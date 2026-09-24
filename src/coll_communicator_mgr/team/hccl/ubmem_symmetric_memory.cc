/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ubmem_symmetric_memory.h"

#include <algorithm>
#include <cstring>
#include <iterator>
#include <list>
#include <limits>
#include <new>
#include <utility>

#include "adapter_rts_common.h"
#include "coll_comm.h"
#include "hccl_common.h"
#include "hcomm_team.h"
#include "log.h"

namespace hccl {

// 与A3 SimpleVaAllocator保持一致，管理每个LSA成员stride内的Window相对偏移。
class UbMemSymmetricMemory::SimpleVaAllocator {
public:
    HcclResult Init(size_t totalSize)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        CHK_PRT_RET(totalSize == 0, HCCL_ERROR("[%s] total size is zero", __func__), HCCL_E_PARA);
        freeList_.clear();
        HcclResult ret = HCCL_SUCCESS;
        EXCEPTION_CATCH(freeList_.push_back({0, totalSize}), ret = HCCL_E_MEMORY);
        if (ret == HCCL_SUCCESS) {
            totalSize_ = totalSize;
        }
        return ret;
    }

    void Destroy()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        freeList_.clear();
        totalSize_ = 0;
    }

    HcclResult Reserve(size_t size, size_t align, size_t& offset)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        CHK_PRT_RET(
            size == 0 || align == 0 || (align & (align - 1U)) != 0,
            HCCL_ERROR("[%s] invalid size[%zu] or alignment[%zu]", __func__, size, align), HCCL_E_PARA);
        for (auto iter = freeList_.begin(); iter != freeList_.end(); ++iter) {
            size_t alignedOffset = 0;
            if (!AlignOffset(iter->offset, align, alignedOffset)) {
                continue;
            }
            size_t blockEnd = iter->offset + iter->size;
            if (alignedOffset > blockEnd || size > blockEnd - alignedOffset) {
                continue;
            }
            CHK_RET(ReserveFromBlock(iter, alignedOffset, size));
            offset = alignedOffset;
            return HCCL_SUCCESS;
        }
        HCCL_ERROR("[%s] no free VA block for size[%zu], alignment[%zu]", __func__, size, align);
        return HCCL_E_MEMORY;
    }

    HcclResult Release(size_t offset, size_t size)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        CHK_PRT_RET(
            size == 0 || offset > totalSize_ || size > totalSize_ - offset,
            HCCL_ERROR(
                "[%s] release range[offset:%zu,size:%zu] exceeds total[%zu]", __func__, offset, size, totalSize_),
            HCCL_E_PARA);
        auto next = freeList_.begin();
        while (next != freeList_.end() && next->offset < offset) {
            ++next;
        }
        CHK_RET(CheckReleaseRange(next, offset, size));
        std::list<FreeBlock>::iterator released;
        HcclResult ret = HCCL_SUCCESS;
        EXCEPTION_CATCH(released = freeList_.insert(next, {offset, size}), ret = HCCL_E_MEMORY);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
        MergeAdjacentBlocks(released);
        return HCCL_SUCCESS;
    }

private:
    struct FreeBlock {
        size_t offset;
        size_t size;
    };

    using FreeBlockIterator = std::list<FreeBlock>::iterator;

    static bool AlignOffset(size_t offset, size_t align, size_t& alignedOffset)
    {
        if (offset > std::numeric_limits<size_t>::max() - (align - 1U)) {
            return false;
        }
        alignedOffset = (offset + align - 1U) & ~(align - 1U);
        return true;
    }

    HcclResult ReserveFromBlock(FreeBlockIterator iter, size_t offset, size_t size)
    {
        size_t blockEnd = iter->offset + iter->size;
        size_t frontSize = offset - iter->offset;
        size_t backSize = blockEnd - offset - size;
        if (frontSize > 0 && backSize > 0) {
            HcclResult ret = HCCL_SUCCESS;
            EXCEPTION_CATCH(freeList_.insert(std::next(iter), {offset + size, backSize}), ret = HCCL_E_MEMORY);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            iter->size = frontSize;
        } else if (frontSize > 0) {
            iter->size = frontSize;
        } else if (backSize > 0) {
            iter->offset = offset + size;
            iter->size = backSize;
        } else {
            freeList_.erase(iter);
        }
        return HCCL_SUCCESS;
    }

    HcclResult CheckReleaseRange(FreeBlockIterator next, size_t offset, size_t size) const
    {
        if (next != freeList_.begin()) {
            auto previous = std::prev(next);
            CHK_PRT_RET(
                previous->offset + previous->size > offset,
                HCCL_ERROR("[%s] release range overlaps previous free block", __func__), HCCL_E_PARA);
        }
        CHK_PRT_RET(
            next != freeList_.end() && offset + size > next->offset,
            HCCL_ERROR("[%s] release range overlaps next free block", __func__), HCCL_E_PARA);
        return HCCL_SUCCESS;
    }

    void MergeAdjacentBlocks(FreeBlockIterator released)
    {
        auto next = std::next(released);
        if (next != freeList_.end() && released->offset + released->size == next->offset) {
            released->size += next->size;
            freeList_.erase(next);
        }
        if (released != freeList_.begin()) {
            auto previous = std::prev(released);
            if (previous->offset + previous->size == released->offset) {
                previous->size += released->size;
                freeList_.erase(released);
            }
        }
    }

    std::list<FreeBlock> freeList_;
    std::mutex mutex_;
    size_t totalSize_{0};
};

static_assert(
    sizeof(UbmemShareableInfo) <= UBMEM_PACKET_DATA_MAX_LEN,
    "UB Memory shareable information exceeds the Agent packet payload");

UbMemSymmetricMemory::UbMemSymmetricMemory(
    CollComm* collComm, HcommTeamHandle lsaTeam, uint32_t netLayer, const std::vector<uint32_t>& worldRankIds)
    : collComm_(collComm),
      lsaTeamSize_(static_cast<uint32_t>(worldRankIds.size())),
      netLayer_(netLayer),
      worldRankIds_(worldRankIds),
      lsaTeam_(lsaTeam)
{
    vaAllocator_.reset(new (std::nothrow) SimpleVaAllocator());
}

UbMemSymmetricMemory::~UbMemSymmetricMemory()
{
    Finalize();
    // Finalize清理失败时记录会保留；对象销毁前仍需移除Owner索引，避免留下悬空通信域指针。
    for (const auto& entry : windowsByHandle_) {
        EraseHcommWindowOwner(entry.first);
    }
}

HcclResult UbMemSymmetricMemory::Init()
{
    CHK_PTR_NULL(collComm_);
    rankGraph_ = collComm_->GetRankGraph();
    selfRank_ = collComm_->GetMyRankId();
    commId_ = collComm_->GetCommId();
    CHK_PTR_NULL(rankGraph_);
    auto selfIter = std::find(worldRankIds_.begin(), worldRankIds_.end(), selfRank_);
    CHK_PRT_RET(
        selfIter == worldRankIds_.end(),
        HCCL_ERROR("[%s] self rank[%u] is not in UB Memory LSA team", __func__, selfRank_), HCCL_E_PARA);
    selfMember_ = static_cast<uint32_t>(selfIter - worldRankIds_.begin());

    EXCEPTION_CATCH(
        agent_ = std::make_unique<UbMemSymmetricMemoryAgent>(
            rankGraph_, collComm_->GetDeviceLogicId(), selfRank_, worldRankIds_, netLayer_, commId_),
        return HCCL_E_MEMORY);
    CHK_SMART_PTR_NULL(agent_);
    return HCCL_SUCCESS;
}

HcclResult UbMemSymmetricMemory::EnsureInit()
{
    CHK_PTR_NULL(collComm_);
    CHK_SMART_PTR_NULL(agent_);
    std::call_once(runtimeInitFlag_, [this]() {
        // 与A3一致：先准备对称VA，再建立用于成员信息交换的LSA Ring。
        runtimeInitResult_ = InitSymmetricVa();
        if (runtimeInitResult_ != HCCL_SUCCESS) {
            HCCL_ERROR("[%s] initialize UB Memory symmetric VA failed, ret[%d]", __func__, runtimeInitResult_);
            FinalizeSymmetricVa();
            return;
        }
        runtimeInitResult_ = agent_->Init();
        if (runtimeInitResult_ != HCCL_SUCCESS) {
            HCCL_ERROR("[%s] initialize UB Memory LSA ring failed, ret[%d]", __func__, runtimeInitResult_);
            FinalizeSymmetricVa();
            return;
        }
        runtimeInitResult_ = GetAllMemberPids();
        if (runtimeInitResult_ != HCCL_SUCCESS) {
            HCCL_ERROR("[%s] exchange UB Memory LSA member pids failed, ret[%d]", __func__, runtimeInitResult_);
            agent_->Finalize();
            FinalizeSymmetricVa();
        }
    });
    return runtimeInitResult_;
}

HcclResult UbMemSymmetricMemory::GetAllMemberPids()
{
    // 交换各LSA成员的bare TGID，用于为Fabric Shareable Handle配置跨进程访问权限。
    int32_t localPid = 0;
    aclError ret = aclrtDeviceGetBareTgid(&localPid);
    CHK_PRT_RET(
        ret != ACL_SUCCESS || localPid <= 0,
        HCCL_ERROR("[%s] get local bare tgid failed, ret[%d], pid[%d]", __func__, ret, localPid), HCCL_E_RUNTIME);
    HCCL_INFO("[%s] local bare tgid[%d]", __func__, localPid);
    EXCEPTION_CATCH(memberPids_.resize(lsaTeamSize_), return HCCL_E_MEMORY);
    CHK_RET(agent_->ExchangeInfo(&localPid, memberPids_.data(), sizeof(localPid)));

    std::string memberPidInfo;
    for (int32_t memberPid : memberPids_) {
        memberPidInfo += std::to_string(memberPid);
        memberPidInfo += "; ";
    }
    HCCL_INFO("[%s] member pids[%s]", __func__, memberPidInfo.c_str());
    return HCCL_SUCCESS;
}

HcclResult UbMemSymmetricMemory::InitSymmetricVa()
{
    if (heapBase_ != nullptr) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(collComm_);
    CHK_SMART_PTR_NULL(vaAllocator_);
    uint64_t strideGB = collComm_->GetCommConfig().GetConfigSymmetricMemoryStride();
    CHK_PRT_RET(
        lsaTeamSize_ == 0 || selfMember_ >= lsaTeamSize_ || strideGB == 0,
        HCCL_ERROR("[%s] invalid symmetric VA configuration", __func__), HCCL_E_PARA);
    CHK_PRT_RET(
        strideGB > std::numeric_limits<size_t>::max() / BYTES_PER_GB,
        HCCL_ERROR("[%s] stride[%llu]GB overflows", __func__, static_cast<unsigned long long>(strideGB)), HCCL_E_PARA);
    stride_ = static_cast<size_t>(strideGB * BYTES_PER_GB);

    size_t freeHbmSize = 0;
    size_t totalHbmSize = 0;
    aclError ret = aclrtGetMemInfo(ACL_HBM_MEM_HUGE, &freeHbmSize, &totalHbmSize);
    CHK_PRT_RET(
        ret != ACL_SUCCESS, HCCL_ERROR("[%s] get HBM memory information failed, ret[%d]", __func__, ret),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(
        stride_ > totalHbmSize,
        HCCL_ERROR("[%s] stride[%zu] exceeds total HBM size[%zu]", __func__, stride_, totalHbmSize), HCCL_E_PARA);
    CHK_RET(InitGranularity());
    CHK_RET(ReserveSymmetricVa());
    HcclResult allocatorRet = vaAllocator_->Init(stride_);
    if (allocatorRet != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] initialize Window offset allocator failed, ret[%d]", __func__, allocatorRet);
        FinalizeSymmetricVa();
        return allocatorRet;
    }
    return HCCL_SUCCESS;
}

HcclResult UbMemSymmetricMemory::InitGranularity()
{
    int32_t deviceId = 0;
    CHK_PRT_RET(
        aclrtGetDevice(&deviceId) != ACL_SUCCESS, HCCL_ERROR("[%s] get device failed", __func__), HCCL_E_RUNTIME);
    aclrtPhysicalMemProp property{};
    property.handleType = ACL_MEM_HANDLE_TYPE_NONE;
    property.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
    property.memAttr = ACL_HBM_MEM_HUGE;
    property.location.id = deviceId;
    property.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
    aclError ret = aclrtMemGetAllocationGranularity(&property, ACL_RT_MEM_ALLOC_GRANULARITY_RECOMMENDED, &granularity_);
    CHK_PRT_RET(
        ret != ACL_SUCCESS || granularity_ == 0,
        HCCL_ERROR("[%s] get memory granularity failed, ret[%d], granularity[%zu]", __func__, ret, granularity_),
        HCCL_E_RUNTIME);
    CHK_PRT_RET(
        stride_ % granularity_ != 0,
        HCCL_ERROR("[%s] stride[%zu] is not aligned to[%zu]", __func__, stride_, granularity_), HCCL_E_PARA);
    return HCCL_SUCCESS;
}

HcclResult UbMemSymmetricMemory::ReserveSymmetricVa()
{
    CHK_PRT_RET(lsaTeamSize_ == 0, HCCL_ERROR("[%s] LSA team size is zero", __func__), HCCL_E_PARA);
    CHK_PRT_RET(
        stride_ > std::numeric_limits<size_t>::max() / lsaTeamSize_,
        HCCL_ERROR("[%s] total symmetric VA size overflows", __func__), HCCL_E_PARA);
    size_t totalHeapSize = stride_ * lsaTeamSize_;
    void* hint = reinterpret_cast<void*>(SYMMETRIC_MEMORY_VA_HINT);
    aclError ret = aclrtReserveMemAddressNoUCMemory(&heapBase_, totalHeapSize, 0, hint, 0);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR("[%s] reserve UB Memory symmetric VA failed, size[%zu], ret[%d]", __func__, totalHeapSize, ret);
        heapBase_ = nullptr;
        return HCCL_E_RUNTIME;
    }
    uintptr_t base = reinterpret_cast<uintptr_t>(heapBase_);
    if (totalHeapSize > std::numeric_limits<uintptr_t>::max() - base) {
        HCCL_ERROR("[%s] reserved UB Memory symmetric VA range overflows", __func__);
        (void)aclrtReleaseMemAddress(heapBase_);
        heapBase_ = nullptr;
        return HCCL_E_RUNTIME;
    }
    HCCL_RUN_INFO(
        "[%s] UB Memory symmetric VA ready, base[%p], stride[%zu], lsaTeamSize[%u]", __func__, heapBase_, stride_,
        lsaTeamSize_);
    return HCCL_SUCCESS;
}

void UbMemSymmetricMemory::FinalizeSymmetricVa()
{
    if (activeMappingCount_ != 0) {
        HCCL_ERROR("[%s] refuse to release symmetric VA with[%zu] active mappings", __func__, activeMappingCount_);
        return;
    }
    if (heapBase_ != nullptr) {
        aclError ret = aclrtReleaseMemAddress(heapBase_);
        if (ret != ACL_SUCCESS) {
            HCCL_ERROR("[%s] release symmetric VA[%p] failed, ret[%d]", __func__, heapBase_, ret);
            return;
        }
    }
    if (vaAllocator_ != nullptr) {
        vaAllocator_->Destroy();
    }
    heapBase_ = nullptr;
    stride_ = 0;
    granularity_ = 0;
}

HcclResult UbMemSymmetricMemory::ValidateRegisterRange(void* ptr, size_t size) const
{
    CHK_PTR_NULL(ptr);
    CHK_PRT_RET(size == 0, HCCL_ERROR("[%s] window size is zero", __func__), HCCL_E_PARA);
    uintptr_t begin = reinterpret_cast<uintptr_t>(ptr);
    CHK_PRT_RET(begin + size < begin, HCCL_ERROR("[%s] address overflow", __func__), HCCL_E_PARA);
    return HCCL_SUCCESS;
}

HcclResult UbMemSymmetricMemory::GetMemoryInfo(
    void* ptr, size_t size, void*& baseUserVa, size_t& baseVaSize, aclrtDrvMemHandle& paHandle,
    size_t& offsetInBaseVa) const
{
    CHK_RET(ValidateRegisterRange(ptr, size));
    CHK_PRT_RET(heapBase_ == nullptr, HCCL_ERROR("[%s] symmetric VA is not initialized", __func__), HCCL_E_UNAVAIL);
    HCCL_INFO("[%s] get memory info, ptr[%p], size[%zu], granularity[%zu]", __func__, ptr, size, granularity_);
    aclError ret = aclrtMemGetAddressRange(ptr, &baseUserVa, &baseVaSize);
    CHK_PRT_RET(
        ret != ACL_SUCCESS || baseUserVa == nullptr || baseVaSize == 0,
        HCCL_ERROR("[%s] get base memory range failed, ptr[%p], ret[%d]", __func__, ptr, ret), HCCL_E_PARA);
    CHK_PRT_RET(
        reinterpret_cast<uintptr_t>(ptr) < reinterpret_cast<uintptr_t>(baseUserVa),
        HCCL_ERROR("[%s] requested address is before baseUserVa", __func__), HCCL_E_PARA);
    CHK_PRT_RET(
        size > baseVaSize
            || reinterpret_cast<uintptr_t>(ptr) - reinterpret_cast<uintptr_t>(baseUserVa) > baseVaSize - size,
        HCCL_ERROR("[%s] requested range exceeds base memory range", __func__), HCCL_E_PARA);
    CHK_PRT_RET(
        granularity_ == 0 || baseVaSize % granularity_ != 0,
        HCCL_ERROR("[%s] baseVaSize[%zu] is not aligned to[%zu]", __func__, baseVaSize, granularity_), HCCL_E_PARA);
    offsetInBaseVa = static_cast<size_t>(reinterpret_cast<uintptr_t>(ptr) - reinterpret_cast<uintptr_t>(baseUserVa));

    ret = aclrtMemRetainAllocationHandle(baseUserVa, &paHandle);
    CHK_PRT_RET(
        ret != ACL_SUCCESS || paHandle == nullptr,
        HCCL_ERROR("[%s] retain local physical handle failed, ret[%d]", __func__, ret), HCCL_E_RUNTIME);
    HCCL_INFO(
        "[%s] retained PA handle[%p], baseUserVa[%p], baseVaSize[%zu]", __func__, paHandle, baseUserVa, baseVaSize);
    return HCCL_SUCCESS;
}

HcclResult UbMemSymmetricMemory::ExportLocalMapping(PaMappingInfo& mapping) const
{
    CHK_PTR_NULL(mapping.paHandle);
    aclError ret = aclrtMemExportToShareableHandleV2(
        mapping.paHandle, 0, ACL_MEM_SHARE_HANDLE_TYPE_FABRIC, static_cast<void*>(&mapping.shareableHandle));
    CHK_PRT_RET(
        ret != ACL_SUCCESS, HCCL_ERROR("[%s] export local physical handle failed, ret[%d]", __func__, ret),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
}

HcclResult UbMemSymmetricMemory::GrantLocalMemory(const PaMappingInfo& mapping)
{
    CHK_PRT_RET(
        memberPids_.size() != lsaTeamSize_ || mapping.paHandle == nullptr,
        HCCL_ERROR("[%s] invalid local grant parameters", __func__), HCCL_E_PARA);
    // 将本地Shareable Handle授权给全部LSA成员进程，供远端Import和Map。
    aclrtMemFabricHandle shareableHandle = mapping.shareableHandle;
    aclError ret = aclrtMemSetPidToShareableHandleV2(
        static_cast<void*>(&shareableHandle), ACL_MEM_SHARE_HANDLE_TYPE_FABRIC, memberPids_.data(), memberPids_.size());
    CHK_PRT_RET(
        ret != ACL_SUCCESS, HCCL_ERROR("[%s] grant local memory failed, ret[%d]", __func__, ret), HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
}

HcclResult UbMemSymmetricMemory::ValidateMappingRange(uint64_t windowOffset, size_t mapSize) const
{
    CHK_PRT_RET(
        heapBase_ == nullptr || windowOffset > stride_ || mapSize > stride_ - windowOffset,
        HCCL_ERROR(
            "[%s] mapping exceeds stride, offset[%llu], size[%zu], stride[%zu]", __func__,
            static_cast<unsigned long long>(windowOffset), mapSize, stride_),
        HCCL_E_PARA);
    CHK_PRT_RET(
        granularity_ == 0 || windowOffset % granularity_ != 0 || mapSize % granularity_ != 0,
        HCCL_ERROR("[%s] mapping is not aligned to granularity[%zu]", __func__, granularity_), HCCL_E_PARA);
    return HCCL_SUCCESS;
}

HcclResult UbMemSymmetricMemory::ImportMemberHandle(
    uint32_t member, const std::vector<uint8_t>& shareableDesc, const PaMappingInfo& mapping, aclrtDrvMemHandle& handle,
    bool& ownsHandle) const
{
    if (member == selfMember_) {
        handle = mapping.paHandle;
        ownsHandle = false;
        CHK_PRT_RET(handle == nullptr, HCCL_ERROR("[%s] local physical handle is null", __func__), HCCL_E_PTR);
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(
        shareableDesc.size() != sizeof(aclrtMemFabricHandle),
        HCCL_ERROR("[%s] member[%u] invalid shareable descriptor size[%zu]", __func__, member, shareableDesc.size()),
        HCCL_E_PARA);
    aclrtMemFabricHandle remoteHandle{};
    CHK_PRT_RET(
        memcpy_s(&remoteHandle, sizeof(remoteHandle), shareableDesc.data(), shareableDesc.size()) != EOK,
        HCCL_ERROR("[%s] copy member[%u] shareable handle failed", __func__, member), HCCL_E_MEMORY);
    aclError ret = aclrtMemImportFromShareableHandleV2(
        static_cast<void*>(&remoteHandle), ACL_MEM_SHARE_HANDLE_TYPE_FABRIC, 0, &handle);
    CHK_PRT_RET(
        ret != ACL_SUCCESS || handle == nullptr,
        HCCL_ERROR("[%s] import member[%u] physical handle failed, ret[%d]", __func__, member, ret), HCCL_E_RUNTIME);
    ownsHandle = true;
    return HCCL_SUCCESS;
}

HcclResult UbMemSymmetricMemory::MapAllMembers(
    uint64_t windowOffset, const std::vector<std::vector<uint8_t>>& shareableDescs, PaMappingInfo& mapping,
    std::vector<CommMem>& memberMems)
{
    CHK_RET(ValidateMappingRange(windowOffset, mapping.baseVaSize));
    CHK_PRT_RET(
        shareableDescs.size() != lsaTeamSize_, HCCL_ERROR("[%s] member descriptor count mismatch", __func__),
        HCCL_E_PARA);
    EXCEPTION_CATCH(mapping.peerMappings.assign(lsaTeamSize_, PeerMappingInfo{}), return HCCL_E_MEMORY);
    EXCEPTION_CATCH(memberMems.assign(lsaTeamSize_, CommMem{}), return HCCL_E_MEMORY);
    for (uint32_t member = 0; member < lsaTeamSize_; ++member) {
        PeerMappingInfo& peer = mapping.peerMappings[member];
        HcclResult ret = ImportMemberHandle(member, shareableDescs[member], mapping, peer.handle, peer.ownsHandle);
        if (ret != HCCL_SUCCESS) {
            return CleanupPartialMapping(mapping, ret);
        }
        uintptr_t base = reinterpret_cast<uintptr_t>(heapBase_);
        uint64_t memberOffset = static_cast<uint64_t>(member) * stride_ + windowOffset;
        peer.address = reinterpret_cast<void*>(base + memberOffset);
        aclError aclRet = aclrtMapMem(peer.address, mapping.baseVaSize, 0, peer.handle, 0);
        if (aclRet != ACL_SUCCESS) {
            HCCL_ERROR("[%s] map member[%u] at[%p] failed, ret[%d]", __func__, member, peer.address, aclRet);
            return CleanupPartialMapping(mapping, HCCL_E_RUNTIME);
        }
        peer.mapped = true;
        ++activeMappingCount_;
        memberMems[member].addr = peer.address;
        memberMems[member].size = mapping.baseVaSize;
        memberMems[member].type = COMM_MEM_TYPE_DEVICE;
    }
    return HCCL_SUCCESS;
}

HcclResult UbMemSymmetricMemory::CleanupPartialMapping(PaMappingInfo& mapping, HcclResult originalResult)
{
    HcclResult cleanupResult = ReleaseMemberMappings(mapping);
    if (cleanupResult != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[%s] cleanup partial mapping failed, original[%d], cleanup[%d]", __func__, originalResult, cleanupResult);
    }
    return originalResult;
}

HcclResult UbMemSymmetricMemory::ReleasePeerMapping(PeerMappingInfo& peer)
{
    if (peer.mapped) {
        aclError ret = aclrtUnmapMem(peer.address);
        if (ret != ACL_SUCCESS) {
            HCCL_ERROR("[%s] unmap address[%p] failed, ret[%d]", __func__, peer.address, ret);
            return HCCL_E_RUNTIME;
        }
        peer.mapped = false;
        --activeMappingCount_;
    }
    if (peer.handle != nullptr && peer.ownsHandle) {
        aclError ret = aclrtFreePhysical(peer.handle);
        if (ret != ACL_SUCCESS) {
            HCCL_ERROR("[%s] free imported physical handle[%p] failed, ret[%d]", __func__, peer.handle, ret);
            return HCCL_E_RUNTIME;
        }
    }
    peer = PeerMappingInfo{};
    return HCCL_SUCCESS;
}

HcclResult UbMemSymmetricMemory::ReleaseMemberMappings(PaMappingInfo& mapping)
{
    // 解除各LSA成员的VA映射并释放导入的远端物理Handle；本地物理内存由申请者释放。
    HcclResult firstError = HCCL_SUCCESS;
    for (PeerMappingInfo& peer : mapping.peerMappings) {
        HcclResult ret = ReleasePeerMapping(peer);
        if (ret != HCCL_SUCCESS && firstError == HCCL_SUCCESS) {
            firstError = ret;
        }
    }
    const bool allReleased
        = std::all_of(mapping.peerMappings.begin(), mapping.peerMappings.end(), [](const PeerMappingInfo& peer) {
              return !peer.mapped && peer.handle == nullptr;
          });
    if (allReleased) {
        mapping.peerMappings.clear();
    }
    return firstError;
}

HcclResult UbMemSymmetricMemory::RegisterInternal(PaMappingInfo& paMapping)
{
    // 与A3不同，A5 Fabric Handle完成Export后不能再次Export；
    // 同一内存重新注册时复用首次Export及授权信息。
    if (!paMapping.shareableHandleReady) {
        CHK_RET(ExportLocalMapping(paMapping));
        paMapping.shareableHandleReady = true;
    }
    if (!paMapping.memberAccessGranted) {
        CHK_RET(GrantLocalMemory(paMapping));
        paMapping.memberAccessGranted = true;
    }

    // Shareable Handle完成PID授权后，再交换给其他LSA成员执行Import和Map。
    UbmemShareableInfo localInfo{
        static_cast<uint64_t>(paMapping.heapBaseOffset), static_cast<uint64_t>(paMapping.baseVaSize),
        paMapping.shareableHandle};
    std::vector<UbmemShareableInfo> memberInfos;
    EXCEPTION_CATCH(memberInfos.resize(lsaTeamSize_), return HCCL_E_MEMORY);
    HcclResult exchangeResult = agent_->ExchangeInfo(&localInfo, memberInfos.data(), sizeof(localInfo));
    CHK_PRT_RET(
        exchangeResult != HCCL_SUCCESS,
        HCCL_ERROR("[%s] exchange UB Memory shareable info failed, ret[%d]", __func__, exchangeResult), exchangeResult);
    for (uint32_t member = 0; member < lsaTeamSize_; ++member) {
        CHK_PRT_RET(
            memberInfos[member].offset != static_cast<uint64_t>(paMapping.heapBaseOffset)
                || memberInfos[member].size != static_cast<uint64_t>(paMapping.baseVaSize),
            HCCL_ERROR(
                "[%s] member[%u] layout[offset:%llu,size:%llu] differs from local[offset:%llu,size:%zu]; "
                "ensure every LSA member invokes registration in the same order",
                __func__, member, static_cast<unsigned long long>(memberInfos[member].offset),
                static_cast<unsigned long long>(memberInfos[member].size),
                static_cast<unsigned long long>(paMapping.heapBaseOffset), paMapping.baseVaSize),
            HCCL_E_PARA);
    }

    // 重叠Window直接共享已有映射；全部注销后peerMappings为空，再注册时需重建映射。
    if (!paMapping.peerMappings.empty()) {
        return HCCL_SUCCESS;
    }

    std::vector<std::vector<uint8_t>> shareableDescs;
    EXCEPTION_CATCH(shareableDescs.resize(lsaTeamSize_), return HCCL_E_MEMORY);
    for (uint32_t member = 0; member < lsaTeamSize_; ++member) {
        const auto* begin = reinterpret_cast<const uint8_t*>(&memberInfos[member].handle);
        EXCEPTION_CATCH(
            shareableDescs[member].assign(begin, begin + sizeof(memberInfos[member].handle)), return HCCL_E_MEMORY);
    }
    HcclResult mapResult = MapAllMembers(paMapping.heapBaseOffset, shareableDescs, paMapping, paMapping.memberMems);
    CHK_PRT_RET(
        mapResult != HCCL_SUCCESS, HCCL_ERROR("[%s] map LSA member memory failed, ret[%d]", __func__, mapResult),
        mapResult);
    return HCCL_SUCCESS;
}

HcclResult UbMemSymmetricMemory::PublishWindow(WindowRecord& record)
{
    CHK_SMART_PTR_NULL(record.paMapInfo);
    CHK_PTR_NULL(record.devWin);
    // 底层共享完整内存块映射，发布Window时再定位到用户注册的子区间。
    std::vector<CommMem> userMemberMems;
    EXCEPTION_CATCH(userMemberMems = record.paMapInfo->memberMems, return HCCL_E_MEMORY);
    for (CommMem& memberMem : userMemberMems) {
        memberMem.addr = static_cast<uint8_t*>(memberMem.addr) + record.offsetInBaseVa;
        memberMem.size = record.userSize;
    }
    void* baseVa = static_cast<uint8_t*>(heapBase_) + record.paMapInfo->heapBaseOffset + record.offsetInBaseVa;
    HcommResult ret = HcommTeamBindUbSymmetricWindow(
        record.devWin, lsaTeam_, agent_->GetNetLayer(), userMemberMems.data(), lsaTeamSize_, baseVa, stride_,
        record.userSize);
    CHK_PRT_RET(
        ret != HCOMM_SUCCESS,
        HCCL_ERROR("[%s] fill HcommWindow LSA information failed, window[%p], ret[%d]", __func__, record.devWin, ret),
        static_cast<HcclResult>(ret));
    return HCCL_SUCCESS;
}

HcclResult UbMemSymmetricMemory::AddWindowRecord(std::unique_ptr<WindowRecord>& record, HcclComm comm)
{
    // windowsByAddress_持有Window所有权，windowsByHandle_提供句柄索引；任一步失败时回滚已插入记录。
    uintptr_t userAddress = reinterpret_cast<uintptr_t>(record->userVa);
    HcclCommSymWindow handle = record->devWin;
    WindowRecord* recordPtr = record.get();
    decltype(windowsByAddress_)::iterator addressIter;
    HcclResult ret = HCCL_SUCCESS;
    EXCEPTION_CATCH(addressIter = windowsByAddress_.emplace(userAddress, nullptr), ret = HCCL_E_MEMORY);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    addressIter->second = std::move(record);

    bool inserted = false;
    EXCEPTION_CATCH(inserted = windowsByHandle_.emplace(handle, recordPtr).second, ret = HCCL_E_MEMORY);
    if (ret == HCCL_SUCCESS && !inserted) {
        HCCL_ERROR("[%s] window handle[%p] already exists", __func__, handle);
        ret = HCCL_E_INTERNAL;
    }
    if (ret == HCCL_SUCCESS) {
        ret = RecordHcommWindowOwner(handle, comm);
    }
    if (ret != HCCL_SUCCESS) {
        if (inserted) {
            windowsByHandle_.erase(handle);
        }
        record = std::move(addressIter->second);
        windowsByAddress_.erase(addressIter);
        return ret;
    }
    return HCCL_SUCCESS;
}

HcclResult UbMemSymmetricMemory::ReleasePaMapping(WindowRecord& record)
{
    if (record.paMapInfo == nullptr) {
        return HCCL_SUCCESS;
    }
    std::shared_ptr<PaMappingInfo> paMapping = record.paMapInfo;
    CHK_PRT_RET(paMapping->refCount == 0, HCCL_ERROR("[%s] invalid PA mapping reference", __func__), HCCL_E_INTERNAL);
    if (paMapping->refCount > 1U) {
        // 仍有Window复用当前PA映射时仅减少引用。
        --paMapping->refCount;
    } else {
        HcclResult ret = ReleaseMemberMappings(*paMapping);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS, HCCL_ERROR("[%s] release member mappings failed, ret[%d]", __func__, ret), ret);
        paMapping->memberMems.clear();
        ret = vaAllocator_->Release(paMapping->heapBaseOffset, paMapping->baseVaSize);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[%s] release VA offset failed, ret[%d]", __func__, ret), ret);
        paMapping->vaOffsetReserved = false;
        // 与A3不同，A5不能在释放后重新Export同一Fabric Handle；
        // 最后一个Window注销后仍保留Fabric信息，供同一内存重新注册。
        paMapping->refCount = 0;
    }
    record.paMapInfo.reset();
    return HCCL_SUCCESS;
}

HcclResult UbMemSymmetricMemory::CleanupWindow(WindowRecord& record)
{
    // HcommWindow由CollComm统一创建和销毁，此处只释放UB Memory的LSA对称VA映射。
    return ReleasePaMapping(record);
}

HcclResult UbMemSymmetricMemory::RegisterWindow(void* ptr, size_t size, HcclCommSymWindow winHandle, HcclComm comm)
{
    CHK_PTR_NULL(winHandle);
    CHK_PTR_NULL(comm);
    std::lock_guard<std::mutex> lock(mutex_);
    CHK_PRT_RET(finalized_, HCCL_ERROR("[%s] symmetric memory manager is finalized", __func__), HCCL_E_UNAVAIL);
    CHK_RET(EnsureInit());

    void* baseUserVa = nullptr;
    size_t baseVaSize = 0;
    aclrtDrvMemHandle paHandle = nullptr;
    size_t offsetInBaseVa = 0;
    CHK_RET(GetMemoryInfo(ptr, size, baseUserVa, baseVaSize, paHandle, offsetInBaseVa));
    std::unique_ptr<WindowRecord> window;
    HcclResult ret = HCCL_SUCCESS;
    EXCEPTION_CATCH(window = std::make_unique<WindowRecord>(), ret = HCCL_E_MEMORY);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }

    std::shared_ptr<PaMappingInfo> paMapping;
    auto mappingIter = paMappingMap_.find(paHandle);
    if (mappingIter != paMappingMap_.end()) {
        paMapping = mappingIter->second;
        if (paMapping == nullptr || paMapping->baseUserVa != baseUserVa || paMapping->baseVaSize != baseVaSize) {
            HCCL_ERROR("[%s] PA handle is associated with a different base memory range", __func__);
            return HCCL_E_INTERNAL;
        }
        if (paMapping->refCount == 0 && !paMapping->vaOffsetReserved) {
            size_t heapBaseOffset = 0;
            ret = vaAllocator_->Reserve(baseVaSize, granularity_, heapBaseOffset);
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR(
                    "[%s] reserve VA space for cached PA mapping failed, size[%zu], alignment[%zu], stride[%zu]",
                    __func__, baseVaSize, granularity_, stride_);
                return ret;
            }
            paMapping->heapBaseOffset = heapBaseOffset;
            paMapping->vaOffsetReserved = true;
        }
        CHK_PRT_RET(
            paMapping->refCount == std::numeric_limits<uint32_t>::max(),
            HCCL_ERROR("[%s] PA mapping reference is exhausted", __func__), HCCL_E_UNAVAIL);
        ++paMapping->refCount;
        HCCL_INFO("[%s] reuse PA handle[%p], refCount[%u]", __func__, paMapping->paHandle, paMapping->refCount);
    } else {
        size_t heapBaseOffset = 0;
        ret = vaAllocator_->Reserve(baseVaSize, granularity_, heapBaseOffset);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[%s] reserve VA space failed, size[%zu], alignment[%zu], stride[%zu]", __func__, baseVaSize,
                granularity_, stride_);
            return ret;
        }
        EXCEPTION_CATCH(paMapping = std::make_shared<PaMappingInfo>(), ret = HCCL_E_MEMORY);
        if (ret != HCCL_SUCCESS) {
            (void)vaAllocator_->Release(heapBaseOffset, baseVaSize);
            return ret;
        }
        paMapping->baseUserVa = baseUserVa;
        paMapping->baseVaSize = baseVaSize;
        paMapping->paHandle = paHandle;
        paMapping->heapBaseOffset = heapBaseOffset;
        paMapping->vaOffsetReserved = true;
        paMapping->refCount = 1U;
        bool inserted = false;
        EXCEPTION_CATCH(inserted = paMappingMap_.emplace(paMapping->paHandle, paMapping).second, ret = HCCL_E_MEMORY);
        if (ret != HCCL_SUCCESS || !inserted) {
            (void)vaAllocator_->Release(heapBaseOffset, paMapping->baseVaSize);
            (void)ReleaseMemberMappings(*paMapping);
            return ret != HCCL_SUCCESS ? ret : HCCL_E_INTERNAL;
        }
    }

    window->userVa = ptr;
    window->userSize = size;
    window->offsetInBaseVa = offsetInBaseVa;
    window->devWin = winHandle;
    window->paMapInfo = paMapping;

    ret = RegisterInternal(*paMapping);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] register UB Memory mapping failed, ret[%d]", __func__, ret);
        (void)CleanupWindow(*window);
        return ret;
    }
    ret = PublishWindow(*window);
    if (ret != HCCL_SUCCESS) {
        (void)CleanupWindow(*window);
        return ret;
    }
    ret = AddWindowRecord(window, comm);
    if (ret != HCCL_SUCCESS) {
        if (window != nullptr) {
            (void)CleanupWindow(*window);
        }
        return ret;
    }
    HCCL_RUN_INFO(
        "[%s] A5 UB Memory symmetric window registered, comm[%s], userVa[%p], userSize[%zu], "
        "baseUserVa[%p], baseVaSize[%zu], offsetInBaseVa[%zu], window[%p]",
        __func__, commId_.c_str(), ptr, size, paMapping->baseUserVa, paMapping->baseVaSize, offsetInBaseVa, winHandle);
    return HCCL_SUCCESS;
}

void UbMemSymmetricMemory::EraseWindowAddressRecord(uintptr_t userAddress, const WindowRecord* record)
{
    auto range = windowsByAddress_.equal_range(userAddress);
    for (auto iter = range.first; iter != range.second; ++iter) {
        if (iter->second.get() == record) {
            windowsByAddress_.erase(iter);
            return;
        }
    }
}

HcclResult UbMemSymmetricMemory::DeregisterWindow(HcclCommSymWindow winHandle)
{
    CHK_PTR_NULL(winHandle);
    std::lock_guard<std::mutex> lock(mutex_);
    CHK_PRT_RET(finalized_, HCCL_ERROR("[%s] symmetric memory manager is finalized", __func__), HCCL_E_UNAVAIL);
    auto handleIter = windowsByHandle_.find(winHandle);
    CHK_PRT_RET(
        handleIter == windowsByHandle_.end(), HCCL_ERROR("[%s] window[%p] not found", __func__, winHandle),
        HCCL_E_NOT_FOUND);
    WindowRecord* record = handleIter->second;
    CHK_PTR_NULL(record);
    uintptr_t userAddress = reinterpret_cast<uintptr_t>(record->userVa);
    HcclResult ret = CleanupWindow(*record);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    windowsByHandle_.erase(handleIter);
    EraseWindowAddressRecord(userAddress, record);
    return HCCL_SUCCESS;
}

void UbMemSymmetricMemory::Finalize()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (finalized_) {
        return;
    }
    for (auto iter = windowsByAddress_.begin(); iter != windowsByAddress_.end();) {
        WindowRecord& record = *iter->second;
        HcclResult ret = CleanupWindow(record);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[%s] cleanup window[%p] failed, ret[%d]", __func__, record.devWin, ret);
            ++iter;
            continue;
        }
        EraseHcommWindowOwner(record.devWin);
        windowsByHandle_.erase(record.devWin);
        iter = windowsByAddress_.erase(iter);
    }
    if (!windowsByAddress_.empty()) {
        HCCL_ERROR("[%s] UB Memory still has windows[%zu]", __func__, windowsByAddress_.size());
        return;
    }
    // refCount归零后PA记录会保留以支持同一内存重新注册，通信域销毁时再统一清理。
    for (const auto& entry : paMappingMap_) {
        const std::shared_ptr<PaMappingInfo>& paMapping = entry.second;
        if (paMapping == nullptr || paMapping->refCount != 0 || !paMapping->peerMappings.empty()
            || paMapping->vaOffsetReserved) {
            HCCL_ERROR("[%s] invalid cached PA mapping", __func__);
            return;
        }
    }
    // VA offset由FinalizeSymmetricVa销毁allocator时整体回收，无需逐条Release。
    paMappingMap_.clear();
    windowsByHandle_.clear();
    if (agent_ != nullptr) {
        agent_->Finalize();
    }
    FinalizeSymmetricVa();
    finalized_ = true;
}

} // namespace hccl
