/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * Description: ccu executor resource manager
 * Author: caiyifan
 */

#include "ccu_resource_manager.h"

#include <cstdint>
#include <cstring>

#include "ccu_microcode_v1.h"
#include "sim_log.h"
#include "ccu_reduce_operator.h"
#include "runtime_state/sim_models.h"
#include "runtime_state/db_sim_runner_common.h"
#include "store_sim_shm_memory_common.h"

using namespace std;
using namespace hcomm::CcuRep;

void CcuResourceManager::Reset()
{
    HCCL_VM_DEBUG("Clearing all CCU resource state (KE/XN/MS/simulators)");
    for (size_t rankId = 0; rankId < ccuResData_.v1Res.size(); rankId++) {
        if (ccuResData_.v1Res[rankId] != nullptr) {
            ccuResData_.v1Res[rankId]->Reset();
        }
    }
    for (size_t rankId = 0; rankId < ccuResData_.v2Res.size(); rankId++) {
        if (ccuResData_.v2Res[rankId] != nullptr) {
            ccuResData_.v2Res[rankId]->Reset();
        }
    }
}

void CcuResourceManager::Init(
    int rankId, int rankSize, RunnerCcuVersion version, const std::vector<uint64_t>& ccuResourceBaseAddr)
{
    ccuResData_.version = version;
    if (ccuResData_.version == RunnerCcuVersion::CCU_V1) {
        if (static_cast<int>(ccuResData_.v1Res.size()) < rankSize) {
            ccuResData_.v1Res.resize(rankSize);
        }
        ccuResData_.v1Res[rankId] = std::make_unique<CcuResourceV1>(rankId, rankSize);
    } else if (ccuResData_.version == RunnerCcuVersion::CCU_V2) {
        if (static_cast<int>(ccuResData_.v2Res.size()) < rankSize) {
            ccuResData_.v2Res.resize(rankSize);
        }
        ccuResData_.v2Res[rankId] = std::make_unique<CcuResourceV2>(rankId, rankSize);
    } else {
        HCCL_VM_ERROR("ccu version {} not supported", static_cast<int>(ccuResData_.version));
    }
    // ★ per-(rank, die) 存储：原实现 `ccuResourceBaseAddr_ = ccuResourceBaseAddr` 是
    //   last-writer-wins——真实基址 per-rank 唯一，多 device 初始化后只剩最后一个 device 的
    //   基址，其余 rank 的地址反查全部错位（设计文档 §4.5.4 P3）。
    ccuResourceBaseAddr_[rankId] = ccuResourceBaseAddr;
}

void CcuResourceManager::InitInstrInfo(int rankId, int dieId, const CcuInstrData& ccuInstrInfo)
{
    auto version = ccuResData_.version;
    if (version == RunnerCcuVersion::CCU_V1) {
        ccuResData_.v1Res[rankId]->instrSpace_[dieId] = ccuInstrInfo;
    } else if (ccuResData_.version == RunnerCcuVersion::CCU_V2) {
        ccuResData_.v2Res[rankId]->instrSpace_[dieId] = ccuInstrInfo;
    } else {
        HCCL_VM_ERROR("ccu version {} not supported", static_cast<int>(version));
    }
}

bool CcuResourceManager::HasResource(int index) const
{
    if (index < 0) {
        return false;
    }
    auto version = ccuResData_.version;
    if (version == RunnerCcuVersion::CCU_V1) {
        return static_cast<size_t>(index) < ccuResData_.v1Res.size() && ccuResData_.v1Res[index] != nullptr;
    }
    if (version == RunnerCcuVersion::CCU_V2) {
        return static_cast<size_t>(index) < ccuResData_.v2Res.size() && ccuResData_.v2Res[index] != nullptr;
    }
    return false;
}

void CcuResourceManager::InitChannelInfo(int rankId, const RankChannelInfo& channelInfo)
{
    auto version = ccuResData_.version;
    if (version == RunnerCcuVersion::CCU_V1) {
        ccuResData_.v1Res[rankId]->channelId2RmtRankMap_ = channelInfo;
    } else if (ccuResData_.version == RunnerCcuVersion::CCU_V2) {
        ccuResData_.v2Res[rankId]->channelId2RmtRankMap_ = channelInfo;
    } else {
        HCCL_VM_ERROR("ccu version {} not supported", static_cast<int>(version));
    }
}

void CcuResourceManager::AddTaskInfo(int rankId, const HcclTaskMetaData& task)
{
    auto dieId = task.taskData.ccu.dieId;
    if (dieId >= HcclSim::DIE_NUM) {
        HCCL_VM_ERROR("invalid dieId[{}].", dieId);
        return;
    }
    auto version = ccuResData_.version;
    if (version == RunnerCcuVersion::CCU_V1) {
        ccuResData_.v1Res[rankId]->ccuTaskInfos_[dieId] = task.taskData.ccu;
    } else if (ccuResData_.version == RunnerCcuVersion::CCU_V2) {
        ccuResData_.v2Res[rankId]->ccuTaskInfos_[dieId] = task.taskData.ccu;
    } else {
        HCCL_VM_ERROR("ccu version {} not supported", static_cast<int>(version));
    }
}

uint64_t CcuResourceManager::GetSqeArgValue(int rankId, int dieId, uint16_t argId) const
{
    auto version = ccuResData_.version;
    if (version == RunnerCcuVersion::CCU_V1) {
        return ccuResData_.v1Res[rankId]->ccuTaskInfos_[dieId].args[argId];
    } else if (ccuResData_.version == RunnerCcuVersion::CCU_V2) {
        return ccuResData_.v2Res[rankId]->ccuTaskInfos_[dieId].args[argId];
    } else {
        HCCL_VM_ERROR("ccu version {} not supported", static_cast<int>(version));
        return U64_INVALID;
    }
}

uint64_t CcuResourceManager::GetXnValue(int rankId, int dieId, uint16_t xnId) const
{
    auto version = ccuResData_.version;
    if (version == RunnerCcuVersion::CCU_V1) {
        return ccuResData_.v1Res[rankId]->xn_[dieId][xnId];
    } else if (ccuResData_.version == RunnerCcuVersion::CCU_V2) {
        return ccuResData_.v2Res[rankId]->xn_[dieId][xnId];
    } else {
        HCCL_VM_ERROR("ccu version {} not supported", static_cast<int>(version));
        return U64_INVALID;
    }
}

uint64_t CcuResourceManager::GetGsaValue(int rankId, int dieId, uint16_t gsaId) const
{
    auto version = ccuResData_.version;
    if (version == RunnerCcuVersion::CCU_V1) {
        return ccuResData_.v1Res[rankId]->gsa_[dieId][gsaId];
    } else if (ccuResData_.version == RunnerCcuVersion::CCU_V2) {
        return ccuResData_.v2Res[rankId]->gsa_[dieId][gsaId];
    } else {
        HCCL_VM_ERROR("ccu version {} not supported", static_cast<int>(version));
        return U64_INVALID;
    }
}

uint16_t CcuResourceManager::GetCkeValue(int rankId, int dieId, uint16_t ckeId) const
{
    auto version = ccuResData_.version;
    if (version == RunnerCcuVersion::CCU_V1) {
        return ccuResData_.v1Res[rankId]->cke_[dieId][ckeId];
    } else if (ccuResData_.version == RunnerCcuVersion::CCU_V2) {
        return ccuResData_.v2Res[rankId]->cke_[dieId][ckeId];
    } else {
        HCCL_VM_ERROR("ccu version {} not supported", static_cast<int>(version));
        return U16_INVALID;
    }
}

std::pair<int, int> CcuResourceManager::GetRmtCcu(int rankId, int dieId, uint16_t channelId) const
{
    int rmtRank = S32_INVALID;
    int rmtDie = S32_INVALID;

    if (channelId >= SimCcuV1::MAX_CCU_CHANNEL_NUM) {
        HCCL_VM_ERROR("invalid channelId[{}], max={}", channelId, SimCcuV1::MAX_CCU_CHANNEL_NUM);
        return std::make_pair(S32_INVALID, S32_INVALID);
    }

    auto version = ccuResData_.version;
    if (version == RunnerCcuVersion::CCU_V1) {
        rmtRank = ccuResData_.v1Res[rankId]->channelId2RmtRankMap_[dieId][channelId].rankId;
        rmtDie = ccuResData_.v1Res[rankId]->channelId2RmtRankMap_[dieId][channelId].dieId;
    } else if (ccuResData_.version == RunnerCcuVersion::CCU_V2) {
        rmtRank = ccuResData_.v2Res[rankId]->channelId2RmtRankMap_[dieId][channelId].rankId;
        rmtDie = ccuResData_.v2Res[rankId]->channelId2RmtRankMap_[dieId][channelId].dieId;
    } else {
        HCCL_VM_ERROR("ccu version {} not supported", static_cast<int>(version));
        return std::make_pair(S32_INVALID, S32_INVALID);
    }

    if (dieId >= HcclSim::DIE_NUM || rmtRank == S32_INVALID || rmtDie == S32_INVALID) {
        HCCL_VM_ERROR("get invalid value, dieId{}, rmtRank{}, rmtDie{}", dieId, rmtRank, rmtDie);
        return std::make_pair(S32_INVALID, S32_INVALID);
    }

    return std::make_pair(rmtRank, rmtDie);
}

void CcuResourceManager::UpdateXnValue(int rankId, int dieId, uint16_t xnId, uint64_t value)
{
    auto version = ccuResData_.version;
    if (version == RunnerCcuVersion::CCU_V1) {
        ccuResData_.v1Res[rankId]->xn_[dieId][xnId] = value;
    } else if (ccuResData_.version == RunnerCcuVersion::CCU_V2) {
        ccuResData_.v2Res[rankId]->xn_[dieId][xnId] = value;
    } else {
        HCCL_VM_ERROR("ccu version {} not supported", static_cast<int>(version));
    }
}

void CcuResourceManager::UpdateGsaValue(int rankId, int dieId, uint16_t gsaId, uint64_t value)
{
    auto version = ccuResData_.version;
    if (version == RunnerCcuVersion::CCU_V1) {
        ccuResData_.v1Res[rankId]->gsa_[dieId][gsaId] = value;
    } else if (ccuResData_.version == RunnerCcuVersion::CCU_V2) {
        ccuResData_.v2Res[rankId]->gsa_[dieId][gsaId] = value;
    } else {
        HCCL_VM_ERROR("ccu version {} not supported", static_cast<int>(version));
    }
}

void CcuResourceManager::UpdateCkeValue(int rankId, int dieId, uint16_t ckeId, uint16_t value)
{
    auto version = ccuResData_.version;
    if (version == RunnerCcuVersion::CCU_V1) {
        ccuResData_.v1Res[rankId]->cke_[dieId][ckeId] = value;
    } else if (ccuResData_.version == RunnerCcuVersion::CCU_V2) {
        ccuResData_.v2Res[rankId]->cke_[dieId][ckeId] = value;
    } else {
        HCCL_VM_ERROR("ccu version {} not supported", static_cast<int>(version));
    }
}

void CcuResourceManager::TransMemToMem(
    void* srcBuf, void* dstBuf, uint64_t length, bool reduceEn, uint16_t reduceOp, uint16_t dataType)
{
    HCCL_VM_DEBUG("srcBuf={}, dstBuf={}, length={}", srcBuf, dstBuf, length);
    if (srcBuf == nullptr) {
        HCCL_VM_ERROR("srcBuf invalid...");
        return;
    }

    if (dstBuf == nullptr) {
        HCCL_VM_ERROR("dstBuf invalid...");
        return;
    }

    sim::runtime::PhyMemBlock srcPhyMem{};
    auto srcAddr = sim::AcquireDevPtrInNoHostProcess((void*)srcBuf, srcPhyMem);
    if (srcAddr == nullptr) {
        HCCL_VM_ERROR("无法获取srcBuf的设备地址(addr= {:x})！", reinterpret_cast<uintptr_t>(srcBuf));
        return;
    }

    sim::runtime::PhyMemBlock dstPhyMem{};
    auto dstAddr = sim::AcquireDevPtrInNoHostProcess((void*)dstBuf, dstPhyMem);
    if (dstAddr == nullptr) {
        HCCL_VM_ERROR("无法获取dstBuf的设备地址(addr= {:x})！", reinterpret_cast<uintptr_t>(dstBuf));
        return;
    }

    if (reduceEn) {
        ReduceProcess(srcAddr, dstAddr, length, dataType, reduceOp, ccuResData_.version);
    } else {
        memcpy(dstAddr, srcAddr, length);
    }

    sim::ReleaseInNoHostProcess(srcPhyMem);
    sim::ReleaseInNoHostProcess(dstPhyMem);
}

void CcuResourceManager::TransMSToMS(
    int srcRank, int srcDie, int dstRank, int dstDie, uint16_t srcMsId, uint16_t dstMsId, uint16_t length)
{
    memcpy(GetMsAddr(dstRank, dstDie, dstMsId), GetMsAddr(srcRank, srcDie, srcMsId), length);
}

bool CcuResourceManager::TransMSToMem(int rankId, int dieId, uint16_t msId, void* buf, uint16_t length)
{
    if (buf == nullptr) {
        HCCL_VM_ERROR("param invalid...");
        return false;
    }

    sim::runtime::PhyMemBlock dstPhyMem{};
    auto dstAddr = sim::AcquireDevPtrInNoHostProcess((void*)buf, dstPhyMem);
    if (dstAddr == nullptr) {
        HCCL_VM_ERROR("无法获取buf的设备地址(addr= {:x})！", reinterpret_cast<uintptr_t>(buf));
        return false;
    }

    memcpy(dstAddr, GetMsAddr(rankId, dieId, msId), length);

    sim::ReleaseInNoHostProcess(dstPhyMem);
    return true;
}

bool CcuResourceManager::TransMemToXn(int rankId, int dieId, uint16_t xnId, uint64_t buf, uint16_t length)
{
    sim::runtime::PhyMemBlock srcPhyMem{};
    auto srcAddr = sim::AcquireDevPtrInNoHostProcess((void*)buf, srcPhyMem);
    if (srcAddr == nullptr) {
        HCCL_VM_ERROR("无法获取buf的设备地址(addr= {:x})！", buf);
        return false;
    }
    memcpy(GetXnAddr(rankId, dieId, xnId), srcAddr, length);

    sim::ReleaseInNoHostProcess(srcPhyMem);
    return true;
}

bool CcuResourceManager::TransXnToMem(int rankId, int dieId, uint16_t xnId, uint64_t buf, uint16_t length)
{
    sim::runtime::PhyMemBlock dstPhyMem{};
    auto dstAddr = sim::AcquireDevPtrInNoHostProcess((void*)buf, dstPhyMem);
    if (dstAddr == nullptr) {
        HCCL_VM_ERROR("无法获取buf的设备地址(addr= {:x})！", buf);
        return false;
    }
    memcpy(dstAddr, GetXnAddr(rankId, dieId, xnId), length);

    sim::ReleaseInNoHostProcess(dstPhyMem);
    return true;
}

bool CcuResourceManager::TransMemToMS(int rankId, int dieId, uint16_t msId, void* buf, uint16_t length)
{
    if (buf == nullptr) {
        HCCL_VM_ERROR("param invalid...");
        return false;
    }

    sim::runtime::PhyMemBlock srcPhyMem{};
    auto srcAddr = sim::AcquireDevPtrInNoHostProcess((void*)buf, srcPhyMem);
    if (srcAddr == nullptr) {
        HCCL_VM_ERROR("无法获取buf的设备地址(addr= {:x})！", reinterpret_cast<uintptr_t>(buf));
        return false;
    }
    memcpy(GetMsAddr(rankId, dieId, msId), srcAddr, length);

    sim::ReleaseInNoHostProcess(srcPhyMem);
    return true;
}

char* CcuResourceManager::GetMsAddr(int rankId, int dieId, uint16_t msId) const
{
    auto version = ccuResData_.version;
    uint32_t offset = static_cast<uint32_t>(msId) * HcclSim::BYTE_NUM_4K; // 一个MS容量为4KB
    if (version == RunnerCcuVersion::CCU_V1) {
        return (ccuResData_.v1Res[rankId]->ms_[dieId].data() + offset);
    } else if (ccuResData_.version == RunnerCcuVersion::CCU_V2) {
        return (ccuResData_.v2Res[rankId]->ms_[dieId].data() + offset);
    } else {
        HCCL_VM_ERROR("ccu version {} not supported", static_cast<int>(version));
        return nullptr;
    }
}

uint64_t* CcuResourceManager::GetXnAddr(int rankId, int dieId, uint16_t xnId) const
{
    auto version = ccuResData_.version;
    uint32_t offset = static_cast<uint32_t>(xnId) * 64; // 一个MS容量为4KB
    if (version == RunnerCcuVersion::CCU_V1) {
        return (ccuResData_.v1Res[rankId]->xn_[dieId].data() + offset);
    } else if (ccuResData_.version == RunnerCcuVersion::CCU_V2) {
        return (ccuResData_.v2Res[rankId]->xn_[dieId].data() + offset);
    } else {
        HCCL_VM_ERROR("ccu version {} not supported", static_cast<int>(version));
        return nullptr;
    }
}

namespace {
/**
 * CcuComponerntType（runner/checker 对外枚举）⟷ devspec::CcuResType（device 规格区域类型）。
 *
 * ★ 地址↔id 的区偏移/步长/容量/MS 几何全部由 CcuAddressSpace 按 device 规格提供，
 *   此处仅做类型标签映射——取代原 XN_*_ADDR_A6 / CCU_MS_START_ADDR / MS_INTERLEAVE_NUM
 *   等字面量常量块与 findBaseAddr/findTypeByAddr/findBaseSize 三份手写转换
 *   （设计文档 §2.3/§2.4/§2.5、§4.5.4：改为持有 CcuAddressSpace）。
 */
sim::devspec::CcuResType ToResType(CcuComponerntType type)
{
    switch (type) {
        case CcuComponerntType::XN_A6:
            return sim::devspec::CcuResType::XN;
        case CcuComponerntType::CKE_A6:
            return sim::devspec::CcuResType::CKE;
        case CcuComponerntType::PFE_A6:
            return sim::devspec::CcuResType::PFE;
        case CcuComponerntType::CHANNEL_A6:
            return sim::devspec::CcuResType::CHANNEL;
        case CcuComponerntType::JETTY_A6:
            return sim::devspec::CcuResType::JETTY;
        case CcuComponerntType::MISSION_A6:
            return sim::devspec::CcuResType::MISSION;
        case CcuComponerntType::LOOP_A6:
            return sim::devspec::CcuResType::LOOP;
        default:
            return sim::devspec::CcuResType::UNKNOWN;
    }
}

CcuComponerntType ToComponentType(sim::devspec::CcuResType type)
{
    switch (type) {
        case sim::devspec::CcuResType::XN:
            return CcuComponerntType::XN_A6;
        case sim::devspec::CcuResType::CKE:
            return CcuComponerntType::CKE_A6;
        case sim::devspec::CcuResType::PFE:
            return CcuComponerntType::PFE_A6;
        case sim::devspec::CcuResType::CHANNEL:
            return CcuComponerntType::CHANNEL_A6;
        case sim::devspec::CcuResType::JETTY:
            return CcuComponerntType::JETTY_A6;
        case sim::devspec::CcuResType::MISSION:
            return CcuComponerntType::MISSION_A6;
        case sim::devspec::CcuResType::LOOP:
            return CcuComponerntType::LOOP_A6;
        default:
            return CcuComponerntType::UNKNOWN;
    }
}
} // namespace

uint64_t CcuResourceManager::BaseAddr(int rankId, uint32_t dieId) const
{
    const auto it = ccuResourceBaseAddr_.find(rankId);
    if (it == ccuResourceBaseAddr_.end() || dieId >= it->second.size()) {
        HCCL_VM_ERROR("ccu base addr not initialized, rankId=[{}] dieId=[{}]", rankId, dieId);
        return 0;
    }
    return it->second[dieId];
}

const sim::devspec::CcuAddressSpace* CcuResourceManager::AddrSpace(int rankId)
{
    const auto cached = addrSpaces_.find(rankId);
    if (cached != addrSpaces_.end()) {
        return cached->second.get();
    }
    const auto* spec = sim::runtime::GetDeviceSpec(static_cast<uint64_t>(rankId));
    if (spec == nullptr) {
        HCCL_VM_ERROR("resolve device spec failed, rankId=[{}]", rankId);
        return nullptr;
    }
    auto space = std::make_unique<sim::devspec::CcuAddressSpace>(*spec);
    const auto* raw = space.get();
    addrSpaces_.emplace(rankId, std::move(space));
    return raw;
}

bool CcuResourceManager::GetMSIdByAddr(int rankId, uint32_t dieId, uint64_t addr, uint16_t& msId)
{
    const uint64_t base = BaseAddr(rankId, dieId);
    const auto* space = AddrSpace(rankId);
    if (base == 0 || space == nullptr) {
        return false;
    }
    uint32_t id = 0;
    if (!space->MsAddrToId(base, addr, id) || id > UINT16_MAX) {
        HCCL_VM_ERROR("ms addr decode failed, rankId=[{}] dieId=[{}] addr=[0x{:x}]", rankId, dieId, addr);
        return false;
    }
    msId = static_cast<uint16_t>(id);
    return true;
}

bool CcuResourceManager::GetXnAndTypeIdByAddr(
    int rankId, uint32_t dieId, uint64_t xnAddr, CcuComponerntType& type, uint16_t& xnId)
{
    type = CcuComponerntType::UNKNOWN;
    const uint64_t base = BaseAddr(rankId, dieId);
    const auto* space = AddrSpace(rankId);
    if (base == 0 || space == nullptr) {
        return false;
    }
    sim::devspec::CcuResType resType = sim::devspec::CcuResType::UNKNOWN;
    uint64_t id = 0;
    if (!space->AddrToResId(base, xnAddr, resType, id)) {
        HCCL_VM_ERROR("unknown type, rankId=[{}] dieId=[{}] addr=[0x{:x}] base=[0x{:x}]", rankId, dieId, xnAddr, base);
        return false;
    }
    if (id > UINT16_MAX) {
        HCCL_VM_ERROR("res id out of range, addr=[0x{:x}] id=[{}]", xnAddr, id);
        return false;
    }
    type = ToComponentType(resType);
    xnId = static_cast<uint16_t>(id);
    return true;
}

bool CcuResourceManager::GetXnIdByAddr(
    int rankId, uint32_t dieId, CcuComponerntType type, uint64_t xnAddr, uint16_t& xnId)
{
    const uint64_t base = BaseAddr(rankId, dieId);
    const auto* space = AddrSpace(rankId);
    if (base == 0 || space == nullptr) {
        return false;
    }
    uint64_t id = 0;
    // ★ 带上界校验（AddrToIdOfType 内置 id < count），修掉原 "需要判断界限 todo"（P10）。
    if (!space->AddrToIdOfType(base, ToResType(type), xnAddr, id) || id > UINT16_MAX) {
        HCCL_VM_ERROR(
            "addr not in region, rankId=[{}] dieId=[{}] type=[{}] addr=[0x{:x}]", rankId, dieId, static_cast<int>(type),
            xnAddr);
        return false;
    }
    xnId = static_cast<uint16_t>(id);
    return true;
}

bool CcuResourceManager::GetAddrByXnId(
    int rankId, uint32_t dieId, CcuComponerntType type, uint16_t xnId, uint64_t& xnAddr)
{
    const uint64_t base = BaseAddr(rankId, dieId);
    const auto* space = AddrSpace(rankId);
    if (base == 0 || space == nullptr) {
        return false;
    }
    if (!space->ResIdToAddr(base, ToResType(type), xnId, xnAddr)) {
        HCCL_VM_ERROR(
            "id out of region, rankId=[{}] dieId=[{}] type=[{}] id=[{}]", rankId, dieId, static_cast<int>(type), xnId);
        return false;
    }
    return true;
}

uint16_t CcuResourceManager::GetInstrCnt(int rankId, int dieId) const
{
    auto version = ccuResData_.version;
    if (version == RunnerCcuVersion::CCU_V1) {
        return ccuResData_.v1Res[rankId]->instrSpace_[dieId].instrCnt;
    } else if (ccuResData_.version == RunnerCcuVersion::CCU_V2) {
        return ccuResData_.v2Res[rankId]->instrSpace_[dieId].instrCnt;
    } else {
        HCCL_VM_ERROR("ccu version {} not supported", static_cast<int>(version));
        return U16_INVALID;
    }
}

std::vector<hcomm::CcuRep::CcuInstr> CcuResourceManager::GetInstrData(int rankId, int dieId) const
{
    auto version = ccuResData_.version;
    if (version == RunnerCcuVersion::CCU_V1) {
        return ccuResData_.v1Res[rankId]->instrSpace_[dieId].instrData;
    } else if (ccuResData_.version == RunnerCcuVersion::CCU_V2) {
        return ccuResData_.v2Res[rankId]->instrSpace_[dieId].instrData;
    } else {
        HCCL_VM_ERROR("ccu version {} not supported", static_cast<int>(version));
        return {};
    }
}

std::string CcuResourceManager::GetInstrDescribe(int rankId, int dieId, int instrId) const
{
#ifndef DEVICE_STUB
    auto version = ccuResData_.version;
    if (version == RunnerCcuVersion::CCU_V1) {
        // return hcomm::CcuRep::ParseInstr(&(ccuResData_.v1Res[rankId]->instrSpace_[dieId].instrData[instrId]));
    } else if (ccuResData_.version == RunnerCcuVersion::CCU_V2) {
        // return hcomm::CcuRep::ParseInstr(&(ccuResData_.v2Res[rankId]->instrSpace_[dieId].instrData[instrId]));
    } else {
        HCCL_VM_ERROR("ccu version {} not supported", static_cast<int>(version));
        return "";
    }
#endif
    return "";
}

std::shared_ptr<CcuSimulator>
CcuResourceManager::InitSimulator(int rankId, int dieId, uint16_t instrStartId, uint16_t endInstrId, uint16_t instCnt)
{
    if (dieId < 0 || dieId >= HcclSim::DIE_NUM || !HasResource(rankId)) {
        HCCL_VM_ERROR("invalid CCU simulator resource: rankId={}, dieId={}", rankId, dieId);
        return nullptr;
    }
    auto version = ccuResData_.version;
    if (version == RunnerCcuVersion::CCU_V1) {
        if (ccuResData_.v1Res[rankId]->simulators_[dieId] == nullptr) {
            ccuResData_.v1Res[rankId]->simulators_[dieId]
                = std::make_shared<CcuSimulator>(rankId, dieId, instrStartId, endInstrId, instCnt, version);
            return ccuResData_.v1Res[rankId]->simulators_[dieId];
        } else {
            ccuResData_.v1Res[rankId]->simulators_[dieId]->Init(instrStartId, endInstrId, instCnt, version);
            return ccuResData_.v1Res[rankId]->simulators_[dieId];
        }
    } else if (version == RunnerCcuVersion::CCU_V2) {
        if (ccuResData_.v2Res[rankId]->simulators_[dieId] == nullptr) {
            ccuResData_.v2Res[rankId]->simulators_[dieId]
                = std::make_shared<CcuSimulator>(rankId, dieId, instrStartId, endInstrId, instCnt, version);
            return ccuResData_.v2Res[rankId]->simulators_[dieId];
        } else {
            ccuResData_.v2Res[rankId]->simulators_[dieId]->Init(instrStartId, endInstrId, instCnt, version);
            return ccuResData_.v2Res[rankId]->simulators_[dieId];
        }
    } else {
        HCCL_VM_ERROR("ccu version {} not supported", static_cast<int>(version));
        return nullptr;
    }
}

void CcuResourceManager::DumpCcuInstructions(int rankId) const
{
    if (!enableDump_) {
        return;
    }
#ifndef DEVICE_STUB
    for (uint32_t dieId = 0; dieId < HcclSim::DIE_NUM; dieId++) {
        HCCL_VM_DEBUG(
            "==========================print ccu instructions start: rank[{}], dieId[{}]==========================",
            rankId, dieId);
        auto instrInfoCnt = GetInstrCnt(rankId, dieId);
        HCCL_VM_DEBUG("*******************rankId=[{}], dieId=[{}], instrCnt=[{}]", rankId, dieId, instrInfoCnt);

        for (uint32_t i = 0; i < instrInfoCnt; i++) {
            HCCL_VM_TRACE("ccu instruction info: {}: {}", i, GetInstrDescribe(rankId, dieId, i));
        }
        HCCL_VM_DEBUG(
            "==========================print ccu instructions end: rank[{}], dieId[{}]==========================",
            rankId, dieId);
    }
#endif
}

void CcuResourceManager::DumpCcuXnResouceInfo(int rankId) const
{
    for (uint32_t dieId = 0; dieId < HcclSim::DIE_NUM; dieId++) {
        HCCL_VM_DEBUG(
            "==========================All CCU XN Resouce Info Start: rank[{}], dieId[{}]===========================",
            rankId, dieId);
        HCCL_VM_DEBUG("-------------------------DieId[{}] XN Resouce Info Start-------------------------", dieId);
        for (uint32_t xnId = 0; xnId < SimCcuV1::CCU_RESOURCE_XN_MAX; xnId++) {
            auto xnValue = GetXnValue(rankId, dieId, xnId);
            if (xnValue != 0) {
                HCCL_VM_TRACE("[XN: Id = {}, Value = {:x}]", xnId, xnValue);
            }
        }
        HCCL_VM_DEBUG("-------------------------DieId[{}] XN Resouce Info End-------------------------", dieId);
        HCCL_VM_DEBUG(
            "==========================All CCU XN Resouce Info End: rank[{}], dieId[{}]===========================",
            rankId, dieId);
    }
}

void CcuResourceManager::DumpCcuGsaResouceInfo(int rankId) const
{
    for (uint32_t dieId = 0; dieId < HcclSim::DIE_NUM; dieId++) {
        HCCL_VM_DEBUG(
            "==========================All CCU GSA Resouce Info Start: rank[{}], dieId[{}]===========================",
            rankId, dieId);
        HCCL_VM_DEBUG("-------------------------DieId[{}] GSA Resouce Info Start-------------------------", dieId);
        for (uint32_t gsaId = 0; gsaId < SimCcuV1::CCU_RESOURCE_GSA_MAX; gsaId++) {
            auto gsaValue = GetGsaValue(rankId, dieId, gsaId);
            if (gsaValue != 0) {
                HCCL_VM_TRACE("[GSA: Id = {}, Value = {:x}]", gsaId, gsaValue);
            }
        }
        HCCL_VM_DEBUG("-------------------------DieId[{}] GSA Resouce Info End-------------------------", dieId);
        HCCL_VM_DEBUG(
            "==========================All CCU GSA Resouce Info End: rank[{}], dieId[{}]===========================",
            rankId, dieId);
    }
}

void CcuResourceManager::DumpCcuCkeResouceInfo(int rankId) const
{
    for (uint32_t dieId = 0; dieId < HcclSim::DIE_NUM; dieId++) {
        HCCL_VM_DEBUG(
            "==========================All CCU CKE Resouce Info Start: rank[{}], dieId[{}]===========================",
            rankId, dieId);
        HCCL_VM_DEBUG("-------------------------DieId[{}] CKE Resouce Info Start-------------------------", dieId);
        for (uint32_t ckeId = 0; ckeId < SimCcuV1::CCU_RESOURCE_MS_NUM; ckeId++) {
            auto ckeValue = GetCkeValue(rankId, dieId, ckeId);
            if (ckeValue != 0) {
                HCCL_VM_TRACE("[CKE: Id = {}, Value = {:x}]", ckeId, ckeValue);
            }
        }
        HCCL_VM_DEBUG("-------------------------DieId[{}] CKE Resouce Info End-------------------------", dieId);
        HCCL_VM_DEBUG(
            "==========================All CCU CKE Resouce Info End: rank[{}], dieId[{}]===========================",
            rankId, dieId);
    }
}

void CcuResourceManager::DumpCcuChannelResouceInfo(int rankId) const
{
    for (int dieId = 0; dieId < HcclSim::DIE_NUM; dieId++) {
        HCCL_VM_DEBUG(
            "==========================All CCU CHANNEL Resouce Info Start: rank[{}], "
            "dieId[{}]===========================",
            rankId, dieId);
        HCCL_VM_DEBUG("-------------------------DieId[{}] CHANNEL Resouce Info Start-------------------------", dieId);
        for (uint32_t chId = 0; chId < SimCcuV1::MAX_CCU_CHANNEL_NUM; chId++) {
            auto rmtCcu = GetRmtCcu(rankId, dieId, chId);
            if (rmtCcu.first != S32_INVALID && rmtCcu.second != S32_INVALID) {
                HCCL_VM_TRACE(
                    "[CHANNEL: Id = {}, Local[{}:{}] -> Remote[{}:{}]]", chId, rankId, dieId, rmtCcu.first,
                    rmtCcu.second);
            }
        }
        HCCL_VM_DEBUG("-------------------------DieId[{}] CHANNEL Resouce Info End-------------------------", dieId);
        HCCL_VM_DEBUG(
            "==========================All CCU CHANNEL Resouce Info End: rank[{}], "
            "dieId[{}]===========================",
            rankId, dieId);
    }
}

void CcuResourceManager::DumpChannelId2RmtRank(int rankId, int dieId) const
{
    if (!enableDump_) {
        return;
    }

    auto version = ccuResData_.version;
    for (uint32_t i = 0; i < SimCcuV1::MAX_CCU_CHANNEL_NUM; i++) {
        auto rmtCcu = GetRmtCcu(rankId, dieId, i);
        if (rmtCcu.first == INT32_MAX || rmtCcu.second == INT32_MAX) {
            continue;
        }
        HCCL_VM_DEBUG("channelId[{}], ccu[{}:{} --> {}:{}]", i, rankId, dieId, rmtCcu.first, rmtCcu.second);
    }
}

void CcuResourceManager::DumpCcuAllResouceInfo(int rankId) const
{
    DumpCcuXnResouceInfo(rankId);
    DumpCcuGsaResouceInfo(rankId);
    DumpCcuCkeResouceInfo(rankId);
    DumpCcuChannelResouceInfo(rankId);
}
