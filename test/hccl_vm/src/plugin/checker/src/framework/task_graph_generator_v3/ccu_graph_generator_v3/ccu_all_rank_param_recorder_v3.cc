/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_all_rank_param_recorder_v3.h"
#include <sys/types.h>
#include <cstdint>
#include "dfx/ccu_resource_stats_dfx.h"
#include "device_spec/ccu_address_space.h"
#include "utils/ccu_base_resolver.h"
#include "sim_log.h"
#include "utils/error_codes.h"

namespace HcclSim {
namespace TaskGraphGeneratorV3 {

    AllRankParamRecorder* AllRankParamRecorder::Global()
    {
        static AllRankParamRecorder* allRankParamRecorder = new AllRankParamRecorder;
        return allRankParamRecorder;
    }

    void AllRankParamRecorder::Reset()
    {
        curXn.clear();
        curGSA.clear();
        curCKE.clear();
        curHBM.clear();
        seenPost.clear();
        postNodeMeta.clear();
        loopGroupRegSnapshot.clear();
        devType_ = DevType::DEV_TYPE_COUNT;
        ccu_resource_base_addr_.clear();
        return;
    }

    void AllRankParamRecorder::InitParam()
    {
        curXn.clear();
        curGSA.clear();
        curHBM.clear();
        seenPost.clear();
        postNodeMeta.clear();
        loopGroupRegSnapshot.clear();
    }

    void AllRankParamRecorder::RegisterPostNode(TaskNode* node, const CcuPostNodeMetaV3& meta)
    {
        if (node == nullptr) {
            return;
        }
        postNodeMeta[node] = meta;
    }

    uint32_t AllRankParamRecorder::GetPostRemainingCkeMask(const TaskNode* node) const
    {
        auto it = postNodeMeta.find(node);
        if (it == postNodeMeta.end()) {
            return 0;
        }
        return it->second.remainingCkeMask;
    }

    void AllRankParamRecorder::SetPostRemainingCkeMask(TaskNode* node, uint32_t remainingCkeMask)
    {
        auto it = postNodeMeta.find(node);
        if (it == postNodeMeta.end()) {
            return;
        }
        it->second.remainingCkeMask = static_cast<uint16_t>(remainingCkeMask);
    }

    const CcuPostNodeMetaV3* AllRankParamRecorder::GetPostNodeMeta(const TaskNode* node) const
    {
        auto it = postNodeMeta.find(node);
        if (it == postNodeMeta.end()) {
            return nullptr;
        }
        return &it->second;
    }

    HcclResult AllRankParamRecorder::SetXn(DeviceId deviceId, uint32_t dieId, uint16_t xnId, uint64_t xnValue)
    {
        curXn[deviceId][dieId][xnId] = xnValue;
        // CCU 资源统计：写入时按 id 记账（开关关闭/无算子上下文时钩子内部短路）
        CcuResourceStats::Collector::GetInstance().RecordXn(deviceId, dieId, xnId);
        return HCCL_SUCCESS;
    }

    HcclResult AllRankParamRecorder::SetGSA(DeviceId deviceId, uint32_t dieId, uint16_t gsaId, uint64_t gsaValue)
    {
        curGSA[deviceId][dieId][gsaId] = gsaValue;
        CcuResourceStats::Collector::GetInstance().RecordGsa(deviceId, dieId, gsaId);
        return HCCL_SUCCESS;
    }

    HcclResult AllRankParamRecorder::SetCKE(DeviceId deviceId, uint32_t dieId, uint16_t ckeId, uint16_t ckeValue)
    {
        curCKE[deviceId][dieId][ckeId] = ckeValue;
        CcuResourceStats::Collector::GetInstance().RecordCke(deviceId, dieId, ckeId);
        return HCCL_SUCCESS;
    }

    HcclResult AllRankParamRecorder::GetXn(DeviceId deviceId, uint32_t dieId, uint16_t xnId, uint64_t& xnValue)
    {
        if (curXn.find(deviceId) != curXn.end()) {
            if (curXn[deviceId].find(dieId) != curXn[deviceId].end()) {
                if (curXn[deviceId][dieId].find(xnId) != curXn[deviceId][dieId].end()) {
                    xnValue = curXn[deviceId][dieId][xnId];
                    return HCCL_SUCCESS;
                }
            }
        }
        xnValue = 0;
        return HCCL_E_PARA;
    }

    HcclResult AllRankParamRecorder::GetGSA(DeviceId deviceId, uint32_t dieId, uint16_t gsaId, uint64_t& gsaValue)
    {
        if (curGSA.find(deviceId) != curGSA.end()) {
            if (curGSA[deviceId].find(dieId) != curGSA[deviceId].end()) {
                if (curGSA[deviceId][dieId].find(gsaId) != curGSA[deviceId][dieId].end()) {
                    gsaValue = curGSA[deviceId][dieId][gsaId];
                    return HCCL_SUCCESS;
                }
            }
        }
        gsaValue = 0;
        return HCCL_E_PARA;
    }

    HcclResult AllRankParamRecorder::GetCKE(DeviceId deviceId, uint32_t dieId, uint16_t ckeId, uint16_t& ckeValue)
    {
        if (curCKE.find(deviceId) != curCKE.end()) {
            if (curCKE[deviceId].find(dieId) != curCKE[deviceId].end()) {
                if (curCKE[deviceId][dieId].find(ckeId) != curCKE[deviceId][dieId].end()) {
                    ckeValue = curCKE[deviceId][dieId][ckeId];
                    return HCCL_SUCCESS;
                }
            }
        }

        ckeValue = 0;
        return HCCL_SUCCESS;
    }

    std::map<uint16_t, uint64_t> AllRankParamRecorder::GetXnSnapshot(DeviceId deviceId, uint32_t dieId) const
    {
        auto deviceIt = curXn.find(deviceId);
        if (deviceIt == curXn.end()) {
            return {};
        }
        auto dieIt = deviceIt->second.find(dieId);
        if (dieIt == deviceIt->second.end()) {
            return {};
        }
        return dieIt->second;
    }

    std::map<uint16_t, uint64_t> AllRankParamRecorder::GetGSASnapshot(DeviceId deviceId, uint32_t dieId) const
    {
        auto deviceIt = curGSA.find(deviceId);
        if (deviceIt == curGSA.end()) {
            return {};
        }
        auto dieIt = deviceIt->second.find(dieId);
        if (dieIt == deviceIt->second.end()) {
            return {};
        }
        return dieIt->second;
    }

    std::map<uint16_t, uint16_t> AllRankParamRecorder::GetCKESnapshot(DeviceId deviceId, uint32_t dieId) const
    {
        auto deviceIt = curCKE.find(deviceId);
        if (deviceIt == curCKE.end()) {
            return {};
        }
        auto dieIt = deviceIt->second.find(dieId);
        if (dieIt == deviceIt->second.end()) {
            return {};
        }
        return dieIt->second;
    }

    void AllRankParamRecorder::RecordLoopGroupRegSnapshot(
        DeviceId deviceId, uint32_t dieId, uint32_t pc, uint64_t xpValue, uint64_t xmValue)
    {
        loopGroupRegSnapshot[deviceId][dieId][pc] = std::make_pair(xpValue, xmValue);
    }

    uint16_t AllRankParamRecorder::GetMaxUsedXnId(DeviceId deviceId, uint32_t dieId) const
    {
        auto deviceIt = curXn.find(deviceId);
        if (deviceIt == curXn.end()) {
            return UINT16_MAX;
        }
        auto dieIt = deviceIt->second.find(dieId);
        if (dieIt == deviceIt->second.end() || dieIt->second.empty()) {
            return UINT16_MAX;
        }
        // std::map 按 key 升序排列，最后一条即最大 xnId
        return dieIt->second.rbegin()->first;
    }

    HcclResult AllRankParamRecorder::CheckAllPostMatch()
    {
        for (const auto& devicePair : seenPost) {
            for (const auto& diePair : devicePair.second) {
                for (const auto& regPair : diePair.second) {
                    for (const auto* post : regPair.second) {
                        const auto* meta = GetPostNodeMeta(post);
                        if (meta != nullptr) {
                            HCCL_VM_WARN(
                                "{} Found CCU post/local-post tasks that were never consumed by "
                                "any Wait task:\n  unconsumedPostCount={}, remainingCkeMask=0x{:x}, isLocal={}\n"
                                "  firstUnconsumedPostNode={}",
                                MakeErrorCodeText(ErrorCode::GRAPH_UNMATCHED).c_str(), regPair.second.size(),
                                meta->remainingCkeMask, meta->isLocal,
                                post == nullptr ? "node=null" : post->Describe().c_str());
                            continue;
                        }
                        HCCL_VM_WARN(
                            "{} Found CCU post/local-post tasks that were never consumed by any "
                            "Wait task:\n  unconsumedPostCount={}\n  firstUnconsumedPostNode={}",
                            MakeErrorCodeText(ErrorCode::GRAPH_UNMATCHED).c_str(), regPair.second.size(),
                            post == nullptr ? "node=null" : post->Describe().c_str());
                    }
                }
            }
        }

        return HCCL_SUCCESS;
    }

    HcclResult
    AllRankParamRecorder::SetHBM(DeviceId deviceId, uint32_t dieId, uint64_t hbmAddr, const std::vector<uint64_t>& data)
    {
        if (data.size() % 8 != 0) {
            return HCCL_E_PARA;
        }
        curHBM[deviceId][dieId][hbmAddr] = data;
        return HCCL_SUCCESS;
    }

    HcclResult
    AllRankParamRecorder::GetHBM(DeviceId deviceId, uint32_t dieId, uint64_t hbmAddr, std::vector<uint64_t>& data)
    {
        if (curHBM.find(deviceId) != curHBM.end()) {
            if (curHBM[deviceId].find(dieId) != curHBM[deviceId].end()) {
                if (curHBM[deviceId][dieId].find(hbmAddr) != curHBM[deviceId][dieId].end()) {
                    data = curHBM[deviceId][dieId][hbmAddr];
                    return HCCL_SUCCESS;
                }
            }
        }
        data.clear();
        return HCCL_E_PARA;
    }

    namespace {
        /**
         * CcuComponerntType（checker 对外枚举）⟷ devspec::CcuResType（device 规格区域类型）。
         *
         * ★ 地址↔id 的区偏移/步长/容量/MS 几何全部由 CcuAddressSpace 按 device 规格提供，
         *   此处仅做类型标签映射——取代原 XN_*_ADDR_A6 / CCU_MS_START_ADDR / MS_INTERLEAVE_NUM
         *   等字面量常量块与 findBaseAddr/findTypeByAddr/findBaseSize 手写转换
         *   （设计文档 §2.3/§2.4/§2.5、§4.5.5：删除常量块与 4 个转换函数，改持 CcuAddressSpace）。
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

    uint64_t AllRankParamRecorder::GetCcuBaseAddr(DeviceId deviceId, uint32_t dieId) const
    {
        const auto devIter = ccu_resource_base_addr_.find(deviceId);
        if (devIter == ccu_resource_base_addr_.end() || dieId >= devIter->second.size()) {
            HCCL_VM_ERROR(
                "{} ccu base addr not initialized, deviceId={}, dieId={}",
                MakeErrorCodeText(ErrorCode::GRAPH_OUT_OF_RANGE).c_str(), deviceId, dieId);
            return 0;
        }
        return devIter->second[dieId];
    }

    HcclResult AllRankParamRecorder::GetMSIdByAddr(DeviceId deviceId, uint32_t dieId, uint64_t addr, uint16_t& msId)
    {
        const uint64_t base = GetCcuBaseAddr(deviceId, dieId);
        const auto* space = GetCcuAddrSpaceByDeviceId(deviceId);
        if (base == 0 || space == nullptr) {
            return HCCL_E_PARA;
        }
        uint32_t id = 0;
        if (!space->MsAddrToId(base, addr, id) || id > UINT16_MAX) {
            HCCL_VM_ERROR(
                "{} Address does not fall into any known MS address range, deviceId={}, dieId={}, rawAddr={}",
                MakeErrorCodeText(ErrorCode::GRAPH_ADDRESS_INVALID).c_str(), deviceId, dieId, addr);
            return HCCL_E_PARA;
        }
        msId = static_cast<uint16_t>(id);
        return HCCL_SUCCESS;
    }

    // 通过XnId所在的地址值来找到XnId以及寄存器的类型
    HcclResult AllRankParamRecorder::GetXnAndTypeIdByAddr(
        DeviceId deviceId, uint32_t dieId, uint64_t xnAddr, CcuComponerntType& type, uint16_t& xnId)
    {
        type = CcuComponerntType::UNKNOWN;
        const uint64_t base = GetCcuBaseAddr(deviceId, dieId);
        const auto* space = GetCcuAddrSpaceByDeviceId(deviceId);
        if (base == 0 || space == nullptr) {
            return HCCL_E_PARA;
        }
        sim::devspec::CcuResType resType = sim::devspec::CcuResType::UNKNOWN;
        uint64_t id = 0;
        if (!space->AddrToResId(base, xnAddr, resType, id)) {
            HCCL_VM_ERROR(
                "{} Address does not belong to any known CCU component range, addr={}, deviceId={}, dieId={}, "
                "dieBaseAddr={}",
                MakeErrorCodeText(ErrorCode::GRAPH_ADDRESS_INVALID).c_str(), xnAddr, deviceId, dieId, base);
            return HCCL_E_PARA;
        }
        if (id > UINT16_MAX) {
            HCCL_VM_ERROR(
                "{} register id out of range, addr={}, id={}", MakeErrorCodeText(ErrorCode::GRAPH_OUT_OF_RANGE).c_str(),
                xnAddr, id);
            return HCCL_E_PARA;
        }
        type = ToComponentType(resType);
        xnId = static_cast<uint16_t>(id);
        return HCCL_SUCCESS;
    }

    // 通过XnId所在的地址值来找到XnId
    HcclResult AllRankParamRecorder::GetXnIdByAddr(
        DeviceId deviceId, uint32_t dieId, CcuComponerntType type, uint64_t xnAddr, uint16_t& xnId)
    {
        const uint64_t base = GetCcuBaseAddr(deviceId, dieId);
        const auto* space = GetCcuAddrSpaceByDeviceId(deviceId);
        if (base == 0 || space == nullptr) {
            return HCCL_E_PARA;
        }
        uint64_t id = 0;
        // ★ 带完整边界校验（AddrToIdOfType 内置 id < count），修掉原 "需要判断界限 todo"。
        if (!space->AddrToIdOfType(base, ToResType(type), xnAddr, id) || id > UINT16_MAX) {
            HCCL_VM_ERROR(
                "{} Address does not belong to the requested CCU component range, addr={}, deviceId={}, dieId={}, "
                "type={}",
                MakeErrorCodeText(ErrorCode::GRAPH_ADDRESS_INVALID).c_str(), xnAddr, deviceId, dieId,
                static_cast<uint32_t>(type));
            return HCCL_E_PARA;
        }
        xnId = static_cast<uint16_t>(id);
        return HCCL_SUCCESS;
    }

    // 通过资源id反查地址
    HcclResult AllRankParamRecorder::GetAddrByXnId(
        DeviceId deviceId, uint32_t dieId, CcuComponerntType type, uint16_t xnId, uint64_t& xnAddr)
    {
        const uint64_t base = GetCcuBaseAddr(deviceId, dieId);
        const auto* space = GetCcuAddrSpaceByDeviceId(deviceId);
        if (base == 0 || space == nullptr) {
            return HCCL_E_PARA;
        }
        if (!space->ResIdToAddr(base, ToResType(type), xnId, xnAddr)) {
            HCCL_VM_ERROR(
                "{} register id out of range when converting to address, deviceId={}, dieId={}, type={}, id={}",
                MakeErrorCodeText(ErrorCode::GRAPH_OUT_OF_RANGE).c_str(), deviceId, dieId, static_cast<uint32_t>(type),
                xnId);
            return HCCL_E_PARA;
        }
        return HCCL_SUCCESS;
    }

} // namespace TaskGraphGeneratorV3
} // namespace HcclSim
