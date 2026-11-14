/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define HCCL_VM_MODULE "CCU_L1_RES"

#include "ccu_level1_res.h"

#include <cerrno>
#include <csignal>
#include <cstring>

#include "sim_log.h"
#include "runtime_state/sim_models.h"
#include "sim_common_api.h"
#include "level1_proxy_common.h"
#include "hccl_proxy_common.h"
#include "runtime_state/db_sim_runner_common.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "storage/table_access.h"
#include "device_spec/device_spec_types.h"

namespace HcclSim {
namespace CcuSim {

    namespace {
        /**
         * @brief 解析当前设备的 device 规格（进程内所有容量/偏移/基址的唯一来源）。
         *
         * ★ 取代此前本文件内的字面量常量表（设计文档 §4.5.6）：
         *   CCU_V1_CAPACITY/CCU_V2_CAPACITY（容量）、DIE_BASE_ADDR（0x123123123/0x456456456
         *   假基址）、V1_GSA/XN/CKE_OFFSET、V2_XN/CKE_OFFSET（区域偏移）、RES_STRIDE（步长）
         *   —— 全部改读 config/device/*.yaml（sim::runtime::GetDeviceSpec，DB 优先/yaml 回退）。
         *
         * @return 未命中返回 nullptr 并打 ERROR；调用方须显式失败，不得回退到硬编码常量。
         */
        const sim::devspec::DeviceSpec* CurrDeviceSpec(const char* who)
        {
            const uint64_t deviceId = sim::runtime::GetCurrDeviceId();
            const auto* spec = sim::runtime::GetDeviceSpec(deviceId);
            if (spec == nullptr) {
                HCCL_VM_ERROR("{}: device spec unavailable, deviceId={}", who, deviceId);
            }
            return spec;
        }

        /** @brief 进程是否仍存活（用于交换表 run 内清理：只删"写方进程已消失"的上一轮遗留行）。 */
        bool IsProcessAlive(uint64_t pid)
        {
            if (pid == 0) {
                return false;
            }
            if (kill(static_cast<pid_t>(pid), 0) == 0) {
                return true;
            }
            return errno == EPERM; // 进程存在但无信号权限 → 视为存活
        }
    } // namespace

    /** @brief 获取资源管理器进程级单例（首次调用时构造）。 */
    CcuResMgr& CcuResMgr::Instance()
    {
        static CcuResMgr instance;
        return instance;
    }

    /** @brief 签发下一个 64bit 资源句柄（进程内自增，恒非零）。 */
    uint64_t CcuResMgr::NextHandle() { return nextHandle_++; }

    /*==================== 实例生命周期 ====================*/

    uint64_t CcuResMgr::CreateIns(uint32_t dieMask)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        // 交换表 run 内清理：DB(hccl_sim.db) 跨 run 保留，而交换键的 round 从 1 重新计数，
        // 若不清会命中上一轮遗留的同 round 行。本进程首次建实例时清理一次（早于任何 launch/publish）。
        // **只删"写方进程已不存在"的行**：两进程启动无同步，可能对端已抢跑并发布了本轮的行；
        // 无条件 DeleteAll 会误删对端刚写入的行，导致读方查不到 → 等满超时（实测 3 次×10s）。
        if (!exchangeTableCleared_) {
            const auto rows = sim::runtime::Db::GetByPred<sim::runtime::CcuSyncResTab>(
                HcclSim::Storage::All<sim::runtime::CcuSyncResTab>());
            uint32_t removed = 0;
            if (rows.ok() && rows.value.has_value()) {
                for (const auto& row : *rows.value) {
                    if (!IsProcessAlive(row.ownerPid)) {
                        (void)sim::runtime::Db::Delete<sim::runtime::CcuSyncResTab>(
                            HcclSim::Storage::Eq(&sim::runtime::CcuSyncResTab::id, row.id));
                        ++removed;
                    }
                }
            }
            exchangeTableCleared_ = true;
            if (removed != 0) {
                HCCL_VM_INFO("ccu sync table run-start cleanup, removedStaleRows={}", removed);
            }
        }
        const uint64_t handle = NextHandle();
        SimCcuInsDesc desc;
        desc.insHandle = handle;
        desc.dieMask = dieMask;
        insPool_[handle] = desc;
        HCCL_VM_INFO("ccu ins created, insHandle={}, dieMask=0x{:x}", handle, dieMask);
        return handle;
    }

    /**
     * @brief 销毁 CCU 实例：级联标记其描述符/清零预约，并擦除实例条目、清理指向它的通信域绑定。
     */
    bool CcuResMgr::DestroyIns(uint64_t insHandle)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = insPool_.find(insHandle);
        if (it == insPool_.end()) {
            HCCL_VM_WARN("ccu ins destroy: insHandle={} not found", insHandle);
            return false;
        }
        // 级联清理：标记实例/描述符、清零预约，并**擦除实例条目 + 清理指向它的通信域绑定**。
        // 否则 HcclCommQueryAssignedCcuIns 会继续返回已销毁句柄、hccl 复用死实例（review #1.3）。
        it->second.destroyed = true;
        for (auto& kv : resDescPool_) {
            if (kv.second.insHandle == insHandle) {
                kv.second.destroyed = true;
            }
        }
        for (auto& kv : acqVarPool_) {
            if (kv.second.insHandle == insHandle) {
                kv.second.num = 0;
            }
        }
        for (auto bindIt = commBinding_.begin(); bindIt != commBinding_.end();) {
            if (bindIt->second == insHandle) {
                HCCL_VM_INFO("ccu ins destroy: unbind commId={} from destroyed insHandle={}", bindIt->first, insHandle);
                bindIt = commBinding_.erase(bindIt);
            } else {
                ++bindIt;
            }
        }
        insPool_.erase(it);
        HCCL_VM_INFO("ccu ins destroyed, insHandle={}", insHandle);
        return true;
    }

    /** @brief 实例句柄是否有效（存在且未被销毁）。 */
    bool CcuResMgr::IsValidIns(uint64_t insHandle) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = insPool_.find(insHandle);
        return (it != insPool_.end()) && !it->second.destroyed;
    }

    /*==================== 通信域绑定 ====================*/

    bool CcuResMgr::BindInsToComm(uint64_t commId, uint64_t insHandle)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        // 只允许绑定仍存活的实例（对齐接口契约"实例无效返回 HCCL_E_PARA"；review #1.3）。
        const auto insIt = insPool_.find(insHandle);
        if (insIt == insPool_.end() || insIt->second.destroyed) {
            HCCL_VM_ERROR("ccu ins bind rejected: insHandle=0x{:x} invalid or destroyed", insHandle);
            return false;
        }
        auto it = commBinding_.find(commId);
        if (it != commBinding_.end()) {
            if (it->second == insHandle) {
                return true; /* 幂等 */
            }
            HCCL_VM_ERROR("ccu ins bind rejected: commId={} already bound to insHandle={}", commId, it->second);
            return false;
        }
        commBinding_[commId] = insHandle;
        HCCL_VM_INFO("ccu ins bound, commId={}, insHandle={}", commId, insHandle);
        return true;
    }

    /** @brief 解绑通信域与 CCU 实例的映射（当前无调用者，见检视 1.3 遗留项）。 */
    bool CcuResMgr::UnbindInsFromComm(uint64_t commId)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = commBinding_.find(commId);
        if (it == commBinding_.end()) {
            return false;
        }
        HCCL_VM_INFO("ccu ins unbound, commId={}, insHandle={}", commId, it->second);
        commBinding_.erase(it);
        return true;
    }

    /** @brief 查询通信域绑定的实例句柄；若绑定项指向的实例已失效则视为未绑定。 */
    bool CcuResMgr::QueryBoundIns(uint64_t commId, uint64_t& insHandle) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = commBinding_.find(commId);
        if (it == commBinding_.end()) {
            return false;
        }
        // 防御：绑定项指向的实例必须仍存活；否则视为未绑定（防脏映射返回已销毁句柄，review #1.3）。
        const auto insIt = insPool_.find(it->second);
        if (insIt == insPool_.end() || insIt->second.destroyed) {
            HCCL_VM_WARN("ccu query bound ins: commId={} bound to invalid insHandle=0x{:x}", commId, it->second);
            return false;
        }
        insHandle = it->second;
        return true;
    }

    /*==================== 资源描述符（多资源类型） ====================*/

    uint64_t CcuResMgr::CreateResDesc(uint32_t dieId)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const uint64_t handle = NextHandle();
        SimCcuInsResDesc desc;
        desc.descHandle = handle;
        desc.dieId = dieId;
        resDescPool_[handle] = desc;
        HCCL_VM_INFO("ccu ins res desc created, descHandle={}, dieId={}", handle, dieId);
        return handle;
    }

    /** @brief 销毁资源描述符（仅置 destroyed 标记，不擦除条目）。 */
    bool CcuResMgr::DestroyResDesc(uint64_t descHandle)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = resDescPool_.find(descHandle);
        if (it == resDescPool_.end()) {
            HCCL_VM_WARN("ccu ins res desc destroy: descHandle={} not found", descHandle);
            return false;
        }
        it->second.destroyed = true;
        return true;
    }

    /** @brief 设置描述符某资源类型的请求量 reqNum（描述符缺失/已销毁时拒绝）。 */
    bool CcuResMgr::SetResDescNum(uint64_t descHandle, uint32_t resType, uint32_t reqNum)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = resDescPool_.find(descHandle);
        if (it == resDescPool_.end() || it->second.destroyed) {
            HCCL_VM_ERROR("ccu ins res desc set-num: descHandle={} not found or destroyed", descHandle);
            return false;
        }
        if (resType >= CCU_RES_TYPE_MAX) {
            HCCL_VM_ERROR("ccu ins res desc set-num: resType={} out of range", resType);
            return false;
        }
        it->second.reqNum[resType] = reqNum;
        return true;
    }

    /** @brief 读取描述符某资源类型的请求量 reqNum。 */
    bool CcuResMgr::QueryResDescNum(uint64_t descHandle, uint32_t resType, uint32_t& reqNum) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = resDescPool_.find(descHandle);
        if (it == resDescPool_.end() || it->second.destroyed || resType >= CCU_RES_TYPE_MAX) {
            return false;
        }
        reqNum = it->second.reqNum[resType];
        return true;
    }

    /** @brief 读取描述符所属的 dieId。 */
    bool CcuResMgr::QueryResDescDieId(uint64_t descHandle, uint32_t& dieId) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = resDescPool_.find(descHandle);
        if (it == resDescPool_.end() || it->second.destroyed) {
            return false;
        }
        dieId = it->second.dieId;
        return true;
    }

    /**
     * @brief 按请求量计算实批量 allocNum = min(reqNum, remaining)，并回填描述符所属实例句柄。
     * @return 描述符缺失/已销毁返回 false。
     */
    bool CcuResMgr::QueryResDescIns(uint64_t insHandle, uint64_t descHandle)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = resDescPool_.find(descHandle);
        if (it == resDescPool_.end() || it->second.destroyed) {
            return false;
        }
        it->second.insHandle = insHandle;
        /* 按类型计算实批 = min(请求, 剩余) */
        const uint32_t dieId = it->second.dieId;
        const CcuCapacity cap = GetCapacity();
        for (uint32_t t = 0; t < CCU_RES_TYPE_MAX; t++) {
            if (it->second.reqNum[t] == 0) {
                continue;
            }
            uint32_t remaining = 0;
            switch (t) {
                case 2:
                    remaining = RemainingXn(dieId);
                    break; /* VARIABLE */
                case 4:
                    remaining = RemainingCke(dieId);
                    break; /* EVENT */
                case 1:
                    remaining = cap.msNum;
                    break; /* CCU_BUF */
                case 0:
                    remaining = cap.loopEngineNum;
                    break; /* LOOP */
                case 6:
                    remaining = cap.instructionNum;
                    break; /* INSTRUCTION */
                default:
                    remaining = it->second.reqNum[t];
                    break;
            }
            it->second.allocNum[t] = (it->second.reqNum[t] <= remaining) ? it->second.reqNum[t] : remaining;
        }
        return true;
    }

    /** @brief 回填描述符各资源类型的剩余量 remainNum（供 hccl 做 CCU→AICPU 回退判定）。 */
    bool CcuResMgr::QueryRemainResDesc(uint64_t descHandle)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = resDescPool_.find(descHandle);
        if (it == resDescPool_.end() || it->second.destroyed) {
            return false;
        }
        const uint32_t dieId = it->second.dieId;
        const CcuCapacity cap = GetCapacity();
        for (uint32_t t = 0; t < CCU_RES_TYPE_MAX; t++) {
            uint32_t remain = 0;
            switch (t) {
                case 2:
                    remain = RemainingXn(dieId);
                    break;
                case 4:
                    remain = RemainingCke(dieId);
                    break;
                case 1:
                    remain = cap.msNum;
                    break;
                case 0:
                    remain = cap.loopEngineNum;
                    break;
                case 6:
                    remain = cap.instructionNum;
                    break;
                default:
                    remain = 0;
                    break;
            }
            it->second.remainNum[t] = remain;
        }
        HCCL_VM_INFO("ccu remain res desc, descHandle={}, dieId={}", descHandle, dieId);
        return true;
    }

    /*==================== 查询模式资源统计 ====================*/

    bool CcuResMgr::QueryResourceReq(
        uint64_t descHandle, const void* kernelFunc, const void** kernelArgs, uint32_t argNum, uint32_t dieId)
    {
        (void)kernelFunc;
        (void)kernelArgs;
        (void)argNum;
        (void)dieId;
        /* TODO: 以查询模式执行 kernelFunc 统计资源——需要与录制引擎配合：
         * 1. 开临时录制事务（queryOnly，不注册）
         * 2. 执行 kernelFunc
         * 3. 从录制上下文提取 handleSeq 按类型计数
         * 4. 写入 desc.reqNum[]
         * 当前阶段 desc.reqNum 由 hccl 侧 SetNum 预估，已覆盖回退判定需求。 */
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = resDescPool_.find(descHandle);
        if (it == resDescPool_.end() || it->second.destroyed) {
            return false;
        }
        HCCL_VM_INFO("ccu query resource req (pass-through), descHandle=0x{:x}", descHandle);
        return true;
    }

    /*==================== 实例级变量/事件 ====================*/

    uint64_t CcuResMgr::VariableAlloc(uint64_t insHandle, uint32_t dieId, uint32_t num)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        SeedWaterlineLocked();
        // 已持 mutex_，禁止调用 IsValidIns（其内部对同一把非递归 mutex_ 二次加锁会自死锁），锁内直查。
        const auto insIter = insPool_.find(insHandle);
        if (insIter == insPool_.end() || insIter->second.destroyed) {
            HCCL_VM_ERROR("ccu variable alloc: insHandle={} invalid", insHandle);
            return 0;
        }
        if (dieId >= 2 || num == 0) {
            HCCL_VM_ERROR("ccu variable alloc: dieId={} or num={} invalid", dieId, num);
            return 0;
        }
        if (xnAllocated_[dieId] + num > GetCapacity().xnNum) {
            HCCL_VM_ERROR(
                "ccu variable alloc: xn exhausted, dieId={}, allocated={}, reqNum={}, capacity={}", dieId,
                xnAllocated_[dieId], num, GetCapacity().xnNum);
            return 0;
        }
        const uint64_t handle = NextHandle();
        SimCcuAcquiredVar acq;
        acq.acqHandle = handle;
        acq.insHandle = insHandle;
        acq.dieId = dieId;
        acq.num = num;
        acq.isEvent = false;
        const CcuResLayout layout = GetLayout();
        acq.baseVa = GetXnBaseAddr(dieId) + static_cast<uint64_t>(xnAllocated_[dieId]) * layout.xnStride;
        acqVarPool_[handle] = acq;
        xnAllocated_[dieId] += num;
        HCCL_VM_INFO(
            "ccu variable acquired, acqHandle={}, insHandle={}, dieId={}, num={}, baseVa=0x{:x}", handle, insHandle,
            dieId, num, acq.baseVa);
        return handle;
    }

    /**
     * @brief 预约一段连续 Event(CKE) 资源并记录 VA：baseVa = CKE 区基址 + 分配水位*步长。
     * @return 预约句柄；实例失效/参数非法/容量不足返回 0。
     */
    uint64_t CcuResMgr::EventAlloc(uint64_t insHandle, uint32_t dieId, uint32_t num)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        SeedWaterlineLocked();
        // 同 VariableAlloc：锁内直查，避免 IsValidIns 二次加锁自死锁。
        const auto insIter = insPool_.find(insHandle);
        if (insIter == insPool_.end() || insIter->second.destroyed) {
            HCCL_VM_ERROR("ccu event alloc: insHandle={} invalid", insHandle);
            return 0;
        }
        if (dieId >= 2 || num == 0) {
            HCCL_VM_ERROR("ccu event alloc: dieId={} or num={} invalid", dieId, num);
            return 0;
        }
        if (ckeAllocated_[dieId] + num > GetCapacity().ckeNum) {
            HCCL_VM_ERROR(
                "ccu event alloc: cke exhausted, dieId={}, allocated={}, reqNum={}, capacity={}", dieId,
                ckeAllocated_[dieId], num, GetCapacity().ckeNum);
            return 0;
        }
        const uint64_t handle = NextHandle();
        SimCcuAcquiredVar acq;
        acq.acqHandle = handle;
        acq.insHandle = insHandle;
        acq.dieId = dieId;
        acq.num = num;
        acq.isEvent = true;
        const CcuResLayout layout = GetLayout();
        acq.baseVa = GetCkeBaseAddr(dieId) + static_cast<uint64_t>(ckeAllocated_[dieId]) * layout.ckeStride;
        acqVarPool_[handle] = acq;
        ckeAllocated_[dieId] += num;
        HCCL_VM_INFO(
            "ccu event acquired, acqHandle={}, insHandle={}, dieId={}, num={}, baseVa=0x{:x}", handle, insHandle, dieId,
            num, acq.baseVa);
        return handle;
    }

    /** @brief 取 Variable 预约第 index 个元素的 VA（越界/类型不符/已失效返回 false）。 */
    bool CcuResMgr::VariableGetAddr(uint64_t acqHandle, uint32_t index, uint64_t& va) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = acqVarPool_.find(acqHandle);
        if (it == acqVarPool_.end() || it->second.num == 0 || it->second.isEvent || index >= it->second.num) {
            return false;
        }
        va = it->second.baseVa + index * sizeof(uint64_t);
        return true;
    }

    /** @brief 取 Event 预约第 index 个元素的 VA（越界/类型不符/已失效返回 false）。 */
    bool CcuResMgr::EventGetAddr(uint64_t acqHandle, uint32_t index, uint64_t& va) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = acqVarPool_.find(acqHandle);
        if (it == acqVarPool_.end() || it->second.num == 0 || !it->second.isEvent || index >= it->second.num) {
            return false;
        }
        va = it->second.baseVa + index * GetLayout().ckeStride;
        return true;
    }

    /*==================== 容量查询 ====================*/

    SimCcuVersion CcuResMgr::GetCcuVersion() const
    {
        // ★ 版本改由 device 规格的 ccu_version 决定，取代原先对 soc_version 的字面量
        //   strcmp（"Ascend950"/"Ascend960"，设计文档 §4.5.6：5 处 strncmp 单点化）。
        const auto* spec = CurrDeviceSpec("GetCcuVersion");
        if (spec == nullptr) {
            HCCL_VM_WARN("ccu version: device spec unavailable, fallback to V1");
            return SimCcuVersion::V1;
        }
        switch (spec->ccuVersion) {
            case sim::devspec::CcuVersion::CCU_V2:
                return SimCcuVersion::V2;
            case sim::devspec::CcuVersion::CCU_V1:
                return SimCcuVersion::V1;
            default:
                HCCL_VM_WARN(
                    "ccu version: unsupported ccu_version={}, fallback to V1",
                    sim::devspec::CcuVersionToString(spec->ccuVersion));
                return SimCcuVersion::V1;
        }
    }

    /** @brief 资源地址布局：区偏移/步长全部取自 device 规格 regions（无 GSA 区域时 gsaOffset=0）。 */
    CcuResLayout CcuResMgr::GetLayout() const
    {
        CcuResLayout layout;
        layout.version = GetCcuVersion();
        const auto* spec = CurrDeviceSpec("GetLayout");
        if (spec == nullptr) {
            return layout; // 偏移/步长保持 0，调用方按越界/失败处理，不回退硬编码
        }
        auto fillRegion = [spec](sim::devspec::CcuResType type, uint64_t& offset, uint32_t& stride) {
            const auto* region = spec->FindRegion(type);
            if (region == nullptr) {
                offset = 0;
                stride = 0;
                return;
            }
            offset = region->offset;
            stride = region->entrySize;
        };
        fillRegion(sim::devspec::CcuResType::XN, layout.xnOffset, layout.xnStride);
        fillRegion(sim::devspec::CcuResType::GSA, layout.gsaOffset, layout.gsaStride); // A6 无 GSA → 0
        fillRegion(sim::devspec::CcuResType::CKE, layout.ckeOffset, layout.ckeStride);
        return layout;
    }

    /** @brief XN 区基址 = dieBase + xnOffset。 */
    uint64_t CcuResMgr::GetXnBaseAddr(uint32_t dieId) const { return GetDieBaseAddr(dieId) + GetLayout().xnOffset; }

    /** @brief GSA 区基址 = dieBase + gsaOffset（无 GSA 区域返回 0）。 */
    uint64_t CcuResMgr::GetGsaBaseAddr(uint32_t dieId) const
    {
        const CcuResLayout layout = GetLayout();
        if (layout.gsaOffset == 0) {
            return 0; // 无 GSA 区域（A6）
        }
        return GetDieBaseAddr(dieId) + layout.gsaOffset;
    }

    /** @brief CKE 区基址 = dieBase + ckeOffset。 */
    uint64_t CcuResMgr::GetCkeBaseAddr(uint32_t dieId) const { return GetDieBaseAddr(dieId) + GetLayout().ckeOffset; }

    /** @brief 当前设备的 CCU 容量：全部取自 device 规格 regions/ms（GetCaps 派生）。 */
    CcuCapacity CcuResMgr::GetCapacity() const
    {
        CcuCapacity cap{};
        const auto* spec = CurrDeviceSpec("GetCapacity");
        if (spec == nullptr) {
            return cap; // 容量全 0：后续分配将因 exhausted 显式失败，不回退硬编码常量
        }
        const sim::devspec::CcuCaps caps = spec->GetCaps();
        cap.xnNum = caps.xnNum;
        cap.gsaNum = caps.gsaNum;
        cap.ckeNum = caps.ckeNum;
        cap.msNum = caps.msNum;
        cap.loopEngineNum = caps.loopEngineNum;
        cap.missionNum = caps.missionNum;
        cap.instructionNum = caps.instructionNum;
        return cap;
    }

    /**
     * @brief 指定 die 的 CCU 资源基址（per-rank per-die 唯一真实地址）。
     *
     * ★ 取代原 per-die 固定假基址（die0=0x123123123, die1=0x456456456，设计文档 §2.2）：
     *   走全项目唯一入口 sim::runtime::GetCcuResourceBaseAddr（DB Ccu.resource_addr 优先、
     *   yaml 规格回退），与 level2 resourceAddr 同源。解析失败返回 0 并打 ERROR。
     */
    uint64_t CcuResMgr::GetDieBaseAddr(uint32_t dieId) const
    {
        if (dieId >= 2) {
            HCCL_VM_ERROR("GetDieBaseAddr: dieId {} out of range", dieId);
            return 0;
        }
        const uint64_t deviceId = sim::runtime::GetCurrDeviceId();
        uint64_t base = 0;
        if (!sim::runtime::GetCcuResourceBaseAddr(deviceId, dieId, base) || base == 0) {
            HCCL_VM_ERROR("GetDieBaseAddr: resolve failed, deviceId={}, dieId={}", deviceId, dieId);
            return 0;
        }
        return base;
    }

    uint32_t CcuResMgr::SyncSlotNum() const
    {
        const auto* spec = CurrDeviceSpec("SyncSlotNum");
        if (spec == nullptr) {
            HCCL_VM_ERROR("SyncSlotNum: device spec unavailable, sync slots fall back to 0（不预留同步区）");
            return 0;
        }
        return spec->reservedSyncSlots;
    }

    void CcuResMgr::SeedWaterlineLocked() const
    {
        if (waterlineSeeded_) {
            return;
        }
        const uint32_t slots = SyncSlotNum();
        for (uint32_t d = 0; d < 2; ++d) {
            xnAllocated_[d] = slots;
            gsaAllocated_[d] = slots;
            ckeAllocated_[d] = slots;
        }
        waterlineSeeded_ = true;
    }

    /** @brief 指定 die 的剩余 XN 数 = capacity - 分配水位（dieId 越界返回 0）。 */
    uint32_t CcuResMgr::RemainingXn(uint32_t dieId) const
    {
        if (dieId >= 2) {
            return 0;
        }
        SeedWaterlineLocked();
        const uint32_t cap = GetCapacity().xnNum;
        return (xnAllocated_[dieId] < cap) ? (cap - xnAllocated_[dieId]) : 0;
    }

    /** @brief 指定 die 的剩余 GSA 数（无 GSA 区域容量为 0）。 */
    uint32_t CcuResMgr::RemainingGsa(uint32_t dieId) const
    {
        if (dieId >= 2) {
            return 0;
        }
        SeedWaterlineLocked();
        const uint32_t cap = GetCapacity().gsaNum; // A6 无 GSA = 0
        return (gsaAllocated_[dieId] < cap) ? (cap - gsaAllocated_[dieId]) : 0;
    }

    /** @brief 指定 die 的剩余 CKE 数。 */
    uint32_t CcuResMgr::RemainingCke(uint32_t dieId) const
    {
        if (dieId >= 2) {
            return 0;
        }
        SeedWaterlineLocked();
        const uint32_t cap = GetCapacity().ckeNum;
        return (ckeAllocated_[dieId] < cap) ? (cap - ckeAllocated_[dieId]) : 0;
    }

} // namespace CcuSim
} // namespace HcclSim
