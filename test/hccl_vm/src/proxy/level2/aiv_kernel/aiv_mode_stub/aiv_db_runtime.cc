/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

/**
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * for the full text of the License.
 */

#include "aiv_db_runtime.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unistd.h>

#include "ai_core_stub.h"
#include "db_sim_runner_common.h"
#include "db_sim_runner_db.h"
#include "db_sim_runner_ops.h"
#include "sim_log.h"
#include "store_sim_device_memory_manager.h"

namespace AivSim {
namespace {
std::atomic<uint64_t> g_aivUbSequence{0};

bool ResolveCurrentMemoryContext(sim::Runner &runner, uint64_t &deviceId) {
    const uint64_t serverId = sim::GetCurServerId();
    if (serverId == 0 || !sim::GetCurrRunnerTls(serverId, runner)) {
        HCCL_VM_ERROR(
            "cannot resolve current runner for AIV UB allocation, serverId={}",
            serverId);
        return false;
    }

    deviceId = sim::GetCurrDeviceId();
    if (runner.current_ctx_id == 0 || deviceId == 0) {
        HCCL_VM_ERROR(
            "invalid AIV UB memory context, contextId={}, deviceId={}",
            runner.current_ctx_id, deviceId);
        return false;
    }
    return true;
}

std::string MakeAivUbShmName() {
    char name[sizeof(decltype(sim::PhyMemBlock::name))] = {};
    const uint64_t sequence =
        g_aivUbSequence.fetch_add(1, std::memory_order_relaxed);
    std::snprintf(name, sizeof(name), "DEV_AIV_UB_%d_%llu",
                  static_cast<int>(getpid()),
                  static_cast<unsigned long long>(sequence));
    return name;
}

bool AllocateAivUb(AivKernelExecutor &executor, const sim::Runner &runner,
                   uint64_t deviceId, size_t blockId) {
    const std::string shmName = MakeAivUbShmName();
    void *hostAddress = sim::DeviceMemoryManager::GetInstance().AllocPhyMem(
        shmName.c_str(), deviceId, AIV_UB_SIZE);
    if (hostAddress == nullptr) {
        HCCL_VM_ERROR("failed to allocate AIV UB shared memory, blockId={}, "
                      "name={}, size={}",
                      blockId, shmName, AIV_UB_SIZE);
        return false;
    }

    sim::PhyMemBlock physicalRecord{};
    physicalRecord.device_id = deviceId;
    std::snprintf(physicalRecord.name, sizeof(physicalRecord.name), "%s",
                  shmName.c_str());
    physicalRecord.size = AIV_UB_SIZE;
    physicalRecord.ref_count = 1;
    physicalRecord.is_freed = 0;
    const uint64_t physicalMemId =
        RunnerDB::Add<sim::PhyMemBlock>(physicalRecord);
    if (physicalMemId == 0) {
        HCCL_VM_ERROR(
            "failed to insert AIV UB PhyMemBlock, blockId={}, name={}", blockId,
            shmName);
        return false;
    }

    void *virtualAddress = sim::DeviceMemoryManager::GetInstance().AllocVirMem(
        deviceId, AIV_UB_SIZE);
    if (virtualAddress == nullptr) {
        HCCL_VM_ERROR(
            "failed to allocate AIV UB virtual address, blockId={}, size={}",
            blockId, AIV_UB_SIZE);
        return false;
    }

    sim::VirtualMemBlock virtualRecord{};
    virtualRecord.start_ptr = reinterpret_cast<uint64_t>(virtualAddress);
    virtualRecord.dev_mapped_ptr = virtualRecord.start_ptr;
    virtualRecord.is_dev_access = 0;
    virtualRecord.size = AIV_UB_SIZE;
    virtualRecord.ctx_id = runner.current_ctx_id;
    virtualRecord.rank_id = executor.GetRankId();
    virtualRecord.device_id = deviceId;
    virtualRecord.phy_mem_id = physicalMemId;
    virtualRecord.owner_pid =
        runner.pid == 0 ? static_cast<uint64_t>(getpid()) : runner.pid;
    virtualRecord.src_type = static_cast<uint8_t>(sim::VIR_MEM_TYPE_DEV);
    virtualRecord.policy = 0;
    const uint64_t virtualMemId =
        RunnerDB::Add<sim::VirtualMemBlock>(virtualRecord);
    if (virtualMemId == 0) {
        HCCL_VM_ERROR("failed to insert AIV UB VirtualMemBlock, blockId={}, "
                      "address={:p}, phyMemId={}",
                      blockId, virtualAddress, physicalMemId);
        return false;
    }

    sim::DeviceMemoryManager::GetInstance().MapDevPtrHostPtr(virtualAddress,
                                                             hostAddress);
    if (!executor.AddUbBuffer({virtualRecord.start_ptr, virtualRecord.size})) {
        HCCL_VM_ERROR(
            "failed to retain AIV UB address, blockId={}, virtualMemId={}",
            blockId, virtualMemId);
        return false;
    }

    HCCL_VM_INFO("allocated persistent AIV UB, rank={}, blockId={}, "
                 "virtualAddress={:p}, hostAddress={:p}, "
                 "size={}, virtualMemId={}, physicalMemId={}, shmName={}",
                 executor.GetRankId(), blockId, virtualAddress, hostAddress,
                 AIV_UB_SIZE, virtualMemId, physicalMemId, shmName);
    return true;
}
} // namespace

bool InitAivUB(AivKernelExecutor &executor, size_t blockNum) {
    const size_t retainedBufferCount = executor.GetUbBufferInfos().size();
    if (retainedBufferCount >= blockNum) {
        HCCL_VM_DEBUG("reuse persistent AIV hardware UB addresses, "
                      "retained={}, requested={}",
                      retainedBufferCount, blockNum);
        return true;
    }

    sim::Runner runner{};
    uint64_t deviceId = 0;
    if (!ResolveCurrentMemoryContext(runner, deviceId)) {
        return false;
    }

    // UB is modeled as hardware state, not operator state. The first AIV kernel
    // initialization allocates one shared-memory-backed virtual range for every
    // requested core. If a later launch requests more cores, or a previous
    // attempt stopped partway, only the missing tail is allocated. Smaller
    // later launches retain and reuse the full address set without consulting
    // operator identity. The simulator host owns the DB records and shared
    // memory until simulator exit.
    for (size_t blockId = retainedBufferCount; blockId < blockNum; ++blockId) {
        if (!AllocateAivUb(executor, runner, deviceId, blockId)) {
            return false;
        }
    }
    HCCL_VM_INFO("initialized persistent AIV hardware UB addresses, "
                 "coreCount={}, ubSizePerCore={}",
                 blockNum, AIV_UB_SIZE);
    return true;
}
} // namespace AivSim
