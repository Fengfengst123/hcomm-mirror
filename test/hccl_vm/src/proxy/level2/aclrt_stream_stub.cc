/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 日志染色: 模块 tag (须在 include sim_log.h 之前)
#include "db_sim_runner_db.h"
#include "sim_models.h"
#define HCCL_VM_MODULE "STREAM_STUB"

#include <cstdint>
#include <ctime>
#include <fcntl.h>
#include <iostream>
#include <sys/file.h>
#include <unistd.h>
#include <vector>

#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "db_sim_op_db_ops.h"
#include "db_sim_runner_common.h"
#include "db_sim_runner_ops.h"
#include "runtime/base.h"
#include "sim_log.h"
#include "store_dump_shm_data.h"
#include "store_sim_store_pub.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

aclError aclrtCreateStreamWithConfig(aclrtStream* stream, uint32_t priority, uint32_t flag)
{
    (void)flag;
    (void)priority;
    auto serverId = sim::GetCurServerId();
    if (serverId == 0) {
        return ACL_ERROR_INVALID_PARAM;
    }
    sim::Runner runner;
    if (!sim::GetCurrRunnerTls(serverId, runner)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    if (runner.current_ctx_id == 0) {
        HCCL_VM_ERROR("invalid param");
        return ACL_ERROR_INVALID_PARAM;
    }
    auto currCtx = RunnerDB::GetById<sim::Context>(runner.current_ctx_id);
    if (!currCtx.has_value()) {
        HCCL_VM_ERROR("ctx not found:{:d}", runner.current_ctx_id);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto& currCtxId = currCtx->id;

    sim::Stream streamTmp{};
    streamTmp.ctx_id = currCtxId;
    streamTmp.activated = 1;
    streamTmp.priority = priority;
    streamTmp.user_tag = flag;
    auto res = RunnerDB::Add<sim::Stream>(streamTmp);

    *stream = (aclrtStream)res;
    HCCL_VM_DEBUG("id:{:d}", res);
    sim::SetLastStreamIdTls(res);
    sim::AddSyncStreamIter(res);
    return ACL_SUCCESS;
}

aclError aclrtCreateStream(aclrtStream* stream) { return aclrtCreateStreamWithConfig(stream, 0, 0); }

aclError aclrtDestroyStream(aclrtStream stream)
{
    uint64_t streamId = (uint64_t)(uintptr_t)stream;
    HCCL_VM_DEBUG("id:{:d}", streamId);
    RunnerDB::Delete<sim::Stream>(streamId);
    sim::RemoveSyncStreamIter(streamId);
    return ACL_SUCCESS;
}

aclError aclrtDestroyStreamForce(aclrtStream stream) { return aclrtDestroyStream(stream); }

aclError aclrtActiveStream(aclrtStream activeStream, aclrtStream stream)
{
    (void)stream;
    uint64_t activStreamId = (uint64_t)(uintptr_t)activeStream;
    HCCL_VM_DEBUG("id:{:d}", activStreamId);
    auto res = RunnerDB::Update<sim::Stream>(activStreamId, [](sim::Stream& stm) {
        stm.activated = 1;
    });
    if (!res) {
        HCCL_VM_ERROR("stream not found:{:d}", activStreamId);
        return ACL_ERROR_INVALID_PARAM;
    }
    return ACL_SUCCESS;
}

aclError aclrtSetStreamFailureMode(aclrtStream stream, uint64_t mode)
{
    uint64_t streamId = (uint64_t)(uintptr_t)stream;
    HCCL_VM_DEBUG("id:{:d} mode:{:d}", streamId, mode);
    RunnerDB::Update<sim::Stream>(streamId, [streamId, mode](sim::Stream& stm) {
        stm.failure_mode = mode;
    });

    return ACL_SUCCESS;
}

aclError aclrtSynchronizeStreamWithTimeout(aclrtStream stream, int32_t timeout)
{
    (void)timeout;
    uint64_t streamId = (uint64_t)(uintptr_t)stream;
    // GetMode();
    std::string mode = "checker";
    const int WAIT_COUNTDOWN = 10; // 等待20s
    HCCL_VM_DEBUG("id:{:d}", streamId);
    return ACL_SUCCESS;
}

aclError aclrtSynchronizeStream(aclrtStream stream)
{
    uint64_t streamId = (uint64_t)(uintptr_t)stream;
    uint64_t syncIdx{UINT64_MAX};
    if (!sim::NextStreamSyncIdx(streamId, syncIdx)) {
        HCCL_VM_ERROR("Get stream syncIdx failed, streamId={:d}", streamId);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    HcclTaskMetaData taskMetaData;
    taskMetaData.taskType = HccLTaskMetaType::SYNC_STREAM;
    taskMetaData.commId = 0;
    taskMetaData.deviceId = sim::GetCurrDeviceId();
    taskMetaData.rankId = UINT32_MAX;
    taskMetaData.streamId = streamId;
    taskMetaData.taskData.syncStreamTask.syncIdx = syncIdx;

    uint32_t index{0};
    auto ret = InsertTaskToCollection(&taskMetaData, &index);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("InsertTaskToCollection fail");
        return ACL_ERROR_INTERNAL_ERROR;
    }
    HCCL_VM_DEBUG("Add SYNC_STREAM task, streamId={:d}, syncIdx={:d}", streamId, syncIdx);

    // 记录 synchronize_strategy：aclrtSynchronizeStream 是 DeviceStatus
    // 的写入源， 查到对应 device 的记录就 Update，查不到就 Add。
    auto streamRecord = RunnerDB::GetById<sim::Stream>(streamId);
    if (!streamRecord.has_value()) {
        HCCL_VM_ERROR("stream not found:{:d}", streamId);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto context = RunnerDB::GetById<sim::Context>(streamRecord->ctx_id);
    if (!context.has_value()) {
        HCCL_VM_ERROR("context not found:{:d}", streamRecord->ctx_id);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto device = RunnerDB::GetById<sim::Device>(context->device_id);
    if (!device.has_value()) {
        HCCL_VM_ERROR("device not found:{:d}", context->device_id);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto deviceStatus
        = RunnerDB::GetOneByPred<sim::DeviceStatus>([deviceId = device->id](const sim::DeviceStatus& status) {
              return status.device_id == deviceId;
          });
    if (deviceStatus.second) {
        RunnerDB::Update<sim::DeviceStatus>(deviceStatus.first.id, [](sim::DeviceStatus& status) {
            status.synchronize_strategy = 1;
        });
    } else {
        sim::DeviceStatus newStatus{};
        newStatus.device_id = device->id;
        newStatus.synchronize_strategy = 1;
        RunnerDB::Add<sim::DeviceStatus>(newStatus);
    }

    auto pluginRows = RunnerDB::GetByPred<sim::Plugin>([](const sim::Plugin& plugin) {
        return std::string(plugin.tag) == "runner";
    });
    if (pluginRows.empty()) { // Runner 未启动
        return ACL_SUCCESS;
    }

    // Runner 已启动，等待stream上所有task运行结束
    while (true) {
        std::vector<sim::OpTaskTab> tasks;
        if (sim::QueryOpTasksByStreamId(streamId, tasks) != 0) {
            HCCL_VM_ERROR("QueryOpTasksByStreamId fail");
            return ACL_ERROR_INTERNAL_ERROR;
        }
        uint64_t isDoneNum = 0;
        for (const auto& task : tasks) {
            if (task.isDone) {
                isDoneNum++;
            }
        }

        HCCL_VM_INFO("Waiting for stream[{:d}]... {:d}/{:d}", streamId, isDoneNum, tasks.size());

        if (isDoneNum == tasks.size()) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return ACL_SUCCESS;
}

aclError aclrtStreamAbort(aclrtStream stream)
{
    uint64_t streamId = (uint64_t)(uintptr_t)stream;
    HCCL_VM_DEBUG("id:{:d}", streamId);
    RunnerDB::Update<sim::Stream>(streamId, [streamId](sim::Stream& stm) {
        stm.activated = 0;
    });

    return ACL_SUCCESS;
}

aclError aclrtStreamQuery(aclrtStream stream, aclrtStreamStatus* status)
{
    uint64_t streamId = (uint64_t)(uintptr_t)stream;
    HCCL_VM_DEBUG("id:{:d}", streamId);
    auto res = RunnerDB::GetById<sim::Stream>(streamId);
    if (!res.has_value()) {
        HCCL_VM_ERROR("stream not found:{:d}", streamId);
        return ACL_ERROR_INVALID_PARAM;
    }
    *status = (aclrtStreamStatus)res->task_complete_status;
    return ACL_SUCCESS;
}

aclError aclrtGetStreamAvailableNum(uint32_t* streamCount)
{
    (void)streamCount;
    auto serverId = sim::GetCurServerId();
    if (serverId == 0) {
        return ACL_ERROR_INVALID_PARAM;
    }
    sim::Runner runner;
    if (!sim::GetCurrRunnerTls(serverId, runner)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    if (runner.current_ctx_id == 0) {
        HCCL_VM_ERROR("invalid param");
        return ACL_ERROR_INVALID_PARAM;
    }
    auto currCtx = RunnerDB::GetById<sim::Context>(runner.current_ctx_id);
    if (!currCtx.has_value()) {
        HCCL_VM_ERROR("ctx not found:{:d}", runner.current_ctx_id);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto& devId = currCtx->device_id;

    auto device = RunnerDB::GetById<sim::Device>(devId);
    if (!device.has_value()) {
        HCCL_VM_ERROR("device not found:{:d}", devId);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto currCtxs = RunnerDB::GetByPred<sim::Context>([devId](const sim::Context& ctx) {
        return ctx.device_id == devId;
    });

    auto currStreams = RunnerDB::GetByPred<sim::Stream>([currCtxs](const sim::Stream& stm) {
        for (auto& ctx : currCtxs) {
            if (ctx.id == stm.ctx_id) {
                return true;
            }
        }
        return false;
    });

    if (memcmp(device->soc_version, "A3", strlen("A3") == 0)) {
        *streamCount = 1984 - currStreams.size();
    }

    return ACL_SUCCESS;
}

aclError aclrtStreamGetId(aclrtStream stream, int32_t* streamId)
{
    *streamId = (uint32_t)(uintptr_t)stream;
    HCCL_VM_DEBUG("id:{:d}", *streamId);
    return ACL_SUCCESS;
}

aclError aclrtSetStreamOverflowSwitch(aclrtStream stream, uint32_t flag)
{
    uint64_t streamId = (uint64_t)(uintptr_t)stream;
    HCCL_VM_DEBUG("id:{:d} flag:{:d}", streamId, flag);
    RunnerDB::Update<sim::Stream>(streamId, [streamId, flag](sim::Stream& stm) {
        stm.overflow_switch = flag;
    });
    return ACL_SUCCESS;
}

aclError aclrtGetStreamOverflowSwitch(aclrtStream stream, uint32_t* flag)
{
    uint64_t streamId = (uint64_t)(uintptr_t)stream;
    HCCL_VM_DEBUG("id:{:d}", streamId);
    auto res = RunnerDB::GetById<sim::Stream>(streamId);
    if (!res.has_value()) {
        HCCL_VM_ERROR("stream not found:{:d}", streamId);
        return ACL_ERROR_INVALID_PARAM;
    }
    *flag = res->overflow_switch;
    return ACL_SUCCESS;
}

aclError aclrtSetStreamAttribute(aclrtStream stream, aclrtStreamAttr stmAttrType, aclrtStreamAttrValue* value)
{
    uint64_t streamId = (uint64_t)(uintptr_t)stream;
    HCCL_VM_DEBUG("id:{:d}", streamId);
    auto res = RunnerDB::GetById<sim::Stream>(streamId);
    if (!res.has_value()) {
        HCCL_VM_ERROR("stream not found:{:d}", streamId);
        return ACL_ERROR_INVALID_PARAM;
    }

    aclrtStreamAttrValue tmp = *value;
    RunnerDB::Update<sim::Stream>(streamId, [streamId, stmAttrType, tmp](sim::Stream& stm) {
        if (stmAttrType == ACL_STREAM_ATTR_FAILURE_MODE) {
            stm.failure_mode = tmp.failureMode;
        } else if (stmAttrType == ACL_STREAM_ATTR_FLOAT_OVERFLOW_CHECK) {
            stm.overflow_switch = tmp.overflowSwitch;
        } else if (stmAttrType == ACL_STREAM_ATTR_USER_CUSTOM_TAG) {
            stm.user_tag = tmp.userCustomTag;
        }
    });

    return ACL_SUCCESS;
}

aclError aclrtGetStreamAttribute(aclrtStream stream, aclrtStreamAttr stmAttrType, aclrtStreamAttrValue* value)
{
    uint64_t streamId = (uint64_t)(uintptr_t)stream;
    HCCL_VM_DEBUG("id:{:d}", streamId);
    auto res = RunnerDB::GetById<sim::Stream>(streamId);
    if (!res.has_value()) {
        HCCL_VM_ERROR("stream not found:{:d}", streamId);
        return ACL_ERROR_INVALID_PARAM;
    }

    if (stmAttrType == ACL_STREAM_ATTR_FAILURE_MODE) {
        value->failureMode = res->failure_mode;
    } else if (stmAttrType == ACL_STREAM_ATTR_FLOAT_OVERFLOW_CHECK) {
        value->overflowSwitch = res->overflow_switch;
    } else if (stmAttrType == ACL_STREAM_ATTR_USER_CUSTOM_TAG) {
        value->userCustomTag = res->user_tag;
    }
    return ACL_SUCCESS;
}

aclError aclrtStreamStop(aclrtStream stream)
{
    (void)stream;
    return ACL_SUCCESS;
}

rtError_t rtStreamSynchronize(rtStream_t stream)
{
    uint64_t streamId = (uint64_t)(uintptr_t)stream;
    HCCL_VM_DEBUG("id:{:d}", streamId);
    return aclrtSynchronizeStream((aclrtStream)stream);
}

rtError_t rtStreamCreateWithFlags(rtStream_t* stm, int32_t priority, uint32_t flags)
{
    return aclrtCreateStreamWithConfig((aclrtStream*)stm, priority, flags);
}

rtError_t rtStreamGetSqid(const rtStream_t stream, uint32_t* sqId)
{
    uint64_t streamId = (uint64_t)(uintptr_t)stream;
    *sqId = streamId;
    HCCL_VM_DEBUG("id:{:d}", streamId);
    return ACL_SUCCESS;
}

rtError_t rtGetTaskIdAndStreamID(uint32_t* taskId, uint32_t* streamId)
{
    *streamId = (uint32_t)sim::GetLastStreamIdTls();
    *taskId = (uint32_t)sim::GetLastTaskIdTls();
    return ACL_SUCCESS;
}

#ifdef __cplusplus
}
#endif // __cplusplus
