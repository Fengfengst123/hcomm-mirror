/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sim_aicpu_pipe_handler.h"
#include "aicpu_args_stub.h"
#include "hccl_device_pub.h"
#include "operation_data/operation_data_ops.h"
#include "sim_aicpu_pipe_msg.h"
#include "sim_common_api.h"
#include "sim_kernel_lib_mgr.h"
#include "sim_log.h"
#include "store_sim_memory_manager.h"
#include <dlfcn.h>
#include <string>
#include <unistd.h>

namespace sim {

int HandlePipeCmdSetDevId(uint8_t* payload, uint32_t payloadLen)
{
    SetDevIdPayload* req = reinterpret_cast<SetDevIdPayload*>(payload);
    uint32_t curRankId = static_cast<uint32_t>(req->rankId);
    uint64_t curDeviceKey = req->deviceKey;
    SetCurRankId(curRankId);
    SetCurDeviceKey(curDeviceKey);
    HCCL_VM_INFO(
        "Process[{}] PIPE_RSP_SET_DEV_ID, set rankId = [{}], deviceKey = [{}]", getpid(), curRankId, curDeviceKey);
    uint64_t donePayload = 0;
    DeviceSendMsg(PIPE_RSP_SET_DEV_ID, &donePayload, sizeof(donePayload));
    return 0;
}

int HandlePipeCmdGetDevPtr(uint8_t* payload, uint32_t payloadLen)
{
    DevMemOpPayload* req = reinterpret_cast<DevMemOpPayload*>(payload);
    RspGetDevPtrPayload donePayload{};
    void* shmptr = sim::MemoryManager::GetInstance().AcquireMemByName(req->memName);
    if (shmptr == nullptr) {
        donePayload.ptr = 0;
        HCCL_VM_ERROR("acquire {} shm failed.", req->memName);
    } else {
        donePayload.ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(shmptr));
    }
    HCCL_VM_INFO("[device] acquire {} shm ptr:{:p}", req->memName, shmptr);
    DeviceSendMsg(PIPE_RSP_GET_DEV_PTR, &donePayload, sizeof(donePayload));
    return 0;
}

static void ExecuteAicpuKernel(uint32_t rankId, ExecKernelPayload* kernelReq, void* argsPtr)
{
    std::string kernelSo = kernelReq->soName;
    std::string kernelName = kernelReq->kernelName;

    HCCL_VM_INFO("rankId[{}] kernel[{}] start run...", rankId, kernelName);
    std::string libDir = InstallPath::ResolveToInstallRoot("lib/" + GetArchStr()) + "/" + kernelSo;
    KernelFn fn = sim::KernelLibManager::GetInstance().GetOrLoadFunc(libDir, kernelName);
    if (!fn) {
        HCCL_VM_ERROR("failed to resolve function");
        return;
    }

    if (argsPtr == nullptr) {
        HCCL_VM_ERROR("[device] rankId[{}] init func handle failed null ptr.", rankId);
        return;
    }

    // CCU退化AICPU场景此处使用的内存先于device进程启动前分配，使用前需转换
    if (kernelName == "RunAicpuIndOpCommInit" || kernelName == "RunAicpuCommInit") {
        CommAicpuParam* param = reinterpret_cast<CommAicpuParam*>(argsPtr);
        param->kfcControlTransferH2DParams.deviceAddr
            = GetDevMapperAddrByDevAddr(param->kfcControlTransferH2DParams.deviceAddr);
        param->kfcControlTransferH2DParams.readCacheAddr
            = GetDevMapperAddrByDevAddr(param->kfcControlTransferH2DParams.readCacheAddr);
        param->kfcStatusTransferD2HParams.deviceAddr
            = GetDevMapperAddrByDevAddr(param->kfcStatusTransferD2HParams.deviceAddr);
    }
    fn(argsPtr);
    HCCL_VM_INFO("rankId[{}] kernel[{}] finish run...", rankId, kernelName);
}

int HandlePipeCmdExecKernel(uint8_t* payload, uint32_t payloadLen)
{
    if (payloadLen < sizeof(ExecKernelPayload)) {
        HCCL_VM_ERROR("[device] EXEC_KERNEL payload too small: {} < {}", payloadLen, sizeof(ExecKernelPayload));
        RspExecKernelPayload donePayload{};
        donePayload.status = -1;
        DeviceSendMsg(PIPE_RSP_EXEC_KERNEL, &donePayload, sizeof(donePayload));
        return -1;
    }

    ExecKernelPayload* kernelReq = reinterpret_cast<ExecKernelPayload*>(payload);
    uint32_t rankId = 0;
    GetCurRankId(&rankId);
    std::string kernelName(kernelReq->kernelName);
    uint32_t argsLen = payloadLen - sizeof(ExecKernelPayload);
    void* argsPtr = payload + sizeof(ExecKernelPayload);
    HCCL_VM_INFO("[device] rankId[{}] executing kernel:{}, argsLen:{}", rankId, kernelName, argsLen);

    uint64_t commId = 0;
    uint32_t opRankId = 0;
    uint32_t opDeviceId = 0;
    if (kernelReq->opDetailId != 0) {
        // 只使用 host 随本次 kernel 显式传入的算子记录，不按时间或 pid 猜测。
        if (sim::operation::QueryOpDetailIdentity(kernelReq->opDetailId, commId, opRankId, opDeviceId) != 0
            || commId == 0) {
            HCCL_VM_ERROR("[device] invalid task identity for opDetailId={}, commId={}", kernelReq->opDetailId, commId);
            RspExecKernelPayload donePayload{-1};
            DeviceSendMsg(PIPE_RSP_EXEC_KERNEL, &donePayload, sizeof(donePayload));
            return -1;
        }
        // InsertOpTask 仍通过该值把 device 任务写入本次 opDetail 的任务表；
        // 它不参与任务 metadata 的身份生成。
        sim::operation::g_currOpDetailId = kernelReq->opDetailId;
    } else {
        // 普通 kernel 不产生带通信域身份的
        // SQE，执行前必须清掉上一次算子的上下文。
        sim::operation::g_currOpDetailId = 0;
    }

    ExecuteAicpuKernel(rankId, kernelReq, argsPtr);
    sim::operation::g_currOpDetailId = 0;
    RspExecKernelPayload donePayload{};
    donePayload.status = 0;
    DeviceSendMsg(PIPE_RSP_EXEC_KERNEL, &donePayload, sizeof(donePayload));
    return 0;
}

int HandlePipeCmdFreeDevPtr(uint8_t* payload, uint32_t payloadLen)
{
    DevMemOpPayload* req = reinterpret_cast<DevMemOpPayload*>(payload);
    sim::MemoryManager::GetInstance().ReleaseMemByName(req->memName);
    HCCL_VM_INFO("[device] release memName:{} shm", req->memName);
    RspFreeDevPtrPayload donePayload{};
    donePayload.status = 0;
    DeviceSendMsg(PIPE_RSP_FREE_DEV_PTR, &donePayload, sizeof(donePayload));
    return 0;
}

int HandlePipeCmdGetWqePtr(uint8_t* payload, uint32_t payloadLen)
{
    RspGetDevPtrPayload donePayload{};
    void* ptr = malloc(2 * 1024 * 1024);
    if (ptr == nullptr) {
        donePayload.ptr = 0;
    } else {
        donePayload.ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
    }
    HCCL_VM_INFO("[device] alloc wqe buffer :{:p}", ptr);

    DeviceSendMsg(PIPE_RSP_GET_WQE_PTR, &donePayload, sizeof(donePayload));
    return 0;
}

int HandlePipeCmdFreeWqePtr(uint8_t* payload, uint32_t payloadLen)
{
    ReqDevPtrPayload* req = reinterpret_cast<ReqDevPtrPayload*>(payload);
    void* ptr = (void*)(req->ptr);
    free(ptr);
    RspFreeDevPtrPayload donePayload{};
    donePayload.status = 0;
    HCCL_VM_INFO("[device] free wqe buffer :{:p}", ptr);

    DeviceSendMsg(PIPE_RSP_FREE_WQE_PTR, &donePayload, sizeof(donePayload));
    return 0;
}

} // namespace sim
