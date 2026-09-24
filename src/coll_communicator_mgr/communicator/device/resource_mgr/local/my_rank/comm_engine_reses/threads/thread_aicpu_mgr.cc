/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "thread_aicpu_mgr.h"
#include "aicpu_ts_thread.h"
#include "hcclCommOp.h"
#include "aicpu_task_cache_manager.h"
#include "stream_lite.h"
#include "rtsq_a5.h"
#include "log.h"
#include <sstream>
#include <iomanip>

ThreadAicpuMgr::ThreadAicpuMgr(hccl::HcclCommDfxLite& dfx, std::function<HcclResult(bool)> checkExecStatusCallback)
    : dfx_(dfx),
      checkExecStatusCallback_(std::move(checkExecStatusCallback))
{}

ThreadAicpuMgr::~ThreadAicpuMgr()
{
    std::unique_lock<std::shared_mutex> rwLock(threadMutex_);
    for (auto& thread : threads_) {
        HcommThreadRegisterCheckExecStatus(reinterpret_cast<ThreadHandle>(thread.get()), nullptr);
    }
    threads_.clear();
}

HcclResult ThreadAicpuMgr::InitThreads(ThreadMgrAicpuParam* param)
{
    CHK_PTR_NULL(param);
    u32 threadNum = param->threadNum;
    ThreadHandle* threadArray = static_cast<ThreadHandle*>(param->deviceHandle);
    CHK_PTR_NULL(threadArray);
    std::string hcomId(param->hcomId);
    // 创建期即分流：真线程入outThreads，桩线程入stubThreads暂存（后置注册/入表零判别）。
    // stubThreads须在threadMutex_内并入cpuExportThread_（与threads_锁纪律一致）
    std::vector<std::shared_ptr<hccl::Thread>> outThreads;
    std::vector<std::shared_ptr<hccl::Thread>> stubThreads;
    outThreads.reserve(threadNum);
    stubThreads.reserve(threadNum);
    for (u32 i = 0; i < threadNum; ++i) {
        std::string thdUniqueId(param->threadParam[i], THREAD_UNIQUE_ID_MAX_SIZE);
        if (UNLIKELY(HcclCheckLogLevel(HCCL_LOG_INFO))) {
            std::ostringstream oss;
            oss << "threadParam[" << i << "] raw bytes: ";
            constexpr u32 HEX_WIDTH = 2;
            for (u32 j = 0; j < THREAD_UNIQUE_ID_MAX_SIZE; ++j) {
                oss << std::hex << std::setw(HEX_WIDTH) << std::setfill('0')
                    << static_cast<unsigned int>(static_cast<unsigned char>(param->threadParam[i][j])) << " ";
            }
            HCCL_INFO("[ThreadAicpuMgr][%s] %s", __func__, oss.str().c_str());
        }
        std::shared_ptr<hccl::AicpuTsThread> thread;
        EXCEPTION_CATCH((thread = std::make_shared<hccl::AicpuTsThread>(thdUniqueId)), return HCCL_E_PTR);
        HcclResult ret = thread->Init();
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[ThreadAicpuMgr][%s] comm identifier[%s], init threads num[%u] failed at index %u", __func__,
                hcomId.c_str(), param->threadNum, i);
            return ret;
        }
        // 设备句柄按【原始输入序】写回（threadArray与threadParam下标对齐）——真/桩都写：
        // host侧ThreadKernelLaunchImpl Step 6无条件读回整个deviceHandle缓冲区并按原始序
        // 配对注册（AddThreadHandleToMap/FillThreadD2HMap），漏写桩槽位=注册未初始化内存为句柄，
        // 按紧缩序写入=混合批次真线程句柄整体错位
        threadArray[i] = reinterpret_cast<ThreadHandle>(thread.get());
        HCCL_INFO(
            "[ThreadAicpuMgr][%s] threadArray[%u] = [%llu]", __func__, i,
            static_cast<unsigned long long>(threadArray[i]));
        if (thread->IsFakeDeviceRes()) {
            stubThreads.emplace_back(std::move(thread));
        } else {
            outThreads.emplace_back(std::move(thread));
        }
    }

    // 注册仅真线程（桩线程无StreamLite/Rtsq，dfx与缓存回调均无可挂载对象）
    for (size_t i = 0; i < outThreads.size(); ++i) {
        CHK_RET(RegisterThreadAddDfxTaskInfo(reinterpret_cast<ThreadHandle>(outThreads[i].get())));
        CHK_RET(RegisterThreadCacheCallback(static_cast<hccl::AicpuTsThread*>(outThreads[i].get())));
    }
    std::unique_lock<std::shared_mutex> rwLock(threadMutex_);
    threads_.insert(
        threads_.end(), std::make_move_iterator(outThreads.begin()), std::make_move_iterator(outThreads.end()));
    cpuExportThread_.insert(
        cpuExportThread_.end(), std::make_move_iterator(stubThreads.begin()),
        std::make_move_iterator(stubThreads.end()));
    HCCL_INFO(
        "[ThreadAicpuMgr][%s] comm identifier[%s], init threads num[%u] success", __func__, hcomId.c_str(), threadNum);
    return HCCL_SUCCESS;
}

HcclResult ThreadAicpuMgr::RegisterThreadAddDfxTaskInfo(ThreadHandle thread)
{
    int32_t ret = HcommThreadRegisterCheckExecStatus(thread, checkExecStatusCallback_);
    if (ret != 0) {
        HCCL_ERROR(
            "[%s]HcommThreadRegisterCheckExecStatus failed, ret[%d], thread[0x%llx], checkExecStatusCallback[%p]",
            __func__, ret, static_cast<unsigned long long>(thread),
            static_cast<const void*>(&checkExecStatusCallback_));
        return HCCL_E_PTR;
    }

    std::function<void(Hccl::TaskInfoCircularQueue*)> reportCallback = [this](Hccl::TaskInfoCircularQueue* taskQueue) {
        dfx_.ReportStreamTask(taskQueue);
    };
    ret = HcommNewThreadRegisterDfx(thread, reportCallback);
    if (ret != 0) {
        HCCL_ERROR("[%s] HcommNewThreadRegisterDfx failed, ret[%d], thread[0x%llx]", __func__, ret, thread);
        return HCCL_E_PTR;
    }

    std::function<const void*()> getLatestOpInfoCallback = [this]() -> const void* {
        return dfx_.GetLatestDfxOpInfo();
    };
    ret = HcommNewThreadRegisterGetLatestDfxOpInfo(thread, getLatestOpInfoCallback);
    if (ret != 0) {
        HCCL_ERROR(
            "[%s] HcommNewThreadRegisterGetLatestDfxOpInfo failed, ret[%d], thread[0x%llx]", __func__, ret, thread);
        return HCCL_E_PTR;
    }

    return HCCL_SUCCESS;
}

HcclResult ThreadAicpuMgr::RegisterThreadCacheCallback(hccl::AicpuTsThread* thread)
{
    HCCL_INFO("[ThreadAicpuMgr][%s] register cache callback for thread[%p]", __func__, thread);
    CHK_PTR_NULL(thread);
    auto* const streamLitePtr = static_cast<Hccl::StreamLite*>(thread->GetStreamLitePtr());
    CHK_PTR_NULL(streamLitePtr);
    Hccl::RtsqA5* rtsqA5 = static_cast<Hccl::RtsqA5*>(streamLitePtr->GetRtsq());
    CHK_PTR_NULL(rtsqA5);
    CHK_RET(rtsqA5->SetAicpuTsThreadPtr(thread));
    CHK_RET(rtsqA5->SetNeedCacheTaskCallback(hcomm::AicpuTaskCacheManager::NeedCacheTask));
    CHK_RET(rtsqA5->SetAddSqeArrayCallback(hcomm::AicpuTaskCacheManager::AddSqeArray));
    HCCL_INFO("[ThreadAicpuMgr][%s] register cache callback for thread[%p] success", __func__, thread);
    return HCCL_SUCCESS;
}
