/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef THREAD_AICPU_MGR_H
#define THREAD_AICPU_MGR_H

#include "common.h"
#include "aicpu_launch_manager.h"
#include "thread.h"
#include "hcclCommDfxLite.h"
#include <shared_mutex>
#include <vector>
#include <memory>
#include <functional>

namespace hccl {
class AicpuTsThread;
}

class ThreadAicpuMgr {
public:
    ThreadAicpuMgr(hccl::HcclCommDfxLite& dfx, std::function<HcclResult(bool)> checkExecStatusCallback);
    ~ThreadAicpuMgr();
    HcclResult InitThreads(ThreadMgrAicpuParam* param);
    const std::vector<std::shared_ptr<hccl::Thread>>& GetAllThread() { return threads_; }
    std::shared_mutex& GetThreadMutex() { return threadMutex_; }

private:
    HcclResult RegisterThreadAddDfxTaskInfo(ThreadHandle thread);
    HcclResult RegisterThreadCacheCallback(hccl::AicpuTsThread* thread);

    std::shared_mutex threadMutex_;
    // threads_：持有设备流资源的真线程（Resume/异常CQE/dfx遍历均依赖其StreamLite/Rtsq）。
    // cpuExportThread_：host侧（CPU_TS）导出的notify桩线程（fakeDeviceRes_，GE保序线程）
    std::vector<std::shared_ptr<hccl::Thread>> threads_;
    std::vector<std::shared_ptr<hccl::Thread>> cpuExportThread_;
    hccl::HcclCommDfxLite& dfx_;
    std::function<HcclResult(bool)> checkExecStatusCallback_;
};

#endif // THREAD_AICPU_MGR_H
