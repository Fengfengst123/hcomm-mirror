/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_rtsq_poll_daemon.h"
#include "log.h"
#include "thread.h"
#include "aicpu_thread_process.h"

namespace hcomm {

RtsqPollCompletionDaemon& RtsqPollCompletionDaemon::GetInstance()
{
    static RtsqPollCompletionDaemon instance;
    return instance;
}

void RtsqPollCompletionDaemon::Call()
{
    std::shared_lock<std::shared_mutex> rwlock(AicpuThreadProcess::GetMutex());
    const auto& threads = AicpuThreadProcess::GetThreads();
    for (const auto& thread : threads) {
        if (thread == nullptr) {
            continue;
        }
        Hccl::StreamLite* streamLite = static_cast<Hccl::StreamLite*>(thread->GetStreamLitePtr());
        if (streamLite == nullptr) {
            continue;
        }
        Hccl::RtsqBase* rtsq = streamLite->GetRtsq();
        if (rtsq == nullptr) {
            continue;
        }
        rtsq->PollCompletion();
    }
}

} // namespace hcomm
