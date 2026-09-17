/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "channel_profiling_adpt.h"
#include <atomic>

namespace hcomm {
namespace {
    // 静态初始化
    std::atomic<GetChannelRemoteRankIdFunc> g_getChannelRemoteRankIdFunc{nullptr};
} // namespace

void RegisterGetChannelRemoteRankId(GetChannelRemoteRankIdFunc func)
{
    g_getChannelRemoteRankIdFunc.store(func, std::memory_order_release);
}

GetChannelRemoteRankIdFunc GetChannelRemoteRankIdFuncImpl()
{
    return g_getChannelRemoteRankIdFunc.load(std::memory_order_acquire);
}

} // namespace hcomm
