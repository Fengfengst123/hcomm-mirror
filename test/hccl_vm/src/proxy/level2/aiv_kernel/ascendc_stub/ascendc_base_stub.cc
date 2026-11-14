/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascendc_base_stub.h"

#include <cstdint>

namespace AscendC {
// block_num 与 block_idx 同为 kernel 执行的"硬件视角"状态：单进程多 rank 线程并发
// 下发时各自写入本线程 launch 的核数，必须线程私有，否则并发 launch 互相覆盖。
thread_local int64_t block_num = 0;
thread_local int64_t block_idx = 0;

std::string GetPipeName(pipe_t pipe)
{
    switch (pipe) {
        case PIPE_ALL:
            return "ALL";
        case PIPE_MTE2:
            return "MTE2";
        case PIPE_MTE3:
            return "MTE3";
        case PIPE_S:
            return "SCALAR";
        default:
            return "UNKNOWN";
    }
}
} // namespace AscendC
