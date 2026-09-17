/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_RES_EXPT_H
#define HCOMM_RES_EXPT_H

#include "hcomm_res_defs.h"
#include "acl/acl_rt.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * @brief 基于已有 aclrtStream 分配单个通信线程，并返回线程句柄。
 * @param[in] engine 通信引擎类型，仅支持 COMM_ENGINE_CPU 和 COMM_ENGINE_CPU_TS
 * @param[in] stream 已有的 runtime stream 句柄，由调用方通过 aclrtCreateStream 等接口创建
 * @param[in] notifyNum 该线程所需的同步资源（Notify）数量
 * @param[out] thread 输出的通信线程句柄
 * @return HcommResult 成功返回 0，失败返回对应错误码
 * @note 调用前需通过 aclrtSetDevice 设置当前 Device。
 */
extern HcommResult
HcommThreadAllocWithStream(CommEngine engine, aclrtStream stream, uint32_t notifyNum, ThreadHandle* thread);

#ifdef __cplusplus
}
#endif // __cplusplus
#endif
