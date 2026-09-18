/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_THREAD_C_ADPT_H
#define HCOMM_THREAD_C_ADPT_H

#include "hcomm_res_defs.h"
#include "hccl/hccl_res.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * @brief 批量补充线程 notify 资源；AICPU 引擎额外触发 device 侧批量 kernel launch
 * @param[in] engine 通信引擎类型，仅 COMM_ENGINE_AICPU 触发 device kernel launch
 * @param[in] handles 线程句柄数组
 * @param[in] threadNum 线程句柄数量
 * @param[in] supplementNotifyNums 每个线程需补充的 notify 增量数量数组，长度须为 threadNum；增量为 0 的线程跳过
 * @return HcommResult 成功返回 HCOMM_SUCCESS，失败返回对应错误码
 */
HcommResult
HcommThreadSupplementNotify(const ThreadHandle* handles, uint32_t threadNum, const uint32_t* supplementNotifyNums);

/**
 * @brief 查询单个线程当前已分配的 notify 数量
 * @param[in] thread 线程句柄
 * @param[out] notifyNum 输出的 notify 数量
 * @return HcommResult 成功返回 HCOMM_SUCCESS，失败返回对应错误码
 */
HcommResult HcommThreadGetNotifyNum(ThreadHandle thread, uint32_t* notifyNum);

/**
 * @brief 重置线程内全部本地 notify 的触发态（逐个调用 hrtNotifyReset，重复调用幂等）
 * @param[in] thread 线程句柄
 * @return HcommResult 成功返回 HCOMM_SUCCESS，失败返回对应错误码
 */
HcommResult HcommThreadResetNotifies(ThreadHandle thread);

/**
 * @brief 把线程句柄换成目标引擎可用的句柄，按目标引擎分两条路径：
 *        - CPU/CCU：从 device→host 映射表里查出对应的 host 侧句柄
 *        - AICPU：先在线程内部映射表里找，找不到就批量下发 kernel 创建 AICPU 线程并登记映射
 * @param[in] handles 源线程句柄数组
 * @param[in] commId 通信域标识字符串，AICPU 路径用于 kernel launch；可为空
 * @param[in] threadNum 线程句柄数量
 * @param[in] dstEngine 目标引擎类型
 * @param[out] outHandles 导出的句柄数组，长度须为 threadNum
 * @return HcommResult 成功返回 HCOMM_SUCCESS，失败返回对应错误码
 */
HcommResult HcommThreadExportToCommEngine(
    const ThreadHandle* handles, const char* commId, uint32_t threadNum, CommEngine dstEngine,
    ThreadHandle* outHandles);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // HCOMM_THREAD_C_ADPT_H
