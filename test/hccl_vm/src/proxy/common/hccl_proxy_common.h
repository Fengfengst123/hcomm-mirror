/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _SIM_HCCL_PROXY_COMMON_H_
#define _SIM_HCCL_PROXY_COMMON_H_

#include "hccl/hccl_types.h"
#include <map>
#include <string>

#include <cstdint>
#include <map>

#include "storage/database_result.h"

namespace sim {
inline const std::map<HcclDataType, uint32_t> DATA_TYPE_SIZE_MAP
    = {{HcclDataType::HCCL_DATA_TYPE_INT8, 1},    {HcclDataType::HCCL_DATA_TYPE_INT16, 2},
       {HcclDataType::HCCL_DATA_TYPE_INT32, 4},   {HcclDataType::HCCL_DATA_TYPE_FP16, 2},
       {HcclDataType::HCCL_DATA_TYPE_FP32, 4},    {HcclDataType::HCCL_DATA_TYPE_INT64, 8},
       {HcclDataType::HCCL_DATA_TYPE_UINT64, 8},  {HcclDataType::HCCL_DATA_TYPE_UINT8, 1},
       {HcclDataType::HCCL_DATA_TYPE_UINT16, 2},  {HcclDataType::HCCL_DATA_TYPE_UINT32, 4},
       {HcclDataType::HCCL_DATA_TYPE_FP64, 8},    {HcclDataType::HCCL_DATA_TYPE_BFP16, 2},
       {HcclDataType::HCCL_DATA_TYPE_INT128, 16}, {HcclDataType::HCCL_DATA_TYPE_HIF8, 1},
       {HcclDataType::HCCL_DATA_TYPE_FP8E4M3, 1}, {HcclDataType::HCCL_DATA_TYPE_FP8E5M2, 1},
       {HcclDataType::HCCL_DATA_TYPE_FP8E8M0, 1}};

inline int GetDataTypeSize(HcclDataType dataType, uint32_t& size)
{
    auto iter = DATA_TYPE_SIZE_MAP.find(dataType);
    if (iter == DATA_TYPE_SIZE_MAP.end()) {
        return 1;
    }
    size = iter->second;
    return 0;
}

/// 地址分类（设备地址包含查询 src_type=DEV 且
/// dev_mapped_ptr<=addr<dev_mapped_end_ptr）。
/// 三态合同——"确定未命中"与"查询无法完成"是不同状态，不得互相冒充：
///   OK+true    查询完成且命中设备映射区间；
///   OK+false   查询完成且数据库明确 NOT_FOUND（host 地址，不是错误）；
///   其它 code
///   查询无法完成（CONFLICT、RESOURCE_EXHAUSTED、BACKEND_UNAVAILABLE、
///              协议/解码错误等），保留原错误码与诊断向上传播，调用方不得据此
///              判定地址类别，也不得视为未命中。
HcclSim::Storage::DbResult<bool> IsDeviceAddress(void* addr);

bool ParseKernelJson(const std::string& jsonPath, std::map<std::string, std::string>& out);

uint64_t GetVirPtrByDevPtr(uint64_t devAddr);

// CCU SQE arguments contain both device addresses and scalar values. Return
// zero for non-address arguments so callers can translate only mapped values.
uint64_t TryGetVirPtrByDevPtr(uint64_t devAddr);

uint64_t TransRemoteAddrToVirtualByRank(uint64_t devAddr, uint32_t rankId);

uint64_t TransRemoteAddrToVirtualByDeviceId(uint64_t devAddr, uint32_t deviceId);

void* GetDevMapperAddrByHostPtr(void* hostPtr);

bool IsAICPUExpMode();

} // namespace sim

// 查询真实 HcclComm 在当前进程中绑定的本地 Communicator 表行 id.
extern "C" bool SimFindHcclCommMember(HcclComm comm, uint64_t* communicatorId);
extern "C" bool SimFindHcclCommHandle(uint64_t communicatorId, HcclComm* comm);
#endif
