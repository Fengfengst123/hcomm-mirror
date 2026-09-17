/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opbase_adpt.h"
#include "adapter_rts_common.h"
#include "device_capacity.h"
#include "log.h"

thread_local s32 g_hcclDeviceId = INVALID_INT;

s32 HcclGetThreadDeviceId()
{
    if (g_hcclDeviceId == INVALID_INT) {
        s32 devId = INVALID_INT;
        if (hrtGetDevice(&devId) != HCCL_SUCCESS) {
            HCCL_WARNING("[HcclGetThreadDeviceId] get fail deviceLogicId[%d]", devId);
            return INVALID_INT;
        }
        g_hcclDeviceId = devId;
    }

    u32 maxDeviceNum = 0;
    if (hccl::GetMaxDevNum(maxDeviceNum) != HCCL_SUCCESS) {
        HCCL_WARNING("[HcclGetThreadDeviceId] get maxDeviceNum fail, deviceLogicId[%d]", g_hcclDeviceId);
        return INVALID_INT;
    }

    if (static_cast<u32>(g_hcclDeviceId) >= maxDeviceNum) {
        HCCL_WARNING(
            "[HcclGetThreadDeviceId] deviceLogicId[%d] is bigger than maxDeviceNum:[%u]", g_hcclDeviceId, maxDeviceNum);
        g_hcclDeviceId = INVALID_INT;
        return INVALID_INT;
    }

    HCCL_INFO("[HcclGetThreadDeviceId] deviceLogicId[%d]", g_hcclDeviceId);
    return g_hcclDeviceId;
}

HcclResult HcclDeviceRefresh(s32& deviceLogicId)
{
    HcclResult ret = hrtGetDeviceRefresh(&g_hcclDeviceId);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Get][DeviceRefresh]errNo[0x%016llx] g_hcclDeviceId[%d]"
            "get device refresh error.",
            ret, g_hcclDeviceId),
        ret);
    deviceLogicId = g_hcclDeviceId;
    return HCCL_SUCCESS;
}
