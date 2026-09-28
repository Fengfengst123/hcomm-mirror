/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DEV_TYPE_UTILS_H
#define DEV_TYPE_UTILS_H

#include <initializer_list>
#include "dtype_common.h"
#include "hcomm_res_defs.h"

inline const char* GetDevTypeStr(DevType devType)
{
    switch (devType) {
        case DevType::DEV_TYPE_910:
            return "DEV_TYPE_910";
        case DevType::DEV_TYPE_310P3:
            return "DEV_TYPE_310P3";
        case DevType::DEV_TYPE_910B:
            return "DEV_TYPE_910B";
        case DevType::DEV_TYPE_310P1:
            return "DEV_TYPE_310P1";
        case DevType::DEV_TYPE_910_93:
            return "DEV_TYPE_910_93";
        case DevType::DEV_TYPE_NOSOC:
            return "DEV_TYPE_NOSOC";
        case DevType::DEV_TYPE_950:
            return "DEV_TYPE_950";
        case DevType::DEV_TYPE_MC62:
            return "DEV_TYPE_MC62";
        case DevType::DEV_TYPE_960:
            return "DEV_TYPE_960";
        case DevType::DEV_TYPE_COUNT:
            return "DEV_TYPE_COUNT";
        default:
            return "UNKNOWN";
    }
}

inline const char* GetEndpointLocTypeStr(EndpointLocType locType)
{
    switch (locType) {
        case ENDPOINT_LOC_TYPE_RESERVED:
            return "RESERVED";
        case ENDPOINT_LOC_TYPE_DEVICE:
            return "DEVICE";
        case ENDPOINT_LOC_TYPE_HOST:
            return "HOST";
        default:
            return "UNKNOWN";
    }
}

// 仅校验当前芯片是否在 supportedTypes 中；protocol / locType 只用于失败日志，不参与判断。
HcclResult CheckDevTypeSupport(
    const std::initializer_list<DevType>& supportedTypes, CommProtocol protocol, EndpointLocType locType);

#endif
