/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INDOP_CHANNEL_PARAM_H
#define INDOP_CHANNEL_PARAM_H

#include "channel_param.h"
struct HcclIndOpChannelRemoteResV2 {
    u64 p2pNotifyNum = 0;        // 用于linkp2p添加notify信息
    u64 roceNotifyNum = 0;       // 主链路：用于linkroce添加notify信息
    u64 qpNum = 0;               // 主链路：QP计数，支持多个，用于linkroce添加QP信息
    u32 remoteRank = 0;          // 远端rankId
    u32 remoteWorldRank = 0;     // 远端rankWorldId
    bool isUsedRdma = false;     // 是否使用RDMA，对应Roce和P2p
    HcclChannelP2p channelP2p;   // P2p资源
    HcclChannelRoce channelRoce; // Roce资源
};

struct HcclIndOpChannelRemoteResV3 {
    char hcomId[HCOMID_MAX_LENGTH]{};                   // 通信域ID 最大长度待修改
    char channelTag[TAG_MAX_LENGTH]{};                  // channelTag 最大长度待修改
    CommEngine engine{};                                // 通信引擎类型
    u32 localUserRank{0};                               // 本地rankId
    u32 multiQpThreshold{0};                            // 多QP每个QP分担数据量最小阈值
    void* channelList{nullptr};                         // device侧channelList地址
    u32 listNum = 0;                                    // 建链channel的总数量
    HcclIndOpChannelRemoteResV2* remoteResV2 = nullptr; // 不同remoteRank建链的资源
};

#endif
