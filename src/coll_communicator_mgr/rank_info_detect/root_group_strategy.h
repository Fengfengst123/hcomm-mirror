/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_ROOT_GROUP_STRATEGY_H
#define HCCL_ROOT_GROUP_STRATEGY_H

#include "types.h"

namespace Hccl {

// scalable 多 root 建链的分组策略回调表。默认注册平均分组（rpr/rmr 均分、余数摊前，组内最小 rankId 为 root）。
// 注册回调形式为后续用户自定义分组策略预留，本次不对外导出。
struct RootGroupStrategy {
    u32 (*rootIdFromRank)(u32 nRanks, u32 nRoot, u32 rank);         // rank 所属 rootIndex
    u32 (*firstRankFromRoot)(u32 nRanks, u32 nRoot, u32 rootIndex); // 组内首 rank
    u32 (*nRankFromRoot)(u32 nRanks, u32 nRoot, u32 rootIndex);     // 组内 rank 数
    bool (*rankHasRoot)(u32 nRanks, u32 nRoot, u32 rank);           // 该 rank 是否为本组 root
};

// hcomm 内部接口：注册分组策略（装载期一次性设置，运行期只读；重复注册返回错误）。
HcclResult RegisterRootGroupStrategy(const RootGroupStrategy& strategy);

// 查询当前分组策略（未注册时返回 nullptr，调用方须兜底报错）。
const RootGroupStrategy* GetRootGroupStrategy();

} // namespace Hccl

#endif // HCCL_ROOT_GROUP_STRATEGY_H
