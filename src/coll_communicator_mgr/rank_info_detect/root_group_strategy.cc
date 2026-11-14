/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "root_group_strategy.h"

#include <mutex>

#include "log.h"

namespace Hccl {
namespace {
    // 默认平均分组（与 A3 topoinfo_detect_scalable 算法一致）：
    // rpr = nRanks/nRoot，rmr = nRanks%nRoot，前 rmr 个 root 各 rpr+1 个 rank，
    // 其余各 rpr 个；组内 rankId 最小者为 root。
    u32 RootIdFromRankImpl(u32 nRanks, u32 nRoot, u32 rank)
    {
        const u32 rpr = nRanks / nRoot;
        const u32 rmr = nRanks % nRoot;
        const u32 rlim = rmr * (rpr + 1); // 前 rmr 个 root 的 rank 总数边界
        if (rank < rlim) {
            return rank / (rpr + 1);
        }
        return (rank - rlim) / rpr + rmr;
    }

    u32 FirstRankFromRootImpl(u32 nRanks, u32 nRoot, u32 rootIndex)
    {
        const u32 rpr = nRanks / nRoot;
        const u32 rmr = nRanks % nRoot;
        const u32 head = (rootIndex < rmr) ? rootIndex : rmr; // rootIndex 之前多带 rank 的 root 个数
        return rootIndex * rpr + head;
    }

    u32 NRankFromRootImpl(u32 nRanks, u32 nRoot, u32 rootIndex)
    {
        const u32 rpr = nRanks / nRoot;
        const u32 rmr = nRanks % nRoot;
        return rpr + (rootIndex < rmr ? 1 : 0); // 前 rmr 个 root 各 rpr+1，其余 rpr
    }

    bool RankHasRootImpl(u32 nRanks, u32 nRoot, u32 rank)
    {
        const u32 rpr = nRanks / nRoot;
        const u32 rmr = nRanks % nRoot;
        const u32 rlim = rmr * (rpr + 1); // 前 rmr 个 root 的 rank 总数边界
        if (rank < rlim) {
            // 在前 rmr 个 root 的范围内，每 rpr+1 个 rank 为一组，组内第一个 rank 是 root
            return (rank % (rpr + 1)) == 0;
        }
        // 在后 nRoot-rmr 个 root 的范围内，每 rpr 个 rank 为一组，组内第一个 rank 是 root
        return ((rank - rlim) % rpr) == 0;
    }

    const RootGroupStrategy DEFAULT_ROOT_GROUP_STRATEGY
        = {RootIdFromRankImpl, FirstRankFromRootImpl, NRankFromRootImpl, RankHasRootImpl};

    RootGroupStrategy& StrategySlot()
    {
        static RootGroupStrategy strategy = DEFAULT_ROOT_GROUP_STRATEGY;
        return strategy;
    }

    // 策略槽位互斥锁：注册（检查-赋值）与读取之间串行化
    std::mutex& StrategyMutex()
    {
        static std::mutex mutex;
        return mutex;
    }

    // 库装载期静态注册默认平均分组。
    // 注册默认策略本身幂等，仅自定义覆盖时受 RegisterRootGroupStrategy 的防重复约束。
    struct RootGroupStrategyRegistrar {
        RootGroupStrategyRegistrar() { (void)RegisterRootGroupStrategy(DEFAULT_ROOT_GROUP_STRATEGY); }
    };
    const RootGroupStrategyRegistrar g_rootGroupStrategyRegistrar;
} // namespace

HcclResult RegisterRootGroupStrategy(const RootGroupStrategy& strategy)
{
    if (strategy.rootIdFromRank == nullptr || strategy.firstRankFromRoot == nullptr || strategy.nRankFromRoot == nullptr
        || strategy.rankHasRoot == nullptr) {
        HCCL_ERROR("[%s] root group strategy has null function pointer.", __func__);
        return HCCL_E_PTR;
    }
    std::lock_guard<std::mutex> lock(StrategyMutex());
    auto& slot = StrategySlot();
    // 已有自定义策略时禁止再次覆盖
    if (slot.rootIdFromRank != DEFAULT_ROOT_GROUP_STRATEGY.rootIdFromRank) {
        HCCL_ERROR("[%s] root group strategy has already been registered.", __func__);
        return HCCL_E_INTERNAL;
    }
    slot = strategy;
    return HCCL_SUCCESS;
}

const RootGroupStrategy* GetRootGroupStrategy()
{
    std::lock_guard<std::mutex> lock(StrategyMutex());
    auto& slot = StrategySlot();
    if (slot.rootIdFromRank == nullptr) {
        return nullptr;
    }
    return &slot;
}

} // namespace Hccl
