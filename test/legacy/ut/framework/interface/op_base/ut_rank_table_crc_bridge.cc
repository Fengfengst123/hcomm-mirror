/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include <algorithm>
#include <future>
#include <string>
#include <thread>
#include <vector>

#include "checkcrc.h"
#include "rank_table_crc_bridge.h"

using namespace Hccl;

namespace {
constexpr size_t CONCURRENT_THREAD_NUM = 32;
constexpr s32 DEVICE_LOGIC_ID_BASE = 1000;
constexpr s32 SHARED_DEVICE_LOGIC_ID = 2000;

class RankTableCrcBridgeTest : public testing::Test {
protected:
    void TearDown() override
    {
        auto& bridge = RankTableCrcBridge::GetInstance();
        for (size_t i = 0; i < CONCURRENT_THREAD_NUM; ++i) {
            (void)bridge.ConsumeRankTableJsonCrc(DEVICE_LOGIC_ID_BASE + static_cast<s32>(i));
        }
        (void)bridge.ConsumeRankTableJsonCrc(SHARED_DEVICE_LOGIC_ID);
    }
};

TEST_F(RankTableCrcBridgeTest, Ut_ConcurrentRecordAndConsumeDifferentDevices_ExpectExactCrc)
{
    auto& bridge = RankTableCrcBridge::GetInstance();
    std::vector<std::string> rankTableJsons;
    std::vector<u32> expectedCrcs(CONCURRENT_THREAD_NUM, 0);
    std::vector<u32> actualCrcs(CONCURRENT_THREAD_NUM, 0);
    rankTableJsons.reserve(CONCURRENT_THREAD_NUM);

    for (size_t i = 0; i < CONCURRENT_THREAD_NUM; ++i) {
        rankTableJsons.emplace_back("rank_table_json_" + std::to_string(i));
        CheckCrc checkCrc;
        ASSERT_EQ(checkCrc.CalcStringCrc(rankTableJsons[i].c_str(), &expectedCrcs[i]), HCCL_SUCCESS);
        ASSERT_NE(expectedCrcs[i], 0U);
    }

    std::promise<void> startPromise;
    std::shared_future<void> startSignal = startPromise.get_future().share();
    std::vector<std::thread> threads;
    threads.reserve(CONCURRENT_THREAD_NUM);
    for (size_t i = 0; i < CONCURRENT_THREAD_NUM; ++i) {
        threads.emplace_back([&, i, startSignal]() {
            startSignal.wait();
            s32 deviceLogicId = DEVICE_LOGIC_ID_BASE + static_cast<s32>(i);
            bridge.RecordRankTableJsonCrc(deviceLogicId, rankTableJsons[i]);
            actualCrcs[i] = bridge.ConsumeRankTableJsonCrc(deviceLogicId);
        });
    }

    startPromise.set_value();
    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(actualCrcs, expectedCrcs);
}

TEST_F(RankTableCrcBridgeTest, Ut_ConcurrentConsumeSameDevice_ExpectSingleConsumer)
{
    auto& bridge = RankTableCrcBridge::GetInstance();
    const std::string rankTableJson = "shared_rank_table_json";
    CheckCrc checkCrc;
    u32 expectedCrc = 0;
    ASSERT_EQ(checkCrc.CalcStringCrc(rankTableJson.c_str(), &expectedCrc), HCCL_SUCCESS);
    ASSERT_NE(expectedCrc, 0U);
    bridge.RecordRankTableJsonCrc(SHARED_DEVICE_LOGIC_ID, rankTableJson);

    std::promise<void> startPromise;
    std::shared_future<void> startSignal = startPromise.get_future().share();
    std::vector<u32> consumedCrcs(CONCURRENT_THREAD_NUM, 0);
    std::vector<std::thread> threads;
    threads.reserve(CONCURRENT_THREAD_NUM);
    for (size_t i = 0; i < CONCURRENT_THREAD_NUM; ++i) {
        threads.emplace_back([&, i, startSignal]() {
            startSignal.wait();
            consumedCrcs[i] = bridge.ConsumeRankTableJsonCrc(SHARED_DEVICE_LOGIC_ID);
        });
    }

    startPromise.set_value();
    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(std::count(consumedCrcs.begin(), consumedCrcs.end(), expectedCrc), 1);
    EXPECT_EQ(std::count(consumedCrcs.begin(), consumedCrcs.end(), 0U), CONCURRENT_THREAD_NUM - 1);
    EXPECT_EQ(bridge.ConsumeRankTableJsonCrc(SHARED_DEVICE_LOGIC_ID), 0U);
}
} // namespace
