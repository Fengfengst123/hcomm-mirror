/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include "channel_profiling_adpt.h"

namespace {
HcclResult StubGetChannelRemoteRankId(const std::string& commTag, u64 handle, u32& remoteRankId)
{
    (void)commTag;
    (void)handle;
    remoteRankId = 42;
    return HCCL_SUCCESS;
}
} // namespace

class UtChannelProfilingAdptTest : public testing::Test {
protected:
    void TearDown() override { hcomm::RegisterGetChannelRemoteRankId(nullptr); }
};

TEST_F(UtChannelProfilingAdptTest, GetFuncImplReturnNullptrWhenNotRegistered)
{
    hcomm::RegisterGetChannelRemoteRankId(nullptr);
    EXPECT_EQ(hcomm::GetChannelRemoteRankIdFuncImpl(), nullptr);
}

TEST_F(UtChannelProfilingAdptTest, GetFuncImplReturnFuncAfterRegister)
{
    hcomm::RegisterGetChannelRemoteRankId(&StubGetChannelRemoteRankId);
    EXPECT_EQ(hcomm::GetChannelRemoteRankIdFuncImpl(), &StubGetChannelRemoteRankId);
}

TEST_F(UtChannelProfilingAdptTest, RegisteredFuncCanBeInvoked)
{
    hcomm::RegisterGetChannelRemoteRankId(&StubGetChannelRemoteRankId);
    hcomm::GetChannelRemoteRankIdFunc getRemoteRankIdFunc = hcomm::GetChannelRemoteRankIdFuncImpl();
    ASSERT_NE(getRemoteRankIdFunc, nullptr);
    u32 remoteRankId = 0;
    EXPECT_EQ(getRemoteRankIdFunc("ut_comm_tag", 1, remoteRankId), HCCL_SUCCESS);
    EXPECT_EQ(remoteRankId, 42U);
}

TEST_F(UtChannelProfilingAdptTest, ConcurrentRegisterAndGetFuncImpl)
{
    const u32 loopTimes = 1000;
    std::atomic<bool> readerResultValid{true};
    std::thread writerThread([&loopTimes] {
        for (u32 idx = 0; idx < loopTimes; ++idx) {
            hcomm::RegisterGetChannelRemoteRankId(&StubGetChannelRemoteRankId);
        }
    });
    std::thread readerThread([&loopTimes, &readerResultValid] {
        for (u32 idx = 0; idx < loopTimes; ++idx) {
            hcomm::GetChannelRemoteRankIdFunc getRemoteRankIdFunc = hcomm::GetChannelRemoteRankIdFuncImpl();
            if ((getRemoteRankIdFunc != nullptr) && (getRemoteRankIdFunc != &StubGetChannelRemoteRankId)) {
                readerResultValid.store(false);
            }
        }
    });
    writerThread.join();
    readerThread.join();
    EXPECT_TRUE(readerResultValid.load());
}
