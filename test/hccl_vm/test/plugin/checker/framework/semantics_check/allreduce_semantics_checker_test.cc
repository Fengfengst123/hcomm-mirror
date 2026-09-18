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
#include <map>
#include <vector>

#include "allreduce_semantics_checker.h"
#include "check_utils.h"

namespace HcclSim {
class AllReduceSemanticsCheckerTest : public testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}

    // Helper function to create valid AllReduce semantics
    void CreateValidAllReduceSemantics(
        std::map<DeviceId, RankMemorySemantics>& allRankMemSemantics, u32 rankSize, u64 dataSize,
        HcclReduceOp reduceType)
    {
        for (RankId rankId = 0; rankId < rankSize; rankId++) {
            RankMemorySemantics rankMemSemantics;
            BufferSemanticMap outputSemantics;

            BufferSemantic bufSem(0, dataSize, true, reduceType);
            for (RankId srcRank = 0; srcRank < rankSize; srcRank++) {
                bufSem.srcBufs.insert(SrcBufDes(srcRank, BufferType::INPUT, 0));
            }
            outputSemantics.emplace(bufSem.startAddr, bufSem);

            rankMemSemantics[BufferType::OUTPUT] = outputSemantics;
            allRankMemSemantics[rankId] = rankMemSemantics;
        }
    }
};

// Test normal case: valid AllReduce semantics with 2 ranks
TEST_F(AllReduceSemanticsCheckerTest, ValidSemantics_TwoRanks)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 2;
    u64 dataSize = 1024;

    CreateValidAllReduceSemantics(allRankMemSemantics, rankSize, dataSize, HcclReduceOp::HCCL_REDUCE_SUM);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result
        = TaskCheckAllReduceSemantics(allRankMemSemantics, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test normal case: valid AllReduce semantics with 4 ranks
TEST_F(AllReduceSemanticsCheckerTest, ValidSemantics_FourRanks)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 4;
    u64 dataSize = 2048;

    CreateValidAllReduceSemantics(allRankMemSemantics, rankSize, dataSize, HcclReduceOp::HCCL_REDUCE_SUM);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result
        = TaskCheckAllReduceSemantics(allRankMemSemantics, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test normal case: different reduce operations
TEST_F(AllReduceSemanticsCheckerTest, ValidSemantics_ReduceProd)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 4;
    u64 dataSize = 1024;

    CreateValidAllReduceSemantics(allRankMemSemantics, rankSize, dataSize, HcclReduceOp::HCCL_REDUCE_PROD);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result
        = TaskCheckAllReduceSemantics(allRankMemSemantics, dataSize, HcclReduceOp::HCCL_REDUCE_PROD, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceSemanticsCheckerTest, ValidSemantics_ReduceMax)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 4;
    u64 dataSize = 1024;

    CreateValidAllReduceSemantics(allRankMemSemantics, rankSize, dataSize, HcclReduceOp::HCCL_REDUCE_MAX);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result
        = TaskCheckAllReduceSemantics(allRankMemSemantics, dataSize, HcclReduceOp::HCCL_REDUCE_MAX, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceSemanticsCheckerTest, ValidSemantics_ReduceMin)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 4;
    u64 dataSize = 1024;

    CreateValidAllReduceSemantics(allRankMemSemantics, rankSize, dataSize, HcclReduceOp::HCCL_REDUCE_MIN);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result
        = TaskCheckAllReduceSemantics(allRankMemSemantics, dataSize, HcclReduceOp::HCCL_REDUCE_MIN, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test boundary case: single rank
TEST_F(AllReduceSemanticsCheckerTest, ValidSemantics_SingleRank)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 1;
    u64 dataSize = 1024;

    CreateValidAllReduceSemantics(allRankMemSemantics, rankSize, dataSize, HcclReduceOp::HCCL_REDUCE_SUM);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result
        = TaskCheckAllReduceSemantics(allRankMemSemantics, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test boundary case: zero data size
TEST_F(AllReduceSemanticsCheckerTest, ValidSemantics_ZeroDataSize)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 2;
    u64 dataSize = 0;

    CreateValidAllReduceSemantics(allRankMemSemantics, rankSize, dataSize, HcclReduceOp::HCCL_REDUCE_SUM);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result
        = TaskCheckAllReduceSemantics(allRankMemSemantics, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test abnormal case: missing rank
TEST_F(AllReduceSemanticsCheckerTest, Abnormal_MissingRank)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 2;
    u64 dataSize = 1024;

    // Only add rank 0
    RankMemorySemantics rankMemSemantics;
    BufferSemanticMap outputSemantics;

    BufferSemantic bufSem(0, dataSize, true, HcclReduceOp::HCCL_REDUCE_SUM);
    bufSem.srcBufs.insert(SrcBufDes(0, BufferType::INPUT, 0));
    bufSem.srcBufs.insert(SrcBufDes(1, BufferType::INPUT, 0));
    outputSemantics.emplace(bufSem.startAddr, bufSem);

    rankMemSemantics[BufferType::OUTPUT] = outputSemantics;
    allRankMemSemantics[0] = rankMemSemantics;

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result
        = TaskCheckAllReduceSemantics(allRankMemSemantics, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: wrong reduce type
TEST_F(AllReduceSemanticsCheckerTest, Abnormal_WrongReduceType)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 2;
    u64 dataSize = 1024;

    // Create with SUM but check with PROD
    CreateValidAllReduceSemantics(allRankMemSemantics, rankSize, dataSize, HcclReduceOp::HCCL_REDUCE_SUM);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result
        = TaskCheckAllReduceSemantics(allRankMemSemantics, dataSize, HcclReduceOp::HCCL_REDUCE_PROD, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: insufficient source buffers
TEST_F(AllReduceSemanticsCheckerTest, Abnormal_InsufficientSourceBuffers)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 4;
    u64 dataSize = 1024;

    RankMemorySemantics rankMemSemantics;
    BufferSemanticMap outputSemantics;

    BufferSemantic bufSem(0, dataSize, true, HcclReduceOp::HCCL_REDUCE_SUM);
    // Only add 2 source buffers instead of 4
    bufSem.srcBufs.insert(SrcBufDes(0, BufferType::INPUT, 0));
    bufSem.srcBufs.insert(SrcBufDes(1, BufferType::INPUT, 0));
    outputSemantics.emplace(bufSem.startAddr, bufSem);

    rankMemSemantics[BufferType::OUTPUT] = outputSemantics;
    allRankMemSemantics[0] = rankMemSemantics;
    allRankMemSemantics[1] = rankMemSemantics;
    allRankMemSemantics[2] = rankMemSemantics;
    allRankMemSemantics[3] = rankMemSemantics;

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result
        = TaskCheckAllReduceSemantics(allRankMemSemantics, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: wrong source buffer type
TEST_F(AllReduceSemanticsCheckerTest, Abnormal_WrongBufferType)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 2;
    u64 dataSize = 1024;

    RankMemorySemantics rankMemSemantics;
    BufferSemanticMap outputSemantics;

    BufferSemantic bufSem(0, dataSize, true, HcclReduceOp::HCCL_REDUCE_SUM);
    bufSem.srcBufs.insert(SrcBufDes(0, BufferType::OUTPUT, 0)); // Wrong buffer type
    bufSem.srcBufs.insert(SrcBufDes(1, BufferType::INPUT, 0));
    outputSemantics.emplace(bufSem.startAddr, bufSem);

    rankMemSemantics[BufferType::OUTPUT] = outputSemantics;
    allRankMemSemantics[0] = rankMemSemantics;
    allRankMemSemantics[1] = rankMemSemantics;

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result
        = TaskCheckAllReduceSemantics(allRankMemSemantics, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: invalid source rank range
TEST_F(AllReduceSemanticsCheckerTest, Abnormal_InvalidSourceRankRange)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 2;
    u64 dataSize = 1024;

    RankMemorySemantics rankMemSemantics;
    BufferSemanticMap outputSemantics;

    BufferSemantic bufSem(0, dataSize, true, HcclReduceOp::HCCL_REDUCE_SUM);
    bufSem.srcBufs.insert(SrcBufDes(0, BufferType::INPUT, 0));
    bufSem.srcBufs.insert(SrcBufDes(2, BufferType::INPUT, 0)); // Invalid rank (out of range)
    outputSemantics.emplace(bufSem.startAddr, bufSem);

    rankMemSemantics[BufferType::OUTPUT] = outputSemantics;
    allRankMemSemantics[0] = rankMemSemantics;
    allRankMemSemantics[1] = rankMemSemantics;

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result
        = TaskCheckAllReduceSemantics(allRankMemSemantics, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: wrong total size
TEST_F(AllReduceSemanticsCheckerTest, Abnormal_WrongTotalSize)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 2;
    u64 dataSize = 1024;

    RankMemorySemantics rankMemSemantics;
    BufferSemanticMap outputSemantics;

    // Create buffer with wrong size
    BufferSemantic bufSem(0, 512, true,
                          HcclReduceOp::HCCL_REDUCE_SUM); // Wrong size
    bufSem.srcBufs.insert(SrcBufDes(0, BufferType::INPUT, 0));
    bufSem.srcBufs.insert(SrcBufDes(1, BufferType::INPUT, 0));
    outputSemantics.emplace(bufSem.startAddr, bufSem);

    rankMemSemantics[BufferType::OUTPUT] = outputSemantics;
    allRankMemSemantics[0] = rankMemSemantics;
    allRankMemSemantics[1] = rankMemSemantics;

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result
        = TaskCheckAllReduceSemantics(allRankMemSemantics, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test boundary case: large number of ranks
TEST_F(AllReduceSemanticsCheckerTest, ValidSemantics_LargeRankSize)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 128;
    u64 dataSize = 1024;

    CreateValidAllReduceSemantics(allRankMemSemantics, rankSize, dataSize, HcclReduceOp::HCCL_REDUCE_SUM);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result
        = TaskCheckAllReduceSemantics(allRankMemSemantics, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test boundary case: large data size
TEST_F(AllReduceSemanticsCheckerTest, ValidSemantics_LargeDataSize)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 4;
    u64 dataSize = 1024 * 1024 * 1024; // 1GB

    CreateValidAllReduceSemantics(allRankMemSemantics, rankSize, dataSize, HcclReduceOp::HCCL_REDUCE_SUM);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result
        = TaskCheckAllReduceSemantics(allRankMemSemantics, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}
} // namespace HcclSim
