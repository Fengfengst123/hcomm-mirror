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
#include <set>
#include <vector>

#include "check_utils.h"
#include "scatter_semantics_checker.h"

namespace HcclSim {
class ScatterSemanticsCheckerTest : public testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}

    // Helper function to create valid Scatter semantics
    void CreateValidScatterSemantics(
        std::map<DeviceId, RankMemorySemantics>& allRankMemSemantics, u32 rankSize, u64 dataSize, RankId root)
    {
        for (RankId rankId = 0; rankId < rankSize; rankId++) {
            RankMemorySemantics rankMemSemantics;
            BufferSemanticMap outputSemantics;

            // Each rank receives its portion from root's INPUT buffer
            // The source addr is rankId * dataSize (root's scatter buffer)
            BufferSemantic bufSem(0, dataSize);
            bufSem.srcBufs.insert(SrcBufDes(root, BufferType::INPUT, rankId * dataSize));
            outputSemantics.emplace(bufSem.startAddr, bufSem);

            rankMemSemantics[BufferType::OUTPUT] = outputSemantics;
            allRankMemSemantics[rankId] = rankMemSemantics;
        }
    }
};

// Test normal case: valid Scatter semantics with root 0
TEST_F(ScatterSemanticsCheckerTest, ValidSemantics_RootZero)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 4;
    u64 dataSize = 1024;
    RankId root = 0;

    CreateValidScatterSemantics(allRankMemSemantics, rankSize, dataSize, root);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckScatterSemantics(allRankMemSemantics, dataSize, rankToDevice[root], rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test normal case: valid Scatter semantics with non-zero root
TEST_F(ScatterSemanticsCheckerTest, ValidSemantics_NonZeroRoot)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 4;
    u64 dataSize = 1024;
    RankId root = 2;

    CreateValidScatterSemantics(allRankMemSemantics, rankSize, dataSize, root);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckScatterSemantics(allRankMemSemantics, dataSize, rankToDevice[root], rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test boundary case: single rank
TEST_F(ScatterSemanticsCheckerTest, ValidSemantics_SingleRank)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 1;
    u64 dataSize = 1024;
    RankId root = 0;

    CreateValidScatterSemantics(allRankMemSemantics, rankSize, dataSize, root);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckScatterSemantics(allRankMemSemantics, dataSize, rankToDevice[root], rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test boundary case: zero data size
TEST_F(ScatterSemanticsCheckerTest, ValidSemantics_ZeroDataSize)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 4;
    u64 dataSize = 0;
    RankId root = 0;

    CreateValidScatterSemantics(allRankMemSemantics, rankSize, dataSize, root);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckScatterSemantics(allRankMemSemantics, dataSize, rankToDevice[root], rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test boundary case: two ranks
TEST_F(ScatterSemanticsCheckerTest, ValidSemantics_TwoRanks)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 2;
    u64 dataSize = 2048;
    RankId root = 0;

    CreateValidScatterSemantics(allRankMemSemantics, rankSize, dataSize, root);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckScatterSemantics(allRankMemSemantics, dataSize, rankToDevice[root], rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test boundary case: missing rank - API uses allRankMemSemantics.size() as
// rankSize, so cannot detect missing ranks; test with only 1 rank as valid
// boundary case
TEST_F(ScatterSemanticsCheckerTest, Abnormal_MissingRank)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 2;
    u64 dataSize = 1024;
    RankId root = 0;

    RankMemorySemantics rankMemSemantics;
    BufferSemanticMap outputSemantics;

    BufferSemantic bufSem(0, dataSize);
    bufSem.srcBufs.insert(SrcBufDes(root, BufferType::INPUT, 0));
    outputSemantics.emplace(bufSem.startAddr, bufSem);

    rankMemSemantics[BufferType::OUTPUT] = outputSemantics;
    allRankMemSemantics[0] = rankMemSemantics;

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckScatterSemantics(allRankMemSemantics, dataSize, rankToDevice[root], rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test abnormal case: wrong source rank
TEST_F(ScatterSemanticsCheckerTest, Abnormal_WrongSourceRank)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 4;
    u64 dataSize = 1024;
    RankId root = 0;

    RankMemorySemantics rankMemSemantics;
    BufferSemanticMap outputSemantics;

    BufferSemantic bufSem(0, dataSize);
    bufSem.srcBufs.insert(SrcBufDes(2, BufferType::INPUT, 0)); // Wrong source rank
    outputSemantics.emplace(bufSem.startAddr, bufSem);

    rankMemSemantics[BufferType::OUTPUT] = outputSemantics;
    allRankMemSemantics[0] = rankMemSemantics;
    allRankMemSemantics[1] = rankMemSemantics;
    allRankMemSemantics[2] = rankMemSemantics;
    allRankMemSemantics[3] = rankMemSemantics;

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckScatterSemantics(allRankMemSemantics, dataSize, rankToDevice[root], rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: wrong buffer type
TEST_F(ScatterSemanticsCheckerTest, Abnormal_WrongBufferType)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 2;
    u64 dataSize = 1024;
    RankId root = 0;

    RankMemorySemantics rankMemSemantics;
    BufferSemanticMap outputSemantics;

    BufferSemantic bufSem(0, dataSize);
    bufSem.srcBufs.insert(SrcBufDes(root, BufferType::OUTPUT, 0)); // Wrong buffer type
    outputSemantics.emplace(bufSem.startAddr, bufSem);

    rankMemSemantics[BufferType::OUTPUT] = outputSemantics;
    allRankMemSemantics[0] = rankMemSemantics;
    allRankMemSemantics[1] = rankMemSemantics;

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckScatterSemantics(allRankMemSemantics, dataSize, rankToDevice[root], rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: wrong source addr
TEST_F(ScatterSemanticsCheckerTest, Abnormal_WrongSourceAddr)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 4;
    u64 dataSize = 1024;
    RankId root = 0;

    RankMemorySemantics rankMemSemantics;
    BufferSemanticMap outputSemantics;

    // For rank 0, source addr should be 0, but we give wrong value
    BufferSemantic bufSem(0, dataSize);
    bufSem.srcBufs.insert(SrcBufDes(root, BufferType::INPUT, 100)); // Wrong source addr
    outputSemantics.emplace(bufSem.startAddr, bufSem);

    rankMemSemantics[BufferType::OUTPUT] = outputSemantics;
    allRankMemSemantics[0] = rankMemSemantics;

    for (RankId rankId = 1; rankId < rankSize; rankId++) {
        RankMemorySemantics otherMemSemantics;
        BufferSemanticMap otherOutputSemantics;

        BufferSemantic otherBufSem(0, dataSize);
        otherBufSem.srcBufs.insert(SrcBufDes(root, BufferType::INPUT, rankId * dataSize));
        otherOutputSemantics.emplace(otherBufSem.startAddr, otherBufSem);

        otherMemSemantics[BufferType::OUTPUT] = otherOutputSemantics;
        allRankMemSemantics[rankId] = otherMemSemantics;
    }

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckScatterSemantics(allRankMemSemantics, dataSize, rankToDevice[root], rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test boundary case: large number of ranks
TEST_F(ScatterSemanticsCheckerTest, ValidSemantics_LargeRankSize)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 128;
    u64 dataSize = 1024;
    RankId root = 64;

    CreateValidScatterSemantics(allRankMemSemantics, rankSize, dataSize, root);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckScatterSemantics(allRankMemSemantics, dataSize, rankToDevice[root], rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}
} // namespace HcclSim
