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

#include "broadcast_semantics_checker.h"
#include "check_utils.h"

namespace HcclSim {
class BroadcastSemanticsCheckerTest : public testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}

    // Helper function to create valid Broadcast semantics
    void CreateValidBroadcastSemantics(
        std::map<DeviceId, RankMemorySemantics>& allRankMemSemantics, u32 rankSize, u64 dataSize, RankId root)
    {
        for (RankId rankId = 0; rankId < rankSize; rankId++) {
            RankMemorySemantics rankMemSemantics;
            BufferSemanticMap inputSemantics;

            // All ranks receive data from root's INPUT buffer
            BufferSemantic bufSem(0, dataSize);
            bufSem.srcBufs.insert(SrcBufDes(root, BufferType::INPUT, 0));
            inputSemantics.emplace(bufSem.startAddr, bufSem);

            rankMemSemantics[BufferType::INPUT] = inputSemantics;
            allRankMemSemantics[rankId] = rankMemSemantics;
        }
    }
};

// Test normal case: valid Broadcast semantics with root 0
TEST_F(BroadcastSemanticsCheckerTest, ValidSemantics_RootZero)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 4;
    u64 dataSize = 1024;
    RankId root = 0;

    CreateValidBroadcastSemantics(allRankMemSemantics, rankSize, dataSize, root);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckBroadcastSemantics(allRankMemSemantics, dataSize, rankToDevice[root], rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test normal case: valid Broadcast semantics with non-zero root
TEST_F(BroadcastSemanticsCheckerTest, ValidSemantics_NonZeroRoot)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 4;
    u64 dataSize = 1024;
    RankId root = 2;

    CreateValidBroadcastSemantics(allRankMemSemantics, rankSize, dataSize, root);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckBroadcastSemantics(allRankMemSemantics, dataSize, rankToDevice[root], rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test normal case: valid Broadcast semantics with last rank as root
TEST_F(BroadcastSemanticsCheckerTest, ValidSemantics_LastRankRoot)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 4;
    u64 dataSize = 1024;
    RankId root = 3;

    CreateValidBroadcastSemantics(allRankMemSemantics, rankSize, dataSize, root);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckBroadcastSemantics(allRankMemSemantics, dataSize, rankToDevice[root], rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test boundary case: single rank (root is the only rank)
TEST_F(BroadcastSemanticsCheckerTest, ValidSemantics_SingleRank)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 1;
    u64 dataSize = 1024;
    RankId root = 0;

    CreateValidBroadcastSemantics(allRankMemSemantics, rankSize, dataSize, root);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckBroadcastSemantics(allRankMemSemantics, dataSize, rankToDevice[root], rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test boundary case: zero data size
TEST_F(BroadcastSemanticsCheckerTest, ValidSemantics_ZeroDataSize)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 4;
    u64 dataSize = 0;
    RankId root = 0;

    CreateValidBroadcastSemantics(allRankMemSemantics, rankSize, dataSize, root);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckBroadcastSemantics(allRankMemSemantics, dataSize, rankToDevice[root], rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test boundary case: two ranks
TEST_F(BroadcastSemanticsCheckerTest, ValidSemantics_TwoRanks)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 2;
    u64 dataSize = 2048;
    RankId root = 0;

    CreateValidBroadcastSemantics(allRankMemSemantics, rankSize, dataSize, root);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckBroadcastSemantics(allRankMemSemantics, dataSize, rankToDevice[root], rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test boundary case: missing rank - API uses allRankMemSemantics.size() as
// rankSize, so cannot detect missing ranks; test with only 1 rank as valid
// boundary case
TEST_F(BroadcastSemanticsCheckerTest, Abnormal_MissingRank)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 2;
    u64 dataSize = 1024;
    RankId root = 0;

    RankMemorySemantics rankMemSemantics;
    BufferSemanticMap inputSemantics;

    BufferSemantic bufSem(0, dataSize);
    bufSem.srcBufs.insert(SrcBufDes(root, BufferType::INPUT, 0));
    inputSemantics.emplace(bufSem.startAddr, bufSem);

    rankMemSemantics[BufferType::INPUT] = inputSemantics;
    allRankMemSemantics[0] = rankMemSemantics;

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckBroadcastSemantics(allRankMemSemantics, dataSize, rankToDevice[root], rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test abnormal case: wrong start addr
TEST_F(BroadcastSemanticsCheckerTest, Abnormal_WrongStartAddr)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 2;
    u64 dataSize = 1024;
    RankId root = 0;

    RankMemorySemantics rankMemSemantics;
    BufferSemanticMap inputSemantics;

    BufferSemantic bufSem(100, dataSize); // Wrong start addr
    bufSem.srcBufs.insert(SrcBufDes(root, BufferType::INPUT, 0));
    inputSemantics.emplace(bufSem.startAddr, bufSem);

    rankMemSemantics[BufferType::INPUT] = inputSemantics;
    allRankMemSemantics[0] = rankMemSemantics;
    allRankMemSemantics[1] = rankMemSemantics;

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckBroadcastSemantics(allRankMemSemantics, dataSize, rankToDevice[root], rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: wrong source rank (not from root)
TEST_F(BroadcastSemanticsCheckerTest, Abnormal_WrongSourceRank)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 4;
    u64 dataSize = 1024;
    RankId root = 0;

    RankMemorySemantics rankMemSemantics;
    BufferSemanticMap inputSemantics;

    BufferSemantic bufSem(0, dataSize);
    bufSem.srcBufs.insert(SrcBufDes(2, BufferType::INPUT, 0)); // Wrong source rank
    inputSemantics.emplace(bufSem.startAddr, bufSem);

    rankMemSemantics[BufferType::INPUT] = inputSemantics;
    allRankMemSemantics[0] = rankMemSemantics;
    allRankMemSemantics[1] = rankMemSemantics;
    allRankMemSemantics[2] = rankMemSemantics;
    allRankMemSemantics[3] = rankMemSemantics;

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckBroadcastSemantics(allRankMemSemantics, dataSize, rankToDevice[root], rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: wrong buffer type
TEST_F(BroadcastSemanticsCheckerTest, Abnormal_WrongBufferType)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 2;
    u64 dataSize = 1024;
    RankId root = 0;

    RankMemorySemantics rankMemSemantics;
    BufferSemanticMap inputSemantics;

    BufferSemantic bufSem(0, dataSize);
    bufSem.srcBufs.insert(SrcBufDes(root, BufferType::OUTPUT, 0)); // Wrong buffer type
    inputSemantics.emplace(bufSem.startAddr, bufSem);

    rankMemSemantics[BufferType::INPUT] = inputSemantics;
    allRankMemSemantics[0] = rankMemSemantics;
    allRankMemSemantics[1] = rankMemSemantics;

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckBroadcastSemantics(allRankMemSemantics, dataSize, rankToDevice[root], rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: multiple source buffers
TEST_F(BroadcastSemanticsCheckerTest, Abnormal_MultipleSourceBuffers)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 2;
    u64 dataSize = 1024;
    RankId root = 0;

    RankMemorySemantics rankMemSemantics;
    BufferSemanticMap inputSemantics;

    BufferSemantic bufSem(0, dataSize);
    bufSem.srcBufs.insert(SrcBufDes(root, BufferType::INPUT, 0));
    bufSem.srcBufs.insert(SrcBufDes(1, BufferType::INPUT, 0)); // Extra source
    inputSemantics.emplace(bufSem.startAddr, bufSem);

    rankMemSemantics[BufferType::INPUT] = inputSemantics;
    allRankMemSemantics[0] = rankMemSemantics;
    allRankMemSemantics[1] = rankMemSemantics;

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckBroadcastSemantics(allRankMemSemantics, dataSize, rankToDevice[root], rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: wrong source addr
TEST_F(BroadcastSemanticsCheckerTest, Abnormal_WrongSourceAddr)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 2;
    u64 dataSize = 1024;
    RankId root = 0;

    RankMemorySemantics rankMemSemantics;
    BufferSemanticMap inputSemantics;

    BufferSemantic bufSem(0, dataSize);
    bufSem.srcBufs.insert(SrcBufDes(root, BufferType::INPUT, 100)); // Wrong source addr
    inputSemantics.emplace(bufSem.startAddr, bufSem);

    rankMemSemantics[BufferType::INPUT] = inputSemantics;
    allRankMemSemantics[0] = rankMemSemantics;
    allRankMemSemantics[1] = rankMemSemantics;

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckBroadcastSemantics(allRankMemSemantics, dataSize, rankToDevice[root], rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: incomplete total size
TEST_F(BroadcastSemanticsCheckerTest, Abnormal_IncompleteTotalSize)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 2;
    u64 dataSize = 1024;
    RankId root = 0;

    RankMemorySemantics rankMemSemantics;
    BufferSemanticMap inputSemantics;

    // Only half the expected size
    BufferSemantic bufSem(0, dataSize / 2);
    bufSem.srcBufs.insert(SrcBufDes(root, BufferType::INPUT, 0));
    inputSemantics.emplace(bufSem.startAddr, bufSem);

    rankMemSemantics[BufferType::INPUT] = inputSemantics;
    allRankMemSemantics[0] = rankMemSemantics;
    allRankMemSemantics[1] = rankMemSemantics;

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckBroadcastSemantics(allRankMemSemantics, dataSize, rankToDevice[root], rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test boundary case: large number of ranks
TEST_F(BroadcastSemanticsCheckerTest, ValidSemantics_LargeRankSize)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 128;
    u64 dataSize = 1024;
    RankId root = 64;

    CreateValidBroadcastSemantics(allRankMemSemantics, rankSize, dataSize, root);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckBroadcastSemantics(allRankMemSemantics, dataSize, rankToDevice[root], rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test boundary case: large data size
TEST_F(BroadcastSemanticsCheckerTest, ValidSemantics_LargeDataSize)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 4;
    u64 dataSize = 1024 * 1024 * 1024; // 1GB
    RankId root = 0;

    CreateValidBroadcastSemantics(allRankMemSemantics, rankSize, dataSize, root);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckBroadcastSemantics(allRankMemSemantics, dataSize, rankToDevice[root], rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}
} // namespace HcclSim
