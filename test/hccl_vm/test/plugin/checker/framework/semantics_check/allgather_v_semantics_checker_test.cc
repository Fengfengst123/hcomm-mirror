/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <gtest/gtest.h>
#include <map>
#include <vector>

#include "allgather_v_semantics_checker.h"
#include "check_utils.h"
#include "checker_def.h"

namespace HcclSim {
class AllGatherVSemanticsCheckerTest : public testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}

    // Helper function to create valid AllGatherV semantics
    void CreateValidAllGatherVSemantics(
        std::map<DeviceId, RankMemorySemantics>& allRankMemSemantics, VDataDesTagInner& vDataDes, u32 rankSize,
        const std::vector<uint64_t>& counts)
    {
        vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
        vDataDes.counts.clear();
        for (auto count : counts) {
            vDataDes.counts.push_back(count);
        }

        for (RankId rankId = 0; rankId < rankSize; rankId++) {
            RankMemorySemantics rankMemSemantics;
            BufferSemanticMap outputSemantics;

            u64 totalSize = 0;
            for (RankId srcRank = 0; srcRank < rankSize; srcRank++) {
                u64 curDataSize = counts[srcRank] * CHECK_SIZE_TABLE[vDataDes.dataType];
                if (curDataSize == 0) {
                    continue;
                }

                BufferSemantic bufSem(totalSize, curDataSize);
                bufSem.srcBufs.insert(SrcBufDes(srcRank, BufferType::INPUT, 0));
                outputSemantics.emplace(bufSem.startAddr, bufSem);
                totalSize += curDataSize;
            }

            rankMemSemantics[BufferType::OUTPUT] = outputSemantics;
            allRankMemSemantics[rankId] = rankMemSemantics;
        }
    }
};

// Test normal case: valid AllGatherV semantics with equal counts
TEST_F(AllGatherVSemanticsCheckerTest, ValidSemantics_EqualCounts)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    VDataDesTagInner vDataDes;
    u32 rankSize = 4;
    std::vector<uint64_t> counts = {100, 100, 100, 100};

    CreateValidAllGatherVSemantics(allRankMemSemantics, vDataDes, rankSize, counts);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckAllGatherVSemantics(allRankMemSemantics, vDataDes, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test normal case: valid AllGatherV semantics with different counts
TEST_F(AllGatherVSemanticsCheckerTest, ValidSemantics_DifferentCounts)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    VDataDesTagInner vDataDes;
    u32 rankSize = 4;
    std::vector<uint64_t> counts = {50, 100, 150, 200};

    CreateValidAllGatherVSemantics(allRankMemSemantics, vDataDes, rankSize, counts);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckAllGatherVSemantics(allRankMemSemantics, vDataDes, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test boundary case: single rank
TEST_F(AllGatherVSemanticsCheckerTest, ValidSemantics_SingleRank)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    VDataDesTagInner vDataDes;
    u32 rankSize = 1;
    std::vector<uint64_t> counts = {100};

    CreateValidAllGatherVSemantics(allRankMemSemantics, vDataDes, rankSize, counts);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckAllGatherVSemantics(allRankMemSemantics, vDataDes, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test boundary case: zero counts for some ranks
TEST_F(AllGatherVSemanticsCheckerTest, ValidSemantics_ZeroCounts)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    VDataDesTagInner vDataDes;
    u32 rankSize = 4;
    std::vector<uint64_t> counts = {0, 100, 0, 200};

    CreateValidAllGatherVSemantics(allRankMemSemantics, vDataDes, rankSize, counts);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckAllGatherVSemantics(allRankMemSemantics, vDataDes, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test boundary case: all zero counts
TEST_F(AllGatherVSemanticsCheckerTest, ValidSemantics_AllZeroCounts)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    VDataDesTagInner vDataDes;
    u32 rankSize = 4;
    std::vector<uint64_t> counts = {0, 0, 0, 0};

    CreateValidAllGatherVSemantics(allRankMemSemantics, vDataDes, rankSize, counts);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckAllGatherVSemantics(allRankMemSemantics, vDataDes, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test abnormal case: missing rank
TEST_F(AllGatherVSemanticsCheckerTest, Abnormal_MissingRank)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    VDataDesTagInner vDataDes;
    u32 rankSize = 2;
    std::vector<uint64_t> counts = {100, 100};

    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    vDataDes.counts = counts;

    // Only add rank 0, missing rank 1
    RankMemorySemantics rankMemSemantics;
    BufferSemanticMap outputSemantics;

    u64 dataSize = counts[0] * CHECK_SIZE_TABLE[vDataDes.dataType];
    BufferSemantic bufSem0(0, dataSize);
    bufSem0.srcBufs.insert(SrcBufDes(0, BufferType::INPUT, 0));
    outputSemantics.emplace(bufSem0.startAddr, bufSem0);

    dataSize = counts[1] * CHECK_SIZE_TABLE[vDataDes.dataType];
    BufferSemantic bufSem1(400, dataSize);
    bufSem1.srcBufs.insert(SrcBufDes(1, BufferType::INPUT, 0));
    outputSemantics.emplace(bufSem1.startAddr, bufSem1);

    rankMemSemantics[BufferType::OUTPUT] = outputSemantics;
    allRankMemSemantics[0] = rankMemSemantics;

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckAllGatherVSemantics(allRankMemSemantics, vDataDes, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: wrong start addr
TEST_F(AllGatherVSemanticsCheckerTest, Abnormal_WrongStartAddr)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    VDataDesTagInner vDataDes;
    u32 rankSize = 2;
    std::vector<uint64_t> counts = {100, 100};

    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    vDataDes.counts = counts;

    RankMemorySemantics rankMemSemantics;
    BufferSemanticMap outputSemantics;

    u64 dataSize = counts[0] * CHECK_SIZE_TABLE[vDataDes.dataType];
    BufferSemantic bufSem0(100, dataSize); // Wrong start addr
    bufSem0.srcBufs.insert(SrcBufDes(0, BufferType::INPUT, 0));
    outputSemantics.emplace(bufSem0.startAddr, bufSem0);

    dataSize = counts[1] * CHECK_SIZE_TABLE[vDataDes.dataType];
    BufferSemantic bufSem1(500, dataSize);
    bufSem1.srcBufs.insert(SrcBufDes(1, BufferType::INPUT, 0));
    outputSemantics.emplace(bufSem1.startAddr, bufSem1);

    rankMemSemantics[BufferType::OUTPUT] = outputSemantics;
    allRankMemSemantics[0] = rankMemSemantics;
    allRankMemSemantics[1] = rankMemSemantics;

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckAllGatherVSemantics(allRankMemSemantics, vDataDes, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: wrong source rank
TEST_F(AllGatherVSemanticsCheckerTest, Abnormal_WrongSourceRank)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    VDataDesTagInner vDataDes;
    u32 rankSize = 2;
    std::vector<uint64_t> counts = {100, 100};

    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    vDataDes.counts = counts;

    RankMemorySemantics rankMemSemantics;
    BufferSemanticMap outputSemantics;

    u64 dataSize = counts[0] * CHECK_SIZE_TABLE[vDataDes.dataType];
    BufferSemantic bufSem0(0, dataSize);
    bufSem0.srcBufs.insert(SrcBufDes(1, BufferType::INPUT, 0)); // Wrong source rank
    outputSemantics.emplace(bufSem0.startAddr, bufSem0);

    dataSize = counts[1] * CHECK_SIZE_TABLE[vDataDes.dataType];
    BufferSemantic bufSem1(400, dataSize);
    bufSem1.srcBufs.insert(SrcBufDes(0, BufferType::INPUT, 0)); // Wrong source rank
    outputSemantics.emplace(bufSem1.startAddr, bufSem1);

    rankMemSemantics[BufferType::OUTPUT] = outputSemantics;
    allRankMemSemantics[0] = rankMemSemantics;
    allRankMemSemantics[1] = rankMemSemantics;

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckAllGatherVSemantics(allRankMemSemantics, vDataDes, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: incomplete total size
TEST_F(AllGatherVSemanticsCheckerTest, Abnormal_IncompleteTotalSize)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    VDataDesTagInner vDataDes;
    u32 rankSize = 2;
    std::vector<uint64_t> counts = {100, 100};

    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    vDataDes.counts = counts;

    RankMemorySemantics rankMemSemantics;
    BufferSemanticMap outputSemantics;

    // Only add one buffer, missing the second
    u64 dataSize = counts[0] * CHECK_SIZE_TABLE[vDataDes.dataType];
    BufferSemantic bufSem0(0, dataSize);
    bufSem0.srcBufs.insert(SrcBufDes(0, BufferType::INPUT, 0));
    outputSemantics.emplace(bufSem0.startAddr, bufSem0);

    rankMemSemantics[BufferType::OUTPUT] = outputSemantics;
    allRankMemSemantics[0] = rankMemSemantics;
    allRankMemSemantics[1] = rankMemSemantics;

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckAllGatherVSemantics(allRankMemSemantics, vDataDes, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test boundary case: different data types
TEST_F(AllGatherVSemanticsCheckerTest, ValidSemantics_DifferentDataTypes)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    VDataDesTagInner vDataDes;
    u32 rankSize = 2;
    std::vector<uint64_t> counts = {100, 200};

    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP16;
    vDataDes.counts = counts;

    for (RankId rankId = 0; rankId < rankSize; rankId++) {
        RankMemorySemantics rankMemSemantics;
        BufferSemanticMap outputSemantics;

        u64 totalSize = 0;
        for (RankId srcRank = 0; srcRank < rankSize; srcRank++) {
            u64 curDataSize = counts[srcRank] * CHECK_SIZE_TABLE[vDataDes.dataType];

            BufferSemantic bufSem(totalSize, curDataSize);
            bufSem.srcBufs.insert(SrcBufDes(srcRank, BufferType::INPUT, 0));
            outputSemantics.emplace(bufSem.startAddr, bufSem);
            totalSize += curDataSize;
        }

        rankMemSemantics[BufferType::OUTPUT] = outputSemantics;
        allRankMemSemantics[rankId] = rankMemSemantics;
    }

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckAllGatherVSemantics(allRankMemSemantics, vDataDes, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test boundary case: large counts
TEST_F(AllGatherVSemanticsCheckerTest, ValidSemantics_LargeCounts)
{
    std::map<DeviceId, RankMemorySemantics> allRankMemSemantics;
    VDataDesTagInner vDataDes;
    u32 rankSize = 4;
    std::vector<uint64_t> counts = {1000000, 2000000, 3000000, 4000000};

    CreateValidAllGatherVSemantics(allRankMemSemantics, vDataDes, rankSize, counts);

    std::vector<DeviceId> rankToDevice(rankSize);
    for (DeviceId i = 0; i < rankSize; i++)
        rankToDevice[i] = i;
    HcclResult result = TaskCheckAllGatherVSemantics(allRankMemSemantics, vDataDes, rankToDevice);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}
} // namespace HcclSim
