/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>
#include <memory>
#include <string>
#include <vector>
#define private public
#define protected public
#include "alltoall_operator.h"
#include "coll_alg_operator.h"
#include "topo_matcher.h"
#include "alg_configurator.h"
#include "ccl_buffer_manager.h"
#include "coll_alg_utils.h"
#include "workflow.h"
#undef private
#undef protected
#include "stream_utils.h"

using namespace hccl;
using namespace std;

namespace {
constexpr u32 RANK_SIZE_TWO = 2;
constexpr u32 RANK_SIZE_EIGHT = 8;
constexpr u64 TEST_CCL_BUFFER_SIZE = 1 * 1024 * 1024;
} // namespace

class AlltoAllOperatorTest : public testing::Test {
protected:
    void SetUp() override
    {
        HcclTopoAttr topoAttr;
        topoAttr.deviceType = DevType::DEV_TYPE_910B;
        topoAttr.userRank = 0;
        topoAttr.userRankSize = RANK_SIZE_TWO;
        topoAttr.serverNum = RANK_SIZE_TWO;
        topoAttr.superPodNum = RANK_SIZE_TWO;
        topoAttr.useSuperPodMode = false;
        // 未设置时默认0，会导致选路条件取模除零
        topoAttr.meshAggregationRankSize = 1;
        HcclAlgoAttr algoAttr;
        algConfigurator = std::make_unique<AlgConfigurator>(algoAttr, topoAttr);
        cclBufferManager.InitCCLbuffer(TEST_CCL_BUFFER_SIZE, TEST_CCL_BUFFER_SIZE);
        auto topoMatcher = std::make_unique<TopoMatcher>(
            commPlaneRanks, isBridgeVector, topoInfo, algoInfo, externalEnable, serverAndsuperPodToRank);
        op = std::make_unique<AlltoAllOperator>(algConfigurator.get(), cclBufferManager, nullptr, topoMatcher);
    }

    void TearDown() override
    {
        // 校验 mock 调用并清理 mock hook，避免影响同进程后续测试
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    }

    // mock 三个选路条件函数，聚焦验证 JudgeIfNeedPreProcessAndGetParam 的门控逻辑；
    // 不 mock 其内部调用的短函数（如 GetWorkflowMode），因其入口可能被前序套件的
    // mock hook 破坏，且选路条件函数本身为20+行大函数，hook 安全
    void MockRouteConditions()
    {
        MOCKER_CPP(&AlltoAllOperator::IsSatisfyAlltoAllAivCondition).stubs().will(returnValue(false));
        MOCKER(IsSupportDirectFullmeshForAlltoallv).stubs().will(returnValue(false));
        MOCKER_CPP(&AlltoAllOperator::IsSatisfyAlltoallContinuousPipelineCondition).stubs().will(returnValue(false));
    }

    void MakeAlltoallvParam(OpParam& param, u64* sendCounts, u64* sdispls, u64* recvCounts, u64* rdispls)
    {
        param.opType = HcclCMDType::HCCL_CMD_ALLTOALLV;
        param.All2AllDataDes.sendType = HcclDataType::HCCL_DATA_TYPE_INT8;
        param.All2AllDataDes.recvType = HcclDataType::HCCL_DATA_TYPE_INT8;
        param.All2AllDataDes.sendCounts = sendCounts;
        param.All2AllDataDes.sdispls = sdispls;
        param.All2AllDataDes.recvCounts = recvCounts;
        param.All2AllDataDes.rdispls = rdispls;
    }

    std::unique_ptr<AlgConfigurator> algConfigurator;
    CCLBufferManager cclBufferManager;
    std::unique_ptr<AlltoAllOperator> op;
    HcclTopoInfo topoInfo;
    HcclAlgoInfo algoInfo;
    HcclExternalEnable externalEnable;
    std::vector<std::vector<std::vector<u32>>> commPlaneRanks;
    std::vector<bool> isBridgeVector;
    std::vector<std::vector<std::vector<u32>>> serverAndsuperPodToRank;
};

/* 910B + 单算子模式 + unfold + 双机各1卡：不再跳过 PreProcess，经 ALLGATHER 收集全量 SendRecvInfo */
TEST_F(AlltoAllOperatorTest, Ut_JudgeIfNeedPreProcess_UnfoldRank2_Expect_PreProcess)
{
    SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    MockRouteConditions();
    OpParam param;
    param.aicpuUnfoldMode = true;
    u64 sendCounts[RANK_SIZE_TWO] = {1, 2};
    u64 sdispls[RANK_SIZE_TWO] = {0, 1};
    u64 recvCounts[RANK_SIZE_TWO] = {2, 1};
    u64 rdispls[RANK_SIZE_TWO] = {1, 0};
    MakeAlltoallvParam(param, sendCounts, sdispls, recvCounts, rdispls);

    std::unique_ptr<PreProcessMetaInfo> preMetaInfo = std::make_unique<PreProcessMetaInfo>();
    EXPECT_TRUE(op->JudgeIfNeedPreProcessAndGetParam(param, preMetaInfo));
    EXPECT_EQ(preMetaInfo->opType, HcclCMDType::HCCL_CMD_ALLGATHER);
}

/* 910B + 单算子模式 + unfold + rankSize > 2：维持跳过 PreProcess 的存量行为 */
TEST_F(AlltoAllOperatorTest, Ut_JudgeIfNeedPreProcess_UnfoldRank8_Expect_Skip)
{
    SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    MockRouteConditions();
    // 双机各4卡
    op->userRankSize_ = RANK_SIZE_EIGHT;
    op->serverNum_ = RANK_SIZE_TWO;
    OpParam param;
    param.aicpuUnfoldMode = true;
    u64 sendCounts[RANK_SIZE_EIGHT] = {1, 2, 3, 4, 5, 6, 7, 8};
    u64 sdispls[RANK_SIZE_EIGHT] = {0, 1, 2, 3, 4, 5, 6, 7};
    u64 recvCounts[RANK_SIZE_EIGHT] = {8, 7, 6, 5, 4, 3, 2, 1};
    u64 rdispls[RANK_SIZE_EIGHT] = {7, 6, 5, 4, 3, 2, 1, 0};
    MakeAlltoallvParam(param, sendCounts, sdispls, recvCounts, rdispls);

    std::unique_ptr<PreProcessMetaInfo> preMetaInfo = std::make_unique<PreProcessMetaInfo>();
    EXPECT_FALSE(op->JudgeIfNeedPreProcessAndGetParam(param, preMetaInfo));
}

/* 910B + 单算子模式 + 非 unfold + 双机各1卡：存量行为回归，仍执行 PreProcess */
TEST_F(AlltoAllOperatorTest, Ut_JudgeIfNeedPreProcess_OpbaseRank2_Expect_PreProcess)
{
    SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    MockRouteConditions();
    OpParam param;
    u64 sendCounts[RANK_SIZE_TWO] = {1, 2};
    u64 sdispls[RANK_SIZE_TWO] = {0, 1};
    u64 recvCounts[RANK_SIZE_TWO] = {2, 1};
    u64 rdispls[RANK_SIZE_TWO] = {1, 0};
    MakeAlltoallvParam(param, sendCounts, sdispls, recvCounts, rdispls);

    std::unique_ptr<PreProcessMetaInfo> preMetaInfo = std::make_unique<PreProcessMetaInfo>();
    EXPECT_TRUE(op->JudgeIfNeedPreProcessAndGetParam(param, preMetaInfo));
    EXPECT_EQ(preMetaInfo->opType, HcclCMDType::HCCL_CMD_ALLGATHER);
}
