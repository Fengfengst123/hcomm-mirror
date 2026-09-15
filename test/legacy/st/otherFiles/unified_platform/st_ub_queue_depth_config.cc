/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define private public
#define protected public
#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include "dev_ub_connection.h"
#include "host_ub_connection.h"
#include "rma_conn_manager.h"
#include "socket.h"
#include "orion_adapter_rts.h"
#include "not_support_exception.h"
#include "rma_conn_exception.h"
#include "orion_adpt_utils.h"
#include "rdma_handle_manager.h"
#undef private

using namespace Hccl;
using Hccl::JfcHandle; // 消除 JfcHandle 歧义：Hccl::JfcHandle=u64, hcomm::JfcHandle=void*

static constexpr u32 TEST_SQ_DEPTH = 256U;
static constexpr u32 TEST_SCQ_DEPTH_HOST = 1024U;
static constexpr u32 TEST_SCQ_DEPTH_AICPU = 512U;
static constexpr u32 EXPECTED_DEFAULT_SQ_DEPTH = 8192U;
static constexpr u32 EXPECTED_OFFLOAD_SQ_DEPTH = 128U;

class UbQueueDepthConfigTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "UbQueueDepthConfigTest set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "UbQueueDepthConfigTest tear down." << std::endl; }

    virtual void SetUp()
    {
        rdmaHandle = reinterpret_cast<RdmaHandle>(0x1000000);
        localIp = IpAddress();
        remoteIp = IpAddress();

        MOCKER_CPP(&RdmaHandleManager::GetDieAndFuncId)
            .defaults()
            .will(returnValue(std::make_pair(static_cast<uint32_t>(0), static_cast<uint32_t>(0))));
        MOCKER_CPP(&RdmaHandleManager::GetJfcHandle).defaults().will(returnValue(static_cast<JfcHandle>(0x5000)));
        MOCKER_CPP(&RdmaHandleManager::IsHandleValid).defaults().will(returnValue(true));
    }

    virtual void TearDown() { GlobalMockObject::verify(); }

    RdmaHandle rdmaHandle;
    IpAddress localIp;
    IpAddress remoteIp;
};

// ============================================================================
// 场景1：单边通信 host、aicpu 的 ub 场景
// 验证指定 scqDepth 时创建独占 JFC，不指定时使用共享 JFC
// ============================================================================

/**
 * 场景：Host UB TP 连接，指定 scqDepth，应创建独占 JFC
 * 预期：HrtRaUbCreateJfc 被调用且 cqDepth 参数等于指定值
 */
TEST_F(UbQueueDepthConfigTest, HostUbTp_WithScqDepth_CreatesExclusiveJfc)
{
    MOCKER(HrtRaUbCreateJfc)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), eq(TEST_SCQ_DEPTH_HOST))
        .will(returnValue(static_cast<JfcHandle>(0x2000)));

    HostUbTpConnection conn(
        rdmaHandle, localIp, remoteIp, OpMode::OPBASE, HrtUbJfcMode::NORMAL, static_cast<u8>(UB_QOS_DEFAULT),
        TEST_SQ_DEPTH, TEST_SCQ_DEPTH_HOST);

    EXPECT_TRUE(conn.isExclusiveJfc);
    EXPECT_EQ(conn.scqDepth, TEST_SCQ_DEPTH_HOST);
    EXPECT_EQ(conn.jfcHandle, static_cast<JfcHandle>(0x2000));

    GlobalMockObject::verify();
}

/**
 * 场景：Host UB CTP 连接，指定 scqDepth，应创建独占 JFC
 */
TEST_F(UbQueueDepthConfigTest, HostUbCtp_WithScqDepth_CreatesExclusiveJfc)
{
    MOCKER(HrtRaUbCreateJfc)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), eq(TEST_SCQ_DEPTH_HOST))
        .will(returnValue(static_cast<JfcHandle>(0x2001)));

    HostUbCtpConnection conn(
        rdmaHandle, localIp, remoteIp, OpMode::OPBASE, HrtUbJfcMode::NORMAL, static_cast<u8>(UB_QOS_DEFAULT),
        TEST_SQ_DEPTH, TEST_SCQ_DEPTH_HOST);

    EXPECT_TRUE(conn.isExclusiveJfc);
    EXPECT_EQ(conn.scqDepth, TEST_SCQ_DEPTH_HOST);

    GlobalMockObject::verify();
}

/**
 * 场景：AICPU UB TP 连接，指定 scqDepth，应创建独占 JFC（STARS_POLL 模式）
 */
TEST_F(UbQueueDepthConfigTest, HostUbTp_WithoutScqDepth_UsesSharedJfc)
{
    HostUbTpConnection conn(
        rdmaHandle, localIp, remoteIp, OpMode::OPBASE, HrtUbJfcMode::NORMAL, static_cast<u8>(UB_QOS_DEFAULT),
        UB_SQ_DEPTH_NOT_SET, UB_SQ_DEPTH_NOT_SET);

    EXPECT_FALSE(conn.isExclusiveJfc);
    EXPECT_EQ(conn.jfcHandle, static_cast<JfcHandle>(0x5000));

    GlobalMockObject::verify();
}

/**
 * 场景：AICPU UB TP 连接，指定 scqDepth，应创建独占 JFC（STARS_POLL 模式）
 */
TEST_F(UbQueueDepthConfigTest, AicpuUbTp_WithScqDepth_CreatesExclusiveJfc)
{
    MOCKER(HrtRaUbCreateJfc)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), eq(HrtUbJfcMode::STARS_POLL), eq(TEST_SCQ_DEPTH_AICPU))
        .will(returnValue(static_cast<JfcHandle>(0x2002)));

    DevUbTpConnection conn(
        rdmaHandle, localIp, remoteIp, OpMode::OPBASE, true, HrtUbJfcMode::STARS_POLL, localIp, remoteIp,
        static_cast<u8>(UB_QOS_DEFAULT), static_cast<u8>(TpManager::TA_TIMEOUT_NOT_SET), COMM_ENGINE_AICPU_TS,
        TEST_SQ_DEPTH, TEST_SCQ_DEPTH_AICPU);

    EXPECT_TRUE(conn.isExclusiveJfc);
    EXPECT_EQ(conn.scqDepth, TEST_SCQ_DEPTH_AICPU);

    GlobalMockObject::verify();
}

/**
 * 场景：AICPU UB CTP 连接，指定 scqDepth，应创建独占 JFC
 */
TEST_F(UbQueueDepthConfigTest, AicpuUbCtp_WithScqDepth_CreatesExclusiveJfc)
{
    MOCKER(HrtRaUbCreateJfc)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), eq(HrtUbJfcMode::STARS_POLL), eq(TEST_SCQ_DEPTH_AICPU))
        .will(returnValue(static_cast<JfcHandle>(0x2003)));

    DevUbCtpConnection conn(
        rdmaHandle, localIp, remoteIp, OpMode::OPBASE, true, HrtUbJfcMode::STARS_POLL, localIp, remoteIp,
        static_cast<u8>(UB_QOS_DEFAULT), static_cast<u8>(TpManager::TA_TIMEOUT_NOT_SET), COMM_ENGINE_AICPU_TS,
        TEST_SQ_DEPTH, TEST_SCQ_DEPTH_AICPU);

    EXPECT_TRUE(conn.isExclusiveJfc);
    EXPECT_EQ(conn.scqDepth, TEST_SCQ_DEPTH_AICPU);

    GlobalMockObject::verify();
}

/**
 * 场景：AICPU UB UBOE 连接，指定 scqDepth，应创建独占 JFC
 */
TEST_F(UbQueueDepthConfigTest, AicpuUbUboe_WithScqDepth_CreatesExclusiveJfc)
{
    MOCKER(HrtRaUbCreateJfc)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), eq(HrtUbJfcMode::STARS_POLL), eq(TEST_SCQ_DEPTH_AICPU))
        .will(returnValue(static_cast<JfcHandle>(0x2004)));

    DevUbUboeConnection conn(
        rdmaHandle, localIp, remoteIp, OpMode::OPBASE, true, HrtUbJfcMode::STARS_POLL, localIp, remoteIp,
        static_cast<u8>(UB_QOS_DEFAULT), static_cast<u8>(TpManager::TA_TIMEOUT_NOT_SET), COMM_ENGINE_AICPU_TS,
        TEST_SQ_DEPTH, TEST_SCQ_DEPTH_AICPU);

    EXPECT_TRUE(conn.isExclusiveJfc);
    EXPECT_EQ(conn.scqDepth, TEST_SCQ_DEPTH_AICPU);

    GlobalMockObject::verify();
}

/**
 * 场景：AICPU UB RTP 连接，指定 scqDepth，应创建独占 JFC
 */
TEST_F(UbQueueDepthConfigTest, AicpuUbRtp_WithScqDepth_CreatesExclusiveJfc)
{
    MOCKER(HrtRaUbCreateJfc)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), eq(HrtUbJfcMode::STARS_POLL), eq(TEST_SCQ_DEPTH_AICPU))
        .will(returnValue(static_cast<JfcHandle>(0x2005)));

    DevUbRtpConnection conn(
        rdmaHandle, localIp, remoteIp, OpMode::OPBASE, true, HrtUbJfcMode::STARS_POLL, localIp, remoteIp,
        static_cast<u8>(UB_QOS_DEFAULT), static_cast<u8>(TpManager::TA_TIMEOUT_NOT_SET), COMM_ENGINE_AICPU_TS,
        TEST_SQ_DEPTH, TEST_SCQ_DEPTH_AICPU);

    EXPECT_TRUE(conn.isExclusiveJfc);
    EXPECT_EQ(conn.scqDepth, TEST_SCQ_DEPTH_AICPU);

    GlobalMockObject::verify();
}

/**
 * 场景：独占 JFC 在连接释放时被销毁
 * 预期：HrtRaUbDestroyJfc 被调用一次
 */
TEST_F(UbQueueDepthConfigTest, ExclusiveJfc_ReleasedOnConnectionDestroy)
{
    MOCKER(HrtRaUbCreateJfc)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(static_cast<JfcHandle>(0x2006)));

    MOCKER(HrtRaUbDestroyJfc).expects(1).with(mockcpp::any(), mockcpp::any()).will(ignoreReturnValue());

    {
        HostUbTpConnection conn(
            rdmaHandle, localIp, remoteIp, OpMode::OPBASE, HrtUbJfcMode::NORMAL, static_cast<u8>(UB_QOS_DEFAULT),
            TEST_SQ_DEPTH, TEST_SCQ_DEPTH_HOST);
        EXPECT_TRUE(conn.isExclusiveJfc);
    }

    GlobalMockObject::verify();
}

/**
 * 场景：DevUb TP 连接析构时释放独占 JFC
 * 预期：HrtRaUbDestroyJfc 被调用一次
 */
TEST_F(UbQueueDepthConfigTest, DevUbTp_ExclusiveJfc_ReleasedOnDestroy)
{
    MOCKER(HrtRaUbCreateJfc)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(static_cast<JfcHandle>(0x200A)));

    MOCKER(HrtRaUbDestroyJfc).expects(1).with(mockcpp::any(), mockcpp::any()).will(ignoreReturnValue());

    {
        DevUbTpConnection conn(
            rdmaHandle, localIp, remoteIp, OpMode::OPBASE, true, HrtUbJfcMode::STARS_POLL, localIp, remoteIp,
            static_cast<u8>(UB_QOS_DEFAULT), static_cast<u8>(TpManager::TA_TIMEOUT_NOT_SET), COMM_ENGINE_AICPU_TS,
            TEST_SQ_DEPTH, TEST_SCQ_DEPTH_AICPU);
        EXPECT_TRUE(conn.isExclusiveJfc);
        EXPECT_EQ(conn.scqDepth, TEST_SCQ_DEPTH_AICPU);
    }

    GlobalMockObject::verify();
}

/**
 * 场景：共享 JFC（不指定 scqDepth）在连接析构时不释放
 * 预期：HrtRaUbDestroyJfc 不被调用
 */
TEST_F(UbQueueDepthConfigTest, SharedJfc_NotReleasedOnDestroy)
{
    MOCKER(HrtRaUbDestroyJfc).expects(0).with(mockcpp::any(), mockcpp::any()).will(ignoreReturnValue());

    {
        HostUbTpConnection conn(
            rdmaHandle, localIp, remoteIp, OpMode::OPBASE, HrtUbJfcMode::NORMAL, static_cast<u8>(UB_QOS_DEFAULT),
            UB_SQ_DEPTH_NOT_SET, UB_SQ_DEPTH_NOT_SET);
        EXPECT_FALSE(conn.isExclusiveJfc);
        EXPECT_EQ(conn.jfcHandle, static_cast<JfcHandle>(0x5000));
    }

    GlobalMockObject::verify();
}

// ============================================================================
// 场景2：基础通信层 API 级验证（需覆盖规格测试）
// 验证 CheckUbScqDepth 的规格边界值
// 规格：host（normal）64 ~ 32768，aicpu（stars poll）64 ~ 16384
// ============================================================================

/**
 * 场景：NORMAL 模式，scqDepth 取最小值 64
 * 预期：返回 HCCL_SUCCESS
 */
TEST_F(UbQueueDepthConfigTest, Spec_NormalMode_MinValue_ReturnsSuccess)
{
    hcomm::UbConnBuildContext ctx{};
    ctx.scqDepth = hcomm::UB_SCQ_DEPTH_MIN;

    EXPECT_EQ(hcomm::CheckUbScqDepth(ctx, hcomm::HrtUbJfcMode::NORMAL), HCCL_SUCCESS);
}

/**
 * 场景：NORMAL 模式，scqDepth 取最大值 32768
 * 预期：返回 HCCL_SUCCESS
 */
TEST_F(UbQueueDepthConfigTest, Spec_NormalMode_MaxValue_ReturnsSuccess)
{
    hcomm::UbConnBuildContext ctx{};
    ctx.scqDepth = hcomm::UB_SCQ_DEPTH_MAX_NORMAL;

    EXPECT_EQ(hcomm::CheckUbScqDepth(ctx, hcomm::HrtUbJfcMode::NORMAL), HCCL_SUCCESS);
}

/**
 * 场景：NORMAL 模式，scqDepth 低于最小值 63
 * 预期：返回 HCCL_E_PARA
 */
TEST_F(UbQueueDepthConfigTest, Spec_NormalMode_BelowMin_ReturnsError)
{
    hcomm::UbConnBuildContext ctx{};
    ctx.scqDepth = hcomm::UB_SCQ_DEPTH_MIN - 1;

    EXPECT_EQ(hcomm::CheckUbScqDepth(ctx, hcomm::HrtUbJfcMode::NORMAL), HCCL_E_PARA);
}

/**
 * 场景：NORMAL 模式，scqDepth 超出最大值 32769
 * 预期：返回 HCCL_E_PARA
 */
TEST_F(UbQueueDepthConfigTest, Spec_NormalMode_ExceedMax_ReturnsError)
{
    hcomm::UbConnBuildContext ctx{};
    ctx.scqDepth = hcomm::UB_SCQ_DEPTH_MAX_NORMAL + 1;

    EXPECT_EQ(hcomm::CheckUbScqDepth(ctx, hcomm::HrtUbJfcMode::NORMAL), HCCL_E_PARA);
}

/**
 * 场景：STARS_POLL 模式，scqDepth 取最小值 64
 * 预期：返回 HCCL_SUCCESS
 */
TEST_F(UbQueueDepthConfigTest, Spec_StarsPollMode_MinValue_ReturnsSuccess)
{
    hcomm::UbConnBuildContext ctx{};
    ctx.scqDepth = hcomm::UB_SCQ_DEPTH_MIN;

    EXPECT_EQ(hcomm::CheckUbScqDepth(ctx, hcomm::HrtUbJfcMode::STARS_POLL), HCCL_SUCCESS);
}

/**
 * 场景：STARS_POLL 模式，scqDepth 取最大值 16384
 * 预期：返回 HCCL_SUCCESS
 */
TEST_F(UbQueueDepthConfigTest, Spec_StarsPollMode_MaxValue_ReturnsSuccess)
{
    hcomm::UbConnBuildContext ctx{};
    ctx.scqDepth = hcomm::UB_SCQ_DEPTH_MAX_STARS_POLL;

    EXPECT_EQ(hcomm::CheckUbScqDepth(ctx, hcomm::HrtUbJfcMode::STARS_POLL), HCCL_SUCCESS);
}

/**
 * 场景：STARS_POLL 模式，scqDepth 超出最大值 16385
 * 预期：返回 HCCL_E_PARA
 */
TEST_F(UbQueueDepthConfigTest, Spec_StarsPollMode_ExceedMax_ReturnsError)
{
    hcomm::UbConnBuildContext ctx{};
    ctx.scqDepth = hcomm::UB_SCQ_DEPTH_MAX_STARS_POLL + 1;

    EXPECT_EQ(hcomm::CheckUbScqDepth(ctx, hcomm::HrtUbJfcMode::STARS_POLL), HCCL_E_PARA);
}

/**
 * 场景：USER_CTL 模式（AIV），scqDepth 取最小值 64
 * 预期：返回 HCCL_SUCCESS
 */
TEST_F(UbQueueDepthConfigTest, Spec_UserCtlMode_MinValue_ReturnsSuccess)
{
    hcomm::UbConnBuildContext ctx{};
    ctx.scqDepth = hcomm::UB_SCQ_DEPTH_MIN;

    EXPECT_EQ(hcomm::CheckUbScqDepth(ctx, hcomm::HrtUbJfcMode::USER_CTL), HCCL_SUCCESS);
}

/**
 * 场景：USER_CTL 模式（AIV），scqDepth 取最大值 32768
 * 预期：返回 HCCL_SUCCESS
 */
TEST_F(UbQueueDepthConfigTest, Spec_UserCtlMode_MaxValue_ReturnsSuccess)
{
    hcomm::UbConnBuildContext ctx{};
    ctx.scqDepth = hcomm::UB_SCQ_DEPTH_MAX_NORMAL;

    EXPECT_EQ(hcomm::CheckUbScqDepth(ctx, hcomm::HrtUbJfcMode::USER_CTL), HCCL_SUCCESS);
}

/**
 * 场景：USER_CTL 模式（AIV），scqDepth 超出最大值 32769
 * 预期：返回 HCCL_E_PARA
 */
TEST_F(UbQueueDepthConfigTest, Spec_UserCtlMode_ExceedMax_ReturnsError)
{
    hcomm::UbConnBuildContext ctx{};
    ctx.scqDepth = hcomm::UB_SCQ_DEPTH_MAX_NORMAL + 1;

    EXPECT_EQ(hcomm::CheckUbScqDepth(ctx, hcomm::HrtUbJfcMode::USER_CTL), HCCL_E_PARA);
}

/**
 * 场景：scqDepth 为默认值 0，应跳过校验返回 SUCCESS
 */
TEST_F(UbQueueDepthConfigTest, Spec_DefaultZero_ReturnsSuccess)
{
    hcomm::UbConnBuildContext ctx{};
    ctx.scqDepth = 0;

    EXPECT_EQ(hcomm::CheckUbScqDepth(ctx, hcomm::HrtUbJfcMode::NORMAL), HCCL_SUCCESS);
}

/**
 * 场景：scqDepth 为哨兵值 0xFFFFFFFF，应跳过校验返回 SUCCESS
 */
TEST_F(UbQueueDepthConfigTest, Spec_SentinelValue_ReturnsSuccess)
{
    hcomm::UbConnBuildContext ctx{};
    ctx.scqDepth = 0xFFFFFFFFU;

    EXPECT_EQ(hcomm::CheckUbScqDepth(ctx, hcomm::HrtUbJfcMode::NORMAL), HCCL_SUCCESS);
}

// ============================================================================
// 场景3：集合通信 host、aicpu ub 基础功能验证
// 以下用例验证 scqDepth 配置后连接的基本通信能力
// 完整功能验证需在 A5 板上执行（需硬件环境）
// ============================================================================

/**
 * 场景：Host UB TP 连接创建后，SQ 深度按指定值生效
 * 预期：conn.sqDepth 等于传入的 TEST_SQ_DEPTH
 */
TEST_F(UbQueueDepthConfigTest, HostUbTp_SqDepth_AppliedCorrectly)
{
    MOCKER(HrtRaUbCreateJfc)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(static_cast<JfcHandle>(0x2007)));

    HostUbTpConnection conn(
        rdmaHandle, localIp, remoteIp, OpMode::OPBASE, HrtUbJfcMode::NORMAL, static_cast<u8>(UB_QOS_DEFAULT),
        TEST_SQ_DEPTH, TEST_SCQ_DEPTH_HOST);

    EXPECT_EQ(conn.sqDepth, TEST_SQ_DEPTH);

    GlobalMockObject::verify();
}

/**
 * 场景：Host UB TP 连接创建后，SCQ 深度按指定值生效
 * 预期：conn.scqDepth 等于传入的 TEST_SCQ_DEPTH_HOST
 */
TEST_F(UbQueueDepthConfigTest, HostUbTp_ScqDepth_AppliedCorrectly)
{
    MOCKER(HrtRaUbCreateJfc)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(static_cast<JfcHandle>(0x2008)));

    HostUbTpConnection conn(
        rdmaHandle, localIp, remoteIp, OpMode::OPBASE, HrtUbJfcMode::NORMAL, static_cast<u8>(UB_QOS_DEFAULT),
        TEST_SQ_DEPTH, TEST_SCQ_DEPTH_HOST);

    EXPECT_EQ(conn.scqDepth, TEST_SCQ_DEPTH_HOST);

    GlobalMockObject::verify();
}

/**
 * 场景：Host UB TP 连接不指定 sqDepth 时，使用默认深度
 * 预期：conn.sqDepth 等于 8192（OpMode::OPBASE 默认值）
 */
TEST_F(UbQueueDepthConfigTest, HostUbTp_DefaultSqDepth_UsesDefaultValue)
{
    HostUbTpConnection conn(
        rdmaHandle, localIp, remoteIp, OpMode::OPBASE, HrtUbJfcMode::NORMAL, static_cast<u8>(UB_QOS_DEFAULT),
        UB_SQ_DEPTH_NOT_SET, UB_SQ_DEPTH_NOT_SET);

    EXPECT_EQ(conn.sqDepth, EXPECTED_DEFAULT_SQ_DEPTH);

    GlobalMockObject::verify();
}

/**
 * 场景：Host UB TP 连接配置 sqDepth=0 时（未显式初始化结构体的常见值），
 * 按头文件承诺"0 表示默认值"，使用默认深度而非建成 0 深度 SQ
 * 预期：conn.sqDepth 等于 8192（OpMode::OPBASE 默认值）
 */
TEST_F(UbQueueDepthConfigTest, HostUbTp_SqDepthZero_UsesDefaultValue)
{
    HostUbTpConnection conn(
        rdmaHandle, localIp, remoteIp, OpMode::OPBASE, HrtUbJfcMode::NORMAL, static_cast<u8>(UB_QOS_DEFAULT), 0,
        UB_SQ_DEPTH_NOT_SET);

    EXPECT_EQ(conn.sqDepth, EXPECTED_DEFAULT_SQ_DEPTH);

    GlobalMockObject::verify();
}

/**
 * 场景：AICPU DevUb TP 连接创建后，SQ 和 SCQ 深度按指定值生效
 * 预期：conn.sqDepth 和 conn.scqDepth 等于传入值
 */
TEST_F(UbQueueDepthConfigTest, AicpuUbTp_SqScqDepth_AppliedCorrectly)
{
    MOCKER(HrtRaUbCreateJfc)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(static_cast<JfcHandle>(0x2009)));

    DevUbTpConnection conn(
        rdmaHandle, localIp, remoteIp, OpMode::OPBASE, true, HrtUbJfcMode::STARS_POLL, localIp, remoteIp,
        static_cast<u8>(UB_QOS_DEFAULT), static_cast<u8>(TpManager::TA_TIMEOUT_NOT_SET), COMM_ENGINE_AICPU_TS,
        TEST_SQ_DEPTH, TEST_SCQ_DEPTH_AICPU);

    EXPECT_EQ(conn.sqDepth, TEST_SQ_DEPTH);
    EXPECT_EQ(conn.scqDepth, TEST_SCQ_DEPTH_AICPU);
    EXPECT_TRUE(conn.isExclusiveJfc);

    GlobalMockObject::verify();
}

/**
 * 场景：Host UB TP 连接 OFFLOAD 模式，不指定 sqDepth 时使用 OFFLOAD 默认深度
 * 预期：conn.sqDepth 等于 128（OFFLOAD 默认值）
 */
TEST_F(UbQueueDepthConfigTest, HostUbTp_OffloadMode_DefaultSqDepth_UsesOffloadValue)
{
    HostUbTpConnection conn(
        rdmaHandle, localIp, remoteIp, OpMode::OFFLOAD, HrtUbJfcMode::NORMAL, static_cast<u8>(UB_QOS_DEFAULT),
        UB_SQ_DEPTH_NOT_SET, UB_SQ_DEPTH_NOT_SET);

    EXPECT_EQ(conn.sqDepth, EXPECTED_OFFLOAD_SQ_DEPTH);

    GlobalMockObject::verify();
}
