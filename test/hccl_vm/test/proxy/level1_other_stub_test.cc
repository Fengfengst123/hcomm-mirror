/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * Description: level1 HcclAicpuKernelLaunch 桩单测（AI_CPU 展开模式 P2P 下发路径）。
 * 下游北向桩（HcclThreadAcquireWithStream / HcommThreadNotifyRecordOnThread）以
 * 测试替身隔离 DB 环境；HcclThreadExportToCommEngine 直接用被测文件内的真实现。
 * Create: 2026-11-11
 */

#include <gtest/gtest.h>

#include "hccl/hccl_launch.h"

// 被测函数（定义在 src/proxy/level1/level1_other_stub.cc）
extern "C" HcclResult HcclAicpuKernelLaunch(
    HcclComm comm, const HcclOpDesc* opInfo, const HcclKernelFuncInfo* funcInfo, ThreadHandle aicpuThreadHandle,
    aclrtStream userStream, const HcclKernelLaunchCfg* kernelLaunchCfg);

namespace {
// 测试替身行为控制
HcclResult g_fakeAcquireRet = HCCL_SUCCESS;
int32_t g_fakeRecordRet = 0;
int g_acquireCalls = 0;
int g_recordCalls = 0;
ThreadHandle g_fakeCpuTsThread = 100;

/**
 * @brief 重置测试替身状态，每个用例前调用，保证用例间互不污染。
 */
void ResetFakes()
{
    g_fakeAcquireRet = HCCL_SUCCESS;
    g_fakeRecordRet = 0;
    g_acquireCalls = 0;
    g_recordCalls = 0;
    g_fakeCpuTsThread = 100;
}
} // namespace

// 下游北向桩的测试替身（隔离真实 DB 建模，仅验证本桩的调用序列与错误传播）
extern "C" HcclResult HcclThreadAcquireWithStream(HcclComm, CommEngine, aclrtStream, uint32_t, ThreadHandle* thread)
{
    g_acquireCalls++;
    if (g_fakeAcquireRet == HCCL_SUCCESS) {
        *thread = g_fakeCpuTsThread;
    }
    return g_fakeAcquireRet;
}

/**
 * @brief HcommThreadNotifyRecordOnThread 测试替身：计数并返回预设结果。
 *
 * @return int32_t 预设的 g_fakeRecordRet。
 */
extern "C" int32_t HcommThreadNotifyRecordOnThread(ThreadHandle, ThreadHandle, uint32_t)
{
    g_recordCalls++;
    return g_fakeRecordRet;
}

class HcclAicpuKernelLaunchStubTest : public ::testing::Test {
protected:
    /**
     * @brief 每个用例前重置测试替身，避免状态串扰。
     */
    void SetUp() override { ResetFakes(); }
};

/**
 * @brief comm 为空指针时应返回 HCCL_E_PTR，且不触碰任何下游桩。
 */
TEST_F(HcclAicpuKernelLaunchStubTest, NullCommReturnsPtrError)
{
    HcclOpDesc opInfo{};
    HcclKernelFuncInfo funcInfo{};
    HcclKernelLaunchCfg cfg{};
    EXPECT_EQ(
        HcclAicpuKernelLaunch(nullptr, &opInfo, &funcInfo, 1, reinterpret_cast<aclrtStream>(2), &cfg), HCCL_E_PTR);
    EXPECT_EQ(g_acquireCalls, 0);
}

/**
 * @brief opInfo 为空指针时应返回 HCCL_E_PTR。
 */
TEST_F(HcclAicpuKernelLaunchStubTest, NullOpInfoReturnsPtrError)
{
    HcclKernelFuncInfo funcInfo{};
    HcclKernelLaunchCfg cfg{};
    EXPECT_EQ(
        HcclAicpuKernelLaunch(
            reinterpret_cast<HcclComm>(8), nullptr, &funcInfo, 1, reinterpret_cast<aclrtStream>(2), &cfg),
        HCCL_E_PTR);
}

/**
 * @brief funcInfo 为空指针时应返回 HCCL_E_PTR。
 */
TEST_F(HcclAicpuKernelLaunchStubTest, NullFuncInfoReturnsPtrError)
{
    HcclOpDesc opInfo{};
    HcclKernelLaunchCfg cfg{};
    EXPECT_EQ(
        HcclAicpuKernelLaunch(
            reinterpret_cast<HcclComm>(8), &opInfo, nullptr, 1, reinterpret_cast<aclrtStream>(2), &cfg),
        HCCL_E_PTR);
}

/**
 * @brief userStream 为空指针时应返回 HCCL_E_PTR。
 */
TEST_F(HcclAicpuKernelLaunchStubTest, NullStreamReturnsPtrError)
{
    HcclOpDesc opInfo{};
    HcclKernelFuncInfo funcInfo{};
    HcclKernelLaunchCfg cfg{};
    EXPECT_EQ(HcclAicpuKernelLaunch(reinterpret_cast<HcclComm>(8), &opInfo, &funcInfo, 1, nullptr, &cfg), HCCL_E_PTR);
}

/**
 * @brief kernelLaunchCfg 为空指针时应返回 HCCL_E_PTR。
 */
TEST_F(HcclAicpuKernelLaunchStubTest, NullLaunchCfgReturnsPtrError)
{
    HcclOpDesc opInfo{};
    HcclKernelFuncInfo funcInfo{};
    EXPECT_EQ(
        HcclAicpuKernelLaunch(
            reinterpret_cast<HcclComm>(8), &opInfo, &funcInfo, 1, reinterpret_cast<aclrtStream>(2), nullptr),
        HCCL_E_PTR);
}

/**
 * @brief funcInfo->args 为空但 argSize>0 时应返回 HCCL_E_PTR（与 hcomm 真实现同口径）。
 */
TEST_F(HcclAicpuKernelLaunchStubTest, NullArgsWithPositiveArgSizeReturnsPtrError)
{
    HcclOpDesc opInfo{};
    HcclKernelFuncInfo funcInfo{};
    HcclKernelLaunchCfg cfg{};
    funcInfo.argSize = 16;
    funcInfo.args = nullptr;
    EXPECT_EQ(
        HcclAicpuKernelLaunch(
            reinterpret_cast<HcclComm>(8), &opInfo, &funcInfo, 1, reinterpret_cast<aclrtStream>(2), &cfg),
        HCCL_E_PTR);
}

/**
 * @brief 线程获取失败时错误码应原样向上传播，且不执行 notify record。
 */
TEST_F(HcclAicpuKernelLaunchStubTest, AcquireFailurePropagates)
{
    HcclOpDesc opInfo{};
    HcclKernelFuncInfo funcInfo{};
    HcclKernelLaunchCfg cfg{};
    g_fakeAcquireRet = HCCL_E_INTERNAL;
    EXPECT_EQ(
        HcclAicpuKernelLaunch(
            reinterpret_cast<HcclComm>(8), &opInfo, &funcInfo, 1, reinterpret_cast<aclrtStream>(2), &cfg),
        HCCL_E_INTERNAL);
    EXPECT_EQ(g_acquireCalls, 1);
    EXPECT_EQ(g_recordCalls, 0);
}

/**
 * @brief argSize 超过 P2P_MAX_ARG_SIZE 时应返回 HCCL_E_PARA（对齐 hcomm 入口校验口径），
 *        且不触碰任何下游桩。
 */
TEST_F(HcclAicpuKernelLaunchStubTest, OverLimitArgSizeReturnsParaError)
{
    HcclOpDesc opInfo{};
    HcclKernelFuncInfo funcInfo{};
    HcclKernelLaunchCfg cfg{};
    uint8_t args[P2P_MAX_ARG_SIZE + 1] = {0};
    funcInfo.argSize = P2P_MAX_ARG_SIZE + 1;
    funcInfo.args = args;
    EXPECT_EQ(
        HcclAicpuKernelLaunch(
            reinterpret_cast<HcclComm>(8), &opInfo, &funcInfo, 1, reinterpret_cast<aclrtStream>(2), &cfg),
        HCCL_E_PARA);
    EXPECT_EQ(g_acquireCalls, 0);
    EXPECT_EQ(g_recordCalls, 0);
}
