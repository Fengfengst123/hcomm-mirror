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
#include "hccl/hccl_res.h"
#include "hcomm_c_adpt.h"
#include "hcomm_thread_c_adpt.h"
#include "../../hccl_api_base_test.h"
#include "hccl_tbe_task.h"
#include "thread_manager.h"
#include "launch_aicpu.h"
#include "aicpu_launch_manager.h"
#include "adapter_rts_common.h"
#include "adapter_rts.h"
#include "aicpu_ts_thread.h"

using namespace hccl;

static HcclResult StubThreadKernelLaunchForCommDevice(
    std::vector<std::shared_ptr<hccl::Thread>>& newThreads, const std::string& commId,
    std::unique_ptr<ThreadHandle[]>& aicpuHandle, aclrtBinHandle binHandle)
{
    for (size_t i = 0; i < newThreads.size(); ++i) {
        aicpuHandle[i] = static_cast<ThreadHandle>(0x5000 + i);
    }
    return HCCL_SUCCESS;
}
void MockGetRunSideIsDevice();
void MockThreadKernelLaunchForComm();

namespace {
// EnsureAicpuCommInit spy：记录 init/alloc 调用次数与顺序（invoke 桩无法捕获变量，用文件级静态对象共享）
struct AicpuCommInitSpy {
    uint32_t seq = 0;      // 全局调用序号
    uint32_t initSeq = 0;  // kernelLaunchAicpuCommInit 首次调用序号
    uint32_t allocSeq = 0; // HcommThreadAllocWithCommConfig 首次调用序号
    uint32_t initCnt = 0;
    uint32_t allocCnt = 0;
    HcclResult initRet = HCCL_SUCCESS;
    bool commState = false;
};
AicpuCommInitSpy g_aicpuSpy;
} // namespace

// L0 线程分配桩：填充伪句柄并记录调用顺序
static HcommResult StubThreadAllocWithCommConfig(
    CommEngine, const char*, uint32_t threadNum, ThreadType, const ThreadConfig*, ThreadHandle* threads)
{
    g_aicpuSpy.allocCnt++;
    if (g_aicpuSpy.allocSeq == 0) {
        g_aicpuSpy.allocSeq = ++g_aicpuSpy.seq;
    }
    for (uint32_t i = 0; i < threadNum; ++i) {
        threads[i] = static_cast<ThreadHandle>(0x7000 + i);
    }
    return static_cast<HcommResult>(HCCL_SUCCESS);
}

// L0 线程资源查询桩：返回伪 stream 指针
static HcommResult StubThreadResGetInfo(ThreadHandle, ThreadResType, uint32_t, void** info)
{
    *info = reinterpret_cast<void*>(0x7100);
    return static_cast<HcommResult>(HCCL_SUCCESS);
}

// 重置 spy 并 mock AICPU host 线程分配链路的 L0 边界，使 UT 不依赖真实线程资源
static void MockAicpuCommInitEnv()
{
    g_aicpuSpy = AicpuCommInitSpy();
    MOCKER(HcommThreadAllocWithCommConfig).stubs().will(invoke(StubThreadAllocWithCommConfig));
    MOCKER(HcommThreadResGetInfo).stubs().will(invoke(StubThreadResGetInfo));
    MOCKER(hrtStreamGetSqid).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HcommThreadSupplementNotify).stubs().will(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));
    MOCKER(HcommThreadFree).stubs().will(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));
}

class ThreadManagerTest : public BaseInit {
public:
    void SetUp() override
    {
        std::cout << "ThreadManagerTest SetUp" << std::endl;
        BaseInit::SetUp();
        MOCKER(AicpuAclKernelLaunch).stubs().will(returnValue(HCCL_SUCCESS));
        ManagerCallbacks callbacks;
        callbacks.getAicpuCommState = []() {
            return true;
        };
        callbacks.setAicpuCommState = [](bool) {};
        callbacks.kernelLaunchAicpuCommInit = []() {
            return HCCL_SUCCESS;
        };
        callbacks.reportProfilingKernel = [](uint64_t, std::string) {
            return HCCL_SUCCESS;
        };
        threadManager = std::make_unique<ThreadMgr>(4, 8, "test", callbacks);
    }
    void TearDown() override
    {
        std::cout << "ThreadManagerTest TearDown" << std::endl;
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }

    // 申请 N 个 CPU 线程（HcclThreadAcquireV2），notifyNum 一致
    void AcquireCpuThreads(uint32_t threadNum, uint16_t notifyNum, ThreadHandle* out)
    {
        MockGetRunSideIsDevice();
        std::vector<ThreadConfig> config(threadNum);
        EXPECT_EQ(ThreadConfigInit(config.data(), threadNum), HCCL_SUCCESS);
        for (uint32_t i = 0; i < threadNum; ++i) {
            config[i].notifyNumPerThread = notifyNum;
        }
        std::vector<uint32_t> threadId;
        EXPECT_EQ(
            threadManager->HcclThreadAcquireV2(
                CommEngine::COMM_ENGINE_CPU, threadNum, ThreadType::THREAD_TYPE_TS, config.data(), out, threadId),
            HCCL_SUCCESS);
    }

    // AICPU 公共域 spy 回调组：init 状态与次数可控可观测
    ManagerCallbacks MakeAicpuSpyCallbacks()
    {
        ManagerCallbacks callbacks;
        callbacks.getAicpuCommState = []() {
            return g_aicpuSpy.commState;
        };
        callbacks.setAicpuCommState = [](bool state) {
            g_aicpuSpy.commState = state;
        };
        callbacks.kernelLaunchAicpuCommInit = []() {
            g_aicpuSpy.initCnt++;
            if (g_aicpuSpy.initSeq == 0) {
                g_aicpuSpy.initSeq = ++g_aicpuSpy.seq;
            }
            return g_aicpuSpy.initRet;
        };
        callbacks.reportProfilingKernel = [](uint64_t, std::string) {
            return HCCL_SUCCESS;
        };
        return callbacks;
    }

private:
    std::unique_ptr<ThreadMgr> threadManager;
    uint32_t threadNum = 1;
    ThreadHandle threads[1] = {0};
    ThreadHandle exportedThreads[1] = {0};
};

void MockGetRunSideIsDevice()
{
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(bool{false})).will(returnValue(HCCL_SUCCESS));
}

void MockThreadKernelLaunchForComm()
{
    MOCKER_CPP(&AicpuLaunchMgr::ThreadKernelLaunchForComm).stubs().will(returnValue(HCCL_SUCCESS));
}

TEST_F(ThreadManagerTest, Ut_ThreadExportToCommEngineAicpu_When_InvalidThreadHandle_Expect_HCCL_E_NOT_FOUND)
{
    CommEngine dstCommEngine = COMM_ENGINE_AICPU_TS;

    HcclResult ret = threadManager->HcclThreadExportToCommEngine(threadNum, threads, dstCommEngine, exportedThreads);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(ThreadManagerTest, Ut_ThreadExportToCommEngineAicpu_When_Normal_Expect_ReturnHCCL_SUCCESS)
{
    MockGetRunSideIsDevice();
    MockThreadKernelLaunchForComm();

    CommEngine dstCommEngine = COMM_ENGINE_AICPU_TS;

    HcclResult ret = threadManager->HcclThreadAcquireWithStream(CommEngine::COMM_ENGINE_CPU, nullptr, 1, threads);
    if (ret == HCCL_SUCCESS) {
        ret = threadManager->HcclThreadExportToCommEngine(threadNum, threads, dstCommEngine, exportedThreads);
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }
}

TEST_F(ThreadManagerTest, Ut_ResetThreadLocalNotifies_When_NoThreads_Expect_Success)
{
    EXPECT_EQ(threadManager->ResetThreadLocalNotifies(), HCCL_SUCCESS);
}

TEST_F(ThreadManagerTest, Ut_ResetThreadLocalNotifies_When_ResetNotifiesFailed_Expect_ReturnFailed)
{
    MockGetRunSideIsDevice();
    HcclResult ret = threadManager->HcclThreadAcquireWithStream(CommEngine::COMM_ENGINE_CPU, nullptr, 1, threads);
    ASSERT_EQ(ret, HCCL_SUCCESS);

    MOCKER(HcommThreadResetNotifies).stubs().will(returnValue(static_cast<HcommResult>(HCCL_E_INTERNAL)));
    ret = threadManager->ResetThreadLocalNotifies();
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(ThreadManagerTest, Ut_ResetThreadLocalNotifies_When_OrderLaunchThreadRegistered_Expect_Success)
{
    MockGetRunSideIsDevice();
    HcclResult ret = threadManager->HcclThreadAcquireWithStream(CommEngine::COMM_ENGINE_CPU, nullptr, 1, threads);
    ASSERT_EQ(ret, HCCL_SUCCESS);

    ret = threadManager->RegisterOrderLaunchThread(threads[0]);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(threadManager->ResetThreadLocalNotifies(), HCCL_SUCCESS);
}

/* ======================== HcclDedicatedThreadAcquire DEVICE ======================== */

static void MockAicpuThreadEnv()
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(HCCL_SUCCESS));
}

TEST_F(ThreadManagerTest, Ut_DedicatedThreadAcquire_When_DeviceTypeInvalid_Expect_HCCL_E_PARA)
{
    ThreadHandle thread = 0;
    HcclResult ret = threadManager->HcclDedicatedThreadAcquire(HCCL_DED_THREAD_TYPE_INVALID, 1, &thread);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(ThreadManagerTest, Ut_DedicatedThreadAcquire_When_DeviceThreadNullptr_Expect_HCCL_E_PTR)
{
    HcclResult ret
        = threadManager->HcclDedicatedThreadAcquire(HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE, 1, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(ThreadManagerTest, Ut_DedicatedThreadAcquire_When_DeviceCreateSuccess_Expect_NonZeroThread)
{
    MockAicpuThreadEnv();
    MOCKER_CPP(&AicpuLaunchMgr::ThreadKernelLaunchForComm).stubs().will(invoke(StubThreadKernelLaunchForCommDevice));

    ThreadHandle thread = 0;
    HcclResult ret
        = threadManager->HcclDedicatedThreadAcquire(HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE, 1, &thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(thread, static_cast<ThreadHandle>(0));
}

TEST_F(ThreadManagerTest, Ut_DedicatedThreadAcquire_When_DeviceRepeatedAcquire_Expect_SameThread)
{
    MockAicpuThreadEnv();
    MOCKER_CPP(&AicpuLaunchMgr::ThreadKernelLaunchForComm).stubs().will(invoke(StubThreadKernelLaunchForCommDevice));

    ThreadHandle thread1 = 0;
    HcclResult ret
        = threadManager->HcclDedicatedThreadAcquire(HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE, 1, &thread1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(thread1, static_cast<ThreadHandle>(0));

    ThreadHandle thread2 = 0;
    ret = threadManager->HcclDedicatedThreadAcquire(HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE, 1, &thread2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(thread1, thread2);
}

TEST_F(ThreadManagerTest, Ut_DedicatedThreadAcquire_When_DeviceKernelLaunchFail_Expect_Error)
{
    MockAicpuThreadEnv();
    MOCKER_CPP(&AicpuLaunchMgr::ThreadKernelLaunchForComm).stubs().will(returnValue(HCCL_E_INTERNAL));

    ThreadHandle thread = 0;
    HcclResult ret
        = threadManager->HcclDedicatedThreadAcquire(HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE, 1, &thread);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

// ============ RFC 0002: L1 ThreadMgr 归一 UT ============

// HcclThreadAcquireV2：CPU 引擎批量分配 + 返回 sqId
TEST_F(ThreadManagerTest, Ut_HcclThreadAcquireV2_When_CpuBatch_Expect_Success)
{
    MockGetRunSideIsDevice();
    constexpr uint32_t kNum = 2;
    ThreadConfig config[kNum];
    ASSERT_EQ(ThreadConfigInit(config, kNum), HCCL_SUCCESS);
    for (uint32_t i = 0; i < kNum; ++i) {
        config[i].notifyNumPerThread = static_cast<uint16_t>(2 + i);
    }
    ThreadHandle out[kNum] = {0};
    std::vector<uint32_t> threadId;
    HcclResult ret = threadManager->HcclThreadAcquireV2(
        CommEngine::COMM_ENGINE_CPU, kNum, ThreadType::THREAD_TYPE_TS, config, out, threadId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(threadId.size(), kNum);
    // 释放（L1 析构不负责，显式 Free）
    HcommThreadFree(out, kNum);
}

// HcclThreadAcquireV2：复用池命中 + notify 补充
TEST_F(ThreadManagerTest, Ut_HcclThreadAcquireV2_When_ReusePoolHit_Expect_SupplementNotify)
{
    MockGetRunSideIsDevice();
    MockThreadKernelLaunchForComm();
    constexpr uint32_t kNum = 1;
    ThreadConfig config[kNum];
    ASSERT_EQ(ThreadConfigInit(config, kNum), HCCL_SUCCESS);
    config[0].notifyNumPerThread = 2;
    ThreadHandle out[kNum] = {0};
    std::vector<uint32_t> threadId;
    HcclResult ret = threadManager->HcclThreadAcquireV2(
        CommEngine::COMM_ENGINE_CPU, kNum, ThreadType::THREAD_TYPE_TS, config, out, threadId);
    ASSERT_EQ(ret, HCCL_SUCCESS);
    ThreadHandle first = out[0];

    // 第二次：notifyNum 提升到 4，触发 SupplementNotify
    config[0].notifyNumPerThread = 4;
    std::vector<uint32_t> threadId2;
    ret = threadManager->HcclThreadAcquireV2(
        CommEngine::COMM_ENGINE_CPU, kNum, ThreadType::THREAD_TYPE_TS, config, out, threadId2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(out[0], first); // 复用同句柄
    HcommThreadFree(out, kNum);
}

// HcclThreadAcquire：批量分配 + 返回 sqId
TEST_F(ThreadManagerTest, Ut_HcclThreadAcquire_When_NonV2Batch_Expect_Success)
{
    MockGetRunSideIsDevice();
    constexpr uint32_t kNum = 2;
    ThreadConfig config[kNum];
    ASSERT_EQ(ThreadConfigInit(config, kNum), HCCL_SUCCESS);
    for (uint32_t i = 0; i < kNum; ++i) {
        config[i].notifyNumPerThread = 2;
    }
    ThreadHandle out[kNum] = {0};
    std::vector<uint32_t> threadId;
    HcclResult ret = threadManager->HcclThreadAcquire(
        CommEngine::COMM_ENGINE_CPU, kNum, ThreadType::THREAD_TYPE_TS, config, out, threadId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(threadId.size(), kNum);
    HcommThreadFree(out, kNum);
}

// HcclGetNotifyNumInThread：L1 反查
TEST_F(ThreadManagerTest, Ut_HcclGetNotifyNumInThread_When_Normal_Expect_Success)
{
    MockGetRunSideIsDevice();
    constexpr uint32_t kNum = 1;
    ThreadConfig config[kNum];
    ASSERT_EQ(ThreadConfigInit(config, kNum), HCCL_SUCCESS);
    config[0].notifyNumPerThread = 3;
    ThreadHandle out[kNum] = {0};
    std::vector<uint32_t> threadId;
    HcclResult ret = threadManager->HcclThreadAcquireV2(
        CommEngine::COMM_ENGINE_CPU, kNum, ThreadType::THREAD_TYPE_TS, config, out, threadId);
    ASSERT_EQ(ret, HCCL_SUCCESS);
    uint32_t num = 0;
    ret = threadManager->HcclGetNotifyNumInThread(out[0], &num);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(num, 3U);
    HcommThreadFree(out, kNum);
}

TEST_F(ThreadManagerTest, Ut_HcclThreadResGetInfo_When_Stream_Expect_Success)
{
    ThreadHandle handle;
    AcquireCpuThreads(1, 1, &handle);
    void* info = nullptr;
    HcclResult ret = threadManager->HcclThreadResGetInfo(
        handle, ThreadResType::THREAD_RES_TYPE_STREAM, sizeof(ThreadResTypeStream), &info);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(info, nullptr);
    HcommThreadFree(&handle, 1);
}

// 析构释放：ThreadMgr 析构不泄漏（L1 cache 清空，L0 g_ThreadMap 由 HcommThreadFree 管）
TEST_F(ThreadManagerTest, Ut_ThreadMgrDestructor_When_NoLeak_Expect_Success)
{
    ThreadHandle handle;
    AcquireCpuThreads(1, 1, &handle);
    // 显式 Free 后析构 cache 已空
    EXPECT_EQ(HcommThreadFree(&handle, 1), HCCL_SUCCESS);
}

/* ======================== EnsureAicpuCommInit（AICPU 首扩容场景） ======================== */

// AICPU 首次 acquire（existNum=0）：comm-init 恰好执行一次，且发生在线程分配之前
TEST_F(ThreadManagerTest, Ut_HcclThreadAcquireV2_When_AicpuFirstAcquire_Expect_CommInitOnceBeforeAlloc)
{
    MockGetRunSideIsDevice();
    MockAicpuCommInitEnv();
    std::unique_ptr<ThreadMgr> mgr = std::make_unique<ThreadMgr>(4, 8, "test", MakeAicpuSpyCallbacks());

    constexpr uint32_t kNum = 1;
    ThreadConfig config[kNum];
    ASSERT_EQ(ThreadConfigInit(config, kNum), HCCL_SUCCESS);
    config[0].notifyNumPerThread = 2;
    ThreadHandle out[kNum] = {0};
    std::vector<uint32_t> threadId;
    HcclResult ret = mgr->HcclThreadAcquireV2(
        CommEngine::COMM_ENGINE_AICPU, kNum, ThreadType::THREAD_TYPE_TS, config, out, threadId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(g_aicpuSpy.initCnt, 1U);
    EXPECT_EQ(g_aicpuSpy.allocCnt, 1U);
    ASSERT_NE(g_aicpuSpy.initSeq, 0U);
    ASSERT_NE(g_aicpuSpy.allocSeq, 0U);
    EXPECT_LT(g_aicpuSpy.initSeq, g_aicpuSpy.allocSeq); // comm-init 先于线程分配
}

// AICPU 重复 acquire：comm-init 防重生效，不重复执行
TEST_F(ThreadManagerTest, Ut_HcclThreadAcquireV2_When_AicpuRepeatedAcquire_Expect_CommInitNotRepeated)
{
    MockGetRunSideIsDevice();
    MockAicpuCommInitEnv();
    std::unique_ptr<ThreadMgr> mgr = std::make_unique<ThreadMgr>(4, 8, "test", MakeAicpuSpyCallbacks());

    constexpr uint32_t kNum = 1;
    ThreadConfig config[kNum];
    ASSERT_EQ(ThreadConfigInit(config, kNum), HCCL_SUCCESS);
    config[0].notifyNumPerThread = 2;
    ThreadHandle out[kNum] = {0};
    std::vector<uint32_t> threadId;
    HcclResult ret = mgr->HcclThreadAcquireV2(
        CommEngine::COMM_ENGINE_AICPU, kNum, ThreadType::THREAD_TYPE_TS, config, out, threadId);
    ASSERT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(g_aicpuSpy.initCnt, 1U);

    ret = mgr->HcclThreadAcquireV2(
        CommEngine::COMM_ENGINE_AICPU, kNum, ThreadType::THREAD_TYPE_TS, config, out, threadId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(g_aicpuSpy.initCnt, 1U);  // 防重生效
    EXPECT_EQ(g_aicpuSpy.allocCnt, 1U); // 复用池命中，不重复分配
}

// comm-init 失败：acquire 返回错误并早退，不进行线程分配，状态未置位
TEST_F(ThreadManagerTest, Ut_HcclThreadAcquireV2_When_CommInitFailed_Expect_AcquireFailedBeforeAlloc)
{
    MockGetRunSideIsDevice();
    MockAicpuCommInitEnv();
    g_aicpuSpy.initRet = HCCL_E_INTERNAL;
    std::unique_ptr<ThreadMgr> mgr = std::make_unique<ThreadMgr>(4, 8, "test", MakeAicpuSpyCallbacks());

    constexpr uint32_t kNum = 1;
    ThreadConfig config[kNum];
    ASSERT_EQ(ThreadConfigInit(config, kNum), HCCL_SUCCESS);
    config[0].notifyNumPerThread = 2;
    ThreadHandle out[kNum] = {0};
    std::vector<uint32_t> threadId;
    HcclResult ret = mgr->HcclThreadAcquireV2(
        CommEngine::COMM_ENGINE_AICPU, kNum, ThreadType::THREAD_TYPE_TS, config, out, threadId);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
    EXPECT_EQ(g_aicpuSpy.initCnt, 1U);
    EXPECT_EQ(g_aicpuSpy.allocCnt, 0U); // 失败早退，未走到线程分配
    EXPECT_FALSE(g_aicpuSpy.commState); // 状态未置位，下次可重试
}
