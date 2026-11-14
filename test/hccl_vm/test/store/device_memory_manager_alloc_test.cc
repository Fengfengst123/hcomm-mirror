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
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "device_spec/device_spec_registry.h"
#include "device_spec/rank_address_space.h"
#include "store_sim_device_memory_manager.h"

namespace {

using sim::DeviceMemoryManager;
using sim::devspec::AddrSpace;
using sim::devspec::DeviceSpecRegistry;
using sim::devspec::RankAddressSpace;

const std::string kDeviceDir = HCCLVM_DEVICE_SPEC_TEST_DIR;

/// rank 窗口内的 CCU 窗口位置（需求 3.2，被第三方基址要求锁定，见设计文档 §4.4.2）
constexpr uint64_t RANK_BASE = 0x8000000000ULL;
constexpr uint64_t RANK_STRIDE = 0x10000000000ULL;
constexpr uint64_t CCU_WIN_OFF = 0xE00000000ULL; // 56 GiB
constexpr uint64_t CCU_WIN_SIZE = 0x80000000ULL; // 2 GiB

uint64_t Va(const void* p) { return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p)); }

class AllocVirMemTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        auto& reg = DeviceSpecRegistry::GetInstance();
        reg.Reset();
        ASSERT_TRUE(reg.LoadAll(kDeviceDir)) << reg.LastError();
        m_addr = reg.CommonAddrSpace();
        ASSERT_NE(m_addr, nullptr) << "A5/A6 的 address_space 必须逐字段相同（验收 A14）";
        m_ras = std::make_unique<RankAddressSpace>(*m_addr);
    }
    void TearDown() override { DeviceSpecRegistry::GetInstance().Reset(); }

    /// 用互不相同的 devPhyId 隔离各用例的水位（DeviceMemoryManager 是进程级单例）
    static uint64_t NextDev()
    {
        static uint64_t s_dev = 1000;
        return ++s_dev;
    }

    const AddrSpace* m_addr{nullptr};
    std::unique_ptr<RankAddressSpace> m_ras;
};

/// ① 段 0 内分配的 VA 与改造前的公式 deviceBase + (devPhyId-1)*step + used 逐位相同（零回归）
TEST_F(AllocVirMemTest, ZeroRegressionWithinSeg0)
{
    const uint64_t dev = NextDev();
    const uint32_t rankIdx = static_cast<uint32_t>(dev - 1);
    uint64_t used = 0;
    const size_t sizes[] = {8, 64, 4096, 1 << 20, 100, 7};
    for (size_t sz : sizes) {
        void* p = DeviceMemoryManager::GetInstance().AllocVirMem(dev, sz);
        ASSERT_NE(p, nullptr);
        const uint64_t aligned = (static_cast<uint64_t>(sz) + 7) & ~7ULL;
        const uint64_t expect = RANK_BASE + rankIdx * RANK_STRIDE + used; // 改造前的公式
        EXPECT_EQ(Va(p), expect) << "size=" << sz << " used=" << std::hex << used;
        used += aligned;
    }
}

/// ② 跨段跳跃：段 0 用尽后跳到段 1，中间 2 GiB CCU 窗口永不被返回
TEST_F(AllocVirMemTest, JumpsOverCcuWindow)
{
    const uint64_t dev = NextDev();
    const uint32_t rankIdx = static_cast<uint32_t>(dev - 1);
    const uint64_t rankBase = m_ras->RankBase(rankIdx);
    ASSERT_EQ(rankBase, RANK_BASE + rankIdx * RANK_STRIDE);

    const uint64_t seg0Size = m_ras->UserMemSegSize(0);
    ASSERT_EQ(seg0Size, CCU_WIN_OFF);
    ASSERT_EQ(m_ras->UserMemSegCount(), 2U);

    // (a) 段内增量分配：VA 必须连续推进，且等于 rankBase + used（零回归）
    const uint64_t chunk = 0x10000000ULL; // 256 MiB
    uint64_t used = 0;
    for (int i = 0; i < 3; ++i) {
        void* p = DeviceMemoryManager::GetInstance().AllocVirMem(dev, chunk);
        ASSERT_NE(p, nullptr);
        EXPECT_EQ(Va(p), rankBase + used) << "第 " << i << " 次段内分配";
        used += chunk;
    }

    // (b) 把段 0 填到只剩 0x100（避免 chunk 整除 seg0 导致 remain==0 的边界）
    const uint64_t targetRemain = 0x100;
    const uint64_t fill = seg0Size - targetRemain - used;
    ASSERT_GT(fill, 0ULL);
    ASSERT_EQ(fill % 8, 0ULL);
    void* pf = DeviceMemoryManager::GetInstance().AllocVirMem(dev, fill);
    ASSERT_NE(pf, nullptr);
    EXPECT_EQ(Va(pf), rankBase + used);
    used += fill;
    ASSERT_EQ(seg0Size - used, targetRemain);

    // (c) 请求一个比段 0 余量大、但远小于段 1 容量的块 → 必须跳到段 1
    const uint64_t bigReq = targetRemain + 0x1000;
    void* jumped = DeviceMemoryManager::GetInstance().AllocVirMem(dev, bigReq);
    ASSERT_NE(jumped, nullptr);
    const uint64_t seg1Va = rankBase + m_ras->UserMemSegOffset(1);
    EXPECT_EQ(Va(jumped), seg1Va) << "应跳到段1 起点，而非在段0 尾部跨界";

    // 段1 起点 == CCU 窗口末端
    EXPECT_EQ(seg1Va, rankBase + CCU_WIN_OFF + CCU_WIN_SIZE);

    // 被跳过的区间恰为 CCU 窗口（2 GiB）
    EXPECT_EQ(seg1Va - (rankBase + CCU_WIN_OFF), CCU_WIN_SIZE) << "跳过长度须为 CCU 窗口 2 GiB";
    // 块首/块尾都不在 CCU 内
    uint32_t r = 0, d = 0;
    EXPECT_FALSE(m_ras->IsInCcuWindow(Va(jumped), r, d));
    EXPECT_FALSE(m_ras->IsInCcuWindow(Va(jumped) + bigReq - 1, r, d));
    // 浪费量（段0 尾部余量）严格小于单次请求大小（设计文档 §4.5.7.2）
    EXPECT_LT(targetRemain, bigReq);

    // (d) 反证：若用"单一全局线性水位"，VA 会是 rankBase + (seg0Size - 0x100)，
    //     块尾 [rankBase+0xE00000000, +0xE00001100) 侵入 CCU die0
    const uint64_t badVa = rankBase + used;
    EXPECT_TRUE(m_ras->IsInCcuWindow(badVa + bigReq - 1, r, d)) << "跨洞反例必须成立";
    EXPECT_EQ(d, 0U) << "侵入的是 die0";
    EXPECT_NE(Va(jumped), badVa) << "实际实现未踩洞";
}

/// ③ 任意成功分配的块，首尾双端都不得落在 CCU 窗口内，且必须在 rank 窗口内
TEST_F(AllocVirMemTest, NoAllocationEverTouchesCcu)
{
    const uint64_t dev = NextDev();
    const uint32_t rankIdx = static_cast<uint32_t>(dev - 1);
    const uint64_t rankBase = m_ras->RankBase(rankIdx);
    const uint64_t windowTop = rankBase + m_ras->RankWindowSize();

    // 混合大小，跨越段边界
    const size_t sizes[] = {8, 0x1000, 0x40000000ULL /*1GiB*/, 64, 0x100000, 0x20000000ULL, 4096};
    for (size_t sz : sizes) {
        void* p = DeviceMemoryManager::GetInstance().AllocVirMem(dev, sz);
        ASSERT_NE(p, nullptr) << "size=" << std::hex << sz;
        const uint64_t va = Va(p);
        const uint64_t aligned = (static_cast<uint64_t>(sz) + 7) & ~7ULL;
        uint32_t r = 0, d = 0;
        EXPECT_FALSE(m_ras->IsInCcuWindow(va, r, d)) << "块首落入 CCU, va=" << std::hex << va;
        EXPECT_FALSE(m_ras->IsInCcuWindow(va + aligned - 1, r, d))
            << "块尾落入 CCU, va=" << std::hex << va << " size=" << aligned;
        EXPECT_GE(va, rankBase);
        EXPECT_LE(va + aligned, windowTop) << "块尾超出 rank 窗口";
        EXPECT_TRUE(m_ras->IsUserMemVa(va, rankIdx));
        EXPECT_TRUE(m_ras->IsUserMemVa(va + aligned - 1, rankIdx));
    }
}

/// ④ 所有段耗尽 → 返回 nullptr（VA 超出 device 1 TiB 上限必须报错，不静默回绕）
TEST_F(AllocVirMemTest, ReturnsNullWhenExhausted)
{
    const uint64_t dev = NextDev();
    // 一次性请求超过 user_mem 总量（1022 GiB）
    const uint64_t overTotal = m_ras->UserMemTotal() + 0x1000;
    void* p = DeviceMemoryManager::GetInstance().AllocVirMem(dev, static_cast<size_t>(overTotal));
    EXPECT_EQ(p, nullptr) << "超过 user_mem 总量必须失败";

    // 超过单个段但小于总量：seg0 放不下 → 跳 seg1；seg1 也放不下 → 失败
    const uint64_t overSeg1 = m_ras->UserMemSegSize(1) + 0x1000;
    p = DeviceMemoryManager::GetInstance().AllocVirMem(dev, static_cast<size_t>(overSeg1));
    EXPECT_EQ(p, nullptr) << "超过最大段容量必须失败";
}

/// ⑤ latent bug B3 的守卫：size==0 必须返回 nullptr
/// ★ devPhyId==0 不在此列 —— 它是既有的 id 空间不一致问题（physical_id 0 起 vs
///   Device 主键 1 起，见 O14），S4 刻意保持改造前的行为以避免回归。
TEST_F(AllocVirMemTest, RejectsInvalidArgs)
{
    // B3: size == 0 原本会返回非空指针 + 插入零长度 VirtualMemBlock（范围查询永不命中）
    EXPECT_EQ(DeviceMemoryManager::GetInstance().AllocVirMem(NextDev(), 0), nullptr);
    // 对齐后回绕为 0 的极端 size
    EXPECT_EQ(DeviceMemoryManager::GetInstance().AllocVirMem(NextDev(), SIZE_MAX), nullptr);
    // devPhyId == 0：走降级路径，保持改造前的回绕行为（非 nullptr）
    void* legacy = DeviceMemoryManager::GetInstance().AllocVirMem(0, 0x1000);
    EXPECT_NE(legacy, nullptr) << "devPhyId==0 须保持既有行为（O14），不得因 S4 而回归";
}

/// 每 device 的水位互相独立，且按 rankIdx 落在各自窗口
TEST_F(AllocVirMemTest, PerDeviceWatermarksAreIndependent)
{
    const uint64_t devA = NextDev();
    const uint64_t devB = NextDev();
    ASSERT_NE(devA, devB);
    void* a = DeviceMemoryManager::GetInstance().AllocVirMem(devA, 0x1000);
    void* b = DeviceMemoryManager::GetInstance().AllocVirMem(devB, 0x1000);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    EXPECT_EQ(Va(a), m_ras->RankBase(static_cast<uint32_t>(devA - 1)));
    EXPECT_EQ(Va(b), m_ras->RankBase(static_cast<uint32_t>(devB - 1)));
    EXPECT_NE(Va(a), Va(b));
    // 各自水位独立推进
    void* a2 = DeviceMemoryManager::GetInstance().AllocVirMem(devA, 0x1000);
    ASSERT_NE(a2, nullptr);
    EXPECT_EQ(Va(a2), Va(a) + 0x1000);
    const auto usedA = DeviceMemoryManager::GetInstance().GetUsedPerSegment(devA);
    const auto usedB = DeviceMemoryManager::GetInstance().GetUsedPerSegment(devB);
    ASSERT_FALSE(usedA.empty());
    ASSERT_FALSE(usedB.empty());
    EXPECT_EQ(usedA[0], 0x2000ULL);
    EXPECT_EQ(usedB[0], 0x1000ULL);
}

/// 地址对齐：所有返回值必须 8 字节对齐
TEST_F(AllocVirMemTest, AllAllocationsAre8ByteAligned)
{
    const uint64_t dev = NextDev();
    for (size_t sz : {1, 2, 3, 5, 7, 9, 15, 17, 100, 4095}) {
        void* p = DeviceMemoryManager::GetInstance().AllocVirMem(dev, sz);
        ASSERT_NE(p, nullptr) << "size=" << sz;
        EXPECT_EQ(Va(p) % 8, 0ULL) << "size=" << sz;
    }
}

/// FreeVirMem 不回收水位（D17）：释放后再分配得到的是新地址，不复用
TEST_F(AllocVirMemTest, FreeDoesNotReclaimVa)
{
    const uint64_t dev = NextDev();
    void* p1 = DeviceMemoryManager::GetInstance().AllocVirMem(dev, 0x1000);
    ASSERT_NE(p1, nullptr);
    DeviceMemoryManager::GetInstance().FreeVirMem(dev, p1);
    void* p2 = DeviceMemoryManager::GetInstance().AllocVirMem(dev, 0x1000);
    ASSERT_NE(p2, nullptr);
    EXPECT_NE(Va(p1), Va(p2)) << "VA 不得复用（否则破坏 is_freed 行与活跃行不同址共存的不变量）";
    EXPECT_EQ(Va(p2), Va(p1) + 0x1000);
}

} // namespace
