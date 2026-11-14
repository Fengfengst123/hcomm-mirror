/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "device_spec/ccu_address_space.h"
#include "device_spec/rank_address_space.h"
#include "device_spec_yaml_loader.h"

namespace {

using sim::devspec::CcuAddressSpace;
using sim::devspec::CcuResType;
using sim::devspec::DeviceSpec;
using sim::devspec::LoadDeviceSpecYaml;
using sim::devspec::RankAddressSpace;

const std::string kDir = HCCLVM_DEVICE_SPEC_TEST_DIR;
constexpr uint64_t DIE_BASE_A6 = 0x8e00000000ULL; // rank0 die0
constexpr uint64_t DIE_BASE_A5 = 0x8e00000000ULL;

class CcuAddrSpaceA6Test : public ::testing::Test {
protected:
    void SetUp() override
    {
        std::string err;
        ASSERT_TRUE(LoadDeviceSpecYaml(kDir + "/device_spec_yaml_demo_ascend960.yaml", m_spec, err)) << err;
        m_as = std::make_unique<CcuAddressSpace>(m_spec);
    }
    DeviceSpec m_spec;
    std::unique_ptr<CcuAddressSpace> m_as;
};

class CcuAddrSpaceA5Test : public ::testing::Test {
protected:
    void SetUp() override
    {
        std::string err;
        ASSERT_TRUE(LoadDeviceSpecYaml(kDir + "/device_spec_yaml_demo_ascend950.yaml", m_spec, err)) << err;
        m_as = std::make_unique<CcuAddressSpace>(m_spec);
    }
    DeviceSpec m_spec;
    std::unique_ptr<CcuAddressSpace> m_as;
};

// ===================== TypeByOffset / RegionOffset / EntrySize =====================

TEST_F(CcuAddrSpaceA6Test, TypeByOffsetMatchesLegacyA6Table)
{
    // 对照 runner ccu_resource_manager.cc:378-396 的 XN_*_ADDR_A6 常量块
    EXPECT_EQ(m_as->TypeByOffset(0x100000), CcuResType::XN);
    EXPECT_EQ(m_as->TypeByOffset(0x13FFFF), CcuResType::XN);
    EXPECT_EQ(m_as->TypeByOffset(0x140000), CcuResType::CKE); // 旧名 XN_CKB_ADDR_A6
    EXPECT_EQ(m_as->TypeByOffset(0x148000), CcuResType::PFE);
    EXPECT_EQ(m_as->TypeByOffset(0x150000), CcuResType::CHANNEL);
    EXPECT_EQ(m_as->TypeByOffset(0x170000), CcuResType::JETTY);
    EXPECT_EQ(m_as->TypeByOffset(0x178000), CcuResType::MISSION);
    EXPECT_EQ(m_as->TypeByOffset(0x180000), CcuResType::LOOP);
    EXPECT_EQ(m_as->TypeByOffset(0x0), CcuResType::INSTR);
    // MS 窗口
    EXPECT_EQ(m_as->TypeByOffset(0x4000000), CcuResType::MS);
    EXPECT_EQ(m_as->TypeByOffset(0x47FFFFF), CcuResType::MS);
    // 空洞
    EXPECT_EQ(m_as->TypeByOffset(0x188000), CcuResType::UNKNOWN);
    EXPECT_EQ(m_as->TypeByOffset(0x1000000), CcuResType::UNKNOWN);
    EXPECT_EQ(m_as->TypeByOffset(0x4800000), CcuResType::UNKNOWN);
}

TEST_F(CcuAddrSpaceA6Test, RegionOffsetAndEntrySize)
{
    bool ok = false;
    EXPECT_EQ(m_as->RegionOffset(CcuResType::XN, ok), 0x100000ULL);
    EXPECT_TRUE(ok);
    EXPECT_EQ(m_as->EntrySize(CcuResType::XN, ok), 8U);
    EXPECT_TRUE(ok);
    EXPECT_EQ(m_as->EntrySize(CcuResType::CHANNEL, ok), 32U);
    EXPECT_EQ(m_as->EntrySize(CcuResType::MISSION, ok), 64U);
    EXPECT_EQ(m_as->RegionCount(CcuResType::XN, ok), 16384U);
    EXPECT_TRUE(ok);
    // A6 无 GSA
    m_as->RegionOffset(CcuResType::GSA, ok);
    EXPECT_FALSE(ok);
    m_as->EntrySize(CcuResType::GSA, ok);
    EXPECT_FALSE(ok);
    // UNKNOWN
    m_as->EntrySize(CcuResType::UNKNOWN, ok);
    EXPECT_FALSE(ok);
}

TEST_F(CcuAddrSpaceA5Test, A5HasGsaRegion)
{
    bool ok = false;
    EXPECT_EQ(m_as->RegionOffset(CcuResType::GSA, ok), 0x900000ULL);
    EXPECT_TRUE(ok);
    EXPECT_EQ(m_as->RegionCount(CcuResType::GSA, ok), 3072U);
    EXPECT_EQ(m_as->TypeByOffset(0x900000), CcuResType::GSA);
    EXPECT_EQ(m_as->TypeByOffset(0x800000), CcuResType::INSTR);
    // A5 的 MS 在低地址
    EXPECT_EQ(m_as->TypeByOffset(0x0), CcuResType::MS);
    EXPECT_EQ(m_as->TypeByOffset(0x7FFFFF), CcuResType::MS);
    EXPECT_EQ(m_as->TypeByOffset(0x800000), CcuResType::INSTR);
    EXPECT_FALSE(m_spec.SupportsAddrDecode()) << "A5 用 id 编址";
}

// ===================== AddrToResId ↔ ResIdToAddr 互逆 =====================

TEST_F(CcuAddrSpaceA6Test, RoundTripAllRegions)
{
    for (const auto& r : m_spec.regions) {
        if (r.entrySize == 0) {
            continue;
        }
        // 首、尾、中间各测一次
        const std::vector<uint64_t> ids = {0, 1, r.count / 2, r.count - 1};
        for (uint64_t id : ids) {
            uint64_t addr = 0;
            ASSERT_TRUE(m_as->ResIdToAddr(DIE_BASE_A6, r.type, id, addr)) << "region=" << r.name << " id=" << id;
            EXPECT_GE(addr, DIE_BASE_A6 + r.offset);
            EXPECT_LT(addr, DIE_BASE_A6 + r.offset + r.size);
            CcuResType type = CcuResType::UNKNOWN;
            uint64_t backId = UINT64_MAX;
            ASSERT_TRUE(m_as->AddrToResId(DIE_BASE_A6, addr, type, backId))
                << "region=" << r.name << " addr=" << std::hex << addr;
            EXPECT_EQ(type, r.type) << "region=" << r.name;
            EXPECT_EQ(backId, id) << "region=" << r.name << " id=" << id;
        }
    }
}

/// 全 id 空间遍历互逆（A6 XN 16384、CKE 1024、CHANNEL 1024 …）
TEST_F(CcuAddrSpaceA6Test, ExhaustiveRoundTrip)
{
    for (const auto& r : m_spec.regions) {
        if (r.entrySize == 0) {
            continue;
        }
        for (uint64_t id = 0; id < r.count; ++id) {
            uint64_t addr = 0;
            ASSERT_TRUE(m_as->ResIdToAddr(DIE_BASE_A6, r.type, id, addr)) << r.name << " id=" << id;
            CcuResType type = CcuResType::UNKNOWN;
            uint64_t backId = UINT64_MAX;
            ASSERT_TRUE(m_as->AddrToResId(DIE_BASE_A6, addr, type, backId)) << r.name << " id=" << id;
            ASSERT_EQ(type, r.type) << r.name << " id=" << id;
            ASSERT_EQ(backId, id) << r.name << " id=" << id;
        }
    }
}

TEST_F(CcuAddrSpaceA6Test, AddrToIdOfTypeAgreesWithAddrToResId)
{
    uint64_t addr = 0;
    ASSERT_TRUE(m_as->ResIdToAddr(DIE_BASE_A6, CcuResType::CKE, 7, addr));
    uint64_t id = 0;
    ASSERT_TRUE(m_as->AddrToIdOfType(DIE_BASE_A6, CcuResType::CKE, addr, id));
    EXPECT_EQ(id, 7U);
    // 类型不匹配时必须失败（地址在 XN 区却按 CKE 解析）
    ASSERT_TRUE(m_as->ResIdToAddr(DIE_BASE_A6, CcuResType::XN, 3, addr));
    EXPECT_FALSE(m_as->AddrToIdOfType(DIE_BASE_A6, CcuResType::CKE, addr, id));
}

// ===================== 边界检查（修掉 ccu_resource_manager.cc:512,522 的 todo）=====================

TEST_F(CcuAddrSpaceA6Test, RejectsIdBeyondCount)
{
    uint64_t addr = 0;
    EXPECT_FALSE(m_as->ResIdToAddr(DIE_BASE_A6, CcuResType::XN, 16384, addr));
    EXPECT_FALSE(m_as->ResIdToAddr(DIE_BASE_A6, CcuResType::CKE, 1024, addr));
    EXPECT_FALSE(m_as->ResIdToAddr(DIE_BASE_A6, CcuResType::MISSION, 16, addr));
    EXPECT_FALSE(m_as->ResIdToAddr(DIE_BASE_A6, CcuResType::GSA, 0, addr)) << "A6 无 GSA region";
    EXPECT_EQ(addr, 0ULL);
}

TEST_F(CcuAddrSpaceA6Test, RejectsAddrOutsideRegions)
{
    CcuResType type = CcuResType::XN;
    uint64_t id = 0;
    // dieBase 之前
    EXPECT_FALSE(m_as->AddrToResId(DIE_BASE_A6, DIE_BASE_A6 - 1, type, id));
    // 空洞内
    EXPECT_FALSE(m_as->AddrToResId(DIE_BASE_A6, DIE_BASE_A6 + 0x188000, type, id));
    EXPECT_FALSE(m_as->AddrToResId(DIE_BASE_A6, DIE_BASE_A6 + 0x1000000, type, id));
    // 资源空间之外
    EXPECT_FALSE(m_as->AddrToResId(DIE_BASE_A6, DIE_BASE_A6 + 0x4800000, type, id));
    EXPECT_FALSE(m_as->AddrToResId(DIE_BASE_A6, DIE_BASE_A6 + 0x40000000, type, id));
    // dieBase 为 0
    EXPECT_FALSE(m_as->AddrToResId(0, 0x100000, type, id));
}

TEST_F(CcuAddrSpaceA6Test, RejectsUnalignedAddr)
{
    CcuResType type = CcuResType::XN;
    uint64_t id = 0;
    // XN entry_size = 8，偏移 3 字节不对齐
    EXPECT_FALSE(m_as->AddrToResId(DIE_BASE_A6, DIE_BASE_A6 + 0x100003, type, id));
    // CHANNEL entry_size = 32
    EXPECT_FALSE(m_as->AddrToResId(DIE_BASE_A6, DIE_BASE_A6 + 0x150008, type, id));
}

/**
 * ★ 超出 count 但仍在 region size 内的地址必须被拒。
 * A6 的 XN region size = 0x40000（可容 32768 槽），而 count = 16384 ——
 * 旧实现（ccu_resource_manager.cc:506-517）无上界校验，会把 id 16384..32767
 * 当成合法值返回，进而越界写 runner 的 xn_ 数组（设计文档 §2.5 / R6）。
 */
TEST_F(CcuAddrSpaceA6Test, RejectsIdWithinSizeButBeyondCount)
{
    CcuResType type = CcuResType::UNKNOWN;
    uint64_t id = 0;
    const uint64_t xnBase = 0x100000;
    // id 16384 → 偏移 0x20000，仍在 [0x100000, 0x140000) 内
    const uint64_t addr = DIE_BASE_A6 + xnBase + 16384ULL * 8;
    ASSERT_LT(addr - DIE_BASE_A6, 0x140000ULL) << "前置条件：地址仍在 XN region 内";
    EXPECT_FALSE(m_as->AddrToResId(DIE_BASE_A6, addr, type, id)) << "id ≥ count 必须拒绝";
    // 最后一个合法 id
    const uint64_t lastOk = DIE_BASE_A6 + xnBase + 16383ULL * 8;
    EXPECT_TRUE(m_as->AddrToResId(DIE_BASE_A6, lastOk, type, id));
    EXPECT_EQ(type, CcuResType::XN);
    EXPECT_EQ(id, 16383ULL);
}

// ===================== MS 交织 =====================

TEST_F(CcuAddrSpaceA6Test, MsRoundTripExhaustive)
{
    for (uint32_t msId = 0; msId < m_spec.ms.msNum; ++msId) {
        uint64_t addr = 0;
        ASSERT_TRUE(m_as->MsIdToAddr(DIE_BASE_A6, msId, addr)) << "msId=" << msId;
        // 地址必须落在某个 CCUA 窗口内
        const uint64_t off = addr - DIE_BASE_A6;
        EXPECT_GE(off, 0x4000000ULL);
        EXPECT_LT(off, 0x4800000ULL);
        uint32_t backId = UINT32_MAX;
        ASSERT_TRUE(m_as->MsAddrToId(DIE_BASE_A6, addr, backId)) << "msId=" << msId;
        ASSERT_EQ(backId, msId) << "msId=" << msId;
        // 与 TypeByOffset 一致
        EXPECT_EQ(m_as->TypeByOffset(off), CcuResType::MS);
        // AddrToResId 也应识别为 MS
        CcuResType type = CcuResType::UNKNOWN;
        uint64_t id = 0;
        ASSERT_TRUE(m_as->AddrToResId(DIE_BASE_A6, addr, type, id)) << "msId=" << msId;
        EXPECT_EQ(type, CcuResType::MS);
        EXPECT_EQ(id, msId);
    }
}

/// 交织语义抽查：msId 0..7 应落在 CCUA0 的前 8 个槽，msId 8..15 落在 CCUA1
TEST_F(CcuAddrSpaceA6Test, MsInterleaveSemantics)
{
    uint64_t addr = 0;
    for (uint32_t i = 0; i < 8; ++i) {
        ASSERT_TRUE(m_as->MsIdToAddr(DIE_BASE_A6, i, addr));
        EXPECT_EQ(addr - DIE_BASE_A6, 0x4000000ULL + i * 4096ULL) << "msId=" << i;
    }
    for (uint32_t i = 8; i < 16; ++i) {
        ASSERT_TRUE(m_as->MsIdToAddr(DIE_BASE_A6, i, addr));
        EXPECT_EQ(addr - DIE_BASE_A6, 0x4200000ULL + (i - 8) * 4096ULL) << "msId=" << i;
    }
    for (uint32_t i = 16; i < 24; ++i) {
        ASSERT_TRUE(m_as->MsIdToAddr(DIE_BASE_A6, i, addr));
        EXPECT_EQ(addr - DIE_BASE_A6, 0x4400000ULL + (i - 16) * 4096ULL) << "msId=" << i;
    }
    for (uint32_t i = 24; i < 32; ++i) {
        ASSERT_TRUE(m_as->MsIdToAddr(DIE_BASE_A6, i, addr));
        EXPECT_EQ(addr - DIE_BASE_A6, 0x4600000ULL + (i - 24) * 4096ULL) << "msId=" << i;
    }
    // 第二组（groupIdx=1）回到 CCUA0
    ASSERT_TRUE(m_as->MsIdToAddr(DIE_BASE_A6, 32, addr));
    EXPECT_EQ(addr - DIE_BASE_A6, 0x4000000ULL + 8 * 4096ULL);
    // 最后一个
    ASSERT_TRUE(m_as->MsIdToAddr(DIE_BASE_A6, 1535, addr));
    EXPECT_LT(addr - DIE_BASE_A6, 0x4800000ULL);
}

/// A5 的 MS 基址为 0x0，其余几何与 A6 相同
TEST_F(CcuAddrSpaceA5Test, MsRoundTripWithBaseZero)
{
    for (uint32_t msId = 0; msId < m_spec.ms.msNum; ++msId) {
        uint64_t addr = 0;
        ASSERT_TRUE(m_as->MsIdToAddr(DIE_BASE_A5, msId, addr)) << "msId=" << msId;
        EXPECT_LT(addr - DIE_BASE_A5, 0x800000ULL) << "A5 MS 全部落在 [0, 0x800000)";
        uint32_t backId = UINT32_MAX;
        ASSERT_TRUE(m_as->MsAddrToId(DIE_BASE_A5, addr, backId)) << "msId=" << msId;
        ASSERT_EQ(backId, msId);
    }
    uint64_t addr = 0;
    ASSERT_TRUE(m_as->MsIdToAddr(DIE_BASE_A5, 0, addr));
    EXPECT_EQ(addr, DIE_BASE_A5);
    ASSERT_TRUE(m_as->MsIdToAddr(DIE_BASE_A5, 8, addr));
    EXPECT_EQ(addr - DIE_BASE_A5, 0x200000ULL) << "msId 8 → CCUA1 起点";
}

TEST_F(CcuAddrSpaceA6Test, MsRejectsOutOfRange)
{
    uint64_t addr = 0;
    EXPECT_FALSE(m_as->MsIdToAddr(DIE_BASE_A6, 1536, addr));
    EXPECT_FALSE(m_as->MsIdToAddr(DIE_BASE_A6, UINT32_MAX, addr));
    uint32_t msId = 0;
    EXPECT_FALSE(m_as->MsAddrToId(DIE_BASE_A6, DIE_BASE_A6 + 0x4800000, msId));
    EXPECT_FALSE(m_as->MsAddrToId(DIE_BASE_A6, DIE_BASE_A6 + 0x3FFFFFF, msId));
    // 未按 ms_size 对齐
    EXPECT_FALSE(m_as->MsAddrToId(DIE_BASE_A6, DIE_BASE_A6 + 0x4000001, msId));
}

// ===================== 资源空间判定 =====================

TEST_F(CcuAddrSpaceA6Test, IsInResourceSpace)
{
    EXPECT_TRUE(m_as->IsInResourceSpace(DIE_BASE_A6, DIE_BASE_A6));
    EXPECT_TRUE(m_as->IsInResourceSpace(DIE_BASE_A6, DIE_BASE_A6 + 0x4800000 - 1));
    EXPECT_FALSE(m_as->IsInResourceSpace(DIE_BASE_A6, DIE_BASE_A6 + 0x4800000));
    EXPECT_FALSE(m_as->IsInResourceSpace(DIE_BASE_A6, DIE_BASE_A6 - 1));
    EXPECT_FALSE(m_as->IsInResourceSpace(DIE_BASE_A6, DIE_BASE_A6 + 0x40000000));
    EXPECT_FALSE(m_as->IsInResourceSpace(0, 0x100000));
}

/// ★ O12 的安全性依据：hcomm 的 SQ buffer 全范围必须落在 backing 内，
///   即使本期不建模 WQE region（D16）
TEST_F(CcuAddrSpaceA5Test, UnmodelledSqBufferRangeStillBacked)
{
    // hcomm ccu_jetty_ctx_mgr.cc:172: ccuResBaseVa + 0x1000000 + jettyCtxId × 256K
    const uint64_t sqBegin = 0x1000000ULL;
    const uint64_t sqEnd = 0x1000000ULL + 128ULL * 0x40000ULL; // = 0x3000000
    EXPECT_EQ(sqEnd, 0x3000000ULL);
    EXPECT_TRUE(m_as->IsInResourceSpace(DIE_BASE_A5, DIE_BASE_A5 + sqBegin));
    EXPECT_TRUE(m_as->IsInResourceSpace(DIE_BASE_A5, DIE_BASE_A5 + sqEnd - 1));
    // 但确实未被建模
    EXPECT_EQ(m_as->TypeByOffset(sqBegin), CcuResType::UNKNOWN);
    EXPECT_EQ(m_as->Spec().FindRegion(CcuResType::WQE), nullptr);
    // 且 backingSize 覆盖了它
    EXPECT_GE(m_spec.BackingSize(), sqEnd);
}

// ===================== 与 dieBase 解耦（per-rank 唯一）=====================

TEST_F(CcuAddrSpaceA6Test, WorksForAnyRankDieBase)
{
    std::string err;
    DeviceSpec spec;
    ASSERT_TRUE(LoadDeviceSpecYaml(kDir + "/device_spec_yaml_demo_ascend960.yaml", spec, err)) << err;
    const RankAddressSpace ras(spec.addr);
    for (uint32_t rank = 0; rank < 4; ++rank) {
        for (uint32_t die = 0; die < 2; ++die) {
            const uint64_t base = ras.CcuDieBase(rank, die);
            ASSERT_NE(base, 0ULL);
            uint64_t addr = 0;
            ASSERT_TRUE(m_as->ResIdToAddr(base, CcuResType::CKE, 5, addr));
            EXPECT_EQ(addr, base + 0x140000 + 5 * 8);
            CcuResType type = CcuResType::UNKNOWN;
            uint64_t id = 0;
            ASSERT_TRUE(m_as->AddrToResId(base, addr, type, id));
            EXPECT_EQ(type, CcuResType::CKE);
            EXPECT_EQ(id, 5ULL);
            // 用另一个 rank 的 base 解析同一 addr 必须失败或得到不同结果
            if (rank > 0) {
                const uint64_t otherBase = ras.CcuDieBase(0, die);
                EXPECT_FALSE(m_as->IsInResourceSpace(otherBase, addr))
                    << "跨 rank 地址不应被误判为在其它 rank 的资源空间内";
            }
        }
    }
}

} // namespace
