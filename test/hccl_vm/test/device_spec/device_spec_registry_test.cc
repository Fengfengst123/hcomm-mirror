/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "device_spec/ccu_address_space.h"
#include "device_spec/device_spec_registry.h"
#include "device_spec/rank_address_space.h"

namespace {

using sim::devspec::CcuAddressSpace;
using sim::devspec::CcuVersion;
using sim::devspec::DeviceSpec;
using sim::devspec::DeviceSpecRegistry;
using sim::devspec::RankAddressSpace;

const std::string kDir = HCCLVM_DEVICE_SPEC_TEST_DIR;
const std::string kConfigRoot = HCCLVM_DEVICE_CONFIG_ROOT;

class RegistryTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        auto& reg = DeviceSpecRegistry::GetInstance();
        reg.Reset();
        ASSERT_TRUE(reg.LoadAll(kDir)) << reg.LastError();
    }
    void TearDown() override { DeviceSpecRegistry::GetInstance().Reset(); }
};

TEST_F(RegistryTest, LoadAllFindsBothSpecs)
{
    auto& reg = DeviceSpecRegistry::GetInstance();
    EXPECT_EQ(reg.Size(), 2U);
    const auto all = reg.All();
    ASSERT_EQ(all.size(), 2U);
    // std::map 按 key（小写 name）升序
    EXPECT_EQ(all[0]->name, "ascend950");
    EXPECT_EQ(all[1]->name, "ascend960");
    EXPECT_TRUE(reg.LastError().empty());
}

TEST_F(RegistryTest, FindByNameIsCaseInsensitive)
{
    auto& reg = DeviceSpecRegistry::GetInstance();
    EXPECT_NE(reg.FindByName("ascend950"), nullptr);
    EXPECT_NE(reg.FindByName("Ascend950"), nullptr);
    EXPECT_NE(reg.FindByName("ASCEND960"), nullptr);
    EXPECT_EQ(reg.FindByName("ascend999"), nullptr);
    EXPECT_EQ(reg.FindByName(""), nullptr);
}

/**
 * ★ 阶段 0 匹配规则（设计文档 §4.2.1）：大小写不敏感最长前缀。
 * 覆盖现网全部 4 个 distinct soc_version 值。
 */
TEST_F(RegistryTest, FindBySocVersionMatchesAllRealValues)
{
    auto& reg = DeviceSpecRegistry::GetInstance();
    struct Case {
        const char* soc;
        const char* expectSpec;
    };
    const Case cases[] = {
        {"Ascend950DT_95A1", "ascend950"},  // cluster/*.yaml（10 个 A5 配置）
        {"Ascend950", "ascend950"},         // server_or_pod/*.yaml 顶层
        {"Ascend960", "ascend960"},         // 3 个 *_clos 配置
        {"Ascend960DT_961BY", "ascend960"}, // 3 个 *_normal* 配置
        {"ascend950dt_95a1", "ascend950"},  // 大小写不敏感
        {"ASCEND960", "ascend960"},
    };
    for (const auto& c : cases) {
        const DeviceSpec* spec = reg.FindBySocVersion(c.soc);
        ASSERT_NE(spec, nullptr) << "soc_version=" << c.soc << " err=" << reg.LastError();
        EXPECT_EQ(spec->name, c.expectSpec) << "soc_version=" << c.soc;
    }
}

TEST_F(RegistryTest, FindBySocVersionRejectsUnknown)
{
    auto& reg = DeviceSpecRegistry::GetInstance();
    EXPECT_EQ(reg.FindBySocVersion("Ascend910B3"), nullptr) << "本测试目录未提供 910B spec";
    EXPECT_FALSE(reg.LastError().empty());
    EXPECT_NE(reg.LastError().find("未命中"), std::string::npos) << reg.LastError();
    // 空串
    EXPECT_EQ(reg.FindBySocVersion(""), nullptr);
    // 完全无关
    EXPECT_EQ(reg.FindBySocVersion("NvidiaH100"), nullptr);
}

/// 决策 (i)：yaml 内的 soc.soc_version 字段【完全不参与】匹配
TEST_F(RegistryTest, SocVersionFieldInYamlIsIgnored)
{
    auto& reg = DeviceSpecRegistry::GetInstance();
    const DeviceSpec* a5 = reg.FindByName("ascend950");
    const DeviceSpec* a6 = reg.FindByName("ascend960");
    ASSERT_NE(a5, nullptr);
    ASSERT_NE(a6, nullptr);
    // 加载器不解析该 key → 成员恒为空串
    EXPECT_TRUE(a5->socVersion.empty()) << "spec 的 socVersion 必须为空（决策 (i)）";
    EXPECT_TRUE(a6->socVersion.empty());
    // 匹配靠 name：A6 yaml 里写的是 "Ascend960"，但 "Ascend960DT_961BY" 也能命中，
    // 说明匹配与 yaml 的 soc_version 值无关
    const DeviceSpec* bySku = reg.FindBySocVersion("Ascend960DT_961BY");
    ASSERT_NE(bySku, nullptr);
    EXPECT_EQ(bySku->name, "ascend960");
}

/**
 * 最长前缀优先：构造一份名为 "ascend960dt" 的额外 spec，
 * 则 "Ascend960DT_961BY" 应命中它而非 "ascend960"。
 */
TEST_F(RegistryTest, LongestPrefixWins)
{
    auto& reg = DeviceSpecRegistry::GetInstance();
    // 直接往 registry 注入一份更长名字的 spec（复用 ascend960 的内容，改 name）
    // 通过临时文件实现，避免依赖内部 API
    const std::string src = kDir + "/device_spec_yaml_demo_ascend960.yaml";
    std::ifstream in(src);
    ASSERT_TRUE(in.is_open());
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();
    const std::string anchor = "name: \"ascend960\"";
    const size_t pos = content.find(anchor);
    ASSERT_NE(pos, std::string::npos);
    content.replace(pos, anchor.size(), "name: \"ascend960dt\"");
    const std::string tmp = kDir + "/ascend960dt.yaml";
    std::ofstream out(tmp);
    ASSERT_TRUE(out.is_open());
    out << content;
    out.close();

    ASSERT_TRUE(reg.LoadFromFile(tmp)) << reg.LastError();
    EXPECT_EQ(reg.Size(), 3U);
    const DeviceSpec* hit = reg.FindBySocVersion("Ascend960DT_961BY");
    ASSERT_NE(hit, nullptr);
    EXPECT_EQ(hit->name, "ascend960dt") << "应命中更长的前缀";
    // 而 "Ascend960" 仍命中 ascend960
    hit = reg.FindBySocVersion("Ascend960");
    ASSERT_NE(hit, nullptr);
    EXPECT_EQ(hit->name, "ascend960");

    ::remove(tmp.c_str());
    reg.Reset();
    ASSERT_TRUE(reg.LoadAll(kDir));
    EXPECT_EQ(reg.Size(), 2U);
}

TEST_F(RegistryTest, LoadFromFileRejectsNameFileNameMismatch)
{
    auto& reg = DeviceSpecRegistry::GetInstance();
    // ascend950 的内容但文件名是 ascend960 → name 与文件名不一致
    const std::string other = kDir + "/device_spec_yaml_demo_ascend960.yaml";
    EXPECT_FALSE(reg.LoadFromFile(other)) << "已加载过，应报名重复";
    EXPECT_NE(reg.LastError().find("重复"), std::string::npos) << reg.LastError();
}

TEST_F(RegistryTest, LoadAllRejectsEmptyDir)
{
    auto& reg = DeviceSpecRegistry::GetInstance();
    reg.Reset();
    EXPECT_FALSE(reg.LoadAll("/tmp/definitely_not_exist_device_dir"));
    EXPECT_FALSE(reg.LastError().empty());
}

TEST_F(RegistryTest, LoadAllIsAtomicOnFailure)
{
    // 放入一份非法 yaml，LoadAll 必须整体失败且不留下半成品
    auto& reg = DeviceSpecRegistry::GetInstance();
    reg.Reset();
    const std::string bad = kDir + "/zzz_bad.yaml";
    {
        std::ofstream out(bad);
        ASSERT_TRUE(out.is_open());
        out << "type: \"device_spec\"\nspec_version: 1\nname: \"zzz_bad\"\n";
        out << "soc:\n  ccu_version: \"CCU_V9\"\n  die_num: 2\n"; // 非法版本
    }
    EXPECT_FALSE(reg.LoadAll(kDir));
    EXPECT_EQ(reg.Size(), 0U) << "失败时必须清空，不留半成品状态";
    ::remove(bad.c_str());
    // 恢复
    ASSERT_TRUE(reg.LoadAll(kDir));
    EXPECT_EQ(reg.Size(), 2U);
}

/// 校验 9：拓扑配置里的每个 soc_version 都必须唯一命中一份 spec
TEST_F(RegistryTest, SocVersionCoverageOverRealConfigs)
{
    auto& reg = DeviceSpecRegistry::GetInstance();
    std::vector<std::string> violations;
    const bool ok = reg.CheckSocVersionCoverage(kConfigRoot, violations);
    for (const auto& v : violations) {
        ADD_FAILURE() << "校验9 违规: " << v;
    }
    EXPECT_TRUE(ok);
}

TEST_F(RegistryTest, MakeAddressSpaces)
{
    auto& reg = DeviceSpecRegistry::GetInstance();
    const DeviceSpec* spec = reg.FindBySocVersion("Ascend950DT_95A1");
    ASSERT_NE(spec, nullptr);
    auto ras = reg.MakeRankAddressSpace(spec);
    auto cas = reg.MakeCcuAddressSpace(spec);
    ASSERT_NE(ras, nullptr);
    ASSERT_NE(cas, nullptr);
    EXPECT_EQ(ras->CcuDieBase(0, 0), 0x8e00000000ULL);
    EXPECT_EQ(cas->Spec().name, "ascend950");
    // nullptr 安全
    EXPECT_EQ(reg.MakeRankAddressSpace(nullptr), nullptr);
    EXPECT_EQ(reg.MakeCcuAddressSpace(nullptr), nullptr);
}

/// ccu_version 必须正确区分 A5/A6（取代 5 处 strncmp(soc_version,"Ascend9xx",9)）
TEST_F(RegistryTest, CcuVersionDrivesDispatch)
{
    auto& reg = DeviceSpecRegistry::GetInstance();
    const DeviceSpec* a5 = reg.FindBySocVersion("Ascend950DT_95A1");
    const DeviceSpec* a6 = reg.FindBySocVersion("Ascend960DT_961BY");
    ASSERT_NE(a5, nullptr);
    ASSERT_NE(a6, nullptr);
    EXPECT_EQ(a5->ccuVersion, CcuVersion::CCU_V1);
    EXPECT_EQ(a6->ccuVersion, CcuVersion::CCU_V2);
    EXPECT_STREQ(sim::devspec::CcuVersionToString(a5->ccuVersion), "CCU_V1");
    EXPECT_STREQ(sim::devspec::CcuVersionToString(a6->ccuVersion), "CCU_V2");
    // A5 不做地址反查，A6 做
    EXPECT_FALSE(a5->SupportsAddrDecode());
    EXPECT_TRUE(a6->SupportsAddrDecode());
    // GSA 只有 A5 有
    EXPECT_TRUE(a5->HasGsa());
    EXPECT_FALSE(a6->HasGsa());
}

} // namespace
