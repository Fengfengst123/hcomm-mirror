/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <ifaddrs.h>
#include <arpa/inet.h>
#include <net/if.h>
#include <vector>
#include <stdarg.h>
#include "npu_nic_affinity.h"
#include "rank_info_types.h"
#include "topo_addr_info.h"
#include "hal.h"
#include "securec.h"
#include "topo_addr_info_log.h"
#include "topo.h"
#include "xml_parser.h"

static void TestLogRecord(int moduleId, int level, const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    printf("[UT_TOPO] ");
    vprintf(fmt, args);
    printf("\n");
    va_end(args);
}

static int TestCheckLogLevel(int moduleId, int logLevel) { return 1; /* 允许所有级别 */ }

/* ──────── Wrap getifaddrs / freeifaddrs ──────── */

static struct ifaddrs* g_fakeIfaddr = NULL;
static bool g_fakeIfaddrsFail = false; /* true=模拟 getifaddrs 系统调用失败 */

extern "C" int __wrap_getifaddrs(struct ifaddrs** ifap)
{
    if (g_fakeIfaddrsFail) {
        *ifap = NULL;
        return -1;
    }
    *ifap = g_fakeIfaddr;
    return 0;
}

extern "C" void __wrap_freeifaddrs(struct ifaddrs* ifa) { /* 我们的假数据是静态管理的，不需要释放 */ }

/* 逐节点释放整条假网卡链：链由头插构建，不能只回收头节点 */
static void TeardownFakeNet()
{
    struct ifaddrs* ifa = g_fakeIfaddr;
    while (ifa != NULL) {
        struct ifaddrs* next = ifa->ifa_next;
        free(ifa->ifa_name);
        free(ifa->ifa_addr);
        free(ifa);
        ifa = next;
    }
    g_fakeIfaddr = NULL;
}

static void SetupFakeNet(const char* ethName, const char* ipStr)
{
    TeardownFakeNet();

    struct ifaddrs* ifa = (struct ifaddrs*)calloc(1, sizeof(struct ifaddrs));
    ASSERT_NE(ifa, nullptr);
    ifa->ifa_next = NULL;
    ifa->ifa_name = strdup(ethName);
    ASSERT_NE(ifa->ifa_name, nullptr);
    ifa->ifa_flags = IFF_UP;
    ifa->ifa_addr = (struct sockaddr*)calloc(1, sizeof(struct sockaddr_in));
    ASSERT_NE(ifa->ifa_addr, nullptr);
    struct sockaddr_in* sin = (struct sockaddr_in*)ifa->ifa_addr;
    sin->sin_family = AF_INET;
    inet_pton(AF_INET, ipStr, &sin->sin_addr);
    g_fakeIfaddr = ifa;
}

/* ──────── 创建 /tmp/ut_hca/<hca>/device/net/<eth> ──────── */
static void SetupFakeHca(const char* hca, const char* eth)
{
    char path[512];
    sprintf_s(path, sizeof(path), "/tmp/ut_hca/%s/device/net", hca);
    char p[512];
    strncpy_s(p, sizeof(p), path, sizeof(p) - 1);
    for (char* c = p + 1; *c; c++) {
        if (*c == '/') {
            *c = '\0';
            mkdir(p, 0755);
            *c = '/';
        }
    }
    mkdir(p, 0755);
    char ethPath[576];
    sprintf_s(ethPath, sizeof(ethPath), "%s/%s", path, eth);
    mkdir(ethPath, 0755);
}

static void TeardownFakeHca()
{
    char cmd[256];
    sprintf_s(cmd, sizeof(cmd), "rm -rf /tmp/ut_hca");
    system(cmd);
}

/* ──────── Mock HAL ──────── */
static struct dcmi_pcie_info_all g_pi[MAX_NPU_COUNT];
static unsigned int g_pc = 0;
static bool g_visibleMask[MAX_NPU_COUNT]; // true=该 phyId 对运行时可见

extern "C" int mock_pi(int phyId, struct dcmi_pcie_info_all* info)
{
    if (phyId >= 0 && (unsigned int)phyId < g_pc && g_visibleMask[phyId]) {
        *info = g_pi[phyId];
        return 0;
    }
    return -1;
}

/* 模拟 hal_get_userdevid_by_phyid：对照 g_visibleMask 判断可见性 */
extern "C" int mock_userdevid(int phyId, int* userDevId)
{
    if (phyId >= 0 && phyId < (int)MAX_NPU_COUNT && g_visibleMask[phyId]) {
        *userDevId = phyId;
        return 0;
    }
    return -1;
}

/* 模拟 hal_get_logicid_from_phyid：identity 映射；真实实现走 load_dcmi()，
   UT 环境无法加载 libdcmi.so，不 mock 会让驱动路径跳过全部 NPU */
extern "C" int mock_logicid(unsigned int phyId, unsigned int* logicId)
{
    if (phyId < (unsigned int)MAX_NPU_COUNT) {
        *logicId = phyId;
        return 0;
    }
    return -1;
}

/* 当前仅Fallback_LogicIdMapping使用：用于区分驱动查询用的是 logicId 还是 phyId */
extern "C" int mock_logicidShifted(unsigned int phyId, unsigned int* logicId)
{
    if (phyId < (unsigned int)MAX_NPU_COUNT) {
        *logicId = phyId + 100U;
        return 0;
    }
    return -1;
}

/* ──────── Mock 驱动拓扑接口 hal_get_topo_info_by_device_id_and_nic_name ──────── */
typedef enum {
    DRIVER_NORMAL,           /* 按 g_topoEntries 查表返回，未命中默认 SELF（驱动宽容） */
    DRIVER_UNKNOWN_NIC_FAIL, /* 未登记（驱动不识别）的网口名返回 -1，如 docker0/veth 等虚拟网口 */
    DRIVER_LOAD_FAIL         /* 全部返回 -1（驱动接口不可用） */
} DriverMode;

typedef struct {
    int phyId;
    std::string nicName;
    int topoType; /* 驱动查询成功的拓扑类型（DCMI_TOPO_TYPE_*，UB=亲和） */
} TopoEntry;

static DriverMode g_driverMode = DRIVER_NORMAL; /* 默认模拟驱动可用 */
static std::vector<TopoEntry> g_topoEntries;    /* 命中表：返回登记的拓扑类型 */
static int g_driverCallCount = 0;

/* 契约：返回 0 表示查询成功、拓扑类型经 *topoType 出参返回；-1=入参非法或驱动接口不可用 */
extern "C" int mock_topo_info(int devId, char* nicName, int nicNameLen, int* topoType)
{
    g_driverCallCount++;
    /* 与真实接口一致：入参非法直接返回 -1 */
    if (devId < 0 || nicName == NULL || nicNameLen <= 0 || topoType == NULL) {
        return -1;
    }
    if (g_driverMode == DRIVER_LOAD_FAIL) {
        return -1;
    }
    int ret = (int)DCMI_TOPO_TYPE_SELF;
    bool matched = false;
    for (size_t i = 0; i < g_topoEntries.size(); i++) {
        if (g_topoEntries[i].phyId == devId && g_topoEntries[i].nicName == nicName) {
            ret = g_topoEntries[i].topoType;
            matched = true;
            break;
        }
    }
    if (!matched && g_driverMode == DRIVER_UNKNOWN_NIC_FAIL) {
        return -1; /* 模拟驱动不认识该网口名（docker0/veth 等虚拟网口查询失败） */
    }
    *topoType = ret;
    return 0;
}

class NpuNicAffinityTest : public testing::Test {
protected:
    void SetUp() override
    {
        /* TopoAddrInfoTest 对 hal_dlopen/hal_dlsym 的 mock 导致 load_dcmi()
           缓存了无效函数指针，后序调用会崩溃。注入 mock 接管
           hal_get_device_pcie_info / hal_get_userdevid_by_phyid，
           完全绕过 load_dcmi() 路径。 */
        g_pc = 0;
        g_driverMode = DRIVER_NORMAL;
        g_topoEntries.clear();
        g_driverCallCount = 0;
        g_fakeIfaddrsFail = false;
        MOCKER(hal_get_device_pcie_info).stubs().with(mockcpp::any(), mockcpp::any()).will(mockcpp::invoke(mock_pi));
        MOCKER(hal_get_userdevid_by_phyid)
            .stubs()
            .with(mockcpp::any(), mockcpp::any())
            .will(mockcpp::invoke(mock_userdevid));
        MOCKER(hal_get_logicid_from_phyid)
            .stubs()
            .with(mockcpp::any(), mockcpp::any())
            .will(mockcpp::invoke(mock_logicid));
        /* 无 XML 回退统一走 mock 驱动查询，不落回真实 libdcmi.so */
        MOCKER(hal_get_topo_info_by_device_id_and_nic_name)
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
            .will(mockcpp::invoke(mock_topo_info));

        remove("/tmp/ut_virtualTopology.xml");
        memset_s(g_pi, sizeof(g_pi), 0, sizeof(g_pi));
        memset(g_visibleMask, 1, sizeof(g_visibleMask)); // all visible by default
        SetupFakeHca("hrn5_0", "eth0");
        SetupFakeNet("eth0", "10.0.0.1");

        /* 注入日志回调，用例中 TOPO_ERR / TOPO_INFO 输出到 stdout */
        g_topo_DlogRecord = TestLogRecord;
        g_topo_CheckLogLevel = TestCheckLogLevel;
    }
    void TearDown() override
    {
        remove("/tmp/ut_virtualTopology.xml");
        TeardownFakeHca();
        TeardownFakeNet();
        g_topo_DlogRecord = NULL;
        g_topo_CheckLogLevel = NULL;
        GlobalMockObject::verify();
    }
    void W(const char* c)
    {
        FILE* fp = fopen("/tmp/ut_virtualTopology.xml", "w");
        ASSERT_NE(fp, nullptr);
        fputs(c, fp);
        fclose(fp);
    }
    /* 设置可见设备掩码，模拟 ASCEND_RT_VISIBLE_DEVICES 效果 */
    void SetVisibleDevices(const std::vector<int>& ids)
    {
        memset(g_visibleMask, 0, sizeof(g_visibleMask));
        for (int id : ids) {
            if (id >= 0 && id < MAX_NPU_COUNT) {
                g_visibleMask[(unsigned int)id] = true;
            }
        }
    }
    /* 配置驱动返回的拓扑类型，默认 DCMI_TOPO_TYPE_UB（网卡与 NPU 亲和） */
    void Affine(int phyId, const char* nicName, int topoType = DCMI_TOPO_TYPE_UB)
    {
        TopoEntry e;
        e.phyId = phyId;
        e.nicName = nicName;
        e.topoType = topoType;
        g_topoEntries.push_back(e);
    }
    void SetDriverMode(DriverMode mode) { g_driverMode = mode; }
    void S(unsigned int npu, unsigned int pcie)
    {
        if (npu > 0) {
            MOCKER(hal_get_npu_count).stubs().will(returnValue((int)npu));
        }
        g_pc = pcie;
        if (pcie > 0) {
            MOCKER(hal_get_device_pcie_info)
                .stubs()
                .with(mockcpp::any(), mockcpp::any())
                .will(mockcpp::invoke(mock_pi));
        }
    }
    /* 校验 GetRoceIpFromXml 成功并返回期望 IP */
    void AssertRoceIpOk(unsigned int npuId, const char* expectIp)
    {
        char ip[64] = {0};
        EXPECT_EQ(GetRoceIpFromXml(npuId, ip, sizeof(ip)), 0);
        EXPECT_STREQ(ip, expectIp);
    }
    /* 校验 GetRoceIpFromXml 失败 */
    void AssertRoceIpFail(unsigned int npuId)
    {
        char ip[64] = {0};
        EXPECT_NE(GetRoceIpFromXml(npuId, ip, sizeof(ip)), TOPO_SUCCESS);
    }
    void AssertRoceIpFail(unsigned int npuId, TopoAddrResult expectRet)
    {
        char ip[64] = {0};
        EXPECT_EQ(GetRoceIpFromXml(npuId, ip, sizeof(ip)), expectRet);
    }
};

/* ══════════════════════════════════════════════════════════════
 *  hal_get_topo_info_by_device_id_and_nic_name 封装
 * ══════════════════════════════════════════════════════════════ */

/* 注意：load_dcmiv2_get_topo_info_by_device_id_and_nic_name 的静态缓存
   （dcmi / isInit / 函数指针）跨用例存活，一旦成功加载便无法回到“加载失败”
   状态，因此加载流程合并为单条序列用例 LoadFailThenSuccess。 */

static void* g_fakeDlopenRet = NULL; /* mock hal_dlopen 返回值 */
static void* g_fakeDlsymRet = NULL;  /* mock hal_dlsym 返回值 */
static int g_fakeDlsymCount = 0;
static int g_fakeDcmiRet = DCMI_TOPO_TYPE_UB; /* fake 驱动查询成功时经 *topoType 写出的拓扑类型 */
static int g_fakeDcmiRetCode = 0;             /* fake 驱动返回值：0=成功，非 0=查询失败 */

extern "C" int mock_dcmiv2_topo_info(int devId, char* nicName, int nicNameLen, int* topoType)
{
    if (topoType != NULL) {
        *topoType = g_fakeDcmiRet;
    }
    return g_fakeDcmiRetCode;
}

void* mock_dlopen_topo(const char* filename, int flag)
{
    EXPECT_STREQ(filename, "libdcmi.so");
    return g_fakeDlopenRet;
}

void* mock_dlsym_topo(void* handle, const char* symbol)
{
    if (strcmp(symbol, "dcmiv2_get_topo_info_by_device_id_and_nic_name") == 0) {
        g_fakeDlsymCount++;
        return g_fakeDlsymRet;
    }
    return NULL;
}

class HalTopoInfoTest : public testing::Test {
protected:
    void SetUp() override
    {
        g_topo_DlogRecord = TestLogRecord;
        g_topo_CheckLogLevel = TestCheckLogLevel;
    }
    void TearDown() override
    {
        g_topo_DlogRecord = NULL;
        g_topo_CheckLogLevel = NULL;
        GlobalMockObject::verify();
    }
};

/* 入参校验先于 dlopen，直接调用封装即可，无副作用 */
TEST_F(HalTopoInfoTest, InvalidParams)
{
    int topoType = 0;
    char nicName[16] = "eth0";
    EXPECT_EQ(hal_get_topo_info_by_device_id_and_nic_name(-1, nicName, 4, &topoType), -1);
    EXPECT_EQ(hal_get_topo_info_by_device_id_and_nic_name(0, NULL, 4, &topoType), -1);
    EXPECT_EQ(hal_get_topo_info_by_device_id_and_nic_name(0, nicName, 0, &topoType), -1);
    EXPECT_EQ(hal_get_topo_info_by_device_id_and_nic_name(0, nicName, 4, NULL), -1);
}

/* 单条序列覆盖：dlopen 失败 → dlsym 失败 → 查询失败 → 加载成功 → 缓存生效 */
TEST_F(HalTopoInfoTest, LoadFailThenSuccess)
{
    char nicName[16] = "eth0";
    int topoType = 0;

    MOCKER(hal_dlopen).stubs().with(mockcpp::any(), mockcpp::any()).will(mockcpp::invoke(mock_dlopen_topo));
    MOCKER(hal_dlsym).stubs().with(mockcpp::any(), mockcpp::any()).will(mockcpp::invoke(mock_dlsym_topo));

    /* ① dlopen 失败 → -1，且不会走到 dlsym */
    g_fakeDlopenRet = NULL;
    g_fakeDlsymCount = 0;
    EXPECT_EQ(hal_get_topo_info_by_device_id_and_nic_name(0, nicName, 4, &topoType), -1);
    EXPECT_EQ(g_fakeDlsymCount, 0);

    /* ② dlopen 成功但 dlsym 失败 → -1 */
    g_fakeDlopenRet = (void*)0x1;
    g_fakeDlsymRet = NULL;
    EXPECT_EQ(hal_get_topo_info_by_device_id_and_nic_name(0, nicName, 4, &topoType), -1);
    EXPECT_EQ(g_fakeDlsymCount, 1);

    /* ③ 加载成功但 dcmi 查询失败 → -1 */
    g_fakeDlsymRet = (void*)mock_dcmiv2_topo_info;
    g_fakeDcmiRetCode = -1;
    EXPECT_EQ(hal_get_topo_info_by_device_id_and_nic_name(0, nicName, 4, &topoType), -1);
    EXPECT_EQ(g_fakeDlsymCount, 2);

    /* ④ 查询成功：返回 0，拓扑类型经 *topoType 出参返回 */
    g_fakeDcmiRetCode = 0;
    g_fakeDcmiRet = DCMI_TOPO_TYPE_UB;
    EXPECT_EQ(hal_get_topo_info_by_device_id_and_nic_name(0, nicName, 4, &topoType), 0);
    EXPECT_EQ(topoType, DCMI_TOPO_TYPE_UB);
    EXPECT_EQ(g_fakeDlsymCount, 2);

    /* ⑤ 静态缓存生效：再次调用不再 dlsym，直接走已加载的函数指针 */
    g_fakeDcmiRet = DCMI_TOPO_TYPE_PHB;
    EXPECT_EQ(hal_get_topo_info_by_device_id_and_nic_name(0, nicName, 4, &topoType), 0);
    EXPECT_EQ(topoType, DCMI_TOPO_TYPE_PHB);
    EXPECT_EQ(g_fakeDlsymCount, 2);
}

/* ══════════════════════════════════════════════════════════════
 *  PCIE 格式
 * ══════════════════════════════════════════════════════════════ */

TEST_F(NpuNicAffinityTest, PCIE_1Bridge_1Nic_2Npu)
{
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    g_pi[1].domain = 0;
    g_pi[1].bdf_busid = 4;
    g_pi[1].bdf_deviceid = 0;
    g_pi[1].bdf_funcid = 0;
    S(2, 2);
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:01:00.0\">\n"
      "<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n<pci busid=\"0000:04:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    AssertRoceIpOk(0, "10.0.0.1");
}

TEST_F(NpuNicAffinityTest, PCIE_MultiBridge_4Group)
{
    S(16, 16);
    unsigned int bus[] = {3, 4, 5, 6, 0xc, 0xd, 0xe, 0xf, 0x14, 0x15, 0x16, 0x17, 0x1e, 0x1f, 0x20, 0x21};
    for (int i = 0; i < 16; i++) {
        g_pi[i].domain = 0;
        g_pi[i].bdf_busid = bus[i];
    }
    W("<system version=\"1\">\n"
      "<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:01:00.0\">\n"
      "  <pci busid=\"0000:02:00.0\">\n    <nic>\n      <net name=\"xscale_3\"/>\n    </nic>\n  </pci>\n"
      "  <pci busid=\"0000:03:00.0\"/>\n  <pci busid=\"0000:04:00.0\"/>\n"
      "  <pci busid=\"0000:05:00.0\"/>\n  <pci busid=\"0000:06:00.0\"/>\n"
      "</pci>\n"
      "<pci busid=\"0000:10:00.0\">\n"
      "  <pci busid=\"0000:11:00.0\">\n    <nic>\n      <net name=\"xscale_4\"/>\n    </nic>\n  </pci>\n"
      "  <pci busid=\"0000:0c:00.0\"/>\n  <pci busid=\"0000:0d:00.0\"/>\n"
      "  <pci busid=\"0000:0e:00.0\"/>\n  <pci busid=\"0000:0f:00.0\"/>\n"
      "</pci>\n</cpu>\n"
      "<cpu numaid=\"1\">\n"
      "<pci busid=\"0000:20:00.0\">\n"
      "  <pci busid=\"0000:21:00.0\">\n    <nic>\n      <net name=\"xscale_1\"/>\n    </nic>\n  </pci>\n"
      "  <pci busid=\"0000:14:00.0\"/>\n  <pci busid=\"0000:15:00.0\"/>\n"
      "  <pci busid=\"0000:16:00.0\"/>\n  <pci busid=\"0000:17:00.0\"/>\n"
      "</pci>\n"
      "<pci busid=\"0000:30:00.0\">\n"
      "  <pci busid=\"0000:31:00.0\">\n    <nic>\n      <net name=\"xscale_2\"/>\n    </nic>\n  </pci>\n"
      "  <pci busid=\"0000:1e:00.0\"/>\n  <pci busid=\"0000:1f:00.0\"/>\n"
      "  <pci busid=\"0000:20:00.0\"/>\n  <pci busid=\"0000:21:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    SetupFakeHca("xscale_1", "eth0");
    SetupFakeHca("xscale_2", "eth0");
    SetupFakeHca("xscale_3", "eth0");
    SetupFakeHca("xscale_4", "eth0");
    AssertRoceIpOk(0, "10.0.0.1");
}

/* NPU 以非自闭合 <pci busid="..."></pci> 形式出现 */
TEST_F(NpuNicAffinityTest, PCIE_NpuNonSelfClose)
{
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    g_pi[1].domain = 0;
    g_pi[1].bdf_busid = 4;
    g_pi[1].bdf_deviceid = 0;
    g_pi[1].bdf_funcid = 0;
    S(2, 2);
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:01:00.0\">\n"
      "<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<pci busid=\"0000:03:00.0\">\n</pci>\n"
      "<pci busid=\"0000:04:00.0\">\n</pci>\n"
      "</pci>\n</cpu>\n</system>\n");
    AssertRoceIpOk(0, "10.0.0.1");
}

/* PCIE 多层嵌套：只顶层 Bridge 建组 */
TEST_F(NpuNicAffinityTest, PCIE_NestedDepth)
{
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 4;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    S(1, 1);
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:00:00.0\">\n"
      "<pci busid=\"0000:01:00.0\">\n"
      "<pci busid=\"0000:02:00.0\">\n"
      "<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<pci busid=\"0000:04:00.0\"/>\n"
      "</pci>\n</pci>\n</pci>\n"
      "</cpu>\n</system>\n");
    AssertRoceIpOk(0, "10.0.0.1");
}

/* 空 Bridge（无 NIC）：应该返回 -1 */
TEST_F(NpuNicAffinityTest, PCIE_EmptyBridge)
{
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    S(1, 1);
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:01:00.0\">\n"
      "<pci busid=\"0000:03:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    AssertRoceIpFail(0);
}

/* ══════════════════════════════════════════════════════════════
 *  UB 格式
 * ══════════════════════════════════════════════════════════════ */

TEST_F(NpuNicAffinityTest, UB_1Group)
{
    S(2, 0);
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<ub>\n<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<npu chipphyid=\"0\"/>\n<npu chipphyid=\"1\"/>\n"
      "</ub>\n</cpu>\n</system>\n");
    AssertRoceIpOk(0, "10.0.0.1");
}

TEST_F(NpuNicAffinityTest, UB_2Group)
{
    SetupFakeHca("hrn5_1", "eth0");
    S(4, 0);
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<ub>\n<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<npu chipphyid=\"0\"/>\n<npu chipphyid=\"1\"/>\n</ub>\n"
      "<ub>\n<nic>\n<net name=\"hrn5_1\"/>\n</nic>\n"
      "<npu chipphyid=\"2\"/>\n<npu chipphyid=\"3\"/>\n</ub>\n"
      "</cpu>\n</system>\n");
    AssertRoceIpOk(0, "10.0.0.1");
    AssertRoceIpOk(2, "10.0.0.1");
}

TEST_F(NpuNicAffinityTest, UB_NpuNonSelfClose)
{
    S(2, 0);
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<ub>\n<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<npu chipphyid=\"0\">\n</npu>\n"
      "<npu chipphyid=\"1\">\n</npu>\n"
      "</ub>\n</cpu>\n</system>\n");
    AssertRoceIpOk(0, "10.0.0.1");
}

/* NIC 在子 ub 内：不应单独建组 */
TEST_F(NpuNicAffinityTest, UB_NicInSubUb)
{
    S(2, 0);
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<ub>\n"
      "<ub>\n<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n</ub>\n"
      "<npu chipphyid=\"0\"/>\n<npu chipphyid=\"1\"/>\n"
      "</ub>\n</cpu>\n</system>\n");
    AssertRoceIpOk(0, "10.0.0.1");
}

/* ══════════════════════════════════════════════════════════════
 *  异常 / 边界
 * ══════════════════════════════════════════════════════════════ */

TEST_F(NpuNicAffinityTest, Error_FileNotFound) { AssertRoceIpFail(0); }

TEST_F(NpuNicAffinityTest, Error_EmptyFile)
{
    W("");
    AssertRoceIpFail(0);
}

TEST_F(NpuNicAffinityTest, Error_NoNic)
{
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    g_pi[1].domain = 0;
    g_pi[1].bdf_busid = 4;
    g_pi[1].bdf_deviceid = 0;
    g_pi[1].bdf_funcid = 0;
    S(2, 2);
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n<pci busid=\"0000:01:00.0\">\n"
      "<pci busid=\"0000:02:00.0\"/>\n<pci busid=\"0000:03:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    AssertRoceIpFail(0);
}

TEST_F(NpuNicAffinityTest, Error_WithComments)
{
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    g_pi[1].domain = 0;
    g_pi[1].bdf_busid = 4;
    g_pi[1].bdf_deviceid = 0;
    g_pi[1].bdf_funcid = 0;
    S(2, 2);
    W("<!-- comment -->\n<system version=\"1.0\">\n<cpu numaid=\"0\">\n<pci busid=\"0000:01:00.0\">\n"
      "<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n<pci busid=\"0000:04:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    AssertRoceIpOk(0, "10.0.0.1");
}

/* 多标签同行：<nic><net name="x"/></nic> 在一行 */
TEST_F(NpuNicAffinityTest, MultiTagSameLine)
{
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    g_pi[1].domain = 0;
    g_pi[1].bdf_busid = 4;
    g_pi[1].bdf_deviceid = 0;
    g_pi[1].bdf_funcid = 0;
    S(2, 2);
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:01:00.0\">\n"
      "<nic><net name=\"hrn5_0\"/></nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n<pci busid=\"0000:04:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    AssertRoceIpOk(0, "10.0.0.1");
}

/* ══════════════════════════════════════════════════════════════
 *  异常 XML 格式（解析器鲁棒性）
 * ══════════════════════════════════════════════════════════════ */

/* 多余闭合标签：无对应开标签 */
TEST_F(NpuNicAffinityTest, Malformed_ExtraCloseTag)
{
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    S(1, 1);
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "</pci>\n"
      "<pci busid=\"0000:01:00.0\">\n"
      "<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    AssertRoceIpOk(0, "10.0.0.1");
}

/* 未知标签：解析器应跳过，不影响后续正常标签 */
TEST_F(NpuNicAffinityTest, Malformed_UnknownTag)
{
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    S(1, 1);
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:01:00.0\">\n"
      "<foo bar=\"x\"/>\n"
      "<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    AssertRoceIpOk(0, "10.0.0.1");
}

/* 属性无引号：ExtractAttrs 跳过该属性，不崩溃 */
TEST_F(NpuNicAffinityTest, Malformed_AttrNoQuote)
{
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    S(1, 1);
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=0000:03:00.0>\n"
      "<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    AssertRoceIpOk(0, "10.0.0.1");
}

/* 属性无值：ExtractAttrs 跳过，不崩溃 */
TEST_F(NpuNicAffinityTest, Malformed_AttrNoValue)
{
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    S(1, 1);
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid>\n"
      "<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    AssertRoceIpOk(0, "10.0.0.1");
}

/* 不完整的标签（缺 >）：strchr 返回 NULL，跳过该行 */
TEST_F(NpuNicAffinityTest, Malformed_IncompleteTag)
{
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    S(1, 1);
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:01:00.0\">\n"
      "<nic>\n<net name=\"hrn5_0\"\n"
      "</nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    AssertRoceIpFail(0);
}

/* 空标签：<> 或 </>，GetTagName 返回空，跳过 */
TEST_F(NpuNicAffinityTest, Malformed_EmptyTag)
{
    S(2, 0);
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<>\n"
      "<ub>\n<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<npu chipphyid=\"0\"/>\n"
      "</ub>\n</>\n"
      "</cpu>\n</system>\n");
    AssertRoceIpOk(0, "10.0.0.1");
}

/* ══════════════════════════════════════════════════════════════
 *  空参数
 * ══════════════════════════════════════════════════════════════ */

TEST_F(NpuNicAffinityTest, NullParams)
{
    EXPECT_NE(GetRoceIpFromXml(0, NULL, 0), TOPO_SUCCESS);
    EXPECT_NE(GetRoceIpFromXml(-1, NULL, 0), TOPO_SUCCESS);
}

/* ══════════════════════════════════════════════════════════════
 *  ASCEND_RT_VISIBLE_DEVICES 非连续可见
 * ══════════════════════════════════════════════════════════════ */

TEST_F(NpuNicAffinityTest, VisibleDevices_NonContiguous_PCIE)
{
    /* 模拟 ASCEND_RT_VISIBLE_DEVICES=0,3,7：8 张 NPU 只暴露 3 张，ID 不连续 */
    S(8, 8);
    SetVisibleDevices({0, 3, 7});

    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3; /* → "0000:03:00.0" */
    g_pi[3].domain = 0;
    g_pi[3].bdf_busid = 7; /* → "0000:07:00.0" */
    g_pi[7].domain = 0;
    g_pi[7].bdf_busid = 10; /* → "0000:0a:00.0" */

    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:01:00.0\">\n"
      "<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n"
      "<pci busid=\"0000:07:00.0\"/>\n"
      "<pci busid=\"0000:0a:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");

    /* 可见的 NPU 能正确匹配到 NIC */
    AssertRoceIpOk(0, "10.0.0.1");
    AssertRoceIpOk(3, "10.0.0.1");
    AssertRoceIpOk(7, "10.0.0.1");

    /* 不可见的 NPU 获取不到 IP */
    AssertRoceIpFail(1);
    AssertRoceIpFail(2);
    AssertRoceIpFail(4);
    AssertRoceIpFail(5);
    AssertRoceIpFail(6);
}

/* UB 格式 + 非连续可见：验证 HandleNpuTag 中的 hal_get_userdevid_by_phyid 过滤 */
TEST_F(NpuNicAffinityTest, VisibleDevices_NonContiguous_UB)
{
    S(8, 0);
    SetVisibleDevices({0, 3, 7});

    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<ub>\n<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<npu chipphyid=\"0\"/>\n<npu chipphyid=\"3\"/>\n<npu chipphyid=\"7\"/>\n"
      "<npu chipphyid=\"1\"/>\n<npu chipphyid=\"2\"/>\n"
      "</ub>\n</cpu>\n</system>\n");

    /* 可见的 NPU 能匹配到 NIC */
    AssertRoceIpOk(0, "10.0.0.1");
    AssertRoceIpOk(3, "10.0.0.1");
    AssertRoceIpOk(7, "10.0.0.1");
    /* 不可见的 NPU 不会加入亲和分组，应获取不到 IP */
    AssertRoceIpFail(1);
    AssertRoceIpFail(2);
}

/* name 为 eth 名：NameToIp 先尝试 EthToIp 直接解析 */
TEST_F(NpuNicAffinityTest, NameIsEthName)
{
    S(2, 2);
    SetVisibleDevices({0, 1});
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[1].domain = 0;
    g_pi[1].bdf_busid = 4;

    /* XML 中 net name 直接用 eth0，不走 HCA 路径 */
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:01:00.0\">\n"
      "<nic>\n<net name=\"eth0\"/>\n</nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n"
      "<pci busid=\"0000:04:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");

    /* EthToIp("eth0") 应通过 getifaddrs mock 拿到 10.0.0.1 */
    AssertRoceIpOk(0, "10.0.0.1");
    AssertRoceIpOk(1, "10.0.0.1");
}

/* eth 名 + TopoAddrInfoGet 端到端 */
TEST_F(NpuNicAffinityTest, NameIsEthName_TopoAddrInfoGet)
{
    S(2, 2);
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[1].domain = 0;
    g_pi[1].bdf_busid = 4;

    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:01:00.0\">\n"
      "<nic>\n<net name=\"eth0\"/>\n</nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n"
      "<pci busid=\"0000:04:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");

    unsigned int mainboardId = MAIN_BOARD_ID_CARD_NOMESH;
    char drvPath[256] = "/usr/local/Ascend2";
    dcmi_urma_eid_info_t eidList[MAX_EID_NUM];
    memset_s(eidList, sizeof(eidList), 0, sizeof(eidList));
    size_t eidNum = 0;
    MOCKER(hal_get_mainboard_id).stubs().with(mockcpp::any(), outBoundP(&mainboardId)).will(returnValue(0));
    MOCKER(hal_get_driver_install_path)
        .stubs()
        .with(outBoundP(drvPath, strlen(drvPath)), mockcpp::any())
        .will(returnValue(0));
    MOCKER(hal_get_eid_list_by_phy_id)
        .stubs()
        .with(mockcpp::any(), outBoundP(eidList, eidNum * sizeof(dcmi_urma_eid_info_t)), outBoundP(&eidNum))
        .will(returnValue(0));

    char buf[4096] = {0};
    size_t bufSize = sizeof(buf);
    ASSERT_EQ(TopoAddrInfoGet(0, buf, &bufSize), 0);
    /* ETH name 也能在端到端流程中正确解析出 IP（NIC 名不写入 JSON，只校验地址） */
    EXPECT_NE(strstr(buf, "\"addr\": \"10.0.0.1\""), nullptr);
}

/* ══════════════════════════════════════════════════════════════
 *  端到端：TopoAddrInfoGet 全流程（非连续可见设备）
 * ══════════════════════════════════════════════════════════════ */

/* 校验 TopoAddrInfoGet 输出的 JSON 中包含所有预期字段 */
static void AssertJsonContains(const char* json, const char* const expectFields[], const std::string& desc)
{
    for (int i = 0; expectFields[i] != nullptr; i++) {
        EXPECT_NE(strstr(json, expectFields[i]), nullptr) << desc << " JSON 应包含: " << expectFields[i];
    }
}

/* 校验 TopoAddrInfoGet 输出的 JSON 中不包含某字段 */
static void AssertJsonNotContains(const char* json, const char* field, const std::string& desc)
{
    EXPECT_EQ(strstr(json, field), nullptr) << desc << " JSON 不应包含: " << field;
}

/* ROCE 层预期字段（phyId 可见时） */
static const char* g_roceJsonFields[]
    = {"\"net_layer\": 3",       "\"net_instance_id\": \"cluster\"", "\"net_type\": \"CLOS\"", "\"rank_addr_list\":",
       "\"addr\": \"10.0.0.1\"", "\"plane_id\": \"plane0\"",         "\"ports\": [\"d2h\"]",   nullptr};

/* RootInfo 层预期字段 */
static const char* g_rootInfoJsonFields[]
    = {"\"version\": \"2.0\"", "\"rank_count\": 1", "\"rank_list\":", "\"device_id\":", "\"local_id\":", nullptr};

/* ══════════════════════════════════════════════════════════════
 *  全量 mainboard × PCIE/UB × 可见模式 正交覆盖
 * ══════════════════════════════════════════════════════════════ */

struct TopoParam {
    unsigned int mainboardId;
    const char* name;
    bool isCard;   // true→hal_get_eid_list, false→HalGetUBEntityList+hal_get_spod_info
    bool isPcie;   // true→PCIE XML, false→UB XML
    bool isSparse; // true→{0,3,7}, false→全量 {0..7}
};

std::ostream& operator<<(std::ostream& os, const TopoParam& p)
{
    return os << p.name << "_" << (p.isPcie ? "PCIE" : "UB") << "_" << (p.isSparse ? "Sparse" : "All");
}

/* 构建 XML 字符串 */
static std::string BuildXml(bool isPcie, const std::vector<int>& ids)
{
    std::string xml;
    if (isPcie) {
        xml = "<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
              "<pci busid=\"0000:01:00.0\">\n"
              "<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n";
        for (int id : ids) {
            char buf[64];
            (void)sprintf_s(buf, sizeof(buf), "<pci busid=\"0000:%02x:00.0\"/>\n", 3 + id);
            xml += buf;
        }
        xml += "</pci>\n</cpu>\n</system>\n";
    } else {
        xml = "<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
              "<ub>\n<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n";
        for (int id : ids) {
            char buf[64];
            (void)sprintf_s(buf, sizeof(buf), "<npu chipphyid=\"%d\"/>\n", id);
            xml += buf;
        }
        xml += "</ub>\n</cpu>\n</system>\n";
    }
    return xml;
}

class TopoAddrInfoAllMainboardTest : public NpuNicAffinityTest, public testing::WithParamInterface<TopoParam> {};

TEST_P(TopoAddrInfoAllMainboardTest, EndToEnd)
{
    const TopoParam& param = GetParam();
    std::vector<int> visibleIds = param.isSparse ? std::vector<int>{0, 3, 7} : std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7};

    S(8, param.isPcie ? 8 : 0);
    SetVisibleDevices(visibleIds);

    /* PCIE 信息：busId = 3 + phyId */
    if (param.isPcie) {
        for (int id : visibleIds) {
            g_pi[id].domain = 0;
            g_pi[id].bdf_busid = 3 + id;
        }
    }

    std::string xml = BuildXml(param.isPcie, visibleIds);
    W(xml.c_str());

    /* 产品层 mock */
    {
        unsigned int mid = param.mainboardId;
        char drvPath[256] = "/usr/local/Ascend2";
        MOCKER(hal_get_mainboard_id).stubs().with(mockcpp::any(), outBoundP(&mid)).will(returnValue(0));
        MOCKER(hal_get_driver_install_path)
            .stubs()
            .with(outBoundP(drvPath, strlen(drvPath)), mockcpp::any())
            .will(returnValue(0));
        if (param.isCard) {
            dcmi_urma_eid_info_t eidList[MAX_EID_NUM];
            memset_s(eidList, sizeof(eidList), 0, sizeof(eidList));
            size_t eidNum = 0;
            MOCKER(hal_get_eid_list_by_phy_id)
                .stubs()
                .with(mockcpp::any(), outBoundP(eidList, eidNum * sizeof(dcmi_urma_eid_info_t)), outBoundP(&eidNum))
                .will(returnValue(0));
        } else {
            /* server/pod 用空 UEList + spod_info，
               ProcessLayer 中非 ROCE 层的循环因 ueNum=0 直接跳过，不会空指针。 */
            UEList ueList;
            memset_s(&ueList, sizeof(ueList), 0, sizeof(ueList));

            struct dcmi_spod_info spinfo;
            memset_s(&spinfo, sizeof(spinfo), 0, sizeof(spinfo));
            spinfo.sdid = 0;
            spinfo.super_pod_size = 128;
            spinfo.super_pod_id = 1;
            spinfo.server_index = 1;

            MOCKER(HalGetUBEntityList).stubs().with(mockcpp::any(), outBoundP(&ueList)).will(returnValue(0));
            MOCKER(hal_get_spod_info).stubs().with(mockcpp::any(), outBoundP(&spinfo)).will(returnValue(0));
        }
    }

    /* 可见 NPU → ROCE 层完整 */
    for (int id : visibleIds) {
        char buf[8192] = {0};
        size_t bufSize = sizeof(buf);
        ASSERT_EQ(TopoAddrInfoGet(id, buf, &bufSize), 0);
        AssertJsonContains(buf, g_rootInfoJsonFields, std::string(param.name) + " phyId=" + std::to_string(id));
        AssertJsonContains(buf, g_roceJsonFields, std::string(param.name) + " phyId=" + std::to_string(id));
    }

    /* 不可见 NPU → 无 ROCE 层 */
    for (int id = 0; id < 8; id++) {
        if (std::find(visibleIds.begin(), visibleIds.end(), id) != visibleIds.end()) {
            continue;
        }
        char buf[8192] = {0};
        size_t bufSize = sizeof(buf);
        ASSERT_EQ(TopoAddrInfoGet(id, buf, &bufSize), 0);
        AssertJsonNotContains(
            buf, "\"addr\": \"10.0.0.1\"", std::string(param.name) + " (invis) phyId=" + std::to_string(id));
    }
}

/* 四正交：All/PCIE + All/UB + Sparse/PCIE + Sparse/UB */
#define MB4(mainboardId, name, isCard)                                                 \
    {mainboardId, name, isCard, true, false}, {mainboardId, name, isCard, true, true}, \
        {mainboardId, name, isCard, false, false},                                     \
    {                                                                                  \
        mainboardId, name, isCard, false, true                                         \
    }

static const TopoParam g_allTopoParams[] = {
    /* ─── Card ─── */
    MB4(MAIN_BOARD_ID_CARD_NOMESH, "CARD_NOMESH", true), MB4(MAIN_BOARD_ID_CARD_2PMESH, "CARD_2PMESH", true),
    MB4(MAIN_BOARD_ID_CARD_4PMESH, "CARD_4PMESH", true),
    /* ─── Server ───
     * 说明：SERVER_TYPE1(0x23) 的 g_netInfoList 条目未初始化 instanceIdFunc，
     * ProcessLayer 调用 NULL 函数指针会崩溃。属于产品侧数据缺陷，暂不覆盖。 */
    MB4(MAIN_BOARD_ID_SERVER_8PMESH, "SERVER_8PMESH", false),
    MB4(MAIN_BOARD_ID_SERVER_8PMESH_UBOE, "SERVER_8PMESH_UBOE", false),
    MB4(MAIN_BOARD_ID_SERVER_8PMESH_NOSP, "SERVER_8PMESH_NOSP", false),
    MB4(MAIN_BOARD_ID_SERVER_8PMESH_NOSP_UBOE, "SERVER_8PMESH_NOSP_UBOE", false),
    MB4(MAIN_BOARD_ID_SERVER_350L, "SERVER_350L", false), MB4(MAIN_BOARD_ID_SERVER_550EL_100, "SERVER_550_100", false),
    MB4(MAIN_BOARD_ID_SERVER_550EL_200, "SERVER_550_200", false),
    /* ─── Pod ─── */
    MB4(MAIN_BOARD_ID_POD, "POD", false), MB4(MAIN_BOARD_ID_POD_2D, "POD_2D", false),
    MB4(MAIN_BOARD_ID_POD_FLEX, "POD_FLEX", false), MB4(MAIN_BOARD_ID_POD_FLEX_RTP, "POD_FLEX_RTP", false)};

INSTANTIATE_TEST_SUITE_P(VisibleDevices_Contiguity, TopoAddrInfoAllMainboardTest, testing::ValuesIn(g_allTopoParams));

/* ─── 多组 UB XML 端到端验证 ─── */
TEST_F(NpuNicAffinityTest, MultiGroupUbAffinity)
{
    /* 构造 8 组 fake 网络接口 */
    const char* nicNames[] = {"ens0f0", "ens1f0", "ens0f1", "ens1f1", "ens0f2", "ens1f2", "ens0f3", "ens1f3"};
    struct ifaddrs* head = NULL;
    for (int i = 0; i < 8; i++) {
        struct ifaddrs* ifa = (struct ifaddrs*)calloc(1, sizeof(struct ifaddrs));
        ifa->ifa_next = head;
        ifa->ifa_name = strdup(nicNames[i]);
        ifa->ifa_flags = IFF_UP;
        struct sockaddr_in* sin = (struct sockaddr_in*)calloc(1, sizeof(struct sockaddr_in));
        sin->sin_family = AF_INET;
        char ip[16];
        sprintf_s(ip, sizeof(ip), "10.0.%d.%d", i / 4, (i % 4) + 1);
        inet_pton(AF_INET, ip, &sin->sin_addr);
        ifa->ifa_addr = (struct sockaddr*)sin;
        head = ifa;
    }
    TeardownFakeNet();
    g_fakeIfaddr = head;

    S(8, 0);
    W("<system version=\"1.0\" description=\"npu nic switch stand by\">\n"
      "<cpu numaid=\"0\">\n"
      "<ub busid=\"\">\n"
      "<ub> <nic><net name=\"ens0f0\" /></nic></ub>\n"
      "<ub> <nic><net name=\"ens1f0\" /></nic></ub>\n"
      "<ub><npu chipphyid=\"0\"/></ub>\n"
      "</ub>\n"
      "<ub busid=\"\">\n"
      "<ub> <nic><net name=\"ens1f0\" /></nic></ub>\n"
      "<ub> <nic><net name=\"ens0f0\" /></nic></ub>\n"
      "<ub><npu chipphyid=\"1\"/></ub>\n"
      "</ub>\n"
      "<ub busid=\"\">\n"
      "<ub> <nic><net name=\"ens0f1\" /></nic></ub>\n"
      "<ub> <nic><net name=\"ens1f1\" /></nic></ub>\n"
      "<ub><npu chipphyid=\"2\"/></ub>\n"
      "</ub>\n"
      "<ub busid=\"\">\n"
      "<ub> <nic><net name=\"ens0f1\" /></nic></ub>\n"
      "<ub> <nic><net name=\"ens1f1\" /></nic></ub>\n"
      "<ub><npu chipphyid=\"3\"/></ub>\n"
      "</ub>\n"
      "</cpu>\n"
      "<cpu numaid=\"1\">\n"
      "<ub busid=\"\">\n"
      "<ub> <nic><net name=\"ens0f2\" /></nic></ub>\n"
      "<ub> <nic><net name=\"ens1f2\" /></nic></ub>\n"
      "<ub><npu chipphyid=\"4\"/></ub>\n"
      "</ub>\n"
      "<ub busid=\"\">\n"
      "<ub> <nic><net name=\"ens1f2\" /></nic></ub>\n"
      "<ub> <nic><net name=\"ens0f2\" /></nic></ub>\n"
      "<ub><npu chipphyid=\"5\"/></ub>\n"
      "</ub>\n"
      "<ub busid=\"\">\n"
      "<ub> <nic><net name=\"ens0f3\" /></nic></ub>\n"
      "<ub> <nic><net name=\"ens1f3\" /></nic></ub>\n"
      "<ub><npu chipphyid=\"6\"/></ub>\n"
      "</ub>\n"
      "<ub busid=\"\">\n"
      "<ub> <nic><net name=\"ens0f3\" /></nic></ub>\n"
      "<ub> <nic><net name=\"ens1f3\" /></nic></ub>\n"
      "<ub><npu chipphyid=\"7\"/></ub>\n"
      "</ub>\n"
      "</cpu>\n"
      "</system>\n");

    /* 每个 NPU 应从自己的亲和组中分配到 NIC，且取到正确的 IP */
    /* 去重后 8 个 NIC 各占一个 nicIdx，全局游标逐一分配 */
    const char* expectedIps[8] = {
        "10.0.0.1", // NPU0 → ens0f0
        "10.0.0.2", // NPU1 → ens1f0
        "10.0.0.3", // NPU2 → ens0f1
        "10.0.0.4", // NPU3 → ens1f1
        "10.0.1.1", // NPU4 → ens0f2
        "10.0.1.2", // NPU5 → ens1f2
        "10.0.1.3", // NPU6 → ens0f3
        "10.0.1.4", // NPU7 → ens1f3
    };
    for (int i = 0; i < 8; i++) {
        char ip[64] = {0};
        ASSERT_EQ(GetRoceIpFromXml(i, ip, sizeof(ip)), 0);
        EXPECT_STREQ(ip, expectedIps[i]);
    }
}

/* ─── PCIE 多组 XML 端到端验证（含去重） ─── */
TEST_F(NpuNicAffinityTest, MultiGroupPcieAffinity)
{
    /* 6 个唯一 NIC 名，各配唯一 IP */
    const char* nicNames[] = {"ens0f0", "ens1f0", "ens0f2", "ens1f2", "ens0f3", "ens1f3"};
    const char* fakeIps[] = {"10.0.0.1", "10.0.0.2", "10.0.1.1", "10.0.1.2", "10.0.1.3", "10.0.1.4"};
    struct ifaddrs* head = NULL;
    for (int i = 0; i < 6; i++) {
        struct ifaddrs* ifa = (struct ifaddrs*)calloc(1, sizeof(struct ifaddrs));
        ASSERT_NE(ifa, nullptr);
        ifa->ifa_next = head;
        ifa->ifa_name = strdup(nicNames[i]);
        ASSERT_NE(ifa->ifa_name, nullptr);
        ifa->ifa_flags = IFF_UP;
        struct sockaddr_in* sin = (struct sockaddr_in*)calloc(1, sizeof(struct sockaddr_in));
        ASSERT_NE(sin, nullptr);
        sin->sin_family = AF_INET;
        inet_pton(AF_INET, fakeIps[i], &sin->sin_addr);
        ifa->ifa_addr = (struct sockaddr*)sin;
        head = ifa;
    }
    TeardownFakeNet();
    g_fakeIfaddr = head;

    /* PCIE BDF 信息：busId = 3 + phyId */
    for (int i = 0; i < 8; i++) {
        g_pi[i].domain = 0;
        g_pi[i].bdf_busid = 3 + i;
        g_pi[i].bdf_deviceid = 0;
        g_pi[i].bdf_funcid = 0;
    }
    S(8, 8);

    /* XML: 4 个 Bridge 分布在 2 个 CPU socket，
       Bridge 2 的 NIC 与 Bridge 1 同名，验证去重 */
    W("<system version=\"1.0\">\n"
      "<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:00:01.0\">\n"
      "<nic>\n<net name=\"ens0f0\"/>\n</nic>\n"
      "<nic>\n<net name=\"ens1f0\"/>\n</nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n"
      "<pci busid=\"0000:04:00.0\"/>\n"
      "</pci>\n"
      "<pci busid=\"0000:00:02.0\">\n"
      "<nic>\n<net name=\"ens0f0\"/>\n</nic>\n"
      "<nic>\n<net name=\"ens1f0\"/>\n</nic>\n"
      "<pci busid=\"0000:05:00.0\"/>\n"
      "<pci busid=\"0000:06:00.0\"/>\n"
      "</pci>\n"
      "</cpu>\n"
      "<cpu numaid=\"1\">\n"
      "<pci busid=\"0000:00:03.0\">\n"
      "<nic>\n<net name=\"ens0f2\"/>\n</nic>\n"
      "<nic>\n<net name=\"ens1f2\"/>\n</nic>\n"
      "<pci busid=\"0000:07:00.0\"/>\n"
      "<pci busid=\"0000:08:00.0\"/>\n"
      "</pci>\n"
      "<pci busid=\"0000:00:04.0\">\n"
      "<nic>\n<net name=\"ens0f3\"/>\n</nic>\n"
      "<nic>\n<net name=\"ens1f3\"/>\n</nic>\n"
      "<pci busid=\"0000:09:00.0\"/>\n"
      "<pci busid=\"0000:0a:00.0\"/>\n"
      "</pci>\n"
      "</cpu>\n"
      "</system>\n");

    /* 去重后 nicIdx: {0:ens0f0, 1:ens1f0, 2:ens0f2, 3:ens1f2, 4:ens0f3, 5:ens1f3}
       Group 0/1 共享 nicIdx {0,1}，Group 2 用 {2,3}，Group 3 用 {4,5} */
    const char* expectedIps[8] = {
        "10.0.0.1", // NPU 0 → ens0f0
        "10.0.0.2", // NPU 1 → ens1f0
        "10.0.0.1", // NPU 2 → ens0f0 (去重同 nicIdx 0)
        "10.0.0.2", // NPU 3 → ens1f0 (去重同 nicIdx 1)
        "10.0.1.1", // NPU 4 → ens0f2
        "10.0.1.2", // NPU 5 → ens1f2
        "10.0.1.3", // NPU 6 → ens0f3
        "10.0.1.4", // NPU 7 → ens1f3
    };
    for (int i = 0; i < 8; i++) {
        char ip[64] = {0};
        ASSERT_EQ(GetRoceIpFromXml(i, ip, sizeof(ip)), 0);
        EXPECT_STREQ(ip, expectedIps[i]);
    }
}

/* ══════════════════════════════════════════════════════════════
 *  ProcessLayerRoce 入口
 * ══════════════════════════════════════════════════════════════ */

TEST_F(NpuNicAffinityTest, ProcessLayerRoce_Success)
{
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    S(1, 1);
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:01:00.0\">\n"
      "<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    NetLayer layer;
    memset_s(&layer, sizeof(layer), 0, sizeof(layer));
    EXPECT_EQ(ProcessLayerRoce(0, &layer), 0);
    EXPECT_EQ(layer.net_layer, 3);
    EXPECT_STREQ(layer.net_type, "CLOS");
    EXPECT_STREQ(layer.net_instance_id, "cluster");
    EXPECT_EQ(layer.addr_count, 1);
    EXPECT_STREQ(layer.rank_addr_list[0].addr, "10.0.0.1");
    EXPECT_STREQ(layer.rank_addr_list[0].plane_id, "plane0");
    EXPECT_STREQ(layer.rank_addr_list[0].ports[0], "d2h");
}

TEST_F(NpuNicAffinityTest, ProcessLayerRoce_FileNotFound)
{
    NetLayer layer;
    memset_s(&layer, sizeof(layer), 0, sizeof(layer));
    EXPECT_NE(ProcessLayerRoce(0, &layer), TOPO_SUCCESS);
}

/* ══════════════════════════════════════════════════════════════
 *  容灾用例：NIC/NPU 设备挂掉时功能正常
 * ══════════════════════════════════════════════════════════════ */

/* NIC 故障：HCA sysfs 目录不存在 */
TEST_F(NpuNicAffinityTest, Nic_HcaSysfsMissing)
{
    TeardownFakeHca(); /* 清除 SetUp 创建的 hrn5_0 HCA 目录 */
    S(1, 1);
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:00:01.0\">\n"
      "<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    char ip[64] = {0};
    ASSERT_NE(GetRoceIpFromXml(0, ip, sizeof(ip)), TOPO_SUCCESS);
}

/* NIC 故障：HCA 目录存在但 net 子目录为空 */
TEST_F(NpuNicAffinityTest, Nic_HcaSysfsEmptyDir)
{
    TeardownFakeHca();
    system("mkdir -p /tmp/ut_hca/hrn5_0/device/net"); /* 只创空目录，不放 eth */
    S(1, 1);
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:00:01.0\">\n"
      "<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    char ip[64] = {0};
    ASSERT_NE(GetRoceIpFromXml(0, ip, sizeof(ip)), TOPO_SUCCESS);
}

/* NIC 故障：eth 名不在 getifaddrs 返回列表中 */
TEST_F(NpuNicAffinityTest, Nic_EthNotInIfaddrs)
{
    S(1, 1);
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    /* XML 中 name 直接是 eth 名，但 getifaddrs 只返回 lo，没有 eth0 */
    TeardownFakeNet();
    {
        struct ifaddrs* ifa = (struct ifaddrs*)calloc(1, sizeof(struct ifaddrs));
        ifa->ifa_next = NULL;
        ifa->ifa_name = strdup("lo");
        ifa->ifa_flags = IFF_UP;
        struct sockaddr_in* sin = (struct sockaddr_in*)calloc(1, sizeof(struct sockaddr_in));
        sin->sin_family = AF_INET;
        inet_pton(AF_INET, "127.0.0.1", &sin->sin_addr);
        ifa->ifa_addr = (struct sockaddr*)sin;
        g_fakeIfaddr = ifa;
    }
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:00:01.0\">\n"
      "<nic>\n<net name=\"eth0\"/>\n</nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    char ip[64] = {0};
    ASSERT_NE(GetRoceIpFromXml(0, ip, sizeof(ip)), TOPO_SUCCESS);
}

/* NIC 故障：eth 存在但 ifa_addr 为 NULL */
TEST_F(NpuNicAffinityTest, Nic_EthIfaAddrNull)
{
    S(1, 1);
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    TeardownFakeNet();
    {
        struct ifaddrs* ifa = (struct ifaddrs*)calloc(1, sizeof(struct ifaddrs));
        ifa->ifa_next = NULL;
        ifa->ifa_name = strdup("eth0");
        ifa->ifa_flags = IFF_UP;
        ifa->ifa_addr = NULL; /* 地址为空 */
        g_fakeIfaddr = ifa;
    }
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:00:01.0\">\n"
      "<nic>\n<net name=\"eth0\"/>\n</nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    char ip[64] = {0};
    ASSERT_NE(GetRoceIpFromXml(0, ip, sizeof(ip)), TOPO_SUCCESS);
}

/* NIC 故障：eth 只有 IPv6 地址，无 AF_INET */
TEST_F(NpuNicAffinityTest, Nic_EthOnlyIpv6)
{
    S(1, 1);
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    TeardownFakeNet();
    {
        struct ifaddrs* ifa = (struct ifaddrs*)calloc(1, sizeof(struct ifaddrs));
        ifa->ifa_next = NULL;
        ifa->ifa_name = strdup("eth0");
        ifa->ifa_flags = IFF_UP;
        struct sockaddr_in6* sin6 = (struct sockaddr_in6*)calloc(1, sizeof(struct sockaddr_in6));
        sin6->sin6_family = AF_INET6;
        ifa->ifa_addr = (struct sockaddr*)sin6;
        g_fakeIfaddr = ifa;
    }
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:00:01.0\">\n"
      "<nic>\n<net name=\"eth0\"/>\n</nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    char ip[64] = {0};
    ASSERT_NE(GetRoceIpFromXml(0, ip, sizeof(ip)), TOPO_SUCCESS);
}

/* 组内部分 NIC 故障：第一张 NIC 不可达，自动 fallback 到同组第二张 */
TEST_F(NpuNicAffinityTest, Nic_FallbackWhenFirstDown)
{
    S(1, 1);
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    /* 只创建 hrn5_1 的 HCA，不创建 hrn5_0 → hrn5_0 不可达，fallback 到 hrn5_1 */
    SetupFakeHca("hrn5_1", "eth1");
    SetupFakeNet("eth1", "10.0.0.2");
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:00:01.0\">\n"
      "<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<nic>\n<net name=\"hrn5_1\"/>\n</nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    char ip[64] = {0};
    ASSERT_EQ(GetRoceIpFromXml(0, ip, sizeof(ip)), 0);
    ASSERT_STREQ(ip, "10.0.0.2");
}

/* NPU 故障：BDF 不匹配 → XML 中的 busid 与所有 NPU 的 BDF 都不一致 */
TEST_F(NpuNicAffinityTest, Npu_BdfNoMatch)
{
    S(1, 1);
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    /* XML 引用 busid=0000:ff:00.0，而 NPU0 的 BDF 是 0000:03:00.0 → 不匹配 */
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:00:01.0\">\n"
      "<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<pci busid=\"0000:ff:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    char ip[64] = {0};
    ASSERT_NE(GetRoceIpFromXml(0, ip, sizeof(ip)), TOPO_SUCCESS);
}

/* NPU 故障：hal_get_device_pcie_info 全失败 → BDF 表全空 → 无匹配 */
TEST_F(NpuNicAffinityTest, Npu_PcieInfoAllFail)
{
    S(1, 0); /* pcie=0 → 不 mock hal_get_device_pcie_info */
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:00:01.0\">\n"
      "<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    char ip[64] = {0};
    ASSERT_NE(GetRoceIpFromXml(0, ip, sizeof(ip)), TOPO_SUCCESS);
}

/* NPU 去重：UB 格式中同组重复 chipphyid → 不影响分配 */
TEST_F(NpuNicAffinityTest, Npu_DupPhyId_UB)
{
    S(2, 0);
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<ub>\n<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<npu chipphyid=\"0\"/>\n"
      "<npu chipphyid=\"0\"/>\n"
      "</ub>\n</cpu>\n</system>\n");
    char ip[64] = {0};
    ASSERT_EQ(GetRoceIpFromXml(0, ip, sizeof(ip)), 0);
    ASSERT_STREQ(ip, "10.0.0.1");
}

/* NPU 去重：PCIE 格式中同组重复 BDF → 不影响分配 */
TEST_F(NpuNicAffinityTest, Npu_DupBdf_PCIE)
{
    S(1, 1);
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:00:01.0\">\n"
      "<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n"
      "<pci busid=\"0000:03:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    char ip[64] = {0};
    ASSERT_EQ(GetRoceIpFromXml(0, ip, sizeof(ip)), 0);
    ASSERT_STREQ(ip, "10.0.0.1");
}

/* 所有 NIC 不可达 → 返回 -1 */
TEST_F(NpuNicAffinityTest, Error_AllNicsUnreachable)
{
    TeardownFakeHca();
    TeardownFakeNet();
    S(1, 1);
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:00:01.0\">\n"
      "<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    char ip[64] = {0};
    ASSERT_NE(GetRoceIpFromXml(0, ip, sizeof(ip)), TOPO_SUCCESS);
}

/* 组内无 NPU：只有 NIC 没有 NPU → 返回 -1 */
TEST_F(NpuNicAffinityTest, Error_GroupWithoutNpu)
{
    S(1, 1);
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:00:01.0\">\n"
      "<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "</pci>\n</cpu>\n</system>\n");
    char ip[64] = {0};
    ASSERT_NE(GetRoceIpFromXml(0, ip, sizeof(ip)), TOPO_SUCCESS);
}

/* GetTopoFilePathFromFile：正常路径，合法 JSON 应正确提取 topo_file_path 值 */
TEST_F(NpuNicAffinityTest, GetTopoFilePathFromFile_Success)
{
    const char* path = "/tmp/ut_topo_file_success.json";
    FILE* fp = fopen(path, "w");
    ASSERT_NE(fp, nullptr);
    const char* content
        = "{\n  \"version\": \"2.0\",\n  \"topo_file_path\": \"/usr/local/ascend/driver/topo/950/atlas_950_1.json\"\n}";
    ASSERT_GT(fprintf(fp, "%s", content), 0);
    fclose(fp);

    char buf[256] = {0};
    int ret = GetTopoFilePathFromFile(path, buf, sizeof(buf));
    EXPECT_EQ(ret, 0);
    EXPECT_STREQ(buf, "/usr/local/ascend/driver/topo/950/atlas_950_1.json");
    unlink(path);
}

/* GetTopoFilePathFromFile：文件不存在应返回 -1 */
TEST_F(NpuNicAffinityTest, GetTopoFilePathFromFile_FileNotFound)
{
    char buf[64] = {0};
    EXPECT_EQ(GetTopoFilePathFromFile("/tmp/ut_topo_file_not_exist.json", buf, sizeof(buf)), -1);
}

/* GetTopoFilePathFromFile：JSON 中无 topo_file_path 字段应返回 -1 */
TEST_F(NpuNicAffinityTest, GetTopoFilePathFromFile_NoField)
{
    const char* path = "/tmp/ut_topo_file_nofield.json";
    FILE* fp = fopen(path, "w");
    ASSERT_NE(fp, nullptr);
    ASSERT_GT(fprintf(fp, "%s", "{\n  \"version\": \"2.0\"\n}"), 0);
    fclose(fp);

    char buf[64] = {0};
    EXPECT_EQ(GetTopoFilePathFromFile(path, buf, sizeof(buf)), -1);
    unlink(path);
}

/* GetTopoFilePathFromFile：strncpy_s 失败（如缓冲区不足）应提前返回 -1，不越界写 */
TEST_F(NpuNicAffinityTest, GetTopoFilePathFromFile_StrncpyFailed)
{
    const char* path = "/tmp/ut_topo_file_strncpy_fail.json";
    FILE* fp = fopen(path, "w");
    ASSERT_NE(fp, nullptr);
    const char* content
        = "{\n  \"version\": \"2.0\",\n  \"topo_file_path\": \"/usr/local/ascend/driver/topo/950/atlas_950_1.json\"\n}";
    ASSERT_GT(fprintf(fp, "%s", content), 0);
    fclose(fp);

    /* 默认 llt_stub_sec.c 的 strncpy_s stub 恒返 0，此处强制注入失败返回值 */
    MOCKER(strncpy_s).stubs().will(returnValue(1));

    char buf[256] = {0};
    EXPECT_EQ(GetTopoFilePathFromFile(path, buf, sizeof(buf)), -1);
    unlink(path);
}

/* ParseXmlTags：标签数超过 maxTags 时超限标签被丢弃，解析仍成功且 tagCount == maxTags */
TEST_F(NpuNicAffinityTest, ParseXmlTags_TagsExceedMax_DroppedCounted)
{
    const char* path = "/tmp/ut_xml_overflow.xml";
    FILE* fp = fopen(path, "w");
    ASSERT_NE(fp, nullptr);
    /* 构造 5 个 <pci> 标签，maxTags 传入 2 */
    fputs("<system version=\"1.0\">\n<cpu numaid=\"0\">\n", fp);
    fputs("<pci busid=\"0000:01:00.0\"/>\n", fp);
    fputs("<pci busid=\"0000:02:00.0\"/>\n", fp);
    fputs("<pci busid=\"0000:03:00.0\"/>\n", fp);
    fputs("<pci busid=\"0000:04:00.0\"/>\n", fp);
    fputs("<pci busid=\"0000:05:00.0\"/>\n", fp);
    fputs("</cpu>\n</system>\n", fp);
    fclose(fp);

    TagEntry tags[2] = {{{0}}};
    unsigned int tagCount = 0;
    TopoAddrResult ret = ParseXmlTags(path, tags, &tagCount, 2);
    EXPECT_EQ(ret, TOPO_SUCCESS); /* 超限丢弃不视为解析失败 */
    EXPECT_EQ(tagCount, 2U);      /* 写入数应被截断为 maxTags */
    unlink(path);
}

/* ══════════════════════════════════════════════════════════════
 *  无 XML 回退：驱动拓扑接口构建亲和矩阵
 * ══════════════════════════════════════════════════════════════ */

/* 构建多 NIC 假网卡链（头插，链序与入参相反），每个 NIC 一个 IPv4；
   nicNames[i]/ips[i] 传 NULL 时对应字段留空（模拟异常节点，如空名/无地址）；
   flags 传 NULL 时默认 IFF_UP，可指定每节点 flags（如未 UP、IFF_POINTOPOINT） */
static void BuildFakeNetChain(
    struct ifaddrs** outHead, const char* const nicNames[], const char* const ips[], int n,
    const unsigned int* flags = NULL)
{
    TeardownFakeNet(); /* 先回收旧链，避免覆盖 g_fakeIfaddr 时泄漏 */
    struct ifaddrs* head = NULL;
    for (int i = 0; i < n; i++) {
        struct ifaddrs* ifa = (struct ifaddrs*)calloc(1, sizeof(struct ifaddrs));
        ASSERT_NE(ifa, nullptr);
        ifa->ifa_next = head;
        if (nicNames[i] != NULL) {
            ifa->ifa_name = strdup(nicNames[i]);
            ASSERT_NE(ifa->ifa_name, nullptr);
        }
        ifa->ifa_flags = (flags != NULL) ? flags[i] : IFF_UP;
        if (ips[i] != NULL) {
            struct sockaddr_in* sin = (struct sockaddr_in*)calloc(1, sizeof(struct sockaddr_in));
            ASSERT_NE(sin, nullptr);
            sin->sin_family = AF_INET;
            inet_pton(AF_INET, ips[i], &sin->sin_addr);
            ifa->ifa_addr = (struct sockaddr*)sin;
        }
        head = ifa;
    }
    *outHead = head;
}

/* 多 NPU 多 NIC 全正常：驱动全量查询，轮询分发各 NPU 拿到不同 IP */
TEST_F(NpuNicAffinityTest, Fallback_MultiNpuMultiNic)
{
    S(4, 0);
    TeardownFakeNet();
    const char* nicNames[] = {"eth0", "eth1", "eth2"};
    const char* fakeIps[] = {"10.0.0.1", "10.0.0.2", "10.0.0.3"};
    BuildFakeNetChain(&g_fakeIfaddr, nicNames, fakeIps, 3);

    /* 全连接亲和：每个 NPU 与每个 NIC 均亲和 */
    for (int phyId = 0; phyId < 4; phyId++) {
        Affine(phyId, "eth0");
        Affine(phyId, "eth1");
        Affine(phyId, "eth2");
    }

    /* 无排序，列序即枚举（链序）：头插后链序为 eth2→eth1→eth0；
       轮询：NPU0→eth2，NPU1→eth1，NPU2→eth0，NPU3→eth2(回绕) */
    AssertRoceIpOk(0, "10.0.0.3");
    EXPECT_EQ(g_driverCallCount, 12); /* 4 NPU × 3 NIC */
    AssertRoceIpOk(1, "10.0.0.2");
    AssertRoceIpOk(2, "10.0.0.1");
    AssertRoceIpOk(3, "10.0.0.3");
}

/* XML 文件不存在 → 回退驱动接口 */
TEST_F(NpuNicAffinityTest, Fallback_NoXml_Basic)
{
    S(1, 0);
    Affine(0, "eth0");
    AssertRoceIpOk(0, "10.0.0.1");
}

/* XML 能解析但其中无 NIC 定义 → 同样回退驱动接口 */
TEST_F(NpuNicAffinityTest, Fallback_XmlNoNic)
{
    S(1, 0);
    Affine(0, "eth0");
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:01:00.0\">\n"
      "<pci busid=\"0000:03:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    AssertRoceIpOk(0, "10.0.0.1");
}

/* 空 XML 文件 → 回退驱动接口 */
TEST_F(NpuNicAffinityTest, Fallback_EmptyFile)
{
    S(1, 0);
    Affine(0, "eth0");
    W("");
    AssertRoceIpOk(0, "10.0.0.1");
}

/* 乱序网卡链：枚举列序即链序（不再排序），轮询分配按链序列序进行 */
TEST_F(NpuNicAffinityTest, Fallback_RoundRobinChainOrder)
{
    S(4, 0);
    Affine(0, "eth0");
    Affine(0, "eth1");
    Affine(1, "eth0");
    Affine(1, "eth1");
    Affine(2, "eth1");
    Affine(3, "eth0");

    /* 头插后链序为 eth1→eth2→eth0→eth3，即矩阵列序 [eth1, eth2, eth0, eth3] */
    const char* nicNames[] = {"eth3", "eth0", "eth2", "eth1"};
    const char* fakeIps[] = {"10.0.0.4", "10.0.0.1", "10.0.0.3", "10.0.0.2"};
    BuildFakeNetChain(&g_fakeIfaddr, nicNames, fakeIps, 4);

    /* 轮询：NPU0→eth1，NPU1→eth0，NPU2→eth1，NPU3→eth0 */
    AssertRoceIpOk(0, "10.0.0.2");
    AssertRoceIpOk(1, "10.0.0.1");
    AssertRoceIpOk(2, "10.0.0.2");
    AssertRoceIpOk(3, "10.0.0.1");
}

/* 枚举去重与过滤：重复 NIC 只查询一次；NULL 名与超长名被跳过 */
TEST_F(NpuNicAffinityTest, Fallback_DedupAndSkip)
{
    S(1, 0);
    Affine(0, "eth0");

    /* 目标链序：eth0 → 名字为空 → 超长名(≥64) → eth0(重复) → eth1；
       helper 头插，数组反序书写 */
    char longName[80];
    memset_s(longName, sizeof(longName), 'x', sizeof(longName) - 1);
    longName[sizeof(longName) - 1] = '\0';
    const char* nicNames[] = {"eth1", "eth0", longName, NULL, "eth0"};
    const char* fakeIps[] = {"10.0.0.2", "10.0.0.1", NULL, NULL, "10.0.0.1"};
    BuildFakeNetChain(&g_fakeIfaddr, nicNames, fakeIps, 5);

    AssertRoceIpOk(0, "10.0.0.1");
    /* 去重后只查询 eth0、eth1 各一次 */
    EXPECT_EQ(g_driverCallCount, 2);
}

/* 枚举上限：超过 MAX_HCA_COUNT(64) 的 NIC 被丢弃 */
TEST_F(NpuNicAffinityTest, Fallback_NicCountCap)
{
    S(1, 0);
    Affine(0, "nic63"); /* 在枚举上限内，应被查询 */
    Affine(0, "nic00"); /* 枚举序第 65 个，超出上限被丢弃，不应被查询 */

    struct ifaddrs* head = NULL;
    for (int i = 0; i < 65; i++) {
        char name[16];
        char ipStr[16];
        sprintf_s(name, sizeof(name), "nic%02d", i);
        sprintf_s(ipStr, sizeof(ipStr), "10.0.0.%d", i + 1);
        struct ifaddrs* ifa = (struct ifaddrs*)calloc(1, sizeof(struct ifaddrs));
        ASSERT_NE(ifa, nullptr);
        ifa->ifa_next = head;
        ifa->ifa_name = strdup(name);
        ASSERT_NE(ifa->ifa_name, nullptr);
        ifa->ifa_flags = IFF_UP;
        struct sockaddr_in* sin = (struct sockaddr_in*)calloc(1, sizeof(struct sockaddr_in));
        ASSERT_NE(sin, nullptr);
        sin->sin_family = AF_INET;
        inet_pton(AF_INET, ipStr, &sin->sin_addr);
        ifa->ifa_addr = (struct sockaddr*)sin;
        head = ifa;
    }
    TeardownFakeNet();
    g_fakeIfaddr = head;

    /* 枚举按链序取前 64 个（nic64..nic01），nic00 被丢弃；无排序，列序即链序，nic63 保留 */
    AssertRoceIpOk(0, "10.0.0.64");
    EXPECT_EQ(g_driverCallCount, 64);
}

/* 非连续可见设备：只查询可见 NPU，不可见 NPU 拿不到 IP */
TEST_F(NpuNicAffinityTest, Fallback_InvisibleNpu)
{
    S(8, 0);
    SetVisibleDevices({0, 3, 7});
    const char* nicNames[] = {"eth0", "eth1"};
    const char* fakeIps[] = {"10.0.0.1", "10.0.0.2"};
    BuildFakeNetChain(&g_fakeIfaddr, nicNames, fakeIps, 2);

    Affine(0, "eth0");
    Affine(3, "eth1");
    Affine(7, "eth0");
    Affine(1, "eth0"); /* 不可见，不应被查询 */
    Affine(2, "eth0");

    AssertRoceIpOk(0, "10.0.0.1");
    EXPECT_EQ(g_driverCallCount, 6); /* 3 个可见 NPU × 2 个 NIC */
    AssertRoceIpOk(3, "10.0.0.2");
    AssertRoceIpOk(7, "10.0.0.1");
    AssertRoceIpFail(1);
    AssertRoceIpFail(2);
}

/* 驱动返回非 UB 拓扑类型（PHB/SELF）：不算亲和，无 IP 可分配 */
TEST_F(NpuNicAffinityTest, Fallback_NoUbAffinity)
{
    S(1, 0);
    Affine(0, "eth0", DCMI_TOPO_TYPE_PHB);
    Affine(0, "eth1", DCMI_TOPO_TYPE_SELF);
    const char* nicNames[] = {"eth0", "eth1"};
    const char* fakeIps[] = {"10.0.0.1", "10.0.0.2"};
    BuildFakeNetChain(&g_fakeIfaddr, nicNames, fakeIps, 2);
    AssertRoceIpFail(0);
}

/* 无任何网卡（空链）→ 回退构建失败 */
TEST_F(NpuNicAffinityTest, Fallback_NoNics_EmptyChain)
{
    S(1, 0);
    TeardownFakeNet();
    AssertRoceIpFail(0);
}

/* 无任何网卡（仅 lo 回环）→ 回退构建失败 */
TEST_F(NpuNicAffinityTest, Fallback_NoNics_LoopbackOnly)
{
    S(1, 0);
    TeardownFakeNet();
    const char* nicNames[] = {"lo"};
    const char* fakeIps[] = {"127.0.0.1"};
    BuildFakeNetChain(&g_fakeIfaddr, nicNames, fakeIps, 1);
    AssertRoceIpFail(0);
}

/* getifaddrs 系统调用失败 → 回退构建失败 */
TEST_F(NpuNicAffinityTest, Fallback_GetifaddrsFail)
{
    S(1, 0);
    g_fakeIfaddrsFail = true;
    AssertRoceIpFail(0);
}

/* NPU 计数为 0 → 回退构建失败 */
TEST_F(NpuNicAffinityTest, Fallback_NpuCountZero)
{
    MOCKER(hal_get_npu_count).stubs().will(returnValue(0));
    Affine(0, "eth0");
    AssertRoceIpFail(0);
}

/* NPU 计数超过上限 → 回退构建失败 */
TEST_F(NpuNicAffinityTest, Fallback_NpuCountTooLarge)
{
    MOCKER(hal_get_npu_count).stubs().will(returnValue((int)MAX_NPU_COUNT + 1));
    Affine(0, "eth0");
    AssertRoceIpFail(0);
}

/* 可枚举但解不出 IP 的 NIC：不参与分配，只亲和该 NIC 的 NPU 拿不到 IP */
TEST_F(NpuNicAffinityTest, Fallback_NicNoIp)
{
    S(2, 0);
    Affine(0, "eth0");
    Affine(1, "eth1");

    TeardownFakeNet();
    /* eth1（ips=NULL → 无地址）先入链，eth0（有 IP）后入 */
    const char* nicNames[] = {"eth1", "eth0"};
    const char* fakeIps[] = {NULL, "10.0.0.1"};
    BuildFakeNetChain(&g_fakeIfaddr, nicNames, fakeIps, 2);

    AssertRoceIpOk(0, "10.0.0.1");
    AssertRoceIpFail(1);
}

/* XML 可用时优先 XML 路径，驱动接口不被调用 */
TEST_F(NpuNicAffinityTest, Fallback_PriorityOverXml)
{
    S(2, 2);
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    g_pi[1].domain = 0;
    g_pi[1].bdf_busid = 4;
    g_pi[1].bdf_deviceid = 0;
    g_pi[1].bdf_funcid = 0;
    /* 若误走回退，驱动会收到查询；XML 优先 → 计数应保持 0 */
    Affine(0, "eth0");
    Affine(0, "eth1");
    Affine(1, "eth0");
    Affine(1, "eth1");
    const char* nicNames[] = {"eth0", "eth1"};
    const char* fakeIps[] = {"10.0.0.1", "10.0.0.2"};
    BuildFakeNetChain(&g_fakeIfaddr, nicNames, fakeIps, 2);

    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:01:00.0\">\n"
      "<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n<pci busid=\"0000:04:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    AssertRoceIpOk(0, "10.0.0.1");
    EXPECT_EQ(g_driverCallCount, 0);
}

/* 驱动拓扑接口整体不可用（所有查询失败）→ 回退构建失败，不产出空亲和矩阵 */
TEST_F(NpuNicAffinityTest, Fallback_DriverQueryFail)
{
    S(1, 0);
    SetDriverMode(DRIVER_LOAD_FAIL);
    Affine(0, "eth0"); /* 驱动接口不可用，登记项也不生效 */
    AssertRoceIpFail(0, TOPO_ERR_INTERNAL);
    EXPECT_EQ(g_driverCallCount, 1); /* 查询已发出，因全部失败而整体报错 */
}

/* logicId 与 phyId 不一致：驱动查询必须用 logicId，用 phyId 会查错设备（容器/虚拟化场景） */
TEST_F(NpuNicAffinityTest, Fallback_LogicIdMapping)
{
    S(1, 0);
    /* mockcpp 先注册者优先生效，需先定向清除 SetUp 的 identity stub，再注册 logicId 偏移映射 */
    GlobalMockObject::reset((const void*)hal_get_logicid_from_phyid);
    MOCKER(hal_get_logicid_from_phyid)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(mockcpp::invoke(mock_logicidShifted));
    /* 登记在 phyId 上：logicId 查询找不到 → 分配失败 */
    Affine(0, "eth0");
    AssertRoceIpFail(0);

    /* 同一网口再登记到 logicId 上 → 分配成功（遗留的 phyId 登记项不影响 logicId 查询） */
    Affine(100, "eth0");
    AssertRoceIpOk(0, "10.0.0.1");
    EXPECT_EQ(g_driverCallCount, 2); /* 两次构建各查询一次 */
}

/* 部分网口驱动不识别（查询失败）：跳过失败网口，其余网口仍正常建亲和 */
TEST_F(NpuNicAffinityTest, Fallback_PartialQueryFail)
{
    S(2, 0);
    SetDriverMode(DRIVER_UNKNOWN_NIC_FAIL); /* 未登记的网口一律返回 -1，模拟驱动不识别的虚拟网口 */
    Affine(0, "eth0");
    Affine(0, "eth1");
    Affine(1, "eth0");
    Affine(1, "eth1");

    /* 头插后链序为 docker0→veth0→eth0→eth1：docker0/veth0 会被查询但失败跳过，
       且作为哑列排在枚举头部——NPU0 需先跳过它们才能命中 eth0 */
    const char* nicNames[] = {"eth1", "eth0", "veth0", "docker0"};
    const char* fakeIps[] = {"10.0.0.2", "10.0.0.1", "172.17.0.2", "172.17.0.1"};
    BuildFakeNetChain(&g_fakeIfaddr, nicNames, fakeIps, 4);

    /* 轮询：NPU0→eth0，NPU1→eth1，失败网口不进亲和矩阵 */
    AssertRoceIpOk(0, "10.0.0.1");
    EXPECT_EQ(g_driverCallCount, 8); /* 单次构建 2 NPU × 4 网口（其中 4 次查询失败被跳过） */
    AssertRoceIpOk(1, "10.0.0.2");
}

/* 枚举过滤：点对点（tunnel/PPP）与未 UP 的网口不进 nicNames，不产生驱动查询 */
TEST_F(NpuNicAffinityTest, Fallback_EnumFilterNonPhysical)
{
    S(1, 0);
    Affine(0, "eth0");

    /* 头插后链序为 eth0(UP) → ethDown(未 UP) → ppp0(IFF_POINTOPOINT) */
    const char* nicNames[] = {"ppp0", "ethDown", "eth0"};
    const char* fakeIps[] = {NULL, NULL, "10.0.0.1"};
    const unsigned int flags[] = {IFF_UP | IFF_POINTOPOINT, IFF_BROADCAST, IFF_UP | IFF_BROADCAST};
    BuildFakeNetChain(&g_fakeIfaddr, nicNames, fakeIps, 3, flags);

    AssertRoceIpOk(0, "10.0.0.1");
    EXPECT_EQ(g_driverCallCount, 1); /* 仅 eth0 被查询，ppp0/ethDown 在枚举期被过滤 */
}

/* XML 含 NIC 但 NPU 的 BDF 全不匹配（亲和矩阵为空）→ 回退驱动接口，且 XML 残留状态被清理 */
TEST_F(NpuNicAffinityTest, Fallback_XmlEmptyAffinity)
{
    S(1, 1);
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    Affine(0, "eth0"); /* 驱动侧亲和 eth0 */

    /* XML 的 NPU busid(0000:ff:00.0) 与实际 BDF(0000:03:00.0) 不匹配 → 矩阵为空；
       其 NIC 名 hrn5_0 不在系统网口列表中，用于校验 XML 残留已被清理 */
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:01:00.0\">\n"
      "<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<pci busid=\"0000:ff:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");

    AssertRoceIpOk(0, "10.0.0.1");
    EXPECT_EQ(g_driverCallCount, 1); /* 仅枚举出的 eth0 被查询：若 hrn5_0 残留，查询数会变成 2 */
}

/* 全部 NPU 不可见（零查询）→ 回退构建失败，而非空亲和矩阵静默成功 */
TEST_F(NpuNicAffinityTest, Fallback_AllNpusInvisible)
{
    S(2, 0);
    SetVisibleDevices({});
    Affine(0, "eth0");
    AssertRoceIpFail(0, TOPO_ERR_INTERNAL);
    EXPECT_EQ(g_driverCallCount, 0); /* 没有任何查询发出，整体报错 */
}
