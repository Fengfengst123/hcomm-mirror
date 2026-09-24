/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

/**
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * for the full text of the License.
 */

#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <unistd.h>
#include <vector>

#include "db_sim_runner_common.h"
#include "db_sim_runner_db.h"
#include "db_sim_sqlite_db.h"
#include "hccl_device_pub.h"
#include "store_sim_memory_manager.h"

using namespace HcclSim;

extern uint32_t g_rankId;
extern std::map<uint32_t, uint32_t> sqTailMap;

namespace {
const std::string kTestDbPath = "/tmp/test_hccl_device_pub.db";

void CleanUpDb() {
    std::remove(kTestDbPath.c_str());
    std::remove((kTestDbPath + "-wal").c_str());
    std::remove((kTestDbPath + "-shm").c_str());
}

void SetupTestData() {
    CleanUpDb();
    sim::SqliteDatabase::SetDbPath(kTestDbPath);
    SimRunnerSqliteDB::Instance().ClearAll();

    sim::Server server{};
    server.pod_id = 100;
    snprintf(server.version, sizeof(server.version), "v1.0");
    RunnerDB::Add<sim::Server>(server);

    sim::Host host{};
    host.server_id = 1;
    snprintf(host.ip_addr, sizeof(host.ip_addr), "192.168.1.100");
    host.arch = 1;
    RunnerDB::Add<sim::Host>(host);

    sim::Device device{};
    device.server_id = 1;
    device.logic_id = 0;
    device.physical_id = 0;
    device.super_device_id = 0;
    RunnerDB::Add<sim::Device>(device);
    g_cur_comm_key = 0;
}
} // namespace

class HcclDevicePubTest : public testing::Test {
  protected:
    void SetUp() override {
        g_rankId = 0;
        sqTailMap.clear();
        SetupTestData();
    }

    void TearDown() override {
        RunnerDB::DeleteAll<sim::EndPoint>();
        RunnerDB::DeleteAll<sim::VirtualMemBlock>();
        RunnerDB::DeleteAll<sim::PhyMemBlock>();
        CleanUpDb();
    }
};

TEST_F(HcclDevicePubTest, SetCurRankId_Valid) {
    uint32_t rankId = 5;
    HcclVmResult result = SetCurRankId(rankId);
    EXPECT_EQ(result, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(HcclDevicePubTest, SetCurRankId_Zero) {
    HcclVmResult result = SetCurRankId(0);
    EXPECT_EQ(result, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(HcclDevicePubTest, SetCurRankId_MaxValue) {
    HcclVmResult result = SetCurRankId(UINT32_MAX);
    EXPECT_EQ(result, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(HcclDevicePubTest, GetCurRankId_AfterSet) {
    uint32_t setRankId = 10;
    SetCurRankId(setRankId);

    uint32_t getRankId = 0;
    HcclVmResult result = GetCurRankId(&getRankId);
    EXPECT_EQ(result, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(getRankId, setRankId);
}

TEST_F(HcclDevicePubTest, GetSqBufferAddr_Valid) {
    uint8_t *sqBuff = nullptr;
    HcclVmResult result = GetSqBufferAddr(&sqBuff);
    EXPECT_EQ(result, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_NE(sqBuff, nullptr);
}

TEST_F(HcclDevicePubTest, GetSqBufferAddr_BufferSize) {
    uint8_t *sqBuff = nullptr;
    GetSqBufferAddr(&sqBuff);
    EXPECT_NE(sqBuff, nullptr);
}

TEST_F(HcclDevicePubTest, UpdatePiValByJettyId_Valid) {
    uint32_t jettyId = 1;
    uint32_t piValue = 100;

    HcclVmResult result = UpdatePiValByJettyId(jettyId, piValue);
    EXPECT_EQ(result, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(HcclDevicePubTest, GetPiValByJettyId_AfterUpdate) {
    uint32_t jettyId = 2;
    uint32_t piValue = 200;

    UpdatePiValByJettyId(jettyId, piValue);

    uint32_t getValue = 0;
    HcclVmResult result = GetPiValByJettyId(jettyId, &getValue);
    EXPECT_EQ(result, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(getValue, piValue);
}

TEST_F(HcclDevicePubTest, GetPiValByJettyId_NotExist) {
    uint32_t jettyId = 9999;
    uint32_t getValue = 0;

    HcclVmResult result = GetPiValByJettyId(jettyId, &getValue);
    EXPECT_EQ(result, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(HcclDevicePubTest, UpdateKfcStatus_ZeroAddr) {
    EXPECT_NO_THROW(UpdateKfcStatus(0));
}

TEST_F(HcclDevicePubTest, UpdateKfcStatus_ValidAddr) {
    constexpr uint32_t kfcBufSize = 4096;
    std::vector<uint8_t> kfcBuf(kfcBufSize, 0);
    uint64_t d2hAddr = reinterpret_cast<uint64_t>(kfcBuf.data());
    EXPECT_NO_THROW(UpdateKfcStatus(d2hAddr));
}

TEST_F(HcclDevicePubTest, RegisterSignalHandler_NoThrow) {
    EXPECT_NO_THROW(RegisterSignalHandler());
}

// ==================== GetSqTail / UpdateSqTail Tests ====================

TEST_F(HcclDevicePubTest, UpdateSqTail_And_GetSqTail) {
    uint32_t sqId = 1;
    uint32_t newTail = 42;

    UpdateSqTail(sqId, newTail);
    uint32_t tail = GetSqTail(sqId);

    EXPECT_EQ(tail, newTail);
}

TEST_F(HcclDevicePubTest, GetSqTail_NotExist) {
    uint32_t sqId = 99;
    uint32_t tail = GetSqTail(sqId);

    EXPECT_EQ(tail, 0);
}

TEST_F(HcclDevicePubTest, UpdateSqTail_Overwrite) {
    uint32_t sqId = 2;
    UpdateSqTail(sqId, 10);
    UpdateSqTail(sqId, 20);

    uint32_t tail = GetSqTail(sqId);
    EXPECT_EQ(tail, 20);
}

// ==================== GetDeviceIdByDevAddr Tests ====================

TEST_F(HcclDevicePubTest, GetDeviceIdByDevAddr_NotFound) {
    uint64_t devAddr = 0x1000;
    uint32_t deviceId = GetDeviceIdByDevAddr(devAddr);

    EXPECT_EQ(deviceId, 0);
}

TEST_F(HcclDevicePubTest, GetDeviceIdByDevAddr_Found) {
    auto devRet = RunnerDB::GetOneByPred<sim::Device>(
        [](const sim::Device &d) { return d.physical_id == 0; });
    ASSERT_TRUE(devRet.second);
    uint64_t deviceId = devRet.first.id;

    sim::PhyMemBlock phyMem{};
    phyMem.device_id = deviceId;
    snprintf(phyMem.name, sizeof(phyMem.name), "test_phy_mem");
    phyMem.size = 4096;
    phyMem.type = 0;
    phyMem.ref_count = 1;
    uint64_t phyMemId = RunnerDB::Add<sim::PhyMemBlock>(phyMem);

    sim::VirtualMemBlock virMem{};
    virMem.start_ptr = 0x10000000;
    virMem.size = 4096;
    virMem.ctx_id = 1;
    virMem.phy_mem_id = phyMemId;
    virMem.owner_pid = 0;
    virMem.src_type = (uint8_t)sim::VIR_MEM_TYPE_DEV;
    virMem.policy = 0;
    RunnerDB::Add<sim::VirtualMemBlock>(virMem);

    sim::Communicator comm{};
    std::strncpy(comm.comm_id, "test_comm", sizeof(comm.comm_id) - 1);
    comm.rank_size = 8;
    comm.rank_id = 5;
    comm.device_id = deviceId;
    g_cur_comm_key = RunnerDB::Add<sim::Communicator>(comm);

    uint64_t devAddr = virMem.start_ptr + 0x100;
    uint32_t resultDeviceId = GetDeviceIdByDevAddr(devAddr);

    EXPECT_EQ(resultDeviceId, deviceId);
}

TEST_F(HcclDevicePubTest, GetDeviceIdByDevAddr_VirMemFound_ButPhyMemNotFound) {
    auto devRet = RunnerDB::GetOneByPred<sim::Device>(
        [](const sim::Device &d) { return d.physical_id == 0; });
    ASSERT_TRUE(devRet.second);
    uint64_t deviceId = devRet.first.id;

    sim::VirtualMemBlock virMem{};
    virMem.start_ptr = 0x20000000;
    virMem.size = 4096;
    virMem.ctx_id = 1;
    virMem.phy_mem_id = 99999;
    virMem.owner_pid = 0;
    virMem.src_type = (uint8_t)sim::VIR_MEM_TYPE_DEV;
    virMem.policy = 0;
    RunnerDB::Add<sim::VirtualMemBlock>(virMem);

    uint64_t devAddr = virMem.start_ptr + 0x100;
    uint32_t resultDeviceId = GetDeviceIdByDevAddr(devAddr);

    EXPECT_EQ(resultDeviceId, 0);
}

TEST_F(HcclDevicePubTest, GetDeviceIdByDevAddr_DeviceNotFound) {
    auto devRet = RunnerDB::GetOneByPred<sim::Device>(
        [](const sim::Device &d) { return d.physical_id == 0; });
    ASSERT_TRUE(devRet.second);
    uint64_t deviceId = devRet.first.id;

    sim::PhyMemBlock phyMem{};
    phyMem.device_id = deviceId;
    snprintf(phyMem.name, sizeof(phyMem.name), "test_phy_mem2");
    phyMem.size = 4096;
    phyMem.type = 0;
    phyMem.ref_count = 1;
    uint64_t phyMemId = RunnerDB::Add<sim::PhyMemBlock>(phyMem);

    sim::VirtualMemBlock virMem{};
    virMem.start_ptr = 0x30000000;
    virMem.size = 4096;
    virMem.ctx_id = 1;
    virMem.phy_mem_id = phyMemId;
    virMem.owner_pid = 0;
    virMem.src_type = (uint8_t)sim::VIR_MEM_TYPE_DEV;
    virMem.policy = 0;
    RunnerDB::Add<sim::VirtualMemBlock>(virMem);

    uint64_t devAddr = virMem.start_ptr + 0x100;
    uint32_t resultDeviceId = GetDeviceIdByDevAddr(devAddr);

    EXPECT_EQ(resultDeviceId, 0);
}

// ==================== GetDeviceIdByIpAddr Tests ====================

TEST_F(HcclDevicePubTest, GetDeviceIdByIpAddr_NotFound) {
    uint32_t deviceId = 0;

    EXPECT_FALSE(GetDeviceIdByIpAddr("192.168.1.999", deviceId));
}

TEST_F(HcclDevicePubTest, GetDeviceIdByIpAddr_Found) {
    auto devRet = RunnerDB::GetOneByPred<sim::Device>(
        [](const sim::Device &d) { return d.physical_id == 0; });
    ASSERT_TRUE(devRet.second);
    uint64_t deviceId = devRet.first.id;

    sim::EndPoint ep{};
    ep.device_id = deviceId;
    snprintf(ep.ip_addr, sizeof(ep.ip_addr), "192.168.1.200");
    RunnerDB::Add<sim::EndPoint>(ep);

    sim::Communicator comm{};
    std::strncpy(comm.comm_id, "test_comm", sizeof(comm.comm_id) - 1);
    comm.rank_size = 8;
    comm.rank_id = 7;
    comm.device_id = deviceId;
    g_cur_comm_key = RunnerDB::Add<sim::Communicator>(comm);

    uint32_t resultDeviceId = 0;

    EXPECT_TRUE(GetDeviceIdByIpAddr("192.168.1.200", resultDeviceId));
    EXPECT_EQ(resultDeviceId, deviceId);
}

TEST_F(HcclDevicePubTest, GetDeviceIdByIpAddr_DeviceNotFound) {
    auto devRet = RunnerDB::GetOneByPred<sim::Device>(
        [](const sim::Device &d) { return d.physical_id == 0; });
    ASSERT_TRUE(devRet.second);
    uint64_t deviceId = devRet.first.id;

    sim::EndPoint ep{};
    ep.device_id = deviceId;
    snprintf(ep.ip_addr, sizeof(ep.ip_addr), "192.168.1.201");
    RunnerDB::Add<sim::EndPoint>(ep);

    uint32_t resultDeviceId = 0;

    EXPECT_TRUE(GetDeviceIdByIpAddr("192.168.1.201", resultDeviceId));
    EXPECT_EQ(resultDeviceId, deviceId);
}

// ==================== GetDevMapperAddrByDevAddr Tests ====================

TEST_F(HcclDevicePubTest, GetDevMapperAddrByDevAddr_NotFound) {
    uint64_t devAddr = 0xDEADBEEF;
    uint64_t realAddr = GetDevMapperAddrByDevAddr(devAddr);

    EXPECT_EQ(realAddr, 0ULL);
}

TEST_F(HcclDevicePubTest, GetDevMapperAddrByDevAddr_FoundButShmNotExist) {
    auto devRet = RunnerDB::GetOneByPred<sim::Device>(
        [](const sim::Device &d) { return d.physical_id == 0; });
    ASSERT_TRUE(devRet.second);
    uint64_t deviceId = devRet.first.id;

    sim::PhyMemBlock phyMem{};
    phyMem.device_id = deviceId;
    snprintf(phyMem.name, sizeof(phyMem.name), "test_real_ptr_mem");
    phyMem.size = 4096;
    phyMem.type = 0;
    phyMem.ref_count = 1;
    uint64_t phyMemId = RunnerDB::Add<sim::PhyMemBlock>(phyMem);

    sim::VirtualMemBlock virMem{};
    virMem.start_ptr = 0x40000000;
    virMem.dev_mapped_ptr = 0x40000000;
    virMem.size = 4096;
    virMem.ctx_id = 1;
    virMem.phy_mem_id = phyMemId;
    virMem.owner_pid = (uint64_t)getppid();
    virMem.src_type = (uint8_t)sim::VIR_MEM_TYPE_DEV;
    virMem.is_dev_access = 0;
    virMem.policy = 0;
    RunnerDB::Add<sim::VirtualMemBlock>(virMem);

    uint64_t devAddr = virMem.dev_mapped_ptr + 0x100;
    uint64_t realAddr = GetDevMapperAddrByDevAddr(devAddr);

    // PhyMem name "test_real_ptr_mem" has no actual shm, AcquireMemByName
    // returns nullptr
    EXPECT_EQ(realAddr, 0ULL);
}

TEST_F(HcclDevicePubTest, GetDevMapperAddrByDevAddr_PhyMemNotFound) {
    auto devRet = RunnerDB::GetOneByPred<sim::Device>(
        [](const sim::Device &d) { return d.physical_id == 0; });
    ASSERT_TRUE(devRet.second);
    uint64_t deviceId = devRet.first.id;

    sim::PhyMemBlock phyMem{};
    phyMem.device_id = deviceId;
    snprintf(phyMem.name, sizeof(phyMem.name), "test_real_ptr_mem2");
    phyMem.size = 4096;
    phyMem.type = 0;
    phyMem.ref_count = 1;
    uint64_t phyMemId = RunnerDB::Add<sim::PhyMemBlock>(phyMem);

    sim::VirtualMemBlock virMem{};
    virMem.start_ptr = 0x50000000;
    virMem.dev_mapped_ptr = 0x50000000;
    virMem.size = 4096;
    virMem.ctx_id = 1;
    virMem.phy_mem_id = phyMemId;
    virMem.owner_pid = (uint64_t)getppid();
    virMem.src_type = (uint8_t)sim::VIR_MEM_TYPE_DEV;
    virMem.is_dev_access = 0;
    virMem.policy = 0;
    RunnerDB::Add<sim::VirtualMemBlock>(virMem);

    // Delete the phyMem so GetById returns empty
    RunnerDB::DeleteAll<sim::PhyMemBlock>();

    uint64_t devAddr = virMem.dev_mapped_ptr + 0x100;
    uint64_t realAddr = GetDevMapperAddrByDevAddr(devAddr);

    EXPECT_EQ(realAddr, 0ULL);
}
