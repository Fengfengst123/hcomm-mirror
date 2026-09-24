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

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>
#include <gtest/gtest.h>

#include "db_sim_runner_common.h"
#include "db_sim_runner_db.h"
#include "db_sim_sqlite_db.h"

extern uint64_t g_cur_server_key;

namespace sim {
aclError GetServerByKey(uint64_t serverKey, sim::Server &server);
uint32_t GetCubeCoreCount(uint64_t deviceId);
} // namespace sim

namespace {
const std::string kTestDbPath = "/tmp/test_sim_runner_common.db";

void CleanUpDb() {
    std::remove(kTestDbPath.c_str());
    std::remove((kTestDbPath + "-wal").c_str());
    std::remove((kTestDbPath + "-shm").c_str());
}

void SetupTestData() {
    CleanUpDb();
    g_cur_comm_key = 0;
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

    device.logic_id = 1;
    device.physical_id = 1;
    RunnerDB::Add<sim::Device>(device);

    g_cur_server_key = 1;
}
} // namespace

class SimRunnerCommonTest : public testing::Test {
  protected:
    void SetUp() override { SetupTestData(); }

    void TearDown() override { CleanUpDb(); }
};

TEST_F(SimRunnerCommonTest, GetDeviceByLogicId_WhenDeviceExists_ReturnSuccess) {
    sim::Device device{};
    auto ret = sim::GetDeviceByLogicId(0, device);

    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_EQ(device.logic_id, 0);
    EXPECT_EQ(device.physical_id, 0);
    EXPECT_EQ(device.server_id, 1);
}

TEST_F(SimRunnerCommonTest,
       GetDeviceByLogicId_WhenDeviceNotExists_ReturnError) {
    sim::Device device{};
    auto ret = sim::GetDeviceByLogicId(999, device);

    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

TEST_F(SimRunnerCommonTest,
       GetDeviceByPhysicalId_WhenDeviceExists_ReturnSuccess) {
    sim::Device device{};
    auto ret = sim::GetDeviceByPhysicalId(0, device);

    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_EQ(device.physical_id, 0);
    EXPECT_EQ(device.logic_id, 0);
}

TEST_F(SimRunnerCommonTest,
       GetDeviceByPhysicalId_WhenDeviceNotExists_ReturnError) {
    sim::Device device{};
    auto ret = sim::GetDeviceByPhysicalId(999, device);

    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

TEST_F(SimRunnerCommonTest,
       UpdateDeviceLogicId_WhenValidParams_UpdateSuccessfully) {
    auto ret = sim::UpdateDeviceLogicId(1, 0, 100, 0);

    EXPECT_EQ(ret, ACL_SUCCESS);

    sim::Device device{};
    auto getRet = sim::GetDeviceByPhysicalId(0, device);
    EXPECT_EQ(getRet, ACL_SUCCESS);
    EXPECT_EQ(device.logic_id, 100);
    EXPECT_EQ(device.status, 1);
}

TEST_F(SimRunnerCommonTest,
       UpdateDeviceLogicId_WhenServerKeyInvalid_ReturnError) {
    auto ret = sim::UpdateDeviceLogicId(999, 0, 100, 0);

    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

TEST_F(SimRunnerCommonTest,
       UpdateDeviceLogicId_WhenPhyDevIdInvalid_ReturnError) {
    auto ret = sim::UpdateDeviceLogicId(1, 999, 100, 0);

    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

TEST_F(SimRunnerCommonTest,
       UpdateSuperDeviceId_WhenValidParams_UpdateSuccessfully) {
    auto ret = sim::UpdateSuperDeviceId(0, 200);

    EXPECT_EQ(ret, ACL_SUCCESS);

    sim::Device device{};
    auto getRet = sim::GetDeviceByLogicId(0, device);
    EXPECT_EQ(getRet, ACL_SUCCESS);
    EXPECT_EQ(device.super_device_id, 200);
}

TEST_F(SimRunnerCommonTest,
       UpdateSuperDeviceId_WhenLogicIdInvalid_ReturnError) {
    auto ret = sim::UpdateSuperDeviceId(999, 200);

    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

TEST_F(SimRunnerCommonTest, GetServerByKey_WhenServerExists_ReturnSuccess) {
    sim::Server server{};
    auto ret = RunnerDB::GetOneByPred<sim::Server>(
        [](const sim::Server &s) { return s.id == 1; });

    EXPECT_TRUE(ret.second);
    EXPECT_EQ(ret.first.pod_id, 100);
}

TEST_F(SimRunnerCommonTest, GetServerByKey_WhenServerNotExists_ReturnError) {
    auto ret = RunnerDB::GetOneByPred<sim::Server>(
        [](const sim::Server &s) { return s.id == 999; });

    EXPECT_FALSE(ret.second);
}

TEST_F(SimRunnerCommonTest,
       GetDeviceByServerKeyAndPhysicalId_WhenValid_ReturnSuccess) {
    auto ret = RunnerDB::GetOneByPred<sim::Device>([](const sim::Device &d) {
        return d.server_id == 1 && d.physical_id == 0;
    });

    EXPECT_TRUE(ret.second);
    EXPECT_EQ(ret.first.physical_id, 0);
    EXPECT_EQ(ret.first.server_id, 1);
}

TEST_F(SimRunnerCommonTest,
       GetDeviceByServerKeyAndPhysicalId_WhenInvalidServerKey_ReturnError) {
    auto ret = RunnerDB::GetOneByPred<sim::Device>([](const sim::Device &d) {
        return d.server_id == 999 && d.physical_id == 0;
    });

    EXPECT_FALSE(ret.second);
}

TEST_F(SimRunnerCommonTest,
       GetDeviceByServerKeyAndPhysicalId_WhenInvalidPhysicalId_ReturnError) {
    auto ret = RunnerDB::GetOneByPred<sim::Device>([](const sim::Device &d) {
        return d.server_id == 1 && d.physical_id == 999;
    });

    EXPECT_FALSE(ret.second);
}

TEST_F(SimRunnerCommonTest, GetServerByKey_Found_ReturnSuccess) {
    sim::Server server{};
    auto ret = sim::GetServerByKey(1, server);
    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_EQ(server.pod_id, 100);
}

TEST_F(SimRunnerCommonTest, GetServerByKey_NotFound_ReturnError) {
    sim::Server server{};
    auto ret = sim::GetServerByKey(999, server);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

class SimRunnerCommonCcuTest : public testing::Test {
  protected:
    void SetUp() override {
        SetupTestData();

        sim::Device device{};
        auto ret = sim::GetDeviceByPhysicalId(0, device);
        ASSERT_EQ(ret, ACL_SUCCESS);

        sim::Ccu ccu{};
        ccu.device_id = device.id;
        ccu.die_id = 0;
        ccu.status = 1;
        RunnerDB::Add<sim::Ccu>(ccu);
    }

    void TearDown() override { CleanUpDb(); }
};

TEST_F(SimRunnerCommonCcuTest,
       GetCcuFromDeviceByDieId_WhenValid_ReturnSuccess) {
    sim::Ccu ccu{};
    auto ret = sim::GetCcuFromDeviceByDieId(1, 0, ccu);

    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_EQ(ccu.die_id, 0);
    EXPECT_EQ(ccu.device_id, 1);
}

TEST_F(SimRunnerCommonCcuTest,
       GetCcuFromDeviceByDieId_WhenNotFound_ReturnError) {
    sim::Ccu ccu{};
    auto ret = sim::GetCcuFromDeviceByDieId(1, 99, ccu);

    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

class SimRunnerCommonContextTest : public testing::Test {
  protected:
    void SetUp() override {
        SetupTestData();

        sim::Device device{};
        auto ret = sim::GetDeviceByPhysicalId(0, device);
        ASSERT_EQ(ret, ACL_SUCCESS);

        sim::Context ctx{};
        ctx.device_id = device.id;
        ctx.is_default = 1;
        ctx.run_id = 1;
        RunnerDB::Add<sim::Context>(ctx);
    }

    void TearDown() override { CleanUpDb(); }
};

TEST_F(SimRunnerCommonContextTest, GetContextByDevId_WhenValid_ReturnSuccess) {
    sim::Device device{};
    auto ret = sim::GetDeviceByPhysicalId(0, device);
    ASSERT_EQ(ret, ACL_SUCCESS);

    sim::Context ctx{};
    auto getRet = sim::GetContextByDevId(device.id, ctx);

    EXPECT_EQ(getRet, ACL_SUCCESS);
    EXPECT_EQ(ctx.device_id, device.id);
    EXPECT_EQ(ctx.is_default, 1);
}

TEST_F(SimRunnerCommonContextTest, GetContextByDevId_WhenNotFound_ReturnError) {
    sim::Context ctx{};
    auto ret = sim::GetContextByDevId(999, ctx);

    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

class SimRunnerCommonPortTest : public testing::Test {
  protected:
    void SetUp() override {
        SetupTestData();

        sim::Device device{};
        auto ret = sim::GetDeviceByPhysicalId(0, device);
        ASSERT_EQ(ret, ACL_SUCCESS);

        sim::Port port{};
        port.device_id = device.id;
        snprintf(port.name, sizeof(port.name), "eth0");
        port.status = 1;
        RunnerDB::Add<sim::Port>(port);

        sim::Port port2{};
        port2.device_id = device.id;
        snprintf(port2.name, sizeof(port2.name), "eth1");
        port2.status = 1;
        RunnerDB::Add<sim::Port>(port2);
    }

    void TearDown() override { CleanUpDb(); }
};

TEST_F(SimRunnerCommonPortTest, GetPortByName_WhenValid_ReturnSuccess) {
    sim::Port port{};
    auto ret = sim::GetPortByName(1, 0, "eth0", port);

    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_STREQ(port.name, "eth0");
}

TEST_F(SimRunnerCommonPortTest, GetPortByName_WhenNotFound_ReturnError) {
    sim::Port port{};
    auto ret = sim::GetPortByName(1, 0, "nonexistent", port);

    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

class SimRunnerCommonCountTest : public testing::Test {
  protected:
    void SetUp() override {
        SetupTestData();

        sim::Device device{};
        auto ret = sim::GetDeviceByLogicId(0, device);
        ASSERT_EQ(ret, ACL_SUCCESS);

        sim::TaskSchedulerDevice tsDev{};
        tsDev.device_id = device.id;
        tsDev.type = static_cast<uint8_t>(sim::TS_DEV_TYPE_CPU);
        RunnerDB::Add<sim::TaskSchedulerDevice>(tsDev);

        tsDev.type = static_cast<uint8_t>(sim::TS_DEV_TYPE_SCALAR);
        RunnerDB::Add<sim::TaskSchedulerDevice>(tsDev);
    }

    void TearDown() override { CleanUpDb(); }
};

TEST_F(SimRunnerCommonCountTest,
       GetAICpuCount_WhenDeviceExists_ReturnCorrectCount) {
    auto count = sim::GetAICpuCount(0);

    EXPECT_EQ(count, 1);
}

TEST_F(SimRunnerCommonCountTest, GetAICpuCount_WhenDeviceNotExists_ReturnZero) {
    auto count = sim::GetAICpuCount(999);

    EXPECT_EQ(count, 0);
}

TEST_F(SimRunnerCommonCountTest,
       GetAICoreCount_WhenDeviceExists_ReturnCorrectCount) {
    auto count = sim::GetAICoreCount(0);

    EXPECT_GE(count, 1);
}

TEST_F(SimRunnerCommonCountTest,
       GetAICoreCount_WhenDeviceNotExists_ReturnZero) {
    auto count = sim::GetAICoreCount(999);

    EXPECT_EQ(count, 0);
}

class SimRunnerCommonVectorCubeTest : public testing::Test {
  protected:
    void SetUp() override {
        SetupTestData();

        sim::Device device{};
        auto ret = sim::GetDeviceByLogicId(0, device);
        ASSERT_EQ(ret, ACL_SUCCESS);

        sim::TaskSchedulerDevice tsDev{};
        tsDev.device_id = device.id;
        tsDev.type = static_cast<uint8_t>(sim::TS_DEV_TYPE_SCALAR);
        uint64_t tsId = RunnerDB::Add<sim::TaskSchedulerDevice>(tsDev);

        sim::ComputeDie die{};
        die.ts_id = tsId;
        die.type = static_cast<uint8_t>(sim::COMPUTE_TYPE_VECTOR);
        RunnerDB::Add<sim::ComputeDie>(die);

        die.type = static_cast<uint8_t>(sim::COMPUTE_TYPE_CUBE);
        RunnerDB::Add<sim::ComputeDie>(die);
    }

    void TearDown() override { CleanUpDb(); }
};

TEST_F(SimRunnerCommonVectorCubeTest,
       GetVectorCoreCount_WhenDeviceExists_ReturnCorrectCount) {
    auto count = sim::GetVectorCoreCount(0);

    EXPECT_GE(count, 1);
}

TEST_F(SimRunnerCommonVectorCubeTest,
       GetVectorCoreCount_WhenDeviceNotExists_ReturnZero) {
    auto count = sim::GetVectorCoreCount(999);

    EXPECT_EQ(count, 0);
}

TEST_F(SimRunnerCommonVectorCubeTest,
       GetCubeCoreCount_WhenDeviceExists_ReturnCorrectCount) {
    auto count = sim::GetCubeCoreCount(0);
    EXPECT_GE(count, 1);
}

TEST_F(SimRunnerCommonVectorCubeTest,
       GetCubeCoreCount_WhenDeviceNotExists_ReturnZero) {
    auto count = sim::GetCubeCoreCount(999);
    EXPECT_EQ(count, 0);
}

class SimRunnerCommonRankEndpointTest : public testing::Test {
  protected:
    void SetUp() override {
        SetupTestData();

        sim::Device device{};
        auto ret = sim::GetDeviceByLogicId(0, device);
        ASSERT_EQ(ret, ACL_SUCCESS);

        sim::Communicator comm{};
        std::strncpy(comm.comm_id, "test_comm", sizeof(comm.comm_id) - 1);
        comm.rank_size = 8;
        comm.rank_id = 5;
        comm.device_id = device.id;
        g_cur_comm_key = RunnerDB::Add<sim::Communicator>(comm);
        ASSERT_NE(g_cur_comm_key, 0U);
    }

    void TearDown() override { CleanUpDb(); }
};

TEST_F(SimRunnerCommonRankEndpointTest,
       GetCommRankByDeviceId_WhenExists_ReturnsRankId) {
    const auto comm = RunnerDB::GetById<sim::Communicator>(g_cur_comm_key);
    ASSERT_TRUE(comm.has_value());

    uint32_t rankId = UINT32_MAX;
    ASSERT_TRUE(
        sim::GetCommRankByDeviceId(g_cur_comm_key, comm->device_id, rankId));
    EXPECT_EQ(rankId, 5);
}

TEST_F(SimRunnerCommonRankEndpointTest,
       GetCommRankByDeviceId_WhenNotExists_ReturnsFalse) {
    uint32_t rankId = UINT32_MAX;
    EXPECT_FALSE(sim::GetCommRankByDeviceId(g_cur_comm_key, 999, rankId));
    EXPECT_EQ(rankId, UINT32_MAX);
}

TEST_F(SimRunnerCommonTest, CommunicatorMemberIdResolvesPeersInSameDomain) {
    uint64_t firstCommId = 0;
    uint64_t secondCommId = 0;
    ASSERT_TRUE(
        sim::GetOrInsertCommunicator("shared_domain", 2, 0, 1, 0, firstCommId));
    ASSERT_TRUE(sim::GetOrInsertCommunicator("shared_domain", 2, 1, 2, 0,
                                             secondCommId));
    EXPECT_NE(firstCommId, 0);
    EXPECT_NE(firstCommId, secondCommId);

    sim::Device device{};
    ASSERT_EQ(sim::GetDeviceByCommRank(firstCommId, 1, device), ACL_SUCCESS);
    EXPECT_EQ(device.id, 2);
    ASSERT_EQ(sim::GetDeviceByCommRank(secondCommId, 0, device), ACL_SUCCESS);
    EXPECT_EQ(device.id, 1);

    std::vector<sim::CommunicatorMemberInfo> members;
    ASSERT_TRUE(sim::GetCommunicatorMembers(secondCommId, members));
    ASSERT_EQ(members.size(), 2U);
    EXPECT_TRUE(
        std::any_of(members.begin(), members.end(),
                    [&firstCommId](const sim::CommunicatorMemberInfo &member) {
                        return member.memberId == firstCommId &&
                               member.rankId == 0 && member.deviceId == 1;
                    }));
    EXPECT_TRUE(
        std::any_of(members.begin(), members.end(),
                    [&secondCommId](const sim::CommunicatorMemberInfo &member) {
                        return member.memberId == secondCommId &&
                               member.rankId == 1 && member.deviceId == 2;
                    }));

    std::string commName;
    ASSERT_TRUE(sim::GetCommunicatorName(secondCommId, commName));
    EXPECT_EQ(commName, "shared_domain");
}

TEST_F(SimRunnerCommonTest,
       CommunicatorMemberIdSeparatesSameNameDomainsByHash) {
    uint64_t firstDomainMemberId = 0;
    uint64_t secondDomainMemberId = 0;
    uint64_t unusedMemberId = 0;
    ASSERT_TRUE(sim::GetOrInsertCommunicator("shared_sub_domain", 2, 0, 1, 100,
                                             firstDomainMemberId));
    ASSERT_TRUE(sim::GetOrInsertCommunicator("shared_sub_domain", 2, 1, 2, 100,
                                             unusedMemberId));
    ASSERT_TRUE(sim::GetOrInsertCommunicator("shared_sub_domain", 2, 0, 2, 200,
                                             unusedMemberId));
    ASSERT_TRUE(sim::GetOrInsertCommunicator("shared_sub_domain", 2, 1, 1, 200,
                                             secondDomainMemberId));

    sim::Device device{};
    ASSERT_EQ(sim::GetDeviceByCommRank(firstDomainMemberId, 1, device),
              ACL_SUCCESS);
    EXPECT_EQ(device.id, 2U);
    ASSERT_EQ(sim::GetDeviceByCommRank(secondDomainMemberId, 1, device),
              ACL_SUCCESS);
    EXPECT_EQ(device.id, 1U);

    std::vector<sim::CommunicatorMemberInfo> members;
    ASSERT_TRUE(sim::GetCommunicatorMembers(firstDomainMemberId, members));
    ASSERT_EQ(members.size(), 2U);
    ASSERT_TRUE(sim::GetCommunicatorMembers(secondDomainMemberId, members));
    ASSERT_EQ(members.size(), 2U);
}

TEST_F(SimRunnerCommonTest, CommunicatorDestroyBarrierWaitsForEveryRank) {
    uint64_t firstCommId = 0;
    uint64_t secondCommId = 0;
    ASSERT_TRUE(sim::GetOrInsertCommunicator("destroy_barrier", 2, 0, 1, 0,
                                             firstCommId));
    ASSERT_TRUE(sim::GetOrInsertCommunicator("destroy_barrier", 2, 1, 2, 0,
                                             secondCommId));

    auto firstWait = std::async(std::launch::async, [firstCommId] {
        return sim::WaitCommunicatorDestroyReady(firstCommId);
    });
    EXPECT_EQ(firstWait.wait_for(std::chrono::milliseconds(200)),
              std::future_status::timeout);

    EXPECT_TRUE(sim::WaitCommunicatorDestroyReady(secondCommId));
    EXPECT_EQ(firstWait.wait_for(std::chrono::seconds(2)),
              std::future_status::ready);
    EXPECT_TRUE(firstWait.get());
}

TEST_F(
    SimRunnerCommonTest,
    CommunicatorNameLengthIsValidatedWithoutChangingSameNameDomainSemantics) {
    const std::string validName(127, 'a');
    uint64_t commId = 0;
    ASSERT_TRUE(
        sim::GetOrInsertCommunicator(validName.c_str(), 1, 0, 1, 0, commId));
    EXPECT_NE(commId, 0U);

    const std::string tooLongName(128, 'b');
    EXPECT_FALSE(
        sim::GetOrInsertCommunicator(tooLongName.c_str(), 1, 0, 1, 0, commId));
    EXPECT_FALSE(sim::WaitCommunicatorReady(tooLongName.c_str(), 0, 1));
}

TEST_F(SimRunnerCommonRankEndpointTest,
       GetEndPointByIpAddr_WhenNotExists_ReturnError) {
    sim::EndPoint ep{};
    auto ret = sim::GetEndPointByIpAddr("192.168.99.99", ep);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

class SimRunnerCommonFullPortTest : public testing::Test {
  protected:
    void SetUp() override {
        SetupTestData();

        sim::Device device{};
        auto ret = sim::GetDeviceByPhysicalId(0, device);
        ASSERT_EQ(ret, ACL_SUCCESS);

        sim::Port port{};
        port.device_id = device.id;
        port.die_id = 0;
        snprintf(port.name, sizeof(port.name), "eth0");
        port.status = 1;
        portId_ = RunnerDB::Add<sim::Port>(port);

        sim::Ccu ccu{};
        ccu.device_id = device.id;
        ccu.die_id = 0;
        ccu.status = 1;
        ccuId_ = RunnerDB::Add<sim::Ccu>(ccu);

        snprintf(port.name, sizeof(port.name), "ccu_eth0");
        RunnerDB::Add<sim::Port>(port);
    }

    void TearDown() override { CleanUpDb(); }

    uint64_t portId_{0};
    uint64_t ccuId_{0};
};

TEST_F(SimRunnerCommonFullPortTest, GetPortById_WhenValid_ReturnSuccess) {
    sim::Port port{};
    auto ret = sim::GetPortById(portId_, port);

    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_EQ(port.id, portId_);
}

TEST_F(SimRunnerCommonFullPortTest, GetPortById_WhenNotFound_ReturnError) {
    sim::Port port{};
    auto ret = sim::GetPortById(999, port);

    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

TEST_F(SimRunnerCommonFullPortTest, GetPortByName_DeviceNotFound_ReturnError) {
    sim::Port port{};
    auto ret = sim::GetPortByName(999, 0, "eth0", port);

    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}
