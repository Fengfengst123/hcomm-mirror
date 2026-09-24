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

#include <arpa/inet.h>
#include <cstdio>
#include <cstring>

#include <gtest/gtest.h>

#include "hccl/hccl_types.h"
#include "hcomm/hcomm_res_defs.h"
#include "runnerdb/db_sim_runner_ops.h"
#include "runnerdb/db_sim_sqlite_db.h"

extern uint64_t g_cur_server_key;

extern "C" HcommResult HcommEndpointCreate(const EndpointDesc *endpoint,
                                           EndpointHandle *endpointHandle);

namespace {

const char *const TEST_DB_PATH = "/tmp/test_ctrl_base_resource_stub.db";

uint64_t HandleToId(EndpointHandle handle) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(handle));
}

class CtrlBaseResourceStubTest : public testing::Test {
  protected:
    static void SetUpTestSuite() {
        std::remove(TEST_DB_PATH);
        std::remove("/tmp/test_ctrl_base_resource_stub.db-wal");
        std::remove("/tmp/test_ctrl_base_resource_stub.db-shm");
        sim::SqliteDatabase::SetDbPath(TEST_DB_PATH);

        auto &db = SimRunnerSqliteDB::Instance();
        db.ClearAll();

        sim::Server server{};
        uint64_t serverId = db.Add<sim::Server>(server);
        ASSERT_GT(serverId, 0u);

        sim::Host host{};
        host.server_id = serverId;
        uint64_t hostId = db.Add<sim::Host>(host);
        ASSERT_GT(hostId, 0u);

        g_cur_server_key = serverId;
        sim::Runner runner{};
        ASSERT_TRUE(sim::GetCurrRunnerTls(serverId, runner));
        ASSERT_GT(runner.id, 0u);
    }

    static void TearDownTestSuite() {
        std::remove(TEST_DB_PATH);
        std::remove("/tmp/test_ctrl_base_resource_stub.db-wal");
        std::remove("/tmp/test_ctrl_base_resource_stub.db-shm");
    }

    void SetUp() override {
        auto &db = SimRunnerSqliteDB::Instance();
        ASSERT_TRUE(db.DeleteAll<sim::HcommEndpoint>());
        ASSERT_TRUE(db.DeleteAll<sim::EndPoint>());
        ASSERT_TRUE(db.DeleteAll<sim::Device>());
    }

    uint64_t AddDevice(uint32_t physicalId) {
        sim::Device device{};
        device.server_id = static_cast<uint32_t>(g_cur_server_key);
        device.physical_id = physicalId;
        return SimRunnerSqliteDB::Instance().Add<sim::Device>(device);
    }
};

TEST_F(CtrlBaseResourceStubTest, EndpointCreateRejectsNullAndInvalidArguments) {
    EndpointDesc endpoint{};
    endpoint.protocol = COMM_PROTOCOL_HCCS;
    endpoint.commAddr.type = COMM_ADDR_TYPE_EID;
    endpoint.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    EndpointHandle handle = reinterpret_cast<EndpointHandle>(0x1234);

    EXPECT_EQ(HcommEndpointCreate(nullptr, &handle), HCCL_E_PTR);
    EXPECT_EQ(HcommEndpointCreate(&endpoint, nullptr), HCCL_E_PTR);

    endpoint.loc.locType = ENDPOINT_LOC_TYPE_RESERVED;
    EXPECT_EQ(HcommEndpointCreate(&endpoint, &handle), HCCL_E_PARA);
    EXPECT_EQ(handle, nullptr);

    auto records = SimRunnerSqliteDB::Instance().QueryList<sim::HcommEndpoint>(
        [](const sim::HcommEndpoint &) { return true; });
    EXPECT_TRUE(records.empty());
}

TEST_F(CtrlBaseResourceStubTest,
       EndpointCreateStoresHostDescriptorWithoutSouthBinding) {
    EndpointDesc endpoint{};
    endpoint.protocol = COMM_PROTOCOL_ROCE;
    endpoint.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "192.168.10.8", &endpoint.commAddr.addr), 1);
    endpoint.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    endpoint.loc.host.id = 9u;
    endpoint.raws[0] = 0x5Au;

    EndpointHandle handle = nullptr;
    ASSERT_EQ(HcommEndpointCreate(&endpoint, &handle), HCCL_SUCCESS);
    ASSERT_NE(handle, nullptr);

    auto record = SimRunnerSqliteDB::Instance().Find<sim::HcommEndpoint>(
        HandleToId(handle));
    ASSERT_TRUE(record.has_value());
    EXPECT_GT(record->runner_id, 0u);
    EXPECT_EQ(record->south_endpoint_id, 0u);
    EXPECT_EQ(record->protocol, COMM_PROTOCOL_ROCE);
    EXPECT_EQ(record->addr_type, COMM_ADDR_TYPE_IP_V4);
    EXPECT_EQ(record->loc_type, ENDPOINT_LOC_TYPE_HOST);
    EXPECT_EQ(record->loc[0], 9u);
    EXPECT_EQ(record->extension[0], 0x5Au);
    EXPECT_EQ(record->bind_state, sim::HCOMM_ENDPOINT_NOT_APPLICABLE);
}

TEST_F(CtrlBaseResourceStubTest,
       EndpointCreateBindsDeviceEidToUniqueSouthEndpoint) {
    constexpr uint32_t physicalId = 3u;
    uint64_t deviceId = AddDevice(physicalId);
    ASSERT_GT(deviceId, 0u);

    uint8_t eid[COMM_ADDR_EID_LEN] = {0x11, 0x22, 0x33, 0x44};
    sim::EndPoint southEndpoint{};
    southEndpoint.device_id = static_cast<uint32_t>(deviceId);
    std::memcpy(southEndpoint.eid, eid, sizeof(eid));
    uint64_t southEndpointId =
        SimRunnerSqliteDB::Instance().Add<sim::EndPoint>(southEndpoint);
    ASSERT_GT(southEndpointId, 0u);

    EndpointDesc endpoint{};
    endpoint.protocol = COMM_PROTOCOL_UBC_CTP;
    endpoint.commAddr.type = COMM_ADDR_TYPE_EID;
    std::memcpy(endpoint.commAddr.eid, eid, sizeof(eid));
    endpoint.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    endpoint.loc.device.devPhyId = physicalId;

    EndpointHandle handle = nullptr;
    ASSERT_EQ(HcommEndpointCreate(&endpoint, &handle), HCCL_SUCCESS);

    auto record = SimRunnerSqliteDB::Instance().Find<sim::HcommEndpoint>(
        HandleToId(handle));
    ASSERT_TRUE(record.has_value());
    EXPECT_EQ(record->south_endpoint_id, southEndpointId);
    EXPECT_EQ(record->bind_state, sim::HCOMM_ENDPOINT_BOUND);
}

TEST_F(CtrlBaseResourceStubTest, EndpointCreateBindsCanonicalIpv4Address) {
    constexpr uint32_t physicalId = 4u;
    uint64_t deviceId = AddDevice(physicalId);
    ASSERT_GT(deviceId, 0u);

    sim::EndPoint southEndpoint{};
    southEndpoint.device_id = static_cast<uint32_t>(deviceId);
    std::snprintf(southEndpoint.ip_addr, sizeof(southEndpoint.ip_addr),
                  "10.20.30.40");
    uint64_t southEndpointId =
        SimRunnerSqliteDB::Instance().Add<sim::EndPoint>(southEndpoint);
    ASSERT_GT(southEndpointId, 0u);

    EndpointDesc endpoint{};
    endpoint.protocol = COMM_PROTOCOL_ROCE;
    endpoint.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "10.20.30.40", &endpoint.commAddr.addr), 1);
    endpoint.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    endpoint.loc.device.devPhyId = physicalId;

    EndpointHandle handle = nullptr;
    ASSERT_EQ(HcommEndpointCreate(&endpoint, &handle), HCCL_SUCCESS);

    auto record = SimRunnerSqliteDB::Instance().Find<sim::HcommEndpoint>(
        HandleToId(handle));
    ASSERT_TRUE(record.has_value());
    EXPECT_EQ(record->south_endpoint_id, southEndpointId);
    EXPECT_EQ(record->bind_state, sim::HCOMM_ENDPOINT_BOUND);
}

TEST_F(CtrlBaseResourceStubTest,
       EndpointCreateDoesNotPersistWhenRequiredSouthEndpointIsMissing) {
    constexpr uint32_t physicalId = 5u;
    ASSERT_GT(AddDevice(physicalId), 0u);

    EndpointDesc endpoint{};
    endpoint.protocol = COMM_PROTOCOL_UBOE;
    endpoint.commAddr.type = COMM_ADDR_TYPE_EID;
    endpoint.commAddr.eid[0] = 0x88;
    endpoint.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    endpoint.loc.device.devPhyId = physicalId;

    EndpointHandle handle = reinterpret_cast<EndpointHandle>(0x1234);
    EXPECT_EQ(HcommEndpointCreate(&endpoint, &handle), HCCL_E_NOT_FOUND);
    EXPECT_EQ(handle, nullptr);

    auto records = SimRunnerSqliteDB::Instance().QueryList<sim::HcommEndpoint>(
        [](const sim::HcommEndpoint &) { return true; });
    EXPECT_TRUE(records.empty());
}

TEST_F(CtrlBaseResourceStubTest,
       EndpointCreateRejectsAmbiguousSouthEndpointBinding) {
    constexpr uint32_t physicalId = 6u;
    uint64_t deviceId = AddDevice(physicalId);
    ASSERT_GT(deviceId, 0u);

    sim::EndPoint southEndpoint{};
    southEndpoint.device_id = static_cast<uint32_t>(deviceId);
    southEndpoint.eid[0] = 0x99;
    ASSERT_GT(SimRunnerSqliteDB::Instance().Add<sim::EndPoint>(southEndpoint),
              0u);
    southEndpoint.id = 0u;
    ASSERT_GT(SimRunnerSqliteDB::Instance().Add<sim::EndPoint>(southEndpoint),
              0u);

    EndpointDesc endpoint{};
    endpoint.protocol = COMM_PROTOCOL_UBC_TP;
    endpoint.commAddr.type = COMM_ADDR_TYPE_EID;
    endpoint.commAddr.eid[0] = 0x99;
    endpoint.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    endpoint.loc.device.devPhyId = physicalId;

    EndpointHandle handle = nullptr;
    EXPECT_EQ(HcommEndpointCreate(&endpoint, &handle), HCCL_E_INTERNAL);
    EXPECT_EQ(handle, nullptr);

    auto records = SimRunnerSqliteDB::Instance().QueryList<sim::HcommEndpoint>(
        [](const sim::HcommEndpoint &) { return true; });
    EXPECT_TRUE(records.empty());
}

} // namespace
