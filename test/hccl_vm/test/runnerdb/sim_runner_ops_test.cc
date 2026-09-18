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
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>
#include <unistd.h>

#include "db_sim_runner_common.h"
#include "db_sim_runner_db.h"
#include "db_sim_runner_ops.h"
#include "db_sim_sqlite_db.h"

namespace {
const std::string kTestDbPath = "/tmp/test_sim_runner_ops.db";

void CleanUpDb()
{
    std::remove(kTestDbPath.c_str());
    std::remove((kTestDbPath + "-wal").c_str());
    std::remove((kTestDbPath + "-shm").c_str());
}

void SetupTestData()
{
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
    RunnerDB::Add<sim::Device>(device);

    sim::Context ctx{};
    ctx.device_id = 1;
    ctx.is_default = 1;
    ctx.run_id = 1;
    RunnerDB::Add<sim::Context>(ctx);

    sim::Communicator comm{};
    std::strncpy(comm.comm_id, "test_comm", sizeof(comm.comm_id) - 1);
    comm.rank_size = 1;
    comm.rank_id = 0;
    comm.device_id = 1;
    g_cur_comm_key = RunnerDB::Add<sim::Communicator>(comm);
}
} // namespace

class SimRunnerOpsTest : public testing::Test {
protected:
    void SetUp() override { SetupTestData(); }

    void TearDown() override { CleanUpDb(); }
};

TEST_F(SimRunnerOpsTest, SetAndGetLastStreamId)
{
    sim::SetLastStreamIdTls(12345);

    uint64_t streamId = sim::GetLastStreamIdTls();

    EXPECT_EQ(streamId, 12345);
}

TEST_F(SimRunnerOpsTest, SetAndGetLastTaskId)
{
    sim::SetLastTaskIdTls(67890);

    uint64_t taskId = sim::GetLastTaskIdTls();

    EXPECT_EQ(taskId, 67890);
}

TEST_F(SimRunnerOpsTest, SetTsDevice)
{
    sim::SetTsDevice(42);
    sim::SetTsDevice(0);
}

TEST_F(SimRunnerOpsTest, GetRankSize_ReturnsCorrectCount)
{
    uint32_t size = sim::GetRankSize();

    EXPECT_GE(size, 1);
}

TEST_F(SimRunnerOpsTest, GetHostSize_ReturnsCorrectCount)
{
    uint32_t size = sim::GetHostSize();

    EXPECT_GE(size, 1);
}

TEST_F(SimRunnerOpsTest, SetCurrCtxTls_UpdatesContext)
{
    bool ret = sim::SetCurrCtxTls(1);

    EXPECT_TRUE(ret);
}

TEST_F(SimRunnerOpsTest, GetCurrDeviceKey_WithValidCtx_ReturnsDeviceKey)
{
    sim::SetCurrCtxTls(1);

    uint64_t deviceKey = sim::GetCurrDeviceKey();

    EXPECT_EQ(deviceKey, 1);
}

TEST_F(SimRunnerOpsTest, GetCurrDeviceKey_WithInvalidCtx_ReturnsZero)
{
    sim::SetCurrCtxTls(99999);

    uint64_t deviceKey = sim::GetCurrDeviceKey();

    EXPECT_EQ(deviceKey, 0);
}

TEST_F(SimRunnerOpsTest, GetDeviceIdByCtxId_WithValidCtx_ReturnsDeviceId)
{
    uint64_t deviceId = sim::GetDeviceIdByCtxId(1);

    EXPECT_EQ(deviceId, 1);
}

TEST_F(SimRunnerOpsTest, GetDeviceIdByCtxId_WithInvalidCtx_ReturnsZero)
{
    uint64_t deviceId = sim::GetDeviceIdByCtxId(99999);

    EXPECT_EQ(deviceId, 0);
}

TEST_F(SimRunnerOpsTest, GetCurrDeviceId_WithValidContext)
{
    sim::SetCurrCtxTls(1);

    uint64_t deviceId = sim::GetCurrDeviceId();

    EXPECT_EQ(deviceId, 1);
}

// ==================== GetCurrDeviceId 错误路径 ====================

class SimRunnerOpsGetCurrDeviceIdTest : public testing::Test {
protected:
    void SetUp() override
    {
        SetupTestData();
        sim::Runner runner;
        sim::GetCurrRunnerTls(1, runner);
    }

    void TearDown() override { CleanUpDb(); }
};

TEST_F(SimRunnerOpsGetCurrDeviceIdTest, GetCurrDeviceId_InvalidCtx_ReturnsZero)
{
    // 设置一个不存在的 context id
    sim::SetCurrCtxTls(99999);
    uint64_t deviceId = sim::GetCurrDeviceId();
    EXPECT_EQ(deviceId, 0);
}

TEST_F(SimRunnerOpsGetCurrDeviceIdTest, GetCurrDeviceId_MissingContextDevice_ReturnsZero)
{
    // 添加一个 context，其 device_id 没有对应的设备。
    sim::Context ctx2{};
    ctx2.device_id = 999;
    ctx2.is_default = 1;
    ctx2.run_id = 1;
    const uint64_t ctxId = RunnerDB::Add<sim::Context>(ctx2);

    sim::SetCurrCtxTls(ctxId);
    uint64_t deviceId = sim::GetCurrDeviceId();
    EXPECT_EQ(deviceId, 0);
}

// ==================== GetDeviceIdByCtxId 错误路径 ====================

class SimRunnerOpsGetDeviceIdByCtxIdTest : public testing::Test {
protected:
    void SetUp() override { SetupTestData(); }

    void TearDown() override { CleanUpDb(); }
};

TEST_F(SimRunnerOpsGetDeviceIdByCtxIdTest, GetDeviceIdByCtxId_InvalidCtx_ReturnsZero)
{
    uint64_t deviceId = sim::GetDeviceIdByCtxId(99999);
    EXPECT_EQ(deviceId, 0);
}

TEST_F(SimRunnerOpsGetDeviceIdByCtxIdTest, GetDeviceIdByCtxId_NoDevice_ReturnsZero)
{
    // 添加一个 context，其 device_id 没有对应的 device
    sim::Context ctx2{};
    ctx2.device_id = 999; // device_id=999 不存在
    ctx2.is_default = 1;
    ctx2.run_id = 1;
    const uint64_t ctxId = RunnerDB::Add<sim::Context>(ctx2);

    uint64_t deviceId = sim::GetDeviceIdByCtxId(ctxId);
    EXPECT_EQ(deviceId, 0);
}

// ==================== 新增：GetCurrDeviceKey 错误路径 ====================

TEST_F(SimRunnerOpsGetCurrDeviceIdTest, GetCurrDeviceKey_InvalidCtx_ReturnsZero)
{
    sim::SetCurrCtxTls(99999);
    uint64_t deviceKey = sim::GetCurrDeviceKey();
    EXPECT_EQ(deviceKey, 0);
}
