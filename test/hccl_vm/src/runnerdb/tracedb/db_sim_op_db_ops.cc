/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "db_sim_op_db_ops.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <utility>
#include <variant>

#include "db_hccl_op_db_ops.h"
#include "db_sim_communicator.h"
#include "db_sim_runner_db.h"
#include "sim_common_macro.h"
#include "sim_log.h"
#include "sim_models.h"

using HcclSim::DB::BlobData;
using HcclSim::DB::ExRows;
using HcclSim::DB::Field;
using HcclSim::DB::OpDbOps;
using HcclSim::DB::Value;

namespace sim {
using CommunicatorIdentity = std::pair<std::string, uint64_t>;
std::map<uint64_t, std::atomic<uint64_t>> g_syncStreamIterMap{};
std::map<CommunicatorIdentity, std::atomic<uint32_t>> g_opIterCounter{};
std::mutex g_opIterCounterMutex{};
uint32_t g_syncIterCounter = 0; // todo 可删
uint32_t g_currOpDetailId = 0;
uint32_t g_currOpMemId = 0;

int InitOpDataDb()
{
    OpDbOps::Instance();
    return 0;
}

int SetDbConfig(DBConfig& config) { return OpDbOps::Instance().SetDbConfig(config); }

// ============================================================
// Insert APIs
// ============================================================

int InsertOpDetail(OpDetailTab& rec)
{
    std::string commName;
    uint64_t commHash = 0;
    if (!GetCommunicatorIdentity(rec.commId, commName, commHash)) {
        HCCL_VM_ERROR(
            "Failed to resolve communicator identity before "
            "assigning opIter, commId={}",
            rec.commId);
        return -1;
    }
    {
        std::lock_guard<std::mutex> lock(g_opIterCounterMutex);
        auto iter = g_opIterCounter.try_emplace(std::make_pair(commName, commHash), 0u).first;
        rec.opIter = iter->second.fetch_add(1, std::memory_order_relaxed);
    }
    rec.syncIter = g_syncIterCounter;

    const std::string sql = "INSERT INTO opDetails (pid, deviceId, rankId, commId, opIter, "
                            "syncIter, streamId, root, opExpansionMode, devType, rankSize, "
                            "srcRank, dstRank, opDetail, opExtInfo) VALUES (?, ?, ?, ?, ?, ?, ?, "
                            "?, ?, ?, ?, ?, ?, ?, ?)";
    std::vector<Value> params
        = {(int64_t)rec.pid,
           (int64_t)rec.deviceId,
           (int64_t)rec.rankId,
           (int64_t)rec.commId,
           (int64_t)rec.opIter,
           (int64_t)rec.syncIter,
           (int64_t)rec.streamId,
           (int64_t)rec.root,
           (int64_t)rec.opExpansionMode,
           (int64_t)rec.devType,
           (int64_t)rec.rankSize,
           (int64_t)rec.srcRank,
           (int64_t)rec.dstRank,
           BlobData{rec.opDetail.data(), rec.opDetail.size()},
           BlobData{rec.opExtInfo.data(), rec.opExtInfo.size()}};

    int ret = OpDbOps::Instance().ExecInsert(sql, params, rec.id);
    if (ret == 0) {
        g_currOpDetailId = rec.id;
    }
    return ret;
}

int InsertOpMem(OpMemInfoTab& rec)
{
    rec.opDetailId = g_currOpDetailId;

    const std::string sql = "INSERT INTO opMemInfo (opDetailId, inputAddr, inputSize, outputAddr, "
                            "outputSize, cclAddr, cclSize) VALUES (?, ?, ?, ?, ?, ?, ?)";
    std::vector<Value> params
        = {(int64_t)rec.opDetailId, (int64_t)rec.inputAddr, (int64_t)rec.inputSize, (int64_t)rec.outputAddr,
           (int64_t)rec.outputSize, (int64_t)rec.cclAddr,   (int64_t)rec.cclSize};
    return OpDbOps::Instance().ExecInsert(sql, params, g_currOpMemId);
}

int InsertOpDetailAndMem(OpDetailTab& detail, OpMemInfoTab& mem)
{
    return OpDbOps::Instance().RunInTransaction([&]() -> int {
        int ret = InsertOpDetail(detail);
        if (ret != 0) {
            return ret;
        }
        mem.opDetailId = detail.id;
        return InsertOpMem(mem);
    });
}

int InsertCcuChannel(CcuChannelTab& rec)
{
    const std::string sql = "INSERT INTO ccuChannels (channelId, srcDieId, dstDieId, srcDeviceId, "
                            "dstDeviceId, srcRankId, dstRankId, leid, reid, protocol, jettyNum, "
                            "jettyId) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";
    std::vector<uint8_t> jettyId((uint8_t*)rec.jettyId, (uint8_t*)rec.jettyId + rec.jettyNum * 4);
    std::vector<Value> params
        = {(int64_t)rec.channelId,   (int64_t)rec.srcDieId,    (int64_t)rec.dstDieId,
           (int64_t)rec.srcDeviceId, (int64_t)rec.dstDeviceId, (int64_t)rec.srcRankId,
           (int64_t)rec.dstRankId,   BlobData{rec.leid, 16},   BlobData{rec.reid, 16},
           (int64_t)rec.protocol,    (int64_t)rec.jettyNum,    BlobData{jettyId.data(), jettyId.size()}};
    uint32_t dummy;
    return OpDbOps::Instance().ExecInsert(sql, params, dummy);
}

int InsertJettyMap(JettyMapTab& rec)
{
    rec.opDetailId = g_currOpDetailId;
    const std::string sql = "INSERT INTO JettyMaps (opDetailId, srcDieId, dstDieId, srcRankId, "
                            "dstRankId, leid, reid, protocol) VALUES (?, ?, ?, ?, ?, ?, ?, ?)";
    std::vector<Value> params
        = {(int64_t)rec.opDetailId, (int64_t)rec.srcDieId,  (int64_t)rec.dstDieId,  (int64_t)rec.srcRankId,
           (int64_t)rec.dstRankId,  BlobData{rec.leid, 16}, BlobData{rec.reid, 16}, (int64_t)rec.protocol};
    uint32_t dummy;
    return OpDbOps::Instance().ExecInsert(sql, params, dummy);
}

int InsertOpTask(OpTaskTab& rec, bool isDevice)
{
    rec.opDetailId = g_currOpDetailId;
    // host进程使用pid，AICPU模式的device进程使用ppid(即host进程的pid)
    auto tabPid = isDevice ? getppid() : getpid();
    rec.pid = tabPid;

    const std::string tabName = "opTask_P_" + std::to_string(tabPid);
    const std::string sql = "INSERT INTO " + tabName
                            + " (pid, opDetailId, deviceId, streamId, taskType, "
                              "isDone, opTaskMeta) VALUES (?, ?, ?, ?, ?, ?, ?)";
    std::vector<Value> params
        = {(int64_t)rec.pid,
           (int64_t)rec.opDetailId,
           (int64_t)rec.deviceId,
           (int64_t)rec.streamId,
           (int64_t)rec.taskType,
           (int64_t)rec.isDone,
           BlobData{rec.optaskMeta.data(), rec.optaskMeta.size()}};
    uint32_t dummy;
    return OpDbOps::Instance().ExecInsert(sql, params, dummy);
}

int InsertSyncRecord(SyncRecordTab& rec)
{
    rec.syncIter = g_syncIterCounter++;
    const std::string sql = "INSERT INTO syncRecords (pid, rankId, rankSize, syncIter, streamId, "
                            "status) VALUES (?, ?, ?, ?, ?, ?)";
    std::vector<Value> params = {(int64_t)rec.pid,      (int64_t)rec.rankId,   (int64_t)rec.rankSize,
                                 (int64_t)rec.syncIter, (int64_t)rec.streamId, (int64_t)rec.status};
    return OpDbOps::Instance().ExecInsert(sql, params, rec.id);
}

int InsertCcuInstrRes(CcuInstrResTab& rec)
{
    const std::string sql = "INSERT INTO ccuInstrRes (deviceId, dieId, "
                            "instrCount, instrSpace) VALUES (?, ?, ?, ?)";
    std::vector<Value> params
        = {(int64_t)rec.deviceId, (int64_t)rec.dieId, (int64_t)rec.instrCount,
           BlobData{rec.instrSpace, sizeof(rec.instrSpace)}};
    return OpDbOps::Instance().ExecInsert(sql, params, rec.id);
}

int InsertCcuInstr(CcuInstrTab& rec)
{
    const std::string sql = "INSERT INTO ccuInstr (ccuInstrResId, startId, "
                            "instrInfoSize) VALUES (?, ?, ?)";
    std::vector<Value> params = {(int64_t)rec.ccuInstrResId, (int64_t)rec.startId, (int64_t)rec.instrInfoSize};
    return OpDbOps::Instance().ExecInsert(sql, params, rec.id);
}

int UpdateAndInsertByCcuId(
    uint64_t& ccuId, uint32_t deviceId, uint32_t dieId, uint32_t startId, uint32_t instrCount, uint32_t instrOffset,
    uint32_t instrInfoSize, const void* instrInfo, std::vector<uint8_t>* mergedInstrSpace, uint32_t* totalInstrCount)
{
    constexpr size_t kInstrSpaceSize = HcclSim::CCU_INSTRUCTION_NUM * HcclSim::CCU_INSTRUCTION_SIZE;

    // 使用减法形式校验写入范围，避免 offset + size 的整数溢出。
    if (instrInfo == nullptr || instrOffset > kInstrSpaceSize || instrInfoSize > kInstrSpaceSize - instrOffset) {
        HCCL_VM_ERROR(
            "invalid instruction range: offset={}, size={}, "
            "space={}, instrInfo={}",
            instrOffset, instrInfoSize, kInstrSpaceSize, instrInfo == nullptr ? "null" : "valid");
        return -1;
    }

    const std::string querySql = "SELECT id, instrSpace, instrCount FROM "
                                 "ccuInstrRes WHERE deviceId = ? AND dieId = ?";
    std::vector<Value> queryParams = {(int64_t)deviceId, (int64_t)dieId};
    ExRows rows;
    int ret = OpDbOps::Instance().ExecQueryEx(querySql, queryParams, rows);
    if (ret != 0) {
        HCCL_VM_ERROR("query ccuInstrRes failed for deviceId: {} dieId: {}", deviceId, dieId);
        return -1;
    }

    if (!rows.empty() && rows[0].size() >= 3) {
        uint32_t existId = std::stoul(std::get<std::string>(rows[0][0]));
        std::vector<uint8_t> space(kInstrSpaceSize, 0);
        if (std::holds_alternative<std::vector<uint8_t>>(rows[0][1])) {
            const auto& existBlob = std::get<std::vector<uint8_t>>(rows[0][1]);
            std::memcpy(space.data(), existBlob.data(), std::min(existBlob.size(), kInstrSpaceSize));
        }
        std::memcpy(space.data() + instrOffset, instrInfo, instrInfoSize);

        uint32_t existCount = 0;
        if (std::holds_alternative<std::string>(rows[0][2])) {
            existCount = std::stoul(std::get<std::string>(rows[0][2]));
        }

        // 支持空隙，startId 为指令起始地址，instrCount 为指令数量
        uint32_t newCount = std::max(existCount, startId + instrCount);

        const std::string updateSql = "UPDATE ccuInstrRes SET instrSpace = ?, "
                                      "instrCount = ? WHERE id = ?";
        std::vector<Value> updateParams = {BlobData{space.data(), space.size()}, (int64_t)newCount, (int64_t)existId};
        ret = OpDbOps::Instance().ExecUpdate(updateSql, updateParams);
        if (ret == 0) {
            ccuId = existId;
        }
        if (mergedInstrSpace != nullptr) {
            *mergedInstrSpace = std::move(space);
        }
        if (totalInstrCount != nullptr) {
            *totalInstrCount = newCount;
        }
        return ret;
    } else {
        std::vector<uint8_t> space(kInstrSpaceSize, 0);
        std::memcpy(space.data() + instrOffset, instrInfo, instrInfoSize);

        uint32_t newCount = startId + instrCount;
        const std::string insertSql = "INSERT INTO ccuInstrRes (deviceId, dieId, instrCount, instrSpace) "
                                      "VALUES (?, ?, ?, ?)";
        std::vector<Value> insertParams
            = {(int64_t)deviceId, (int64_t)dieId, (int64_t)newCount, BlobData{space.data(), space.size()}};
        uint32_t newId = 0;
        ret = OpDbOps::Instance().ExecInsert(insertSql, insertParams, newId);
        if (ret == 0) {
            ccuId = newId;
        }
        if (mergedInstrSpace != nullptr) {
            *mergedInstrSpace = std::move(space);
        }
        if (totalInstrCount != nullptr) {
            *totalInstrCount = newCount;
        }
        return ret;
    }
}

int InsertHalfRTT(HalfRTTTab& rec)
{
    const std::string sql = "INSERT INTO halfRTT (deviceId, dieId, wishCntXnIdBegin, "
                            "wishCntXnIdEnd, totalCntId) VALUES (?, ?, ?, ?, ?)";
    std::vector<Value> params
        = {(int64_t)rec.deviceId, (int64_t)rec.dieId, (int64_t)rec.wishCntXnIdBegin, (int64_t)rec.wishCntXnIdEnd,
           (int64_t)rec.totalCntId};
    return OpDbOps::Instance().ExecInsert(sql, params, rec.id);
}

// ============================================================
// Query / Update APIs
// ============================================================

int QuerySyncRecordByStatus(uint8_t status, std::vector<SyncRecordTab>& out)
{
    // 1. 补全缺失的列 (rankId, rankSize)，确保结构体数据完整
    const std::string sql = "SELECT id, pid, rankId, rankSize, syncIter, streamId, status FROM "
                            "syncRecords WHERE status = ?";
    std::vector<Value> params = {(int64_t)status};
    std::vector<std::vector<std::string>> rows;

    if (OpDbOps::Instance().ExecQuery(sql, params, rows) != 0) {
        return -1;
    }

    out.clear();

    // 2. 使用枚举替代魔鬼数字，提高可读性和维护性
    enum : int { COL_ID, COL_PID, COL_RANK_ID, COL_RANK_SIZE, COL_SYNC_ITER, COL_STREAM_ID, COL_STATUS };
    const size_t COL_COUNT = 7;

    for (const auto& r : rows) {
        if (r.size() < COL_COUNT) {
            continue;
        }

        SyncRecordTab rec;
        rec.id = std::stoul(r[COL_ID]);
        rec.pid = std::stoul(r[COL_PID]);
        rec.rankId = std::stoul(r[COL_RANK_ID]);
        rec.rankSize = std::stoul(r[COL_RANK_SIZE]);
        rec.syncIter = std::stoul(r[COL_SYNC_ITER]);
        rec.streamId = std::stoull(r[COL_STREAM_ID]);
        rec.status = static_cast<uint8_t>(std::stoul(r[COL_STATUS]));

        out.push_back(std::move(rec));
    }
    return 0;
}

int UpdateSyncRecordStatus(std::vector<SyncRecordTab>& syncRecord)
{
    return OpDbOps::Instance().RunInTransaction([&]() -> int {
        const std::string sql = "UPDATE syncRecords SET status = ? WHERE id = ?";
        for (const auto& rec : syncRecord) {
            std::vector<Value> params = {(int64_t)rec.status, (int64_t)rec.id};
            int ret = OpDbOps::Instance().ExecUpdate(sql, params);
            if (ret != 0) {
                return ret;
            }
        }
        return 0;
    });
}

int UpdateOpMemCclBuffer(uint64_t cclAddr, uint64_t cclSize)
{
    const std::string sql = "UPDATE opMemInfo SET cclAddr = ?, cclSize = ? WHERE id = ?";
    std::vector<Value> params = {(int64_t)cclAddr, (int64_t)cclSize, (int64_t)g_currOpMemId};
    int ret = OpDbOps::Instance().ExecUpdate(sql, params);
    if (ret != 0) {
        return ret;
    }
    return 0;
}

int QueryCurrentOpMemInfo(uint64_t commId, uint32_t deviceId, OpMemInfoTab& out)
{
    if (commId == 0 || deviceId == 0) {
        HCCL_VM_ERROR("invalid current opMemInfo query, commId={}, deviceId={}", commId, deviceId);
        return -1;
    }

    const std::string sql = "SELECT m.id, m.opDetailId, m.inputAddr, m.inputSize, "
                            "m.outputAddr, m.outputSize, m.cclAddr, m.cclSize "
                            "FROM opDetails d "
                            "JOIN opMemInfo m ON m.opDetailId = d.id "
                            "WHERE d.commId = ? AND d.deviceId = ? "
                            "ORDER BY d.id DESC LIMIT 1";
    std::vector<Value> params = {(int64_t)commId, (int64_t)deviceId};
    std::vector<std::vector<std::string>> rows;
    if (OpDbOps::Instance().ExecQuery(sql, params, rows) != 0 || rows.empty()) {
        HCCL_VM_ERROR("query current opMemInfo failed, commId={}, deviceId={}", commId, deviceId);
        return -1;
    }

    const auto& row = rows[0];
    if (row.size() < 8) {
        HCCL_VM_ERROR("invalid current opMemInfo row size={}, commId={}, deviceId={}", row.size(), commId, deviceId);
        return -1;
    }

    out.id = std::stoul(row[0]);
    out.opDetailId = std::stoul(row[1]);
    out.inputAddr = std::stoull(row[2]);
    out.inputSize = std::stoull(row[3]);
    out.outputAddr = std::stoull(row[4]);
    out.outputSize = std::stoull(row[5]);
    out.cclAddr = std::stoull(row[6]);
    out.cclSize = std::stoull(row[7]);
    return 0;
}

int UpdateOpExpansionMode(uint8_t mode)
{
    const std::string sql = "UPDATE opDetails SET opExpansionMode = ? WHERE id = ?";
    std::vector<Value> params = {(int64_t)mode, (int64_t)g_currOpDetailId};
    int ret = OpDbOps::Instance().ExecUpdate(sql, params);
    if (ret != 0) {
        HCCL_VM_ERROR("Failed to update mode={}, id={}", mode, g_currOpDetailId);
    }
    return ret;
}

int QueryLatestOpExpansionMode()
{
    const std::string sql = "SELECT opExpansionMode FROM opDetails WHERE id = ?";
    std::vector<Value> params = {(int64_t)g_currOpDetailId};
    std::vector<std::vector<std::string>> rows;
    if (OpDbOps::Instance().ExecQuery(sql, params, rows) != 0 || rows.empty() || rows[0].empty()) {
        HCCL_VM_ERROR("Failed to query, id={}", g_currOpDetailId);
        return static_cast<int>(SimOpExpansionMode::SIM_OP_EXPANSION_MODE_RESERVED);
    }
    return static_cast<int>(std::stoul(rows[0][0]));
}

int QueryOpExecutionIndexEntries(std::vector<OpExecutionIndexEntry>& out)
{
    using HcclSim::DB::Field;
    enum : int { ID, DEVICE_ID, RANK_ID, COMM_ID, OP_ITER, RANK_SIZE };
    constexpr size_t COLUMN_COUNT = 6;
    const std::string sql = "SELECT id, deviceId, rankId, commId, opIter, rankSize "
                            "FROM opDetails ORDER BY id";
    HcclSim::DB::ExRows rows;
    if (OpDbOps::Instance().ExecQueryEx(sql, {}, rows) != 0) {
        HCCL_VM_ERROR("Failed to query operator execution index");
        return -1;
    }

    auto toStr = [](const Field& field) -> std::string {
        if (std::holds_alternative<std::string>(field)) {
            return std::get<std::string>(field);
        }
        return {};
    };
    out.clear();
    out.reserve(rows.size());
    for (const auto& row : rows) {
        if (row.size() < COLUMN_COUNT) {
            HCCL_VM_ERROR("Invalid operator execution index row, columnCount={}", row.size());
            return -1;
        }
        OpExecutionIndexEntry entry;
        entry.opDetailId = std::stoul(toStr(row[ID]));
        entry.deviceId = std::stoul(toStr(row[DEVICE_ID]));
        entry.rankId = std::stoul(toStr(row[RANK_ID]));
        entry.commId = std::stoull(toStr(row[COMM_ID]));
        entry.opIter = std::stoul(toStr(row[OP_ITER]));
        entry.rankSize = std::stoul(toStr(row[RANK_SIZE]));
        out.push_back(entry);
    }
    return 0;
}

int QueryCompositeOpDetailBySyncIter(uint32_t syncIter, std::map<uint32_t, std::vector<CompositeOpDetail>>& detail)
{
    // 1. 定义 opDetails 表的列索引枚举 (共 16 列)
    enum : int {
        OD_ID,
        OD_PID,
        OD_DEVICE_ID,
        OD_RANK_ID,
        OD_COMM_ID,
        OD_OP_ITER,
        OD_SYNC_ITER,
        OD_STREAM_ID,
        OD_ROOT,
        OD_OP_EXPANSION,
        OD_DEV_TYPE,
        OD_RANK_SIZE,
        OD_SRC_RANK,
        OD_DST_RANK,
        OD_OP_DETAIL,
        OD_OP_EXT_INFO
    };
    const size_t OD_COL_COUNT = 16;

    // 2. 定义 opMemInfo 表的列索引枚举 (共 8 列)
    enum : int {
        MEM_ID,
        MEM_OP_DETAIL_ID,
        MEM_INPUT_ADDR,
        MEM_INPUT_SIZE,
        MEM_OUTPUT_ADDR,
        MEM_OUTPUT_SIZE,
        MEM_CCL_ADDR,
        MEM_CCL_SIZE
    };
    const size_t MEM_COL_COUNT = 8;

    // 3. 定义 opTask 表的列索引枚举 (共 6 列)
    enum : int {
        TASK_ID,
        TASK_PID,
        TASK_OP_DETAIL_ID,
        TASK_DEVICE_ID,
        TASK_STREAM_ID,
        TASK_TYPE,
        TASK_IS_DONE,
        TASK_META
    };
    const size_t TASK_COL_COUNT = 8;

    using HcclSim::DB::Field;

    detail.clear();

    const std::string opSql = "SELECT id, pid, deviceId, rankId, commId, opIter, syncIter, streamId, "
                              "root, opExpansionMode, devType, rankSize, srcRank, dstRank, "
                              "opDetail, opExtInfo FROM opDetails WHERE syncIter = ? ORDER BY id";
    std::vector<Value> opParams = {(int64_t)syncIter};
    HcclSim::DB::ExRows opRows;
    if (OpDbOps::Instance().ExecQueryEx(opSql, opParams, opRows) != 0) {
        HCCL_VM_ERROR("Query opDetails failed for syncIter: {}", syncIter);
        return -1;
    }

    const std::string memSql = "SELECT id, opDetailId, inputAddr, inputSize, "
                               "outputAddr, outputSize, cclAddr, cclSize "
                               "FROM opMemInfo WHERE opDetailId = ?";

    // 提取 Lambda 到循环外，避免重复定义
    auto toStr = [](const Field& f) -> std::string {
        if (std::holds_alternative<std::string>(f)) {
            return std::get<std::string>(f);
        }
        return {};
    };

    for (const auto& r : opRows) {
        // 使用枚举常量替代魔鬼数字 14
        if (r.size() < OD_COL_COUNT) {
            continue;
        }

        CompositeOpDetail comp = {};
        comp.deviceId = static_cast<uint32_t>(std::stoul(toStr(r[OD_DEVICE_ID])));
        comp.rankId = static_cast<uint32_t>(std::stoul(toStr(r[OD_RANK_ID])));
        comp.commId = std::stoull(toStr(r[OD_COMM_ID]));
        comp.detail.id = std::stoul(toStr(r[OD_ID]));
        comp.detail.pid = std::stoul(toStr(r[OD_PID]));
        comp.detail.deviceId = comp.deviceId;
        comp.detail.rankId = std::stoul(toStr(r[OD_RANK_ID]));
        comp.detail.commId = comp.commId;
        comp.detail.opIter = std::stoul(toStr(r[OD_OP_ITER]));
        comp.detail.syncIter = std::stoul(toStr(r[OD_SYNC_ITER]));
        comp.detail.streamId = std::stoull(toStr(r[OD_STREAM_ID]));
        comp.detail.root = std::stoul(toStr(r[OD_ROOT]));
        comp.detail.opExpansionMode = std::stoul(toStr(r[OD_OP_EXPANSION]));
        comp.detail.devType = std::stoul(toStr(r[OD_DEV_TYPE]));
        comp.detail.rankSize = std::stoul(toStr(r[OD_RANK_SIZE]));
        comp.detail.srcRank = std::stoul(toStr(r[OD_SRC_RANK]));
        comp.detail.dstRank = std::stoul(toStr(r[OD_DST_RANK]));

        if (std::holds_alternative<std::vector<uint8_t>>(r[OD_OP_DETAIL])) {
            comp.detail.opDetail = std::get<std::vector<uint8_t>>(r[OD_OP_DETAIL]);
        }
        if (std::holds_alternative<std::vector<uint8_t>>(r[OD_OP_EXT_INFO])) {
            comp.detail.opExtInfo = std::get<std::vector<uint8_t>>(r[OD_OP_EXT_INFO]);
        }

        const uint32_t opDetailId = comp.detail.id;
        const uint32_t pid = comp.detail.pid;

        // 查询 MemInfo
        std::vector<Value> memParams = {(int64_t)opDetailId};
        std::vector<std::vector<std::string>> memRows;
        if (OpDbOps::Instance().ExecQuery(memSql, memParams, memRows) == 0 && !memRows.empty()) {
            const auto& mr = memRows[0];
            // 使用枚举常量替代魔鬼数字
            if (mr.size() >= MEM_COL_COUNT) {
                comp.memInfo.id = std::stoul(mr[MEM_ID]);
                comp.memInfo.opDetailId = std::stoul(mr[MEM_OP_DETAIL_ID]);
                comp.memInfo.inputAddr = std::stoull(mr[MEM_INPUT_ADDR]);
                comp.memInfo.inputSize = std::stoull(mr[MEM_INPUT_SIZE]);
                comp.memInfo.outputAddr = std::stoull(mr[MEM_OUTPUT_ADDR]);
                comp.memInfo.outputSize = std::stoull(mr[MEM_OUTPUT_SIZE]);
                comp.memInfo.cclAddr = std::stoull(mr[MEM_CCL_ADDR]);
                comp.memInfo.cclSize = std::stoull(mr[MEM_CCL_SIZE]);
            }
        }

        // 查询 Tasks
        const std::string taskTabName = "opTask_P_" + std::to_string(pid);
        const std::string taskSql = "SELECT id, pid, opDetailId, deviceId, streamId, taskType, isDone, "
                                    "opTaskMeta FROM "
                                    + taskTabName + " WHERE opDetailId = ?";
        std::vector<Value> taskParams = {(int64_t)opDetailId};
        HcclSim::DB::ExRows taskRows;
        if (OpDbOps::Instance().ExecQueryEx(taskSql, taskParams, taskRows) == 0) {
            for (const auto& tr : taskRows) {
                // 为确保数据完整性（特别是 opTaskMeta），应检查完整的列数
                if (tr.size() < TASK_COL_COUNT) {
                    continue;
                }
                OpTaskTab taskRec;
                taskRec.id = std::stoul(toStr(tr[TASK_ID]));
                taskRec.pid = std::stoul(toStr(tr[TASK_PID]));
                taskRec.opDetailId = std::stoul(toStr(tr[TASK_OP_DETAIL_ID]));
                taskRec.deviceId = std::stoull(toStr(tr[TASK_DEVICE_ID]));
                taskRec.streamId = std::stoull(toStr(tr[TASK_STREAM_ID]));
                taskRec.taskType = std::stoul(toStr(tr[TASK_TYPE]));
                taskRec.isDone = std::stoul(toStr(tr[TASK_IS_DONE]));

                // 安全访问 Meta 数据
                if (std::holds_alternative<std::vector<uint8_t>>(tr[TASK_META])) {
                    taskRec.optaskMeta = std::get<std::vector<uint8_t>>(tr[TASK_META]);
                }
                comp.tasks.push_back(std::move(taskRec));
            }
        }

        detail[static_cast<int32_t>(comp.rankId)].push_back(std::move(comp));
    }

    return 0;
}

// ============================================================
// Query Single APIs
// ============================================================

int QueryOpDetailIdentity(uint32_t opDetailId, uint64_t& commId, uint32_t& rankId, uint32_t& deviceId)
{
    const std::string sql = "SELECT commId, rankId, deviceId FROM opDetails WHERE id = ?";
    std::vector<Value> params = {(int64_t)opDetailId};
    std::vector<std::vector<std::string>> rows;
    if (OpDbOps::Instance().ExecQuery(sql, params, rows) != 0 || rows.empty() || rows[0].size() < 3) {
        return -1;
    }

    commId = std::stoull(rows[0][0]);
    rankId = std::stoul(rows[0][1]);
    deviceId = std::stoul(rows[0][2]);
    return 0;
}

// ============================================================
// Query All APIs
// ============================================================

int QueryCcuChannelAll(std::vector<CcuChannelTab>& out)
{
    enum : int {
        ID,
        CHANNEL_ID,
        SRC_DIE_ID,
        DST_DIE_ID,
        SRC_DEVICE_ID,
        DST_DEVICE_ID,
        SRC_RANK_ID,
        DST_RANK_ID,
        LEID,
        REID,
        PROTOCOL,
        JETTY_NUM,
        JETTY_ID
    };
    const size_t CHANNEL_COL_COUNT = 13;
    const std::string sql = "SELECT id, channelId, srcDieId, dstDieId, srcDeviceId, dstDeviceId, "
                            "srcRankId, dstRankId, leid, reid, protocol, jettyNum, jettyId FROM "
                            "ccuChannels";
    using HcclSim::DB::Field;
    HcclSim::DB::ExRows rows;
    if (OpDbOps::Instance().ExecQueryEx(sql, {}, rows) != 0) {
        return -1;
    }

    auto toStr = [](const Field& f) -> std::string {
        if (std::holds_alternative<std::string>(f)) {
            return std::get<std::string>(f);
        }
        return {};
    };

    out.clear();
    for (const auto& r : rows) {
        if (r.size() < CHANNEL_COL_COUNT) {
            continue;
        }
        CcuChannelTab rec = {};
        rec.id = std::stoul(toStr(r[ID]));
        rec.channelId = std::stoul(toStr(r[CHANNEL_ID]));
        rec.srcDieId = std::stoul(toStr(r[SRC_DIE_ID]));
        rec.dstDieId = std::stoul(toStr(r[DST_DIE_ID]));
        rec.srcDeviceId = std::stoul(toStr(r[SRC_DEVICE_ID]));
        rec.dstDeviceId = std::stoul(toStr(r[DST_DEVICE_ID]));
        rec.srcRankId = std::stoul(toStr(r[SRC_RANK_ID]));
        rec.dstRankId = std::stoul(toStr(r[DST_RANK_ID]));
        rec.protocol = static_cast<uint16_t>(std::stoul(toStr(r[PROTOCOL])));
        rec.jettyNum = static_cast<uint16_t>(std::stoul(toStr(r[JETTY_NUM])));

        if (std::holds_alternative<std::vector<uint8_t>>(r[LEID])) {
            const auto& blob = std::get<std::vector<uint8_t>>(r[LEID]);
            std::memcpy(rec.leid, blob.data(), std::min(blob.size(), sizeof(rec.leid)));
        }
        if (std::holds_alternative<std::vector<uint8_t>>(r[REID])) {
            const auto& blob = std::get<std::vector<uint8_t>>(r[REID]);
            std::memcpy(rec.reid, blob.data(), std::min(blob.size(), sizeof(rec.reid)));
        }
        if (std::holds_alternative<std::vector<uint8_t>>(r[JETTY_ID])) {
            const auto& blob = std::get<std::vector<uint8_t>>(r[JETTY_ID]);
            std::memcpy(rec.jettyId, blob.data(), std::min(blob.size(), sizeof(rec.jettyId)));
        }

        out.push_back(std::move(rec));
    }
    return 0;
}

int QueryJettyMapAll(std::vector<JettyMapTab>& out)
{
    enum : int { ID, OP_DETAIL_ID, SRC_DIE_ID, DST_DIE_ID, SRC_RANK_ID, DST_RANK_ID, LEID, REID, PROTOCOL };
    const size_t JETTY_COL_COUNT = 9;
    const std::string sql = "SELECT id, opDetailId, srcDieId, dstDieId, srcRankId, dstRankId, "
                            "leid, reid, protocol FROM JettyMaps";
    using HcclSim::DB::Field;
    HcclSim::DB::ExRows rows;
    if (OpDbOps::Instance().ExecQueryEx(sql, {}, rows) != 0) {
        return -1;
    }

    auto toStr = [](const Field& f) -> std::string {
        if (std::holds_alternative<std::string>(f)) {
            return std::get<std::string>(f);
        }
        return {};
    };

    out.clear();
    for (const auto& r : rows) {
        if (r.size() < JETTY_COL_COUNT) {
            continue;
        }
        JettyMapTab rec = {};
        rec.id = std::stoul(toStr(r[ID]));
        rec.opDetailId = std::stoul(toStr(r[OP_DETAIL_ID]));
        rec.srcDieId = std::stoul(toStr(r[SRC_DIE_ID]));
        rec.dstDieId = std::stoul(toStr(r[DST_DIE_ID]));
        rec.srcRankId = std::stoul(toStr(r[SRC_RANK_ID]));
        rec.dstRankId = std::stoul(toStr(r[DST_RANK_ID]));
        rec.protocol = static_cast<uint16_t>(std::stoul(toStr(r[PROTOCOL])));

        if (std::holds_alternative<std::vector<uint8_t>>(r[LEID])) {
            const auto& blob = std::get<std::vector<uint8_t>>(r[LEID]);
            std::memcpy(rec.leid, blob.data(), std::min(blob.size(), sizeof(rec.leid)));
        }
        if (std::holds_alternative<std::vector<uint8_t>>(r[REID])) {
            const auto& blob = std::get<std::vector<uint8_t>>(r[REID]);
            std::memcpy(rec.reid, blob.data(), std::min(blob.size(), sizeof(rec.reid)));
        }

        out.push_back(std::move(rec));
    }
    return 0;
}

int QuerySyncRecordAll(std::vector<SyncRecordTab>& out)
{
    enum : int { ID, PID, RANK_ID, RANK_SIZE, SYNC_ITER, STREAM_ID, STATUS };
    const size_t RECORD_COL_COUNT = 7;
    const std::string sql = "SELECT id, pid, rankId, rankSize, syncIter, "
                            "streamId, status FROM syncRecords";
    std::vector<std::vector<std::string>> rows;
    if (OpDbOps::Instance().ExecQuery(sql, {}, rows) != 0) {
        return -1;
    }

    out.clear();
    for (const auto& r : rows) {
        if (r.size() < RECORD_COL_COUNT) {
            continue;
        }
        SyncRecordTab rec = {};
        rec.id = std::stoul(r[ID]);
        rec.pid = std::stoul(r[PID]);
        rec.rankId = std::stoul(r[RANK_ID]);
        rec.rankSize = std::stoul(r[RANK_SIZE]);
        rec.syncIter = std::stoul(r[SYNC_ITER]);
        rec.streamId = std::stoull(r[STREAM_ID]);
        rec.status = static_cast<uint8_t>(std::stoul(r[STATUS]));
        out.push_back(std::move(rec));
    }
    return 0;
}

int QueryCcuInstrResAll(std::vector<CcuInstrResTab>& out)
{
    enum : int { ID, DEVICE_ID, DIE_ID, INSTR_COUNT, INSTR_SPACE };
    const size_t CCU_COL_COUNT = 5;
    const std::string sql = "SELECT id, deviceId, dieId, instrCount, instrSpace FROM ccuInstrRes";
    ExRows rows;
    if (OpDbOps::Instance().ExecQueryEx(sql, {}, rows) != 0) {
        return -1;
    }

    out.clear();
    for (const auto& r : rows) {
        if (r.size() < CCU_COL_COUNT) {
            continue;
        }
        CcuInstrResTab rec = {};
        rec.id = std::stoul(std::get<std::string>(r[ID]));
        rec.deviceId = std::stoul(std::get<std::string>(r[DEVICE_ID]));
        rec.dieId = std::stoul(std::get<std::string>(r[DIE_ID]));
        rec.instrCount = std::stoul(std::get<std::string>(r[INSTR_COUNT]));
        const auto& blob = std::get<std::vector<uint8_t>>(r[INSTR_SPACE]);
        std::memcpy(rec.instrSpace, blob.data(), std::min(blob.size(), sizeof(rec.instrSpace)));
        out.push_back(std::move(rec));
    }
    return 0;
}

int QueryCcuResourceMetaCount(uint32_t& instrLoadCnt, uint32_t& channelCnt)
{
    // ccuInstr表记录每次微码下发，ccuInstrRes表记录已创建的资源行；
    // 两者行数之和可同时感知微码下发和资源行补齐，避免轮询时漏掉中间状态。
    instrLoadCnt = 0;
    channelCnt = 0;
    const std::string sql = "SELECT (SELECT COUNT(*) FROM ccuInstr) + "
                            "(SELECT COUNT(*) FROM ccuInstrRes), "
                            "(SELECT COUNT(*) FROM ccuChannels)";
    std::vector<std::vector<std::string>> rows;
    if (OpDbOps::Instance().ExecQuery(sql, {}, rows) != 0 || rows.empty() || rows[0].size() < 2) {
        HCCL_VM_ERROR("Query ccu resource meta count failed");
        return -1;
    }
    instrLoadCnt = std::stoul(rows[0][0]);
    channelCnt = std::stoul(rows[0][1]);
    return 0;
}

void AddSyncStreamIter(uint64_t streamId) { g_syncStreamIterMap.try_emplace(streamId, 0u); }

void RemoveSyncStreamIter(uint64_t streamId) { g_syncStreamIterMap.erase(streamId); }

bool NextStreamSyncIdx(uint64_t streamId, uint64_t& syncIdx)
{
    auto it = g_syncStreamIterMap.find(streamId);
    if (it == g_syncStreamIterMap.end()) {
        return false;
    }
    syncIdx = it->second.fetch_add(1);
    return true;
}

int QueryOpTaskTabNames(std::vector<std::string>& names)
{
    std::vector<std::vector<std::string>> rows;
    std::string querySql = "SELECT name FROM sqlite_master WHERE type='table' "
                           "AND name LIKE 'opTask_P_%'";
    auto ret = OpDbOps::Instance().ExecQuery(querySql, {}, rows);
    if (ret != 0) {
        HCCL_VM_ERROR("Get all opTask table name failed!");
        return ret;
    }

    for (const auto& row : rows) {
        if (!row.empty()) {
            names.push_back(row[0]);
        }
    }
    return ret;
}

int QueryCompositeOpDetailByOpIter(
    const std::string& commName, uint64_t commHash, uint32_t opIter, std::vector<CompositeOpDetail>& details)
{
    using HcclSim::DB::Field;

    // 定义 opDetails 表的列索引枚举 (共 16 列)
    enum : int {
        OD_ID,
        OD_PID,
        OD_DEVICE_ID,
        OD_RANK_ID,
        OD_COMM_ID,
        OD_OP_ITER,
        OD_SYNC_ITER,
        OD_STREAM_ID,
        OD_ROOT,
        OD_OP_EXPANSION,
        OD_DEV_TYPE,
        OD_RANK_SIZE,
        OD_SRC_RANK,
        OD_DST_RANK,
        OD_OP_DETAIL,
        OD_OP_EXT_INFO
    };
    const size_t OD_COL_COUNT = 16;

    // 定义 opMemInfo 表的列索引枚举 (共 8 列)
    enum : int {
        MEM_ID,
        MEM_OP_DETAIL_ID,
        MEM_INPUT_ADDR,
        MEM_INPUT_SIZE,
        MEM_OUTPUT_ADDR,
        MEM_OUTPUT_SIZE,
        MEM_CCL_ADDR,
        MEM_CCL_SIZE
    };
    const size_t MEM_COL_COUNT = 8;

    // 定义 opTask 表的列索引枚举 (共 6 列)
    enum : int {
        TASK_ID,
        TASK_PID,
        TASK_OP_DETAIL_ID,
        TASK_DEVICE_ID,
        TASK_STREAM_ID,
        TASK_TYPE,
        TASK_IS_DONE,
        TASK_META
    };
    const size_t TASK_COL_COUNT = 8;

    auto toStr = [](const Field& f) -> std::string {
        if (std::holds_alternative<std::string>(f)) {
            return std::get<std::string>(f);
        }
        return {};
    };

    auto commMembers
        = RunnerDB::GetByPred<sim::Communicator>([&commName, commHash](const sim::Communicator& record) -> bool {
              return std::string(record.comm_id) == commName && record.comm_hash == commHash;
          });

    for (const auto& comm : commMembers) {
        const std::string opSql = "SELECT id, pid, deviceId, rankId, commId, opIter, syncIter, "
                                  "streamId, "
                                  "root, opExpansionMode, devType, rankSize, srcRank, dstRank, "
                                  "opDetail, opExtInfo FROM opDetails WHERE commId = ? AND opIter = "
                                  "?";
        std::vector<Value> opParams = {(int64_t)comm.id, (int64_t)opIter};
        HcclSim::DB::ExRows opRows;
        if (OpDbOps::Instance().ExecQueryEx(opSql, opParams, opRows) != 0 || opRows.empty()
            || opRows[0].size() < OD_COL_COUNT) {
            HCCL_VM_ERROR("Query opDetails failed, commId={}, opIter={}", comm.id, opIter);
            return -1;
        }

        const auto& opRow = opRows[0];
        CompositeOpDetail comp = {};
        comp.deviceId = static_cast<uint32_t>(std::stoul(toStr(opRow[OD_DEVICE_ID])));
        comp.rankId = static_cast<uint32_t>(std::stoul(toStr(opRow[OD_RANK_ID])));
        comp.commId = std::stoull(toStr(opRow[OD_COMM_ID]));
        comp.detail.id = std::stoul(toStr(opRow[OD_ID]));
        comp.detail.pid = std::stoul(toStr(opRow[OD_PID]));
        comp.detail.deviceId = comp.deviceId;
        comp.detail.rankId = std::stoul(toStr(opRow[OD_RANK_ID]));
        comp.detail.commId = comp.commId;
        comp.detail.opIter = std::stoul(toStr(opRow[OD_OP_ITER]));
        comp.detail.syncIter = std::stoul(toStr(opRow[OD_SYNC_ITER]));
        comp.detail.streamId = std::stoull(toStr(opRow[OD_STREAM_ID]));
        comp.detail.root = std::stoul(toStr(opRow[OD_ROOT]));
        comp.detail.opExpansionMode = std::stoul(toStr(opRow[OD_OP_EXPANSION]));
        comp.detail.devType = std::stoul(toStr(opRow[OD_DEV_TYPE]));
        comp.detail.rankSize = std::stoul(toStr(opRow[OD_RANK_SIZE]));
        comp.detail.srcRank = std::stoul(toStr(opRow[OD_SRC_RANK]));
        comp.detail.dstRank = std::stoul(toStr(opRow[OD_DST_RANK]));
        if (std::holds_alternative<std::vector<uint8_t>>(opRow[OD_OP_DETAIL])) {
            comp.detail.opDetail = std::get<std::vector<uint8_t>>(opRow[OD_OP_DETAIL]);
        }
        if (std::holds_alternative<std::vector<uint8_t>>(opRow[OD_OP_EXT_INFO])) {
            comp.detail.opExtInfo = std::get<std::vector<uint8_t>>(opRow[OD_OP_EXT_INFO]);
        }

        // 查询 MemInfo
        const std::string memSql = "SELECT id, opDetailId, inputAddr, inputSize, "
                                   "outputAddr, outputSize, cclAddr, cclSize "
                                   "FROM opMemInfo WHERE opDetailId = ?";
        const uint32_t opDetailId = comp.detail.id;
        std::vector<Value> memParams = {(int64_t)opDetailId};
        std::vector<std::vector<std::string>> memRows;
        if (OpDbOps::Instance().ExecQuery(memSql, memParams, memRows) != 0 || memRows.empty()
            || memRows[0].size() < MEM_COL_COUNT) {
            HCCL_VM_ERROR("Query opDetails mem failed, opDetailId={}", opDetailId);
            return -1;
        }
        const auto& memRow = memRows[0];
        comp.memInfo.id = std::stoul(memRow[MEM_ID]);
        comp.memInfo.opDetailId = std::stoul(memRow[MEM_OP_DETAIL_ID]);
        comp.memInfo.inputAddr = std::stoull(memRow[MEM_INPUT_ADDR]);
        comp.memInfo.inputSize = std::stoull(memRow[MEM_INPUT_SIZE]);
        comp.memInfo.outputAddr = std::stoull(memRow[MEM_OUTPUT_ADDR]);
        comp.memInfo.outputSize = std::stoull(memRow[MEM_OUTPUT_SIZE]);
        comp.memInfo.cclAddr = std::stoull(memRow[MEM_CCL_ADDR]);
        comp.memInfo.cclSize = std::stoull(memRow[MEM_CCL_SIZE]);

        // 查询 Tasks
        const uint32_t pid = comp.detail.pid;
        const std::string taskTabName = "opTask_P_" + std::to_string(pid);
        const std::string taskSql = "SELECT id, pid, opDetailId, deviceId, streamId, taskType, isDone, "
                                    "opTaskMeta FROM "
                                    + taskTabName + " WHERE opDetailId = ?";
        std::vector<Value> taskParams = {(int64_t)opDetailId};
        HcclSim::DB::ExRows taskRows;
        if (OpDbOps::Instance().ExecQueryEx(taskSql, taskParams, taskRows) != 0 || taskRows.empty()) {
            HCCL_VM_ERROR("Query opDetails tasks failed, pid={}, opDetailId={}", pid, opDetailId);
            return -1;
        }
        for (const auto& taskRow : taskRows) {
            if (taskRow.size() < TASK_COL_COUNT) {
                HCCL_VM_ERROR("Query opDetails tasks failed, pid={}, opDetailId={}", pid, opDetailId);
                return -1;
            }
            OpTaskTab taskRec;
            taskRec.id = std::stoul(toStr(taskRow[TASK_ID]));
            taskRec.pid = std::stoul(toStr(taskRow[TASK_PID]));
            taskRec.opDetailId = std::stoul(toStr(taskRow[TASK_OP_DETAIL_ID]));
            taskRec.deviceId = std::stoull(toStr(taskRow[TASK_DEVICE_ID]));
            taskRec.streamId = std::stoull(toStr(taskRow[TASK_STREAM_ID]));
            taskRec.taskType = std::stoul(toStr(taskRow[TASK_TYPE]));
            taskRec.isDone = std::stoul(toStr(taskRow[TASK_IS_DONE]));
            if (std::holds_alternative<std::vector<uint8_t>>(taskRow[TASK_META])) {
                taskRec.optaskMeta = std::get<std::vector<uint8_t>>(taskRow[TASK_META]);
            }
            comp.tasks.push_back(std::move(taskRec));
        }

        details.push_back(std::move(comp));
    }

    return 0;
}

int QueryAllOpTasks(std::vector<OpTaskTab>& tasks, bool filterDone)
{
    std::vector<std::string> taskTabNames{};
    auto ret = QueryOpTaskTabNames(taskTabNames);
    if (ret != 0) {
        HCCL_VM_ERROR("QueryAllOpTasks failed!");
        return ret;
    }

    // 查询 Tasks
    enum : int {
        TASK_ID,
        TASK_PID,
        TASK_OP_DETAIL_ID,
        TASK_DEVICE_ID,
        TASK_STREAM_ID,
        TASK_TYPE,
        TASK_IS_DONE,
        TASK_META
    };
    const size_t TASK_COL_COUNT = 8;

    for (const std::string& taskTabName : taskTabNames) {
        std::string taskSql = "SELECT id, pid, opDetailId, deviceId, streamId, "
                              "taskType, isDone, opTaskMeta FROM "
                              + taskTabName;
        if (filterDone) {
            taskSql += " WHERE isDone = 0";
        }
        taskSql += " ORDER BY id";
        HcclSim::DB::ExRows taskRows;
        if (OpDbOps::Instance().ExecQueryEx(taskSql, {}, taskRows) != 0) {
            HCCL_VM_ERROR("Query OpTasks failed, tableName={}", taskTabName);
            return -1;
        }
        for (const auto& tr : taskRows) {
            // 校验列数
            if (tr.size() < TASK_COL_COUNT) {
                HCCL_VM_ERROR("Query OpTasks failed, tableName={}", taskTabName);
                continue;
            }
            OpTaskTab taskRec;
            taskRec.id = std::stoul(std::get<std::string>(tr[TASK_ID]));
            taskRec.pid = std::stoul(std::get<std::string>(tr[TASK_PID]));
            taskRec.opDetailId = std::stoul(std::get<std::string>(tr[TASK_OP_DETAIL_ID]));
            taskRec.deviceId = std::stoull(std::get<std::string>(tr[TASK_DEVICE_ID]));
            taskRec.streamId = std::stoull(std::get<std::string>(tr[TASK_STREAM_ID]));
            taskRec.taskType = std::stoul(std::get<std::string>(tr[TASK_TYPE]));
            taskRec.isDone = std::stoul(std::get<std::string>(tr[TASK_IS_DONE]));
            // 安全访问 Meta 数据
            if (std::holds_alternative<std::vector<uint8_t>>(tr[TASK_META])) {
                taskRec.optaskMeta = std::get<std::vector<uint8_t>>(tr[TASK_META]);
            }

            tasks.push_back(std::move(taskRec));
        }
    }
    return ret;
}

int QueryOpTasksByStreamId(uint64_t streamId, std::vector<OpTaskTab>& tasks)
{
    // 查询 Tasks
    enum : int {
        TASK_ID,
        TASK_PID,
        TASK_OP_DETAIL_ID,
        TASK_DEVICE_ID,
        TASK_STREAM_ID,
        TASK_TYPE,
        TASK_IS_DONE,
        TASK_META
    };
    const size_t TASK_COL_COUNT = 8;

    const auto tabPid = getpid();
    const std::string tabName = "opTask_P_" + std::to_string(tabPid);
    std::string sql = "SELECT id, pid, opDetailId, deviceId, streamId, "
                      "taskType, isDone, opTaskMeta FROM "
                      + tabName + " WHERE streamId = ?";
    std::vector<Value> taskParams = {(int64_t)streamId};
    HcclSim::DB::ExRows taskRows;
    auto ret = OpDbOps::Instance().ExecQueryEx(sql, taskParams, taskRows);
    if (ret != 0) {
        HCCL_VM_ERROR("Query OpTasks by streamId failed, tableName={:s}, streamId={}", tabName, streamId);
        return ret;
    }

    for (const auto& tr : taskRows) {
        // 校验列数
        if (tr.size() < TASK_COL_COUNT) {
            HCCL_VM_ERROR("Query OpTasks by streamId failed, tableName={:s}, streamId={}", tabName, streamId);
            continue;
        }
        OpTaskTab taskRec;
        taskRec.id = std::stoul(std::get<std::string>(tr[TASK_ID]));
        taskRec.pid = std::stoul(std::get<std::string>(tr[TASK_PID]));
        taskRec.opDetailId = std::stoul(std::get<std::string>(tr[TASK_OP_DETAIL_ID]));
        taskRec.deviceId = std::stoull(std::get<std::string>(tr[TASK_DEVICE_ID]));
        taskRec.streamId = std::stoull(std::get<std::string>(tr[TASK_STREAM_ID]));
        taskRec.taskType = std::stoul(std::get<std::string>(tr[TASK_TYPE]));
        taskRec.isDone = std::stoul(std::get<std::string>(tr[TASK_IS_DONE]));
        // 安全访问 Meta 数据
        if (std::holds_alternative<std::vector<uint8_t>>(tr[TASK_META])) {
            taskRec.optaskMeta = std::get<std::vector<uint8_t>>(tr[TASK_META]);
        }

        tasks.push_back(std::move(taskRec));
    }

    return ret;
}

int FinishOpTask(const OpTaskTab& task)
{
    const std::string taskTabName = "opTask_P_" + std::to_string(task.pid);
    std::string sql = "UPDATE " + taskTabName + " SET isDone = 1 WHERE id = ?";
    std::vector<Value> params = {(int64_t)task.id};
    int ret = OpDbOps::Instance().ExecUpdate(sql, params);
    if (ret != 0) {
        HCCL_VM_ERROR("Update OpTask isDone failed, taskTable:{:s}, id:{}", taskTabName, task.id);
    }
    return ret;
}

int QueryHalfRTTAll(std::vector<HalfRTTTab>& out)
{
    enum : int { ID, DEVICE_ID, DIE_ID, WISH_CNT_XN_BEGIN, WISH_CNT_XN_END, TOTAL_CNT_XN };
    const size_t HALF_RTT_COL_COUNT = 6;
    const std::string sql = "SELECT id, deviceId, dieId, wishCntXnIdBegin, "
                            "wishCntXnIdEnd, totalCntId FROM halfRTT";
    ExRows rows;
    if (OpDbOps::Instance().ExecQueryEx(sql, {}, rows) != 0) {
        return -1;
    }

    out.clear();
    for (const auto& r : rows) {
        if (r.size() < HALF_RTT_COL_COUNT) {
            continue;
        }
        HalfRTTTab rec = {};
        rec.id = std::stoul(std::get<std::string>(r[ID]));
        rec.deviceId = std::stoul(std::get<std::string>(r[DEVICE_ID]));
        rec.dieId = std::stoul(std::get<std::string>(r[DIE_ID]));
        rec.wishCntXnIdBegin = std::stoul(std::get<std::string>(r[WISH_CNT_XN_BEGIN]));
        rec.wishCntXnIdEnd = std::stoul(std::get<std::string>(r[WISH_CNT_XN_END]));
        rec.totalCntId = std::stoul(std::get<std::string>(r[TOTAL_CNT_XN]));
        out.push_back(std::move(rec));
    }
    return 0;
}
} // namespace sim
