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

#ifndef SIM_OP_DB_OPS_H
#define SIM_OP_DB_OPS_H

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "sim_op_db_types.h"

namespace sim {
extern uint32_t g_currOpDetailId;
extern uint32_t g_currOpMemId;

/**
 * @brief 当前进程正在处理的算子迭代（= 最近一次 InsertOpDetail 分配的
 * opIter）。
 *
 * 用途：CCU 交换表（CcuSyncResTab）需要跨 rank 一致的 op
 * 身份来隔离不同算子的交换值。 opIter 由 InsertOpDetail 按 (commName, commHash)
 * 自增，两侧进程按同一算子序列 推进时天然对齐，可直接当 op 身份使用。
 *
 * @return 最新分配的 opIter；尚无算子插入时为 0。
 */
uint32_t GetCurOpIter();

int InitOpDataDb();
int SetDbConfig(DBConfig &config);

void AddSyncStreamIter(uint64_t streamId);
void RemoveSyncStreamIter(uint64_t streamId);
bool NextStreamSyncIdx(uint64_t streamId, uint64_t &syncIdx);

int InsertOpDetail(OpDetailTab &rec);
int InsertOpMem(OpMemInfoTab &rec);
int InsertOpDetailAndMem(OpDetailTab &detail, OpMemInfoTab &mem);
int InsertCcuChannel(CcuChannelTab &rec);
int InsertJettyMap(JettyMapTab &rec);
int InsertOpTask(OpTaskTab &rec, bool isDevice = false);
int InsertSyncRecord(SyncRecordTab &rec);
int InsertCcuInstrRes(CcuInstrResTab &rec);
int InsertCcuInstr(CcuInstrTab &rec);
int InsertHalfRTT(HalfRTTTab &rec);

int UpdateAndInsertByCcuId(uint64_t &ccuId, uint32_t deviceId, uint32_t dieId,
                           uint32_t startId, uint32_t instrCount,
                           uint32_t instrOffset, uint32_t instrInfoSize,
                           const void *instrInfo,
                           std::vector<uint8_t> *mergedInstrSpace = nullptr,
                           uint32_t *totalInstrCount = nullptr);
int UpdateSyncRecordStatus(std::vector<SyncRecordTab> &syncRecord);
int UpdateOpMemCclBuffer(uint64_t cclAddr, uint64_t cclSize);
int UpdateOpExpansionMode(uint8_t mode);
int UpdateOpDetailMainStream(uint32_t opDetailId, uint64_t streamId);

int QueryLatestOpExpansionMode();
int QueryCcuChannelAll(std::vector<CcuChannelTab> &out);
int QueryJettyMapAll(std::vector<JettyMapTab> &out);
int QuerySyncRecordAll(std::vector<SyncRecordTab> &out);
int QueryCcuInstrResAll(std::vector<CcuInstrResTab> &out);
int QueryCcuResourceMetaCount(uint32_t &instrLoadCnt, uint32_t &channelCnt);
int QueryOpDetailIdentity(uint32_t opDetailId, uint64_t &commId,
                          uint32_t &rankId, uint32_t &deviceId);
int QueryOpDetailById(uint32_t opDetailId, OpDetailTab &out);
int QueryLatestOpDetailByDeviceId(uint32_t deviceId, OpDetailTab &out);
int QueryCurrentOpMemInfo(uint64_t commId, uint32_t deviceId,
                          OpMemInfoTab &out);
int QueryOpMemInfoByOpDetailId(uint32_t opDetailId, OpMemInfoTab &out);
int QuerySyncRecordByStatus(uint8_t status, std::vector<SyncRecordTab> &out);
int QueryOpExecutionIndexEntries(std::vector<OpExecutionIndexEntry> &out);
int QueryCompositeOpDetailBySyncIter(
    uint32_t syncIter,
    std::map<uint32_t, std::vector<CompositeOpDetail>> &detail);
int QueryHalfRTTAll(std::vector<HalfRTTTab> &out);
int QueryOpTaskTabNames(std::vector<std::string> &names);
int QueryCompositeOpDetailByOpIter(const std::string &commName,
                                   uint64_t commHash, uint32_t opIter,
                                   std::vector<CompositeOpDetail> &details);
int QueryAllOpTasks(std::vector<OpTaskTab> &tasks, bool filterDone = false);
int QueryOpTasksByStreamId(uint64_t streamId, std::vector<OpTaskTab> &tasks);
int FinishOpTask(const OpTaskTab &task);
int QueryHalfRTTAll(std::vector<HalfRTTTab> &out);
} // namespace sim

#endif
