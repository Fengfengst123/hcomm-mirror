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

#include "storage_manager.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib> // strtoull
#include <cstring>
#include <iostream>
#include <iterator>
#include <limits>
#include <nlohmann_json/json.hpp>
#include <set>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include "binary_data_operator.h"
#include "db_sim_communicator.h"
#include "error_codes.h"
#include "sim_log.h"

std::map<DeviceId, std::map<uint32_t, HcclSim::ChannelsPerDie>>
    g_allRankChannelInfo;
std::map<DeviceId, std::map<uint32_t, std::vector<HcclSim::HalfRTTInfo>>>
    g_allDeviceHalfRTTInfo;

namespace HcclSim {
static const std::string PLUGIN_PATH = "/plugin";
static const std::string DATA_FILE_PATH = "/data";
static const std::string TASK_COLLECTION_FILE = "/%s_task.jsonl.gz";
static const std::string MEM_LAYOUT_FILE = "/%s_mem_layout.jsonl.gz";
static const std::string MODEL_FILE = "/%s_model.jsonl.gz";

static const std::string HCCLVM_TASK_DATA_FILE = "/%s_hcclvm_task_data.bin";
static const std::string HCCLVM_SYN_DATA_FILE = "/%s_hcclvm_syn_data.bin";
static const std::string HCCLVM_INSTR_DATA_FILE = "/%s_hcclvm_instr_data.bin";

namespace {
template <typename T>
bool ReadExtValue(const std::vector<uint8_t> &data, size_t &offset, T &value) {
    if (offset > data.size() || data.size() - offset < sizeof(T)) {
        return false;
    }
    std::memcpy(&value, data.data() + offset, sizeof(T));
    offset += sizeof(T);
    return true;
}

bool ReadVParamExtInfo(const std::vector<uint8_t> &data, uint32_t rankSize,
                       VRankParam &param) {
    if (rankSize == 0) {
        return false;
    }

    constexpr size_t fixedSize = sizeof(uint64_t);
    const uint64_t valueCount = static_cast<uint64_t>(rankSize) * 2U;
    if (valueCount >
        (std::numeric_limits<size_t>::max() - fixedSize) / sizeof(uint64_t)) {
        return false;
    }
    const size_t expectedSize =
        fixedSize + static_cast<size_t>(valueCount) * sizeof(uint64_t);
    if (data.size() != expectedSize) {
        return false;
    }

    VRankParam parsed;
    parsed.counts.resize(rankSize);
    parsed.displs.resize(rankSize);
    size_t offset = 0;
    if (!ReadExtValue(data, offset, parsed.localCount)) {
        return false;
    }
    for (uint32_t i = 0; i < rankSize; ++i) {
        if (!ReadExtValue(data, offset, parsed.counts[i])) {
            return false;
        }
    }
    for (uint32_t i = 0; i < rankSize; ++i) {
        if (!ReadExtValue(data, offset, parsed.displs[i])) {
            return false;
        }
    }
    param = std::move(parsed);
    return true;
}

bool ReadBatchSendRecvExtInfo(const std::vector<uint8_t> &data,
                              BatchSendRecvRankParam &param) {
    if (data.size() != sizeof(uint32_t)) {
        return false;
    }
    std::memcpy(&param.itemNum, data.data(), sizeof(param.itemNum));
    return true;
}

HcclResult FinalizeVDataDes(CheckerParam &checkerParam, bool isAllGatherV) {
    const uint32_t rankSize = checkerParam.rankSize;
    if (rankSize == 0 || checkerParam.vRankParams.size() < rankSize) {
        HCCL_VM_ERROR(
            "Invalid V operator rank parameters, rankSize={}, reportedRanks={}",
            rankSize, checkerParam.vRankParams.size());
        return HcclResult::HCCL_E_PARA;
    }

    std::vector<uint64_t> finalCounts(rankSize, 0);
    for (uint32_t rankId = 0; rankId < rankSize; ++rankId) {
        const VRankParam &param = checkerParam.vRankParams[rankId];
        if (param.counts.size() != rankSize ||
            param.displs.size() != rankSize) {
            HCCL_VM_ERROR("Invalid V operator parameters reported by rank {}, "
                          "countsSize={}, "
                          "displsSize={}, rankSize={}",
                          rankId, param.counts.size(), param.displs.size(),
                          rankSize);
            return HcclResult::HCCL_E_PARA;
        }
        finalCounts[rankId] = param.localCount;
    }

    for (uint32_t sourceRank = 0; sourceRank < rankSize; ++sourceRank) {
        for (uint32_t targetRank = 0; targetRank < rankSize; ++targetRank) {
            const uint64_t reportedCount =
                isAllGatherV
                    ? checkerParam.vRankParams[targetRank].counts[sourceRank]
                    : checkerParam.vRankParams[sourceRank].counts[targetRank];
            if (reportedCount !=
                finalCounts[isAllGatherV ? sourceRank : targetRank]) {
                HCCL_VM_ERROR(
                    "{} V count mismatch, sourceRank={}, targetRank={}, "
                    "reportedCount={}, "
                    "expectedCount={}",
                    isAllGatherV ? "AllGather" : "ReduceScatter", sourceRank,
                    targetRank, reportedCount,
                    finalCounts[isAllGatherV ? sourceRank : targetRank]);
                return HcclResult::HCCL_E_PARA;
            }
        }
    }

    std::vector<uint64_t> displs;
    displs.reserve(rankSize);
    uint64_t offset = 0;
    for (uint64_t count : finalCounts) {
        displs.push_back(offset);
        if (count > std::numeric_limits<uint64_t>::max() - offset) {
            HCCL_VM_ERROR("{} V count prefix sum overflows uint64",
                          isAllGatherV ? "AllGather" : "ReduceScatter");
            return HcclResult::HCCL_E_PARA;
        }
        offset += count;
    }

    checkerParam.vDataDes = VDataDesTagInner{};
    checkerParam.vDataDes.dataType =
        static_cast<uint16_t>(checkerParam.dataType);
    checkerParam.vDataDes.count = rankSize;
    checkerParam.vDataDes.counts = std::move(finalCounts);
    checkerParam.vDataDes.displs = std::move(displs);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult FinalizeBatchSendRecvRing(CheckerParam &checkerParam) {
    const uint32_t rankSize = checkerParam.rankSize;
    if (rankSize < 2 ||
        checkerParam.batchSendRecvRankParams.size() != rankSize) {
        HCCL_VM_ERROR("Invalid BatchSendRecv ring report set, rankSize={}, "
                      "reportedRanks={}",
                      rankSize, checkerParam.batchSendRecvRankParams.size());
        return HcclResult::HCCL_E_PARA;
    }

    const uint64_t peerCount =
        checkerParam.batchSendRecvRankParams[0].peerCount;
    const HcclDataType dataType =
        checkerParam.batchSendRecvRankParams[0].dataType;
    for (uint32_t rankId = 0; rankId < rankSize; ++rankId) {
        const BatchSendRecvRankParam &current =
            checkerParam.batchSendRecvRankParams[rankId];
        const uint32_t expectedSendPeer = (rankId + 1U) % rankSize;
        const uint32_t expectedRecvPeer = (rankId + rankSize - 1U) % rankSize;
        if (current.itemNum != 2 || current.peerCount != peerCount ||
            current.dataType != dataType ||
            current.sendPeer != expectedSendPeer ||
            current.recvPeer != expectedRecvPeer) {
            HCCL_VM_ERROR("Invalid BatchSendRecv ring parameters at rank {}: "
                          "itemNum={}, peerCount={}, "
                          "dataType={}, sendPeer={}, recvPeer={}; expected "
                          "itemNum=2, peerCount={}, dataType={}, "
                          "sendPeer={}, recvPeer={}",
                          rankId, current.itemNum, current.peerCount,
                          static_cast<uint32_t>(current.dataType),
                          current.sendPeer, current.recvPeer, peerCount,
                          static_cast<uint32_t>(dataType), expectedSendPeer,
                          expectedRecvPeer);
            return HcclResult::HCCL_E_PARA;
        }
    }

    checkerParam.dataCount = peerCount;
    checkerParam.dataType = dataType;
    return HcclResult::HCCL_SUCCESS;
}

void UpdateNotifyPeerDevices(HcclVmTaskMetaData &taskMetaData) {
    std::unordered_map<uint64_t, std::set<uint32_t>> notifyId2Devices;
    for (const auto &taskMeta : taskMetaData.task_meta) {
        if (taskMeta.taskType == HccLTaskMetaType::NOTIFY_RECORD ||
            taskMeta.taskType == HccLTaskMetaType::NOTIFY_WAIT) {
            const uint64_t notifyId = taskMeta.taskData.notify.notifyId;
            notifyId2Devices[notifyId].insert(taskMeta.deviceId);
        }
    }

    for (auto &taskMeta : taskMetaData.task_meta) {
        if (taskMeta.taskType != HccLTaskMetaType::NOTIFY_RECORD &&
            taskMeta.taskType != HccLTaskMetaType::NOTIFY_WAIT) {
            continue;
        }
        uint32_t deviceId = taskMeta.deviceId;
        for (const uint32_t id :
             notifyId2Devices[taskMeta.taskData.notify.notifyId]) {
            if (id != deviceId) {
                deviceId = id;
                break;
            }
        }

        if (taskMeta.taskType == HccLTaskMetaType::NOTIFY_RECORD) {
            taskMeta.taskData.notify.dstDeviceId = deviceId;
        } else {
            taskMeta.taskData.notify.srcDeviceId = deviceId;
        }
    }
}

} // namespace

void StorageManager::Reset(bool clearMemLayout) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (clearMemLayout) {
        m_op_mem_layout.clear();
        m_comm_ccl_layout.clear();
        m_comm_device_rank_mappings.clear();
    }
    m_allRankChannelInfo.clear();
    m_main_stream_ids.clear();
    m_checker_param = CheckerParam{};
    m_checker_params.clear();
    m_all2AllvSendMatrices.clear();
    m_current_comm_name.clear();
    m_current_comm_hash = std::numeric_limits<uint64_t>::max();
    m_current_op_iter = 0;
    m_synData = HcclVmSynData{};
    m_instrData = HcclVmInstrData{};
    HCCL_VM_INFO("ZHF==Reset instrData");
    m_taskMeataData = HcclVmTaskMetaData{};
    devType_ = DevType::DEV_TYPE_COUNT;
    g_allRankChannelInfo.clear();
    g_allDeviceHalfRTTInfo.clear();
}

void StorageManager::BeginOpGroup(const std::string &commName,
                                  uint64_t commHash, uint32_t opIter) {
    m_checker_param = CheckerParam{};
    m_main_stream_ids.clear();
    m_all2AllvSendMatrices.clear();
    m_current_comm_name = commName;
    m_current_comm_hash = commHash;
    m_current_op_iter = opIter;
}

std::string StorageManager::GetCurrentCommName() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_current_comm_name;
}

uint64_t StorageManager::GetCurrentCommHash() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_current_comm_hash;
}

uint32_t StorageManager::GetCurrentOpIter() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_current_op_iter;
}

HcclResult StorageManager::Trans2CheckerParam(sim::OpDetailTab &detailTab,
                                              ::OpDetails &detail) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (detailTab.streamId >
        std::numeric_limits<TaskGraphGeneratorV3::StreamId>::max()) {
        HCCL_VM_ERROR(
            "Invalid operator main stream ID, deviceId={}, streamId={}",
            detailTab.deviceId, detailTab.streamId);
        return HcclResult::HCCL_E_PARA;
    }
    const auto mainStreamId =
        static_cast<TaskGraphGeneratorV3::StreamId>(detailTab.streamId);
    const auto mainStreamIt = m_main_stream_ids.find(detailTab.deviceId);
    if (mainStreamIt != m_main_stream_ids.end() &&
        mainStreamIt->second != mainStreamId) {
        HCCL_VM_ERROR("Inconsistent operator main stream for deviceId={}, "
                      "previousStreamId={}, streamId={}",
                      detailTab.deviceId, mainStreamIt->second, mainStreamId);
        return HcclResult::HCCL_E_PARA;
    }
    m_main_stream_ids[detailTab.deviceId] = mainStreamId;

    devType_ = static_cast<DevType>(detailTab.devType);
    auto vRankParams = std::move(m_checker_param.vRankParams);
    auto sendRecvPairs = std::move(m_checker_param.sendRecvPairs);
    auto batchSendRecvRankParams =
        std::move(m_checker_param.batchSendRecvRankParams);
    const CheckerParam previousParam = m_checker_param;
    m_checker_param = CheckerParam{};
    m_checker_param.vRankParams = std::move(vRankParams);
    m_checker_param.sendRecvPairs = std::move(sendRecvPairs);
    m_checker_param.batchSendRecvRankParams =
        std::move(batchSendRecvRankParams);
    m_checker_param.cmdType = static_cast<HcclCMDType>(detail.opType);
    m_checker_param.rankSize = detailTab.rankSize;
    m_checker_param.deviceId = detailTab.deviceId;
    m_checker_param.commId = detailTab.commId;
    m_checker_param.dataType = static_cast<HcclDataType>(detail.dataType);
    m_checker_param.dataCount = detail.opV1.count;
    m_checker_param.reduceType = static_cast<HcclReduceOp>(detail.reduceType);
    m_checker_param.srcRank = detailTab.srcRank;
    m_checker_param.dstRank = detailTab.dstRank;
    m_checker_param.root = detailTab.root;
    m_checker_param.all2AllDataDes.sendType = detail.opV2.sendDataType;
    m_checker_param.all2AllDataDes.recvType = detail.opV2.recvDataType;
    m_checker_param.all2AllDataDes.sendCount = detail.opV2.sendCount;
    m_checker_param.all2AllDataDes.recvCount = detail.opV2.recvCount;
    m_checker_param.all2AllDataDes.count = 0;

    HcclCMDType curCmdType = static_cast<HcclCMDType>(detail.opType);
    const bool isSendRecv = curCmdType == HcclCMDType::HCCL_CMD_SEND ||
                            curCmdType == HcclCMDType::HCCL_CMD_RECEIVE;
    if (isSendRecv) {
        if (!m_checker_param.sendRecvPairs.empty() &&
            (previousParam.rankSize != detailTab.rankSize ||
             previousParam.dataType != m_checker_param.dataType ||
             previousParam.dataCount != m_checker_param.dataCount)) {
            HCCL_VM_ERROR(
                "Inconsistent Send/Recv parameters in one op group, rankId={}",
                detailTab.rankId);
            return HcclResult::HCCL_E_PARA;
        }
        auto pair =
            std::find_if(m_checker_param.sendRecvPairs.begin(),
                         m_checker_param.sendRecvPairs.end(),
                         [&detailTab](const SendRecvPairParam &value) {
                             return value.srcRank == detailTab.srcRank &&
                                    value.dstRank == detailTab.dstRank;
                         });
        if (pair == m_checker_param.sendRecvPairs.end()) {
            m_checker_param.sendRecvPairs.push_back(
                {detailTab.srcRank, detailTab.dstRank, false, false});
            pair = std::prev(m_checker_param.sendRecvPairs.end());
        }
        bool &seen = curCmdType == HcclCMDType::HCCL_CMD_SEND ? pair->sendSeen
                                                              : pair->recvSeen;
        if (seen) {
            HCCL_VM_ERROR("Duplicate {} report for Send/Recv pair {} -> {}",
                          curCmdType == HcclCMDType::HCCL_CMD_SEND ? "Send"
                                                                   : "Recv",
                          pair->srcRank, pair->dstRank);
            return HcclResult::HCCL_E_PARA;
        }
        seen = true;
        // A group contains both entry types; use SEND as the canonical group
        // type.
        m_checker_param.cmdType = HcclCMDType::HCCL_CMD_SEND;
    }
    if (curCmdType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
        if (detailTab.rankSize == 0 || detailTab.rankId >= detailTab.rankSize) {
            HCCL_VM_ERROR("Invalid BatchSendRecv rank id {}, rankSize={}",
                          detailTab.rankId, detailTab.rankSize);
            return HcclResult::HCCL_E_PARA;
        }
        if (m_checker_param.batchSendRecvRankParams.size() <
            detailTab.rankSize) {
            m_checker_param.batchSendRecvRankParams.resize(detailTab.rankSize);
        }
        if (m_checker_param.batchSendRecvRankParams[detailTab.rankId].itemNum !=
            0) {
            HCCL_VM_ERROR(
                "Duplicate BatchSendRecv parameter report from rank {}",
                detailTab.rankId);
            return HcclResult::HCCL_E_PARA;
        }

        BatchSendRecvRankParam rankParam;
        if (!ReadBatchSendRecvExtInfo(detailTab.opExtInfo, rankParam)) {
            HCCL_VM_ERROR(
                "Invalid BatchSendRecv opExtInfo, rankId={}, payloadSize={}",
                detailTab.rankId, detailTab.opExtInfo.size());
            return HcclResult::HCCL_E_PARA;
        }
        const uint32_t expectedSendPeer =
            (detailTab.rankId + 1U) % detailTab.rankSize;
        const uint32_t expectedRecvPeer =
            (detailTab.rankId + detailTab.rankSize - 1U) % detailTab.rankSize;
        if (detailTab.rankSize < 2 || rankParam.itemNum != 2 ||
            detailTab.dstRank != expectedSendPeer ||
            detailTab.srcRank != expectedRecvPeer) {
            HCCL_VM_ERROR("BatchSendRecv is not a valid ring at rank {}: "
                          "itemNum={}, rankSize={}, "
                          "sendPeer={}, recvPeer={}",
                          detailTab.rankId, rankParam.itemNum,
                          detailTab.rankSize, detailTab.dstRank,
                          detailTab.srcRank);
            return HcclResult::HCCL_E_PARA;
        }
        rankParam.peerCount = detail.opV1.count;
        rankParam.dataType = static_cast<HcclDataType>(detail.dataType);
        rankParam.sendPeer = detailTab.dstRank;
        rankParam.recvPeer = detailTab.srcRank;
        m_checker_param.batchSendRecvRankParams[detailTab.rankId] = rankParam;
    }

    const bool isVOp = curCmdType == HcclCMDType::HCCL_CMD_ALLGATHER_V ||
                       curCmdType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V;
    if (isVOp) {
        if (detailTab.rankId >= detailTab.rankSize) {
            HCCL_VM_ERROR("Invalid V operator rank id {}, rankSize={}",
                          detailTab.rankId, detailTab.rankSize);
            return HcclResult::HCCL_E_PARA;
        }
        if (m_checker_param.vRankParams.size() < detailTab.rankSize) {
            m_checker_param.vRankParams.resize(detailTab.rankSize);
        }
        VRankParam &rankParam = m_checker_param.vRankParams[detailTab.rankId];
        if (!rankParam.counts.empty()) {
            HCCL_VM_ERROR("Duplicate V operator parameter report from rank {}",
                          detailTab.rankId);
            return HcclResult::HCCL_E_PARA;
        }
        if (!ReadVParamExtInfo(detailTab.opExtInfo, detailTab.rankSize,
                               rankParam)) {
            HCCL_VM_ERROR("Invalid V operator opExtInfo, opType={}, rankId={}, "
                          "rankSize={}, payloadSize={}",
                          static_cast<uint32_t>(curCmdType), detailTab.rankId,
                          detailTab.rankSize, detailTab.opExtInfo.size());
            return HcclResult::HCCL_E_PARA;
        }
        if (detail.opV1.count != rankParam.localCount) {
            HCCL_VM_ERROR("V operator local count mismatch, opType={}, "
                          "rankId={}, detailCount={}, extInfoCount={}",
                          static_cast<uint32_t>(curCmdType), detailTab.rankId,
                          detail.opV1.count, rankParam.localCount);
            return HcclResult::HCCL_E_PARA;
        }
    } else if ((curCmdType == HcclCMDType::HCCL_CMD_ALLTOALL ||
                curCmdType == HcclCMDType::HCCL_CMD_ALLTOALLV ||
                curCmdType == HcclCMDType::HCCL_CMD_ALLTOALLVC) &&
               detailTab.opExtInfo.size() >= sizeof(uint32_t)) {
        uint32_t count = 0;
        std::memcpy(&count, detailTab.opExtInfo.data(), sizeof(uint32_t));
        m_checker_param.all2AllDataDes.count = count;

        std::vector<uint64_t> currentMatrix;
        for (uint32_t i = 0; i < count; i++) {
            uint64_t val = 0;
            size_t offset = sizeof(uint32_t) + i * sizeof(uint64_t);
            if (offset + sizeof(uint64_t) <= detailTab.opExtInfo.size()) {
                std::memcpy(&val, detailTab.opExtInfo.data() + offset,
                            sizeof(uint64_t));
            }
            currentMatrix.push_back(val);
        }

        bool needMerge = (curCmdType == HcclCMDType::HCCL_CMD_ALLTOALL ||
                          curCmdType == HcclCMDType::HCCL_CMD_ALLTOALLV ||
                          curCmdType == HcclCMDType::HCCL_CMD_ALLTOALLVC);

        if (needMerge) {
            m_all2AllvSendMatrices[detailTab.rankId] = currentMatrix;
        }
        m_checker_param.all2AllDataDes.sendCountMatrix = currentMatrix;
    }

    HCCL_VM_INFO("opIter={}, rankId={}", detailTab.opIter, detailTab.rankId);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult StorageManager::FinalizeOpGroup() {
    switch (m_checker_param.cmdType) {
    case HcclCMDType::HCCL_CMD_ALLGATHER_V:
        return FinalizeVDataDes(m_checker_param, true);
    case HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V:
        return FinalizeVDataDes(m_checker_param, false);
    case HcclCMDType::HCCL_CMD_BATCH_SEND_RECV:
        return FinalizeBatchSendRecvRing(m_checker_param);
    case HcclCMDType::HCCL_CMD_SEND:
    case HcclCMDType::HCCL_CMD_RECEIVE:
        for (const auto &pair : m_checker_param.sendRecvPairs) {
            if (!pair.sendSeen || !pair.recvSeen ||
                pair.srcRank == pair.dstRank ||
                pair.srcRank >= m_checker_param.rankSize ||
                pair.dstRank >= m_checker_param.rankSize) {
                HCCL_VM_ERROR("Incomplete or invalid Send/Recv pair {} -> {} "
                              "(sendSeen={}, recvSeen={})",
                              pair.srcRank, pair.dstRank, pair.sendSeen,
                              pair.recvSeen);
                return HcclResult::HCCL_E_PARA;
            }
        }
        if (m_checker_param.sendRecvPairs.empty()) {
            return HcclResult::HCCL_E_PARA;
        }
        m_checker_param.srcRank = m_checker_param.sendRecvPairs.front().srcRank;
        m_checker_param.dstRank = m_checker_param.sendRecvPairs.front().dstRank;
        return HcclResult::HCCL_SUCCESS;
    default:
        return HcclResult::HCCL_SUCCESS;
    }
}

void StorageManager::MergeAll2AllVSendCountMatrix() {
    if (m_all2AllvSendMatrices.empty()) {
        return;
    }

    uint32_t rankSize = m_checker_param.rankSize;
    uint32_t matrixSize = rankSize * rankSize;
    uint32_t numRanks = static_cast<uint32_t>(m_all2AllvSendMatrices.size());
    std::vector<uint64_t> mergedMatrix(matrixSize, 0);

    for (auto &pair : m_all2AllvSendMatrices) {
        const auto &partial = pair.second;
        if (partial.size() == matrixSize) {
            for (size_t i = 0; i < matrixSize; i++) {
                mergedMatrix[i] |= partial[i];
            }
        }
    }

    m_checker_param.all2AllDataDes.sendCountMatrix = mergedMatrix;
    m_checker_param.all2AllDataDes.count = matrixSize;
    m_all2AllvSendMatrices.clear();

    HCCL_VM_INFO("Merged {} ranks into {}x{} matrix", numRanks, rankSize,
                 rankSize);
}

HcclResult StorageManager::LoadHcclVmSynthesisData(
    DeviceId deviceId, CommId commId, const std::string &commName,
    uint64_t commHash, uint32_t opIter, uint32_t rankId,
    sim::OpMemInfoTab memInfo, std::vector<sim::CcuChannelTab> &channels,
    std::vector<sim::HalfRTTTab> &halfRTT) {
    std::vector<sim::CommunicatorMemberInfo> members;
    if (!sim::GetCommunicatorMembers(commId, members)) {
        HCCL_VM_ERROR("failed to resolve communicator members, commId={}",
                      commId);
        return HCCL_E_PARA;
    }
    for (const sim::CommunicatorMemberInfo &commMember : members) {
        auto &deviceRankMappings =
            m_comm_device_rank_mappings[commMember.memberId];
        for (const sim::CommunicatorMemberInfo &member : members) {
            deviceRankMappings[static_cast<DeviceId>(member.deviceId)] =
                member.rankId;
        }
    }
    // 转换channel映射表，key 使用 deviceId（物理设备 ID）
    for (auto &channel : channels) {
        RemoteDieInfo rmtDieInfo1;
        rmtDieInfo1.dstDeviceId = channel.dstDeviceId;
        rmtDieInfo1.remoteDieId = channel.dstDieId;
        HCCL_VM_INFO("[Channel info] channelId= {}, srcDeviceId= {}, srcDie= "
                     "{}, dstDeviceId= {}, dstDie= {}",
                     channel.channelId, channel.srcDeviceId,
                     static_cast<uint32_t>(channel.srcDieId),
                     channel.dstDeviceId, channel.dstDieId);
        g_allRankChannelInfo[channel.srcDeviceId][channel.srcDieId]
                            [channel.channelId] = rmtDieInfo1;
    }

    const CommIdentity commIdentity{commName, commHash};
    auto &opLayout = m_op_mem_layout[commIdentity][opIter];
    auto &cclLayout = m_comm_ccl_layout[commIdentity];

    // Input/output buffer 属于一次算子执行。
    // 转换half RTT映射表
    for (auto &rtt : halfRTT) {
        HalfRTTInfo rttInfo;
        rttInfo.wishCntXnBegin = rtt.wishCntXnIdBegin;
        rttInfo.wishCntXnEnd = rtt.wishCntXnIdEnd;
        rttInfo.totalCntXn = rtt.totalCntId;
        g_allDeviceHalfRTTInfo[rtt.deviceId][rtt.dieId].push_back(rttInfo);
    }
    // 构造memory layout
    if (memInfo.inputAddr != 0 && memInfo.inputSize > 0) {
        MemBlock memBlock;
        memBlock.bufferType = BufferType::INPUT;
        memBlock.startAddr = memInfo.inputAddr;
        memBlock.size = memInfo.inputSize;
        memBlock.globalOffset =
            0; // todo:
               // 预期一个rank只有一个同类型的buffer时，globalOffset为0。若有多个，需要按照下面json方案计算
        opLayout[deviceId][BufferType::INPUT][memInfo.inputAddr] = memBlock;
        HCCL_VM_INFO("[Init MemLayout] deviceId={}, rankId={}, bufType= {}, "
                     "startAddr={}, size={}, globalOffset= {}",
                     deviceId, rankId, static_cast<int>(BufferType::INPUT),
                     memInfo.inputAddr, memInfo.inputSize,
                     memBlock.globalOffset);
    }

    if (memInfo.outputAddr != 0 && memInfo.outputSize > 0) {
        MemBlock memBlock;
        memBlock.bufferType = BufferType::OUTPUT;
        memBlock.startAddr = memInfo.outputAddr;
        memBlock.size = memInfo.outputSize;
        memBlock.globalOffset =
            0; // todo:
               // 预期一个rank只有一个同类型的buffer时，globalOffset为0。若有多个，需要按照下面json方案计算
        opLayout[deviceId][BufferType::OUTPUT][memInfo.outputAddr] = memBlock;
        HCCL_VM_INFO("[Init MemLayout] deviceId={}, rankId={}, bufType= {}, "
                     "startAddr={}, size={}, globalOffset= {}",
                     deviceId, rankId, static_cast<int>(BufferType::OUTPUT),
                     memInfo.outputAddr, memInfo.outputSize,
                     memBlock.globalOffset);
    }

    if (memInfo.cclAddr != 0 && memInfo.cclSize > 0) {
        MemBlock memBlock;
        memBlock.bufferType = BufferType::CCL;
        memBlock.startAddr = memInfo.cclAddr;
        memBlock.size = memInfo.cclSize;
        memBlock.globalOffset =
            0; // todo:
               // 预期一个rank只有一个同类型的buffer时，globalOffset为0。若有多个，需要按照下面json方案计算
        // CCL buffer 由通信域持有，并在该通信域的多个算子之间共享。
        cclLayout[deviceId][BufferType::CCL][memInfo.cclAddr] = memBlock;
        HCCL_VM_INFO("[Init MemLayout] deviceId={}, rankId={}, bufType= {}, "
                     "startAddr={}, size={}, globalOffset= {}",
                     deviceId, rankId, static_cast<int>(BufferType::CCL),
                     memInfo.cclAddr, memInfo.cclSize, memBlock.globalOffset);
    }
    return HcclResult::HCCL_SUCCESS;
}

std::map<DeviceId, RankId> StorageManager::GetDeviceRankMappings() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_checker_param.commId != INVALID_COMM_ID) {
        const auto iter =
            m_comm_device_rank_mappings.find(m_checker_param.commId);
        if (iter != m_comm_device_rank_mappings.end()) {
            return iter->second;
        }
    }
    if (m_comm_device_rank_mappings.size() == 1) {
        return m_comm_device_rank_mappings.begin()->second;
    }
    return {};
}

std::map<DeviceId, RankId>
StorageManager::GetDeviceRankMappings(CommId commId) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    const auto iter = m_comm_device_rank_mappings.find(commId);
    return iter == m_comm_device_rank_mappings.end()
               ? std::map<DeviceId, RankId>{}
               : iter->second;
}

bool StorageManager::GetDeviceIdByCommRank(CommId commId, RankId rankId,
                                           DeviceId &deviceId) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    const auto commIter = m_comm_device_rank_mappings.find(commId);
    if (commIter == m_comm_device_rank_mappings.end()) {
        return false;
    }
    for (const auto &mapping : commIter->second) {
        if (mapping.second == rankId) {
            deviceId = mapping.first;
            return true;
        }
    }
    return false;
}

bool StorageManager::GetMainStreamId(
    DeviceId deviceId, TaskGraphGeneratorV3::StreamId &streamId) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    const auto iter = m_main_stream_ids.find(deviceId);
    if (iter == m_main_stream_ids.end()) {
        return false;
    }
    streamId = iter->second;
    return true;
}

void StorageManager::InitCcuInfo(DevType &devType,
                                 std::vector<uint64_t> &resourceBaseAddr) {
    devType = devType_;
    resourceBaseAddr.clear();
    resourceBaseAddr.push_back(0x123123123);
    resourceBaseAddr.push_back(0x456456456);
}

HcclResult StorageManager::LoadHcclVmInstrData(
    std::vector<sim::CcuInstrResTab> &instrRes) {
    // The instruction table describes the complete resource snapshot for the
    // current checker run. Replace the previous snapshot so repeated loading
    // from the single-op and big-graph paths cannot duplicate instructions.
    m_instrData.instr_data.clear();
    for (auto &instr : instrRes) {
        if (instr.instrCount == 0 || instr.instrCount > 32 * 1024) {
            HCCL_VM_WARN("invalid instrCount={}, deviceId={}, dieId={}",
                         instr.instrCount, instr.deviceId,
                         static_cast<uint32_t>(instr.dieId));
            continue;
        }

        MicrocodeInstrInner instrInner;
        instrInner.desc.rank_id = instr.deviceId;
        instrInner.desc.die_id = static_cast<uint8_t>(instr.dieId);
        instrInner.desc.count = static_cast<uint16_t>(instr.instrCount);

        instrInner.data.reserve(instr.instrCount);
        for (uint32_t i = 0; i < instr.instrCount; ++i) {
            hcomm::CcuRep::CcuInstr ccuInstr;
            std::memcpy(&ccuInstr, instr.instrSpace[i],
                        sizeof(hcomm::CcuRep::CcuInstr));
            instrInner.data.push_back(ccuInstr);
        }

        m_instrData.instr_data.push_back(std::move(instrInner));

        HCCL_VM_INFO("deviceId={}, dieId={}, count={}", instr.deviceId,
                     static_cast<uint32_t>(instr.dieId), instr.instrCount);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult StorageManager::LoadHcclVmTaskMetaData(
    std::vector<std::vector<sim::OpTaskTab>> &allTasks) {
    size_t totalTasks = 0;
    for (const auto &rankTasks : allTasks) {
        totalTasks += rankTasks.size();
    }
    HCCL_VM_INFO("total tasks: {}, ranks: {}", totalTasks, allTasks.size());
    HcclVmTaskMetaData taskMeataData;
    for (const auto &rankTasks : allTasks) {
        for (const auto &task : rankTasks) {
            if (task.optaskMeta.size() >= sizeof(HcclTaskMetaData)) {
                HcclTaskMetaData metaData;
                std::memcpy(&metaData, task.optaskMeta.data(),
                            sizeof(HcclTaskMetaData));
                taskMeataData.task_meta.push_back(metaData);
            } else {
                HCCL_VM_WARN("optaskMeta too small, src:{:d}, dst:{:d}",
                             task.optaskMeta.size(), sizeof(HcclTaskMetaData));
            }
        }
    }
    m_taskMeataData = taskMeataData;
    UpdateNotifyPeerDevices(m_taskMeataData);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult StorageManager::LoadDecodedHcclVmTaskMetaData(
    const std::vector<std::vector<HcclTaskMetaData>> &allTaskMetas) {
    size_t totalTasks = 0;
    for (const auto &rankTaskMetas : allTaskMetas) {
        totalTasks += rankTaskMetas.size();
    }
    HCCL_VM_INFO("total decoded tasks: {}, ranks: {}", totalTasks,
                 allTaskMetas.size());

    HcclVmTaskMetaData taskMeataData;
    taskMeataData.task_meta.reserve(totalTasks);
    for (const auto &rankTaskMetas : allTaskMetas) {
        taskMeataData.task_meta.insert(taskMeataData.task_meta.end(),
                                       rankTaskMetas.begin(),
                                       rankTaskMetas.end());
    }

    m_taskMeataData = std::move(taskMeataData);
    UpdateNotifyPeerDevices(m_taskMeataData);
    return HcclResult::HCCL_SUCCESS;
}

uint64_t StorageManager::GetBlockSize(const std::string &commName,
                                      uint64_t commHash, uint32_t opIter,
                                      DeviceId deviceId,
                                      BufferType bufferType) {
    const DeviceMemLayout *layout = nullptr;
    if (bufferType == BufferType::CCL) {
        const auto cclIt = m_comm_ccl_layout.find({commName, commHash});
        if (cclIt != m_comm_ccl_layout.end()) {
            layout = &cclIt->second;
        }
    } else {
        const auto commIt = m_op_mem_layout.find({commName, commHash});
        if (commIt != m_op_mem_layout.end()) {
            const auto opIt = commIt->second.find(opIter);
            if (opIt != commIt->second.end()) {
                layout = &opIt->second;
            }
        }
    }
    if (layout == nullptr) {
        return 0;
    }
    auto rankIt = layout->find(deviceId);
    if (rankIt == layout->end()) {
        HCCL_VM_INFO("Cannot find device id from memory layout");
        return 0;
    }

    // 2. 定位 BufferType
    auto typeIt = rankIt->second.find(bufferType);
    if (typeIt == rankIt->second.end()) {
        HCCL_VM_INFO("Cannot find buffer type from memory layout");
        return 0;
    }

    // 3. 获取该类型下的最后一个块 (map 的 rbegin)
    const auto &addrMap = typeIt->second;
    if (addrMap.empty()) {
        HCCL_VM_INFO("Cannot find addr from memory layout");
        return 0;
    }

    // map 是有序的，rbegin() 指向起始地址最大的那个块
    const MemBlock &lastBlock = addrMap.rbegin()->second;

    // 总大小 = 最后一个块的逻辑起始偏移 + 最后一个块的大小
    return lastBlock.globalOffset + lastBlock.size;
}

// 取指定通信域/算子/设备/缓冲类型下的基址（绝对地址基址快照的来源）。
bool StorageManager::GetBufferBaseAddress(const std::string &commName,
                                          uint64_t commHash, uint32_t opIter,
                                          DeviceId deviceId,
                                          BufferType bufferType,
                                          uint64_t &baseAddr) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    const DeviceMemLayout *layout = nullptr;
    if (bufferType == BufferType::CCL) {
        const auto cclIt = m_comm_ccl_layout.find({commName, commHash});
        if (cclIt != m_comm_ccl_layout.end()) {
            layout = &cclIt->second;
        }
    } else {
        const auto commIt = m_op_mem_layout.find({commName, commHash});
        if (commIt != m_op_mem_layout.end()) {
            const auto opIt = commIt->second.find(opIter);
            if (opIt != commIt->second.end()) {
                layout = &opIt->second;
            }
        }
    }
    if (layout == nullptr) {
        return false;
    }
    const auto rankIt = layout->find(deviceId);
    if (rankIt == layout->end()) {
        return false;
    }
    const auto typeIt = rankIt->second.find(bufferType);
    if (typeIt == rankIt->second.end() || typeIt->second.empty()) {
        return false;
    }
    baseAddr = typeIt->second.begin()->second.startAddr;
    return true;
}

// 在设备上注册的全部布局块（op 布局 + CCL 布局）中查找完整包含 [addr, addr+len)
// 的块。 绝对地址包含校验以此为准，不依赖可能被 inplace 首命中误判的 memType。
bool StorageManager::IsMemRangeRegistered(const std::string &commName,
                                          uint64_t commHash, uint32_t opIter,
                                          DeviceId deviceId, uint64_t addr,
                                          uint64_t len) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    const DeviceMemLayout *opLayout = nullptr;
    const auto commIt = m_op_mem_layout.find({commName, commHash});
    if (commIt != m_op_mem_layout.end()) {
        const auto opIt = commIt->second.find(opIter);
        if (opIt != commIt->second.end()) {
            opLayout = &opIt->second;
        }
    }
    const DeviceMemLayout *cclLayout = nullptr;
    const auto cclIt = m_comm_ccl_layout.find({commName, commHash});
    if (cclIt != m_comm_ccl_layout.end()) {
        cclLayout = &cclIt->second;
    }
    const DeviceMemLayout *layouts[] = {opLayout, cclLayout};
    for (const DeviceMemLayout *layout : layouts) {
        if (layout == nullptr) {
            continue;
        }
        const auto rankIt = layout->find(deviceId);
        if (rankIt == layout->end()) {
            continue;
        }
        for (const auto &typeEntry : rankIt->second) {
            for (const auto &addrEntry : typeEntry.second) {
                const MemBlock &block = addrEntry.second;
                if (addr >= block.startAddr && len <= block.size &&
                    addr - block.startAddr <= block.size - len) {
                    return true;
                }
            }
        }
    }
    return false;
}

HcclResult StorageManager::GetSlice(const std::string &commName,
                                    uint64_t commHash, uint32_t opIter,
                                    uint64_t addr, uint64_t len,
                                    DataSlice &dataSlice, DeviceId *deviceId) {
    dataSlice.SetSize(len);

    const auto commIt = m_op_mem_layout.find({commName, commHash});
    const DeviceMemLayout *opLayout = nullptr;
    if (commIt != m_op_mem_layout.end()) {
        const auto opIt = commIt->second.find(opIter);
        if (opIt != commIt->second.end()) {
            opLayout = &opIt->second;
        }
    }
    const auto cclIt = m_comm_ccl_layout.find({commName, commHash});
    const DeviceMemLayout *cclLayout =
        cclIt == m_comm_ccl_layout.end() ? nullptr : &cclIt->second;

    auto findSlice = [&](const DeviceMemLayout &layout) -> HcclResult {
        for (const auto &[actualDeviceId, deviceMem] : layout) {
            for (const auto &typeEntry : deviceMem) {
                const auto &addrMap = typeEntry.second;
                auto it = addrMap.upper_bound(addr);
                if (it == addrMap.begin()) {
                    continue;
                }
                --it;
                const MemBlock &block = it->second;
                if (addr < block.startAddr ||
                    addr >= block.startAddr + block.size) {
                    continue;
                }
                dataSlice.SetBufferType(block.bufferType);
                dataSlice.SetRawAddr(addr);
                dataSlice.SetOffset(block.globalOffset + addr -
                                    block.startAddr);
                if (deviceId != nullptr) {
                    *deviceId = actualDeviceId;
                }
                return HcclResult::HCCL_SUCCESS;
            }
        }
        return HcclResult::HCCL_E_MEMORY;
    };

    if (opLayout != nullptr &&
        findSlice(*opLayout) == HcclResult::HCCL_SUCCESS) {
        return HcclResult::HCCL_SUCCESS;
    }
    if (cclLayout != nullptr &&
        findSlice(*cclLayout) == HcclResult::HCCL_SUCCESS) {
        return HcclResult::HCCL_SUCCESS;
    }
    HCCL_VM_ERROR("{} Failed to resolve data slice from memory layout, "
                  "addr=0x{:X}, len=0x{:X}",
                  MakeErrorCodeText(ErrorCode::GRAPH_ADDRESS_INVALID), addr,
                  len);
    return HcclResult::HCCL_E_MEMORY;
}

std::string StorageManager::FindRootPath() {
    // 使用相对路径前缀：.  ./..  ./../..
    std::string current_search_path = ".";
    for (int i = 0; i <= 3; ++i) {
        char abs_path[PATH_MAX];
        // 尝试获取当前探测点的绝对路径
        if (realpath(current_search_path.c_str(), abs_path) != nullptr) {
            std::string check_target = std::string(abs_path) + PLUGIN_PATH;

            // 检查该绝对路径下的 plugin 目录是否存在
            if (IsDirExists(check_target)) {
                return std::string(abs_path);
            }
        } else {
            // 如果 realpath 失败（例如路径被删除或权限不足）
            HCCL_VM_ERROR("Iteration {}: realpath failed for {}", i,
                          current_search_path);
        }

        // 没找到，将探测路径向上推一级
        current_search_path += "/..";
    }

    HCCL_VM_INFO("RootPath NOT found.");
    return "";
}

bool StorageManager::IsDirExists(const std::string &path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        return false; // 不存在
    }
    return (info.st_mode & S_IFDIR); // 存在且是目录
}

uint32_t StorageManager::GetRankSize() const {
    return m_checker_param.rankSize;
}

HcclVmInstrData StorageManager::GetHvmInstrData() const { return m_instrData; }

HcclVmTaskMetaData StorageManager::GetHvmTaskMetaData() const {
    return m_taskMeataData;
}
} // namespace HcclSim
