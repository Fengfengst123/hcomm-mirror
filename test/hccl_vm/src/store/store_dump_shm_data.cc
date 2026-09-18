/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "store_dump_shm_data.h"

#include <cstdint>
#include <cstdlib> // strtoull
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>
#include <utility>

#include "operation_data/operation_data_ops.h"
#include "runtime_state/db_sim_runner_common.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "runtime_state/sim_models.h"
#include "sim_capacity_limits.h"
#include "sim_common_api.h"
#include "sim_common_defs.h"
#include "sim_ip_address.h"
#include "sim_loader.h"
#include "sim_log.h"
#include "sim_yaml_config.h"
#include "storage/table_access.h"
#include "store_binary_data_operator.h"

uint8_t g_opExpansionMode = 0;
static const std::string HCCLVM_FLAG_DATA_FILE = "/hcclvm_flag_data.bin";
static const std::string HCCLVM_TASK_DATA_FILE = "/%s_hcclvm_task_data.bin";
static const std::string HCCLVM_SYN_DATA_FILE = "/%s_hcclvm_syn_data.bin";
static const std::string HCCLVM_INSTR_DATA_FILE = "/%s_hcclvm_instr_data.bin";

namespace HcclSim {
namespace fs = std::filesystem;

namespace {
    HcclVmResult LoadCompositeDataByOpIter(
        const sim::operation::OpExecutionKey& key,
        std::map<uint32_t, std::vector<sim::operation::CompositeOpDetail>>& compositeDataMap)
    {
        std::vector<sim::operation::CompositeOpDetail> details;
        loader::Loader dataLoader;
        if (dataLoader.LoadCompositeOpDetailByOpIter(key.commName, key.commHash, key.opIter, details)
            != HcclResult::HCCL_SUCCESS) {
            HCCL_VM_ERROR(
                "LoadCompositeOpDetailByOpIter failed, commName={}, "
                "commHash={}, opIter={}",
                key.commName, key.commHash, key.opIter);
            return HcclVmResult::HCCL_SIM_E_INTERNAL;
        }
        for (sim::operation::CompositeOpDetail& detail : details) {
            compositeDataMap[detail.rankId].push_back(std::move(detail));
        }
        return HcclVmResult::HCCL_SIM_SUCCESS;
    }

} // namespace

std::string GetBinLocation()
{
    std::filesystem::path curPath = std::filesystem::current_path();
    return curPath.string();
}

std::string GenDataId()
{
    std::string dataId;

    // 1. 初始化随机数生成器 (static 保证只初始化一次，提高性能和随机性)
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1000, 9999); // 4位随机数
    int32_t retryCount = 10;
    while (true) {
        if (retryCount == 0) {
            HCCL_VM_ERROR("failed to gen dataId.");
            break;
        }
        retryCount--;
        // 2. 获取当前时间戳
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        struct tm buf;
        localtime_r(&in_time_t, &buf);

        // 3. 拼接时间 + 随机数
        std::stringstream ss;
        ss << std::put_time(&buf, "%Y%m%d_%H%M%S") << "_" << dis(gen);
        dataId = ss.str();

        // 4. 冲突检查：检查核心文件是否已存在
        char fileName[256];
        snprintf(fileName, sizeof(fileName), "/%s_model.jsonl.gz", dataId.c_str());
        std::string rootPath = GetBinLocation();
        std::string fullPath = rootPath + DATA_FILE_PATH + fileName;

        if (access(fullPath.c_str(), F_OK) == -1) {
            break; // 文件不存在，DataId 可用，跳出循环
        }
        // 如果冲突，循环会立即重新生成（时间或随机数会变）
    }

    return dataId;
}

HcclVmResult DumpHcclVmFlagData(HcclSim::HcclVmFlagData& flagData)
{
    HCCL_VM_INFO("Start dumping hccl vm flag data...");
    // 1. 构造完整路径
    // 假设 FindRootPath() 已经实现并返回插件根目录
    std::string rootPath = GetBinLocation();
    std::string fullPath = rootPath + DATA_FILE_PATH + HCCLVM_FLAG_DATA_FILE;

    // 2. 写文件，通知runner启动
    FILE* fp = fopen(fullPath.c_str(), "wb");
    if (!fp) {
        HCCL_VM_ERROR("Open file failed: {}, error={}", fullPath, strerror(errno));
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    auto ret = HcclVmFlagDataWrite(fp, flagData);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("HcclVmFlagDataWrite failed. ");
        fclose(fp);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    fclose(fp);
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult GetHcclVmFlagData(HcclSim::HcclVmFlagData& waitFlag)
{
    // 1. 构造完整路径
    // 假设 FindRootPath() 已经实现并返回插件根目录
    std::string rootPath = GetBinLocation();
    std::string fullPath = rootPath + DATA_FILE_PATH + HCCLVM_FLAG_DATA_FILE;
    // 2. 读取文件，等待runner状态
    FILE* fp = fopen(fullPath.c_str(), "rb");
    if (!fp) {
        HCCL_VM_ERROR("Open file failed: {}, error={}", fullPath, strerror(errno));
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    auto ret = HcclVmFlagDataRead(fp, waitFlag, HCCLVM_FLAG_FILE_MAGIC);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("HcclVmFlagDataRead failed. ");
        fclose(fp);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    fclose(fp);
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DumpDataToFile(const std::string& dataId)
{
    loader::Loader dataLoader;
    std::vector<sim::operation::OpExecutionKey> keys;
    if (dataLoader.LoadOpExecutionKeys(keys) != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR("LoadOpExecutionKeys failed, dataId={}", dataId);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    // 仅当恰好存在一个算子执行记录时才 dump; 0 个或大于 1 个都视为异常，不 dump
    // 并返回错误.
    if (keys.size() != 1) {
        HCCL_VM_WARN(
            "Expected exactly one operator execution to dump, but got "
            "{:d}, dataId={}, skip dump",
            keys.size(), dataId);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    const sim::operation::OpExecutionKey& key = keys.front();
    HCCL_VM_INFO(
        "Dumping op dataId={}, commName={}, commHash={}, opIter={}", dataId, key.commName, key.commHash, key.opIter);

    HcclVmResult ret = DumpHcclVmSynthesisData(dataId, key);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("failed to dump hccl vm synthesis data, opIter={}", key.opIter);
        return ret;
    }
    ret = DumpHcclVmInstrData(dataId);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("failed to dump hccl vm instruction data, opIter={}", key.opIter);
        return ret;
    }
    ret = DumpHcclVmTask(dataId, key);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("failed to dump hccl vm task data, opIter={}", key.opIter);
        return ret;
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult CreateMemoryInfo(HcclVmSynData& hvmSynData, const sim::operation::OpMemInfoTab& memInfo, uint32_t rankId)
{
    HCCL_VM_INFO("Enter into create memory info...");
    auto addBuf = [&](uint8_t bufType, uint64_t addr, uint64_t size) {
        if (addr != 0 && size > 0) {
            MemLayoutData memLayoutData{};
            memLayoutData.rank_id = rankId;
            memLayoutData.buffer_type = bufType;
            memLayoutData.start_addr = addr;
            memLayoutData.size = size;
            memLayoutData.global_offset = 0; // 简化约定，与 checker 侧一致
            hvmSynData.memory_info.data.push_back(memLayoutData);
        }
    };
    addBuf(static_cast<uint8_t>(BufferType::INPUT), memInfo.inputAddr, memInfo.inputSize);
    addBuf(static_cast<uint8_t>(BufferType::OUTPUT), memInfo.outputAddr, memInfo.outputSize);
    addBuf(static_cast<uint8_t>(BufferType::CCL), memInfo.cclAddr, memInfo.cclSize);

    hvmSynData.memory_info.count = hvmSynData.memory_info.data.size();
    HCCL_VM_INFO("memory_info count={}, dataSize={}", hvmSynData.memory_info.count, hvmSynData.memory_info.data.size());

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult CreateChannelInfo(HcclVmSynData& hvmSynData)
{
    HCCL_VM_INFO("Enter into create channel info...");

    std::vector<sim::operation::CcuChannelTab> channels;
    loader::Loader dataLoader;
    if (dataLoader.GetCcuChannelInfo(channels) != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR("GetCcuChannelInfo() failed.");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    for (const auto& ch : channels) {
        ChannelData chData{};
        chData.channelId = static_cast<uint16_t>(ch.channelId);
        chData.srcDieId = static_cast<uint8_t>(ch.srcDieId);
        chData.dstDieId = static_cast<uint8_t>(ch.dstDieId);
        chData.srcDeviceId = ch.srcDeviceId;
        chData.dstDeviceId = ch.dstDeviceId;

        std::memcpy(chData.leid, ch.leid, sizeof(chData.leid));
        std::memcpy(chData.reid, ch.reid, sizeof(chData.reid));

        uint8_t* leidPtr = chData.leid;
        auto lEpRet = sim::runtime::Db::GetOneByPred<sim::runtime::EndPoint>(HcclSim::Storage::Eq(
            &sim::runtime::EndPoint::eid, std::vector<uint8_t>(
                                              reinterpret_cast<const uint8_t*>(leidPtr),
                                              reinterpret_cast<const uint8_t*>(leidPtr) + sizeof(chData.leid))));
        if (!lEpRet.ok()) {
            HCCL_VM_ERROR("cannot find EndPoint by ip addr: {}", static_cast<const void*>(leidPtr));
            return HcclVmResult::HCCL_SIM_E_NOT_FOUND;
        }

        uint8_t* reidPtr = chData.reid;
        auto rEpRet = sim::runtime::Db::GetOneByPred<sim::runtime::EndPoint>(HcclSim::Storage::Eq(
            &sim::runtime::EndPoint::eid, std::vector<uint8_t>(
                                              reinterpret_cast<const uint8_t*>(reidPtr),
                                              reinterpret_cast<const uint8_t*>(reidPtr) + sizeof(chData.reid))));
        if (!rEpRet.ok()) {
            HCCL_VM_ERROR("cannot find EndPoint by ip addr:{}", static_cast<const void*>(reidPtr));
            return HcclVmResult::HCCL_SIM_E_NOT_FOUND;
        }

        auto localEpId = lEpRet->id;
        auto remoteEpId = rEpRet->id;

        auto pairOpt = sim::runtime::Db::GetOneByPred<sim::runtime::EndPointPair>(HcclSim::Storage::And(
            HcclSim::Storage::Eq(&sim::runtime::EndPointPair::local_enpoint_id, localEpId),
            HcclSim::Storage::Eq(&sim::runtime::EndPointPair::remote_enpoint_id, remoteEpId)));

        if (!pairOpt.ok()) {
            HCCL_VM_ERROR("cannot find EndPointPair by local: {} remote: {}", localEpId, remoteEpId);
            return HcclVmResult::HCCL_SIM_E_NOT_FOUND;
        }

        chData.protocol = pairOpt->tp_type;
        if (ch.jettyNum > HcclSim::HCCL_VM_MAX_CHANNEL_JETTIES) {
            HCCL_VM_ERROR(
                "channelId={} jettyNum={} exceeds capacity={}", ch.channelId, ch.jettyNum,
                HcclSim::HCCL_VM_MAX_CHANNEL_JETTIES);
            return HcclVmResult::HCCL_SIM_E_PARA;
        }
        chData.jettyNum = ch.jettyNum;
        std::memcpy(chData.jettyId, ch.jettyId, static_cast<size_t>(chData.jettyNum) * sizeof(uint32_t));

        HCCL_VM_INFO(
            "channelId={}, srcDeviceId={}, dstDeviceId={}, "
            "jettyNum={}, protocol={}",
            chData.channelId, chData.srcDeviceId, chData.dstDeviceId, chData.jettyNum, chData.protocol);

        hvmSynData.channel_info.data.push_back(chData);
    }
    hvmSynData.channel_info.count = hvmSynData.channel_info.data.size();
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult CreateJettyInfo(HcclVmSynData& hvmSynData)
{
    HCCL_VM_INFO("Enter into create jetty info...");
    auto endpointPairs
        = sim::runtime::Db::GetByPred<sim::runtime::EndPointPair>(HcclSim::Storage::All<sim::runtime::EndPointPair>());
    if (!endpointPairs.ok() || !endpointPairs.value.has_value()) {
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    for (const auto& endpoint : *endpointPairs.value) {
        uint64_t localEndPointId = endpoint.local_enpoint_id;
        uint64_t rmtEndPointId = endpoint.remote_enpoint_id;
        uint8_t protocol = endpoint.tp_type;

        // 根据enpoint_id查找本端和对端的EndPoint信息
        auto localEndPointOpt = sim::runtime::Db::GetById<sim::runtime::EndPoint>(localEndPointId);
        if (!localEndPointOpt.ok()) {
            HCCL_VM_ERROR("Get EndPoint failed handle = {}", localEndPointId);
            continue;
        }

        auto rmtEndPointOpt = sim::runtime::Db::GetById<sim::runtime::EndPoint>(rmtEndPointId);
        if (!rmtEndPointOpt.ok()) {
            HCCL_VM_ERROR("Get EndPoint failed handle = {}", rmtEndPointId);
            continue;
        }

        ChannelData channelData;
        channelData.channelId = endpoint.id; // AICPU模式无用赋值为主键
        channelData.protocol = protocol;

        memcpy(channelData.leid, localEndPointOpt->eid, sizeof(channelData.leid));
        channelData.srcDieId = localEndPointOpt->die_id;
        channelData.srcDeviceId = localEndPointOpt->device_id;

        memcpy(channelData.reid, rmtEndPointOpt->eid, sizeof(channelData.reid));
        channelData.dstDieId = rmtEndPointOpt->die_id;
        channelData.dstDeviceId = rmtEndPointOpt->device_id;

        // 查表查询JettyNum和JettyId
        auto raCtx = sim::runtime::Db::GetOneByPred<sim::runtime::RaContext>(
            HcclSim::Storage::Eq(&sim::runtime::RaContext::endpoint_id, localEndPointId));
        if (!raCtx.ok()) {
            HCCL_VM_ERROR("Get RaContext failed handle = {}", localEndPointId);
            continue;
        }

        uint64_t raCtxHandle = raCtx->id;
        auto raJettys = sim::runtime::Db::GetByPred<sim::runtime::RaJetty>(HcclSim::Storage::And(
            HcclSim::Storage::Eq(&sim::runtime::RaJetty::ctx_handle, raCtxHandle),
            HcclSim::Storage::Eq(&sim::runtime::RaJetty::mode, 3)));
        if (!raJettys.ok() || !raJettys.value.has_value() || raJettys.value->empty()) {
            continue;
        }
        if (raJettys.value->size() > HcclSim::HCCL_VM_MAX_CHANNEL_JETTIES) {
            HCCL_VM_ERROR(
                "endpointId={} jettyNum={} exceeds capacity={}", endpoint.id, raJettys.value->size(),
                HcclSim::HCCL_VM_MAX_CHANNEL_JETTIES);
            return HcclVmResult::HCCL_SIM_E_PARA;
        }
        channelData.jettyNum = raJettys.value->size();
        for (size_t i = 0; i < raJettys.value->size(); i++) {
            channelData.jettyId[i] = raJettys.value->at(i).jetty_id;
        }

        hvmSynData.channel_info.data.push_back(channelData);
    }
    hvmSynData.channel_info.count = hvmSynData.channel_info.data.size();

    // 打印测试-JettyInfo
    std::stringstream chDataStr;
    for (auto chData : hvmSynData.channel_info.data) {
        chDataStr << "channelId=" << chData.channelId << ", srcDeviceId=" << chData.srcDeviceId
                  << ", dstDeviceId=" << chData.dstDeviceId;
        chDataStr << ", srcDieId=" << (int)chData.srcDieId << ", dstDieId=" << (int)chData.dstDieId;
        chDataStr << ", protocol=" << chData.protocol;
        chDataStr << ", leid=";
        for (int i = 0; i < sizeof(chData.leid); i++) {
            chDataStr << std::hex << std::setw(2) << std::setfill('0') << std::hex << chData.leid[i];
        }

        chDataStr << ", reid=";
        for (int i = 0; i < sizeof(chData.reid); i++) {
            chDataStr << std::hex << std::setw(2) << std::setfill('0') << std::hex << chData.reid[i];
        }
        chDataStr << std::endl;

        chDataStr << "jettyNum=" << chData.jettyNum << " ";
        for (int i = 0; i < chData.jettyNum; i++) {
            chDataStr << chData.jettyId[i] << " ";
        }
        chDataStr << std::endl;
    }
    HCCL_VM_INFO("{}", chDataStr.str());
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult CreateSimSynData(HcclVmSynData& hvmSynData, const sim::operation::OpExecutionKey& key)
{
    HCCL_VM_INFO("Start get simulator synthesis data...");
    // header
    hvmSynData.header.magic = HCCLVM_SYN_FILE_MAGIC;
    hvmSynData.header.version = 1;
    hvmSynData.header.header_size = 20;
    hvmSynData.header.count = 1;

    std::map<uint32_t, std::vector<sim::operation::CompositeOpDetail>> compositeDataMap;
    const auto loadRet = LoadCompositeDataByOpIter(key, compositeDataMap);
    if (loadRet != HcclVmResult::HCCL_SIM_SUCCESS) {
        return loadRet;
    }
    const auto rank0It = compositeDataMap.find(0);
    if (rank0It == compositeDataMap.end() || rank0It->second.empty()) {
        HCCL_VM_ERROR(
            "No op detail found for rankId=0, commName={}, "
            "commHash={}, opIter={}",
            key.commName, key.commHash, key.opIter);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    const auto& compRank0 = rank0It->second.front();
    const auto& opTab = compRank0.detail;
    const auto& memInfo = compRank0.memInfo;
    uint32_t rankId = compRank0.rankId;
    if (opTab.opDetail.size() < sizeof(OpDetails)) {
        HCCL_VM_ERROR("opDetail BLOB too small: {} < {}", opTab.opDetail.size(), sizeof(OpDetails));
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    ::OpDetails opDetails{};
    std::memcpy(&opDetails, opTab.opDetail.data(), sizeof(OpDetails));
    g_opExpansionMode = opTab.opExpansionMode;
    hvmSynData.model_info.comm.root = opTab.root;
    hvmSynData.model_info.comm.rank_size = opTab.rankSize;
    hvmSynData.model_info.comm.chip_type = static_cast<uint16_t>(opTab.devType);
    hvmSynData.model_info.comm.op_type = opDetails.opType;
    hvmSynData.model_info.comm.reduce_op = opDetails.reduceType;
    hvmSynData.model_info.comm.data_type = opDetails.dataType;
    hvmSynData.model_info.comm.data_count = opDetails.opV1.count;
    hvmSynData.model_info.comm.op_expansion_mode = opTab.opExpansionMode;
    // 保留占位（与 checker 侧一致）
    hvmSynData.model_info.comm.ccu0_resource_base_addr = 0x123123123;
    hvmSynData.model_info.comm.ccu1_resource_base_addr = 0x456456456;
    hvmSynData.model_info.all2AllDataDes.sendType = opDetails.opV2.sendDataType;
    hvmSynData.model_info.all2AllDataDes.recvType = opDetails.opV2.recvDataType;
    hvmSynData.model_info.all2AllDataDes.sendCount = opDetails.opV2.sendCount;
    hvmSynData.model_info.all2AllDataDes.recvCount = opDetails.opV2.recvCount;
    // opExtInfo 解析 count + sendCountMatrix
    hvmSynData.model_info.all2AllDataDes.count = 0;
    const HcclCMDType cmdType = static_cast<HcclCMDType>(opDetails.opType);
    const bool isAll2AllOp = cmdType == HcclCMDType::HCCL_CMD_ALLTOALL || cmdType == HcclCMDType::HCCL_CMD_ALLTOALLV
                             || cmdType == HcclCMDType::HCCL_CMD_ALLTOALLVC;
    if (isAll2AllOp && opTab.opExtInfo.size() >= sizeof(uint32_t)) {
        uint32_t cnt = 0;
        std::memcpy(&cnt, opTab.opExtInfo.data(), sizeof(uint32_t));
        hvmSynData.model_info.all2AllDataDes.count = cnt;
        for (uint32_t i = 0; i < cnt; i++) {
            uint64_t val = 0;
            size_t offset = sizeof(uint32_t) + i * sizeof(uint64_t);
            if (offset + sizeof(uint64_t) <= opTab.opExtInfo.size()) {
                std::memcpy(&val, opTab.opExtInfo.data() + offset, sizeof(uint64_t));
            }
            hvmSynData.model_info.all2AllDataDes.sendCountMatrix.push_back(val);
        }
    }
    HCCL_VM_INFO("all2AllDataDes send count: {}", hvmSynData.model_info.all2AllDataDes.count);

    // ChannelInfo(CCU) or JettyInfo(AICPU)
    HcclVmResult channelResult;
    if (hvmSynData.model_info.comm.op_expansion_mode == sim::runtime::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_AICPU) {
        channelResult = CreateJettyInfo(hvmSynData);
    } else {
        channelResult = CreateChannelInfo(hvmSynData);
    }
    if (channelResult != HcclVmResult::HCCL_SIM_SUCCESS) {
        return channelResult;
    }

    // MemoryLayout
    CreateMemoryInfo(hvmSynData, memInfo, rankId);

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DumpHcclVmSynthesisData(const std::string& dataId, const sim::operation::OpExecutionKey& key)
{
    HCCL_VM_INFO("Start dumping hccl vm synthesis data...");
    // 1. 构造完整路径
    char fileName[256];
    snprintf(fileName, sizeof(fileName), HCCLVM_SYN_DATA_FILE.c_str(), dataId.c_str());

    fs::create_directories(fs::path(InstallPath::ResolveToInstallRoot("data")));
    std::string fullPath = InstallPath::ResolveToInstallRoot("data" + std::string(fileName));

    //  构造hccl vm synthesis数据
    HcclVmSynData hvmSynData;
    auto ret = CreateSimSynData(hvmSynData, key);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("Get hccl vm synthesis data failed.");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    FILE* fp = fopen(fullPath.c_str(), "wb");
    if (!fp) {
        HCCL_VM_ERROR("Open file failed: {}, err: {}", fullPath, strerror(errno));
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    // 3. dump为二进制文件
    ret = HcclVmSynDataWrite(fp, hvmSynData);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        fclose(fp);
        HCCL_VM_ERROR("Write file failed.");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    fclose(fp);
    HCCL_VM_INFO("Write file success.");
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult CreateSimInstrData(HcclVmInstrData& hvmInstrData)
{
    HCCL_VM_INFO("Start create simulator instruction data...");
    // 原来实现中 sizeof(hcomm::CcuRep::CcuInstr) == 32判断（与
    // instrSpace[i][32] 对齐），加 static_assert
    static_assert(sizeof(hcomm::CcuRep::CcuInstr) == 32, "CcuInstr must be 32 bytes");

    std::vector<sim::operation::CcuInstrResTab> allInstrRes;
    loader::Loader dataLoader;
    auto ret = dataLoader.GetInstrResInfo(allInstrRes);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR("Get instr res info failed.");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    // 过滤掉 instrCount == 0 的记录 等价原来的 Ccustatus.state == 1 的判断
    std::vector<sim::operation::CcuInstrResTab> validInstrRes;
    for (const auto& instrRes : allInstrRes) {
        if (instrRes.instrCount > 0) {
            validInstrRes.push_back(instrRes);
        }
    }

    // header
    hvmInstrData.header.magic = HCCLVM_INSTR_FILE_MAGIC;
    hvmInstrData.header.version = 1;
    hvmInstrData.header.header_size = 20;
    hvmInstrData.header.count = validInstrRes.size(); // CCU有微码指令的个数

    for (const auto& instrRes : validInstrRes) {
        MicrocodeInstrInner mcInstr;
        mcInstr.desc.rank_id = instrRes.deviceId;
        mcInstr.desc.die_id = instrRes.dieId;
        mcInstr.desc.count = instrRes.instrCount;

        if (mcInstr.desc.count > HcclSim::CCU_INSTRUCTION_NUM) {
            HCCL_VM_ERROR(
                "instrCount {} exceeds instruction capacity {}", mcInstr.desc.count, HcclSim::CCU_INSTRUCTION_NUM);
            return HcclVmResult::HCCL_SIM_E_INTERNAL;
        }

        mcInstr.data.resize(mcInstr.desc.count);
        auto size = sizeof(hcomm::CcuRep::CcuInstr) * mcInstr.desc.count;
        memcpy(mcInstr.data.data(), instrRes.instrSpace, size);

        hvmInstrData.instr_data.push_back(mcInstr);
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DumpHcclVmInstrData(const std::string& dataId)
{
    HCCL_VM_INFO("Start dumping hccl vm instruction data...");
    // 1. 构造完整路径
    char fileName[256];
    snprintf(fileName, sizeof(fileName), HCCLVM_INSTR_DATA_FILE.c_str(), dataId.c_str());

    fs::create_directories(fs::path(InstallPath::ResolveToInstallRoot("data")));
    std::string fullPath = InstallPath::ResolveToInstallRoot("data" + std::string(fileName));
    FILE* fp = fopen(fullPath.c_str(), "wb");
    if (!fp) {
        HCCL_VM_ERROR("Open file failed: {}, err: {}", fullPath, strerror(errno));
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    // 2. 构造hccl vm instruction数据
    HcclVmInstrData hvmInstrData;
    auto ret = CreateSimInstrData(hvmInstrData);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        fclose(fp);
        HCCL_VM_ERROR("Get hccl vm instruction data failed.");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    // 非CCU模式无微码指令，无需执行步骤3、步骤4
    if (g_opExpansionMode != sim::runtime::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_CCU) {
        fclose(fp);
        return HcclVmResult::HCCL_SIM_SUCCESS;
    }

    // 3. 检查指令是否为空
    if (hvmInstrData.instr_data.empty()) {
        fclose(fp);
        HCCL_VM_WARN("There is no microcode instruction.");
        return HcclVmResult::HCCL_SIM_SUCCESS;
    }

    // 4. dump为二进制文件
    ret = HcclVmInstrDataWrite(fp, hvmInstrData);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        fclose(fp);
        HCCL_VM_ERROR("Write file failed.");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    fclose(fp);
    HCCL_VM_INFO("Write file success.");
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult CreateSimTaskMetaData(
    HcclVmTaskMetaData& hvmTaskMetaData,
    const std::map<uint32_t, std::vector<sim::operation::CompositeOpDetail>>& compositeDataMap)
{
    HCCL_VM_INFO("Start create simulator task metadata...");

    // 遍历 compositeDataMap，收集所有 task
    for (const auto& rankEntry : compositeDataMap) {
        for (const auto& comp : rankEntry.second) {
            for (const auto& task : comp.tasks) {
                // 反序列化 optaskMeta BLOB
                if (task.optaskMeta.size() < sizeof(HcclTaskMetaData)) {
                    HCCL_VM_WARN("optaskMeta too small: {} < {}", task.optaskMeta.size(), sizeof(HcclTaskMetaData));
                    continue;
                }

                HcclTaskMetaData taskMeta;
                std::memcpy(&taskMeta, task.optaskMeta.data(), sizeof(HcclTaskMetaData));

                // AIV_GRAPH 特殊处理
                if (taskMeta.taskType == HccLTaskMetaType::AIV_GRAPH) {
                    taskMeta.taskData.aiv.launchIdx = hvmTaskMetaData.task_meta.size();
                }

                hvmTaskMetaData.task_meta.push_back(taskMeta);
            }
        }
    }

    // 构造 header
    hvmTaskMetaData.header.magic = HCCLVM_TASK_FILE_MAGIC;
    hvmTaskMetaData.header.version = 1;
    hvmTaskMetaData.header.header_size = 20;
    hvmTaskMetaData.header.count = hvmTaskMetaData.task_meta.size(); // task个数

    HCCL_VM_INFO("taskCount = {}", hvmTaskMetaData.header.count);

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DumpHcclVmTask(const std::string& dataId, const sim::operation::OpExecutionKey& key)
{
    HCCL_VM_INFO("Start dumping hccl vm task data...");
    // 1. 构造完整路径
    char fileName[256];
    snprintf(fileName, sizeof(fileName), HCCLVM_TASK_DATA_FILE.c_str(), dataId.c_str());

    fs::create_directories(fs::path(InstallPath::ResolveToInstallRoot("data")));
    std::string fullPath = InstallPath::ResolveToInstallRoot("data" + std::string(fileName));

    FILE* fp = fopen(fullPath.c_str(), "wb");
    if (!fp) {
        HCCL_VM_ERROR("Open file failed: {}, err: {}", fullPath, strerror(errno));
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    // 2. 获取 compositeDataMap
    std::map<uint32_t, std::vector<sim::operation::CompositeOpDetail>> compositeDataMap;
    if (LoadCompositeDataByOpIter(key, compositeDataMap) != HcclVmResult::HCCL_SIM_SUCCESS) {
        fclose(fp);
        HCCL_VM_ERROR("LoadCompositeDataByOpIter failed.");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    // 3. 构造 hccl vm task meta 数据
    HcclVmTaskMetaData hvmTaskMeta;
    auto ret = CreateSimTaskMetaData(hvmTaskMeta, compositeDataMap);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        fclose(fp);
        HCCL_VM_ERROR("Get hccl vm task data failed.");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    // 4. 检查任务是否为空
    auto allDevicesResult = sim::runtime::Db::GetByPred<sim::runtime::Device>(
        HcclSim::Storage::Eq(&sim::runtime::Device::status, uint64_t{1}));
    if (!allDevicesResult.ok() || !allDevicesResult.value.has_value()) {
        fclose(fp);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    const auto& allDevices = *allDevicesResult.value;
    // 单卡流程，hccl业务不会生成任务，不需要报错
    if (hvmTaskMeta.task_meta.empty() && allDevices.size() > 1) {
        fclose(fp);
        HCCL_VM_ERROR("There is no task.");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    // 5. dump 为二进制文件
    ret = HcclVmTaskMetaDataWrite(fp, hvmTaskMeta);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        fclose(fp);
        HCCL_VM_ERROR("Write hccl vm task file failed.");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    fclose(fp);
    HCCL_VM_INFO("Write hccl vm task file success.");
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult
GenCaModelCcuInstr(hcomm::CcuRep::CcuInstr* instrData, uint32_t instrCnt, uint32_t deviceId, uint32_t dieId)
{
    char instrFileName[256] = {0};
    sprintf(instrFileName, "data/CaModel_instruction_0_%u_%u_sq.toml", deviceId, dieId + 2);
    HCCL_VM_INFO("instrFileName = {}", instrFileName);
    auto fullPath = InstallPath::ResolveToInstallRoot(instrFileName);

    FILE* fp = fopen(fullPath.c_str(), "w");
    if (fp == nullptr) {
        HCCL_VM_ERROR("open file fail: {}", fullPath.c_str());
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    uint64_t* ptr = (uint64_t*)instrData;
    for (uint32_t idx = 0; idx < instrCnt; idx++) {
        for (uint32_t j = 0; j < 4; j++) {
            if (j == 3) {
                fprintf(fp, "%lu\r\n", ptr[idx * 4 + j]);
            } else {
                fprintf(fp, "%lu\t", ptr[idx * 4 + j]);
            }
        }
    }
    fclose(fp);
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult GenCaModelCcuToml(rtCcuTaskInfo_t* taskInfo, uint32_t deviceId, uint32_t streamId)
{
    char tomlFileName[256] = {0};
    sprintf(tomlFileName, "data/CaModel_ccu_%u_%u_%u_sq.toml", deviceId, taskInfo->dieId + 2, streamId);
    HCCL_VM_INFO("tomlFileName = {}", tomlFileName);

    auto fullPath = InstallPath::ResolveToInstallRoot(tomlFileName);

    FILE* fp = fopen(fullPath.c_str(), "w");
    if (fp == nullptr) {
        HCCL_VM_ERROR("open file fail: {}", fullPath.c_str());
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    fprintf(fp, "name = \" ccu \"\r\n");
    fprintf(fp, "[[tasks]]\r\n");
    fprintf(fp, "sqe_type = \"CCU\"\r\n");
    fprintf(fp, "mission_id = %u\r\n", taskInfo->missionId);
    fprintf(fp, "iodie_id = %u\r\n", taskInfo->dieId);
    fprintf(fp, "inst_start_id = %u\r\n", taskInfo->instStartId);
    fprintf(fp, "inst_cnt = %u\r\n", taskInfo->instCnt);
    // fprintf(fp, "inst_addr_key_value = %u\r\n", taskInfo->instAddrKeyValue);
    // // key是什么？
    fprintf(fp, "args = [");
    for (uint32_t idx = 0; idx < taskInfo->argSize; idx++) {
        fprintf(fp, "%lu", taskInfo->args[idx]);
        if (idx < taskInfo->argSize - 1) {
            fprintf(fp, ",");
        }
    }
    fprintf(fp, "]");

    fclose(fp);

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

} // namespace HcclSim
