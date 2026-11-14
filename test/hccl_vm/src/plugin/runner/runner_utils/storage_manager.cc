/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "storage_manager.h"

#include <cstdint>
#include <cstdlib> // strtoull
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <nlohmann_json/json.hpp>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "store_binary_data_operator.h"
#include "ccu_resource_manager.h"
#include "runtime_state/db_sim_runner_common.h"
#include "sim_log.h"

namespace HcclSim {
static const std::string PLUGIN_PATH = "/plugin";

static const std::string HCCLVM_FLAG_DATA_FILE = "/hcclvm_flag_data.bin";
static const std::string HCCLVM_TASK_DATA_FILE = "/%s_hcclvm_task_data.bin";
static const std::string HCCLVM_SYN_DATA_FILE = "/%s_hcclvm_syn_data.bin";
static const std::string HCCLVM_INSTR_DATA_FILE = "/%s_hcclvm_instr_data.bin";
static const std::string HCCLVM_ALL_RANK_INPUT_OUTPUT_FILE = "/all_rank_input_output.txt";

static const uint32_t PRINT_DATA_PREFIX = 512;          // 打印数据前缀
static const uint32_t PRINT_DATA_SUFFIX = 512;          // 打印数据后缀
static const uint32_t PRINT_DATA_SIZE_THRESHOLD = 1024; // 打印数据最大大小

std::string EidToHexString(uint8_t eid[])
{
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (uint32_t i = 0; i < URMA_EID_LEN; i++) {
        oss << std::setw(2) << static_cast<uint32_t>(eid[i]);
    }
    return oss.str();
}

void StorageManager::ReleasePhyMem()
{
    for (const auto& phyMem : m_allPhyMem) {
        sim::ReleaseInNoHostProcess(phyMem);
    }
    m_allPhyMem.clear();
}

HcclVmResult StorageManager::PrintAllRankInputBuffer()
{
    std::vector<sim::runtime::PhyMemBlock> allPhyMem;
    for (uint32_t i = 0; i < m_synData.memory_info.count; i++) {
        if (m_synData.memory_info.data[i].buffer_type == 0) {
            auto startAddr = m_synData.memory_info.data[i].start_addr;

            sim::runtime::PhyMemBlock srcPhyMem{};
            auto srcAddr = sim::AcquireDevPtrInNoHostProcess((void*)startAddr, srcPhyMem);
            if (srcAddr == nullptr) {
                HCCL_VM_ERROR(
                    "无法获取startAddr的设备地址(addr= {:x})！", reinterpret_cast<uintptr_t>((void*)startAddr));
                return HcclVmResult::HCCL_SIM_E_INTERNAL;
            }
            m_allPhyMem.push_back(srcPhyMem);

            auto size = m_synData.memory_info.data[i].size;
            HCCL_VM_DEBUG(
                "rankId={}, input buffer= {:x}, size={}", m_synData.memory_info.data[i].rank_id, startAddr, size);
            for (uint32_t idx = 0; idx < size; idx++) {
                HCCL_VM_TRACE("idx={}, value={:x}", idx, *((char*)srcAddr + idx));
            }
        }
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult StorageManager::Trans2CheckerParam(sim::operation::OpDetailTab& detailTab, ::OpDetails& detail)
{
    devType_ = static_cast<DevType>(detailTab.devType);
    m_checker_param.cmdType = static_cast<HcclCMDType>(detail.opType);
    m_checker_param.rankSize = detailTab.rankSize;
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
    const HcclCMDType cmdType = static_cast<HcclCMDType>(detail.opType);
    const bool isAll2AllOp = cmdType == HcclCMDType::HCCL_CMD_ALLTOALL || cmdType == HcclCMDType::HCCL_CMD_ALLTOALLV
                             || cmdType == HcclCMDType::HCCL_CMD_ALLTOALLVC;
    if (isAll2AllOp && detailTab.opExtInfo.size() >= sizeof(uint32_t)) {
        uint32_t count = 0;
        std::memcpy(&count, detailTab.opExtInfo.data(), sizeof(uint32_t));
        m_checker_param.all2AllDataDes.count = count;
        for (uint32_t i = 0; i < count; i++) {
            uint64_t val = 0;
            size_t offset = sizeof(uint32_t) + i * sizeof(uint64_t);
            if (offset + sizeof(uint64_t) <= detailTab.opExtInfo.size()) {
                std::memcpy(&val, detailTab.opExtInfo.data() + offset, sizeof(uint64_t));
            }
            m_checker_param.all2AllDataDes.sendCountMatrix.push_back(val);
        }
    }

    HCCL_VM_DEBUG("Success");

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult StorageManager::LoadHcclVmSynthesisData(
    sim::operation::OpDetailTab& detailTab, std::vector<sim::operation::CcuChannelTab>& channels)
{
    if (detailTab.opDetail.size() < sizeof(::OpDetails)) {
        HCCL_VM_ERROR("opDetail BLOB too small");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    ::OpDetails opDetails{};
    std::memcpy(&opDetails, detailTab.opDetail.data(), sizeof(::OpDetails));

    Trans2CheckerParam(detailTab, opDetails);
    auto rankSize = GetRankSize();
    m_allRankChannelInfo.resize(rankSize);

    // 转换channel映射表
    for (auto& channel : channels) {
        CcuInfo rmtDieInfo1;
        rmtDieInfo1.rankId = channel.dstRankId;
        rmtDieInfo1.dieId = channel.dstDieId;
        HCCL_VM_INFO(
            "[Channel info] channelId={}, srcRank={}, srcDie={}, dstRank={}, dstDie={} srcEid={}, dstEid={}",
            channel.channelId, channel.srcRankId, static_cast<uint32_t>(channel.srcDieId), channel.dstRankId,
            channel.dstDieId, EidToHexString(channel.leid), EidToHexString(channel.reid));
        m_allRankChannelInfo[channel.srcRankId][channel.srcDieId][channel.channelId] = rmtDieInfo1;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult StorageManager::LoadHcclVmInstrData(std::vector<sim::operation::CcuInstrResTab>& instrRes)
{
    for (auto& ccuInstr : instrRes) {
        HCCL_VM_DEBUG("deviceId={}, dieId={}, count={}", ccuInstr.deviceId, ccuInstr.dieId, ccuInstr.instrCount);
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult StorageManager::LoadHcclVmTaskMetaData(std::vector<sim::operation::OpTaskTab>& tasks)
{
    HcclVmTaskMetaData taskMeataData;
    for (const auto& task : tasks) {
        if (task.optaskMeta.size() >= sizeof(HcclTaskMetaData)) {
            HcclTaskMetaData metaData;
            std::memcpy(&metaData, task.optaskMeta.data(), sizeof(HcclTaskMetaData));
            taskMeataData.task_meta.push_back(metaData);
        } else {
            HCCL_VM_WARN("optaskMeta too small, src:{:d}, dst:{:d}", task.optaskMeta.size(), sizeof(HcclTaskMetaData));
        }
    }
    for (auto& taskMeta : taskMeataData.task_meta) {
        HCCL_VM_DEBUG(
            "rankId={}, dieId={}, instrCnt={}, argSize={}, streamId={}", taskMeta.rankId,
            static_cast<uint32_t>(taskMeta.taskData.ccu.dieId), taskMeta.taskData.ccu.instCnt,
            taskMeta.taskData.ccu.argSize, taskMeta.streamId);
    }
    ConvertTaskQueue(taskMeataData);
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult StorageManager::GetHcclVmFlagData(HcclSim::HcclVmFlagData& waitFlag)
{
    // 1. 构造路径
    std::string rootPath = FindRootPath();
    if (rootPath.empty()) {
        HCCL_VM_ERROR("Failed to find root path");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    std::string fullPath = rootPath + DATA_FILE_PATH + HCCLVM_FLAG_DATA_FILE;
    // 2. 写文件，通知runner启动
    FILE* fp = fopen(fullPath.c_str(), "rb");
    if (!fp) {
        HCCL_VM_ERROR("Open file failed: {}", fullPath);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    auto ret = HcclVmFlagDataRead(fp, waitFlag, HCCLVM_FLAG_FILE_MAGIC);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("HcclVmFlagDataRead failed");
        fclose(fp);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    fclose(fp);
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult StorageManager::DumpHcclVmFlagData(uint16_t status)
{
    HCCL_VM_DEBUG("Start dumping hccl vm flag data...");
    // 1. 构造完整路径
    std::string rootPath = FindRootPath();
    if (rootPath.empty()) {
        HCCL_VM_ERROR("Failed to find root path");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    std::string fullPath = rootPath + DATA_FILE_PATH + HCCLVM_FLAG_DATA_FILE;
    // 2. 写文件，通知runner启动
    FILE* fp = fopen(fullPath.c_str(), "wb");
    if (!fp) {
        HCCL_VM_ERROR("Open file failed: {}", fullPath);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    HcclSim::HcclVmFlagData flagData{};
    flagData.header.magic = HCCLVM_FLAG_FILE_MAGIC;
    flagData.runner_status = status;
    auto ret = HcclVmFlagDataWrite(fp, flagData);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("HcclVmFlagDataWrite failed");
        fclose(fp);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    fflush(fp);
    fclose(fp);
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

std::string StorageManager::FindRootPath()
{
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
            HCCL_VM_WARN("Iteration {}: realpath failed for {}", i, current_search_path);
        }

        // 没找到，将探测路径向上推一级
        current_search_path += "/..";
    }

    HCCL_VM_WARN("RootPath NOT found");
    return "";
}

HcclVmResult StorageManager::ConvertTaskQueue(const HcclVmTaskMetaData& taskMeataData)
{
    auto rankSize = GetRankSize();
    m_allRankTaskQueues.resize(rankSize);
    HCCL_VM_INFO("rankSize={}", rankSize);

    auto taskMetaVec = taskMeataData.task_meta;
    uint32_t size = taskMetaVec.size();
    for (uint32_t i = 0; i < size; i++) {
        uint32_t rankId = taskMetaVec[i].rankId;
        if (rankId >= rankSize) {
            HCCL_VM_ERROR("rankId={} is out of range. rankSize={}", rankId, rankSize);
            return HcclVmResult::HCCL_SIM_E_PARA;
        }
        uint32_t streamId = (taskMetaVec[i].streamId);
        if (m_allRankTaskQueues[rankId].size() <= streamId) {
            m_allRankTaskQueues[rankId].resize(streamId + 1);
        }
        m_allRankTaskQueues[rankId][streamId].push(taskMetaVec[i]);
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

bool StorageManager::IsDirExists(const std::string& path)
{
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        return false; // 不存在
    }
    return (info.st_mode & S_IFDIR); // 存在且是目录
}

uint32_t StorageManager::GetRankSize() const { return m_checker_param.rankSize; }

std::vector<RankChannelInfo>& StorageManager::GetAllRankChannelInfo() { return m_allRankChannelInfo; }

HcclVmInstrData StorageManager::GetHvmInstrData() const { return m_instrData; }

AllRankTaskQueues& StorageManager::GetAllRankTaskQueues() { return m_allRankTaskQueues; }

HcclVmResult StorageManager::InitCcuResource(std::vector<sim::operation::CcuInstrResTab>& instrRes)
{
    RunnerCcuVersion ccuVersion{RunnerCcuVersion::CCU_INVALID};
    if (devType_ == DevType::DEV_TYPE_950) {
        ccuVersion = RunnerCcuVersion::CCU_V1;
    } else if (devType_ == DevType::DEV_TYPE_960) {
        ccuVersion = RunnerCcuVersion::CCU_V2;
    } else {
        HCCL_VM_ERROR("Wrong devive type: {:d}", static_cast<int>(devType_));
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    // ★ 取代原先的 ccuResourceBaseAddr = {0x123123123, 0x456456456} 假基址（设计文档 §2.2 #7）：
    //   per-(rank, die) 走全项目唯一入口 sim::runtime::GetCcuResourceBaseAddr（DB 优先/yaml 回退）。
    //   注意：本函数属 legacy 路径（RunVirtualRuntime 当前无调用者），rank 槽位沿用
    //   "rankId 即 CcuResourceManager 索引"的旧约定；活跃路径见 hccl_task_sequential_execute_v2.cc
    //   （按 deviceId 索引并逐 device 解析基址）。解析失败写 0（显式"未知"），不回退假地址。
    auto rankSize = GetRankSize();
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    for (uint32_t rankId = 0; rankId < rankSize; rankId++) {
        std::vector<uint64_t> ccuResourceBaseAddr{};
        for (uint32_t dieId = 0; dieId < HcclSim::DIE_NUM; dieId++) {
            uint64_t baseAddr = 0;
            if (!sim::runtime::GetCcuResourceBaseAddr(rankId, dieId, baseAddr) || baseAddr == 0) {
                HCCL_VM_ERROR(
                    "resolve ccu resource base addr failed: rankId={} dieId={}（写入 0，不回退假地址）", rankId, dieId);
                baseAddr = 0;
            }
            ccuResourceBaseAddr.push_back(baseAddr);
        }
        ccuResMgr.Init(rankId, rankSize, ccuVersion, ccuResourceBaseAddr);
        ccuResMgr.InitChannelInfo(rankId, m_allRankChannelInfo[rankId]);
    }
    for (auto& instr : instrRes) {
        CcuInstrData ccuInstr;
        ccuInstr.instrCnt = instr.instrCount;
        ccuInstr.instrData.reserve(instr.instrCount);
        for (uint32_t i = 0; i < instr.instrCount; ++i) {
            hcomm::CcuRep::CcuInstr tmp;
            std::memcpy(&tmp, instr.instrSpace[i], sizeof(hcomm::CcuRep::CcuInstr));
            ccuInstr.instrData.push_back(tmp);
        }
        ccuResMgr.InitInstrInfo(instr.deviceId, instr.dieId, ccuInstr);
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

/**
 * @brief dump 开关查询：HCCLVM_ENABLE_DUMP_DATA 未设置/为空/"0" 均视为未使能。
 * @return 使能返回 true，否则返回 false。
 */
bool StorageManager::IsDumpDataEnabled() const
{
    const char* enableDumpData = std::getenv("HCCLVM_ENABLE_DUMP_DATA");
    return enableDumpData != nullptr && !std::string(enableDumpData).empty() && std::string(enableDumpData) != "0";
}

/**
 * @brief 打开 all_rank_input_output.txt：append=false 截断重写（legacy 整轮 dump），append=true 追加（V2 按 op 增量）。
 * @param ofs 出参，打开成功的输出流
 * @param append true 追加模式，false 截断模式
 * @return 打开成功返回 true；找不到根目录/打开失败时打日志并返回 false。
 */
bool StorageManager::OpenDumpFile(std::ofstream& ofs, bool append)
{
    std::string rootPath = FindRootPath();
    if (rootPath.empty()) {
        HCCL_VM_ERROR("Failed to find root path");
        return false;
    }
    std::string fullPath = rootPath + HCCLVM_ALL_RANK_INPUT_OUTPUT_FILE;
    auto mode = std::ios::out | (append ? std::ios::app : std::ios::trunc);
    ofs.open(fullPath, mode);
    if (!ofs.is_open()) {
        HCCL_VM_ERROR("Open file failed: {}", fullPath);
        return false;
    }
    return true;
}

/**
 * @brief 把一个 rank 的 input/output buffer 映射到本进程地址空间并回填 BufferInfo。
 * @param addr buffer 虚拟地址，为 0 视为无此 buffer（直接成功返回）
 * @param size buffer 大小
 * @param type buffer 类型（仅用于日志）
 * @param out 出参，回填 virtualAddr/phyAddr/size
 * @param phyMemOut 出参，成功获取的物理内存块追加到该 vector，由调用方在写出后统一释放
 * @return 映射成功或 addr 为 0 返回 HCCL_SIM_SUCCESS；映射失败返回 HCCL_SIM_E_INTERNAL。
 */
HcclVmResult StorageManager::AcquireRankBuffer(
    uint64_t addr, uint64_t size, BufferType type, BufferInfo& out, std::vector<sim::runtime::PhyMemBlock>& phyMemOut)
{
    if (addr == 0 || size == 0) {
        return HcclVmResult::HCCL_SIM_SUCCESS;
    }
    sim::runtime::PhyMemBlock srcPhyMem{};
    auto startAddr = sim::AcquireDevPtrInNoHostProcess((void*)addr, srcPhyMem);
    if (startAddr == nullptr) {
        HCCL_VM_ERROR("fail addr(type={}, addr={:x})!", static_cast<int>(type), addr);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    phyMemOut.push_back(srcPhyMem);
    out.virtualAddr = addr;
    out.phyAddr = (char*)startAddr;
    out.size = size;
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

/**
 * @brief 把一个 op（全部 rank）的 input/output 数据写出一个 "=== Op N ===" 段；
 *        legacy 整轮 dump 与 V2 按 op 增量 dump 共用此实现。
 * @param ofs 已打开的输出流
 * @param opIdx 算子序号（写入段头）
 * @param opType 算子类型（HcclCMDType，决定打印 input 还是 input+output）
 * @param rankSize 通信域 rank 数（buffer 向量维度；V2 路径取算子明细行，不能依赖 GetRankSize）
 * @param rankTask 该 op 全部 rank 的复合明细（key = rankId）
 * @param phyMemOut 出参，本段映射到的物理内存块，由调用方在写出后统一释放
 * @return 写出成功返回 HCCL_SIM_SUCCESS；buffer 映射失败返回 HCCL_SIM_E_INTERNAL。
 */
HcclVmResult StorageManager::DumpOneOpGroup(
    std::ofstream& ofs, uint32_t opIdx, uint32_t opType, uint32_t rankSize,
    const std::map<uint32_t, sim::operation::CompositeOpDetail>& rankTask,
    std::vector<sim::runtime::PhyMemBlock>& phyMemOut)
{
    std::vector<BufferInfo> allRankInput(rankSize);
    std::vector<BufferInfo> allRankOutput(rankSize);
    for (auto& it : rankTask) {
        if (it.first >= rankSize) {
            HCCL_VM_WARN("rankId={} out of range, rankSize={}, opIdx={}", it.first, rankSize, opIdx);
            continue;
        }
        auto& memInfo = it.second.memInfo;
        auto inRet = AcquireRankBuffer(
            memInfo.inputAddr, memInfo.inputSize, BufferType::INPUT, allRankInput[it.first], phyMemOut);
        auto outRet = AcquireRankBuffer(
            memInfo.outputAddr, memInfo.outputSize, BufferType::OUTPUT, allRankOutput[it.first], phyMemOut);
        if (inRet != HcclVmResult::HCCL_SIM_SUCCESS || outRet != HcclVmResult::HCCL_SIM_SUCCESS) {
            return HcclVmResult::HCCL_SIM_E_INTERNAL;
        }
        HCCL_VM_INFO(
            "opIdx={}, rankId={}, inputAddr={:x}, size={}, outputAddr={:x}, size={}", opIdx, it.first,
            memInfo.inputAddr, memInfo.inputSize, memInfo.outputAddr, memInfo.outputSize);
    }

    ofs << "\n=== Op " << std::dec << opIdx << " (opType=" << opType << ") ===\n";
    switch (static_cast<HcclCMDType>(opType)) {
        case HcclCMDType::HCCL_CMD_BROADCAST:
            PrintOpData1(ofs, rankSize, allRankInput);
            break;
        case HcclCMDType::HCCL_CMD_SEND:
        case HcclCMDType::HCCL_CMD_RECEIVE:
            // P2P：发送侧有 input、接收侧有 output，按 input+output 同列打印
            PrintOpData2(ofs, rankSize, allRankInput, allRankOutput);
            break;
        case HcclCMDType::HCCL_CMD_ALLGATHER:
        case HcclCMDType::HCCL_CMD_ALLREDUCE:
        case HcclCMDType::HCCL_CMD_SCATTER:
        case HcclCMDType::HCCL_CMD_REDUCE:
        case HcclCMDType::HCCL_CMD_REDUCE_SCATTER:
        case HcclCMDType::HCCL_CMD_ALLTOALL:
        case HcclCMDType::HCCL_CMD_ALLTOALLV:
            PrintOpData2(ofs, rankSize, allRankInput, allRankOutput);
            break;
        default:
            break;
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

/**
 * @brief legacy 整轮 dump 入口：把按 op 分组的全部 rank input/output 数据截断写出
 *        all_rank_input_output.txt（当前无调用者，保留兼容；活跃路径为 CollectOpRankDump）。
 * @param compositeDataMap 按 op 索引、按 rankId 索引的复合算子明细
 * 副作用：使能 HCCLVM_ENABLE_DUMP_DATA 时写文件；映射的物理内存留在 m_allPhyMem，由调用方 ReleasePhyMem。
 */
void StorageManager::DumpAllRankInputOutput(
    std::vector<std::map<uint32_t, sim::operation::CompositeOpDetail>>& compositeDataMap)
{
    if (!IsDumpDataEnabled()) {
        HCCL_VM_DEBUG("HCCLVM_ENABLE_DUMP_DATA is not set");
        return;
    }
    auto rankSize = GetRankSize();
    if (compositeDataMap.empty() || compositeDataMap[0].empty()) {
        HCCL_VM_ERROR("compositeDataMap is empty");
        return;
    }
    std::ofstream ofs;
    if (!OpenDumpFile(ofs, false)) {
        return;
    }
    m_dumpFileTruncated = true; // legacy 整轮 dump 本身截断，同步标记避免后续 V2 flush 再截断

    HCCL_VM_INFO("Dump All Rank Input Output Data. rankSize={}", rankSize);
    for (size_t opIdx = 0; opIdx < compositeDataMap.size(); opIdx++) {
        auto& rankTask = compositeDataMap[opIdx];
        if (rankTask.empty()) {
            continue;
        }
        const auto& firstComp = rankTask.begin()->second;
        if (firstComp.detail.opDetail.size() < sizeof(::OpDetails)) {
            HCCL_VM_ERROR("opDetail BLOB too small, opIdx={}, size={}", opIdx, firstComp.detail.opDetail.size());
            continue;
        }
        ::OpDetails opDetails{};
        std::memcpy(&opDetails, firstComp.detail.opDetail.data(), sizeof(::OpDetails));
        auto ret = DumpOneOpGroup(ofs, opIdx, opDetails.opType, rankSize, rankTask, m_allPhyMem);
        if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
            HCCL_VM_WARN("dump op group failed. opIdx={}, ret={:d}", opIdx, static_cast<int>(ret));
        }
    }
    ofs.close();
    HCCL_VM_INFO("Dump All Rank Input Output Data success.");
}

/**
 * @brief V2 执行路径的 input/output dump 登记入口（详见头文件注释）：只记录 buffer 地址/大小，
 *        不读内容；内容在全部设备同步就绪后由 FlushPendingOpDumps 统一读出写出。
 * @param commHash 该 op 所属通信域的 hash（跨 rank 共享，聚合键的一部分）
 * @param detail 该 rank 的算子明细行
 * @param memInfo 该 rank 的 input/output buffer 地址与大小
 */
void StorageManager::CollectOpRankDump(
    uint64_t commHash, const sim::operation::OpDetailTab& detail, const sim::operation::OpMemInfoTab& memInfo)
{
    if (!IsDumpDataEnabled()) {
        return;
    }
    if (detail.opDetail.size() < sizeof(::OpDetails)) {
        HCCL_VM_ERROR("opDetail BLOB too small, opDetailId={}, size={}", detail.id, detail.opDetail.size());
        return;
    }
    ::OpDetails opDetails{};
    std::memcpy(&opDetails, detail.opDetail.data(), sizeof(::OpDetails));

    // 聚合键用通信域级 commHash（跨 rank 共享）+ syncIter（区分多次 run）+ opIter（区分同周期多 op）；
    // detail.commId 是每 rank 各自的 Communicator 行 id，用作聚合键会导致每组只有 1 个 rank、永远收不齐。
    auto key = std::make_tuple(commHash, detail.syncIter, detail.opIter);
    auto& pending = m_pendingOpDumps[key];
    if (pending.dumped) {
        // 本 op 已落盘；同 opDetailId 的后续 task 完成通知直接忽略。
        return;
    }
    pending.opType = opDetails.opType;
    pending.rankSize = detail.rankSize;
    auto& comp = pending.rankMemInfos[detail.rankId];
    comp.rankId = detail.rankId;
    comp.deviceId = detail.deviceId;
    comp.commId = detail.commId;
    comp.detail = detail;
    comp.memInfo = memInfo;
}

/**
 * @brief 是否存在未落盘的 pending dump 组。
 * @return 存在返回 true，否则返回 false。
 */
bool StorageManager::HasPendingOpDumps() const
{
    for (const auto& [_, pending] : m_pendingOpDumps) {
        if (!pending.dumped) {
            return true;
        }
    }
    return false;
}

/**
 * @brief 全部设备同步就绪后（由调用方判定）把所有未落盘的 pending 组读出并写出
 *        all_rank_input_output.txt；每组仅落盘一次。本会话首次落盘截断重写（清掉历史运行
 *        残留的旧 section），其后追加，保证文件内容只含本次运行产生的段。
 */
void StorageManager::FlushPendingOpDumps()
{
    if (!IsDumpDataEnabled()) {
        return;
    }
    std::ofstream ofs;
    bool opened = false;
    for (auto& [key, pending] : m_pendingOpDumps) {
        if (pending.dumped) {
            continue;
        }
        if (!opened) {
            // 首次打开按截断标记决定模式；打开失败不置标记，下一轮重试仍会先截断
            if (!OpenDumpFile(ofs, m_dumpFileTruncated)) {
                return; // 打开失败，保留 pending 待下一轮就绪信号重试
            }
            m_dumpFileTruncated = true;
            opened = true;
        }
        const auto& [commHash, syncIter, opIter] = key;
        HCCL_VM_INFO(
            "Dump op input/output data. commHash={}, syncIter={}, opIter={}, registeredRanks={}", commHash, syncIter,
            opIter, pending.rankMemInfos.size());
        auto ret = DumpOneOpGroup(ofs, opIter, pending.opType, pending.rankSize, pending.rankMemInfos, m_allPhyMem);
        ReleasePhyMem(); // 本组已写出，释放其映射的设备内存，避免长稳运行无限增长
        if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
            HCCL_VM_WARN("dump op group failed. opIter={}, ret={:d}", opIter, static_cast<int>(ret));
            continue; // 保留未落盘标记，待下一轮就绪信号重试
        }
        pending.dumped = true;
    }
    if (opened) {
        ofs.close();
    }
}

/**
 * @brief 按大小灵活打印 buffer 内容：不超过阈值全量打印，超过阈值只打印前缀+后缀并标注跳过字节数。
 * @param ofs 已打开的输出流
 * @param buffer 已映射到本进程的 buffer（phyAddr 有效）
 */
void StorageManager::FlexiblePrintData(std::ofstream& ofs, const BufferInfo& buffer)
{
    if (buffer.size <= PRINT_DATA_SIZE_THRESHOLD) {
        for (uint64_t i = 0; i < buffer.size; i++) {
            // setw 每次插入即失效须逐字节重设；不补零会导致 0x00/0x04 等打印成 "0"/"4" 错位
            ofs << std::hex << std::setfill('0') << std::setw(2)
                << static_cast<int>(static_cast<unsigned char>(buffer.phyAddr[i]));
        }
        return;
    }

    for (uint32_t i = 0; i < PRINT_DATA_PREFIX; i++) {
        ofs << std::hex << std::setfill('0') << std::setw(2)
            << static_cast<int>(static_cast<unsigned char>(buffer.phyAddr[i]));
    }
    ofs << "\n...[skipped " << std::dec << buffer.size - PRINT_DATA_SIZE_THRESHOLD << " bytes]...\n";
    for (uint64_t i = buffer.size - PRINT_DATA_SUFFIX; i < buffer.size; i++) {
        ofs << std::hex << std::setfill('0') << std::setw(2)
            << static_cast<int>(static_cast<unsigned char>(buffer.phyAddr[i]));
    }
}

/**
 * @brief 打印单 buffer 类算子（如 Broadcast）的全部 rank input 数据。
 * @param ofs 已打开的输出流
 * @param rankSize 通信域 rank 数（显式传入，V2 路径 GetRankSize() 无值）
 * @param allRankInput 按 rankId 索引的 input buffer 向量
 */
void StorageManager::PrintOpData1(std::ofstream& ofs, uint32_t rankSize, const std::vector<BufferInfo>& allRankInput)
{
    for (uint32_t idx = 0; idx < rankSize; idx++) {
        ofs << std::dec << "Rank " << idx << " Input: " << (uint64_t)(void*)allRankInput[idx].virtualAddr
            << ", size=" << allRankInput[idx].size << std::endl;
        FlexiblePrintData(ofs, allRankInput[idx]);
        ofs << std::endl;
    }
}

/**
 * @brief 打印 input+output 类算子的全部 rank input/output 数据。
 * @param ofs 已打开的输出流
 * @param rankSize 通信域 rank 数（显式传入，V2 路径 GetRankSize() 无值）
 * @param allRankInput 按 rankId 索引的 input buffer 向量
 * @param allRankOutput 按 rankId 索引的 output buffer 向量
 */
void StorageManager::PrintOpData2(
    std::ofstream& ofs, uint32_t rankSize, const std::vector<BufferInfo>& allRankInput,
    const std::vector<BufferInfo>& allRankOutput)
{
    for (uint32_t idx = 0; idx < rankSize; idx++) {
        ofs << std::dec << "Rank " << idx << " Input: " << (uint64_t)(void*)allRankInput[idx].virtualAddr
            << ", size=" << allRankInput[idx].size << std::endl;
        FlexiblePrintData(ofs, allRankInput[idx]);
        ofs << std::endl;
        ofs << std::dec << "Rank " << idx << " Output: " << (uint64_t)(void*)allRankOutput[idx].virtualAddr
            << ", size=" << allRankOutput[idx].size << std::endl;
        FlexiblePrintData(ofs, allRankOutput[idx]);
        ofs << "\n" << std::endl;
    }
}
} // namespace HcclSim
