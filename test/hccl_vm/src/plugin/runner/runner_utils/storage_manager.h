/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STORAGE_MANAGER_H
#define STORAGE_MANAGER_H

#include <cstdint>
#include <fstream>
#include <map>
#include <mutex>
#include <queue>
#include <tuple>
#include <vector>

#include "sim_binary_data_type_pub.h"
#include "ccu_resource_common.h"
#include "dtype_common.h"
#include "sim_common_defs.h"
#include "sim_loader.h"
#include "operation_data/operation_data_types.h"
#include "store_sim_shm_memory_common.h"

namespace HcclSim {
struct BufferInfo {
    uint64_t virtualAddr{0};
    char* phyAddr{nullptr};
    uint64_t size{0};
};

struct StreamTag {
    uint32_t rankId{0};
    uint32_t streamId{0};
};

struct CheckerParam {
    HcclCMDType cmdType = static_cast<HcclCMDType>(0);
    uint32_t rankSize = 0;
    HcclDataType dataType = static_cast<HcclDataType>(0);
    uint64_t dataCount = 0;

    // 以下是可能缺失的字段，初始化为安全值
    HcclReduceOp reduceType = static_cast<HcclReduceOp>(0);
    uint32_t srcRank = 0;
    uint32_t dstRank = 0;
    uint32_t root = 0;

    VDataDesTagInner vDataDes;
    All2AllDataDesTagInner all2AllDataDes;
};

// 一个rank的任务队列： {streamId, {tasklist}}
using SingleTaskQueue = std::vector<std::queue<HcclTaskMetaData>>;
using AllRankTaskQueues = std::vector<SingleTaskQueue>;

class StorageManager {
public:
    static StorageManager& GetInstance()
    {
        static StorageManager instance;
        return instance;
    }
    // 禁用拷贝构造和赋值
    StorageManager(const StorageManager&) = delete;
    StorageManager& operator=(const StorageManager&) = delete;

    void SetDataId(const std::string& data_id)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_data_id = data_id;
    }

    std::string GetDataId() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_data_id;
    }

    CheckerParam GetCheckerParam() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_checker_param;
    }

    std::string FindRootPath();

    HcclVmResult InitCcuResource(std::vector<sim::operation::CcuInstrResTab>& instrRes);
    void ReleasePhyMem();

    HcclVmResult
    LoadHcclVmSynthesisData(sim::operation::OpDetailTab& detail, std::vector<sim::operation::CcuChannelTab>& channels);
    HcclVmResult LoadHcclVmInstrData(std::vector<sim::operation::CcuInstrResTab>& instrRes);
    HcclVmResult LoadHcclVmTaskMetaData(std::vector<sim::operation::OpTaskTab>& tasks);
    HcclVmResult ConvertTaskQueue(const HcclVmTaskMetaData& taskMeataData);
    uint32_t GetRankSize() const;

    HcclVmInstrData GetHvmInstrData() const;
    HcclVmResult Trans2CheckerParam(sim::operation::OpDetailTab& detailTab, ::OpDetails& detail);
    HcclVmTaskMetaData GetHvmTaskMetaData() const;
    AllRankTaskQueues& GetAllRankTaskQueues();
    void DumpAllRankInputOutput(std::vector<std::map<uint32_t, sim::operation::CompositeOpDetail>>& compositeDataMap);
    /**
     * @brief V2 执行路径的 input/output dump 登记入口：某个 op 的一个 rank 的任一任务执行成功后登记其
     *        buffer 地址/大小（只记地址，不读内容），按 (commHash, syncIter, opIter) 暂存
     *        （commHash 为通信域级别标识，跨 rank 一致；注意 OpDetailTab.commId 是每 rank 各自的
     *        Communicator 行 id，不能用作聚合键），待全部设备同步就绪后由 FlushPendingOpDumps 统一读出写出。
     * @param commHash 该 op 所属通信域的 hash（跨 rank 共享，由调用方按 detail.commId 反查 Communicator 得到）
     * @param detail 该 rank 的算子明细行（opType BLOB / rankId / opIter / syncIter / rankSize）
     * @param memInfo 该 rank 的 input/output buffer 地址与大小
     * 副作用：使能 HCCLVM_ENABLE_DUMP_DATA 时暂存登记状态；未使能时 no-op。
     */
    void CollectOpRankDump(
        uint64_t commHash, const sim::operation::OpDetailTab& detail, const sim::operation::OpMemInfoTab& memInfo);
    // 是否存在未落盘的 pending dump 组（runner 主循环据此决定是否轮询设备就绪信号）。
    bool HasPendingOpDumps() const;
    /**
     * @brief 全部设备同步就绪后（由调用方判定）把所有未落盘的 pending 组读出并写出
     *        all_rank_input_output.txt；每组仅落盘一次，已落盘组忽略。本会话首次落盘截断
     *        重写（清掉历史运行残留的旧 section），其后追加，保证文件内容只含本次运行的段。
     * 副作用：读设备内存并写文件；写出的映射随写随释放。
     */
    void FlushPendingOpDumps();
    HcclVmResult DumpHcclVmFlagData(uint16_t finishFlag);
    HcclVmResult GetHcclVmFlagData(HcclSim::HcclVmFlagData& flagData);
    // 重置运行期状态（任务队列/通道/物理内存/数据缓存/设备类型），供单测隔离与复用场景调用；
    // 同时复位 dump 文件截断标记，使下一次 flush 重新截断清掉历史残留段。
    void Reset()
    {
        m_allRankTaskQueues.clear();
        m_allRankChannelInfo.clear();
        m_allPhyMem.clear();
        m_synData = {};
        m_instrData = {};
        m_checker_param = {};
        devType_ = DevType::DEV_TYPE_COUNT;
        // dump 文件截断标记随会话重置：下一次 flush 先截断重写，清掉历史运行残留段。
        m_dumpFileTruncated = false;
    }
    std::vector<RankChannelInfo>& GetAllRankChannelInfo();

private:
    StorageManager() = default;
    HcclVmResult PrintAllRankInputBuffer();
    // dump 开关查询：HCCLVM_ENABLE_DUMP_DATA 未设置/为空/"0" 时返回 false。
    bool IsDumpDataEnabled() const;
    // 打开 all_rank_input_output.txt：append=false 截断重写（legacy 整轮 dump），append=true 追加（V2 按 op 增量）。
    // 返回 false 表示未使能/找不到根目录/打开失败（已打日志），调用方应直接返回。
    bool OpenDumpFile(std::ofstream& ofs, bool append);
    // 把一个 rank 的 input/output buffer 映射到本进程地址空间并回填 BufferInfo；
    // 成功获取的物理内存块追加到 phyMemOut（由调用方在写出后统一释放）。addr 为 0 视为无此 buffer。
    HcclVmResult AcquireRankBuffer(
        uint64_t addr, uint64_t size, BufferType type, BufferInfo& out,
        std::vector<sim::runtime::PhyMemBlock>& phyMemOut);
    // 把一个 op（全部 rank）的 input/output 数据写出一个 "=== Op N ===" 段，供 legacy 整轮 dump 与 V2 按 op dump 复用。
    HcclVmResult DumpOneOpGroup(
        std::ofstream& ofs, uint32_t opIdx, uint32_t opType, uint32_t rankSize,
        const std::map<uint32_t, sim::operation::CompositeOpDetail>& rankTask,
        std::vector<sim::runtime::PhyMemBlock>& phyMemOut);
    void FlexiblePrintData(std::ofstream& ofs, const BufferInfo& buffer);
    void PrintOpData1(std::ofstream& ofs, uint32_t rankSize, const std::vector<BufferInfo>& allRankInput);
    void PrintOpData2(
        std::ofstream& ofs, uint32_t rankSize, const std::vector<BufferInfo>& allRankInput,
        const std::vector<BufferInfo>& allRankOutput);

    bool IsDirExists(const std::string& path);

    // V2 路径按 op 聚合 dump 的暂存：key = (commHash, syncIter, opIter)；
    // 只登记 buffer 地址，内容在全部设备同步就绪后由 FlushPendingOpDumps 读出写出（runner 单线程访问，无锁）。
    struct PendingOpDump {
        uint32_t opType{0};
        uint32_t rankSize{0};
        bool dumped{false};
        std::map<uint32_t, sim::operation::CompositeOpDetail> rankMemInfos;
    };

    std::string m_data_id;
    mutable std::mutex m_mutex;

    std::vector<RankChannelInfo> m_allRankChannelInfo;
    CheckerParam m_checker_param;

    HcclVmSynData m_synData;
    HcclVmInstrData m_instrData;
    AllRankTaskQueues m_allRankTaskQueues;
    std::vector<sim::runtime::PhyMemBlock> m_allPhyMem;
    std::map<std::tuple<uint64_t, uint32_t, uint32_t>, PendingOpDump> m_pendingOpDumps;
    // dump 文件截断标记：false=本会话首次落盘须截断重写（清掉历史运行残留段），true=已截断、后续 flush 追加。
    bool m_dumpFileTruncated{false};
    DevType devType_{DevType::DEV_TYPE_COUNT}; // 初始化无效值
};
} // namespace HcclSim

#endif
