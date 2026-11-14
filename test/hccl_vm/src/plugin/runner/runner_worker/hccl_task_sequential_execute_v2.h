/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_TASK_SEQUENTIAL_EXECUTE_V2_H
#define HCCL_TASK_SEQUENTIAL_EXECUTE_V2_H

#include <map>
#include <string>

#include "sim_common_defs.h"
#include "sim_loader.h"
#include "operation_data/operation_data_types.h"
#include "aiv_graph_executor.h"

using namespace HcclSim;

namespace VirtualRunTime {

class SequentialExecutorV2 {
public:
    explicit SequentialExecutorV2(loader::Loader& loader) : loader_(loader) {}
    ~SequentialExecutorV2() = default;

    HcclVmResult Execute();

private:
    /**
     * @brief dump 开关（HCCLVM_ENABLE_DUMP_DATA）使能时，任一任务成功后按 opDetailId 反查算子明细与
     *        内存信息，登记（只记 buffer 地址/大小，不读内容）到 StorageManager 按 op 聚合暂存；
     *        buffer 内容待全部设备同步就绪后由 FlushReadyOpDumps 触发统一读出写出，确保是终态数据。
     * @param task 刚执行成功的任务行（提供 opDetailId 定位算子）
     * 副作用：使能时查询 runner 库并向 StorageManager 登记；查询失败仅告警不影响主流程。
     */
    void DumpTaskInputOutput(const sim::operation::OpTaskTab& task);

    /**
     * @brief 存在未落盘 dump 组时，轮询"全部设备同步就绪"信号（DeviceStatus.synchronize_strategy=1
     *        计数 == 在线 Device 计数，与宿主侧 RunnerListen 判定一致）；就绪后由 StorageManager
     *        把所有 pending 组读出写出 all_rank_input_output.txt。未就绪或查询失败时保留 pending，
     *        下一轮 Execute 再判（主循环每 1ms 重调 Execute，可收敛）。
     * 副作用：就绪时读设备内存并写文件；查询失败仅告警不影响主流程。
     */
    void FlushReadyOpDumps();

    HcclVmResult ExecuteOneTask(HcclTaskMetaData& task);

    // AIV
    HcclVmResult TaskAivGraph(const HcclTaskMetaData& task);
    std::shared_ptr<AivGraphExecutor> GetAivGraphExecutor(uint64_t deviceId, uint64_t launchIdx);
    void RemoveAivGraphExecutor(uint64_t deviceId, uint64_t launchIdx);

    // CCU
    HcclVmResult EnsureCcuResource(); // 按DB变更签名增量刷新CCU资源(设备资源/通道映射/微码指令空间)

private:
    loader::Loader& loader_;

    // <{deviceId, launchIdx}, AivGraphExecutor> 每个AivGraphExecutor对应一个AivGraphTask节点, 因为在顺序执行引擎下,
    // AivGraphTask节点常常不能一次执行完成, 需要记录执行状态并让出
    std::map<std::pair<uint64_t, uint64_t>, std::shared_ptr<AivGraphExecutor>> aivExecutorMap_{};

    bool ccuResourceInited_{false};
    uint32_t lastInstrLoadCnt_{0};
    uint32_t lastChannelCnt_{0};
};

} // namespace VirtualRunTime
#endif
