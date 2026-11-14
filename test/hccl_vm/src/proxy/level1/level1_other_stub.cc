/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * Description: 其他桩函数（北向劫持）——不属于通信域初始化、控制面、数据面、公共结构的接口
 * Create: 2026-07-25
 */

#define HCCL_VM_MODULE "OTHER_STUB"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

#include "level1_proxy_common.h"
#include "hccl/hccl_launch.h"
#include "sim_log.h"
#include "runtime_state/db_sim_runner_common.h"
#include "runtime_state/db_sim_runner_ops.h"

#ifdef __cplusplus
extern "C" {
#endif

// 跨桩文件复用的北向桩声明（定义分别在 ctrl_comm_resource_stub.cc / data_local_op_stub.cc）
HcclResult HcclThreadAcquireWithStream(
    HcclComm comm, CommEngine engine, aclrtStream stream, uint32_t notifyNum, ThreadHandle* thread);
HcclResult HcclThreadExportToCommEngine(
    HcclComm comm, uint32_t threadNum, const ThreadHandle* threads, CommEngine dstCommEngine,
    ThreadHandle* exportedThreads);
int32_t HcommThreadNotifyRecordOnThread(ThreadHandle thread, ThreadHandle dstThread, uint32_t dstNotifyIdx);
int32_t HcommThreadNotifyWaitOnThread(ThreadHandle thread, uint32_t notifyIdx, uint32_t timeout);

#ifdef __cplusplus
}
#endif

namespace {
// 与 hcomm NUM_THREE 同口径：AICPU 主流线程持有的 notify 预算
constexpr uint32_t kAicpuMainThreadNotifyNum = 3;

/**
 * @brief 获取并导出 AICPU kernel launch 所需的线程资源（对齐 hcomm AicpuKernelLaunchDirect 前段）。
 *
 * 依次完成：CPU_TS 主流线程获取（HcclThreadAcquireWithStream 桩）、
 * AICPU_TS/CPU_TS 双向线程导出（HcclThreadExportToCommEngine 桩）、
 * 导出线程序号查询。notifyNum 真实现由 HcclGetNotifyNumInThread 从线程资源管理器读取，
 * 本桩无该接口，改查 DB 中 HcclThread 记录的 notifyNum（同一建模口径）。
 *
 * @param comm 输入：通信域句柄（level1 假句柄，值为 commId）。
 * @param userStream 输入：用户主流。
 * @param aicpuThreadHandle 输入：AICPU 通信主流线程句柄。
 * @param exportedAicpuTsThread 输出：导出到 AICPU_TS 引擎的 cpuTsThread。
 * @param exportedCpuTsThread 输出：导出到 CPU_TS 引擎的 aicpuThreadHandle。
 * @param cpuTsThread 输出：获取到的 CPU_TS 主流线程句柄。
 * @param notifyNum 输出：导出的 cpuTs 线程 notify 数量。
 * @return HcclResult 成功返回 HCCL_SUCCESS；任一步骤失败返回对应错误码。
 */
HcclResult AcquireAndExportAicpuThreads(
    HcclComm comm, aclrtStream userStream, ThreadHandle aicpuThreadHandle, ThreadHandle* exportedAicpuTsThread,
    ThreadHandle* exportedCpuTsThread, ThreadHandle* cpuTsThread, uint32_t* notifyNum)
{
    HcclResult ret
        = HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, userStream, kAicpuMainThreadNotifyNum, cpuTsThread);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    ret = HcclThreadExportToCommEngine(comm, 1, cpuTsThread, COMM_ENGINE_AICPU_TS, exportedAicpuTsThread);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    ret = HcclThreadExportToCommEngine(comm, 1, &aicpuThreadHandle, COMM_ENGINE_CPU_TS, exportedCpuTsThread);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    const auto threadRec = sim::runtime::Db::GetById<sim::runtime::HcclThread>(*exportedCpuTsThread);
    if (!threadRec.ok()) {
        HCCL_VM_ERROR("{}: exported thread {:d} not found", __func__, *exportedCpuTsThread);
        return HCCL_E_INTERNAL;
    }
    *notifyNum = threadRec->notifyNum;
    return HCCL_SUCCESS;
}

/**
 * @brief 建模 hcomm AicpuKernelLaunchDirect 的 h2d 同步段（record + device wait）。
 *
 * 对齐真实现两段动作：host 在用户流上插 h2d NOTIFY_RECORD
 * （HcommThreadNotifyRecordOnThread(cpuTsThread, exportedCpuTsThread, notifyNum-1)），
 * 随后在 unfold 流上下发 RunAicpuNotifyWaitAicpuKernel 等待同一 notify
 * （HcommThreadNotifyWaitOnThread(aicpuThreadHandle, notifyNum-1)）。
 * 导出桩为恒等映射（exportedCpuTsThread == aicpuThreadHandle），两侧 notifyId 取值一致，
 * record/wait 在 DB 中天然配对；wait 任务必须先于数据 kernel 插入，保证流序与真实现一致。
 *
 * @param cpuTsThread 输入：CPU_TS 主流线程（任务落在用户流）。
 * @param exportedCpuTsThread 输入：导出到 CPU_TS 引擎的 aicpuThreadHandle（record 目标）。
 * @param aicpuThreadHandle 输入：AICPU 通信主流线程（wait 任务落在其流上）。
 * @param notifyNum 输入：线程 notify 数量，使用 notifyNum-1 槽位。
 * @return HcclResult 成功返回 HCCL_SUCCESS；record/wait 桩调用失败返回 HCCL_E_INTERNAL。
 * @sideeffect 向 DB 插入 NOTIFY_RECORD（用户流）与 NOTIFY_WAIT（device 流）各一条。
 */
HcclResult InsertH2dNotifySyncTasks(
    ThreadHandle cpuTsThread, ThreadHandle exportedCpuTsThread, ThreadHandle aicpuThreadHandle, uint32_t notifyNum)
{
    const int32_t recordRet = HcommThreadNotifyRecordOnThread(cpuTsThread, exportedCpuTsThread, notifyNum - 1);
    if (recordRet != 0) {
        HCCL_VM_ERROR("{}: HcommThreadNotifyRecordOnThread failed, ret={:d}", __func__, recordRet);
        return HCCL_E_INTERNAL;
    }

    // 建模 LaunchNotifyWaitToThread：真实现为 unfold 流上的 wait kernel，仿真以同名 WAIT 任务代替
    const int32_t waitRet = HcommThreadNotifyWaitOnThread(aicpuThreadHandle, notifyNum - 1, 0);
    if (waitRet != 0) {
        HCCL_VM_ERROR("{}: HcommThreadNotifyWaitOnThread failed, ret={:d}", __func__, waitRet);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

/**
 * @brief 建模 hcomm AicpuKernelLaunchDirect 的 d2h 同步段（record + host wait）。
 *
 * 对齐真实现两段动作：数据 kernel 之后在 unfold 流上下发 RunAicpuNotifyRecordAicpuKernel
 * （HcommThreadNotifyRecordOnThread(aicpuThreadHandle, exportedAicpuTsThread, 0)），
 * 随后 host 在用户流上等待同一 notify
 * （HcommThreadNotifyWaitOnThreadWithDefaultTimeout(cpuTsThread, 0)，本桩无该变体，
 * 以 timeout=0 的 HcommThreadNotifyWaitOnThread 等价建模，仅插任务不真阻塞，不会挂死）。
 * 导出桩为恒等映射（exportedAicpuTsThread == cpuTsThread），两侧 notifyId 取值一致，
 * record/wait 在 DB 中天然配对。
 *
 * @param aicpuThreadHandle 输入：AICPU 通信主流线程（record 任务落在 device 流）。
 * @param exportedAicpuTsThread 输入：导出到 AICPU_TS 引擎的 cpuTsThread（record 目标）。
 * @param cpuTsThread 输入：CPU_TS 主流线程（wait 任务落在用户流）。
 * @return HcclResult 成功返回 HCCL_SUCCESS；record/wait 桩调用失败返回 HCCL_E_INTERNAL。
 * @sideeffect 向 DB 插入 NOTIFY_RECORD（device 流）与 NOTIFY_WAIT（用户流）各一条。
 * @pre 调用方须保证数据 kernel 已同步执行完毕（仿真 launch 桩为同步语义），
 *      以保证 record 任务排在 kernel 内部任务之后，与真实现流序一致。
 */
HcclResult
InsertD2hNotifySyncTasks(ThreadHandle aicpuThreadHandle, ThreadHandle exportedAicpuTsThread, ThreadHandle cpuTsThread)
{
    const int32_t recordRet = HcommThreadNotifyRecordOnThread(aicpuThreadHandle, exportedAicpuTsThread, 0);
    if (recordRet != 0) {
        HCCL_VM_ERROR("{}: HcommThreadNotifyRecordOnThread failed, ret={:d}", __func__, recordRet);
        return HCCL_E_INTERNAL;
    }

    // 建模真实现末尾的 host 阻塞等待（HcommThreadNotifyWaitOnThreadWithDefaultTimeout）
    const int32_t waitRet = HcommThreadNotifyWaitOnThread(cpuTsThread, 0, 0);
    if (waitRet != 0) {
        HCCL_VM_ERROR("{}: HcommThreadNotifyWaitOnThread failed, ret={:d}", __func__, waitRet);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

/**
 * @brief 校验 HcclAicpuKernelLaunch 入参（对齐 hcomm 入口校验口径）。
 *
 * 依次检查：comm/opInfo/funcInfo/userStream/kernelLaunchCfg 非空；
 * argSize>0 时 args 非空；argSize 不超 P2P_MAX_ARG_SIZE（防 opParams 溢出）。
 *
 * @param comm 输入：通信域句柄。
 * @param opInfo 输入：算子描述。
 * @param funcInfo 输入：kernel 函数信息。
 * @param userStream 输入：用户主流。
 * @param kernelLaunchCfg 输入：kernel 下发配置。
 * @return HcclResult 校验通过返回 HCCL_SUCCESS；否则返回 HCCL_E_PTR / HCCL_E_PARA。
 */
HcclResult ValidateAicpuLaunchInput(
    HcclComm comm, const HcclOpDesc* opInfo, const HcclKernelFuncInfo* funcInfo, aclrtStream userStream,
    const HcclKernelLaunchCfg* kernelLaunchCfg)
{
    if (comm == nullptr || opInfo == nullptr || funcInfo == nullptr || userStream == nullptr
        || kernelLaunchCfg == nullptr) {
        HCCL_VM_ERROR("{}: comm, opInfo, funcInfo, userStream or kernelLaunchCfg is nullptr", __func__);
        return HCCL_E_PTR;
    }
    if (funcInfo->argSize > 0 && funcInfo->args == nullptr) {
        HCCL_VM_ERROR("{}: funcInfo->args is nullptr but argSize={:d}", __func__, funcInfo->argSize);
        return HCCL_E_PTR;
    }
    if (funcInfo->argSize > P2P_MAX_ARG_SIZE) {
        HCCL_VM_ERROR("{}: argSize={:d} over P2P_MAX_ARG_SIZE", __func__, funcInfo->argSize);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

/**
 * @brief 加载 AICPU kernel 的 bin handle（kernel 名→so 名映射 json）。
 *
 * 对齐 hcomm 的 GetHcclBinHandle 口径：仿真桩 aclrtBinaryLoadFromFile 按路径缓存，
 * 重复调用同一路径返回同一 handle。json 路径由 ASCEND_OPP_PATH 环境变量推导
 * （与真实进程加载 libhccl_aicpu_kernel.json 的路径一致）。
 *
 * @param binHandle 输出：加载得到的 bin handle。
 * @return HcclResult 成功返回 HCCL_SUCCESS；环境变量缺失/加载失败返回 HCCL_E_RUNTIME。
 */
HcclResult LoadAicpuKernelBinHandle(aclrtBinHandle* binHandle)
{
    const char* oppPath = std::getenv("ASCEND_OPP_PATH");
    if (oppPath == nullptr || std::string(oppPath).empty()) {
        HCCL_VM_ERROR("{}: ASCEND_OPP_PATH is not set", __func__);
        return HCCL_E_RUNTIME;
    }
    const std::string jsonPath = std::string(oppPath) + "/built-in/op_impl/aicpu/config/libhccl_aicpu_kernel.json";
    const aclError ret = aclrtBinaryLoadFromFile(jsonPath.c_str(), nullptr, binHandle);
    if (ret != ACL_SUCCESS) {
        HCCL_VM_ERROR("{}: aclrtBinaryLoadFromFile failed, ret={:d}, path='{}'", __func__, ret, jsonPath);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

/**
 * @brief 准备 P2P kernel 的 args handle（对齐 hcomm LaunchP2pExec 参数装配段）。
 *
 * KernelArgsInit 后构造 HcclP2pKernelParam（sendRecvThread 回填线程句柄，funcInfo->args
 * 拷入 opParams）并 Append、Finalize，供后续 LaunchKernelWithConfig 使用。
 *
 * @param funcHandle 输入：kernel 函数句柄（aclrtBinaryGetFunction 产物）。
 * @param funcInfo 输入：kernel 函数信息（args 拷入 opParams）。
 * @param aicpuThreadHandle 输入：AICPU 通信主流线程句柄（回填 sendRecvThread）。
 * @param argsHandle 输出：装配完成的 args handle。
 * @return HcclResult 成功返回 HCCL_SUCCESS；装配任一步失败返回 HCCL_E_RUNTIME。
 * @pre 调用方须保证 funcInfo->argSize <= P2P_MAX_ARG_SIZE（HcclAicpuKernelLaunch 入口已校验）。
 */
HcclResult PrepareP2pKernelArgs(
    aclrtFuncHandle funcHandle, const HcclKernelFuncInfo* funcInfo, ThreadHandle aicpuThreadHandle,
    aclrtArgsHandle* argsHandle)
{
    aclError aclRet = aclrtKernelArgsInit(funcHandle, argsHandle);
    if (aclRet != ACL_SUCCESS) {
        HCCL_VM_ERROR("{}: aclrtKernelArgsInit failed, ret={:d}", __func__, aclRet);
        return HCCL_E_RUNTIME;
    }

    HcclP2pKernelParam params{};
    params.sendRecvThread = aicpuThreadHandle;
    if (funcInfo->argSize > 0) {
        (void)memcpy(params.opParams, funcInfo->args, funcInfo->argSize);
    }
    aclrtParamHandle paraHandle = nullptr;
    aclRet = aclrtKernelArgsAppend(*argsHandle, &params, sizeof(HcclP2pKernelParam), &paraHandle);
    if (aclRet != ACL_SUCCESS) {
        HCCL_VM_ERROR("{}: aclrtKernelArgsAppend failed, ret={:d}", __func__, aclRet);
        return HCCL_E_RUNTIME;
    }

    aclRet = aclrtKernelArgsFinalize(*argsHandle);
    if (aclRet != ACL_SUCCESS) {
        HCCL_VM_ERROR("{}: aclrtKernelArgsFinalize failed, ret={:d}", __func__, aclRet);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

/**
 * @brief 向 device 真下发 P2P AICPU kernel（HcclLaunchP2pAicpuKernel）。
 *
 * 对齐 hcomm LaunchP2pExec 的 ACL RTS 链：BinaryGetFunction → PrepareP2pKernelArgs →
 * LaunchKernelWithConfig（numBlocks=1，stream 取 opInfo->p2p.unfoldStream，超时属性取
 * kernelLaunchCfg->timeOut）。仿真 aclrt 桩经 pipe 把 kernel 转发给 device 进程真执行。
 * P2P kernel 无 notify 依赖，跨 rank 数据同步由 kernel 二进制内部完成。
 *
 * @param opInfo 输入：算子描述（stream 取其 p2p.unfoldStream）。
 * @param funcInfo 输入：kernel 函数信息（函数名、args，args 拷入 opParams）。
 * @param aicpuThreadHandle 输入：AICPU 通信主流线程句柄（回填 sendRecvThread）。
 * @param kernelLaunchCfg 输入：kernel 下发配置（超时）。
 * @return HcclResult 成功返回 HCCL_SUCCESS；加载/下发任一步失败返回 HCCL_E_RUNTIME。
 * @pre 调用方须保证 funcInfo->argSize <= P2P_MAX_ARG_SIZE（HcclAicpuKernelLaunch 入口已校验）。
 */
HcclResult LaunchP2pAicpuKernelDevice(
    const HcclOpDesc* opInfo, const HcclKernelFuncInfo* funcInfo, ThreadHandle aicpuThreadHandle,
    const HcclKernelLaunchCfg* kernelLaunchCfg)
{
    aclrtBinHandle binHandle = nullptr;
    const HcclResult ret = LoadAicpuKernelBinHandle(&binHandle);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }

    aclrtFuncHandle funcHandle = nullptr;
    aclError aclRet = aclrtBinaryGetFunction(binHandle, funcInfo->kernelFuncName, &funcHandle);
    if (aclRet != ACL_SUCCESS) {
        HCCL_VM_ERROR("{}: aclrtBinaryGetFunction failed, ret={:d}", __func__, aclRet);
        return HCCL_E_RUNTIME;
    }

    aclrtArgsHandle argsHandle = nullptr;
    const HcclResult argsRet = PrepareP2pKernelArgs(funcHandle, funcInfo, aicpuThreadHandle, &argsHandle);
    if (argsRet != HCCL_SUCCESS) {
        return argsRet;
    }

    aclrtLaunchKernelAttr attr{};
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
    attr.value.timeout = static_cast<uint16_t>(kernelLaunchCfg->timeOut);
    aclrtLaunchKernelCfg cfg{};
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    constexpr uint32_t kNumBlocks = 1;
    aclRet = aclrtLaunchKernelWithConfig(funcHandle, kNumBlocks, opInfo->p2p.unfoldStream, &cfg, argsHandle, nullptr);
    if (aclRet != ACL_SUCCESS) {
        HCCL_VM_ERROR("{}: aclrtLaunchKernelWithConfig failed, ret={:d}", __func__, aclRet);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}
} // namespace

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 查询 HCOMM 是否支持 HcclChannelDestroy。
 *
 * HCCL 用该能力探针门控 CCU 跨 rank 协商（CheckCcuResNegotiation /
 * CheckCcuParamAndFallback）：探针为 false 时各 rank 会独立回退，可能出现
 * 一侧回退到 AICPU、另一侧仍在 CCU 的 engine 不一致。HCCL-VM 的
 * HcclChannelDestroy 已由北向桩实现（dpu_control_stub.cc），这里固定返回 true。
 *
 * @return bool 恒返回 true。
 */
bool HcommIsSupportHcclChannelDestroy()
{
    HCCL_VM_INFO("{}: stub returns true", __func__);
    return true;
}

/**
 * @brief 将Thread资源导出到指定通信引擎上。
 *
 * 将传入的Thread句柄数组拷贝一份到exportedThreads中，供目标通信引擎直接引用。
 * 导出后在目标通信引擎直接使用，不需要额外的导入操作。
 *
 * @param comm 输入：通信域句柄，由HcclCommInitXxx系列接口返回。
 * @param threadNum 输入：待导出的Thread数量，不能为0。
 * @param threads 输入：源Thread句柄数组，长度为threadNum。
 * @param dstCommEngine 输入：目标通信引擎类型。
 * @param exportedThreads 输出：导出的Thread句柄数组，长度为threadNum，由调用者分配。
 *
 * @return HcclResult 接口成功返回HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS 导出成功
 * @retval HCCL_E_PTR comm、threads或exportedThreads为空
 * @retval HCCL_E_PARA threadNum为0
 */
HcclResult HcclThreadExportToCommEngine(
    HcclComm comm, uint32_t threadNum, const ThreadHandle* threads, CommEngine dstCommEngine,
    ThreadHandle* exportedThreads)
{
    if (comm == nullptr || threads == nullptr || exportedThreads == nullptr) {
        HCCL_VM_ERROR("{}: comm, threads or exportedThreads is nullptr", __func__);
        return HCCL_E_PTR;
    }
    if (threadNum == 0) {
        HCCL_VM_ERROR("{}: threadNum is 0", __func__);
        return HCCL_E_PARA;
    }
    for (uint32_t i = 0; i < threadNum; ++i) {
        exportedThreads[i] = threads[i];
    }

    uint64_t commId = reinterpret_cast<uint64_t>(comm);
    HCCL_VM_INFO(
        "{} success, commId={:d}, threadNum={:d}, dstCommEngine={:d}", __func__, commId, threadNum,
        static_cast<int>(dstCommEngine));
    return HCCL_SUCCESS;
}

/**
 * @brief 查找aicpu task cache, 判断tag是否已经缓存
 *
 * HCCL-VM 模拟环境不维护 aicpu task cache，桩函数始终返回 cache miss
 * （*isHit=false），强制调用方走完整算子展开路径。
 *
 * @param tag 输入：缓存标识符。
 * @param isHit 输出：是否cache hit，桩函数固定输出 false。
 *
 * @return HcommResult 接口成功返回HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS 查询完成（始终为 miss）
 * @retval HCCL_E_PTR tag或isHit指针为空
 */
int32_t HcommAicpuTsTaskCacheLookup(const char* tag, bool* isHit)
{
    if (tag == nullptr || isHit == nullptr) {
        HCCL_VM_ERROR("{}: tag or isHit is nullptr", __func__);
        return HCCL_E_PTR;
    }

    *isHit = false;
    HCCL_VM_INFO("{} success, tag='{}', isHit=false", __func__, tag);
    return HCCL_SUCCESS;
}

/**
 * @brief cache miss下, 算子展开前, 通知aicpu task cache开始缓存task
 *
 * HCCL-VM 模拟环境不维护 aicpu task cache，桩函数为空操作，直接返回成功。
 *
 * @param tag 输入：缓存标识符。
 * @param addrs 输入：内存基址信息数组。
 * @param sizes 输入：内存大小信息数组。
 * @param count 输入：内存信息数组长度。
 *
 * @return HcommResult 接口成功返回HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS 空操作成功
 * @retval HCCL_E_PTR tag、addrs或sizes指针为空
 */
HcommResult HcommAicpuTsTaskCacheStart(const char* tag, void** addrs, uint64_t* sizes, uint64_t count)
{
    if (tag == nullptr || addrs == nullptr || sizes == nullptr) {
        HCCL_VM_ERROR("{}: tag, addrs or sizes is nullptr", __func__);
        return HCCL_E_PTR;
    }

    HCCL_VM_INFO("{} success, tag='{}', count={:d}", __func__, tag, count);
    return HCCL_SUCCESS;
}

/**
 * @brief cache miss下, 算子展开后, 通知aicpu task cache停止缓存task
 *
 * HCCL-VM 模拟环境不维护 aicpu task cache，桩函数为空操作，直接返回成功。
 *
 * @param tag 输入：缓存标识符。
 *
 * @return HcommResult 接口成功返回HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS 空操作成功
 * @retval HCCL_E_PTR tag指针为空
 */
HcommResult HcommAicpuTsTaskCacheEnd(const char* tag)
{
    if (tag == nullptr) {
        HCCL_VM_ERROR("{}: tag is nullptr", __func__);
        return HCCL_E_PTR;
    }

    HCCL_VM_INFO("{} success, tag='{}'", __func__, tag);
    return HCCL_SUCCESS;
}

/**
 * @brief cache hit下, 刷新task并下发
 *
 * HCCL-VM 模拟环境不维护 aicpu task cache（Lookup 恒返回 miss），本桩函数
 * 理论上不会被调用，保留接口仅做参数校验并返回成功。
 *
 * @param tag 输入：缓存标识符。
 * @param addrs 输入：内存基址信息数组。
 * @param sizes 输入：内存大小信息数组。
 * @param count 输入：内存信息数组长度。
 *
 * @return HcommResult 接口成功返回HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS 空操作成功
 * @retval HCCL_E_PTR tag、addrs或sizes指针为空
 */
HcommResult HcommAicpuTsTaskCacheExecute(const char* tag, void** addrs, uint64_t* sizes, uint64_t count)
{
    if (tag == nullptr || addrs == nullptr || sizes == nullptr) {
        HCCL_VM_ERROR("{}: tag, addrs or sizes is nullptr", __func__);
        return HCCL_E_PTR;
    }

    HCCL_VM_INFO("{} success, tag='{}', count={:d}", __func__, tag, count);
    return HCCL_SUCCESS;
}

/**
 * @brief 清理aicpu task cache中指定tag对应的cache entry
 *
 * HCCL-VM 模拟环境不维护 aicpu task cache，桩函数为空操作，直接返回成功。
 *
 * @param tag 输入：缓存标识符。
 *
 * @return HcommResult 接口成功返回HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS 空操作成功
 * @retval HCCL_E_PTR tag指针为空
 */
HcommResult HcommAicpuTsTaskCacheClear(const char* tag)
{
    if (tag == nullptr) {
        HCCL_VM_ERROR("{}: tag is nullptr", __func__);
        return HCCL_E_PTR;
    }

    HCCL_VM_INFO("{} success, tag='{}'", __func__, tag);
    return HCCL_SUCCESS;
}

/**
 * @brief 启动 AICPU P2P kernel（HcclAicpuKernelEntranceLaunch 经 dlsym 解析后调用）。
 *
 * 与 hcomm 真实现（hccl_kernel_launch_aicpu.cc 的 HcclAicpuKernelLaunch +
 * AicpuKernelLaunchDirect）对齐：线程获取/导出、notify 数查询复用既有北向桩；
 * 同步段按真实现逐段建模——h2d record + device wait（InsertH2dNotifySyncTasks，
 * 对齐真实现数据 kernel 前的 RunAicpuNotifyWaitAicpuKernel）、d2h record + host wait
 * （InsertD2hNotifySyncTasks，对齐数据 kernel 后的 RunAicpuNotifyRecordAicpuKernel 与
 * 末尾 HcommThreadNotifyWaitOnThreadWithDefaultTimeout）。真实现中 wait/record 是
 * unfold 流上的独立 kernel，仿真中不在 kernel json 内、无实际同步语义，改为同名
 * NOTIFY 任务落库（notify 状态仅 runner 进程内 DB/私有 map 维护），DB 记录口径一致。
 * device 侧数据 kernel（HcclLaunchP2pAicpuKernel）经标准 ACL RTS 链真下发到 device
 * 进程执行（对齐 LaunchP2pExec，stream 取 opInfo->p2p.unfoldStream，跨 rank 数据
 * 同步由 kernel 二进制内部完成）。
 *
 * 暂不支持 group 模式（真实现在 hcclGroupDepth>0 时走 AppendGroupP2pTask 累积、
 * 由 groupLaunchA5 批量下发），HcclGroupStart/End 批式 P2P 请勿搭配 AI_CPU 展开模式使用。
 *
 * @param comm 输入：通信域句柄（level1 假句柄，值为 commId），不允许为空。
 * @param opInfo 输入：算子描述（P2P 描述取其 p2p 联合），不允许为空。
 * @param funcInfo 输入：kernel 函数信息（so 名、函数名、args），不允许为空。
 * @param aicpuThreadHandle 输入：AICPU 通信主流线程句柄。
 * @param userStream 输入：用户主流，不允许为空。
 * @param kernelLaunchCfg 输入：kernel 下发配置（超时等），不允许为空。
 * @return HcclResult 成功返回 HCCL_SUCCESS；参数为空返回 HCCL_E_PTR；
 *         线程资源获取/导出失败或线程记录缺失返回对应错误码；
 *         kernel 下发失败返回 HCCL_E_RUNTIME / HCCL_E_PARA。
 */
HcclResult HcclAicpuKernelLaunch(
    HcclComm comm, const HcclOpDesc* opInfo, const HcclKernelFuncInfo* funcInfo, ThreadHandle aicpuThreadHandle,
    aclrtStream userStream, const HcclKernelLaunchCfg* kernelLaunchCfg)
{
    const HcclResult validRet = ValidateAicpuLaunchInput(comm, opInfo, funcInfo, userStream, kernelLaunchCfg);
    if (validRet != HCCL_SUCCESS) {
        return validRet;
    }

    const uint64_t commId = reinterpret_cast<uint64_t>(comm);
    const std::string kernelSo(funcInfo->kernelSoName, strnlen(funcInfo->kernelSoName, HCCL_KERNEL_SO_NAME_MAX_LEN));
    const std::string kernelFunc(
        funcInfo->kernelFuncName, strnlen(funcInfo->kernelFuncName, HCCL_KERNEL_FUNC_NAME_MAX_LEN));
    HCCL_VM_INFO(
        "{} enter, commId={:d}, opDescType={:d}, kernelSo='{}', kernelFunc='{}', argSize={:d}, "
        "aicpuThread={:d}",
        __func__, commId, opInfo->opDescType, kernelSo, kernelFunc, funcInfo->argSize, aicpuThreadHandle);

    ThreadHandle exportedAicpuTsThread = 0;
    ThreadHandle exportedCpuTsThread = 0;
    ThreadHandle cpuTsThread = 0;
    uint32_t notifyNum = 0;
    const HcclResult ret = AcquireAndExportAicpuThreads(
        comm, userStream, aicpuThreadHandle, &exportedAicpuTsThread, &exportedCpuTsThread, &cpuTsThread, &notifyNum);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    if (notifyNum == 0) {
        HCCL_VM_ERROR("{}: exported thread {:d} has no notify", __func__, exportedCpuTsThread);
        return HCCL_E_INTERNAL;
    }

    // h2d 同步段（record + device wait）：对齐真实现，wait 任务须先于数据 kernel 落库
    const HcclResult h2dRet = InsertH2dNotifySyncTasks(cpuTsThread, exportedCpuTsThread, aicpuThreadHandle, notifyNum);
    if (h2dRet != HCCL_SUCCESS) {
        return h2dRet;
    }

    // device 数据 kernel 真下发：对齐 LaunchP2pExec（仿真 launch 桩同步执行，返回即 kernel 完成）
    const HcclResult launchRet = LaunchP2pAicpuKernelDevice(opInfo, funcInfo, aicpuThreadHandle, kernelLaunchCfg);
    if (launchRet != HCCL_SUCCESS) {
        HCCL_VM_ERROR("{}: LaunchP2pAicpuKernelDevice failed, ret={:d}", __func__, static_cast<int>(launchRet));
        return launchRet;
    }

    // d2h 同步段（record + host wait）：对齐真实现，record 任务排在 kernel 内部任务之后
    const HcclResult d2hRet = InsertD2hNotifySyncTasks(aicpuThreadHandle, exportedAicpuTsThread, cpuTsThread);
    if (d2hRet != HCCL_SUCCESS) {
        return d2hRet;
    }

    HCCL_VM_INFO(
        "{} success, commId={:d}, exportedAicpuTsThread={:d}, exportedCpuTsThread={:d}, notifyNum={:d}", __func__,
        commId, exportedAicpuTsThread, exportedCpuTsThread, notifyNum);
    return HCCL_SUCCESS;
}

#ifdef __cplusplus
}
#endif
