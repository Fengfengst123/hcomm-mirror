/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_SUB_PROCESS_MANAGER_H
#define SIM_SUB_PROCESS_MANAGER_H

#include <sys/types.h>
#include <cstdint>
#include <mutex>
#include <map>
#include <string>
#include <vector>
#include <sys/uio.h>
#include "sim_aicpu_pipe_msg.h"

namespace sim {

struct SubProcessConfig {
    std::string executable;
    std::vector<std::string> args;
    std::map<std::string, std::string> envVars;
    std::string workDir;
};

class SubProcessManager {
public:
    SubProcessManager() = default;
    ~SubProcessManager();

    SubProcessManager(const SubProcessManager&) = delete;
    SubProcessManager& operator=(const SubProcessManager&) = delete;

    int CreateProcess(const SubProcessConfig& config);
    int DestroyProcess(int timeoutMs = 3000);

    bool IsAlive() const;
    pid_t GetPid() const { return m_pid; }

    int Request(
        uint8_t reqCmd, const void* reqData, uint32_t reqLen, uint8_t& rspCmd, void* rspData, uint32_t rspMaxLen,
        uint32_t& rspLen);

    int RequestV(
        uint8_t reqCmd, const struct iovec* reqIov, uint32_t reqSegCnt, uint8_t& rspCmd, void* rspData,
        uint32_t rspMaxLen, uint32_t& rspLen);

private:
    static constexpr int kH2dReadFd = 200;
    static constexpr int kD2hWriteFd = 201;
    // QEMU 启动 + 进入 READY 的等待预算（毫秒）：覆盖冷启动抖动，又避免子进程
    // 卡死（而非退出）时阻塞读把 host 线程永久挂住。
    static constexpr int kSpawnReadyTimeoutMs = 15000;

    int m_h2dWriteFd = -1;
    int m_d2hReadFd = -1;
    pid_t m_pid = -1;

    mutable std::mutex m_rpcLock;
    mutable std::mutex m_forkLock;

    // 等待 device 子进程 READY 握手；timeoutMs>0 时先 poll 防子进程卡死把 host 挂住。
    int WaitForReady(int timeoutMs);

    // 单次 device 子进程拉起：建管对 + posix_spawnp + 等 READY。
    // 不用 fork：多线程进程 fork 会继承"蒸发线程"持有的锁，CANN atfork 钩子
    // （libascend_trace）在子进程死锁（futex_do_wait，走不到 exec）。
    // 失败路径已自行 DestroyProcess 清场（m_pid 归 -1），由 CreateProcess 循环重试。
    int SpawnOnce(const SubProcessConfig& config);

    // 拼接 device 进程 exec 参数：config.args 尾部追加 h2d/d2h 管道 fd 参数并以 nullptr 结尾。
    static std::vector<char*> BuildSpawnArgv(const SubProcessConfig& config);

    int HostSendMsg(uint8_t cmd, const void* data, uint32_t len);
    int HostSendMsgV(uint8_t cmd, const struct iovec* iov, uint32_t iovcnt);
    int HostRecvMsg(uint8_t& outCmd, void* outData, uint32_t maxLen, uint32_t& outLen);
};

SubProcessConfig CreateAicpuDeviceConfig(uint32_t rankId, uint32_t deviceKey = 0);

/**
 * @brief 取指定 deviceKey 对应的 AICPU device 子进程管理器。
 *
 * 资源隔离粒度是“每 deviceKey（= rank 身份）一份”，而非“每线程一份”：
 * 单进程多线程 examples 中不同 rank 线程用不同 deviceKey，各得一独立 device 进程；
 * 多进程 hccl_test 中同 rank 的主线程与工作线程（hcomm 内部线程）共用同一 deviceKey，
 * 共享同一 device 进程，避免每个线程重复拉起 QEMU 子进程。
 * 线程安全性：首次调用时登记本线程的 deviceKey；注册表由内部互斥量保护，
 * 返回的管理器可被多线程并发使用（Request 内部已串行化）。
 *
 * @param deviceKey Device 表主键；传 0 表示沿用本线程最近一次登记的值。
 * @return 该 deviceKey 的管理器引用（注册表生命周期内稳定，不会随线程退出销毁）。
 */
SubProcessManager& GetAicpuProcMgr(uint64_t deviceKey = 0);

/**
 * @brief 销毁注册表中全部 AICPU device 子进程并清空注册表。
 *
 * 用于进程级收尾（aclFinalize）：单进程多线程场景下需收掉全部 deviceKey 的子进程，
 * 不能只收当前线程对应的一个。进程正常退出时注册表析构也会逐个销毁（幂等）。
 */
void DestroyAllAicpuProcMgrs();
} // namespace sim

#endif
