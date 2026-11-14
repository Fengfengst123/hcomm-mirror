/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sim_sub_process_manager.h"
#include <sys/wait.h>
#include <sys/utsname.h>
#include <fcntl.h>
#include <poll.h>
#include <spawn.h>
#include <unistd.h>

#include <cerrno>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <sstream>
#include <thread>
#include <unordered_map>
#include "sim_log.h"
#include "sim_common_defs.h"
#include "sim_common_api.h"
#include "sim_pipe_io.h"
#include "store_sim_resource_root.h"

namespace sim {

namespace {
    namespace fs = std::filesystem;

    // HCCL_VM_QEMU_TB_SIZE 未设置时的默认翻译代码缓存（MiB）
    constexpr uint32_t kDefaultQemuTbSizeMiB = 16;
    // 版本探测日志截断长度，避免发行版长版本串刷屏
    constexpr size_t kQemuVersionLogMax = 48;

    bool IsCompleteCannSysroot(const fs::path& sysroot, std::string& missingPath)
    {
        const std::vector<fs::path> requiredPaths{
            sysroot / "lib64/ld-linux-aarch64.so.1", sysroot / "lib64/libc.so.6", sysroot / "usr/lib64/libstdc++.so.6",
            sysroot / "usr/lib64/libgcc_s.so.1"};

        std::error_code ec;
        for (const auto& requiredPath : requiredPaths) {
            ec.clear();
            if (!fs::exists(requiredPath, ec)) {
                missingPath = requiredPath.string();
                if (ec) {
                    missingPath += ": " + ec.message();
                }
                return false;
            }
        }
        return true;
    }

    std::string ResolveCannSysroot(const char* ascendHomePath)
    {
        const char* configuredSysroot = std::getenv("CANN_HCC_SYSROOT");
        if (configuredSysroot != nullptr && configuredSysroot[0] != '\0') {
            std::string missingPath;
            if (IsCompleteCannSysroot(configuredSysroot, missingPath)) {
                return configuredSysroot;
            }
            HCCL_VM_ERROR(
                "invalid CANN_HCC_SYSROOT [{}]: required component [{}] was not found.", configuredSysroot,
                missingPath);
            return {};
        }

        if (ascendHomePath == nullptr || ascendHomePath[0] == '\0') {
            HCCL_VM_ERROR("CANN sysroot is not configured; set ASCEND_HOME_PATH or CANN_HCC_SYSROOT.");
            return {};
        }

        const std::vector<fs::path> candidates{
            fs::path(ascendHomePath) / "tools/hcc/sysroot", fs::path(ascendHomePath) / "toolkit/toolchain/hcc/sysroot"};
        std::ostringstream diagnostics;
        for (const auto& candidate : candidates) {
            std::string missingPath;
            if (IsCompleteCannSysroot(candidate, missingPath)) {
                return candidate.string();
            }
            diagnostics << "\n  " << candidate.string() << ": missing " << missingPath;
        }

        HCCL_VM_ERROR(
            "complete CANN sysroot was not found under ASCEND_HOME_PATH [{}]. Tried:{}", ascendHomePath,
            diagnostics.str());
        return {};
    }

    // HCCL_VM_QEMU_PATH 指定的 QEMU 程序校验：必须是非空绝对路径，目标为可执行的
    // 普通文件（允许最终指向普通文件的符号链接）。失败原因写入 reason 并返回空串，
    // 由调用方决定回退（PATH 查找）或硬失败（全机找不到 QEMU）。
    std::string ResolveConfiguredQemuProgram(const char* configuredPath, std::string& reason)
    {
        reason.clear();
        if (configuredPath == nullptr || configuredPath[0] == '\0') {
            reason = "is not set";
            return {};
        }
        const fs::path givenPath(configuredPath);
        if (!givenPath.is_absolute()) {
            reason = "must be an absolute path: " + std::string(configuredPath);
            return {};
        }
        std::error_code ec;
        const fs::file_status status = fs::status(givenPath, ec); // 跟随符号链接，断链在此暴露
        if (ec || !fs::exists(status)) {
            reason = "target does not exist: " + std::string(configuredPath);
            return {};
        }
        if (!fs::is_regular_file(status)) {
            reason = "target is not a regular file: " + std::string(configuredPath);
            return {};
        }
        if (access(configuredPath, X_OK) != 0) {
            reason = "target is not executable: " + std::string(configuredPath);
            return {};
        }
        return std::string(configuredPath);
    }

    // 按 PATH 顺序查找首个可执行的普通文件（用于 HCCL_VM_QEMU_PATH 未配置/无效时
    // 的回退，以及 timeout 工具的定位）。找不到返回空串。
    std::string ResolveExecutableInPath(const char* name)
    {
        if (name == nullptr || name[0] == '\0') {
            return {};
        }
        const char* pathEnv = std::getenv("PATH");
        if (pathEnv == nullptr || pathEnv[0] == '\0') {
            return {};
        }
        std::istringstream pathStream(pathEnv);
        std::string dir;
        while (std::getline(pathStream, dir, ':')) {
            if (dir.empty()) {
                continue;
            }
            const fs::path candidate = fs::path(dir) / name;
            std::error_code ec;
            if (!fs::is_regular_file(fs::status(candidate, ec)) || ec) {
                continue;
            }
            if (access(candidate.c_str(), X_OK) == 0) {
                return candidate.string();
            }
        }
        return {};
    }

    // 捕获命令输出头部（探测用途，最多 kQemuProbeOutputMax 字节）。返回值：-1 表示
    // popen/pclose 失败或子进程被信号终止；否则为子进程退出码。
    constexpr size_t kQemuProbeOutputMax = 2048;
    int RunCapture(const std::string& command, std::string& output)
    {
        output.clear();
        FILE* pipe = ::popen(command.c_str(), "r");
        if (pipe == nullptr) {
            return -1;
        }
        char buf[512];
        while (output.size() < kQemuProbeOutputMax) {
            size_t n = ::fread(buf, 1, sizeof(buf), pipe);
            if (n == 0) {
                break;
            }
            output.append(buf, n);
        }
        const int rawRc = ::pclose(pipe);
        if (rawRc == -1 || !WIFEXITED(rawRc)) {
            return -1;
        }
        return WEXITSTATUS(rawRc);
    }

    // 一次性探测 QEMU 版本与 -tb-size 能力：优先 `timeout 5 <program> -tb-size 16
    // -version`（rc==0 即支持，输出同时含版本行）；不支持时再取纯 -version 输出
    // 尽力解析版本。每进程仅在创建 AICPU 设备时调用一次。
    void ProbeQemuVersionAndTbSize(const std::string& program, bool& supportsTbSize, std::string& version)
    {
        supportsTbSize = false;
        version = "unknown";
        const std::string quotedProgram = "'" + program + "'";
        // timeout 定位为绝对路径：sh 经 PATH 查找，rank 环境的 PATH 可能不含系统目录
        const std::string timeoutTool = ResolveExecutableInPath("timeout");
        const std::string guard = timeoutTool.empty() ? "" : timeoutTool + " 5 ";
        std::string output;
        const int tbRc = RunCapture(guard + quotedProgram + " -tb-size 16 -version 2>&1", output);
        supportsTbSize = (tbRc == 0);
        if (!supportsTbSize) {
            std::string versionOutput;
            if (RunCapture(guard + quotedProgram + " -version 2>&1", versionOutput) == 0) {
                output = versionOutput;
            }
        }
        const std::string prefix = "qemu-aarch64 version ";
        std::istringstream in(output);
        std::string line;
        while (std::getline(in, line)) {
            const auto pos = line.find(prefix);
            if (pos == std::string::npos) {
                continue;
            }
            std::string ver = line.substr(pos + prefix.size());
            if (ver.size() > kQemuVersionLogMax) {
                ver.resize(kQemuVersionLogMax);
            }
            while (!ver.empty() && (ver.back() == ' ' || ver.back() == '\r' || ver.back() == '\n')) {
                ver.pop_back();
            }
            if (!ver.empty()) {
                version = ver;
            }
            break;
        }
    }

    // HCCL_VM_QEMU_TB_SIZE 解析：未设置时使用默认值；仅接受可完整解析到 uint32_t
    // 的十进制数字串（允许前导 0，如 0016）。空串、正负号、空白、单位、前缀或尾随
    // 字符、超出 uint32_t 范围均视为非法（由调用方告警并回退默认值）。解析成功后
    // 由调用方按十进制重新序列化进 argv，避免 QEMU 对前导 0 串的自动进制解析偏差。
    bool ParseQemuTbSizeMiB(const char* configuredValue, uint32_t& outValueMiB, std::string& reason)
    {
        outValueMiB = kDefaultQemuTbSizeMiB;
        reason.clear();
        if (configuredValue == nullptr) {
            return true;
        }
        const std::string text(configuredValue);
        if (text.empty()) {
            reason = "HCCL_VM_QEMU_TB_SIZE is empty";
            return false;
        }
        for (const char c : text) {
            if (c < '0' || c > '9') {
                reason = "HCCL_VM_QEMU_TB_SIZE must be decimal digits only: " + text;
                return false;
            }
        }
        errno = 0;
        char* endPtr = nullptr;
        const unsigned long long parsed = std::strtoull(text.c_str(), &endPtr, 10);
        if (errno == ERANGE || endPtr == nullptr || *endPtr != '\0' || parsed > std::numeric_limits<uint32_t>::max()) {
            reason = "HCCL_VM_QEMU_TB_SIZE is out of uint32_t range: " + text;
            return false;
        }
        outValueMiB = static_cast<uint32_t>(parsed);
        return true;
    }
} // namespace

// 每 deviceKey 一份 AICPU device 子进程管理器：
// - 单进程多线程 examples：8 个 rank 线程用 8 个不同 deviceKey → 8 个 device 进程（隔离目标）；
// - 多进程 hccl_test：同 rank 的主线程与 hcomm 工作线程同 deviceKey → 共享 1 个 device 进程
//   （与历史“进程级单例”行为一致，避免工作线程重复拉起 QEMU 子进程）。
namespace {
    struct ProcMgrRegistry {
        std::mutex lock;
        std::map<uint64_t, std::shared_ptr<SubProcessManager>> mgrs;
    };

    /**
     * @brief 取全局管理器注册表（函数内静态，避免静态初始化顺序问题）。
     * @return 注册表单例引用。
     */
    ProcMgrRegistry& GetProcMgrRegistry()
    {
        static ProcMgrRegistry registry;
        return registry;
    }

    // 本线程最近一次显式指定（aclrtSetDevice / TryLaunchAicpuDevProcForRank）的 deviceKey。
    // 仅用于把“无参调用”解析到正确槽位；不参与跨线程共享判断。
    thread_local uint64_t g_threadDeviceKey = 0;
} // namespace

/**
 * @brief 取指定 deviceKey 的 AICPU device 子进程管理器（按 key 共享，线程安全）。
 * @param deviceKey Device 表主键；0 表示沿用本线程最近一次登记的值。
 * @return 管理器引用；未登记且无法唯一判定归属时返回 key=0 的空管理槽。
 */
SubProcessManager& GetAicpuProcMgr(uint64_t deviceKey)
{
    if (deviceKey != 0) {
        g_threadDeviceKey = deviceKey;
    }
    uint64_t key = (deviceKey != 0) ? deviceKey : g_threadDeviceKey;

    ProcMgrRegistry& registry = GetProcMgrRegistry();
    std::lock_guard<std::mutex> lock(registry.lock);

    if (key == 0) {
        // 未登记过 deviceKey 的线程（如部分 hcomm 内部线程）回退：全进程只有一个存活
        // 管理器时直接复用，保持历史“进程级单例”语义；多设备并存时无法判定归属，
        // 落到 key=0 的空槽（IsAlive()=false，调用方按未启用 AICPU 模式处理）。
        SubProcessManager* onlyAlive = nullptr;
        size_t aliveCount = 0;
        for (auto& item : registry.mgrs) {
            if (item.second != nullptr && item.second->IsAlive()) {
                onlyAlive = item.second.get();
                ++aliveCount;
            }
        }
        if (aliveCount == 1) {
            return *onlyAlive;
        }
    }

    auto iter = registry.mgrs.find(key);
    if (iter == registry.mgrs.end()) {
        iter = registry.mgrs.emplace(key, std::make_shared<SubProcessManager>()).first;
    }
    return *iter->second;
}

/**
 * @brief 销毁注册表中全部 device 子进程并清空注册表（进程级收尾，幂等）。
 */
void DestroyAllAicpuProcMgrs()
{
    ProcMgrRegistry& registry = GetProcMgrRegistry();
    std::lock_guard<std::mutex> lock(registry.lock);
    for (auto& item : registry.mgrs) {
        if (item.second != nullptr) {
            item.second->DestroyProcess();
        }
    }
    registry.mgrs.clear();
}

SubProcessManager::~SubProcessManager()
{
    if (m_pid > 0) {
        DestroyProcess();
    }
}

// 创建 device 子进程：最多尝试 kMaxSpawnAttempts 次，任一成功即返回。
// 单进程多 rank 并发 spawn 时 QEMU fork/exec 偶发"子进程未就绪"，重试一次可避免整 rank 早退。
int SubProcessManager::CreateProcess(const SubProcessConfig& config)
{
    std::lock_guard<std::mutex> lock(m_forkLock);

    if (config.executable.empty() || config.args.empty()) {
        HCCL_VM_ERROR("Invalid subprocess config: executable or argv is empty.");
        return -1;
    }

    if (m_pid > 0) {
        HCCL_VM_INFO("Process {} already exists, skip.", m_pid);
        return 0;
    }

    // 单进程多 rank 场景下 QEMU fork/exec 偶发"子进程未就绪"（peer 提前关管）；
    // 首次失败时自动重试一次，避免整 rank 早退导致 AllReduce 等不齐而 join 死锁。
    const int kMaxSpawnAttempts = 2;
    for (int attempt = 1; attempt <= kMaxSpawnAttempts; ++attempt) {
        if (attempt > 1) {
            HCCL_VM_WARN("{}: retry subprocess spawn (attempt {}/{})", __func__, attempt, kMaxSpawnAttempts);
        }
        int rc = SpawnOnce(config);
        if (rc == 0) {
            return 0;
        }
    }
    return -1;
}

// 拼接 device 进程 exec 参数：config.args 尾部追加 h2d/d2h 管道 fd 参数并以 nullptr 结尾。
std::vector<char*> SubProcessManager::BuildSpawnArgv(const SubProcessConfig& config)
{
    std::vector<char*> argv;
    for (const auto& arg : config.args) {
        argv.push_back(const_cast<char*>(arg.c_str()));
    }

    // fd 是编译期常量，直接用字面量：std::to_string 的局部 string 在函数返回后
    // 销毁，指针会悬垂（posix_spawnp 异步消费 argv，悬垂即垃圾/崩溃）。
    static_assert(kH2dReadFd == 200 && kD2hWriteFd == 201, "argv fd literals must match kH2dReadFd/kD2hWriteFd");
    argv.push_back(const_cast<char*>("200"));
    argv.push_back(const_cast<char*>("201"));
    argv.push_back(nullptr);
    return argv;
}

// 构造 device 子进程 exec 用环境块（environ 快照 + config.envVars 覆盖）。
// 在父进程侧（posix_spawnp 之前）构建：父进程内存分配安全，规避多线程进程
// fork 后在子进程 malloc/setenv 的死锁风险。返回空串表示构建失败。
// 仅覆盖 config.envVars 里显式声明的键（不设新值时沿用父进程环境，等效原 fork 语义）。
static std::string BuildSpawnEnv(const SubProcessConfig& config)
{
    std::unordered_map<std::string, std::string> env;
    for (char* const* ep = environ; *ep != nullptr; ++ep) {
        const std::string entry(*ep);
        const size_t eq = entry.find('=');
        if (eq != std::string::npos) {
            env[entry.substr(0, eq)] = entry.substr(eq + 1);
        }
    }
    for (const auto& kv : config.envVars) {
        env[kv.first] = kv.second;
    }
    std::string out;
    out.reserve(4096);
    for (const auto& kv : env) {
        out += kv.first;
        out += '=';
        out += kv.second;
        out += '\0';
    }
    return out;
}

// 一次性关闭两对管道全部端（SpawnOnce 失败路径用）。
// @param h2d/d2h 两对管道（fd 会被置 -1）。
static void CloseAllPipes(sim::PipePair& h2d, sim::PipePair& d2h)
{
    sim::PipeClose(h2d.readFd);
    sim::PipeClose(h2d.writeFd);
    sim::PipeClose(d2h.readFd);
    sim::PipeClose(d2h.writeFd);
}

// 打开 stderr 捕获文件（spawn 前在父进程侧完成；子进程不运行我方代码，靠
// file_actions 重定向）。文件名先用序号占位，spawn 拿到 pid 后由调用方 rename。
// @param resRoot 资源根目录；@param seq 本次 spawn 序号（区分重试）。
// @param errPath 输出路径缓冲；@return 文件 fd，打开失败返回 -1。
static int OpenStderrCapture(const std::string& resRoot, uint32_t seq, char* errPath, size_t errPathSize)
{
    snprintf(
        errPath, errPathSize, "%s/device_stderr_tmp_%d_%u.log", resRoot.c_str(), static_cast<int>(::getpid()), seq);
    return open(errPath, O_CREAT | O_WRONLY | O_TRUNC, 0644);
}

// 把 NUL 分隔的环境块（BuildSpawnEnv 产物）转成 envp 指针数组。
// @param envBlock 环境块；@param envp 输出指针数组（末尾补 nullptr）。
static void EnvBlockToEnvp(std::string& envBlock, std::vector<char*>& envp)
{
    if (envBlock.empty()) {
        return;
    }
    envp.reserve(envBlock.size() / 8 + 2);
    for (size_t i = 0; i < envBlock.size();) {
        envp.push_back(envBlock.data() + i);
        i += strlen(envBlock.data() + i) + 1;
    }
    envp.push_back(nullptr);
}

// 登记子进程文件动作：就位 h2d 读端/d2h 写端到固定 fd（dup2 清 CLOEXEC）、
// 关掉子进程不需要的管道端、stderr 重定向到捕获文件。
// @param actions 已 init 的动作表；@param h2dPipe/d2hPipe 两对管道；
// @param targetReadFd/targetWriteFd 子进程约定的固定管道 fd；
// @param errFd stderr 捕获 fd（<0 则跳过重定向）。
// @return 0 成功；非 0 为首个失败的错误码。
static int SetupSpawnFileActions(
    posix_spawn_file_actions_t* actions, const sim::PipePair& h2dPipe, const sim::PipePair& d2hPipe, int targetReadFd,
    int targetWriteFd, int errFd)
{
    auto addDup2 = [actions](int from, int to) {
        if (from == to) {
            // dup2 同 fd 无操作且会清 CLOEXEC；file_actions 无对应动作，
            // 父进程侧先清 CLOEXEC（spawn 后父进程随即关闭该端，无副作用）。
            fcntl(from, F_SETFD, 0);
            return 0;
        }
        return posix_spawn_file_actions_adddup2(actions, from, to);
    };
    int rc = 0;
    rc |= addDup2(h2dPipe.readFd, targetReadFd);
    rc |= addDup2(d2hPipe.writeFd, targetWriteFd);
    rc |= posix_spawn_file_actions_addclose(actions, h2dPipe.writeFd);
    rc |= posix_spawn_file_actions_addclose(actions, d2hPipe.readFd);
    if (h2dPipe.readFd != targetReadFd) {
        rc |= posix_spawn_file_actions_addclose(actions, h2dPipe.readFd);
    }
    if (d2hPipe.writeFd != targetWriteFd) {
        rc |= posix_spawn_file_actions_addclose(actions, d2hPipe.writeFd);
    }
    if (errFd >= 0) {
        rc |= posix_spawn_file_actions_adddup2(actions, errFd, STDERR_FILENO);
        rc |= posix_spawn_file_actions_addclose(actions, errFd);
    }
    return rc;
}

// 创建 h2d/d2h 两对管道（SpawnOnce 前置步骤）。
// @param h2dPipe/d2hPipe 输出管道对；@return 0 成功；非 0 失败（已自行部分清理）。
static int CreatePipePairs(sim::PipePair& h2dPipe, sim::PipePair& d2hPipe)
{
    if (sim::PipeCreate(h2dPipe) != 0) {
        HCCL_VM_ERROR("Failed to create h2d pipe");
        return -1;
    }
    if (sim::PipeCreate(d2hPipe) != 0) {
        HCCL_VM_ERROR("Failed to create d2h pipe");
        sim::PipeClose(h2dPipe.readFd);
        sim::PipeClose(h2dPipe.writeFd);
        return -1;
    }
    return 0;
}

// spawn 失败清场：关管道、关并删除 stderr 捕获文件。
static void CleanupFailedSpawn(sim::PipePair& h2dPipe, sim::PipePair& d2hPipe, int errFd, const char* errPath)
{
    CloseAllPipes(h2dPipe, d2hPipe);
    if (errFd >= 0) {
        close(errFd);
        unlink(errPath);
    }
}

// spawn 成功后收尾：父进程侧关掉 stderr 捕获 fd，并把占位文件名 rename 为
// device_stderr_<pid>.log（保持按子进程 pid 定位的调试约定）。
static void FinalizeStderrCapture(const std::string& resRoot, int errFd, const char* errPath, pid_t childPid)
{
    if (errFd < 0) {
        return;
    }
    close(errFd); // 父进程不需要写端
    char finalPath[512] = {0};
    snprintf(finalPath, sizeof(finalPath), "%s/device_stderr_%d.log", resRoot.c_str(), static_cast<int>(childPid));
    rename(errPath, finalPath);
}

// 创建并启动一次 device 子进程：管对 + posix_spawnp + 等 READY。
// 用 posix_spawnp 替代 fork+exec：glibc 走 clone(CLONE_VM|CLONE_VFORK)，不执行
// pthread_atfork 子进程钩子、不做地址空间拷贝。多线程（每 rank 一线程）+ CANN
// 第三方库（libascend_trace 注册 atfork）场景下，fork 子进程会继承"已蒸发线程"
// 持有的锁并死锁（futex_do_wait，等不到 dup2/exec）；posix_spawnp 从根上规避。
// 每次调用重建一对新 fd；失败路径内部已 DestroyProcess 清场（m_pid 归 -1），调用方可安全重试。
int SubProcessManager::SpawnOnce(const SubProcessConfig& config)
{
    sim::PipePair h2dPipe{}, d2hPipe{};
    if (CreatePipePairs(h2dPipe, d2hPipe) != 0) {
        return -1;
    }

    // stderr 捕获文件与 exec 参数/环境块全部在父进程侧准备（posix_spawnp 无子进程用户代码）。
    static std::atomic<uint32_t> s_stderrSeq{0};
    const std::string& resRoot = sim::SimResourceRoot::GetInstance().GetRoot();
    char errPath[512] = {0};
    const int errFd
        = OpenStderrCapture(resRoot, s_stderrSeq.fetch_add(1, std::memory_order_relaxed), errPath, sizeof(errPath));
    std::vector<char*> argv = BuildSpawnArgv(config);
    std::string envBlock = BuildSpawnEnv(config);
    std::vector<char*> envp;
    EnvBlockToEnvp(envBlock, envp);

    posix_spawn_file_actions_t actions;
    const int initRc = posix_spawn_file_actions_init(&actions);
    const int actionRc
        = initRc == 0 ? SetupSpawnFileActions(&actions, h2dPipe, d2hPipe, kH2dReadFd, kD2hWriteFd, errFd) : initRc;
    if (actionRc != 0) {
        HCCL_VM_ERROR("posix_spawn file actions setup failed: {}", strerror(actionRc));
        if (initRc == 0) {
            posix_spawn_file_actions_destroy(&actions);
        }
        CleanupFailedSpawn(h2dPipe, d2hPipe, errFd, errPath);
        return -1;
    }

    pid_t pid = -1;
    const int spawnRc = posix_spawnp(
        &pid, config.executable.c_str(), &actions, nullptr, argv.data(), envp.empty() ? nullptr : envp.data());
    posix_spawn_file_actions_destroy(&actions);
    if (spawnRc != 0) {
        HCCL_VM_ERROR("posix_spawnp({}) failed: {}", config.executable, strerror(spawnRc));
        CleanupFailedSpawn(h2dPipe, d2hPipe, errFd, errPath);
        return -1;
    }
    FinalizeStderrCapture(resRoot, errFd, errPath, pid);

    m_pid = pid;
    sim::PipeClose(h2dPipe.readFd);  // 主进程不读 h2dPipe
    sim::PipeClose(d2hPipe.writeFd); // 主进程不写 d2hPipe
    m_h2dWriteFd = h2dPipe.writeFd;
    m_d2hReadFd = d2hPipe.readFd;

    HCCL_VM_INFO("Child process {} created (m_h2dWriteFd={}, m_d2hReadFd={}).", m_pid, m_h2dWriteFd, m_d2hReadFd);

    // 等待device进程管道准备就绪进入Ready状态
    if (WaitForReady(kSpawnReadyTimeoutMs) != 0) {
        HCCL_VM_ERROR("Child process {} failed to become READY, aborting.", m_pid);
        DestroyProcess();
        return -1;
    }

    return 0;
}

// 等待 device 子进程发来 READY 握手（pipe 消息 cmd==PIPE_RSP_READY）。
// @param timeoutMs >0 时先 poll 等 d2h 可读（防子进程卡死把 host 线程永久挂住）；
//        <=0 时退化为纯阻塞读（历史行为）。
// @return 0 成功；非 0 表示超时/读失败/子进程退出/握手命令异常。
int SubProcessManager::WaitForReady(int timeoutMs)
{
    if (m_d2hReadFd < 0 || m_pid < 0) {
        HCCL_VM_ERROR("WaitForReady: child not running.");
        return -1;
    }

    // 先 poll 等待 d2h 可读：子进程卡死（而非退出）时阻塞读会挂住 host 线程；
    // 超时返回让上层走重试/报错路径。
    if (timeoutMs > 0) {
        struct pollfd pfd = {m_d2hReadFd, POLLIN, 0};
        int pollRc;
        do {
            pollRc = poll(&pfd, 1, timeoutMs);
        } while (pollRc < 0 && errno == EINTR);
        if (pollRc == 0) {
            HCCL_VM_ERROR("WaitForReady: child pid={} not READY within {} ms.", m_pid, timeoutMs);
            return -1;
        }
        if (pollRc < 0) {
            HCCL_VM_ERROR("WaitForReady: poll failed: {}", strerror(errno));
            return -1;
        }
    }

    uint8_t rspCmd = 0;
    uint8_t buf[16] = {0};
    uint32_t rspLen = 0;
    if (HostRecvMsg(rspCmd, buf, sizeof(buf), rspLen) != 0) {
        HCCL_VM_ERROR("WaitForReady: child pid={} exited or read failed.", m_pid);
        return -1;
    }
    if (rspCmd != PIPE_RSP_READY) {
        HCCL_VM_ERROR("WaitForReady: unexpected msg cmd=0x{:02x}.", rspCmd);
        return -1;
    }
    HCCL_VM_INFO("Child process {} is READY.", m_pid);
    return 0;
}

int SubProcessManager::DestroyProcess(int timeoutMs)
{
    if (m_pid < 0) {
        HCCL_VM_INFO("Process already stopped.");
        return 0;
    }

    HCCL_VM_INFO("Sending SHUTDOWN to process {}.", m_pid);
    {
        std::lock_guard<std::mutex> lock(m_rpcLock);
        HostSendMsg(PIPE_CMD_SHUTDOWN, nullptr, 0);
    }

    sim::PipeClose(m_h2dWriteFd);

    auto startTime = std::chrono::steady_clock::now();
    while (true) {
        int status;
        pid_t wPid = waitpid(m_pid, &status, WNOHANG);
        if (wPid == m_pid) {
            HCCL_VM_INFO("Child process {} exited with status {}.", m_pid, status);
            break;
        } else if (wPid == -1 && errno == ECHILD) {
            break;
        }

        auto elapsed
            = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime)
                  .count();
        if (elapsed >= timeoutMs) {
            HCCL_VM_WARN("Child process {} did not exit within {} ms, SIGKILL.", m_pid, timeoutMs);
            kill(m_pid, SIGKILL);
            waitpid(m_pid, &status, 0);
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    sim::PipeClose(m_d2hReadFd);

    m_pid = -1;
    return 0;
}

bool SubProcessManager::IsAlive() const
{
    if (m_pid < 0) {
        return false;
    }

    int status = 0;
    pid_t wPid = waitpid(m_pid, &status, WNOHANG);
    return wPid == 0;
}

int SubProcessManager::Request(
    uint8_t reqCmd, const void* reqData, uint32_t reqLen, uint8_t& rspCmd, void* rspData, uint32_t rspMaxLen,
    uint32_t& rspLen)
{
    std::lock_guard<std::mutex> lock(m_rpcLock);
    if (HostSendMsg(reqCmd, reqData, reqLen) != 0) {
        return -1;
    }

    if (HostRecvMsg(rspCmd, rspData, rspMaxLen, rspLen) != 0) {
        return -1;
    }

    return 0;
}

int SubProcessManager::RequestV(
    uint8_t reqCmd, const struct iovec* reqIov, uint32_t reqSegCnt, uint8_t& rspCmd, void* rspData, uint32_t rspMaxLen,
    uint32_t& rspLen)
{
    std::lock_guard<std::mutex> lock(m_rpcLock);
    if (HostSendMsgV(reqCmd, reqIov, reqSegCnt) != 0) {
        return -1;
    }

    if (HostRecvMsg(rspCmd, rspData, rspMaxLen, rspLen) != 0) {
        return -1;
    }

    return 0;
}

int SubProcessManager::HostSendMsg(uint8_t cmd, const void* data, uint32_t len)
{
    if (m_h2dWriteFd < 0) {
        HCCL_VM_ERROR("HostSendMsg: h2d pipe not open.");
        return -1;
    }

    return PipeSendMsg(m_h2dWriteFd, cmd, data, len);
}

int SubProcessManager::HostSendMsgV(uint8_t cmd, const struct iovec* iov, uint32_t iovcnt)
{
    if (m_h2dWriteFd < 0) {
        HCCL_VM_ERROR("HostSendMsgV: h2d pipe not open.");
        return -1;
    }

    return PipeSendMsgV(m_h2dWriteFd, cmd, iov, iovcnt);
}

int SubProcessManager::HostRecvMsg(uint8_t& outCmd, void* outData, uint32_t maxLen, uint32_t& outLen)
{
    if (m_d2hReadFd < 0) {
        HCCL_VM_ERROR("HostRecvMsg: d2h pipe not open.");
        return -1;
    }

    return PipeRecvMsg(m_d2hReadFd, outCmd, outData, maxLen, outLen);
}

bool IsAarch64Host()
{
    utsname utsBuf;
    uname(&utsBuf);
    return (std::strstr(utsBuf.machine, "aarch64") != nullptr || std::strstr(utsBuf.machine, "arm") != nullptr);
}

SubProcessConfig CreateAicpuDeviceConfig(uint32_t rankId, uint32_t deviceKey)
{
    SubProcessConfig config;
    const char* ascendHomePath = std::getenv("ASCEND_HOME_PATH");
    const std::string cannSysroot = ResolveCannSysroot(ascendHomePath);
    if (cannSysroot.empty()) {
        return config;
    }

    const std::string devBinPath = InstallPath::ResolveToInstallRoot("bin/device");
    const std::string preloadPath = InstallPath::ResolveToInstallRoot("lib/aarch64/libhccl_device_proxy.so");
    const std::string preloadPathL1 = InstallPath::ResolveToInstallRoot("lib/aarch64/libhccl_device_proxy_level1.so");
    std::string libPath = InstallPath::ResolveToInstallRoot("lib/aarch64");
    if (ascendHomePath != nullptr) {
        libPath += ":" + std::string(ascendHomePath) + "/" + GetArchStr() + "-linux/devlib/device";
    }
    libPath += ":" + cannSysroot + "/lib64";
    libPath += ":" + cannSysroot + "/usr/lib64";

    if (IsAarch64Host()) {
        const std::string cannLoaderPath = (fs::path(cannSysroot) / "lib64/ld-linux-aarch64.so.1").string();
        config.executable = cannLoaderPath;
        config.args = {cannLoaderPath, "--library-path", libPath, devBinPath};
    } else {
        // x86_64 宿主：AICPU device 经外部 QEMU（linux-user 模拟 aarch64）启动。
        // 程序优先取 HCCL_VM_QEMU_PATH 指定的 10.2.4 部署（README §2.3 步骤产物）；
        // 未配置/配置无效/为不支持 -tb-size 的旧版时自动回退（PATH 查找/不传该
        // 选项）并打 WARN（含实际版本与内存提示），保证旧部署仍可运行；仅当全机
        // 找不到 QEMU 时硬失败（走现有空配置路径）。
        std::string reason;
        uint32_t qemuTbSizeMiB = kDefaultQemuTbSizeMiB;
        if (!ParseQemuTbSizeMiB(std::getenv("HCCL_VM_QEMU_TB_SIZE"), qemuTbSizeMiB, reason)) {
            qemuTbSizeMiB = kDefaultQemuTbSizeMiB;
            HCCL_VM_WARN(
                "AICPU QEMU tb-size is invalid ({}), fallback to default {} MiB.", reason, kDefaultQemuTbSizeMiB);
        }
        std::string qemuProgram = ResolveConfiguredQemuProgram(std::getenv("HCCL_VM_QEMU_PATH"), reason);
        bool pathFallback = false;
        if (qemuProgram.empty()) {
            qemuProgram = ResolveExecutableInPath("qemu-aarch64-static");
            if (qemuProgram.empty()) {
                HCCL_VM_ERROR(
                    "AICPU QEMU program not found: HCCL_VM_QEMU_PATH {} and no executable "
                    "qemu-aarch64-static in PATH. Deploy QEMU 10.2.4 per README section 2.3.",
                    reason);
                return config;
            }
            pathFallback = true;
        }
        bool supportsTbSize = false;
        std::string qemuVersion;
        ProbeQemuVersionAndTbSize(qemuProgram, supportsTbSize, qemuVersion);
        config.executable = qemuProgram;
        config.envVars["QEMU_LD_PREFIX"] = cannSysroot;
        if (pathFallback) {
            HCCL_VM_WARN(
                "AICPU QEMU fallback: HCCL_VM_QEMU_PATH is not set or invalid ({}), using "
                "PATH lookup [{}] ({}).",
                reason, qemuProgram, qemuVersion);
        }
        if (supportsTbSize) {
            // 显式 -tb-size 置于 device 路径之前。日志记录请求值；0 表示把缓存
            // 策略委托给 QEMU 自身默认（不是零缓存，也不虚报实际分配量）。
            config.args = {qemuProgram, "-tb-size", std::to_string(qemuTbSizeMiB), devBinPath};
            HCCL_VM_INFO(
                "AICPU QEMU: program=[{}], version=[{}], -tb-size request={} MiB{}.", qemuProgram, qemuVersion,
                qemuTbSizeMiB, qemuTbSizeMiB == 0 ? " (0 delegates cache sizing to QEMU default)" : "");
        } else {
            // 旧版 QEMU 不支持 -tb-size：回退为不限制翻译缓存并告警内存风险
            config.args = {qemuProgram, devBinPath};
            HCCL_VM_WARN(
                "AICPU QEMU fallback: [{}] ({}) does not support -tb-size, running without "
                "translation cache limit; QEMU memory may be large at scale. Deploy QEMU 10.2.4 per "
                "README section 2.3.",
                qemuProgram, qemuVersion);
        }
    }

    config.args.push_back(std::to_string(rankId));
    config.args.push_back(std::to_string(deviceKey));

    uint32_t vmLevel = 2;
    const char* levelEnv = std::getenv("HCCL_VM_LEVEL");
    if (levelEnv != nullptr && levelEnv[0] != '\0') {
        char* endptr = nullptr;
        unsigned long val = std::strtoul(levelEnv, &endptr, 10);
        if (endptr != levelEnv && *endptr == '\0') {
            vmLevel = static_cast<uint32_t>(val);
        }
    }
    if (vmLevel == 1) {
        config.envVars["LD_PRELOAD"] = preloadPathL1 + ":" + preloadPath;
    } else {
        config.envVars["LD_PRELOAD"] = preloadPath;
    }
    config.envVars["LD_LIBRARY_PATH"] = libPath;

    HCCL_VM_INFO("AICPU CANN runtime: sysroot=[{}], executable=[{}].", cannSysroot, config.executable);

    return config;
}

} // namespace sim
