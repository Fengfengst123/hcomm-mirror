# StreamTaskMonitor task 耗时监控模块说明

---

## 功能描述

`StreamTaskMonitor` 是 HCOMM 在 `coll_communicator_mgr/dfx/streamTaskMonitor/` 路径下的 AICPU 背景线程 task 粒度耗时监控模块（L2 层），用于 **在集群训练场景中快速定位性能抖动或卡死的异常根因卡**。

核心能力：
1. **周期性巡检**：注册到 `AicpuDaemonService` 背景线程，每轮遍历所有通信域的所有流，从 SQ 队列 head 取当前 SQE，统计队头 task 的执行耗时。
2. **阈值告警**：task 停留时间超过配置阈值（`taskMonitorInterval`，单位 ms）时输出监控日志，打印后重置计时基线，按阈值间隔重复打印。
3. **安全降级**：QuerySqStatus 失败、GetRtsq 返回空、GetSqeHeaderFieldsBySqIdx 失败等异常路径打 ERROR 并返回错误码，错误传播至 Call() 触发 stopCall_ 永久停止监控；FindDfxTaskInfo 返回 nullptr（notify 类型 SQE 在 DfxTaskInfo 队列中可能无记录，属正常情况）时打印告警并 return SUCCESS 跳过本轮，不停止监控。
4. **卡内 notify wait 跳过**：对卡内 notify wait（remoteRank 为 INVALID）跳过打印，仅重置计时；task_exception 未开启时无法区分卡内/跨卡，notify_wait 整体跳过。

---

## 配置方法

通过环境变量 `HCCL_DFS_CONFIG` 配置：
```bash
export HCCL_DFS_CONFIG="task_exception:on, task_monitor_interval:5000"
```

| 配置项 | 取值范围 | 默认值 | 说明 |
|--------|----------|--------|------|
| `task_monitor_interval` | [0, 7200000] ms | 0 | 0 表示关闭监控；>0 时使能，task 停留超过该阈值时打印告警日志 |

配置经 `DfsConfig` → `DevAicpuCommConfig.taskMonitorInterval` → `CollCommAicpuMgr::InitIndopEnv` → `StreamTaskMonitor::SetInterval` 链路下发到 AICPU 侧。

---

## 目录描述

```text
streamTaskMonitor/
├── CMakeLists.txt              # 顶层构建脚本，仅 add_subdirectory(aicpu)
└── aicpu/
    ├── CMakeLists.txt          # 将 stream_task_monitor.cc 加入 ccl_kernel 目标
    ├── stream_task_monitor.h   # 类声明（单例，继承 Hccl::DaemonFunc）
    └── stream_task_monitor.cc  # Call() 实现：遍历通信域 → 遍历线程 → 耗时判定 → 打印
```

---

## 接口描述

### 公共接口

| 接口 | 说明 |
|------|------|
| `static StreamTaskMonitor& GetInstance()` | 获取单例引用 |
| `void SetInterval(u32 interval)` | 设置监控阈值（ms），同时重置 stopCall_ |
| `void SetTaskExceptionEnable(bool enable)` | 设置 task_exception 开关 |
| `void Init(u32 devId)` | 保存设备 ID |
| `void Call() override` | 守护周期调用入口：遍历所有活跃通信域的流，统计队头 task 耗时 |
| `void OnCommDestroy(CollCommAicpu* aicpuComm)` | 通信域销毁时清理对应流的监控数据 |

### Daemon 注册

| 时机 | 调用方 | 操作 |
|------|--------|------|
| 通信域初始化 | `CollCommAicpuMgr::InitBackGroundThread(devId)` | `StreamTaskMonitor::Init(devId)` + `AicpuDaemonService::Register(&StreamTaskMonitor::GetInstance())` |

---

## 监控日志格式

```
[StreamTaskMonitor]prof monitor streamId:%u, sqid:%u, head:%u, tail:%u, elapsed[%llu us] threshold[%u ms], %s
```

- `elapsed`：task 在队头的停留耗时（us）
- `threshold`：配置的监控阈值（ms）
- 尾部为 taskName（task_exception 开启时，已含 notifyId）或 sqeType/sqeId/notifyId（task_exception 关闭时）
- 告警说明队头 task 确实卡住，但本监控无法区分卡在本卡还是跨卡对端，需结合 task_exception/cqe 日志进一步定位

---

## 使用限制

| 类别 | 约束 |
|------|------|
| **平台依赖** | 在 AICPU 侧设备启用 |
| **线程模型** | Call() 在 AicpuDaemonService 后台线程中执行，约 10ms 周期 |
| **并发安全** | streamTaskMonitor_ map 仅在 Call()/OnCommDestroy() 中访问，由 mutexForFuncs_ 串行保护 |
| **性能影响** | interval=0 时 IsNoNeedMonitor 直接返回，几乎零开销 |
| **层级归属** | L2（coll_communicator_mgr/dfx），符合分层依赖方向约束 |
