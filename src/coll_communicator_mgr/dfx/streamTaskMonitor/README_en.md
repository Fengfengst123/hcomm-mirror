# StreamTaskMonitor Task Execution Time Monitoring Module

---

## Overview

`StreamTaskMonitor` is an AICPU background thread task-level execution time monitoring module located at `coll_communicator_mgr/dfx/streamTaskMonitor/` (L2 layer). It is designed to **quickly identify the root cause card of performance jitter or hang in cluster training scenarios**.

Key capabilities:
1. **Periodic inspection**: Registered to `AicpuDaemonService` background thread, iterates over all communicators and streams each cycle, reads the current SQE from the SQ queue head, and measures the execution time of the head task.
2. **Threshold alerting**: Outputs a monitoring log when a task's dwell time exceeds the configured threshold (`taskMonitorInterval`, in ms). After printing, the timing baseline is reset, so alerts repeat at threshold intervals.
3. **Safe degradation**: QuerySqStatus failure, GetRtsq returning null, GetSqeHeaderFieldsBySqIdx failure, and other error paths log ERROR and return an error code, which propagates to Call() to trigger stopCall_ and permanently stop monitoring. When FindDfxTaskInfo returns nullptr (notify-type SQEs may have no DfxTaskInfo record, which is normal), an alert is logged and SUCCESS is returned to skip the current cycle without stopping monitoring.
4. **Intra-card notify wait skip**: Intra-card notify wait (remoteRank is INVALID) skips printing, only resets timing; when task_exception is disabled, intra/cross-card cannot be distinguished, so all notify_wait is skipped.

---

## Configuration

Configure via the `HCCL_DFS_CONFIG` environment variable:

```bash
export HCCL_DFS_CONFIG="task_exception:on, task_monitor_interval:5000"
```

| Config Item | Range | Default | Description |
|------------|-------|---------|-------------|
| `task_monitor_interval` | [0, 7200000] ms | 0 | 0 disables monitoring; >0 enables it, printing alert logs when a task dwells longer than the threshold |

The configuration is delivered to the AICPU side via `DfsConfig` → `DevAicpuCommConfig.taskMonitorInterval` → `CollCommAicpuMgr::InitIndopEnv` → `StreamTaskMonitor::SetInterval`.

---

## Directory Structure

```text
streamTaskMonitor/
├── CMakeLists.txt              # Top-level build script, add_subdirectory(aicpu) only
└── aicpu/
    ├── CMakeLists.txt          # Adds stream_task_monitor.cc to ccl_kernel target
    ├── stream_task_monitor.h   # Class declaration (singleton, inherits Hccl::DaemonFunc)
    └── stream_task_monitor.cc  # Call() implementation: iterate comms → iterate threads → time check → print
```

---

## API

### Public Interface

| API | Description |
|-----|-------------|
| `static StreamTaskMonitor& GetInstance()` | Get singleton reference |
| `void SetInterval(u32 interval)` | Set monitoring threshold (ms), resets stopCall_ |
| `void SetTaskExceptionEnable(bool enable)` | Set task_exception switch |
| `void Init(u32 devId)` | Save device ID |
| `void Call() override` | Daemon periodic entry: iterate all active comm streams, measure head task dwell time |
| `void OnCommDestroy(CollCommAicpu* aicpuComm)` | Clean up monitoring data for a destroyed communicator's streams |

### Daemon Registration

| Timing | Caller | Operation |
|--------|--------|-----------|
| Communicator init | `CollCommAicpuMgr::InitBackGroundThread(devId)` | `StreamTaskMonitor::Init(devId)` + `AicpuDaemonService::Register(&StreamTaskMonitor::GetInstance())` |

---

## Monitoring Log Format

```
[StreamTaskMonitor]prof monitor streamId:%u, sqid:%u, head:%u, tail:%u, elapsed[%llu us] threshold[%u ms], %s
```

- `elapsed`: task dwell time at queue head (us)
- `threshold`: configured monitoring threshold (ms)
- Trailing field is taskName (when task_exception enabled, includes notifyId) or sqeType/sqeId/notifyId (when disabled)
- The alert indicates the head task is genuinely stuck, but this monitor cannot distinguish whether the cause is local or the cross-card peer; cross-reference task_exception/cqe logs to localize

---

## Constraints

| Category | Constraint |
|----------|-----------|
| **Platform** | Enabled on AICPU device side |
| **Threading** | Call() runs in AicpuDaemonService background thread, ~10ms cycle |
| **Concurrency** | streamTaskMonitor_ map accessed only in Call()/OnCommDestroy(), serialized by mutexForFuncs_ |
| **Performance** | interval=0 returns immediately via IsNoNeedMonitor, near-zero overhead |
| **Layer** | L2 (coll_communicator_mgr/dfx), complies with layered dependency constraints |
