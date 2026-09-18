# CCU Executor Debug Tool Design Document — Feasibility Review and Optimization Report

A comprehensive review based on the code in `src/plugin/solver/virtual_runtime/ccu_executor/` and the design proposal in `runner_debug/debug_tool_design_doc.md`.

---

## I. Feasibility Issues (Must Fix)

### P0-1: CKE Wait Flow Does Not Match Instrumentation Design

**Severity**: P0 (Approach feasibility error)

**Problem Description**:

The instrumentation pseudocode in Section 4.1 assumes that the `waitCKE_` flag can be detected before `ExecuteInstr()` returns, and the delta from `snapshotBefore → snapshotAfter` is computed afterward. However, the actual code flow is:

```
WaitCkeProcess() → CKE not met → ccuSimulator_->SetWaitCKEFlag(true) → return (Process() not called)
→ UpdateLoopStatus() returns false → ExecuteInstr() returns false → Execute() returns false
```

Key point: When CKE is not met, `WaitCkeProcess` inside `Run()` sets the waitCKE flag and returns directly, **without calling Process() and without modifying any resources**. Therefore:
- Pre-execution snapshot = post-execution snapshot (delta is empty)
- snapshotAfter should not be captured and delta should not be computed

**Code Evidence**:

`CcuExecutorBase.cc` lines 162-172:
```cpp
if ((waitCKE & waitCKEMask) == waitCKEMask) {
    Process(ccuResMgr);      // CKE met → execute Process
} else {
    ccuSimulator_->SetWaitCKEFlag(true);  // CKE not met → set flag, do not execute Process
    return;
}
```

**Fix Suggestion**:

When waitCKE is not met, only record key spin fields; do not compute delta or create a full trace entry:
```cpp
// During waitCKE, only record spin information
CcuTraceCollector::RecordWaitSpin(rankId_, dieId_, curInstrId,
                                    waitCKEId, waitCKEMask, actualCKEValue);
// Do not capture snapshotAfter, do not compute delta
// Do not create full CcuTraceEntry
return false;
```

Only create a full entry after CKE passes (merging previous spin information).

---

### P0-2: globalSeqId/execRound Maintenance Should Be in SequentialExecutor

**Severity**: P0 (Approach feasibility error)

**Problem Description**:

The document assumes `CcuTraceCollector` maintains `globalSeqId/execRound/currentSqeTaskId` inside `CcuSimulator::ExecuteInstr()`. However, `CcuSimulator` can only see execution within its own CCU and cannot perceive global scheduling rounds or CCU switching.

`CcuSimulator::ExecuteInstr()` executes within a single CCU and does not know:
- Which round of the outer `while(HasTask())` it is (`execRound`)
- The global execution sequence number (`globalSeqId`)
- Which SQE task is currently being executed (`sqeTaskId`)

This information is only available in the upper call chain of `SequentialExecutor::Execute()` and `TaskCcuGraph()`.

**Code Evidence**:

`hccl_task_sequential_execute.cc` lines 51-72 — Global scheduling loop:
```cpp
while (HasTask()) {
    uint32_t rankId = 0;
    for (auto& rankTasks : allRankTaskQueues_) {  // Iterate all ranks
        for (auto& streamTasks : rankTasks) {     // Iterate each stream
            while (!streamTasks.empty()) {
                auto task = streamTasks.front();
                auto ret = ExecuteOneTask(task);   // ← globalSeqId incremented here
                if (ret == HCCL_SIM_VRT_HOLD_CMD) break;
                streamTasks.pop();
            }
        }
    }
}
```

`hccl_task_thread.cc` lines 135-148 — TaskCcuGraph:
```cpp
auto simulator = ccuResMgr.InitSimulator(rankId, dieId, instrStartId, endInstrId, instCnt);
// sqeTaskId is known here (obtained from task parameters)
if (simulator->Execute() == false) {
    return HCCL_SIM_VRT_HOLD_CMD;
}
```

**Fix Suggestion**:

Global context is managed by `SequentialExecutor` and passed via parameters:
```cpp
// In SequentialExecutor::Execute()
while (HasTask()) {
    uint32_t execRound = 0;
    for (rankTasks : allRankTaskQueues_) {
        for (streamTasks : rankTasks) {
            while (!streamTasks.empty()) {
                CcuTraceCollector::BeginGlobalStep(execRound);
                // globalSeqId incremented inside ExecuteOneTask
                auto ret = ExecuteOneTask(task);
                ...
            }
        }
    }
    execRound++;
}
```

Pass sqeTaskId to `CcuSimulator` in `TaskCcuGraph()`:
```cpp
uint32_t sqeTaskId = CcuTraceCollector::RegisterSqeTask(rankId, dieId, task);
CcuSimulator::SetCurrentSqeTaskId(sqeTaskId);  // Used internally by Simulator
```

---

### P0-3: Full Snapshot Collection Has Excessive Performance Overhead

**Severity**: P0 (Performance bottleneck)

**Problem Description**:

In Section 4.1, a `CaptureResourceSnapshot()` (full read of all resources) is performed before and after each instruction, with significant overhead:

- Single snapshot: 4096×8(XN) + 4096×8(GSA) + 1024×2(CKE) ≈ 67KB
- Before and after each instruction ≈ 134KB
- 10000 instructions ≈ 1.34GB (snapshot data only)

Section 4.5 mentions incremental collection optimization, but the pseudocode in 4.1 still follows the full snapshot + diff computation flow, which contradicts the optimization approach.

**Industry Reference**:

- **rr debugger**: Checkpoint + incremental recording model; full snapshots only at key nodes
- **Chrome DevTools Timeline**: Event stream format; each event contains only changed args
- **Perfetto (Android trace)**: Protobuf binary incremental format

**Fix Suggestion**:

Change to ResourceManager change interception + initial snapshot + periodic checkpoints:

1. **Initial Snapshot**: Full snapshot at run start (stored in `CcuIdentity.initialSnapshot`)
2. **Change Interception**: Intercept changes directly in `CcuResourceManager::UpdateXnValue/UpdateGsaValue/UpdateCkeValue` etc., recording delta:
```cpp
void CcuResourceManager::UpdateXnValue(int rankId, int dieId, uint16_t xnId, uint64_t value) {
    uint64_t oldValue = GetXnValue(rankId, dieId, xnId);
    if (oldValue != value) {
        CcuTraceCollector::RecordXnDelta(rankId, dieId, xnId, oldValue, value);
    }
    xnData_[rankId][dieId][xnId] = value;
}
```
3. **Periodic Checkpoint**: Full snapshot every N instructions (N configurable, default 1000) for quick intermediate state recovery
4. **Restore Arbitrary Point State**: From nearest checkpoint + accumulate subsequent deltas

---

## II. Design Optimization Points (Recommended Improvements)

### P1-1: Loop Offset Formula Is Incomplete

**Severity**: P1 (Information gap)

**Problem Description**:

The offset calculation in `UpdateAddress()` in the code is more complex than the `CcuExecutionContext.gsaOffset` in the document:
```cpp
uint64_t CcuExecutorBase::UpdateAddress(uint64_t addr, uint16_t addrExpandCoef) {
    return addr + ((ccuSimulator_->GetLoopExtendNum() * ccuSimulator_->GetGSAOffset()) << addrExpandCoef)
        + ((ccuSimulator_->GetCurLoopCnt() * ccuSimulator_->GetLoopIterStepGSA()) << addrExpandCoef);
}
```

This involves 5 parameters: `addr` (original address), `addrExpandCoef` (expansion coefficient), `loopExtendNum`, `gsaOffset`, `curLoopCnt`, `loopIterStepGSA`.

The current `CcuExecutionContext` has only a simple `gsaOffset` field and cannot fully reproduce the Loop GSA address offset calculation process.

**Fix Suggestion**:

Add complete offset parameters to `CcuExecutionContext`:
```cpp
struct CcuExecutionContext {
    bool inLoop;
    uint16_t loopRound;          // Current iteration round
    uint16_t loopExtendIndex;    // Expand index

    // Loop offset parameters (valid only when inLoop=true)
    uint64_t gsaAddrOffset;      // = extendIndex * gsaOffset + curLoopCnt * iterStepGSA (computed final offset)
    uint64_t gsaOffset;          // GSA base offset coefficient (GetGSAOffset)
    uint64_t iterStepGSA;        // GSA step per iteration round (GetLoopIterStepGSA)
    uint32_t curLoopCnt;         // Current iteration count (GetCurLoopCnt)
    uint32_t loopExtendNum;      // Expand count (GetLoopExtendNum)
    uint16_t addrExpandCoef;     // Address expansion coefficient

    uint16_t msOffset;           // MS ID offset (GetLoopMsOffset)
    uint16_t ckeOffset;          // CKE ID offset (GetLoopCKEOffset)
    uint16_t xnIdOffset;         // XN ID offset (GetLoopXnIdOffset)
};
```

---

### P1-2: Cross-CCU Changes Should Be Intercepted in ResourceManager

**Severity**: P1 (Collection accuracy)

**Problem Description**:

In Section 4.1, the cross-CCU change collection point is a post-hoc call to `CaptureCrossCcuChanges()` in `CcuSimulator::ExecuteInstr()`. However, cross-CCU operations actually occur in the Executor's `Process()` method (e.g., `SyncCkeExecutor::Process()` calls `SetRmtCKESignal()`).

The post-hoc collection approach requires comparing the remote CCU's before/after state, meaning a remote CCU snapshot would be needed before and after instruction execution — high overhead and potentially missing changes (the remote CCU could be simultaneously modified by other code).

**Fix Suggestion**:

Intercept cross-CCU changes directly in `CcuResourceManager::UpdateCkeValue()`:
```cpp
void CcuResourceManager::UpdateCkeValue(int rankId, int dieId, uint16_t ckeId, uint16_t value) {
    uint16_t oldValue = GetCkeValue(rankId, dieId, ckeId);
    // Detect whether this is a cross-CCU operation
    auto [execRank, execDie] = CcuTraceCollector::GetCurrentExecutingCcu();
    if (rankId != execRank || dieId != execDie) {
        CcuTraceCollector::RecordCrossCcuCkeChange(
            rankId, dieId, ckeId, oldValue, value, execRank, execDie);
    }
    ckeData_[rankId][dieId][ckeId] = value;
}
```

Similarly applicable to `UpdateXnValue()` (TransXnToRmt operations), `UpdateGsaValue()`, `TransMSToMS()`, etc.

---

### P1-3: CcuInstrTraceDetail Polymorphism Is Over-engineered

**Severity**: P1 (Architecture optimization)

**Problem Description**:

The current design has **13** `CcuInstrTraceDetail` subclasses (Load/Arith/Trans/Sync/Loop/LoopGroup/Cke/Jump/Wait/Fence/Reduce/...), but there are the following issues:

1. **Duplication with common layer**: `CcuTransTraceDetail`'s `waitCKEId/setCKEId` duplicates `CcuTraceEntry.waitInfo`; `CcuCkeTraceDetail`'s `ckeId/ckeValueBefore/ckeValueAfter` duplicates `resourceDelta.ckeChanges`
2. **Subclass proliferation**: The code has 30+ instruction types (~20 in V1 + ~10+ in V2); each requires a subclass
3. **Complex frontend rendering**: Requires dispatching different rendering templates based on typeName
4. **High cost of new instructions**: Each new instruction type requires a new subclass struct + CollectTraceDetail implementation

**Industry Reference**:

- **Chrome DevTools Trace Event Format**: Flat `args` field (key-value map), no polymorphic inheritance
- **LLVM Execution Trace**: Flat "additional info" field
- **GDB MI (Machine Interface)**: Key-value property list

**Fix Suggestion**:

Simplify `CcuInstrTraceDetail` to a flat structure:
```cpp
struct CcuInstrTraceDetail {
    std::string typeName;                     // e.g., "TransLocMemToLocMem"
    std::map<std::string, std::string> args;  // Instruction-specific parameter key-value table
    // Examples:
    //   Trans type: {"srcAddr": "0x7f0000", "dstAddr": "0x7f1000", "length": "4KB", "channelId": "0"}
    //   Reduce type: {"op": "Add", "msList": "[0,1,2,3]", "dataType": "FP32", "castEn": "0"}
    //   CKE type: {"ckeOp": "Set", "ckeId": "3", "ckeMask": "0x0001", "isRemote": "true"}
};
```

Advantages:
- No need to define a subclass for each instruction type
- New instructions only need to populate args in `CollectTraceDetail()`
- Unified frontend rendering logic (key-value table)
- `Describe()` output can be used directly as args source

---

### P1-4: JSON Serialization Format vs Large Data Volume

**Severity**: P1 (Performance optimization)

**Problem Description**:

Section 3.4 adopts JSON as the serialization format, but JSON has weaknesses:
- Each uint64_t XN/GSA value requires string representation (`"0x7f3a0000"`), 8 bytes becomes ~16 bytes
- Large amounts of redundant key name repetition (`"xnChanges"` repeated in every entry)
- 10000 instructions' JSON file can reach 50MB+
- Frontend loading and parsing of large JSON files is seriously slow

**Industry Reference**:

| Tool | Format | Characteristics |
|------|------|------|
| Chrome DevTools | JSON streaming (one event per line) | Supports line-by-line parsing |
| Perfetto (Android) | Protobuf binary | Frontend WASM parsing, small size |
| rr debugger | Custom binary | Efficient recording/playback |
| Intel VTune | Custom binary + SQLite | On-demand querying |

**Fix Suggestion**:

Layered serialization strategy:
- **Collection side**: msgpack/flatbuffers binary format, 5-10x smaller, faster writes
- **Frontend loading**: Decode binary into frontend index structures via Worker
- **Export/debug**: Optional export to JSON (for human readability and debugging)
- JSON Schema `metadata.rankId/dieId` should be removed (changed to `rankSize/diePerRank`, consistent with global model)

---

## III. Design Optimization Points (Recommended Improvements, Lower Priority)

### P2-1: Describe() Can Serve as MVP Trace Information Source

**Severity**: P2 (Implementation strategy)

**Problem Description**:

Every Executor in the code implements `Describe()`, returning formatted strings like:
```
"[Simulation Execute] Wait CKE[3:0001], Sync LocCKE[5:00ff] To rmtCKE[8:00ff]
 Use Channel[0], Set CKE[10:0001], clearType[1]"
```

These strings already contain key instruction parameter information. The current design requires each Executor to add a `CollectTraceDetail()` virtual method and implement it, which is a significant effort (30+ instruction types).

**Fix Suggestion**:

Phased implementation strategy:

**Phase 1 (MVP)**:
- Collect only `Describe()` output + resource delta
- `instrDescribe` field directly uses `executor->Describe()` return value
- Do not implement `CollectTraceDetail()`; detail field is empty or contains only typeName

**Phase 2 (Enhancement)**:
- Implement `CollectTraceDetail()` for high-frequency/critical instruction types (e.g., Trans, Reduce, SyncCke)
- Detail uses flat key-value map (see P1-3)
- Progressively cover all instruction types

This enables rapid MVP delivery, reducing initial development cost.

---

### P2-2: Notify and Other Non-CCU Tasks Not in Trace

**Severity**: P2 (Information gap)

**Problem Description**:

`SequentialExecutor::Execute()` includes task types beyond `CCU_GRAPH`, such as `NOTIFY_RECORD/NOTIFY_WAIT/REDUCE/MEM_CPY/AIV_GRAPH`. `NotifyWait` can also return `HOLD_CMD`, with similar blocking behavior to CCU's waitCKE.

The current trace only covers `CCU_GRAPH` type tasks. But Notify's Record/Wait affects execution flow:
- `NotifyWait` blocking causes the outer loop to skip that stream
- After `NotifyRecord` releases the block, the next round allows that stream's CCU task to continue

If not reflected in trace, the frontend replay will show "CCU instructions suddenly jumping to the next round" with no explanation of what happened in between.

**Fix Suggestion**:

Add simplified records for non-CCU task types in `globalEntries[]`:
```cpp
struct CcuTraceNonCcuEntry {
    uint32_t globalSeqId;
    uint32_t execRound;
    int32_t rankId;
    HccLTaskMetaType taskType;    // NOTIFY_RECORD, NOTIFY_WAIT, REDUCE, MEM_CPY...
    HcclVmResult execResult;      // SUCCESS or HOLD_CMD
    std::string description;      // Task description (for frontend display)
};
```

During CCU switch notifications, the frontend can display: "Round 1: Rank1 NotifyWait blocked, skipped".

---

### P2-3: JSON Schema Metadata Model Inconsistency

**Severity**: P2 (Model consistency)

**Problem Description**:

The JSON Schema example in Section 3.4 still includes `rankId`/`dieId` fields in `metadata` (per-CCU model), but Section 3.3.1's `CcuRunMetadata` has already been changed to `rankSize`/`diePerRank` (global model). The two are inconsistent.

Additionally, the JSON Schema example uses `"seqId"` instead of the new design's `"globalSeqId"`, and is missing `"execRound"`/`"sqeTaskId"`/`"ccuRegistry"`/`"sqeTaskRegistry"` and other new fields.

**Fix Suggestion**:

Update the JSON Schema example to be consistent with the data structure definitions in 3.3.1-3.3.2:
- Remove `metadata.rankId`/`metadata.dieId`
- Add `metadata.rankSize`/`metadata.diePerRank`/`metadata.totalExecRounds`
- Add `ccuRegistry[]` and `sqeTaskRegistry[]`
- Change `seqId` to `globalSeqId` in entries
- Add `execRound`/`sqeTaskId`/`rankId`/`dieId` fields

---

## IV. Summary Table

| Priority | ID | Issue | Type | Fix Summary |
|--------|------|------|------|---------|
| P0 | P0-1 | Delta should not be computed during CKE Wait | Feasibility error | Record only key fields during waitCKE; create full entry only after CKE passes |
| P0 | P0-2 | globalSeqId/execRound maintenance location is wrong | Feasibility error | Global context managed by SequentialExecutor, passed to Simulator via parameters |
| P0 | P0-3 | Full snapshot performance bottleneck | Performance bottleneck | ResourceManager change interception + initial snapshot + periodic checkpoints |
| P1 | P1-1 | Loop offset formula incomplete | Information gap | Add iterStepGSA/curLoopCnt/extendNum/addrExpandCoef |
| P1 | P1-2 | Cross-CCU change collection timing | Collection accuracy | Intercept cross-CCU changes inside ResourceManager |
| P1 | P1-3 | Detail polymorphism over-engineered | Architecture optimization | Simplify to flat key-value map |
| P1 | P1-4 | JSON format unsuitable for large data | Performance optimization | Collection-side msgpack/flatbuffers, frontend WASM parsing |
| P2 | P2-1 | Describe() can serve as MVP source | Implementation strategy | Phase 1: only Describe() + delta |
| P2 | P2-2 | Notify and other non-CCU tasks missing | Information gap | Add simplified records |
| P2 | P2-3 | JSON Schema model inconsistency | Model consistency | Align with 3.3.1-3.3.2 |

---

## V. Fix Priority Recommendations

**Step 1 (Required)**: Fix P0-1, P0-2, P0-3; otherwise the approach cannot be implemented

**Step 2 (Recommended)**: Fix P1-1, P1-2, P1-3; improve approach quality and practicality

**Step 3 (Optional)**: Fix P1-4, P2-1, P2-2, P2-3; refine details and user experience
