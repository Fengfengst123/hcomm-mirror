# CCU Executor Debug Tool Design Document

## 1. Overview

### 1.1 Objective

Build a debug replay tool for the CCU Executor module. By collecting trace data during instruction execution, enable step-by-step replay on a frontend UI, displaying for each instruction:
- Instruction information (type, parameters, owning CCU)
- Debug information during instruction execution (key variable values, execution path)
- CCU resource changes (before/after comparison of XN/GSA/CKE/MS/Channel)

### 1.2 Current Code Analysis

**CCU Executor Module Architecture** (`src/plugin/solver/virtual_runtime/ccu_executor/`):

```
CcuExecutorBase (abstract base class)
├── Parser()          // Parse instruction fields
├── Run()             // Execute instruction logic
├── Process()         // Resource operations (implemented by some instructions)
└── Describe()        // Generate instruction description string

CcuSimulator (execution engine)
├── Execute()         // Main loop: serial instruction execution
├── ExecuteInstr()    // Single instruction execution
├── ExecuteLoop()     // Loop execution
└── ExecuteLoopGroup()// LoopGroup expanded execution

CcuResourceManager (resource management - singleton)
├── XN[]    // General registers (V1:3072, V2:4096)
├── GSA[]   // Address registers (V1:3072, V2:4096)
├── CKE[]   // Synchronization signals   (1024)
├── MS[]    // Memory Slice 4K (1536)
├── Channel[] // Communication channels (128)
└── InstrSpace // Instruction space
```

**Instruction Categories (4 major types)**:

| Type | Subtypes (V1) | Subtypes (V2 new) | Operated Resources |
|------|-----------|---------------|-----------|
| **Load** | LoadSqeArgsToGsa, LoadSqeArgsToXn, LoadImdToGsa, LoadImdToXn, LoadGsaXn, LoadGsaGsa, LoadXX | LoadImdToX, LoadX, StoreX, ClearX, Nop, Load, Store, Add, Sub, Mul, And, Or, Not, Xor, Shl, Shr, Popcnt | XN, GSA |
| **Trans** | LocMem↔LocMem, LocMem↔LocMS, LocMem↔RmtMem, LocMS↔LocMS, LocMS↔RmtMS, LocMS↔RmtMem, RmtMem↔LocMem, RmtMS↔LocMem, RmtMS↔LocMS, SyncCke, SyncGsa, SyncXn | TransMem, SyncXnWt, SyncAt | MS, GSA, XN, Channel, CKE |
| **Control** | Loop, LoopGroup, SetCke, ClearCke, Jump | Wait, Fence | CKE, LoopEngine, XN |
| **Reduce** | ReduceAdd, ReduceMax, ReduceMin | (same as V1) | MS, CKE |

**Resource Ownership**:
```
Rank[0..N]
  └── CCU[0..1]  (die0, die1)
       ├── XN[0..max]
       ├── GSA[0..max]
       ├── CKE[0..1023]
       ├── MS[0..1535]  (each 4KB)
       ├── Channel[0..127]
       └── InstrSpace (instruction list)
```

---

## 2. Replay Approach Analysis: Log Replay vs Trace Replay

### 2.1 Approach Comparison

| Dimension | Log Replay | Trace Replay (Structured Trace) |
|------|----------------------|-------------------------------|
| **Data Collection** | Intercept `HCCL_VM_DEBUG/INFO/TRACE` log output | Instrument the execution engine to collect structured data |
| **Data Format** | Unstructured text strings | JSON/MsgPack structured data |
| **Information Completeness** | Only contains information printed in logs, prone to omissions | Can collect complete resource snapshots, no omissions |
| **Parsing Complexity** | Requires regex parsing, fragile and high maintenance cost | Direct deserialization, no parsing cost |
| **Storage Size** | Verbose text, large size | Compact binary encoding, supports incremental recording |
| **Replay Precision** | Cannot precisely restore resource state | Can precisely restore full state at any instruction point |
| **Frontend Integration** | Requires additional log parsing service | Frontend directly loads JSON/MsgPack for rendering |
| **Performance Overhead** | Low (text output only) | Medium (requires serialization + snapshots) |
| **Extensibility** | New instructions require regex updates | New instructions only require extending trace structure |
| **Industry Reference** | GDB textual trace | Chrome DevTools Timeline, LLVM Execution Trace, rr debugger |

### 2.2 Conclusion: Adopt Trace Replay Approach

**Rationale**:
1. **Precision**: The core requirement of CCU debugging is observing resource changes; logs cannot guarantee complete collection of XN/GSA/CKE/MS values
2. **Frontend Friendly**: HVRM Insight already has a mature data loading pipeline (Worker + JSON); structured trace data can be directly integrated
3. **Incremental Recording**: Only record changed resource fields (delta) rather than full snapshots, controlling storage size
4. **Step-by-step Replay**: Trace data naturally supports locating by instruction sequence number; log replay is difficult to achieve precise step control

---

## 3. Trace Data Structure Design

### 3.1 Design Principles

Referencing industry practices:
- **Chrome DevTools Trace Event Format**: Layered structure, each event includes category, name, timestamp, args
- **LLVM Execution Trace**: Instruction-level trace + register snapshots
- **rr (Record and Replay) debugger**: Event stream + key checkpoints

Combined with local code characteristics:
- Instructions naturally fall into 4 major categories (Load/Trans/Control/Reduce), each operating different resources
- CCU resources are independent per die; trace needs to associate (rankId, dieId)
- Instructions execute serially; trace is naturally ordered

### 3.2 Overall Architecture

#### 3.2.1 Execution Model Review (Actual Code Logic)

CCU execution is not per-CCU independent, but rather **globally interleaved scheduling**. The core scheduling logic is in `SequentialExecutor::Execute()`:

```
while (HasTask()) {                          // Outer round loop
    for (rank in allRanks) {                 // Iterate all ranks
        for (stream in rankStreams) {        // Iterate each stream of each rank
            while (stream has tasks) {
                ret = ExecuteOneTask(stream.front());
                if (ret == HOLD_CMD) break;  // waitCKE stuck → skip to next stream
                stream.pop();                // task completed → dequeue
            }
        }
    }
}
```

When a CCU's SQE reaches a waitCKE instruction and the condition is not met, `TaskCcuGraph()` returns `HOLD_CMD`,
the SQE task **is not dequeued**, and the outer loop continues to process the next CCU. When another CCU executes SetCke to set the corresponding CKE,
the next outer loop iteration retries the SQE, the CKE condition is met, and execution continues.

Key facts:
- **CCU resources are independent**: Each CCU (rankId, dieId) has its own XN/GSA/CKE/MS/instruction space
- **One CCU, one Simulator**: Multiple SQEs on the same (rankId, dieId) share the same `CcuSimulator` instance (reset instruction pointer via `Init()`)
- **SQE describes execution range**: Each SQE task contains missionId, start instruction ID, instruction count, and input parameter list
- **Global interleaved execution**: The actual execution order is R0:D0 → R0:D1 → R1:D0 → R1:D1 → R0:D0 → ... (all CCUs take turns)

#### 3.2.2 Trace Overall Architecture

Trace must be **global**, recording the interleaved execution sequence of all CCUs, rather than per-CCU independent recording.

```
CcuTraceRun (one complete virtual runtime execution, covering all ranks and CCUs)
├── runMetadata              // Run-level metadata (operator name, rank count, CCU version, etc.)
├── ccuRegistry[]            // CCU registry: all CCUs participating in execution (rankId, dieId)
│   └── CcuIdentity           //   Identity and initial resource snapshot of each CCU
├── sqeTaskRegistry[]        // SQE task registry: all SQE tasks (referenced by entries)
│   └── CcuSqeTask            //   missionId, instruction range, input parameters, owning CCU
│
│  ┌─────────── Static Configuration Layer (per CCU, immutable after init) ───────────┐
├── instrSpaces[]            // Instruction space (per CCU, pure static snapshot)
│   └── CcuInstrSpace         //   rankId + dieId + instructions[instrId, describe]
├── channelSpaces[]          // Channel mapping table (per CCU, pure static snapshot)
│   └── CcuChannelSpace       //   rankId + dieId + channels[channelId, remoteRankId, remoteDieId]
│  └──────────────────────────────────────────────────────────┘
│
│  ┌─────────── Dynamic Execution Layer (runtime changes) ──────────────────┐
├── globalEntries[]          // Global instruction execution sequence (in actual interleaved order)
│   └── CcuTraceEntry         //   Each entry indexes instruction space via (rankId, dieId, instrId)
│       ├── (rankId, dieId,    //   Triple indexes instruction description in instruction space
│       │    instrId)           //
│       ├── sqeRef              // → SQE task in sqeTaskRegistry
│       ├── globalSeqId         //   Global execution sequence number (reflects actual interleaved order)
│       ├── execRound           //   Outer while(HasTask()) round number
│       ├── resourceDelta       //   Resource changes of this CCU (XN/GSA/CKE/MS, excluding Channel)
│       ├── crossCcuChanges     //   Cross-CCU resource changes
│       ├── waitInfo            //   CKE Wait spin information
│       ├── errorInfo           //   Execution failure information
│       └── detail.args         //   Instruction-specific dynamic parameters (runtime values, may differ across rounds for same instruction)
├── ccuFinalSnapshots{}      // Final resource snapshot of each CCU (key: rankId_dieId)
│  └──────────────────────────────────────────────────────────┘
└── runSummary               // Run-level summary statistics
```

**Static/Dynamic Separation Design**:

| Data | Storage Location | Property | Description |
|------|---------|------|------|
| Instruction Description (Describe) | `instrSpaces[]` | **Static** | Each instruction stored only once, does not change with Loop iterations |
| Instruction Category/Name | `instrSpaces[]` | **Static** | Determined by instruction bytes, fixed |
| Channel Mapping Table | `channelSpaces[]` | **Static** | Immutable after InitChannelInfo initialization, stored alongside instrSpaces |
| Runtime Parameters | `detail.args` | **Dynamic** | Actual values after Loop offset, SQE parameter values, etc. |
| Resource Changes | `resourceDelta` | **Dynamic** | Resource before/after for each execution (excluding Channel) |
| Execution Context | `context` | **Dynamic** | Loop round, offset, etc. |

**Same instruction executed multiple times in a Loop**: The instruction space has only 1 record, but trace produces N entries, each with different `detail.args`, `resourceDelta`, `context`.

### 3.3 Data Structure Detailed Definitions

#### 3.3.1 Top Level: CcuTraceRun, CCU Registry, SQE Registry

```cpp
// ===== One complete virtual runtime execution =====
// Covers the global execution process of all ranks, all CCUs, all SQEs
struct CcuTraceRun {
    CcuRunMetadata runMetadata;                        // Run-level metadata
    std::vector<CcuIdentity> ccuRegistry;              // CCU registry
    std::vector<CcuSqeTask> sqeTaskRegistry;           // SQE task registry
    
    // ===== Static Configuration Layer (per CCU, immutable after init) =====
    std::vector<CcuInstrSpace> instrSpaces;            // Instruction space (per CCU, pure static snapshot)
    std::vector<CcuChannelSpace> channelSpaces;        // Channel mapping table (per CCU, pure static snapshot)
    
    // ===== Dynamic Execution Layer (runtime changes) =====
    std::vector<CcuTraceEntry> globalEntries;          // Global instruction execution sequence (interleaved ordered)
    std::vector<CcuTraceNonCcuEntry> nonCcuEntries;    // Non-CCU task records
    std::map<std::string, CcuResourceSnapshot> ccuFinalSnapshots;  // Final snapshot of each CCU (dynamic resources only)
    CcuRunSummary runSummary;                          // Run-level summary statistics
};

// ===== Run-level metadata =====
struct CcuRunMetadata {
    uint32_t traceFormatVersion;                       // Trace format version number (current = 1)
    std::string algorithmName;                         // Algorithm name (e.g., "AllReduce")
    std::string operatorName;                          // Operator name (e.g., "RingAllReduce_Step1")
    uint32_t rankSize;                                 // Total rank count
    uint32_t diePerRank;                               // Die count per rank (fixed = 2)
    RunnerCcuVersion ccuVersion;                       // CCU microcode version
    uint64_t runTimestampNs;                           // Run start timestamp
    uint32_t totalSqeTaskCount;                        // Total SQE task count
    uint32_t totalExecRounds;                          // Total rounds of outer while(HasTask())
};

// ===== CCU Identity (registry entry) =====
// Each participating CCU has exactly one record in the registry
struct CcuIdentity {
    int32_t rankId;                                    // Rank number
    uint16_t dieId;                                    // Die number (0 or 1)
    RunnerCcuVersion ccuVersion;                       // CCU microcode version
    CcuResourceSnapshot initialSnapshot;               // Initial resource snapshot of this CCU
    std::string ccuKey() const {                       // Unique key: "rankId_dieId"
        return std::to_string(rankId) + "_" + std::to_string(dieId);
    }
};

// ===== SQE Task (registry entry) =====
// Each SQE task has exactly one record in the registry, referenced by CcuTraceEntry via sqeTaskId
struct CcuSqeTask {
    uint32_t sqeTaskId;                                // SQE task ID (unique within Run, incrementing from 0)
    int32_t rankId;                                    // Owning rank
    uint16_t dieId;                                    // Owning die
    uint8_t missionId;                                 // SQE mission number (CcuTask.missionId)
    uint16_t instStartId;                              // Start instruction ID (CcuTask.instStartId)
    uint16_t instCnt;                                  // Instruction count (CcuTask.instCnt)
    uint32_t key;                                      // Task key (CcuTask.key)
    std::vector<uint64_t> args;                        // SQE input parameter list (CcuTask.args[], up to 13)
    uint64_t simulatorPtr;                             // CcuSimulator instance pointer (multiple SQEs on same CCU share one)
    uint32_t firstExecRound;                           // First execution round number
};

// ===== Run-level Summary =====
struct CcuRunSummary {
    uint32_t totalInstrExecuted;                       // Total executed instruction count
    uint32_t totalFailedInstr;                         // Failed instruction count
    uint32_t totalCkeWaitSpins;                        // Total CKE wait spin count
    uint32_t totalHoldEvents;                          // Total HOLD event count (interruptions caused by waitCKE)
    std::map<std::string, uint32_t> instrCountByCategory;  // Statistics by instruction category
    std::map<std::string, uint32_t> instrCountByCcu;       // Statistics by CCU (key: "rankId_dieId")
    uint64_t totalMsBytesTransferred;                  // Total MS transfer bytes
};
```

#### 3.3.2 Instruction Space: CcuInstrSpace (per CCU, pure static snapshot)

Instruction space is stored independently per CCU, containing static information of all instructions of that CCU (instruction ID + precomputed Describe() output).
Instruction space is **fully decoupled** from trace dynamic parameters: trace entries index the instruction space only via the `(rankId, dieId, instrId)` triple.

```cpp
// Single instruction in instruction space (pure static)
struct CcuInstrSpaceEntry {
    uint16_t instrId;                        // Instruction ID (index in instruction space)
    std::string instrDescribe;               // Precomputed Describe() output (static, no runtime values)
};

// Instruction space of a single CCU
struct CcuInstrSpace {
    int32_t rankId;                          // Owning rank
    uint16_t dieId;                          // Owning die
    std::vector<CcuInstrSpaceEntry> instructions;  // All instructions of this CCU
};
```

**Collection Timing**: Before trace starts, iterate each CCU's instruction space, call `Parser()` + `Describe()` for all instructions, and cache the results.

**Frontend Usage**: After loading instruction space, the frontend displays the complete instruction list in the left panel (pure static); during replay, highlight the corresponding row via `(rankId, dieId, instrId)`.

#### 3.3.3 Common Layer: CcuTraceEntry

Each entry records **one instruction execution on one CCU**, obtaining the instruction description by indexing instruction space via the `(rankId, dieId, instrId)` triple.

```cpp
// Trace entry for a single instruction (one point on the global timeline)
struct CcuTraceEntry {
    // === Global Positioning ===
    uint32_t globalSeqId;                   // Global execution sequence number (incrementing from 0, reflects actual interleaved order)
    uint32_t execRound;                     // Outer while(HasTask()) round number (incrementing from 0)

    // === CCU Ownership + Instruction Index (triple indexes instruction space) ===
    int32_t rankId;                         // Owning rank
    uint16_t dieId;                         // Owning die
    uint32_t instrId;                       // Instruction ID, together with (rankId, dieId) indexes instrSpaces
    // (rankId, dieId, instrId) can locate the instruction description in instrSpaces

    // === SQE Ownership ===
    uint32_t sqeTaskId;                     // References CcuSqeTask in sqeTaskRegistry

    // === Instruction Category (for statistical aggregation only, 1 byte) ===
    CcuInstrCategory category;              // Instruction category: Load/Trans/Control/Reduce

    // === Execution State ===
    CcuExecState execState;                 // State after execution

    // === Execution Context ===
    CcuExecutionContext context;            // Loop/Jump context information

    // === Resource Changes (Delta) ===
    CcuResourceDelta resourceDelta;         // Resource changes of this CCU caused by this instruction
    CcuCrossCcuChanges crossCcuChanges;     // Cross-CCU resource changes caused by this instruction (if any)

    // === CKE Wait Spin Information ===
    CcuWaitInfo waitInfo;                   // Wait spin merge information (valid only during waitCKE)

    // === Execution Failure Information ===
    CcuErrorInfo errorInfo;                 // Failure information (valid only when execState == EXEC_FAIL)

    // === Instruction-specific Details (polymorphic) ===
    std::unique_ptr<CcuInstrTraceDetail> detail;  // Instruction-specific debug information
};

// Instruction category enum
enum class CcuInstrCategory : uint8_t {
    LOAD = 0,       // Load/store/arithmetic
    TRANS = 1,      // Data transfer
    CONTROL = 2,    // Control flow
    REDUCE = 3,     // Reduction operations
};

// Execution context (Loop/Jump related, excluding SQE info — SQE obtained via sqeTaskId reference)
// GSA address offset calculation formula within Loop (from CcuExecutorBase::UpdateAddress):
//   finalOffset = (loopExtendNum * gsaOffset << addrExpandCoef)
//               + (curLoopCnt * iterStepGSA << addrExpandCoef)
struct CcuExecutionContext {
    bool inLoop;                            // Whether executing within a Loop
    uint16_t loopRound;                     // Current Loop iteration round
    uint16_t loopExtendIndex;               // Loop expand index

    // === Loop offset parameters (valid only when inLoop=true) ===
    // GSA address offset parameters
    uint32_t gsaOffset;                     // GSA base offset coefficient (GetGSAOffset)
    uint64_t iterStepGSA;                   // GSA step per iteration round (GetLoopIterStepGSA)
    uint32_t loopExtendNum;                 // Expand count (GetLoopExtendNum)
    uint32_t curLoopCnt;                    // Current iteration count (GetCurLoopCnt)
    uint64_t gsaAddrOffset;                 // Computed final GSA address offset value
    uint16_t addrExpandCoef;                // Address expansion coefficient (2nd parameter of UpdateAddress)

    // Resource ID offset parameters
    uint16_t msOffset;                      // MS ID offset (GetLoopMsOffset)
    uint16_t ckeOffset;                     // CKE ID offset (GetLoopCKEOffset)
    uint16_t xnIdOffset;                    // XN ID offset (GetLoopXnIdOffset)
};

// CKE Wait spin merge information
// When an instruction spins waiting because CKE condition is not met, multiple ExecuteInstr calls
// are merged into one CcuTraceEntry, rather than producing multiple duplicate records.
struct CcuWaitInfo {
    bool hadWait;                           // Whether CKE wait occurred
    uint32_t waitRetryCount;                // Number of spin wait retries
    uint16_t waitCKEId;                     // CKE ID being waited on
    uint16_t waitCKEMask;                   // CKE mask being waited on
    uint16_t ckeValueOnFirstCheck;          // CKE value on first check
    uint16_t ckeValueOnPass;                // CKE value when finally passed
    uint64_t waitDurationNs;                // Total wait duration (nanoseconds)
};

// Execution failure information
struct CcuErrorInfo {
    bool hasError;                          // Whether an error occurred
    std::string errorMessage;              // Error description (from HCCL_VM_ERROR log)
    std::string failPhase;                  // Failure phase: "Parser" / "Run" / "Process"
    CcuResourceSnapshot failSnapshot;       // Full resource snapshot at failure (for post-mortem analysis)
};

// Cross-CCU resource changes
// When an instruction operates on remote CCU resources (e.g., SetRmtCKE, TransLocMSToRmtMS),
// record the impact on the remote CCU.
struct CcuCrossCcuChanges {
    bool hasCrossCcuChange;                 // Whether cross-CCU changes exist
    std::vector<CcuRemoteCkeChange> remoteCkeChanges;   // Remote CKE changes
    std::vector<CcuRemoteMsChange> remoteMsChanges;     // Remote MS changes
    std::vector<CcuRemoteMemChange> remoteMemChanges;   // Remote memory changes
};

struct CcuRemoteCkeChange {
    int32_t remoteRankId;
    uint16_t remoteDieId;
    uint16_t ckeId;
    uint16_t valueBefore;
    uint16_t valueAfter;
};

struct CcuRemoteMsChange {
    int32_t remoteRankId;
    uint16_t remoteDieId;
    uint16_t msId;
    uint64_t offset;
    uint32_t length;
    std::vector<uint8_t> dataAfter;         // Only record after (remote before covered by remote trace)
};

struct CcuRemoteMemChange {
    int32_t remoteRankId;
    uint64_t remoteAddr;
    uint64_t length;
    std::vector<uint8_t> dataAfter;
};
```

#### 3.3.4 Resource Layer: CcuChannelSpace & CcuResourceSnapshot & CcuResourceDelta

```cpp
// ===== Channel Space (per CCU, pure static snapshot) =====
// Channel mapping table, like instruction space, is a CCU-specific static resource:
//   - Immutable after InitChannelInfo() initialization
//   - Only read during instruction execution (via GetRmtCcu()), never modified
//   - Stored separately from dynamic resources like XN/GSA/CKE/MS
struct CcuChannelSpace {
    int32_t rankId;
    uint16_t dieId;
    std::vector<CcuChannelRecord> channels;  // channelId → (remoteRankId, remoteDieId)
};

// CCU resource snapshot (dynamic resources only: XN/GSA/CKE/MS)
// Note: Channel has been moved to CcuChannelSpace (static configuration, alongside CcuInstrSpace)
struct CcuResourceSnapshot {
    std::vector<CcuXnRecord> xnRecords;      // XN register snapshot
    std::vector<CcuGsaRecord> gsaRecords;    // GSA register snapshot
    std::vector<CcuCkeRecord> ckeRecords;    // CKE signal snapshot
    std::vector<CcuMsRecord> msRecords;      // MS memory slice snapshot (optional, large size)
};

// Resource change delta (records only changed portions)
// Note: Channel table is static configuration and does not change, so no channelChanges included
struct CcuResourceDelta {
    std::vector<CcuXnChange> xnChanges;      // XN change list
    std::vector<CcuGsaChange> gsaChanges;    // GSA change list
    std::vector<CcuCkeChange> ckeChanges;    // CKE change list
    std::vector<CcuMsChange> msChanges;      // MS change list
};

// ========== Resource Records (for snapshots) ==========

struct CcuXnRecord {
    uint16_t id;
    uint64_t value;
};

struct CcuGsaRecord {
    uint16_t id;
    uint64_t value;
};

struct CcuCkeRecord {
    uint16_t id;
    uint16_t value;
};

struct CcuMsRecord {
    uint16_t id;
    std::vector<uint8_t> data;  // 4KB data
};

struct CcuChannelRecord {
    uint16_t channelId;
    int32_t remoteRankId;
    uint16_t remoteDieId;
};

// ========== Resource Changes (for Delta) ==========

struct CcuXnChange {
    uint16_t id;
    uint64_t valueBefore;
    uint64_t valueAfter;
};

struct CcuGsaChange {
    uint16_t id;
    uint64_t valueBefore;
    uint64_t valueAfter;
};

struct CcuCkeChange {
    uint16_t id;
    uint16_t valueBefore;
    uint16_t valueAfter;
};

struct CcuMsChange {
    uint16_t id;
    uint64_t offset;              // Change start offset
    uint32_t length;              // Change length
    std::vector<uint8_t> dataBefore;  // Data before change
    std::vector<uint8_t> dataAfter;   // Data after change
};

// Note: Channel table is static configuration and does not change, so no CcuChannelChange structure
```

#### 3.3.5 Instruction-specific Layer: CcuInstrTraceDetail (runtime dynamic parameters only)

**Runtime dynamic parameters** specific to each instruction type. Since the code has 30+ instruction types (~20 in V1 + ~10+ in V2),
and different instructions have significantly different parameters, a **flat key-value map** is used instead of polymorphic inheritance to reduce development and maintenance costs.

**Key Distinction**: `CcuInstrTraceDetail.args` **stores only runtime dynamic parameters**, i.e., values that may differ for the same instruction across different Loop rounds or different SQE tasks.
Static information (such as instruction description, fixed parameter IDs) is stored in instruction space and not duplicated here.

```cpp
// Instruction-specific trace detail (runtime dynamic parameters only)
struct CcuInstrTraceDetail {
    std::string typeName;                     // Instruction type name (e.g., "LoadSqeArgsToXn")

    // Runtime dynamic parameters (key-value format)
    // Records only values that change with Loop iterations or SQE tasks
    // Examples:
    //   LoadSqeArgsToXn:
    //     {"sqeArgValue": "0x1000"}           // SQE parameter value may differ across tasks
    //
    //   TransLocMSToLocMem (within Loop):
    //     {"resolvedLocMSId": "15",           // Actual MS ID after offset
    //      "resolvedLocGSAId": "103",         // Actual GSA ID after offset
    //      "resolvedLocXnId": "52"}           // Actual XN ID after offset
    //
    //   ReduceAdd:
    //     {"reduceOp": "Add", "dataType": "FP32"}
    std::map<std::string, std::string> args;
    // Note: does not include describeText (instruction description stored in instruction space, indexed by instrId)
};
```

**Static vs Dynamic Division**:
| Information Type | Storage Location | Property | Description |
|----------|---------|------|------|
| Instruction Description | `instrSpaces[].instructions[].instrDescribe` | **Static** | Precomputed Describe() output |
| Instruction Category/Name | `instrSpaces[].instructions[]` | **Static** | Determined by instruction bytes |
| CKE Wait Information | `CcuTraceEntry.waitInfo` | **Dynamic** | waitCKEId/mask, spin count, etc. |
| CKE Resource Changes | `CcuResourceDelta.ckeChanges` | **Dynamic** | CKE before/after values |
| XN/GSA/MS Changes | `CcuResourceDelta.xnChanges`, etc. | **Dynamic** | Resource before/after values |
| Loop Offset Context | `CcuExecutionContext` | **Dynamic** | loopRound, offset, etc. |
| Instruction-specific Dynamic Parameters | `CcuInstrTraceDetail.args` | **Dynamic** | Runtime resolved values (after offset, SQE parameters, etc.) |

### 3.4 Data Serialization Format

**JSON** is adopted as the serialization format (consistent with the Insight tool's existing data format), with future upgrades to MsgPack/FlatBuffers possible to reduce size.

JSON Schema overview:
```json
{
    "runMetadata": {
        "traceFormatVersion": 1,
        "algorithmName": "AllReduce",
        "operatorName": "RingAllReduce_Step1",
        "rankSize": 2,
        "diePerRank": 2,
        "ccuVersion": "CCU_V2",
        "runTimestampNs": 1717000000000000000,
        "totalSqeTaskCount": 4,
        "totalExecRounds": 3
    },
    "ccuRegistry": [
        {
            "rankId": 0, "dieId": 0, "ccuVersion": "CCU_V2",
            "initialSnapshot": {
                "xnRecords": [{"id": 0, "value": "0x0"}, ...],
                "gsaRecords": [{"id": 0, "value": "0x7f0000"}, ...],
                "ckeRecords": [{"id": 0, "value": 0}, ...]
            }
        },
        { "rankId": 0, "dieId": 1, ... },
        { "rankId": 1, "dieId": 0, ... },
        { "rankId": 1, "dieId": 1, ... }
    ],
    "sqeTaskRegistry": [
        {
            "sqeTaskId": 0, "rankId": 0, "dieId": 0,
            "missionId": 3, "instStartId": 0, "instCnt": 50, "key": 1,
            "args": ["0x1000", "0x2000", ...],
            "simulatorPtr": "0x7f3a0000", "firstExecRound": 0
        },
        { "sqeTaskId": 1, "rankId": 0, "dieId": 1, ... }
    ],
    "instrSpaces": [
        {
            "rankId": 0, "dieId": 0,
            "instructions": [
                {"instrId": 0, "instrDescribe": "[Simulation Execute] locCcu[0:0], Load SqeArg[0] to Xn[5]"},
                {"instrId": 1, "instrDescribe": "ParseTransLocMemToLocMemInstr Wait CKE[0:0000]..."},
                {"instrId": 2, "instrDescribe": "[Simulation Execute] Wait CKE[0:0000], Set CKE[3:0001]..."},
                ...
            ]
        },
        { "rankId": 0, "dieId": 1, "instructions": [...] },
        { "rankId": 1, "dieId": 0, "instructions": [...] },
        { "rankId": 1, "dieId": 1, "instructions": [...] }
    ],
    "channelSpaces": [
        {
            "rankId": 0, "dieId": 0,
            "channels": [
                {"channelId": 0, "remoteRankId": 1, "remoteDieId": 0},
                {"channelId": 1, "remoteRankId": 2, "remoteDieId": 1},
                ...
            ]
        },
        { "rankId": 0, "dieId": 1, "channels": [...] },
        { "rankId": 1, "dieId": 0, "channels": [...] },
        { "rankId": 1, "dieId": 1, "channels": [...] }
    ],
    "globalEntries": [
        {
            "globalSeqId": 0,
            "execRound": 0,
            "rankId": 0,
            "dieId": 0,
            "instrId": 0,
            "sqeTaskId": 0,
            "category": "Load",
            "execState": "EXEC_NORMAL_INSTR",
            "context": {
                "inLoop": false,
                "loopRound": 0,
                "loopExtendIndex": 0,
                "gsaOffset": 0,
                "iterStepGSA": 0,
                "loopExtendNum": 0,
                "curLoopCnt": 0,
                "gsaAddrOffset": 0,
                "addrExpandCoef": 0,
                "msOffset": 0,
                "ckeOffset": 0,
                "xnIdOffset": 0
            },
            "resourceDelta": {
                "xnChanges": [
                    {"id": 5, "valueBefore": "0x0", "valueAfter": "0x1000"}
                ],
                "gsaChanges": [],
                "ckeChanges": [],
                "msChanges": []
            },
            "detail": {
                "typeName": "LoadSqeArgsToXn",
                "args": {"sqeArgValue": "0x1000"}
            }
        },
        {
            "globalSeqId": 1,
            "execRound": 0,
            "rankId": 0,
            "dieId": 1,
            "instrId": 0,
            ...
        }
    ],
    "ccuFinalSnapshots": {
        "0_0": { ... },
        "0_1": { ... }
    },
    "runSummary": {
        "totalInstrExecuted": 1280,
        "totalFailedInstr": 0,
        "totalCkeWaitSpins": 42,
        "totalHoldEvents": 3,
        "instrCountByCategory": {"Load": 400, "Trans": 500, "Control": 300, "Reduce": 80},
        "instrCountByCcu": {"0_0": 320, "0_1": 320, "1_0": 320, "1_1": 320}
    }
}
```

---

## 4. Collection Instrumentation Point Design

### 4.1 Instrumentation Location

In `CcuSimulator::ExecuteInstr()`, one collection point is inserted before and after instruction execution. **Key: CKE Wait spin merging**.

When the CKE condition is not met in `WaitCkeProcess`, `waitCKE_=true` is set causing `ExecuteInstr` to return false, and the upper-level `Execute()` re-invokes the same instruction. Without merging, a single Wait instruction could produce hundreds of duplicate trace entries.

**Solution**: The collector detects consecutive repeated executions of the same instruction (CKE Wait spin) and merges them into one trace entry, recording the spin count.

```cpp
bool CcuSimulator::ExecuteInstr(uint16_t curInstrId)
{
    auto &ccuResMgr = CcuResourceManager::GetInstance();
    auto instrData = ccuResMgr.GetInstrData(rankId_, dieId_);

    // ① Capture pre-execution snapshot
    auto snapshotBefore = CcuTraceCollector::CaptureResourceSnapshot(rankId_, dieId_);

    auto executor = CcuExecutorFactory::MakeCcuExecutorInstance(...);
    executor->Parser();
    executor->Run();

    // ② CKE Wait spin detection and merging
    if (waitCKE_) {
        // This instruction returned false due to CKE wait; collector records one spin
        CcuTraceCollector::RecordWaitSpin(rankId_, dieId_, curInstrId,
                                           snapshotBefore);
        // Do not record trace entry; wait until CKE is satisfied then record final merged result
        return false;
    }

    // ③ CKE passed (or instruction does not involve CKE wait); capture post-execution snapshot
    auto snapshotAfter = CcuTraceCollector::CaptureResourceSnapshot(rankId_, dieId_);
    auto detail = executor->CollectTraceDetail();

    // ④ Get cross-CCU changes (intercepted and recorded in real-time by CcuResourceManager during instruction execution, see Section 4.4)
    auto crossChanges = CcuTraceCollector::ConsumeCrossCcuChanges(rankId_, dieId_);

    // ⑤ Merge CKE Wait spin information (if spins occurred previously); record final trace entry
    auto waitInfo = CcuTraceCollector::FinalizeWaitInfo(rankId_, dieId_, curInstrId,
                                                         snapshotBefore);

    // ⑥ Detect execution failure
    auto errorInfo = CcuTraceCollector::CaptureErrorInfo(rankId_, dieId_, executor.get());

    // ⑦ Compute delta and record trace entry (including wait, error, cross-ccu information)
    // globalSeqId, execRound, sqeTaskId are passed from the SequentialExecutor layer to CcuTraceCollector (see Section 4.3)
    auto globalCtx = CcuTraceCollector::GetCurrentGlobalContext();
    CcuTraceCollector::RecordEntry(rankId_, dieId_, curInstrId,
                                    globalCtx.globalSeqId,
                                    globalCtx.execRound,
                                    globalCtx.currentSqeTaskId,
                                    snapshotBefore, snapshotAfter,
                                    detail, waitInfo, errorInfo, crossChanges);

    UpdateLoopStatus();
    return true;
}
```

### 4.2 CcuExecutorBase Extension

Add a `CollectTraceDetail()` virtual method in the base class, implemented by subclasses as needed:

```cpp
class CcuExecutorBase {
public:
    // New: Collect instruction-specific trace detail (flat key-value structure)
    // Default implementation: includes only typeName and Describe() output
    virtual CcuInstrTraceDetail CollectTraceDetail() {
        CcuInstrTraceDetail detail;
        detail.typeName = "Unknown";
        detail.describeText = Describe();
        return detail;
    }
};
```

Example implementations for Executor subclasses (populating instruction-specific parameters into the args map):

```cpp
// TransLocMemToLocMemExecutor
CcuInstrTraceDetail TransLocMemToLocMemExecutor::CollectTraceDetail() {
    CcuInstrTraceDetail detail;
    detail.typeName = "TransLocMemToLocMem";
    detail.describeText = Describe();
    detail.args = {
        {"srcAddr", std::to_string(srcLocAddr_)},
        {"dstAddr", std::to_string(dstLocAddr_)},
        {"transLength", std::to_string(transLength_)},
    };
    // Note: CKE synchronization info already recorded in CcuWaitInfo and CcuResourceDelta, no need to duplicate
    return detail;
}

// ReduceAddExecutor
CcuInstrTraceDetail ReduceAddExecutor::CollectTraceDetail() {
    CcuInstrTraceDetail detail;
    detail.typeName = "ReduceAdd";
    detail.describeText = Describe();
    detail.args = {
        {"count", std::to_string(count_)},
        {"castEn", std::to_string(castEn_)},
        {"dataType", std::to_string(dataType_)},
        {"msList", ParseMSList()},
    };
    return detail;
}
```

### 4.3 Global Context Management

Global context information such as `globalSeqId`, `execRound`, `currentSqeTaskId` can only be obtained at the `SequentialExecutor` layer and needs to be passed down to `CcuTraceCollector`.

```cpp
// SequentialExecutor::Execute() manages global context
while (HasTask()) {
    CcuTraceCollector::BeginRound(execRound);    // Set current round

    for (rankTasks : allRankTaskQueues_) {
        for (streamTasks : rankTasks) {
            while (!streamTasks.empty()) {
                auto task = streamTasks.front();
                if (task.taskType == CCU_GRAPH) {
                    // Register SQE task and get sqeTaskId
                    uint32_t sqeTaskId = CcuTraceCollector::RegisterSqeTask(
                        task.rankId, task.dieId, task.missionId,
                        task.instStartId, task.instCnt, task.args);
                    CcuTraceCollector::SetCurrentSqeTaskId(sqeTaskId);
                }
                CcuTraceCollector::BeginGlobalStep();  // Atomically increment globalSeqId
                auto ret = ExecuteOneTask(task);
                if (ret == HOLD_CMD) break;
                streamTasks.pop();
            }
        }
    }
    execRound++;
}
```

`CcuSimulator::ExecuteInstr()` reads these values via `CcuTraceCollector::GetCurrentGlobalContext()`.

### 4.4 Cross-CCU Change Collection (ResourceManager Interception)

Cross-CCU changes (e.g., `SetRmtCKESignal`, `TransLocMSToRmtMS`, etc.) occur in the Executor's `Process()`, operating on remote CCU resources. Compared to comparing remote CCU state after the fact, a more reliable approach is to intercept directly in `CcuResourceManager`'s Update methods:

```cpp
// CcuResourceManager::UpdateCkeValue() intercepts cross-CCU changes
void CcuResourceManager::UpdateCkeValue(int rankId, int dieId, uint16_t ckeId, uint16_t value) {
    uint16_t oldValue = GetCkeValue(rankId, dieId, ckeId);

    // Detect whether this is a cross-CCU operation
    auto [execRank, execDie] = CcuTraceCollector::GetCurrentExecutingCcu();
    if (rankId != execRank || dieId != execDie) {
        CcuTraceCollector::RecordCrossCcuCkeChange(
            rankId, dieId, ckeId, oldValue, value, execRank, execDie);
    }

    // Perform actual update
    ckeData_[rankId][dieId][ckeId] = value;
}
```

Similarly applicable to `UpdateXnValue()`, `UpdateGsaValue()`, `TransMSToMS()`, and other methods involving remote CCUs.

Intercepted cross-CCU changes are buffered in `CcuTraceCollector` and consumed by `CcuSimulator::ExecuteInstr()` via `ConsumeCrossCcuChanges()` when recording the trace entry.

### 4.5 Resource Snapshot Collection

```cpp
class CcuTraceCollector {
public:
    static CcuResourceSnapshot CaptureResourceSnapshot(int rankId, int dieId);
    static CcuResourceDelta ComputeDelta(const CcuResourceSnapshot& before,
                                          const CcuResourceSnapshot& after);
    static void RecordEntry(int rankId, int dieId, uint16_t instrId,
                            const CcuResourceSnapshot& before,
                            const CcuResourceSnapshot& after,
                            std::unique_ptr<CcuInstrTraceDetail> detail);
    static void StartSession(int rankId, int dieId, RunnerCcuVersion version);
    static void EndSession(int rankId, int dieId);
    static void DumpToFile(const std::string& outputPath);

    // Global context management (called by SequentialExecutor layer)
    static void BeginRound(uint32_t execRound);
    static void BeginGlobalStep();
    static uint32_t RegisterSqeTask(int rankId, int dieId, uint8_t missionId,
                                     uint16_t instStartId, uint16_t instCnt,
                                     const std::vector<uint64_t>& args);
    static void SetCurrentSqeTaskId(uint32_t sqeTaskId);
    static CcuGlobalContext GetCurrentGlobalContext();

    // Cross-CCU change interception (called by CcuResourceManager)
    static void RecordCrossCcuCkeChange(int rankId, int dieId, uint16_t ckeId,
                                         uint16_t oldValue, uint16_t newValue,
                                         int execRank, int execDie);
    static std::pair<int,int> GetCurrentExecutingCcu();
    static CcuCrossCcuChanges ConsumeCrossCcuChanges(int rankId, int dieId);
};
```

### 4.6 Performance Optimization Strategies

#### 4.6.1 Collection-side Optimization

1. **On-demand Enablement**: Controlled via environment variable `HCCLVM_ENABLE_CCU_TRACE=1`
2. **Incremental Recording**: Only record resource fields with non-zero changes (delta); do not store unchanged resources
3. **MS Data Tiered Collection** (controlled via `HCCLVM_CCU_TRACE_MS_LEVEL`):
   - `none` (default): No MS data collection; only record MS ID and length
   - `digest`: Collect hash digest of MS data (16 bytes) for integrity verification
   - `range`: Only record modified offset ranges (offset + length + before/after)
   - `full`: Record full 4KB (only when data comparison is needed)
4. **Deferred Writing**: Buffer in memory first, batch write to file after execution completes
5. **Optional Full Snapshots**: Enable full snapshot mode via `HCCLVM_CCU_TRACE_FULL_SNAPSHOT=1`

#### 4.6.2 Loop Folding and Sampling

Loop execution can cause trace data explosion (one Loop with 1000 rounds x N instructions = thousands of entries). Adopt folding + sampling strategies to control data volume:

```
Environment variable control:
  HCCLVM_CCU_TRACE_LOOP_MODE=fold|sample|full

fold mode (default):
  Fold multiple iterations of the same Loop into one summary record:
  ┌─────────────────────────────────────────────────┐
  │ CcuLoopFoldEntry                                │
  │ ├── loopInstrId: 5                              │
  │ ├── totalRounds: 1000                           │
  │ ├── firstRoundEntries: [...]   // Full trace of first round │
  │ ├── lastRoundEntries: [...]    // Full trace of last round  │
  │ ├── resourceStats:                              │
  │ │   ├── xnWriteCount: {5: 1000, 8: 500}         │
  │ │   ├── ckeToggleCount: {3: 2000}               │
  │ │   └── avgMsBytesPerRound: 4096                │
  │ └── anomalyRounds: [42, 87]  // Anomalous round numbers │
  └─────────────────────────────────────────────────┘
  Frontend displays folded view by default; click "Expand" to view details of any round.

sample mode:
  Record only first round + last round + sampling every N rounds (N controlled by HCCLVM_CCU_TRACE_SAMPLE_RATE, default 10)

full mode:
  Record all rounds (only for small-scale debugging; large data volume)
```

Anomaly round detection rules:
- Instruction count in this round differs from the first round
- Resource change pattern in this round differs significantly from the first round (delta field set differs)
- This round contains EXEC_FAIL
- CKE Wait count in this round is abnormal (more than 2x the first round)

#### 4.6.3 Frontend Rendering Optimization

When trace entry count exceeds 10,000, the frontend requires special handling:

1. **Virtual Scrolling**: Instruction list uses virtual scroll (e.g., `vue-virtual-scroller`), rendering only visible rows
2. **Lazy Loading**: Initially load only entry summary information (seqId/instrId/name/category); load full detail only on click
3. **Paginated Loading**: Large datasets loaded in chunks of 1000 entries per page; frontend paginates browsing
4. **Resource Heatmap Downsampling**: In the resource overview panel, when resource count is excessive, aggregate display by interval grouping

### 4.7 Non-CCU Task Trace Recording

`SequentialExecutor::Execute()` includes task types beyond `CCU_GRAPH`, such as `NOTIFY_RECORD/NOTIFY_WAIT/REDUCE/MEM_CPY/AIV_GRAPH`. `NotifyWait` can also return `HOLD_CMD`, affecting global scheduling flow. If not recorded in trace, the frontend replay will show "CCU instructions suddenly jumping to the next round".

Add simplified records for non-CCU tasks in `globalEntries[]`:

```cpp
struct CcuTraceNonCcuEntry {
    uint32_t globalSeqId;                   // Global execution sequence number
    uint32_t execRound;                     // Outer scheduling round
    int32_t rankId;                         // Owning rank
    HccLTaskMetaType taskType;              // Task type: NOTIFY_RECORD/NOTIFY_WAIT/REDUCE/MEM_CPY etc.
    HcclVmResult execResult;                // Execution result: SUCCESS or HOLD_CMD
    std::string description;                // Task description (for frontend display)
};
```

During replay, the frontend can display scheduling events such as "Round 1: Rank1 NotifyWait blocked → skipped", helping users understand CCU switching reasons.

### 4.8 MVP Phased Implementation Strategy

**Phase 1 (MVP)**:

Minimal viable trace collection to quickly validate the frontend replay workflow:

1. Collect `Describe()` output + resource delta (via ResourceManager change interception)
2. `CcuInstrTraceDetail`'s `args` is empty; `describeText` directly uses the return value of `executor->Describe()`
3. Do not implement custom `CollectTraceDetail()` for each Executor; use base class default implementation
4. Frontend displays only instruction list panel + resource change panel + instruction description panel
5. No support for Loop folding/sampling, cross-CCU changes, non-CCU task recording

**Phase 2 (Enhancement)**:

1. Implement `CollectTraceDetail()` for high-frequency/critical instruction types (e.g., Trans, Reduce, SyncCke)
2. Frontend displays complete key-value parameter tables
3. Support Loop folding/sampling
4. Support cross-CCU change recording and non-CCU task recording

**Phase 3 (Completion)**:

1. Progressively cover all 30+ instruction types' `CollectTraceDetail()`
2. Support search/filter, data dependency tracing, Trace Diff comparison
3. Support breakpoint functionality
4. Optional upgrade to msgpack/flatbuffers binary format

---

## 5. Frontend Integration Plan

### 5.1 Existing HVRM Insight Architecture

```
App.vue
├── AppTopNav.vue          // Top navigation bar
├── DashboardPage.vue      // Dashboard page
├── MemViewPage.vue        // Correlation analysis page
└── AnalyticPage.vue       // Error diagnostics page
```

Tech stack: Vue 3 + Element Plus + Vite

### 5.2 New Tab: CCU Trace Replay

Add a fourth tab in `App.vue`:

```javascript
const pages = [
  { id: 'dashboard', label: '总览' },
  { id: 'mem-view', label: '关联' },
  { id: 'analytic', label: '报错' },
  { id: 'ccu-trace', label: 'CCU调试' },  // New
]
```

### 5.3 CCU Trace Replay Page Design

#### Overall Layout

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  CCU Trace Replay                                                            │
├────────────┬─────────────────────────────────────────────────────────────────┤
│            │  ┌─ Trace Progress Bar ────────────────────────────────────────┐  │
│  Left      │  │  Trace: Rank [0 ▼]  Die [0 ▼]    #4/1280  Round 0         │  │
│  Panel     │  │  ──────●──────○────────○────────○────────○────────  0.3%   │  │
│            │  └───────────────────────────────────────────────────────────┘  │
│  ────────  │  ┌───────────────────────────────────────────────────────────┐  │
│            │  │         Instruction List Panel  (Rank0:Die0 Instr Space)    │  │
│  Playback  │  │  BP | instrId | Instruction Description (Describe) | Exec State │
│  Controls  │  │   ○ | 0       | [Load] LoadImd to Xn[5]...  | ✓ #0 R0     │  │
│            │  │   ● | 1       | [Trans] LocMem→LocMem...    | ✓ #4 R0     │  │
│  [⏮][⏪]   │  │   ○ | 2       | [SetCke] Set CKE[3:0001]    | ✓ #7 R0     │  │
│  [▶/⏸]    │  │ ▶ ○ | 3       | [WaitCKE] Wait CKE[0:0000]..| ← Current #12 │ │
│  [⏩][⏭]   │  │   ○ | 4       | [Trans] LocMem→RmtMem...    |             │  │
│  [⇥ Next BP]│  │   ○ | 5       | [Reduce] Add MS[0:1,1:2,2:3]|             │  │
│            │  │   ...                                                     │  │
│  ────────  │  │  ▶ = Current replay position  ● = Breakpoint  ○ = No BP    │  │
│            │  │  Exec State: ✓=Executed ←current=Current step HOLD=CKE blocked FAIL=Failed │
│  Current   │  └───────────────────────────────────────────────────────────┘  │
│  CCU:      │  ┌─────────────────────────┬─────────────────────────────────┐  │
│  Rank0:Die0│  │   Resource Change Panel   │   Instruction Detail Panel      │  │
│            │  │                         │                                 │  │
│  ────────  │  │  XN Changes             │  Instruction: WaitCKE           │  │
│            │  │  Xn[5]: 0→0x1000        │  CCU: Rank0:Die0  Round: 0       │  │
│  SQE Task  │  │  Xn[8]: 0x400→0         │                                 │  │
│  (current  │  │                         │  Execution Context:              │  │
│  instr     │  │  GSA Changes            │  Loop Round: 0                   │  │
│  belongs   │  │  (none)                 │  Extend Index: 0                 │  │
│  to)       │  │                         │  GSA Offset: 0x4000              │  │
│            │  │  CKE Changes            │                                 │  │
│  mission: 3│  │  CKE[3]: 0→1            │  CKE Wait Info:                 │  │
│  sim*:     │  │                         │  waitCKE: CKE[0]&0x1 expect=1   │  │
│  0x7f3a..  │  │  MS Changes             │  Wait spins: 5 rounds (first CKE=0) │
│  args[0..12]│ │  MS[0]: @0x0 +4KB       │  Final pass: CKE=1              │  │
│  0x1000..  │  │                         │                                 │  │
│            │  └─────────────────────────┴─────────────────────────────────┘  │
│  ────────  │  ┌───────────────────────────────────────────────────────────┐  │
│            │  │              CCU Resource Overview Panel                    │  │
│  Exec      │  │  XN[0..31]  | GSA[0..31] | CKE[0..15] | MS[0..7]       │  │
│  Summary   │  │  (Shows accumulated resource state of current CCU Rank0:Die0) │
│            │  └───────────────────────────────────────────────────────────┘  │
│  3 instrs  │  ┌───────────────────────────────────────────────────────────┐  │
│  executed  │  │              Breakpoint Hit Bar (shown only when hit)       │  │
│  Round 0   │  │  ⚠ Breakpoint hit: Rank0:Die0 instrId=1 (TransLocMemToLocMem) │
│            │  │  Round 0  missionId=3                                      │
│  ────────  │  │  [Continue ▶]  [Step ⏭]  [Skip to next BP ⇥]              │
│            │  └───────────────────────────────────────────────────────────┘  │
│  Breakpoint│                                                                 │
│  Management│                                                                 │
│  [+ Add]   │                                                                 │
│  [Clear All]│                                                                │
│            │                                                                 │
│  ┌────────┐│                                                                 │
│  │BP List  ││                                                                 │
│  │● R0:D0  ││                                                                 │
│  │  id=1   ││                                                                 │
│  │● R1:D0  ││                                                                 │
│  │  id=5   ││                                                                 │
│  │ [✏][🗑]││                                                                 │
│  └────────┘│                                                                 │
└────────────┴─────────────────────────────────────────────────────────────────┘
```

#### Instruction List Panel Description

The instruction list has 4 columns:

| Column | Content | Description |
|---|------|------|
| **BP** | ○ / ● / ▶ | Breakpoint marker + current replay position (`▶` marks current row) |
| **instrId** | Instruction ID | Index in CCU instruction space |
| **Instruction Description** | `Describe()` output | Contains complete instruction parsing information (parameters, addresses, register IDs, etc.) |
| **Exec State** | Instruction execution state | Shows execution information of this instruction at current replay progress |

**Execution State Column Meanings**:

| Display | Meaning |
|------|------|
| (blank) | Not yet executed |
| `✓ #N RN` | Executed, #N is global sequence number, RN is Round number |
| `← Current #N` | Current replay step is at this instruction |
| `HOLD` | waitCKE blocked (CKE condition not met, waiting), **orange highlight** |
| `FAIL` | Execution failed, **red highlight** |

**Reason for removing Type/SQE columns**:
- Type (Load/Trans/Control/Reduce) is already included in Describe() output; no need for a separate column
- SQE is task-level information, not instruction-level; displayed in the SQE task area of the left panel

#### CCU Instruction Space View and Auto-switching

The instruction list panel always displays **one CCU's instruction space** (determined by the top Rank/Die dropdown). During trace replay, the instruction list automatically switches and positions according to trace execution records:

```
Trace replay advances to next entry
       │
       ▼
  entry's (rankId, dieId) == currently displayed CCU?
       │                              │
       │ Yes (continuous execution     │ No (switch to different CCU)
       │ within same CCU)              │
       ▼                              ▼
  Move ▶ marker to                ① Auto-switch top Rank/Die dropdown
  corresponding row in             ② Instruction list switches to new CCU's instruction space
  current instruction list         ③ Move ▶ marker to corresponding row in new CCU
                                   ④ Status bar shows "CCU switch: R0:D0 → R1:D0"
```

**Current Replay Position Marker**:

`▶` marker is always displayed in the BP column, identifying the row of the current replay step. Overlays with breakpoint markers (e.g., `▶●` indicates the current row is also a breakpoint).

**Breakpoint Setting**:

Users click the BP column circle in the current CCU's instruction list to set breakpoints. Breakpoints are essentially `(rankId, dieId, instrId)` triples, bound to the currently displayed CCU.

#### Core Components

| Component | Function |
|------|------|
| `CcuTraceCcuSelector` | **CCU Selector**: Rank/Die dropdown, selects the CCU instruction space currently displayed |
| `CcuTraceProgressBar` | **Trace Progress Bar**: Shows global progress, current globalSeqId, Round number |
| `CcuTraceInstrList` | **Instruction List Panel**: 4 columns (BP/instrId/Describe/Exec State), auto-highlight/switch during replay |
| `CcuTraceExecSummary` | **Execution Trace Summary**: Left panel shows executed count and Round of current CCU |
| `CcuTraceSqeInfo` | **SQE Task Information**: Left panel shows missionId, simulator pointer, args[0..12] parameter list of the SQE owning the current instruction |
| `CcuTracePlaybackControl` | Playback controller (step forward/backward/auto-play/skip to next breakpoint) |
| `CcuTraceBreakpointPanel` | **Breakpoint Management Panel**: Lists all breakpoints (identified by rankId/dieId/instrId), supports add/delete/edit |
| `CcuTraceBreakpointHitBar` | **Breakpoint Hit Bar**: Pops up when breakpoint is hit, contains CCU/SQE/Round info, provides Continue/Step/Skip |
| `CcuTraceResourceDelta` | Resource change panel, shows delta of current step |
| `CcuTraceInstrDetail` | Instruction detail panel, shows instruction-specific debug information |
| `CcuTraceResourceOverview` | CCU resource overview, shows accumulated resource state of current CCU |

#### Breakpoint Functionality Design (Frontend Replay Breakpoint)

##### Breakpoint Types

| Type | Description | Setting Method |
|------|------|---------|
| **Line Breakpoint** | Pause when reaching specified globalSeqId | Click BP column circle in instruction list |
| **Name Breakpoint** | Pause at all instructions with specified name | Add via breakpoint management panel |
| **Category Breakpoint** | Pause at all instructions of specified category (Load/Trans/Control/Reduce) | Add via breakpoint management panel |
| **Resource Change Breakpoint** | Pause when specified resource is modified (e.g., "break when Xn[5] is written") | Add via breakpoint management panel |
| **Round Breakpoint** | Pause when outer scheduling enters specified Round (e.g., "first instruction of Round 3") | Add via breakpoint management panel |

##### Breakpoint States

- **○** No breakpoint (default)
- **●** Breakpoint enabled
- **⊘** Breakpoint disabled (configuration retained but not triggered)
- **▶●** Current step hit breakpoint (highlighted + hit bar pops up)

##### Auto-play Breakpoint Behavior

```
User clicks [▶ Auto-play]
       │
       ▼
  Advance step by step ────────► Each step checks: does current entry hit a breakpoint?
       │                              │
       │ No                           │ Yes
       ▼                              ▼
  Continue to next step       Pause playback, highlight current row
                                   │
                           Pop up breakpoint hit bar:
                           "⚠ Breakpoint hit: Rank0:Die0 instrId=1 (instruction name)"
                                   │
                   ┌───────────────┼───────────────┐
                   ▼               ▼               ▼
             [Continue ▶]    [Step ⏭]       [Skip to next BP ⇥]
              Resume auto-play  Step forward    Fast-forward to next breakpoint
                              then pause
```

##### Breakpoint Management Panel Interaction

The breakpoint management area at the bottom of the left panel displays all current breakpoints and supports:
1. **Quick Add**: Click BP column in instruction list, or click `[+ Add]` to open dialog
2. **Enable/Disable**: Click the toggle on each breakpoint entry
3. **Jump to Location**: Click a breakpoint entry; instruction list auto-scrolls to corresponding row
4. **Edit/Delete**: Each breakpoint has `[✏ Edit]` and `[🗑 Delete]` buttons on the right
5. **Batch Operations**: `[Clear All]` removes all breakpoints at once

#### Interaction Logic

1. **CCU Selection** → Select the CCU instruction space to display via top Rank/Die dropdown; instruction list switches to corresponding CCU
2. **Replay Advance** → Click playback control buttons; global advance (globalSeqId increments):
   - If current entry belongs to selected CCU → `▶` marker moves to corresponding instrId row; exec state column updates
   - If current entry belongs to another CCU → Auto-switch CCU selector; instruction list switches to new CCU and marks corresponding row
3. **CCU Switch Notification** → When auto-switching CCU, status bar briefly shows "CCU switch: R0:D0 → R1:D0", helping users understand scheduling jumps
4. **Auto-play Mode** → Advance global sequence at set speed, simulating actual interleaved execution; CCU auto-switches with trace
5. **Round Separation** → When execRound changes, the left execution trace summary area shows Round change (e.g., "Round 0 → Round 1")
6. **HOLD State** → Instructions blocked by waitCKE show orange `HOLD` in exec state column; instruction detail panel shows waiting CKE ID, mask, spin count
7. **Click Instruction Row** → Linked update: resource change panel (showing that instruction's delta), instruction detail panel, CCU resource overview (updated to current CCU state)
8. **Click BP Column Circle** → Toggle breakpoint marker for that row (○ ↔ ●); breakpoint triple `(rankId, dieId, instrId)` auto-binds to current CCU; breakpoint management panel syncs
9. **Auto-play Hits Breakpoint** → Pause and pop up hit bar (with CCU/Round info); options to continue/step/skip to next breakpoint
10. **Click [⇥ Next Breakpoint]** → Fast-forward to next breakpoint position; if breakpoint is on another CCU, auto-switch CCU selector
11. **Execution Trace Summary** → Left panel shows real-time executed instruction count, current Round, current SQE summary for current CCU
12. **SQE Info Linkage** → Instruction detail panel shows missionId, instStartId, instCnt, args list of the SQE owning the current instruction
13. **Resource Overview** → Shows accumulated resource state of current CCU (XN/GSA/CKE/MS), updates in real-time with replay
14. **Search Instructions** → Search in current CCU's instruction list by instruction name, description, instrId
15. **Trace Resource Origin** → Click a change value in resource change panel to auto-highlight the instruction that last modified that resource (traced within same CCU)
16. **Cross-CCU Jump** → Click a remote change in crossCcuChanges to auto-switch CCU selector to corresponding CCU and locate the related instruction
17. **Failure Quick Locate** → Click "Jump to first failure" button to auto-switch to the CCU of the failed instruction, locate the first instruction with execState=EXEC_FAIL, highlighted in red
18. **Manual CCU Switch** → Users can manually switch CCU via dropdown at any time; instruction list switches to corresponding CCU's instruction space; replay position unchanged, but highlights the most recently executed instruction in that CCU

### 5.4 Data Loading

Reuse Insight's existing Worker data loading pipeline:

```javascript
// New data loading utility
// src/utils/ccuTraceData.js
export async function loadCcuTrace(datasetPath, rankId, dieId) {
    const response = await fetch(`/api/ccu-trace/${datasetPath}/rank${rankId}_die${dieId}.json`);
    return await response.json();
}
```

Backend new API endpoint:

```python
# server.py new addition
@app.route('/api/ccu-trace/<dataset>/<file>')
def ccu_trace_data(dataset, file):
    trace_path = os.path.join(DATA_DIR, dataset, 'ccu_trace', file)
    return send_file(trace_path, mimetype='application/json')
```

---

## 6. File Organization

### 6.1 Backend (C++)

```
src/plugin/solver/virtual_runtime/ccu_executor/trace/
├── ccu_trace_types.h          // Trace data structure definitions
├── ccu_trace_collector.h      // Trace collector declarations
├── ccu_trace_collector.cc     // Trace collector implementation
├── ccu_trace_serializer.h     // JSON serialization declarations
└── ccu_trace_serializer.cc    // JSON serialization implementation
```

### 6.2 Frontend (Vue)

```
src/plugin/solver/virtual_runtime/insight/frontendV3/src/
├── pages/
│   └── CcuTracePage.vue              // CCU Trace replay page
├── components/ccu-trace/
│   ├── CcuTraceCcuSelector.vue       // CCU selector (Rank/Die dropdown)
│   ├── CcuTraceProgressBar.vue       // Trace progress bar
│   ├── CcuTraceInstrList.vue         // Instruction list panel (current CCU instruction space)
│   ├── CcuTraceExecSummary.vue       // Execution trace summary (left panel)
│   ├── CcuTracePlaybackControl.vue   // Playback controls
│   ├── CcuTraceSqeInfo.vue           // SQE task information (left panel)
│   ├── CcuTraceResourceDelta.vue     // Resource change panel
│   ├── CcuTraceInstrDetail.vue       // Instruction detail panel
│   ├── CcuTraceResourceOverview.vue  // CCU resource overview (current CCU)
│   ├── CcuTraceBreakpointPanel.vue   // Breakpoint management panel
│   └── CcuTraceBreakpointHitBar.vue  // Breakpoint hit bar
└── utils/
    └── ccuTraceData.js               // Data loading + index building utilities
```

---

## 7. Four-way Relationship Summary

The core of this design is to tightly associate **instructions**, **CCUs**, **SQE tasks**, and **CCU resources** through the trace data structure:

```
┌──────────────────────────────────────────────────────────────────────┐
│                         CcuTraceEntry                                │
│                                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌───────────────┐   │
│  │ Global    │  │Instruction│ │  Resource     │  │ Instruction-  │   │
│  │ Positioning│ │ Info     │  │  Changes     │  │ specific      │   │
│  │          │  │          │  │  (Delta)     │  │ Details       │   │
│  │globalSeq │  │ instrId  │  │ xnChanges[]  │  │ (Detail)      │   │
│  │execRound │  │ category │  │ gsaChanges[] │  │ CcuLoadTrace  │   │
│  │rankId    │  │ name     │  │ ckeChanges[] │  │ CcuTransTrace │   │
│  │dieId     │  │ describe │  │ msChanges[]  │  │ CcuLoopTrace  │   │
│  │sqeTaskId─┤──│          │  │ crossCcuChgs │  │ CcuReduceTrace│   │
│  └──────────┘  └──────────┘  └──────────────┘  │ ...           │   │
│       │                                         └───────────────┘   │
│       ▼                                                              │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  SQE Task (looked up in sqeTaskRegistry)                       │   │
│  │  missionId, instStartId, instCnt, args[], simulatorPtr        │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Execution Context (Loop/Jump state)                           │   │
│  │  inLoop, loopRound, gsaOffset, msOffset, ckeOffset, ...      │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

**Key Design Decisions**:
1. **Global Interleaved Timeline**: Instructions from all CCUs are interleaved in actual execution order in `globalEntries[]`; `globalSeqId` reflects true scheduling order
2. **CCU Registry**: Each CCU (rankId, dieId) has a unique entry in `ccuRegistry` containing initial resource snapshot; entries associate via rankId+dieId
3. **SQE Registry**: Each SQE task has a unique entry in `sqeTaskRegistry`; entries reference via `sqeTaskId`, avoiding redundancy
4. **Delta Instead of Full Snapshots**: Each trace records only changed portions; initial and final snapshots used to restore full state
5. **Layered Polymorphism**: Common layer handles information shared by all instructions; instruction-specific layer supports each instruction's unique details through polymorphism
6. **Scheduling Rounds**: `execRound` records the outer `while(HasTask())` round number, helping understand interleaved scheduling timing

---

## 8. Breakpoint Functionality Design

### 8.1 Feasibility Analysis

Breakpoint functionality is fully feasible and can be implemented at two independent levels:

| Level | Name | Trigger Timing | Complexity | Purpose |
|------|------|---------|--------|------|
| **Frontend Replay Breakpoint** | Offline Breakpoint | When replaying collected trace data | Low | Auto-pause at a certain instruction during playback for user inspection |
| **Backend Execution Breakpoint** | Live Breakpoint | During actual CCU instruction execution/recording | High | Real-time pause of simulator execution; interactive review of current state before continuing |

Both can be used independently or combined: first use backend breakpoints to precisely record trace for key intervals, then use frontend breakpoints for repeated replay analysis.

### 8.2 Breakpoint Types

```cpp
// Breakpoint type enum (breakpoint matching strategies, all essentially locating specific rankId+dieId+instrId)
enum class CcuBreakpointType : uint8_t {
    INSTR_ID,           // Instruction ID breakpoint: triggered when reaching specified instrId
    INSTR_NAME,         // Instruction name breakpoint: triggered when reaching instruction with specified name (e.g., "ReduceAdd")
    CATEGORY,           // Instruction category breakpoint: triggered when reaching specified category (e.g., all Trans instructions)
    RESOURCE_CHANGE,    // Resource change breakpoint: triggered when specified resource is modified (e.g., Xn[5] written)
    CONDITION,          // Conditional breakpoint: triggered when resource satisfies condition expression (e.g., Xn[5] == 0x1000)
    ITERATION,          // Iteration breakpoint: triggered at instruction when Loop reaches Nth round
    ROUND,              // Scheduling round breakpoint: triggered when outer while(HasTask()) enters specified round (e.g., all instructions in round 3)
};

// Breakpoint configuration
// Breakpoints are essentially instruction-related only; core triple: (rankId, dieId, instrId)
struct CcuBreakpointConfig {
    uint32_t bpId;                      // Breakpoint unique identifier
    CcuBreakpointType type;             // Breakpoint matching strategy type
    bool enabled;                       // Whether enabled
    bool oneShot;                       // One-shot breakpoint (auto-disabled after trigger)

    // === Breakpoint Core Location Information ===
    int32_t rankId;                     // Target rank (-1 means all ranks)
    int32_t dieId;                      // Target die (-1 means all dies)
    uint16_t instrId;                   // Target instruction ID

    // === Matching Strategy Additional Parameters (used based on type) ===
    std::string targetInstrName;        // INSTR_NAME: target instruction name
    CcuInstrCategory targetCategory;    // CATEGORY: target instruction category
    std::string targetResourceType;     // RESOURCE_CHANGE: resource type "XN"/"GSA"/"CKE"/"MS"
    uint16_t targetResourceId;          // RESOURCE_CHANGE: resource ID
    std::string conditionExpr;          // CONDITION: condition expression (e.g., "XN[5] == 0x1000")
    uint16_t loopRound;                 // ITERATION: target Loop iteration round
    uint32_t targetExecRound;           // ROUND: target outer scheduling round number

    // === Hit Counting ===
    uint32_t hitCount;                  // Hit count so far
    uint32_t skipCount;                 // Skip first N hits (for "break at 3rd loop iteration")
};

// Breakpoint hit information
struct CcuBreakpointHit {
    uint32_t bpId;                      // Hit breakpoint ID
    uint32_t globalSeqId;               // Current instruction global execution sequence number
    uint32_t execRound;                 // Current outer scheduling round
    uint16_t instrId;                   // Current instruction ID
    std::string instrName;              // Current instruction name
    int32_t rankId;                     // Current rank
    int32_t dieId;                      // Current die
    uint32_t sqeTaskId;                 // Owning SQE task ID (references sqeTaskRegistry)
    CcuResourceSnapshot snapshot;       // Resource snapshot at hit time
    CcuExecutionContext context;        // Execution context at hit time
};
```

**Breakpoint and SQE Relationship**:

Breakpoints only concern "which CCU and which instruction to stop at"; SQE is not involved. SQE information is managed via the `sqeTaskRegistry` registry; each `CcuTraceEntry` references the corresponding SQE task via `sqeTaskId`.

When a user replays and jumps to a certain instruction:
1. The **Instruction Detail Panel**'s SQE info area auto-displays missionId, args, instruction range, etc. of the SQE owning that instruction
2. The **SQE Registry Panel** (left panel) auto-highlights the corresponding SQE entry
3. Multiple SQEs on the same CCU share the same CcuSimulator instance (same `simulatorPtr`); SQEs are distinguished by `(sqeTaskId, missionId)`

### 8.3 Approach 1: Frontend Replay Breakpoint (Offline Breakpoint)

**Principle**: Pure frontend implementation; no backend modifications. User clicks a row in the frontend instruction list to set a breakpoint marker; auto-play mode pauses when reaching that instruction.

**Implementation**:

```javascript
// Breakpoint management in CcuTracePlaybackControl.vue
const breakpoints = ref(new Set());  // Set storing breakpoint seqIds

function toggleBreakpoint(seqId) {
    if (breakpoints.value.has(seqId)) {
        breakpoints.value.delete(seqId);
    } else {
        breakpoints.value.add(seqId);
    }
}

// Auto-play logic
function autoPlay() {
    const timer = setInterval(() => {
        currentStep.value++;
        // Pause on breakpoint
        if (breakpoints.value.has(currentStep.value)) {
            clearInterval(timer);
            isPlaying.value = false;
            notifyBreakpointHit(currentStep.value);
        }
        // Playback ended
        if (currentStep.value >= entries.length - 1) {
            clearInterval(timer);
            isPlaying.value = false;
        }
    }, playSpeed.value);
}
```

**Supported Breakpoint Types**:
- Break by seqId (most common: click instruction row to set)
- Break by instruction name (e.g., "break at all ReduceAdd instructions")
- Break by instruction category (e.g., "break at all Trans instructions")
- Break by resource change (e.g., "break at instructions where Xn[5] is modified"; requires scanning delta data)

**Frontend Breakpoint UI Interaction**:

```
Instruction List Panel:
  seqId | instrId | Name        | Type   | BP   | State
  0     | 0       | LoadImd     | Load   |      | ✓
  1     | 1       | TransMM     | Trans  | 🔴   | ✓     ← Click row number area to toggle breakpoint
  2     | 2       | SetCke      | Ctrl   |      | ✓
  ▶ 3   | 3       | ReduceAdd   | Reduce |      | ← Current (stopped here after breakpoint hit)
  4     | 4       | Loop        | Ctrl   |      |
  ...
```

### 8.4 Approach 2: Backend Execution Breakpoint (Live Breakpoint)

**Principle**: During actual CCU instruction execution, when execution reaches the instruction specified by the breakpoint, pause the simulator execution thread, notify the frontend via WebSocket, and the frontend displays current state. After user confirmation, send a continue command via WebSocket to resume execution. This approach is not implemented in the current plan.

#### 8.4.1 Architecture

```
┌──────────────────┐      WebSocket        ┌─────────────────────────┐
│   Frontend (Vue)  │ ◄──────────────────► │  Backend Debug Server    │
│                  │   /ws/ccu-debug       │  (embedded in hccl-vm    │
│  Breakpoint Mgmt │                       │   process)               │
│  Panel           │                       │                         │
│  Real-time State │                       │  CcuBreakpointManager   │
│  View            │                       │    ├── Breakpoint config │
│  Continue/Step   │                       │    │   table             │
│  Buttons         │                       │    ├── Condition         │
│                  │                       │    │   evaluator         │
└──────────────────┘                       │    └── State snapshot    │
                                           │        collection        │
                                           │                         │
                                           │  CcuSimulator           │
                                           │    └── ExecuteInstr()   │
                                           │        └── CheckBreakpoint() │
                                           │            └── Hit? → Pause thread │
                                           └─────────────────────────┘
```

#### 8.4.2 CcuBreakpointManager

```cpp
class CcuBreakpointManager {
public:
    static CcuBreakpointManager& GetInstance();

    // Breakpoint management
    uint32_t AddBreakpoint(const CcuBreakpointConfig& config);
    void RemoveBreakpoint(uint32_t bpId);
    void EnableBreakpoint(uint32_t bpId, bool enabled);
    void ClearAllBreakpoints();
    std::vector<CcuBreakpointConfig> GetAllBreakpoints() const;

    // Runtime check (called in ExecuteInstr)
    // Returns true if pause is needed (breakpoint hit)
    bool CheckBreakpoint(int rankId, int dieId, uint16_t instrId,
                         uint32_t seqId, const CcuExecutionContext& ctx);

    // Pause and resume control
    void RequestPause();          // External pause request (user clicks pause button)
    void RequestContinue();       // Request to continue execution
    void RequestStepOver();       // Request to execute one instruction then pause
    void RequestStepIntoLoop();   // Request to step into Loop body
    bool IsPaused() const;

    // Hit information (read by Debug Server and sent to frontend)
    CcuBreakpointHit GetLastHitInfo() const;

private:
    // Condition expression evaluation
    bool EvaluateCondition(const std::string& expr, int rankId, int dieId);

    std::mutex bpMutex_;
    std::map<uint32_t, CcuBreakpointConfig> breakpoints_;
    std::atomic<bool> paused_{false};
    std::atomic<bool> stepMode_{false};        // Step mode
    std::condition_variable pauseCV_;           // For thread pause/resume
    std::mutex pauseMutex_;
    CcuBreakpointHit lastHitInfo_;
    uint32_t nextBpId_{1};
};
```

#### 8.4.3 Integrating Breakpoint Checks in CcuSimulator

```cpp
bool CcuSimulator::ExecuteInstr(uint16_t curInstrId)
{
    auto& bpMgr = CcuBreakpointManager::GetInstance();
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    auto instrData = ccuResMgr.GetInstrData(rankId_, dieId_);

    // ① Breakpoint check (before execution)
    if (bpMgr.IsEnabled() && bpMgr.CheckBreakpoint(
            rankId_, dieId_, curInstrId, seqId_, GetExecContext())) {
        // Breakpoint hit → pause thread, wait for user action
        // CheckBreakpoint internally captures snapshot and populates lastHitInfo_
        // Debug Server notifies frontend via WebSocket
        bpMgr.WaitWhilePaused();  // Block until user sends Continue/Step
    }

    auto executor = CcuExecutorFactory::MakeCcuExecutorInstance(...);
    executor->Parser();
    executor->Run();

    // ② Step mode: pause after executing one instruction
    if (bpMgr.IsStepMode()) {
        bpMgr.RequestPause();
        bpMgr.WaitWhilePaused();
    }

    // ③ Trace collection (if enabled)
    if (CcuTraceCollector::IsEnabled()) {
        // ... original trace collection logic ...
    }

    UpdateLoopStatus();
    return true;
}
```

#### 8.4.4 Thread Pause/Resume Mechanism

```cpp
// Thread synchronization in CcuBreakpointManager
void CcuBreakpointManager::WaitWhilePaused() {
    std::unique_lock<std::mutex> lock(pauseMutex_);
    pauseCV_.wait(lock, [this] { return !paused_.load(); });
}

void CcuBreakpointManager::RequestPause() {
    paused_.store(true);
}

void CcuBreakpointManager::RequestContinue() {
    stepMode_.store(false);
    paused_.store(false);
    pauseCV_.notify_all();
}

void CcuBreakpointManager::RequestStepOver() {
    stepMode_.store(true);
    paused_.store(false);
    pauseCV_.notify_all();  // Release execution thread; pauses again after one instruction
}
```

#### 8.4.5 WebSocket Debug Server Protocol

The backend embeds a lightweight WebSocket service (started only when `HCCLVM_ENABLE_CCU_DEBUG=1`), communicating with the frontend in real-time:

```
Message Direction    Message Type                Content
───────────────────────────────────────────────────────────
Server → Client   breakpoint_hit          { bpId, seqId, instrId, instrName,
                                            rankId, dieId, snapshot, context }
Server → Client   execution_paused        { reason: "breakpoint"|"step"|"user",
                                            currentSeqId, instrId }
Server → Client   execution_resumed       { }
Server → Client   resource_snapshot       { rankId, dieId, snapshot }

Client → Server   set_breakpoint          { CcuBreakpointConfig }
Client → Server   remove_breakpoint       { bpId }
Client → Server   continue                { }
Client → Server   step_over               { }
Client → Server   pause                   { }
Client → Server   get_snapshot            { rankId, dieId }
Client → Server   get_instr_info          { rankId, dieId, instrId }
```

#### 8.4.6 Conditional Breakpoint Expression Evaluation

Supports simple resource condition expressions for conditional breakpoints:

```
Syntax examples:
  XN[5] == 0x1000                 // Xn register 5 value equals 0x1000
  GSA[3] > 0x7F0000               // GSA register 3 greater than specified value
  CKE[0] & 0x0001 != 0            // CKE signal 0 bit 0 is 1
  MS[10][0x100:0x10] == 0xFF      // MS 10 offset 0x100, 16 bytes equal specified value
  XN[5] == 0x1000 && GSA[3] > 0  // Combined condition
```

Implementation: Lightweight recursive descent parser, directly reading resource values from `CcuResourceManager` for evaluation.

### 8.5 Frontend Breakpoint Panel UI Design

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Breakpoint Management Panel                                             │
├─────────────────────────────────────────────────────────────────────────┤
│  [+ Add Breakpoint]  [Clear All]  [Enable All ✓]                         │
│                                                                         │
│  ┌──────┬──────────────────────────────────┬──────┬────────┬──────────┐ │
│  │State │ Breakpoint Condition              │ Hits │ oneShot│ Actions  │ │
│  ├──────┼──────────────────────────────────┼──────┼────────┼──────────┤ │
│  │ ✓   │ instrId == 42 @ Rank0/Die0       │ 0    │        │ [✏][🗑] │ │
│  │ ✓   │ name == "ReduceAdd" @ All        │ 3    │        │ [✏][🗑] │ │
│  │     │ XN[5] == 0x1000 @ Rank0/Die1     │ 0    │ ✓     │ [✏][🗑] │ │
│  │ ✓   │ category == Trans @ Rank0/Die0   │ 12   │        │ [✏][🗑] │ │
│  │ ✓   │ loopRound == 3 @ Rank0/Die0      │ 1    │        │ [✏][🗑] │ │
│  └──────┴──────────────────────────────────┴──────┴────────┴──────────┘ │
│                                                                         │
│  ┌─── Add Breakpoint Dialog ─────────────────────────────────────────┐ │
│  │  Breakpoint Type: [Instruction ID ▼]                                │ │
│  │  Rank: [0 ▼]  Die: [0 ▼]  InstrId: [42]                           │ │
│  │  Skip Count: [0]   ☐ One-shot breakpoint                           │ │
│  │                                              [Cancel]  [OK]         │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.6 Collaboration Workflow of Two Modes

```
User operation flow:

Phase 1: Recording (optional, using backend breakpoints)
  ① Set backend breakpoint: instrId=42, Rank0/Die0
  ② Start hccl-vm to execute operator
  ③ CCU reaches instruction 42 → auto-pause
  ④ Frontend displays real-time state: resource snapshot, instruction details
  ⑤ User reviews and clicks [Continue]
  ⑥ Execution completes; trace data auto-saved

Phase 2: Replay Analysis (using frontend breakpoints)
  ⑦ Frontend loads trace data
  ⑧ Set frontend breakpoints in instruction list (instructions of interest)
  ⑨ Click [▶ Auto-play]
  ⑩ Auto-pause when playback reaches breakpoint instruction
  ⑪ User reviews resource changes, instruction details
  ⑫ Click [▶ Continue] or [⏭ Skip to next breakpoint]
```

### 8.7 File Organization (Breakpoint Functionality Additions)

```
src/plugin/solver/virtual_runtime/ccu_executor/trace/
├── ccu_trace_types.h              // (existing) Trace data structures
├── ccu_trace_collector.h/cc       // (existing) Trace collector
├── ccu_trace_serializer.h/cc      // (existing) JSON serialization
├── ccu_breakpoint_types.h         // (new) Breakpoint data structures
├── ccu_breakpoint_manager.h       // (new) Breakpoint manager declarations
├── ccu_breakpoint_manager.cc      // (new) Breakpoint manager implementation
├── ccu_condition_evaluator.h      // (new) Condition expression evaluator
├── ccu_condition_evaluator.cc     // (new)
├── ccu_debug_server.h             // (new) WebSocket debug server
└── ccu_debug_server.cc            // (new)

Frontend additions:
src/plugin/solver/virtual_runtime/insight/frontendV3/src/
├── components/ccu-trace/
│   ├── CcuTraceBreakpointPanel.vue    // (new) Breakpoint management panel
│   ├── CcuTraceBreakpointDialog.vue   // (new) Add/edit breakpoint dialog
│   └── CcuTraceLiveControl.vue        // (new) Live debug control (Continue/Step/Pause)
├── composables/
│   └── useCcuDebugWebSocket.js        // (new) WebSocket communication composable
└── utils/
    └── ccuBreakpointUtils.js          // (new) Breakpoint frontend utility functions
```

---

## 9. Advanced Analysis Features

### 9.1 Search and Multi-dimensional Filtering

Quickly locate problem instructions in large trace datasets with the following search and filtering capabilities:

#### 9.1.1 Text Search

A search box at the top of the instruction list supports real-time filtering:
- Search by instruction name (e.g., enter "TransLocMem" to match all local memory transfer instructions)
- Search by instruction description (match keywords in `Describe()` output)
- Search by seqId (e.g., enter "#42" to jump to the 42nd instruction)

#### 9.1.2 Multi-dimensional Filters

Dropdown filters supporting multi-dimensional combined filtering:

| Filter Dimension | Options | Description |
|---------|------|------|
| **Instruction Category** | Load / Trans / Control / Reduce / All | Filter by 4 major categories |
| **Execution State** | Normal / Failed / CKE Wait / All | Filter by execution result |
| **Loop Context** | Non-Loop / Loop First Round / Loop Middle Round / Loop Last Round / All | Filter by Loop position |
| **Resource Change Type** | Modified XN / Modified GSA / Modified CKE / Modified MS / Cross-CCU / All | Filter by changed resource type |
| **CKE Wait** | Has Wait / No Wait / All | Whether CKE wait is included |
| **SQE Range** | SQE 0 / SQE 1 / ... / All | Filter by SQE session |

Filter states can be saved as "filter presets" for convenient reuse.

#### 9.1.3 Search Result Statistics

Statistics displayed at the bottom of the list after filtering:
```
Showing 42 / 1280 instructions  |  Failed: 2  |  CKE Wait: 5  |  Cross-CCU: 8
```

### 9.2 Data Dependency Tracing

Helps users understand "where did a resource value come from", establishing data flow relationships between instructions.

#### 9.2.1 Trace Origin (Trace Backward)

User clicks a change value in the resource change panel to auto-locate the instruction that last modified that resource:

```
User action: Click [Trace] button next to "Xn[5]: 0x400 → 0x1000"

System behavior:
  1. Search backward from current seqId to find the last instruction that modified Xn[5]
  2. Instruction list auto-scrolls to that instruction and highlights it
  3. Resource change panel updates in linkage
  4. Draw dashed arrow between the two instructions (optional, displayed in instruction list)
```

Implementation principle:
```javascript
// Build resource write index (preprocessed during data loading)
const resourceWriteIndex = {
    'XN:5': [
        { seqId: 0, valueAfter: '0x400' },
        { seqId: 15, valueAfter: '0x1000' },
        { seqId: 42, valueAfter: '0x2000' },
    ],
    'GSA:3': [ ... ],
    'CKE:0': [ ... ],
};

// Trace backward: find the last entry that wrote this resource before current seqId
function traceBackward(resourceType, resourceId, currentSeqId) {
    const key = `${resourceType}:${resourceId}`;
    const writes = resourceWriteIndex[key] || [];
    return writes.filter(w => w.seqId < currentSeqId).pop();
}
```

#### 9.2.2 Trace Impact (Trace Forward)

Reverse operation: find which subsequent instructions read the resource modified by the current instruction:

```
User action: Click [Impact] button next to "Xn[5]: 0 → 0x400"

System behavior:
  1. Search forward from current seqId to find all instructions that read Xn[5]
  2. Highlight these instructions in the instruction list (yellow background)
  3. Statistics: Xn[5]=0x400 was read by N subsequent instructions until overwritten at seqId=M
```

#### 9.2.3 CKE Synchronization Relationship Visualization

CKE is the core synchronization mechanism between CCU instructions. A dedicated CKE relationship view is provided:

```
CKE[3] Synchronization Timeline:

seqId=10  SetCke   CKE[3]: 0→1  (set)
  ↓ (waiting for CKE[3]&0x1 == 0x1)
seqId=25  TransMM  waitCKE[3:0001] ✓ passed
  ↓
seqId=26  ClearCke CKE[3]: 1→0  (clear)
  ↓ (waiting for CKE[3]&0x1 == 0x1)
seqId=40  TransMM  waitCKE[3:0001] ⚠ passed after 14 spin rounds
```

### 9.3 Trace Diff Comparison

Compare trace data from two runs to quickly locate differences. Typical scenario: comparing the same operator under correct/incorrect scenarios.

#### 9.3.1 Comparison Mode

```
┌────────────────────────────────────────────────────────────────────────┐
│  Trace Diff Mode                                                       │
├──────────────────────────┬─────────────────────────────────────────────┤
│  Run A (correct)          │  Run B (error)                              │
│  rank0_die0_trace.json   │  rank0_die0_trace.json                     │
│                          │                                             │
│  seqId | Name    | State │  seqId | Name    | State | Diff             │
│  0     | LoadImd | ✓    │  0     | LoadImd | ✓    │ = (identical)    │
│  1     | TransMM | ✓    │  1     | TransMM | ✓    │ ✗ delta differs  │
│  2     | SetCke  | ✓    │  2     | SetCke  | ✓    │ =                │
│  3     | Reduce  | ✓    │  3     | Reduce  | FAIL  │ ✗ B failed       │
│  ...                    │  ...                                       │
│                          │                                             │
│  Diff Summary:                                  │
│  First divergence: seqId=1 (TransLocMemToLocMem)                       │
│  Diff type: Xn[5] value differs (A:0x1000 vs B:0x0)                   │
│  Total diffs: 3 / 128                                                  │
└──────────────────────────┴─────────────────────────────────────────────┘
```

#### 9.3.2 Diff Detection Rules

| Comparison Item | Match Condition | Diff Marker |
|-------|---------|---------|
| Instruction Sequence | Same instrId + instrName | Instruction missing/extra → Red |
| Resource Delta | Same xnChanges/gsaChanges/ckeChanges | Values differ → Yellow |
| Execution State | Same execState | A succeeded, B failed → Red |
| CKE Wait | Same waitRetryCount (tolerance ±1) | Large wait count difference → Orange |
| Cross-CCU Changes | Same remoteCkeChanges | Remote operation inconsistency → Purple |

#### 9.3.3 Diff Data Source

Diff data is computed in real-time by the frontend after loading two trace files; no backend support needed:

```javascript
// Frontend Diff computation
function computeTraceDiff(traceA, traceB) {
    const diffs = [];
    const maxLen = Math.max(traceA.entries.length, traceB.entries.length);

    for (let i = 0; i < maxLen; i++) {
        const entryA = traceA.entries[i];
        const entryB = traceB.entries[i];

        if (!entryA) { diffs.push({ type: 'missing_in_a', entryB }); continue; }
        if (!entryB) { diffs.push({ type: 'missing_in_b', entryA }); continue; }

        if (entryA.instrId !== entryB.instrId ||
            entryA.instrName !== entryB.instrName) {
            diffs.push({ type: 'instr_mismatch', entryA, entryB });
        }

        if (!isEqual(entryA.resourceDelta, entryB.resourceDelta)) {
            diffs.push({ type: 'delta_diff', entryA, entryB,
                         deltaDiff: diffDeltas(entryA.resourceDelta, entryB.resourceDelta) });
        }

        if (entryA.execState !== entryB.execState) {
            diffs.push({ type: 'state_diff', entryA, entryB });
        }
    }
    return diffs;
}
```

---

## 10. Version Compatibility Strategy

### 10.1 Trace Format Version Management

`CcuRunMetadata.traceFormatVersion` identifies the version number of the trace data format. Version numbers follow semantic versioning:

| Version | Meaning | Change Example |
|------|------|---------|
| 1.0 | Initial version | Current design |
| 1.x | Backward-compatible extension | New optional fields (e.g., Detail for new instruction types) |
| 2.0 | Incompatible change | Data structure reorganization (e.g., delta format change) |

### 10.2 CCU Microcode Version Compatibility

CCU V1 and V2 have different resource specifications:

| Resource | V1 Spec | V2 Spec |
|------|--------|--------|
| XN | 3072 | 4096 |
| GSA | 3072 | 4096 |
| CKE | 1024 | 1024 |
| MS | 1536 | 1536 |
| Instruction Types | 4 categories ~20 types | 4 categories ~35 types |

**Compatibility Strategy**:
1. `CcuRunMetadata.ccuVersion` identifies the microcode version of this trace
2. Frontend dynamically adjusts resource display range based on ccuVersion (V1 shows 3072 XNs, V2 shows 4096)
3. Frontend displays corresponding instruction type names based on ccuVersion (V2 has Add/Sub/Mul arithmetic instructions, V1 does not)
4. Trace Diff supports cross-version comparison (V1 trace vs V2 trace), but marks resource specification differences

### 10.3 Frontend Backward Compatibility

Frontend code handles different trace data versions through a version dispatcher:

```javascript
function loadTrace(data) {
    const version = data.runMetadata?.traceFormatVersion || 1;

    switch (version) {
        case 1:
            return parseTraceV1(data);
        case 2:
            return parseTraceV2(data);
        default:
            console.warn(`Unknown trace version ${version}, trying V1 parser`);
            return parseTraceV1(data);
    }
}
```

For backward-compatible extensions (1.x), new fields are given default values:

```javascript
function parseTraceEntry(raw) {
    return {
        seqId: raw.seqId,
        instrId: raw.instrId,
        // ... required fields
        waitInfo: raw.waitInfo || { hadWait: false, waitRetryCount: 0 },  // default value
        errorInfo: raw.errorInfo || { hasError: false },                   // default value
        crossCcuChanges: raw.crossCcuChanges || { hasCrossCcuChange: false }, // default value
    };
}
```

---

## 11. Trace Reliability Under Exception Scenarios

### 11.1 Problem Analysis

The current trace dump flow has a **critical flaw**: all trace data resides in memory during runtime (`m_traceRun.globalEntries`'s `std::vector`), and is serialized to disk only at the **very end** of `SequentialExecutor::Execute()`.

```
Execute() {
    // trace initialization
    while (HasTask()) {       // ← If crash happens here
        ExecuteOneTask();     //   ← Or crash happens here
    }
    // ===== Trace Dump =====  //   ← Will never be reached
    traceCollector.EndRun();
    DumpToFile(...);
}
```

**Three Exception Scenario Analysis**:

| Scenario | Trigger Location | Can trace be dumped | Reason |
|------|---------|:---:|------|
| **Segfault (SIGSEGV)** | Pointer out-of-bounds in instruction `Process()` | **No** | Process terminates immediately; dump code never executes |
| **Out-of-bounds Access** | Array operations like `GetXnValue()` | **No** | Triggers crash/UB, same as above |
| **C++ Exception (throw)** | Exception thrown during instruction execution | **No** | No try/catch in call chain; propagates to top level triggering `std::terminate` |

**Root Causes (4 defects)**:

1. **All trace data in memory**: All `CcuTraceEntry` stored in `m_traceRun.globalEntries` (`std::vector`); never written to disk during runtime
2. **No signal handling**: Global search for `signal`/`sigaction`/`SIGSEGV`/`SIGABRT`/`atexit`/`set_terminate` yields no results; no opportunity to trigger dump on crash
3. **No exception catching**: `Execute()` → `ExecuteOneTask()` → `TaskCcuGraph()` → `CcuSimulator::Execute()` → `ExecuteInstr()` — entire chain has no try/catch
4. **No incremental dump/checkpoint**: No mechanism for periodic disk writes during runtime

### 11.2 Solution

Adopt a combination of **Approach 1 (Signal Handling) + Approach 3 (Incremental Dump)**.

#### 11.2.1 Approach 1: Register Signal Handlers (for segfaults/abort)

Register SIGSEGV/SIGABRT signal handlers in `SequentialExecutor::Execute()` to emergency-dump collected trace data when the process crashes:

```cpp
#include <signal.h>
#include <execinfo.h>

static const char* g_crashDumpPath = nullptr;

static void CrashDumpHandler(int sig) {
    // 1. Restore default signal handling to prevent infinite recursion if dump triggers another signal
    signal(sig, SIG_DFL);

    // 2. Emergency dump collected trace data
    auto& collector = CcuTrace::CcuTraceCollector::GetInstance();
    if (collector.IsEnabled() && g_crashDumpPath != nullptr) {
        collector.EndRun();
        auto traceRun = collector.GetTraceRun();
        CcuTrace::CcuTraceSerializer::DumpToFile(traceRun, g_crashDumpPath);
    }

    // 3. Print call stack for debugging
    void* frames[64];
    int n = backtrace(frames, 64);
    backtrace_symbols_fd(frames, n, STDERR_FILENO);

    // 4. Re-trigger default handling (core dump)
    raise(sig);
}

// Register in Execute()
g_crashDumpPath = crashOutputPath.c_str();
signal(SIGSEGV, CrashDumpHandler);
signal(SIGABRT, CrashDumpHandler);
```

**async-signal-safety Note**:
Strictly speaking, using `std::ostringstream` (used internally by DumpToFile) in a signal handler is not async-signal-safe. However, in practice it usually works because a crash is unlikely to occur precisely inside malloc. Even if the dump fails, the process is no worse off than before (previously 100% data loss). For strict safety, this could later be changed to use `write()` system calls to directly output binary format.

#### 11.2.2 Approach 3: Incremental Dump (periodic checkpoint)

Periodically serialize collected trace data to disk within the main scheduling loop, so that even if the process is killed by `kill -9` or OOM killer, the most recent checkpoint trace data remains on disk.

```cpp
const uint32_t TRACE_FLUSH_INTERVAL = 100;  // Flush every 100 instructions
uint32_t instrCountSinceFlush = 0;

while (HasTask()) {
    // ... execution logic ...
    
    // Incremental dump
    instrCountSinceFlush++;
    if (traceCollector.IsEnabled() && instrCountSinceFlush >= TRACE_FLUSH_INTERVAL) {
        traceCollector.IncrementalDump();
        instrCountSinceFlush = 0;
    }
}
```

**CcuTraceCollector new `IncrementalDump()` interface**:

```cpp
void CcuTraceCollector::IncrementalDump() {
    if (!m_enabled || m_outputPath.empty()) return;
    auto traceRun = GetTraceRun();  // Get current snapshot (with lock protection)
    CcuTraceSerializer::DumpToFile(traceRun, m_outputPath);
}
```

**Performance Impact Assessment**:
- Flush every 100 instructions; serialization overhead ~1-5ms (depending on collected data volume)
- Total instruction count typically < 10000; flush count < 100; total extra overhead < 500ms
- Relative to instruction execution overhead itself, the impact is acceptable
- Interval can be adjusted via `HCCLVM_TRACE_FLUSH_INTERVAL` environment variable

#### 11.2.3 Coverage Summary

| Layer | Mechanism | Covered Scenarios |
|----|------|---------|
| **Signal Handling** (Approach 1) | `signal(SIGSEGV/SIGABRT, handler)` | Segfaults, abort, stack overflow |
| **Incremental Dump** (Approach 3) | Periodic flush | All scenarios (including kill -9, OOM, power loss) |

The combination ensures:
- **Normal exit**: Final complete trace dump
- **Segfault/abort**: Signal handler emergency dump (filename marked with `_crash`)
- **kill -9 / OOM / Power loss**: Most recent incremental checkpoint trace on disk

### 11.3 File Naming Convention

| Scenario | Filename | Description |
|------|--------|------|
| Normal completion | `hccl_trace_output.json` | Complete trace |
| Incremental checkpoint | `hccl_trace_output.json` | Overwrite; always retains latest state |
| Crash dump | `hccl_trace_crash_dump.json` | Trace snapshot at crash time |
