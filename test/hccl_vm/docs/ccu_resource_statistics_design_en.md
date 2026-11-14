# CCU Resource Statistics Feature Design

> Feature objective: After the checker completes graph generation, collect statistics on all CCU resources
> (channel, instruction space, Variable/Address/Event/CcuBuffer registers, loop, mission)
> at the granularity of **each operator invocation (op) x device (+die)**;
> collect and store the data in DFX form during graph generation, and extract from DFX to generate
> a CSV statistics report after all graph generation completes.
>
> Status: Design phase (D1-D7 implemented, revised to V6 per review comments)
> Affected modules: `src/plugin/checker`
>
> V4 key decisions:
> 1. **Resource collection occurs only during the single-operator graph generation phase; statistics are disabled during the large graph generation phase**. The two paths are fully isolated.
> 2. **Statistics are collected per operator invocation only; no run-level or device-level total aggregation is performed** -- when operators process different data volumes,
>    the CCU resources they use differ, so run-level aggregation provides no value for analyzing operator resource usage.
>
> V5 revision: **The primary channel metric changes from "differential" to "translation-time reference collection"** -- channels are persistent resources reused across operators.
> The differential metric almost always produces empty results for subsequent operators in hccl_test batch invocation scenarios (channelNum=0
> appears extensively), which contradicts the per-op statistics premise. The reference metric becomes the primary metric for phase 1; the differential metric is demoted to a reference metric (see Section 3.1).
>
> V6 revision: **The CSV report displays data in operator sections, with each section header containing operator information** (operator type, algorithm name, data type,
> element count, input/output data volume, reduce type, rankSize). Data rows are reduced to 13 columns (see Section 4.2).
> The topology mapping is corrected to the primary path `Device.id -> server_id -> CommConfigServer.serverKey`
> (D7 testing revealed that deviceId and physical_id use different definitions; see Section 4.3).

---

## 1. Background and Requirements

### 1.1 Requirement Description

1. After the checker completes graph generation, collect statistics on all CCU resources used.
2. **Statistics scope (V4 revision)**: Use **each operator invocation** as the statistics unit -- each invocation
   (`OpExecutionKey{commName, commHash, opIter}`) produces a separate statistics record. Reason: When operators process different data volumes,
   the CCU resources they use also differ (for example, hccl_test invokes operators in batches of 64B/128B/.../1G,
   and each invocation has different loop unrolling counts and register usage). Run-level aggregated statistics provide no value for analyzing operator resource usage.
   **No device-level total report is generated**. Composite test cases (one run invokes multiple operators) are also distinguished by operator.
3. **Collection scope decision (V4)**: Resource collection **occurs only during the single-operator graph generation phase** (`enableNewChecker` path,
   `CheckOneOp -> GenAndCheckGraphV3`). **Statistics are disabled during the large graph generation phase (BigGraphCheckerV3)**.
   The two paths are fully isolated, which eliminates duplicate counting and operator attribution difficulties within the large graph window when both checkers are enabled simultaneously.
   Cost: Runs that use only the large graph path (`enableNewChecker=off`) produce no statistics data. A WARN message is issued (see Section 6 Risk List).
4. Statistics dimensions: Within each operator invocation, statistics are organized by device + CCU. Index labels: `superPodId + serverId + phyDevId`
   locate the device, and `dieId` locates the CCU.
5. Statistics resource items:
   - **channel**: Obtained from the channel table of the checker plugin. Records the count of deduplicated channels.
   - **Instruction space**: The CCU instruction space is 32K instructions x 32B = 1MB. Records the number of generated instructions.
   - **Registers**:
      - A5: Variable, Address, Event, CcuBuffer.
      - A6: Variable, Event, CcuBuffer (A6 has no Address registers).
      - **Event statistics use ckeId as the granularity and do not include mask/bit information**.
      - **A6 introduces register address-based addressing. All resources must first convert addresses to IDs before statistics collection**.
   - **loop**: The concurrent execution unit of the CCU. Records only the **static unrolling instance count** (excluding runtime iteration count).
     Computed from LoopGroup instructions: Given loop template count `n`, unrolling offset `o` (`o in [0, n-1]`),
     and extension count `m`, the loop count = **`n + (n - o) * m`**.
   - **mission**: Statistics collection (added in V2).
   - **Explicitly excluded**: queue, loopCtxId, jetty (counted elsewhere), halfRTT (not tracked for now).
6. Two-level collection/reporting structure: Statistics are collected and stored in DFX form during graph generation. After the checker completes all graph generation,
   data is extracted from DFX to generate a per-op granularity CSV report (see Section 4).

### 1.2 Mapping to Existing Code (Verified)

| Requirement Item | Code Basis |
|---|---|
| deviceId is phyDevId | `operation_data_types.h:36`: "Physical device ID (globally unique)" |
| Topology label source | The checker reads directly from the DB: `TopoMetaConfig` table (`include/runtime_state/sim_models.h:23-26`). The checker already has a precedent for direct DB access (`dag_graphviz_dump.cc:648` reads the Communicator table) |
| Centralized register write entry | `AllRankParamRecorder::SetXn/SetGSA/SetCKE` (`ccu_all_rank_param_recorder_v3.cc:84/90/96`), all carrying `(deviceId, dieId)` |
| A6 address-to-ID conversion | `GetXnAndTypeIdByAddr / GetXnIdByAddr / GetMSIdByAddr` (recorder .cc:279/377/406), address segment constants .cc:259-269. The checker already performs address-to-ID conversion uniformly when accessing resources, so statistics collection reuses the same conversion |
| Channel table | `m_allRankChannelInfo: map<DeviceId, map<dieId, map<channelId, RemoteDieInfo>>>` (`storage_manager.h:289`), loaded by `Loader::GetCcuChannelInfo` + `EnsureChannelInfo` (`storage_manager.cc:582`) |
| Instruction space | `CcuInstrResTab{deviceId, dieId, instrCount, instrSpace[32K][32]}` (`operation_data_types.h:117-122`), decoded by `LoadHcclVmInstrData` (`storage_manager.cc:765`). The 32K upper limit check already exists (.cc:771) |
| mission/instruction range | `CcuTaskParam{dieId, missionId, instStartId, instCnt}`. `CcuGraphStateV3::InitInstrInfo` already computes `missionStartInstrId / missionInstrCount` (`ccu_task_common_v3.cc:307-336`) |
| Loop unrolling formula | A6: `ccu_task_transform_instruct_a6_v3.cc:1703` `expandTimes = (curLoopIdx < expandOffset) ? 1 : (expandCnt + 1)`, total unrolling = `o + (n-o)*(m+1)` = `n + (n-o)*m`, **consistent with the requirement formula**. A5 is structurally identical (a5_v3.cc:2056). No nesting (`CheckLoopGroupNotSupport` guarantees this) |
| LoopGroup field bit widths | A6: `loopInsCnt(10bit) / expandOffset(9bit) / expandCnt(9bit)`. A5: all 7bit |
| CcuBuffer usage convergence point | `GenSliceFromMs(msId, len, slice)` (`ccu_task_common_v3.cc:1071`): All A5/A6 CcuBuffer usage points (normal instructions + loop unrolling instructions, approximately 30+ call sites) call this function with the final msId after `UpdateMSId` offset |
| Existing DFX pattern | `src/dfx/dag_graphviz_dump`: Controlled by the `enableDagGraphvizDump` switch, collects data during graph generation and writes to disk after completion |

---

## 2. Overall Architecture

**Collection occurs only on the single-operator graph generation path**. The collector resets at the run level, buckets data by operator invocation (`BeginOp/EndOp`),
and generates a per-op granularity CSV once all operator graph generation completes (**displayed in operator sections**, each section contains an operator information header +
data rows for each (dev,die) of that operator). No cross-operator aggregation is performed.

```
      Single-Op Graph Phase (Collection)              Large Graph Phase            Post Graph Generation (Reporting)
+----------------------------------------+   +------------------+   +------------------------------+
| RunChecker -> iterate ops -> CheckOneOp:|   | ProcessBigGraph   |   | CcuResourceStatsReporter(dfx) |
|                                        |   |                  |   |  - Read topology from DB,   |
|  auto g = BeginOp(opKey) <-- gate      |   |  Statistics      |   |    build deviceId->           |
|  (RAII guard, destructor calls EndOp   |   |  disabled        |   |    (superPod,server) mapping  |
|   to seal bucket;                      |   |  (no BeginOp,    |   |  - Per-op bucket -> sectioned |
|   must precede LoadHcclVmSynthesisData)|   |   hooks no-op,   |   |    ccu_resource_stats.csv    |
|  (1) Has/GetChannelInfo ref collection |   |   zero collection/|   |    (one section per operator: |
|  (2) SetXn/SetGSA/SetCKE hooks         |   |   zero overhead)  |   |     header + (dev,die) rows)  |
|     (A6: address->ID first)            |   +------------------+   |  - summary log                |
|  (3) GenSliceFromMs records msId       |                          |  - Write to DumpV3Manager dir |
|  (4) TransformLoopGroupInstr key record|                          +------------------------------+
|  (5) InitInstrInfo records mission/    |
|      instruction range                 |
|  (6) SetCurrentOpInfo records operator |
|      display information               |
|  ~guard -> EndOp() seals bucket        |
|      (any exit path)                   |
|                                        |
|  CcuResourceStatsCollector (dfx, singl)|
|  map<OpKey, map<dev, map<die, Stats>>> |
+----------------------------------------+
```

- **Collector**: New files `src/plugin/checker/src/dfx/ccu_resource_stats_dfx.h/.cc`, singleton.
  Run-level reset point: At the beginning of `RunChecker` (around main.cc:508) -- **not** the
  per-op Reset inside `CheckOneOp` (main.cc:221, which is the recorder reset point and happens to be at the same location as the `BeginOp` for this feature).
- **Gate mechanism (key to large graph isolation)**: All hooks (Set*/GenSliceFromMs/TransformLoopGroupInstr/
  InitInstrInfo) reside on code paths shared by single-operator and large graph generation. They are uniformly driven by the "current operator context":
   - Inside `CheckOneOp` (after the Reset at main.cc:221 and before `LoadHcclVmSynthesisData`), call
     `collector.BeginOp(opKey)`.
   - **(Mandatory requirement) `BeginOp` must return an RAII guard whose destructor executes `EndOp()` to seal the bucket**.
     Reason: `CheckOneOp` has multiple early return points after BeginOp
     (rank loading/PrepareOpParam/FinalizeOpGroup/taskMeta loading failures, and so on, main.cc:250-301).
     A bare `EndOp()` call would inevitably be missed on failure paths, causing the operator context to remain dangling
     and the next operator collection data to leak into the previous bucket (cross-bucket contamination). RAII guarantees correct bucket sealing on any exit path
     (including exceptions and CHK_RET short circuits).
   - Hook entry check: If no current operator context exists, return directly (no-op).
   - The large graph path (`ProcessBigGraph`) never calls `BeginOp`, so collection is naturally zero.
     Even when both checkers are enabled simultaneously, no duplicate counting occurs.
- **Per-op isolation correctness basis**: In single-operator mode, `AllRankParamRecorder::Reset()` is called before each operator
  (main.cc:221). Register state does not leak across operators -- all Variable/Address/Event/CcuBuffer entries recorded within an operator bucket belong to that operator.
  No "read/write attribution" ambiguity exists. Channels are collected by actual translation-time references (`Has/GetChannelInfo` wrapper, see Section 3.1)
  and naturally belong to the current operator.
- **Reporter**: Generated uniformly at the end of `RunChecker` (after the single-operator summary table, around main.cc:594-654).
- **Switch**: `CheckerSettings` adds `enableCcuResourceStatsDump` (`manifest.json`, disabled by default).
  The pattern aligns with `enableDagGraphvizDump` (`setting_manager.h:21`).
  When the switch is enabled but `enableNewChecker=off`, a WARN message is issued (no collection path, report is empty).

### 2.1 Statistics Data Structures

```cpp
// Internal to DFX, one instance per (opKey, deviceId, dieId); set types are used for deduplication
struct CcuResStats {
    // ---- channel (reference metric, deduplicated) ----
    std::set<uint32_t> channelIds;       // channelIds actually referenced during translation (channelNum = size(), primary metric)
    std::set<uint32_t> newChannelIds;    // Differential metric: channelIds newly introduced by this op (reference only, not in CSV primary columns)

    // ---- Instruction space ----
    uint32_t instrResidentNum = 0;       // CcuInstrResTab.instrCount (run-level snapshot, reference column, not per-op metric)
    std::set<uint32_t> opInstrIds;       // Union of SQE instruction ranges for this op (instructionNum = size(), primary metric)

    // ---- Registers (recorded at write time; A6 converts address to ID first; deduplicated) ----
    std::set<uint16_t> xnIds;            // xnNum = size(); also records xnMaxId peak value
    std::set<uint16_t> gsaIds;           // A5 only; A6 outputs NA
    std::set<uint16_t> ckeIds;           // ckeNum = size(), granularity is ckeId, no mask
    std::set<uint16_t> msIds;            // msNum = size()

    // ---- loop (idempotent recording by key) ----
    struct LoopGroupKey {              // (missionId, instStartId, loopGroup instruction position)
        uint16_t missionId;
        uint32_t instStartId;          // SQE identifier: the same LoopGroup instruction can be referenced by SQEs with different args
        uint32_t loopGroupInstrId;
    };
    struct LoopGroupRecord {
        uint32_t n;                    // loopInsCnt (template count)
        uint32_t expanded;             // n + (n-o)*m (static unrolling instance count)
    };
    std::map<LoopGroupKey, LoopGroupRecord> loopGroups;
    // loopGroupNum = loopGroups.size()
    // loopTemplateNum = sum of record.n
    // loopNum = sum of record.expanded

    // ---- mission (deduplicated) ----
    std::set<uint16_t> missionIds;       // missionNum = size()
};

// Operator identifier
struct OpKey {  // = OpExecutionKey
    std::string commName;
    uint64_t commHash;
    uint32_t opIter;
};

// Operator display information (V6: attached to CSV section header; CheckOneOp calls SetCurrentOpInfo after CheckerParam is ready)
struct OpInfo {
    std::string collectiveType;  // HcclCmdTypeToString(cmdType)
    std::string algName;         // OpDetailTab.algorithmName (first non-empty across ranks, NA if empty)
    std::string dataType;        // HcclDataTypeToString(dataType)
    uint64_t elementCount;       // dataCount
    uint64_t inputBytes;         // CalcInputOutputSize, rank0 perspective
    uint64_t outputBytes;
    std::string reduceType;      // Only for reduce-type operators, NA for others
    uint32_t rankSize;
};
```

Aggregation containers: `std::map<OpKey, std::map<DeviceId, std::map<uint32_t /*dieId*/, CcuResStats>>>`
+ `std::map<OpKey, OpInfo>`.

**No cross-operator aggregation (V4 decision)**: The report outputs each operator bucket directly, with no device-level total merge layer.

> Deduplication semantics: The statistics metric is "which resource IDs have been used after this operator invocation completes graph generation" -- within an operator bucket, deduplication
> is performed by ID set (the same resource being written/read/cleared multiple times counts only once, not accumulated by instruction count; see Section 3.3 General Principle 1).
> Loops use idempotent recording by key (once the instruction space and SQE parameters are determined, the unrolling result is fixed, regardless of unrolling count; see Section 3.5).
> Resource reuse across different operators **is not merged** -- the resource usage of each invocation is presented independently. This is exactly the analysis goal of this feature
> (different data volumes produce different resource usage).

---

## 3. Collection Details for Each Resource Item

### 3.1 channel (channelNum)

- Data source: `g_allRankChannelInfo[dev][die]` (`storage_manager.h:289`).
- **Primary metric (V5 revision): Reference metric** -- collects "the set of channelIds actually referenced during instruction translation for this operator invocation".
  Reason: Channels are persistent resources reused across operators. The differential metric (see below) almost always produces empty results for subsequent operators
  in the V4 core scenario (hccl_test batch invocation on the same communicator), resulting in widespread channelNum=0 values and unusable data.
  **Implementation (completed)**: Added unified wrappers `HasChannelInfo(dev, die, channelId)` (existence check) and
  `GetChannelInfo(dev, die, channelId)` (value retrieval, returns `const RemoteDieInfo&`).
  These replace the existing 32 direct accesses to `g_allRankChannelInfo` (22 in a5_v3.cc, 10 in a6_v3.cc,
  all verified to be concentrated in two translation files with uniform patterns. A6 `IsExistRemoteDieInfo` preserves find semantics
  and performs internal recording). When the wrapper encounters an actually existing channel, it calls
  `collector.RecordChannel(dev, die, channelId)` to record into the current operator bucket. All reference points are during serial translation,
  reusing the same context mechanism as Plan A in Section 3.4, with consistent gating.
  Note: On the graph node side, only `CcuNotify.channelId` (task_def_v3.h:216, synchronization nodes) carries a channel.
  Data transfer nodes do not carry channel information, so the "traverse nodes after graph generation" approach has incomplete coverage and has been excluded.
- **Reference metric (demoted, optional column): Differential metric** -- at `BeginOp`, record a channel set snapshot for each (dev,die).
  At `EndOp`, compute the difference to obtain "channelIds newly introduced by this op". This metric is valuable only when analyzing
  "which operator first introduces a channel". It is not included in CSV primary columns (retained in DFX).
- halfRTT-related resources: **Not collected** (review explicitly excluded them for now).

### 3.2 Instruction Space (instructionNum)

- **Primary metric (per-op, included in CSV)**: For each SQE of this operator, the `[instStartId, instStartId+instCnt)` range
  is unioned per (dev,die) (hook point: `CcuGraphStateV3::InitInstrInfo`, ccu_task_common_v3.cc:307,
  which already has `missionStartInstrId/missionInstrCount`). The result is stored in the current operator bucket `opInstrIds`.
  `instructionNum = opInstrIds.size()` -- semantically "the number of instructions used by this operator invocation on this CCU".
  Note: `missionInstrCount` is uint16 (forced cast at ccu_task_common_v3.cc:329). The instruction count per single mission
  is far below 64K, so truncation does not affect statistics accuracy (confirmed during review, no action needed).
- **Reference metric (optional column)**: `CcuInstrResTab.instrCount` per (dev,die)
  (`LoadHcclVmInstrData`, `storage_manager.cc:765-795`, a complete run-level resource snapshot).
  Semantically "the total number of generated instructions within the 1M instruction space of this CCU", used to compare against the 32K upper limit occupancy rate
  (`instrResidentNum / 32768`). This value is the same for all operator rows within the same run.
- The difference between the two metrics can reveal other resident instructions on this CCU that are not referenced by this operator (including instruction segments from other operators).
- Implementation optimization (non-blocking): `opInstrIds` uses `set<uint32_t>` with at most 32K elements per bucket per die.
  When the product of operator count and die count is large, interval merge storage can replace the set, and the total length is computed at report time.

### 3.3 Registers: Variable / Address / Event

**General Principles:**
1. **Deduplicate by ID set; do not accumulate by instruction count (confirmed during review)** -- the statistics metric is "which resource IDs have been used after this operator invocation completes".
   When the same resource is repeatedly operated on (written, read, cleared) by multiple instructions, it counts as 1:
   for example, after `SetXn(xn[0])` then `ClearX(xn[0])`, xnNum counts 1, not 2. `std::set<id>` naturally satisfies this requirement.
   Hooks only perform `insert(id)`, regardless of instruction count.
   - ClearX metric (confirmed during review): A6 `ClearX` performs `SetXn(i, 0)` for each ID in the range (a6_v3.cc:502).
     After set deduplication, no double counting occurs. **Any ID touched by ClearX is counted as used** (no requirement that it was previously written).
   - ClearX anomaly detection (optional extension, not implemented in phase 1): Clearing IDs that were never written may indicate an algorithm scheduling issue
     (misalignment with SetXn, or redundant clearing). The detection implementation cost is low (verified at a6_v3.cc:485-510) --
     before `TransformClearXInstr` clears the `[xnId, xmId]` range, check whether each ID already exists in
     `curXn[dev][die]` (this map is populated only by SetXn; the recorder needs a `HasXn` query interface).
     If the ID does not exist, it is a suspected misalignment or redundancy. Note two points: (1) When `xnId > xmId`, the semantics clear up to
     `MAX_LOADX_STOREX_ID_NUM(16383)`. Full-range clearing hits many unwritten IDs, so WARN messages must be aggregated by instruction
     into a count summary (for example, "ClearX [a,b]: k ids never written"). Per-ID printing is prohibited.
     (2) At the end of ClearX, `ProcessSetMask -> SetCKE` is also called. The ckeIds it touches follow the "touched means counted"
     metric and are recorded into ckeIds normally. **The current phase performs only statistics collection, not anomaly detection**. This item is listed as an optional phase 2 task.
2. **Record at write time; prohibit end-state snapshots** -- A6 `ClearX`, Event wait bit clearing (`ClearWaitMask`), and `ClearCKE`
   erase the end state. The snapshot approach misses resources that are "used then cleared". Recording at write time + set deduplication simultaneously satisfies
   "no misses" and "no duplicates".
3. **A6 always converts address to ID before statistics collection** -- A6 instruction operands are register addresses (Event/PFE/CHANNEL/JETTY/MISSION/
   LOOP/CcuBuffer segments, recorder .cc:259-269). **Segment classification is completed at the call site**: Translation code first passes through
   `GetXnAndTypeIdByAddr / GetXnIdByAddr` for conversion and dispatches by `CcuComponerntType` before calling
   SetXn/SetCKE (verified: a6_v3.cc:2312-2317 branches by type before calling). Therefore the `SetXn` hook
   naturally receives only Variable-segment IDs, and no additional classification is needed inside the hook.
4. **The ID after loop unrolling offset is the actual occupancy** -- the value after `UpdateXnId/UpdateCKEId` (a6_v3.cc:205/218,
   bit15 offset logic a6:272) is recorded (offset writes still go through Set* functions; verified that no direct map writes bypass the hooks).
5. **Per-op attribution has no ambiguity (V4)** -- in single-operator mode, `AllRankParamRecorder::Reset()` is called before each operator
   (main.cc:221). Register state does not leak across operators. Hooks uniformly write into the current operator bucket set by `BeginOp`.
   Entries recorded within an operator bucket represent "resources written/occupied by this operator invocation". Reading residual values from other operators does not occur in single-operator mode.
   Cross-device writes (remote post/transfer) carry the actual target (dev,die) in `SetXn/SetCKE(rmtDeviceId, rmtDieId, ...)`
   (a6_v3.cc:2314-2317/2385/2454/2534) and are correctly recorded into the peer row.

| Resource | Hook Point | Recorded Content |
|---|---|---|
| Variable | `SetXn` (recorder .cc:84) | `xnIds.insert(id)`, also records `xnMaxId` peak value. A6 segment classification is already completed at the call site, so the hook receives Variable-segment IDs directly (see General Principle 3) |
| Address (A5 only) | `SetGSA` (.cc:90) | `gsaIds.insert(id)`. A6 report outputs `NA` (do not output 0, to avoid misinterpretation as "unused") |
| Event | `SetCKE` (.cc:96) | `ckeIds.insert(ckeId)`, **granularity is ckeId, no mask/bit information** |

### 3.4 CcuBuffer (Plans for Review)

Current state: The recorder has no CcuBuffer registration table, and msId does not pass through Set* hooks. After code verification,
**`GenSliceFromMs(msId, len, slice)` (`ccu_task_common_v3.cc:1071`) is the single convergence function for all A5/A6 CcuBuffer
usage points**: Normal transfer/reduction instructions and loop unrolling instructions (approximately 30+ call sites,
a5_v3.cc:834-1866, A6 is structurally identical) all call it with the final msId after `UpdateMSId` offset.
A6 address-form CcuBuffer values are also first converted to IDs via `GetMSIdByAddr` (recorder .cc:279).
Therefore, only one problem needs to be solved: "this function lacks (dev,die) context". Three candidate plans:

| Plan | Approach | Change Volume | Pros and Cons |
|---|---|---|---|
| **A. Context injection (recommended)** | Reuse the "current operator context" gate from Section 2: After `BeginOp`, at the instruction translation entry points (`TransformInstrQue`/`ProcessCcuNode`, where the current dev/die is known via `curCcuTask->GetDeviceId()/GetDieId()`), write `(opKey, dev, die)` into the collector current context (a plain member; CCU unrolling is single-threaded and serial). `GenSliceFromMs` reads the context internally and calls `RecordMs(...)` | 2 entry points + 1 function body, **zero changes at call sites** | Covers all CcuBuffer usage paths (including loop merge/unrolling). Shares the same context mechanism with the collection gate. Depends on the "serial translation" assumption (verified: ccu_graph_generator_v3 has no threads; multi-threading exists only in the memory conflict check phase). This assumption must be documented in code comments |
| B. Explicit parameter passing | Add `(deviceId, dieId)` parameters to `GenSliceFromMs` and pass them at all call sites | 1 function signature + 30+ call sites, mechanical changes | No implicit state, most robust. Large change scope; both A5/A6 files need modification |
| C. Post-graph node traversal | After graph generation, traverse TransMem/Reduce/Batch nodes for CcuBuffer slices (`BufferType::MS`, `offset = msId*4096` to reverse-derive msId; loop merge nodes have a separate `msIds` set). Aggregate by node `TaskPosition.deviceId` + die | Independent traversal code | Does not intrude on the translation layer. However, it must cover all node types and depends on graph topology (merge/parallelization rewrites). Risk of incomplete coverage exists |

**Recommendation: Choose Plan A** (Plan A+C cross-validation as a fallback). Outstanding confirmation item: A6 `GetMSIdByAddr` uses
4 CCUA bank x interleave 8 addressing (recorder .cc:266-277). Whether the resulting msId is a die-global index
or a bank-local index needs alignment with hardware specifications and must be noted in the report.

### 3.5 loop (loopNum)

- **Formula (verified against code)**: `loopNum = n + (n - o) * m`
  - `n = loopInsCnt` (number of loop templates within the LoopGroup)
  - `o = expandOffset` (the first o loops are not unrolled; each counts as 1 instance)
  - `m = expandCnt` (the remaining n-o loops are each unrolled into m+1 instances)
  - Code basis: a6_v3.cc:1703-1727.
- **Records only static unrolling instance count, excluding runtime iteration count loopIterCnt** (confirmed during review).
- **Hook point**: Where `TransformLoopGroupInstr` finishes parsing the LoopGroup configuration
  (A6: a6_v3.cc:1654; A5: a5_v3.cc:2056).
- **Idempotent recording (V4 semantics)**: Collection occurs only on the single-operator path, and each operator is unrolled exactly once, so simple accumulation would suffice.
  Key-based idempotent recording is retained as a **defense mechanism**:
  1. If collection is enabled on the large graph path in the future, or if operator retry/repeated unrolling logic is introduced, the count result remains independent of unrolling count.
  2. Same-key value conflicts can serve as a data anomaly probe.
  Rule: Use `(dev, die, missionId, instStartId, loopGroupInstrId)` as the key to write into
  `map<LoopGroupKey, LoopGroupRecord>` (see Section 2.1):
   - Duplicate recording with the same key and same value produces automatic deduplication.
   - `instStartId` (SQE identifier) must be part of the key: The same LoopGroup instruction can be referenced by SQEs with different args,
     where n/o/m may differ. These represent "genuinely different invocations" and should be counted separately.
   - `loopGroupInstrId` takes the absolute ID of the LoopGroup instruction in the instruction space
     (`startLoopInstrId`, available at the site). missionId/instStartId are obtained via
     `curCcuTask` (queId -> `CcuTaskParam` of the current SQE; note that when multiple SQEs exist in a queue, use
     `ccuParamIndexs[queId]` to point to the current SQE).
   - Defense: When two recordings with the same key produce different values (which should not occur in theory), issue a WARN and retain the first value.
- Report aggregation: `loopGroupNum = loopGroups.size()`, `loopTemplateNum = sum of record.n`,
  `loopNum = sum of record.expanded`.
- This layer is not affected by loop merge optimization (`TryProcessMergedLoopA6` / `EmitMergedLoopInstrsV3`) or
  queue parallelization rewrites (`ApplyCcuQueueParallelizationV3`).
  **Deriving counts from final graph nodes is prohibited** (batch merge nodes distort the unrolling count).
- No nested loops (`CheckLoopGroupNotSupport` guarantees this). The formula requires no recursion.

### 3.6 mission (missionNum)

- Hook point: `CcuGraphStateV3::InitInstrInfo` (ccu_task_common_v3.cc:307) or the SQE translation entry point.
  Collects `CcuTaskParam.missionId` into the current operator bucket `missionIds` set (deduplicated per (dev,die)).
- Byproduct: At the same site, `missionStartInstrId/missionInstrCount` is available, supporting the per-op instruction range metric in Section 3.2.

### 3.7 Explicitly Excluded Items

| Resource | Reason |
|---|---|
| queue | Review explicitly excluded it |
| loopCtxId (xp) | Review explicitly excluded it |
| jetty | Already tracked by other statistics channels |
| halfRTT (wishCnt/totalCnt Variable) | Not tracked for now |
| PFE / `TransDep.reserve*` reserved resources | Not included in phase 1. If needed later, collect using the same address-to-ID method as in Section 3.3 |

---

## 4. Statistics Report (CSV, V6: Single Per-Op Sectioned Report)

### 4.1 Generation Timing and Location

- Timing: At the end of `RunChecker`, generated once after all operator graph generation and checking completes.
- Location: Reuses the `DumpV3Manager` output directory, single file `ccu_resource_stats.csv` --
  **sectioned by operator** (each operator invocation is one section: `#` header two lines + 13-column data rows), satisfying "each operator invocation is counted separately"
  (hccl_test batch invocations produce sequential sections; composite test cases with multiple operators are distinguished by section).
- **No run-level or device-level total aggregation report is generated** (V4 decision: aggregated data provides no value for analyzing operator resource usage).
- A summary log is also output (style aligned with `[CHECKER_RUN_SUMMARY]`).

### 4.2 CSV Format Definition (V6: Sectioned Display by Operator + Operator Information Header)

Statistics are already isolated by operator invocation. The display is also **sectioned by operator**: Each operator invocation is one section, with blank lines separating sections.
The section header consists of two `#` comment lines (operator identifier + operator display information). **Fields within header lines are separated by `"; "` instead of commas**
(V6.1): This ensures the entire line occupies only the first column in Excel without being split -- the remaining cells in the header row are empty, and Excel automatically
overflows rightward to display the full operator information. The `#` line prefix is preserved so tools like pandas can skip these lines with `comment='#'`.
Within each section, a column header row is followed by data rows for each (dev,die) of that operator (14 columns = 5 topology/device columns + 9 resource columns.
The operator identifier has been moved to the section header and is no longer repeated per row).

```
# ===== op[0]: commName=hccl_world_group; commHash=0; opIter=0 =====
# opInfo: type=AllReduce; alg=CcuMSAllReduceSoleMesh; dataType=INT32; elements=262144; inputBytes=1048576; outputBytes=1048576; reduceType=SUM; rankSize=4
superPodId,serverId,boardId,phyDevId,dieId,channelNum,instructionNum,loopTemplateNum,loopNum,xnNum,gsaNum,ckeNum,msNum,missionNum
0,0,0,1,1,3,235,2,128,69,31,5,256,1
0,0,0,2,1,3,235,2,128,69,31,5,256,1

# ===== op[1]: commName=hccl_world_group; commHash=0; opIter=1 =====
# opInfo: type=AllReduce; alg=CcuMSAllReduceSoleMesh; dataType=INT32; elements=1074791; ...
...
```

**Section header operator information fields** (source: `CheckerParam` + `OpDetailTab.algorithmName`,
set via `Collector::SetCurrentOpInfo` inside CheckOneOp after CheckerParam is ready):

| Field | Description | Source |
|---|---|---|
| type | Operator type | `HcclCmdTypeToString(cmdType)` |
| alg | Algorithm name | `OpDetailTab.algorithmName` (first non-empty across ranks, NA if empty) |
| dataType / elements | Data type / element count | `HcclDataTypeToString(dataType)` / `dataCount` |
| inputBytes / outputBytes | Input/output data volume (**rank0 perspective**; for rank-sensitive operators like send/recv/alltoall, uses myRank=0) | `CalcInputOutputSize` (check_utils) |
| reduceType | Reduce type. Output only for ALLREDUCE/REDUCE/REDUCE_SCATTER/REDUCE_SCATTER_V; `NA` for others | `HcclReduceOpToString(reduceType)` |
| rankSize | Number of ranks in the communicator | `CheckerParam.rankSize` |

**Section data columns** (13 columns):

| # | Column Name | Description | Source |
|---|---|---|---|
| 1 | superPodId | Super node ID | DB topology mapping (Section 4.3) |
| 2 | serverId | Server ID | DB topology mapping |
| 3 | phyDevId | Physical device ID (= checker deviceId, globally unique) | Collection container key |
| 4 | dieId | Die ID (locates the CCU) | Collection container key |
| 5 | channelNum | Number of channels actually referenced by this operator invocation (reference metric, deduplicated) | Section 3.1 |
| 6 | instructionNum | Number of instructions used by this operator invocation (SQE range union) | Section 3.2 |
| 7 | loopTemplateNum | Sum of n (loop template count) | Section 3.5 |
| 8 | loopNum | Sum of (n + (n-o)*m) (static unrolling instance count) | Section 3.5 |
| 9 | xnNum | Number of deduplicated Variable registers | Section 3.3 |
| 10 | gsaNum | Number of deduplicated Address entries. A6 outputs `NA` | Section 3.3 |
| 11 | ckeNum | Number of deduplicated ckeId values (excluding mask) | Section 3.3 |
| 12 | msNum | Number of deduplicated CcuBuffer entries | Section 3.4 |
| 13 | missionNum | Number of deduplicated missions | Section 3.6 |

- Section ordering: By (opIter, commName, commHash). Rows within a section are ordered by (superPodId, serverId, phyDevId, dieId).
- Row coverage: Each section outputs only the (dev,die) pairs actually involved in that operator. Operators where a single rank has no task and is skipped (main.cc:228) do not produce a section.
- Extended data retained in DFX but not yet included in the CSV: `xnMaxId` (peak watermark), `loopGroupNum`,
  `instrResidentNum`, channel differential metric set (newChannelIds). These can be added as columns later if analysis requires, without changing the collection layer.

### 4.3 superPodId / serverId Mapping (Implemented: Checker Reads DB Directly)

Implementation steps (`dfx/ccu_resource_stats_topo.cc`):

1. Read the `TopoMetaConfig` table (`sim::runtime::Db::GetOneByPred<TopoMetaConfig>`.
    The checker already has DB access precedent: `dag_graphviz_dump.cc:648`). The table entry stores the topo YAML file name.
    `"ranktable"` indicates ranktable.json mode.
2. Call `sim::runtime::ParseCommConfigData(fileName, nullptr, commConfig)`
   (exported by runtime_state, already linked by the checker. Both YAML and ranktable.json modes are handled uniformly).
   This produces `commConfig.servers[]` (containing podId/serverId/serverKey/phyIds).
3. **Primary path (D7 testing correction)**: Query the `Device` table to build the
   `deviceId(=Device.id) -> server_id -> CommConfigServer.serverKey -> (podId, serverId)` mapping.
   **Pitfall**: The checker deviceId (Device.id, tested as 1-based) and the topology phyIds
   (Device.physical_id, 0-based) use different definitions. Directly matching deviceId against phyIds causes
   label misalignment or missing labels (the D7 first-round test result of dev4 -> -1 was caused by this).
4. Fallback path: Direct `phyIds` matching (for deployment scenarios where deviceId == phyDevId). The emplace operation does not overwrite the primary path.
5. If any step fails (table entry missing, file unreachable, parsing failure), produce an empty table + WARN. Both columns output `-1`.
   Report generation is not blocked.

---

## 5. Feasibility Conclusion

**The plan is feasible**:

1. **Collection occurs only during the single-operator graph generation phase** (V4 decision): This path executes each operator independently via
   `CheckOneOp -> GenAndCheckGraphV3`, and the recorder resets before each operator (main.cc:221).
   The per-op statistics boundary is naturally clean -- registers do not leak across operators, channels are collected by translation-time reference attribution (V5).
   The answer to "whether operator invocations can be counted separately" is **yes, and both typical scenarios are covered**:
   - hccl_test batch invocation: Each invocation increments opIter, producing an independent OpExecutionKey and an independent statistics bucket.
   - Composite test cases with multiple operators: Different operators always produce different keys (the commName/commHash/opIter triplet is unique),
     producing independent statistics buckets. The distinction is based on the first-class identity of the input data model, not on heuristic splitting.
2. **The large graph path performs no collection**. The "BeginOp context gate" achieves zero collection and zero overhead. Even when both checkers are enabled simultaneously,
   no duplicate counting occurs. Difficulties such as shared state within the large graph window and TaskPosition attribution are fully resolved.
3. Register writes have centralized entry points (SetXn/SetGSA/SetCKE). A6 address-to-ID conversion is already uniformly implemented in the checker.
   Statistics collection reuses the same conversion. CcuBuffer has a single convergence point `GenSliceFromMs` (Plan A in Section 3.4 requires zero call-site changes).
4. The loop formula is consistent with the code implementation (a6_v3.cc:1703). The hook point is in the translation layer and is not affected by merge/parallelization rewrites.
   Key-based idempotent recording serves as a defense mechanism to ensure the count is independent of unrolling repetitions.
5. Topology labels are reachable through direct DB reads. The DFX collection + end-of-run report pattern has an existing reference (dag_graphviz_dump).
6. **Known cost**: Runs that use only the large graph path (`enableNewChecker=off` and `enableBigGraphChecker=on`)
   produce no statistics data -- the statistics feature explicitly depends on the single-operator checker being enabled. When the switch configuration conflicts, a WARN is issued (see Section 6 Risk #4).

---

## 6. Risk and Outstanding Issue List (V5 Revision)

| # | Risk/Gap | Level | Mitigation |
|---|---|---|---|
| 1 | The topology parsing library (src/cmd / src/runtime_state) is not linked by the checker. `TopoMetaConfig` stores only the YAML file name or `"ranktable"` marker. The absolute path resolution for topology files in the checker process runtime environment is not verified | Medium | Sink the common library or implement equivalent logic in the checker. Verify file path reachability before implementation. Fall back to outputting -1 |
| 2 | `ccu_resource_base_addr_` is currently a mock hardcoded value (storage_manager.cc:757-763, only 2 die slots), affecting A6 address-to-ID conversion. When dieId >= 2, `GetXnAndTypeIdByAddr/GetMSIdByAddr` reports errors directly, causing A6 multi-die statistics to fail rather than degrade | Medium | Statistics reuse the existing checker conversion path (same metric as the checking logic, no new errors introduced). When conversion fails, record into an UNKNOWN bucket + WARN without interruption. The base addr formalization is a pre-existing issue and will take effect automatically when fixed |
| 3 | The CcuBuffer statistics plan awaits review decision (Section 3.4 A/B/C). The A6 msId bank/interleave index scope needs alignment with hardware specifications | Medium | Recommend Plan A. After the scope is confirmed, note it in the report |
| 4 | Runs that use only the large graph path produce no statistics data (cost of the V4 collection scope decision) | Medium | When the statistics switch is enabled but `enableNewChecker=off`, issue a WARN. Document the dependency in the report header. If large graph statistics are needed later, phase 2 can extend by TaskPosition.operatorId attribution (key-based idempotent recording already provides a defense against double counting) |
| 5 | The primary instructionNum metric is the per-op SQE range union, while instrResidentNum (run-level snapshot) is a reference value. The semantics of both must be clearly documented in usage instructions | Low | Sections 3.2/4.2 already define them. Note the metrics in the report header comments |
| 6 | Whether to output a partial report when checking fails with early exit | Low | Recommend still outputting (data is whatever has been collected so far). Mark as partial in logs. Retain failed operator buckets and list them in the summary |
| 7 | Set/GenSliceFromMs is a high-frequency path. Collection hooks have performance overhead | Low | Set insertion + switch disabled by default + first-line short-circuit return when no context exists. The collection context is a plain member with no locking |
| 8 | Reset/gate point placement is error-prone: The collector run-level reset must be placed around main.cc:508. `BeginOp` must be placed inside CheckOneOp after the recorder Reset (main.cc:221). The differential snapshot for the reference metric newChannelIds must precede `LoadHcclVmSynthesisData` (main.cc:244) | **High** | **`BeginOp` must return an RAII guard (destructor calls EndOp)**: CheckOneOp has multiple early return points after BeginOp (main.cc:250-301). A bare EndOp is inevitably missed on failure paths, causing context dangling and cross-bucket contamination for the next operator. RAII covers all exit paths (see Section 2 gate mechanism). Unit tests must cover the "operator fails midway, next operator bucket is not contaminated" scenario |
| 9 | Both checkers running in the same run can overlap (main.cc:536-560): V4 large graph path performs zero collection, architecturally eliminating double counting. The remaining risk is someone mistakenly adding BeginOp on the large graph path in the future | Low | Section 3.5 key-based idempotent recording serves as a fallback. Add code comments on the large graph path stating "collection is prohibited". When same-key values are inconsistent, issue a WARN and retain the first value |
| 10 | Multi-die scenario: A5 device dieId value range differs from A6. Report rows must be generated based on (dev,die) pairs that actually appear | Low | Use the actual keys in the collection container |
| 11 | Operator failure rerun/retry scenarios (if introduced in the future) cause multiple writes to the same operator bucket | Low | At BeginOp, if the bucket already exists, issue a WARN and choose to overwrite or merge (decided at implementation time) |
| 12 | ~~Channel differential metric boundary issue~~ (resolved in V5): The differential metric is always empty for subsequent operators in batch invocation scenarios, contradicting the per-op premise | Closed | Section 3.1 primary metric changed to translation-time reference collection (`Has/GetChannelInfo` wrapper replaces 32 direct accesses). The differential metric is demoted to a reference metric (newChannelIds, not in CSV primary columns) |
| 13 | Channel reference collection requires replacing 32 direct accesses to `g_allRankChannelInfo` (22 in a5_v3.cc, 10 in a6_v3.cc). The changes are mechanical but involve many sites. Missing a replacement causes undercounting | Low | The reference pattern is uniform (`g_allRankChannelInfo[dev][die]...`). Use grep to verify that direct access sites are reduced to zero. The wrapper function enforces the recording path. Unit tests compare against a golden channel set |

---

## 7. Implementation Breakdown (Recommended)

1. **M1 DFX collection framework**: `ccu_resource_stats_dfx` (OpKey bucketed container + run-level Reset hook +
   `BeginOp` returns RAII guard (destructor calls EndOp, covering all CheckOneOp early return failure paths) +
   switch + switch conflict WARN).
2. **M2 Register collection**: SetXn/SetGSA/SetCKE hooks (short-circuit when no context; A6 address-to-ID classification before recording).
3. **M3 CcuBuffer collection**: Implement `RecordMs` using the plan selected during review (recommended: Plan 3.4-A context injection, sharing context with M1 gate).
4. **M4 loop/mission/instruction range collection**: Key-based idempotent recording at TransformLoopGroupInstr
   (including same-key value conflict defense). InitInstrInfo collects mission/per-op instruction ranges.
5. **M5 Channel collection**: Add `HasChannelInfo/GetChannelInfo` unified wrappers (embedding `RecordChannel`
   reference collection). Replace 32 direct accesses in a5/a6 translation files. Retain BeginOp/EndOp differential logic as the reference metric (newChannelIds).
6. **M6 Topology mapping + CSV reporter**: Read TopoMetaConfig from DB, parse via `ParseCommConfigData`,
   reverse-lookup Device.server_id to serverKey. Generate a single sectioned `ccu_resource_stats.csv` per operator bucket
   (section header with operator information + 13 data columns, see Section 4.2) + summary log.
7. **M7 Testing**: Loop formula unit tests (o=0, o=n-1, m=0 boundary cases). Multi-op bucketing/deduplication scenarios
   (hccl_test batch simulation, **focus on verifying that channelNum reference metric is non-zero and stable across operators during batch invocation**).
   Channel reference collection comparison against a golden set (verify no missed replacements among 32 direct access sites).
   **Operator fails midway, RAII seals bucket, next operator bucket is not contaminated**.
   A5/A6 dual-version golden CSV. Both checkers enabled simultaneously produces zero collection verification.
