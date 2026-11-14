# CCU Modeling and Refactoring Design Document

**Version:** v0.1 (analysis + design, no code landed)
**Date:** 2026-10-28
**Scope:** `asset/cluster_model/config/device/` (new), `src/proxy/level1/ccu`, `src/proxy/level2`, `src/plugin/runner`, `src/plugin/checker`, `src/store`, `src/topo`

---

## 1. Background and Goals

### 1.1 Background

The CCU modeling in hccl-vm currently has three disconnected layers:

- **hccl-vm/proxy side**: It exposes a set of **fake base addresses** to the northbound hcomm layer (die0 = `0x123123123`, die1 = `0x456456456`). These addresses do not correspond to any real memory.
- **runner side**: It uses process-private C++ arrays (`std::array<uint64_t, N>` / `std::vector<char>`) to simulate Variable/Address/Event/CcuBuffer resources. The execution results exist only in the runner process heap.
- **checker side**: It **duplicates** the fake base addresses from the proxy and the address mapping constants from the runner, and independently performs address-to-id reverse lookups.

The three layers maintain consistency by agreeing on the same magic numbers. No single source of truth exists, and no real memory entities back them.

### 1.2 Goals

| Number | Goal | Acceptance Criteria |
|---|---|---|
| G1 | Establish a **single device specification configuration source** shared by hccl-vm, runner, checker, and third-party plugins | The code no longer contains `0x123123123`, `0x456456456`, or A6 region offset literals |
| G2 | Make CCU resources **memory-backed**: allocate real shared memory during the hccl-vm phase and bind it to normalized device virtual addresses | Variable/Event/CcuBuffer data written by the runner can be read by hccl-vm, proxy, and checker through the same virtual address |
| G3 | Addresses are **unique per rank** and fall within the rank memory window, conforming to real hardware addressing semantics | `ccuBase(rank,die)` is derived from configuration and differs across ranks |
| G4 | rtMalloc **never falls into the CCU window**, and the available user_mem quota equals the **entire remaining space** in the rank window after CCU is subtracted | **Structurally guaranteed by the per-segment first-fit mapping** (not dependent on quota constants); cumulative allocations exceeding 1022 GiB return failure and log an ERROR, without silent wraparound or intrusion into the next rank window (§4.5.7) |
| G5 | Eliminate **duplicate implementations** of address-to-id conversion and resource specifications | The three conversion code locations in runner, checker, and light_performance converge into a single library |
| G6 | Third-party plugins (performance simulation and others) can directly use the **real CCU base address** | The plugin side requires no secondary address conversion |

### 1.3 Non-goals

- Do not change CCU instruction semantics or add new instruction support.
- Do not change the existing field layout of binary dump formats (`*_hcclvm_syn_data.bin` and others) (see §9).
- Do not refactor the graph generation or validation algorithms in the checker; only replace its address resolution layer.

---

## 2. Current State Analysis (Code Facts)

### 2.1 Components and Code Map

```
                          ┌─────────────────────────────────────────────┐
                          │  hcomm (libhcomm.so / ccu_dataplane.so)     │
                          └───────┬─────────────────────┬───────────────┘
               Northbound HcommCcu*│                     │Southbound TLV / rtCCULaunch
                ┌────────────────▼──────┐   ┌──────────▼───────────────────┐
                │ src/proxy/level1/ccu/ │   │ src/proxy/level2/            │
                │  ccu_level1_res.cc    │   │  hccp_ccu_stub.cc            │
                │  ★Real allocation     │   │  hccp_stub.cc:GetCcuMemInfo  │
                │   model (bump)        │   │  ★No resource model,         │
                │  DIE_BASE_ADDR[2]     │   │   returns caps only          │
                └───────────┬───────────┘   │  0x123123123/0x456456456     │
                            │               └──────────┬───────────────────┘
                            │  DB (Ccu / CcuInstrResTab / CcuChannelTab)
                            ▼                          │
                ┌───────────────────────┐              ▼
                │ src/store/            │   ┌─────────────────────────────┐
                │  store_dump_shm_data  │   │ rtMalloc → AllocVirMem      │
                │  ★ccu0/1_base_addr    │   │  0x8000000000 + (devPk-1)*1T│
                │   (placeholder,       │   │  → /dev/shm/hvm_*/ (real)   │
                │    nobody reads)      │   └─────────────────────────────┘
                └───────────┬───────────┘
                            │
             ┌──────────────┴───────────────┐
             ▼                              ▼
 ┌───────────────────────────┐  ┌──────────────────────────────┐
 │ src/plugin/runner/        │  │ src/plugin/checker/          │
 │  ccu_resource_manager.cc  │  │  ccu_all_rank_param_         │
 │  ★XN/GSA/CKE/MS = process │  │  recorder_v3.cc              │
 │   heap                    │  │  ★A6 address table (copy)    │
 │  ★A6 address table (copy) │  │  curXn/curCKE/curHBM = map   │
 │  Not externally visible   │  │                              │
 └───────────────────────────┘  └──────────────────────────────┘
```

### 2.2 Fact 1: The CCU base address is independently hardcoded in 8 locations

| # | Location | Form | Description |
|---|---|---|---|
| 1 | `src/proxy/level2/hccp_ccu_stub.cc:73,75` | `baseinfo.resourceAddr` | `SetCcuV1ResourceBasicInfo`, with `// todo: set different resourceAddr based on dieId in the future` |
| 2 | `src/proxy/level2/hccp_ccu_stub.cc:102,104` | Same as above | `SetCcuV2ResourceBasicInfo` (A6) |
| 3 | `src/proxy/level2/hccp_stub.cc:2098-2099` | `ccu_die0/1_base_addr` | Region table base address in `GetCcuMemInfo` |
| 4 | `src/proxy/level1/ccu/ccu_level1_res.cc:37` | `DIE_BASE_ADDR[2]` | Level1 uses it to **derive the real VA** returned to hcomm |
| 5 | `src/store/store_dump_shm_data.cc:497-498` | `ccu0/1_resource_base_addr` | Comment says "reserved placeholder (consistent with checker side)", **written but nobody reads** |
| 6 | `src/plugin/runner/runner_worker/hccl_task_sequential_execute_v2.cc:39` | `CCU_RESOURCE_BASE_ADDR` | Comment says "consistent with proxy side SetCcuResourceBasicInfo" — **consistency maintained only by comments** |
| 7 | `src/plugin/runner/runner_utils/storage_manager.cc:321-322` | `push_back(...)` | This path is currently dead code (`RunVirtualRuntime` has no callers) |
| 8 | `src/plugin/checker/src/utils/storage_manager.cc:761-762` | `InitCcuInfo` | Populates `AllRankParamRecorder::ccu_resource_base_addr_` |

There are also 2 locations where **the structure is ready but not activated**:

- `sim::runtime::Ccu.resource_addr` (`include/runtime_state/sim_models.h:167`): The field and comment are complete. `FillCcu()` in `src/topo/topo_ascend_cluster_parser.cc:970` **only fills device_id/die_id/status/id**, and `resource_addr` is always 0. The only reader is the table print function at `src/cmd/cmd_table_utils.cc:135`.
- `ModelInfoCommInner.ccu0/ccu1_resource_base_addr` (`include/sim_binary_data_type_pub.h:151-152`): This is serialized into the syn file on disk, but both the runner and checker **ignore it and re-hardcode the values**.

> **Conclusion**: The data pathways have long been in place (DB columns + disk fields). What is missing is a "writer" and a "reader." This is the lowest-cost entry point for this refactoring.

### 2.3 Fact 2: Four mutually conflicting CCU address layouts exist

This is a more severe problem than "fake addresses" — **even if the base addresses are unified, the region offsets still do not match**.

| Layout | Source | INS | GSA | XN | CKE | MS |
|---|---|---|---|---|---|---|
| **L2 `GetCcuMemInfo`** | `hccp_stub.cc:2089-2209` | base+0 (1MB) | base+0x100000 | base+0x108000 | base+0x110000 | Not reported |
| **L1 `CcuResLayout` V1** | `ccu_level1_res.cc:44-46` | base+0x100000 | base+0x900000 | base+0x908000 | base+0x910000 | — |
| **L1 `CcuResLayout` V2(A6)** | `ccu_level1_res.cc:47-48` | base+0 | 0 (A6 has no GSA) | base+0x100000 | base+0x140000 | — |
| **runner/checker A6 table** | `ccu_resource_manager.cc:378-396`<br>`ccu_all_rank_param_recorder_v3.cc:259-277` | base+0 | — | base+0x100000 | base+0x140000 | Starting at base+0x4000000, 4×CCUA, 2MB spacing |
| **level1 MS window (5th layout)** | `ccu_kernel_sim.cc:66` | — | — | — | — | `0x08000000` (128MB), per-die 32MB |

Key defects:

1. **`GetCcuMemInfo` does not distinguish A5 from A6**. `hccp_stub.cc:2089` has no `soc_version` branch. It returns V1-style `+0x8000` stride tables even for A6 chips, while the actual Variable location on A6 is at `+0x100000`. The offsets that hcomm receives do not match the offsets that the runner and checker use for reverse lookup — **they are different sets**.
2. **The stride in `GetCcuMemInfo` contradicts its own comments**: The comment says "actually reserved 32KByte." The Address-to-Variable stride is `0x8000` (32KB), but `mem_size` only reports `0x6000` (24KB), leaving an 8KB gap. Meanwhile PFE reports `0x100` yet occupies an `0x8000` stride. The `size` and `stride` semantics of regions are mixed.
3. **The level1 CcuBuffer base address `0x08000000` differs by a factor of 2 from the `0x4000000` used by the runner and checker**. Level1 uses "die-equal-split 64MB window" while the runner and checker use "4×CCUA×8-way interleave." The same msId produces different addresses on the two sides.
4. `hccp_ccu_stub.cc:400-410` also uses the fake base address **after bit-shifting** as a remote die discriminator: `dstVa == 0x456000000 → die1`, `0x123000000 → die0`. This check is now gated by `devType == DEV_TYPE_950` (**only the A5 path is active**, and the value is retrieved through `GetRemoteCcuVa()`). This approach treats magic numbers as protocol fields. Once the base address changes, this logic must be rewritten in sync.

### 2.4 Fact 3: Four implementations of address-to-id conversion exist

| Implementation | Location | Size | Differences |
|---|---|---|---|
| runner | `ccu_resource_manager.cc:378-531` | ~154 lines | Returns `bool` |
| checker | `ccu_all_rank_param_recorder_v3.cc:259-445` | ~187 lines | Returns `HcclResult` + error code |
| proxy level1 | `ccu_level1_res.cc:504-555` (forward id→addr) | ~50 lines | Forward only, no reverse lookup |
| light_performance | `ccu_graph_builder.py:221-294`, `register_model.py` | Python rewrite | Comment says "ported from checker ...a6_v3.cc" |

The **constant blocks, `findBaseAddr` / `findTypeByAddr` / `findBaseSize` function names, and function bodies are identical** across the four implementations (even the `CcuComponerntType` spelling error is the same: runner `ccu_resource_manager.h:41-50`, checker `ccu_all_rank_param_recorder_v3.h:37-46`).

Additionally, the loop offset helper functions `UpdateXnId/UpdateCkeId/UpdateMSId/GetXnId/UpdateAddress(WithoutStride)` are duplicated in the runner (`ccu_executor_base.cc:90-140`) and the checker (`ccu_task_transform_instruct_a6_v3.cc:205-277`), with identical semantics (bit15 = offset mode).

> **Observed risk**: `ccu_resource_manager.cc:512` and `:522` both contain `// need boundary check  todo`, meaning `GetXnIdByAddr` / `GetAddrByXnId` **have no upper-bound validation**. The same functions on the checker side have `GRAPH_OUT_OF_RANGE` validation. Two implementations of the same logic, with inconsistent robustness.

### 2.5 Fact 4: Resource specification constants are duplicated and mutually contradictory

| Resource | Runner Array Limit | Proxy Reported Capacity | Checker Validation Upper Bound | Global Default | Conflict |
|---|---|---|---|---|---|
| XN (A6/V2) | `SimCcuV2::CCU_RESOURCE_XN_MAX = 4096`<br>`ccu_microcode_common_v2.h:83` | `xnNum = 16384`<br>`hccp_ccu_stub.cc:114` | `MAX_LOADX_STOREX_ID_NUM = 16383`<br>`ccu_task_transform_instruct_a6_v3.cc:77` | `CCU_RESOURCE_XN_MAX = 3072`<br>`sim_common_defs.h:41` | **4096 vs 16384**: The proxy allows hcomm to allocate up to id 16383, but the runner array only has 4096 entries → out-of-bounds write |
| Loop Engine (V2) | `CCU_RESOURCE_LOOP_ENGINE_MAX = 200`<br>`_v2.h:85` | `loopEngineNum = 512`<br>`hccp_ccu_stub.cc:110` | — | 200 | **200 vs 512** |
| Channel | `MAX_CCU_CHANNEL_NUM = 128`<br>`_v1.h:66` (runner array width) | V1=128 / **V2=1024**<br>`hccp_ccu_stub.cc:90,119` | `CCU_MAX_CHANNEL_NUM = 16` (profiling) | — | **V2 reports 1024, runner only has 128 slots** |
| GSA | V2 array still has 4096 entries | V2 `gsaNum = 0` | `curGSA` comment says "A6 has no GSA" | 3072 | V2 GSA retention is inconsistent |
| CKE | 1024 | 1024 | 1024 | 1024 | Consistent ✓ |
| MS | 1536×4KB = 6MB | 1536 | `msId*4096`, **no capacity validation** | 1536 | Checker lacks validation |

Additionally, `DumpCcuCkeResouceInfo` (`ccu_resource_manager.cc:677`) uses `CCU_RESOURCE_MS_NUM` (1536) to iterate over the 1024-entry `cke_` array — an out-of-bounds read once activated.

`GetXnAddr` (`ccu_resource_manager.cc:364-376`) steps by `xnId * 64` `uint64_t` units (= 512B/XN), which contradicts the 8B stride used elsewhere. The comment was copied from `GetMsAddr` and says "one MS capacity is 4KB." For 3072/4096-entry arrays, `xnId ≥ 48/64` causes out-of-bounds access. The callers `load_executor.cc:59` / `store_executor.cc:59` pass the **value** of the XN instead of the id.

> These contradictions have not triggered so far only because the actual allocation volume in hcomm is far below the upper limits. **A unified configuration source must also unify the capacity specifications**; otherwise G1 only solves half the problem.

### 2.6 Fact 5: The CCU state in the runner is process-private heap memory

`CcuResourceV1/V2` (`ccu_resource_v1.h:42-46`):

```cpp
std::array<uint64_t, SimCcuV1::CCU_RESOURCE_XN_MAX>  xn_[HcclSim::DIE_NUM];
std::array<uint64_t, SimCcuV1::CCU_RESOURCE_GSA_MAX> gsa_[HcclSim::DIE_NUM];
std::array<uint64_t, SimCcuV1::CCU_RESOURCE_CKE_MAX> cke_[HcclSim::DIE_NUM];
std::array<std::vector<char>, HcclSim::DIE_NUM>      ms_;   // 6MB per die
```

The `runner` is a **standalone executable** (`src/plugin/runner/CMakeLists.txt:19`), and these arrays reside in the runner process heap.

The runner currently produces only three types of externally visible output, none of which supports programmatic queries:

| Output | Trigger | Limitation |
|---|---|---|
| JSON trace | `HCCLVM_ENABLE_TRACE=1` | **Only initialized in the legacy `SequentialExecutor::Execute()`** (`hccl_task_sequential_execute.cc:95-111`); the production path `SequentialExecutorV2::Execute()` (`_v2.cc:182-238`) never calls `SetEnabled/StartRun` → the collector is always disabled, `RecordEntry` calls are all short-circuited |
| input/output dump | `HCCLVM_ENABLE_DUMP_DATA` | Can only inspect buffers, cannot see Variable/Event/CcuBuffer |
| Logs | — | Not programmatically queryable |

`enableDump_` (`ccu_resource_manager.h:121`) **is never assigned**. `DumpCcuInstructions` (`:607`) and `DumpChannelId2RmtRank` (`:714`) are always no-ops. The only call site for `DumpCcuXn/Gsa/Cke/ChannelResouceInfo` (`ccu_simulator.cc:459`) has been commented out.

> **This is the root cause of problems 1.1 and 2.2**: The execution results of the runner are physically invisible to any second process. hcomm CCU interface test cases (for example, reading back Variable/Event after launch to check synchronization completion) are **fundamentally untestable** on the simulator.

By contrast, **the on-chip memory buffer path is already cross-process visible**: `rtMalloc` → `AllocPhyMem` (`/dev/shm/hvm_*/`) + register `PhyMemBlock`/`VirtualMemBlock` → the runner uses `sim::AcquireDevPtrInNoHostProcess` (`store_sim_shm_memory_common.cc:22`) to query by DB range and obtain the host pointer. CCU resources only need to follow the same mechanism — **no new IPC needs to be invented**.

The cross-process shared memory root directory is also already available: `HCCL_VM_RESOURCE_ROOT` is set by the main process during the `start` phase and inherited through the entire chain (`include/store/store_sim_resource_root.h:26`, `src/proxy/common/hccl_proxy_round_slot.h:57` already has a precedent). The default is `/dev/shm/hvm_<pid>`.

### 2.7 Fact 6: Rank memory addressing already has a 1 TiB window prototype

`src/store/store_sim_device_memory_manager.cc:47-62`:

```cpp
void* DeviceMemoryManager::AllocVirMem(uint64_t devPhyId, size_t size)
{
    size_t alignedSize = (size + 7) & ~7;
    constexpr uint64_t deviceBase     = 0x8000000000;    // = 2^39 = 512 GiB
    constexpr uint64_t deviceAddrStep = 0x10000000000;   // = 2^40 = 1 TiB
    virAddr = deviceBase + ((devPhyId - 1) * deviceAddrStep) + m_deviceMemSizes[devPhyId];
    m_deviceMemSizes[devPhyId] += alignedSize;
}
```

Call sites `aclrt_memory_stub.cc:277,293`: `curDeviceId = GetCurrDeviceId()`, and `GetCurrDeviceId()` (`db_sim_runner_ops.cc:154-171`) returns `Context.device_id`, that is, the **Device table primary key** (auto-increment, starting from 1). `virMem.rank_id = curDeviceId` also stores the primary key.

**This fully aligns with the requirement 3.1 approach**:

| Requirement | Existing Code | Conclusion |
|---|---|---|
| rank0 base = `0x8000000000` | `deviceBase = 0x8000000000`, devPk=1 → `(1-1)*stride` | ✅ Already consistent |
| rank window = 1T | `deviceAddrStep = 0x10000000000` | ✅ Already consistent |
| rank1 base = `0x18000000000` | devPk=2 → `0x8000000000 + 0x10000000000` | ✅ Already consistent |

The checker side already has corresponding constants (`src/plugin/checker/header/internal/sim_common.h:38-39`: `SIM_MEM_BLOCK_SIZE = 0x10000000000`, `SIM_MEM_MASKER = 0xFFFFFF0000000000`), **but nobody currently uses them**.

**Existing gaps**:
- `AllocVirMem` **has no upper-bound check**. `m_deviceMemSizes` has no upper limit → a single rank allocation exceeding 1 TiB overflows into the next rank window; once the CCU window (at 56 GiB) is introduced, exceeding 56 GiB silently corrupts the CCU region.
- `m_deviceMemSizes` is a **process-internal map** (`DeviceMemoryManager` is a process-level singleton). In multi-process per-rank scenarios, each process indexes by a different devPk with no conflict; however, within the same process, the watermarks are shared across ranks during cross-rank allocation. The semantics need confirmation.
- The window index is `devPk - 1`, while the **rankId in hcomm comes from the ranktable**. The two are not guaranteed to be equal (the repository already has `docs/ranktable_rankid_device_conflict_analysis.md` specifically discussing such conflicts).

---

## 3. Problem List

| ID | Problem | Root Cause | Impact | Evidence |
|---|---|---|---|---|
| P1 | The runner is disconnected from hccl-vm; CCU execution results are not inspectable | The runner models CCU with process-private arrays (§2.6) | hcomm CCU interface test cases (reading back Variable/Event after launch) are **fundamentally untestable**; the proxy cannot answer "where has instruction execution reached" | `ccu_resource_v1.h:42-46` |
| P2 | Fake base addresses are meaningless to third parties and require secondary conversion | Magic numbers are hardcoded in 8 locations with no configuration source (§2.2) | Performance simulation and other plugins must carry their own conversion table; addresses cannot be explained in external deliveries | `hccp_ccu_stub.cc:73` and 7 other locations |
| P3 | The CCU base address is the same for all ranks, which does not match real hardware | `DIE_BASE_ADDR[2]` indexes only by die and **does not include the rank dimension** | Cross-rank addresses cannot be distinguished; remote notify target addresses are semantically ambiguous; `hccp_ccu_stub.cc:400-410` (A5-only) can only use magic number bit-shifting as a die discriminator | `ccu_level1_res.cc:37,549` |
| P4 | Four address layouts conflict with each other | No unified layout definition exists (§2.3) | `GetCcuMemInfo` returns V1 offsets for A6; the level1 CcuBuffer base address differs by 2x from the runner/checker; the same msId produces different addresses on the two sides | `hccp_stub.cc:2089` (no version branch), `ccu_kernel_sim.cc:66` vs `ccu_resource_manager.cc:385` |
| P5 | Four duplicate implementations of address-to-id conversion | No shared library exists (§2.4) | Changing one location misses the other three; the runner side lacks boundary validation while the checker side has it | `ccu_resource_manager.cc:512,522` contain `todo` comments |
| P6 | Resource capacity specifications are contradictory | No unified specification source exists (§2.5) | The proxy allows Variable id up to 16383 while the runner array has only 4096 entries → **potential out-of-bounds write**; Channel V2 reports 1024 while the runner has only 128 slots | `_v2.h:83` vs `hccp_ccu_stub.cc:114` |
| P7 | The `Ccu.resource_addr` field is empty | `FillCcu()` does not fill it | The existing DB pathway is wasted; each party can only hardcode its own values | `topo_ascend_cluster_parser.cc:970-975` |
| P8 | The `ccu0/1_resource_base_addr` in the syn file is written but nobody reads it | Consumers re-hardcode the values | The on-disk format and runtime specification may be inconsistent without any warning | `store_dump_shm_data.cc:497` (comment says "placeholder") |
| P9 | rtMalloc has no upper-bound protection | `AllocVirMem` has no checks | Once the CCU window is introduced, it silently corrupts CCU memory | `store_sim_device_memory_manager.cc:57` |
| P10 | `CcuResourceManager` has no boundary checks across the entire chain | Accessors only do `switch(version)` | Out-of-bounds `rankId/dieId/xnId/...` causes undefined behavior; `GetCkeValue` also truncates `uint64_t` to `uint16_t` | `ccu_resource_manager.cc:134-237` |
| P11 | The runner trace is non-functional on the production path | `SequentialExecutorV2` does not initialize the collector | CCU debugging capability is unavailable on the most commonly used path | `_v2.cc:182-238` |
| P12 | Two `StorageManager` implementations with the same name but different code | One each in checker/runner | ~150-200 lines of boilerplate duplication; `FindRootPath`/`IsDirExists`/`GetRankSize` are copied verbatim | See §2.1 and exploration findings |

---

## 4. Refactoring Design

### 4.1 Overall Architecture

Introduce three layers: **a single data source + a single parsing library + a single memory entity**:

```
┌──────────────────────────────────────────────────────────────────────┐
│ ① Configuration Layer  asset/cluster_model/config/device/<soc>.yaml  │
│    Address space partitioning / CCU region table / resource          │
│    capacity / MS layout                                              │
└───────────────────────────┬──────────────────────────────────────────┘
                            │ Parse (yaml-cpp, guarded by HAVE_YAML_CPP)
┌───────────────────────────▼──────────────────────────────────────────┐
│ ② Model Layer  New static library device_spec (src/device_spec/)     │
│    DeviceSpecRegistry : soc_version → DeviceSpec                     │
│    RankAddressSpace   : rank/devPk  → rankBase / ccuDieBase          │
│    CcuAddressSpace    : (rank,die)  ↔ (region,type,id) bidirectional │
│    ★ The sole holder of region offset/stride literals                 │
└───────────┬─────────────────────────┬────────────────┬───────────────┘
            │                         │                │
┌───────────▼──────────┐  ┌───────────▼──────┐  ┌──────▼──────────────┐
│ ③ Entity Layer       │  │ proxy L1/L2      │  │ runner / checker /  │
│ CcuMemoryService     │  │ Return real base │  │ third-party plugins │
│  shm + PhyMemBlock   │  │ address; write   │  │ Read/write directly │
│  + VirtualMemBlock   │  │ Ccu.resource_addr│  │ by VA using the     │
│                      │  │                  │  │ same CcuAddrSpace   │
└──────────────────────┘  └──────────────────┘  └─────────────────────┘
```

Design highlights: **② The model layer is pure computation, stateless, and has no IO** (except loading configuration). Therefore the runner, checker, proxy, and Python plugins can all safely depend on it. **③ The entity layer is only created on the hccl-vm side**; other processes map it through the existing DB + shared memory mechanism, with no new IPC introduced.

### 4.2 Device Configuration File Design

#### 4.2.1 Directory and Naming

```
asset/cluster_model/config/device/
├── ascend950.yaml      # A5 / CCU V1
└── ascend960.yaml      # A6 / CCU V2
```

The naming convention is consistent with `cluster/` and `server_or_pod/`. **Reference implementations are already landed**: `asset/cluster_model/config/device/device_spec_yaml_demo_ascend950.yaml` (A5) and `device_spec_yaml_demo_ascend960.yaml` (A6); they will be renamed to `ascend950.yaml` / `ascend960.yaml` for formal delivery.

**Matching rules (two phases)**:

| | Phase 0 (current) | Target state (phases 1-3, see §4.7) |
|---|---|---|
| Spec selection basis | Match `name` (= file name) against `soc_version` from `cluster/*.yaml` using **case-insensitive longest prefix** | `server_or_pod/*.yaml` contains `device: "ascend950"` as an **explicit reference** |
| Rule | `lower(soc_version).startswith(lower(spec.name))`; if multiple matches, select the one with the longest name; if still tied at the longest → fail-fast; if zero matches → fail-fast and list candidates | Look up directly by name; if not found → fail-fast |
| Covered production values | `ascend950` ↔ `Ascend950DT_95A1`, `Ascend950`; `ascend960` ↔ `Ascend960DT_961BY`, `Ascend960`; `ascend910b` ↔ `Ascend910B3` | Same as left (values provided by the device yaml) |

Case-insensitive prefix matching is chosen over the existing `strncmp(soc_version, "Ascend950", 9)` to cover both SKU-level and family-level granularity simultaneously, and to avoid the defect in `ParseDevType` (`topo_cluster_ir.cc:137-152`), which only recognizes `ascend950/ASCEND950/950` and therefore always returns `UNKNOWN` for real values (that function is deleted in S2).

**Five `strncmp` hardcoded locations to be removed after convergence**: `hccp_ccu_stub.cc:134,258,513`, `ccu_level1_res.cc:494-498`, `aclrt_new_stub.cc:47-52`, `hccl_task_sequential_execute_v2.cc:103-105` (changed to query `DeviceSpec.ccu_version`).

> ⚠ Although the device yaml contains a `soc.soc_version` field, **the Phase 0 loader completely ignores it** (decision (i)). Matching uses `name` only. See §4.7.4 for details.

#### 4.2.2 Schema (using A6/Ascend960 as the sample)

> The authoritative implementation is in `asset/cluster_model/config/device/device_spec_yaml_demo_ascend960.yaml` (with per-field hcomm reference line numbers and computed verification). The following is a condensed view.

```yaml
type: "device_spec"
spec_version: 1                       # schema version for consumer compatibility decisions
name: "ascend960"                     # = file name = spec identifier = Phase 0 matching key
description: "Ascend960(A6) device specification: address space partitioning and CCU resource layout"

soc:
  soc_version: "Ascend960"            # ⚠ Phase 0 loader [completely ignores this], see §4.7.4 decision (i)
  ccu_version: "CCU_V2"               # CCU_V1 | CCU_V2 | CCU_V3 (V3 reserved)
  die_num: 2

# ---------------- Global address space partitioning (chip-independent, A5/A6 identical field by field) ----------------
address_space:
  rank_base:        0x8000000000      # rank0 window start (= 512 GiB)
  rank_stride:      0x10000000000     # 1 TiB per rank
  rank_window_size: 0x10000000000     # must be ≤ rank_stride (validation 5)
  # Sorted by offset ascending, non-overlapping, fully covering [0, rank_window_size); **the same name can have multiple segments**
  partitions:
    - {name: "user_mem",  offset: 0x0,          size: 0xE00000000}   # segment 0:   0 ~   56 GiB
    - {name: "ccu",       offset: 0xE00000000,  size: 0x80000000}    #             56 ~   58 GiB (2 die × 1 GiB)
    - {name: "user_mem",  offset: 0xE80000000,  size: 0xF180000000}  # segment 1:  58 ~ 1024 GiB
  ccu:
    die_offsets: [0x0, 0x40000000]    # Explicitly listed, not derived from stride (D10); length must == soc.die_num
    die_size:    0x40000000           # 1 GiB address window per die
  # ★ No alloc_guard / user_mem_limit: available quota = Σ user_mem segments = 0xFF80000000 (1022 GiB),
  #   derived from partitions. rtMalloc avoids CCU through per-segment first-fit **structural guarantee** (§4.5.7).

# ---------------- CCU region layout (offsets relative to ccuDieBase) --------------------
ccu:
  # Authoritative resource space total size = backingSize (hcomm ccu_res_specs.h:24 CCU_V1_V2_RESOURCE_SIZE,
  # .cc:765 GetResourceSpaceSize()). A5/A6 both equal 72 MiB. **Do not** derive from max(region ∪ ms end).
  resource_space_size: 0x4800000

  # entry_size: byte stride per resource; count: hardware capacity (caps derived from this)
  # No backed field — see §4.5.2 and §6.2 O3 (closed)
  regions:
    - {name: "instruction", type: "INSTR",   offset: 0x0,       size: 0x100000, entry_size: 32, count: 32768}
    - {name: "xn",          type: "XN",      offset: 0x100000,  size: 0x40000,  entry_size: 8,  count: 16384}
    - {name: "cke",         type: "CKE",     offset: 0x140000,  size: 0x8000,   entry_size: 8,  count: 1024}
    - {name: "pfe",         type: "PFE",     offset: 0x148000,  size: 0x8000,   entry_size: 8,  count: 20}
    - {name: "channel",     type: "CHANNEL", offset: 0x150000,  size: 0x20000,  entry_size: 32, count: 1024}
    - {name: "jetty",       type: "JETTY",   offset: 0x170000,  size: 0x8000,   entry_size: 32, count: 128}
    - {name: "mission",     type: "MISSION", offset: 0x178000,  size: 0x8000,   entry_size: 64, count: 16}
    - {name: "loop",        type: "LOOP",    offset: 0x180000,  size: 0x8000,   entry_size: 32, count: 512}
  # A6 has no GSA register file → regions does not contain a GSA entry; GetCaps().gsa_num derives to 0.
  # ★ No independent gsa: key (avoids redundancy with regions; T6 is resolved).
  # ⚠ However, the A6 gsaOffset **bit field is still in use** (ccu_microcode_v1.h:630 CcuV2 section xmId
  #   "gsaOffset[52:21]"). Its semantics is the **address stride** for loop unrolling, not a register index
  #   (ccu_simulator.cc:47-65 → ccu_executor_base.cc:127-136). It must not be removed along with count=0.

  # MS layout: A5/A6 **geometry is identical** (4 CCUA × 2 MiB × interleave 8); only the base address differs
  ms:
    ms_size:    4096                  # hcomm ccu_microcode_v1.h:28 CCU_MS_SIZE
    ms_num:     1536
    ccua_num:   4                     # ccu_block_res_strategy.cc:23 CCU_V1_V2_CCUA_NUM
    interleave: 8                     # ccu_microcode_v1.h:29 CCU_MS_INTERLEAVE
    ccua_windows:                     # A6 base address 0x4000000; **A5 base address 0x0**
      - {index: 0, offset: 0x4000000, size: 0x200000}
      - {index: 1, offset: 0x4200000, size: 0x200000}
      - {index: 2, offset: 0x4400000, size: 0x200000}
      - {index: 3, offset: 0x4600000, size: 0x200000}

  reserved_sync_slots: 4              # L1 SYNC_SLOT_NUM semantics (ccu_level1_res.h:250)
  count_xn_num: 4                     # A6-specific: ccu_res_specs.h:59
                                      # CCU_V2_RESOURCE_TOTAL_CNT_XNS_NUM (0.5RTT 4 groups); A5 is 0
  # ★ No caps section: the 10-item capability specification can be 100% derived from
  #   regions[*].count / ms.ms_num, returned by DeviceSpec::GetCaps() (§4.2.3).
  # ★ No WQE / SQ buffer region: the actual location and offset have not been confirmed
  #   by hardware specification; not modeled in this phase (O12).

compute:                              # T10/T11 placeholder, not consumed this phase (O7)
  ai_core_num: 0
  vector_core_num: 0
  ai_cpu_num: 0
bandwidth:
  hbm_gbps: 4000.0
  hbm_utilization: 0.7
  default_port_gbps: 53.125
```

**A5/`ascend950.yaml` differences** (the `address_space` section is **field-by-field identical** to A6):

| Item | A5 | A6 | Basis |
|---|---|---|---|
| `ccu_version` | `CCU_V1` | `CCU_V2` | — |
| **CCUM/CCUA order** | CCUM is **after** CCUA (`CCU_V1_CCUM_OFFSET = 0x800000`) | CCUM is **before** CCUA (`CCU_V2_CCUM_OFFSET = 0x0`) | `ccu_res_specs.h:26-27` |
| `regions` | **9 entries**: INSTR@`0x800000`, **GSA@`0x900000`**, XN@`0x908000`, CKE@`0x910000`, PFE/CHANNEL/JETTY/MISSION/LOOP starting at `0x918000` with 0x8000 stride | **8 entries** (no GSA) | `ccu_res_specs.cc:372-373,499-500`; matches proxy L1 `V1_*_OFFSET` (`ccu_level1_res.cc:44-46`) one by one |
| Variable reservation | `CCU_RESOURCE_XN_V1_RESERVE_SIZE` = 32K = `0x8000` | `..._V2_...` = 256K = `0x40000` | `ccu_res_specs.h:55-56` |
| `ms.ccua_windows` base address | **`0x0`** (4 banks occupy `[0, 0x800000)`) | `0x4000000` | Derived from CCUM position |
| `count_xn_num` | 0 | 4 | `ccu_res_specs.h:59` |
| caps (derived values) | instr 0x8000 / mission 16 / loop **200** / gsa **3072** / xn **3072** / cke 1024 / ms 1536 / channel **128** / jetty 128 / pfe **16** | loop 512 / gsa 0 / xn 16384 / channel 1024 / pfe 20 | **10/10 match** with `hccp_ccu_stub.cc:69-96`(V1) / `:98-125`(V2) |
| `backingSize` | **72 MiB** (same as A6) | 72 MiB | `CCU_V1_V2_RESOURCE_SIZE` |

⚠ **A5 remaining TODO (T5)**: The offsets for PFE/CHANNEL/JETTY/MISSION/LOOP have **no definition found** in hcomm `ccu_res_specs.{h,cc}` (full repository grep for `0x148000`/`0x150000`/`0x170000`/`0x178000`/`0x180000` yields zero hits). The yaml uses proposed values following the "extend the hcomm V1 chain with 0x8000 reservation convention." The impact is limited: A5 instructions use **id-based addressing**, and the runner/checker do not perform `AddrToResId` reverse lookup, so these five items currently only affect the `GetCcuMemInfo` reported values and **do not affect backingSize** (which is fixed by `resource_space_size`).

> **Parsing note (correcting an erroneous statement in early versions)**: `0x`-prefixed integers are correctly parsed to integer values by both parsers — yaml-cpp `convert<T>::decode` internally calls `unsetf(ios::dec)` for base-0 semantics (`third_party/yaml-cpp/include/yaml-cpp/node/convert.h:180-190`), and PyYAML implements YAML 1.1 native support. **No `strtoull` special handling is needed**. The real pitfall is that all address/size fields **must be read as `uint64_t`** — `rank_stride` (0x10000000000), `user_mem segment 1.size` (0xF180000000), and `rank_base` (0x8000000000) are all ≥ 2^32. Using a 32-bit type causes `num_get` to set failbit and throw `TypedBadConversion`. The `device_spec` library should provide a unified `ReadU64(node, keyPath)` function and report the full key path on failure.

#### 4.2.3 Schema Design Trade-offs

| Decision | Choice | Rationale |
|---|---|---|
| `regions` uses an **explicit list** rather than fixed fields | List | A5 and A6 have different region sets (A5 has GSA, A6 does not); future chips may add or remove regions. A list is naturally extensible and avoids repeated schema changes |
| `offset` and `size` are separate; `stride` is not used for derivation | Explicitly provide all values | Fixes §2.3 defect 2 (`GetCcuMemInfo` mixes size and stride); `entry_size × count ≤ size` is validated by the loader |
| **No `caps` section**; capability specification is derived by `DeviceSpec::GetCaps()` from `regions[*].count` / `ms.ms_num` | Write a `caps` section in yaml alongside, then have the loader assert both are equal | Verified item by item: `instruction/xn/cke/pfe/channel/jetty/mission/loop` 8 items == corresponding `regions[*].count`, `ms_num` == `ms.ms_num`, `gsa_num` == `regions[GSA].count` (0 if absent). **10/10 are derivable**, and A5/A6 each match the current values in `hccp_ccu_stub.cc` V1/V2 at 10/10. The decoupling of "reported specification vs array specification" is already expressed by the separation of region `size` (address space slots) and `count` (capacity) — for example, A6 Variable has `size=0x40000` (32768 slots) while `count=16384`. Keeping a separate `caps` section would just write the same numbers twice and add one more anti-drift validation, violating D10 single authority |
| **No independent `gsa:` key**; GSA is expressed by `regions[type=="GSA"]` | A5 writes a region + an independent `gsa: {count:3072}`; A6 writes `gsa: {count:0}` | The independent key and `regions[GSA].count` are two copies of the same number (confirmed redundant on A5). When A6 has no GSA, "independent key count:0" and "no such entry in regions" express the same fact in two forms, creating schema inconsistency. Unified to "if present, one region entry; if absent, do not write." `GetCaps().gsa_num` derives as "entry absent means 0" |
| **user_mem uses multi-segment `partitions` explicitly covering the full range** (option A), not "complement derivation" (option B) | Option B: delete `partitions`, `ccu.window_offset` is self-describing, user_mem is computed as the complement by the library | Option A makes the window partitioning auditable at a glance in the yaml; adding a new reserved region (AIV UB and others) in the future means **manually adding one partition entry**, without introducing an `other_reserved` extension field (Q1 decision). The only weakness of option A (possible incomplete coverage or overlap) is fully covered by validation 5 coverage assertion |
| **No `alloc_guard` / `user_mem_limit`** | Keep `user_mem_limit: 0xE00000000` as a hard upper bound | That value is **the compensating upper bound for when the linear bump allocator cannot skip the CCU gap**. It becomes invalid after per-segment first-fit is implemented. Keeping it would artificially reduce the available quota from 1022 GiB to 56 GiB (wasting 94.3%), and it contradicts the sum of the two user_mem segments. The available quota is now derived from `Σ user_mem segments` |
| **No `regions[*].backed`** | Boolean flag marking regions that do not need backing | The benefit is always 0 (regions occupy only 2.13% of backingSize; tmpfs sparse files charge by touched pages). It also contradicts premises C-1/C-3 of approach C. See §4.5.2 and §6.2 O3 (closed) |
| **No WQE / SQ buffer region** | Add an entry `offset 0x1000000, size 32 MiB` based on hcomm `CCU_V1_WQE_BASIC_BLOCK_OFFSET` | That offset is reverse-engineered from hcomm code (`ccu_res_specs.h:32` + `ccu_jetty_ctx_mgr.cc:149,172,222`), **not confirmed by hardware specification**. Writing it into the configuration would be mistaken as authoritative. Removal is safe: `backingSize` takes the authoritative value of 72 MiB, which fully covers `[0x1000000, 0x3000000)`. hcomm writes fall within the backing, and the simulator does not interpret them. Tracked in O12 |
| Addresses use `0x` hexadecimal scalars | Yes | Consistent with real hardware documentation for easy manual comparison. Both yaml-cpp (`convert<T>::decode` internally uses `unsetf(ios::dec)` for base-0) and PyYAML (YAML 1.1) **parse correctly** without `strtoull` special handling; however **they must be read as `uint64_t`** (see parsing note at end of §4.2.2) |
| `compute`/`bandwidth` are placeholder-only this phase | Yes | Converging `light_performance/sim_config.json` and the unlanded `TaskSchedulerDevice` is an independent benefit that does not block the main track (O7) |
| Add `spec_version` | Yes | The same repository has a `version` precedent in `topo.json` (`topo_cluster_parser.cc:218-309` reads it). The device yaml has 4 types of C++ consumers plus Python plugins. Without a version number, adding fields in the future makes compatibility decisions impossible |

#### 4.2.4 Device Yaml Description Items Checklist (TODO Tracking)

**Convention**: For A5, A6, and all future chip types, the device specification — address space partitioning, device base address and die offsets, CCU region layout, **MS layout**, resource capacity and reported specification — **all use the device yaml as the single authority** (D10). The following table enumerates the items that each `asset/cluster_model/config/device/ascend*.yaml` must describe and the current status. When delivering a new chip type, verify each item. **Any missing item causes a startup failure (loader fail-fast); falling back to hardcoded values in code is prohibited**.

> **Status note**: The A5 T5/T6/T7 items were originally TODOs. They have now been **verified** against the authoritative hcomm source code (`<hcomm>/src/base_comm/resources/ccu/ccu_device/ccu_res_specs.{h,cc}`, `ccu_block_res_strategy.cc:23`, `ccu_microcode/ccu_microcode_v1.h:28-29`). Both reference implementations are landed and pass all loader validations.

| ID | Yaml Section | Required Description Item | ascend960(A6) | ascend950(A5) | Status / Source |
|---|---|---|---|---|---|
| T1 | `soc` | `soc_version` (single value, **Phase 0 loader ignores**) / `ccu_version` / `die_num` | ✅ `"Ascend960"`, `CCU_V2`, 2 | ✅ `"Ascend950DT_95A1"`, `CCU_V1`, 2 | Phase 0 matching uses `name` longest prefix (§4.2.1); `soc_version` normalization deferred to **O8**. Field semantics see §4.7.4 decision (i) |
| T2 | `address_space` | `rank_base` / `rank_stride` / `rank_window_size` | ✅ Defined | ✅ **Field-by-field identical** | Same source as `AllocVirMem` current `deviceBase/deviceAddrStep` (§2.7); cross-file consistency asserted by UT |
| T3 | `address_space.partitions` | Multiple `user_mem` segments + one `ccu` segment, **fully covering** the window | ✅ 3 segments (56G + 2G + 966G) | ✅ Same partitioning | Validation 5: coverage + unique ccu segment + `userMemTotal = 0xFF80000000` (1022 GiB) > 0. **No more `alloc_guard`** (§4.2.3) |
| T4 | `address_space.ccu` | `die_offsets` (per-die explicit offsets) + `die_size` | ✅ `[0x0, 0x40000000]` / `0x40000000` | ✅ Same | D10: Explicitly listed, not derived from stride; validation 4 ensures die windows do not overlap |
| T5 | `ccu.regions` | Region table: name/type/offset/size/entry_size/count (**no `backed`**, see §4.5.2) | ✅ **8 entries** (no GSA) | ✅ **9 entries** (including GSA@`0x900000`); ⚠ **Remaining TODO**: PFE/CHANNEL/JETTY/MISSION/LOOP five offsets have no hcomm definition; using 0x8000 stride proposed values | Verified against `ccu_res_specs.cc:372-373,499-500`; A5 five remaining items only affect `GetCcuMemInfo` reporting, **do not affect backingSize** (A5 uses id-based addressing, no `AddrToResId` reverse lookup) |
| T6 | ~~`ccu.gsa`~~ → `regions[GSA]` | GSA capacity **expressed through regions**, no independent key | ✅ No GSA entry → `GetCaps().gsa_num` derives to 0 | ✅ One GSA region (count 3072) | Resolved. ⚠ The A6 `gsaOffset` **bit field** is still used in V2 microcode (`ccu_microcode_v1.h:630`). Its semantics is an address stride, not a register index. It must not be removed along with count=0 |
| T7 | `ccu.ms` | **MS layout**: `ms_size`/`ms_num`/`ccua_num`/`interleave`/`ccua_windows` | ✅ Base address `0x4000000` | ✅ Base address **`0x0`** | **Verified: A5/A6 geometry is identical** (4 CCUA × 2 MiB × interleave 8, `ccu_block_res_strategy.cc:23` + `ccu_microcode_v1.h:28-29`); only the base address differs. Based on this, the level1 `CCU_MS_BASE = 0x08000000` (`ccu_kernel_sim.cc:65`) is **confirmed incorrect** and is deleted along with §4.5.6 |
| T8 | `ccu.reserved_sync_slots` | Reserved slot count for cross-rank synchronization | ✅ 4 | ✅ 4 | Follows L1 `SYNC_SLOT_NUM=4` (`ccu_level1_res.h:250`). ⚠ L1 actually has per-type three watermarks (`:293-295` xn/gsa/cke each 4). This scalar has the semantics "applies uniformly to all three types." If per-type configuration is needed in the future, it must change to a map form |
| T9 | ~~`caps`~~ → `DeviceSpec::GetCaps()` | Capability specification reported to hcomm **is not in the yaml**; derived by the library | ✅ Derived values **10/10 match** with `hccp_ccu_stub.cc:98-125` | ✅ **10/10 match** with `:69-96` | Eliminates "same number maintained in two places"; original validation 6 is canceled. Derived values also serve as **cross-validation** for T5/T7: a mismatch indicates an error in regions |
| T10 | `compute` | `ai_core_num` / `vector_core_num` / `ai_cpu_num` | ⏸ Placeholder (0, not consumed) | ⏸ Placeholder | O7: Land `TaskSchedulerDevice`/`ComputeDie` table in phase 2 |
| T11 | `bandwidth` | `hbm_gbps` / `hbm_utilization` / `default_port_gbps` | ⏸ Placeholder | ⏸ Placeholder (0.0) | O7: Converge `light_performance/sim_config.json` in phase 2 |
| T12 | Future chips | New soc files / `die_num>2` / new region additions/removals / **CCU_V3** | — | — | O4: This phase only supports `die_num` 1/2. **V3 extension notes**: `CCU_V3_RESOURCE_SIZE=128 MiB`, CCUA 4→8, `XN_TOTAL_NUM_LIMIT=16K`, `CKE_TOTAL_NUM_LIMIT=2048`, private/shared dual regions + per-block start id (`CcuResSpecInfo`, `ccu_res_specs.h:95-135`). The existing `regions` list + `ms.ccua_num` can express CCUA and capacity changes, but the block semantics requires new `ccu.blocks`/`ccu.shared` sections (S1 reserved; not filled, not consumed). `ccu_version` enumeration must include `CCU_V3` |
| **T13** | ~~`ccu.regions[WQE]`~~ | **WQE / SQ buffer region**: location and offset **not confirmed by hardware specification**, **not modeled** this phase | ➖ Not modeled | ➖ Not modeled | ⚠ **TODO (O12)**: hcomm does have computation paths — `ccu_res_specs.h:32` `CCU_V1_WQE_BASIC_BLOCK_OFFSET = 0x1000000`; `ccu_jetty_ctx_mgr.cc:149` `GetSqeBuffVa` → `:172` `sqeBuffVa = ccuResBaseVa + 0x1000000 + jettyCtxId × CCU_SQ_BUFFER_SIZE(256K)`; `:222` triggered by `jettyType == CCUM_CACHED_JETTY`, **not version-gated** (V1/V2 shared). After confirmation, manually add one region (`offset 0x1000000, size 128×256K = 32 MiB, entry_size 0x40000, count 128`) and recheck validation 8 |

**Out-of-scope list** (drawing the boundary of D10 "single authority") — the following CCU-related address/id spaces **are not governed by the device yaml**, to prevent repeated questions about "why this constant is not in the yaml":

| Item | Location | Nature |
|---|---|---|
| `CCU_CHANNEL_NOTIFY_ID_BASE = 0xD000000000000000` | `ccu_kernel_sim.cc:61` | **id encoding** base address (isolated from event bit id and DB notifyId space); not a memory address |
| notify id 17-bit space | `sim_capacity_limits.h:86-88` | id space partitioning |
| AICPU jetty hardware id segmentation (CCU `[1024,1151]` and others) | `sim_capacity_limits.h:133-147` | Hardware id allocation, corresponds to `CCU_START_TA_JETTY_ID=1024` |
| AIV comm buffer layout (65 MiB fixed offset) | `sim_common_defs.h:144-157` | AIV side, not CCU; if address space reservation is needed in the future, manually add a `partitions` entry (Q1) |
| Host-side shared memory naming / root directory | `store_sim_resource_root.cc:29-42` | Host resource, not device address space |
| **WQE / SQ buffer region** | See T13 / O12 | Location not confirmed; not modeled this phase (but covered by the 72 MiB backing) |

### 4.3 Configuration Loading and Distribution

#### 4.3.1 Loading

- **Path resolution**: `InstallPath::ResolveToInstallRoot("config/device/<name>.yaml")` (`include/sim_common_api.h:21`, `src/utils/sim_common_api.cc:41-78`). The root is determined by `HCCL_VM_INSTALL_ROOT` or by searching upward for `config/log_config.yaml`.
- **New library** `src/device_spec/` (static library, `HCCL_VM_MODULE="DEVSPEC"`):
  ```cpp
  namespace sim::devspec {

  enum class CcuResType : uint16_t {   // Replaces runner/checker's respective CcuComponerntType
      UNKNOWN = 0, INSTR, CCUM, GSA, XN, CKE, PFE, CHANNEL, JETTY, MISSION, LOOP, WQE, MS
  };                                   // ★ Removes _A6 suffix, fixes "Componernt" spelling, adds INSTR/CCUM/GSA/WQE/MS
  enum class CcuVersion : uint8_t { CCU_V1 = 0, CCU_V2 = 1, CCU_V3 = 2 };

  struct RegionSpec { std::string name; CcuResType type; uint64_t offset, size; uint32_t entrySize, count; };
  struct MsWindow   { uint32_t index; uint64_t offset, size; };
  struct MsSpec     { uint32_t msSize, msNum, ccuaNum, interleave; std::vector<MsWindow> windows; };
  struct Partition  { std::string name; uint64_t offset, size; };
  struct AddrSpace  {
      uint64_t rankBase, rankStride, rankWindowSize;
      std::vector<Partition> partitions;
      std::vector<uint64_t>  ccuDieOffsets;  uint64_t ccuDieSize;
      // Derived (computed by loader)
      uint64_t ccuWindowOffset, ccuWindowSize;
      std::vector<Partition> userMemSegs;
      uint64_t userMemTotal;
  };
  struct CcuCaps { uint32_t instructionNum, missionNum, loopEngineNum, gsaNum, xnNum,
                              ckeNum, msNum, channelNum, jettyNum, pfeNum; };

  struct DeviceSpec {
      uint32_t    specVersion{0};
      std::string name;
      std::string socVersion;       // ★ Phase 0 loader does not read this key (decision (i)); always empty
      CcuVersion  ccuVersion{CcuVersion::CCU_V1};
      uint32_t    dieNum{2};
      AddrSpace   addr;
      std::vector<RegionSpec> regions;
      MsSpec      ms;
      uint32_t    reservedSyncSlots{4};
      uint32_t    countXnNum{0};
      uint64_t    resourceSpaceSize{0};

      CcuCaps           GetCaps() const;
      uint64_t          BackingSize() const;
      const RegionSpec* FindRegion(CcuResType) const;
      bool              HasGsa() const;
      bool              SupportsAddrDecode() const;
  };

  class DeviceSpecRegistry {
  public:
      static DeviceSpecRegistry& GetInstance();
      bool LoadAll();
      bool LoadFromFile(const std::string& absPath);
      const DeviceSpec* FindByName(const std::string& specName) const;
      const DeviceSpec* FindBySocVersion(const std::string& socVersion) const;
      const DeviceSpec* FindByDeviceKey(uint64_t deviceKey) const;
      std::vector<const DeviceSpec*> All() const;
      std::string LastError() const;
  };

  class RankAddressSpace {
  public:
      explicit RankAddressSpace(const AddrSpace& a);
      uint64_t RankBase(uint32_t rankIdx) const;
      uint64_t RankWindowSize() const;
      uint32_t MaxRankSlots() const;
      uint64_t CcuWindowOffset() const;
      uint64_t CcuWindowSize() const;
      uint64_t CcuDieBase(uint32_t rankIdx, uint32_t dieId) const;
      bool     IsInCcuWindow(uint64_t va, uint32_t& rankIdx, uint32_t& dieId) const;
      size_t   UserMemSegCount() const;
      uint64_t UserMemSegOffset(size_t i) const;
      uint64_t UserMemSegSize(size_t i) const;
      uint64_t UserMemTotal() const;
      bool     IsUserMemVa(uint64_t va, uint32_t rankIdx) const;
      static uint32_t RankIdxOfDeviceKey(uint64_t deviceKey);
  };
  // ★ Does not provide UserMemVaFromLinear(rankIdx, linear): a single global linear
  //   watermark would cause allocation blocks to cross the CCU gap (counterexample §4.5.7).

  class CcuAddressSpace {
  public:
      explicit CcuAddressSpace(const DeviceSpec& spec);
      bool AddrToResId(uint64_t dieBase, uint64_t addr, CcuResType& type, uint64_t& id) const;
      bool ResIdToAddr(uint64_t dieBase, CcuResType type, uint64_t id, uint64_t& addr) const;
      bool MsAddrToId (uint64_t dieBase, uint64_t addr, uint32_t& msId) const;
      bool MsIdToAddr (uint64_t dieBase, uint32_t msId, uint64_t& addr) const;
      CcuResType TypeByOffset(uint64_t offset) const;
      uint32_t   EntrySize(CcuResType) const;
      uint64_t   RegionOffset(CcuResType) const;
      // ★ All methods include full boundary checks
  };
  }
  ```
- **Loader validations** (fail-fast; configuration errors must not allow startup; both reference implementations have passed computed verification):
  1. Regions do not overlap, and `[offset, offset+size)` ⊂ `[0, die_size)`. ★ **The scope must be a combined check of `regions[] ∪ ms.ccua_windows[]`** — MS is not in `regions[]`; checking separately would miss "a region overlapping with an MS window."
  2. `entry_size × count ≤ size` (opaque blocks with `entry_size == 0` are skipped).
  3. MS **per-window** coverage (★ **do not use** the summation formula `ms_num ≤ Σwindow slots` — the summation formula misses overflow under heterogeneous windows: when window capacities are `[256,512,512,512]`, `Σ=1792 ≥ 1536` passes, but `localIdx_max=383 > 255` causes window 0 overflow):
     `localIdx_max = ((ms_num-1)/(ccua_num×interleave))×interleave + (interleave-1)`, requires `∀i: localIdx_max × ms_size < ccua_windows[i].size`;
     additionally requires `ccua_num == len(ccua_windows)`, `ccua_windows[i].index == i` (ordered, no duplicates), windows do not overlap.
  4. `len(ccu.die_offsets) == soc.die_num`; die windows do not overlap (after sorting, `die_offsets[i] + die_size ≤ die_offsets[i+1]`).
  5. **Partitions coverage and user_mem derivation**: sorted by offset ascending, non-overlapping, **fully covering** `[0, rank_window_size)` (first segment `offset == 0`; last segment `offset+size == rank_window_size`; adjacent segments connect end-to-start); exactly one `name=="ccu"` segment; `name` only allows `user_mem`/`ccu` (current phase whitelist); `partitions[ccu].offset + max(die_offsets) + die_size ≤ partitions[ccu].offset + partitions[ccu].size`; `userMemTotal = Σ user_mem segments > 0` and segment count ≥ 1; `rank_window_size ≤ rank_stride`.
     ★ No more `alloc_guard.user_mem_limit == partitions[user_mem].size` (that field is deleted, §4.2.3).
  6. ~~`caps.*` matches corresponding `regions[*].count`~~ — **Canceled**: `caps` is not in the yaml; derived by `DeviceSpec::GetCaps()` (§4.2.3). No second copy exists (number retained to avoid breaking cross-references).
  7. `rank_base + GetMaxRanks() × rank_stride` bit width check → exceeding 47-bit triggers **WARN** (see §6.1 R4). Note: `GetMaxRanks()` (`sim_capacity_limits.h:30-45`) can be overridden by `HCCL_VM_MAX_RANKS`, so this validation depends on the runtime environment and must execute during the `start` phase, not within `LoadAll()`.
  8. `max(regions ∪ ms end) ≤ ccu.resource_space_size ≤ ccu.die_size` (computed: A5 `0x940000 ≤ 0x4800000`, 62.75 MiB headroom; A6 `0x4800000 ≤ 0x4800000`, equality holds).
  9. Scan all `soc_version` values in `config/cluster/*.yaml` + `config/server_or_pod/*.yaml`; each must **uniquely match** one spec (case-insensitive longest prefix). Reverse: each spec `name` must be matched by at least one `soc_version` (prevent spec decay). Note: the `soc.soc_version` field within the spec **does not participate** in this validation (decision (i)).
- **Dependencies**: `device_spec` depends only on `log` + `yaml-cpp`. **Both checker and runner already link `yaml-cpp::yaml-cpp`** (`src/plugin/checker/CMakeLists.txt:58`, `src/plugin/runner/CMakeLists.txt:62`), so no new third-party dependency is introduced. The library skeleton follows `src/topo/CMakeLists.txt` (`add_library(... STATIC "")` + `target_sources` + `hcclvm_apply_common_flags` + `HCCL_VM_MODULE`).

#### 4.3.2 Distribution (Three Pathways, Covering Three Consumer Types)

| Pathway | Carrier | Consumer | Description |
|---|---|---|---|
| **T1 Read configuration directly** | `config/device/*.yaml` | hccl-vm main process, runner, checker, C++ plugins, third parties | All parties can access the install root; simplest approach with no format change risk. **Single authoritative source** |
| **T2 Persist to DB** | `Ccu.resource_addr` (already exists, `FillCcu()` fills it) + **4 new `DeviceSpec*` tables** (see §4.8) | Proxy stub, runner/checker (prefer DB when available), Python plugins, third parties | `GetCcuFromDeviceByDieId` (`db_sim_runner_common.cc:768`) already reads the `Ccu` row — **zero new query code**; `light_performance/db_reader.py:25,566` already uses `sqlite3.connect(mode=ro)` to connect to `hccl_sim.db`; just add read access |
| **T3 On-disk file** | `ModelInfoCommInner.ccu0/1_resource_base_addr` (already exists) | Offline checker, third-party tools | Change from "placeholder" to "real value" (`store_dump_shm_data.cc:497-498`); consumers change to reading instead of hardcoding |

**The single authoritative source for specification/layout information is the yaml (T1)**: The DB and on-disk files are both **runtime artifacts** and do not serve as specification sources. After loading, **cross-validation** must be performed: if the DB `Ccu.resource_addr` or `DeviceSpec*` table content does not match the locally derived yaml values → log ERROR and refuse to start (to detect "configuration changed but topology was not regenerated"). Third parties that read the DB directly must handle the "table is empty (`start` was never run)" case and fall back to reading yaml.

**Unified fallback chain for consumers** (mandatory requirement, due to the mock-comm gap in §4.8.7):
```
Query DB DeviceSpec → if found, use it; if not found (table empty / start not run) → DeviceSpecRegistry::FindBySocVersion() reads yaml
```

#### 4.3.3 Installation and Packaging

In `CMakeLists.txt`, within `config_hcclvm_install()` (`:154-204`), add a fourth section following the `server_or_pod` pattern (`:182-192`):

```cmake
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/asset/cluster_model/config/device"
        DESTINATION "config"
        FILE_PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
        DIRECTORY_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
```

Supporting changes:
- `test/topo/CMakeLists.txt:21-22`, `test/cmd/CMakeLists.txt:19-20` already copy the entire `asset/cluster_model/config/` directory, so the `device/` directory is **automatically covered**.
- `test/topo/sqlite/topo_golden_test.cc:399-409` uses explicit symlinks; if the generation script needs to read `device/`, add `linkConfig("device")`.
- `asset/generate_cluster_topo.sh:100-113` adds `DEVICE_CONFIG_DIR` (only needed if the generation script must validate soc-to-device configuration existence).

### 4.4 Address Space Partitioning Design

#### 4.4.1 Normalized Formulas

```
rankBase(r)          = RANK_BASE + r × RANK_STRIDE
ccuDieBase(r, d)     = rankBase(r) + CCU_WINDOW_OFFSET + CCU_DIE_OFFSET[d]
resAddr(r, d, t, id) = ccuDieBase(r, d) + regionOffset(t) + id × entrySize(t)
msAddr(r, d, id)     = ccuDieBase(r, d) + ccuaOffset(id) + localIdx(id) × ms_size   // see 4.4.3 interleave
userMemVa(r, seg, off) = rankBase(r) + userMemSegs[seg].offset + off               // see §4.5.7 segmented allocation
```

Where `RANK_BASE = 0x8000000000`, `RANK_STRIDE = 0x10000000000`, `CCU_WINDOW_OFFSET = 0xE00000000`, `CCU_DIE_OFFSET = [0x0, 0x40000000]` (that is, the yaml `ccu.die_offsets`, explicitly listed, not derived from stride; see D10); `userMemSegs = [(0x0, 0xE00000000), (0xE80000000, 0xF180000000)]`, derived from `partitions` entries with `name=="user_mem"` sorted by offset ascending.

#### 4.4.2 Consistency Verification with Requirements 3.1/3.2

| Item | Required Value | Formula Derivation | Check |
|---|---|---|---|
| rank0 base | `0x8000000000` | `0x8000000000 + 0` | ✅ |
| rank1 base | `0x18000000000` | `0x8000000000 + 0x10000000000` | ✅ |
| rank0 die0 ccu | `0x8e00000000` | `0x8000000000 + 0xE00000000` | ✅ |
| rank0 die1 ccu | `0x8e40000000` | `0x8e00000000 + 0x40000000` | ✅ |
| rank1 die0 ccu | `0x18e00000000` | `0x18000000000 + 0xE00000000` | ✅ |
| ccu size | 1 GiB | `0x40000000` | ✅ |
| rank size | 1 TiB | `0x10000000000` | ✅ |
| Requirement 3.3 (rtMalloc does not overlap CCU) | — | **Structurally guaranteed by segmented first-fit**, not dependent on any quota constant | ✅ See §4.5.7 |

**Conclusion: The requirement approach is fully co-sourced with the existing `AllocVirMem` `deviceBase`/`deviceAddrStep`. This is "making the existing implicit convention explicit + opening a CCU window at 56 GiB," not a ground-up rewrite.** This is the most important feasibility basis for this design.

Rank window internal layout:

```
rank r window (1 TiB)
0x000000000 ┌─────────────────────────────────────────────┐
            │  user_mem segment 0 (rtMalloc allocatable)  56 GiB     │
0xE00000000 ├─────────────────────────────────────────────┤
            │  CCU die0                        1 GiB      │  ← rtMalloc never returns
0xE40000000 ├─────────────────────────────────────────────┤
            │  CCU die1                        1 GiB      │  ← rtMalloc never returns
0xE80000000 ├─────────────────────────────────────────────┤
            │  user_mem segment 1 (rtMalloc allocatable) 966 GiB     │
0x10000000000 └───────────────────────────────────────────┘
              user_mem total = 0xFF80000000 = 1022 GiB (1 TiB − 2 GiB CCU)
```

> ★ **The CCU window position is locked at 56 GiB**: Third-party components have hard requirements on the CCU base address. It cannot be moved to the end of the window to achieve a single user_mem segment (moving to the end would eliminate the need for segmented jumps and simplify the allocator, but it would change the external commitment of `0x8e00000000`/`0x8e40000000`). If the base address changes in the future, this section and the §4.5.7 allocator must be re-adapted.

Actual usage range within a die (A6):

```
ccuDieBase + 0x0000000  instruction   1 MiB
           + 0x0100000  XN          256 KiB   (16384 × 8B = 128 KiB effective)
           + 0x0140000  CKE          32 KiB   (1024 × 8B = 8 KiB effective)
           + 0x0148000  PFE          32 KiB
           + 0x0150000  CHANNEL     128 KiB   (1024 × 32B = 32 KiB effective)
           + 0x0170000  JETTY        32 KiB
           + 0x0178000  MISSION      32 KiB
           + 0x0180000  LOOP         32 KiB
           + 0x4000000  MS CCUA0      2 MiB ┐
           + 0x4200000  MS CCUA1      2 MiB │ Total 8 MiB
           + 0x4400000  MS CCUA2      2 MiB │
           + 0x4600000  MS CCUA3      2 MiB ┘
           + 0x4800000  ── Actual usage upper bound ≈ 72 MiB ≪ 1 GiB ✅
```

#### 4.4.3 MS Interleave Address-to-id Formula

Follows existing semantics (`ccu_resource_manager.cc:398-423`, same in checker), but **all parameters come from configuration**:

```
addr → id:
  off       = addr - ccuDieBase
  find ccuaIndex such that ccuaWindows[i].offset ≤ off < offset + size
  localIdx  = (off - ccuaWindows[i].offset) / ms_size
  groupIdx  = localIdx / interleave
  posInGrp  = localIdx % interleave
  msId      = groupIdx × (interleave × ccua_num) + ccuaIndex × interleave + posInGrp

id → addr:
  groupIdx  = msId / (interleave × ccua_num)
  ccuaIndex = (msId % (interleave × ccua_num)) / interleave
  posInGrp  = msId % interleave
  localIdx  = groupIdx × interleave + posInGrp
  addr      = ccuDieBase + ccuaWindows[ccuaIndex].offset + localIdx × ms_size
```

Coverage (per-window; both reference implementations have verified mutual invertibility by traversing `msId ∈ [0,1536)`):
`localIdx_max = ((1536-1)/(4×8))×8 + 7 = 383`, slots per window `0x200000/4096 = 512` → `383 ≤ 511` ✅

> ★ **The A5/A6 MS geometry is identical** (verified against hcomm): `ccua_num = 4` (`ccu_block_res_strategy.cc:23` `CCU_V1_V2_CCUA_NUM`; `ccu_res_specs.h:28` note "V3 CCUA count, V1/V2 have 4"), `ms_size = 4096` and `interleave = 8` (`ccu_microcode_v1.h:28-29`, **V1 header file**), 2 MiB per bank. **Only the base address differs**: A5 = `0x0` (CCUM is after CCUA), A6 = `0x4000000` (CCUM is before CCUA).
>
> ⇒ **Conclusion**: The proxy level1 `CCU_MS_BASE = 0x08000000` (128 MiB, `ccu_kernel_sim.cc:65`) is **confirmed incorrect** — it exceeds both the V1 CCUA range `[0, 0x800000)` and the entire 72 MiB resource space. It appears to be `CCU_V1_CCUM_OFFSET = 0x800000` with an extra zero. This "5th MS address space" is deleted along with §4.5.6.

> ⚠ **Existing defect requiring correction (applies to both A5 and A6)**: `GetMSIdByAddr` performs 4-bank × 8-way **de-interleaving**, while `GetMsAddr` (`ccu_resource_manager.cc:350-362`) and the checker `GenSliceFromMs` (`ccu_task_common_v3.cc:1071-1080`) both use **flat** `msId × 4096` — the same msId has **inconsistent** semantics between the "address reverse lookup" and "direct addressing" paths. Additionally: `GetMSIdByAddr` **has no callers** on the runner side (only the checker uses it); `addr → id` and `id → addr` have only unidirectional implementations.
>
> After S2 unification, the following must also be changed: the checker MS slice offset, the semantic check `MsKey` bucketing (`task_graph_semantic_check_v3.cc:47-48` buckets by `address/4096`). This item and the outstanding confirmation item in `docs/ccu_resource_statistics_design.md` §3.4 ("whether msId is a die-global index or a bank-local index") **are the same problem** and should be resolved together in S0 (see the A9 coordination item in §4.7.6).
>
> Semantic convention: `GetMsAddr` returns "the **host pointer** within the MS pool," and `MsIdToAddr` returns "the **device VA**." The two are linked through the materialized region base (§4.5.4 `CcuDieView`).

#### 4.4.4 Rank Index Specification (Must Be Explicitly Defined)

Currently `AllocVirMem` uses `devPhyId - 1` as the window index, where `devPhyId` = **Device table primary key** (starting from 1). Meanwhile the hcomm rankId comes from the ranktable, and the two are not guaranteed to be equal.

**Design decision**: Introduce `RankIndexResolver` with the explicit definition

```
rankIdx := Device.physical_id - 1     // Fully consistent with existing AllocVirMem, ensuring no allocated addresses change
```

And the requirements:
- The CCU window index, the rtMalloc window index, and `VirtualMemBlock.rank_id` **must use the same rankIdx**.
- Provide `rankIdx ↔ hcclRankId ↔ deviceKey` three-way mapping queries (data comes from the `Communicator` table; `store_dump_shm_data.cc` already constructs a `deviceToRank` mapping).
- During the `InitClusterStaticTopoData` phase, perform a one-time **uniqueness assertion**: `physical_id` must be unique and contiguous across all Device entries; otherwise rank windows would overlap.

> If the future requirement is "windows indexed by hcclRankId," that is a breaking change (it would change all already-allocated VAs) and requires a separate project.

### 4.5 CCU Memory Materialization

This is the core solution for P1/P2/P3.

#### 4.5.1 Approach: Reuse Existing Shared Memory + DB Registration Mechanism

**Do not invent new IPC.** Make CCU memory follow the exact same pathway as on-chip memory buffers:

```
CcuMemoryService::EnsureCcuDieMemory(deviceKey, dieId)
  1. Query DB: Ccu(device_id, die_id).resource_addr != 0 ? → already created, use Acquire
  2. rankIdx = Device.physical_id - 1
     dieBase = RankAddressSpace.CcuDieBase(rankIdx, dieId)
  3. memName = "CCU_R<rankIdx>_D<dieId>"
     hostPtr = MemoryManager::AllocMemByName(memName, backingSize)   // → /dev/shm/$HCCL_VM_RESOURCE_ROOT/
  4. MapDevPtrHostPtr(dieBase, hostPtr)                              // This process VA→host cache
  5. Insert PhyMemBlock{device_id, name=memName, size=backingSize, ref_count=1}
     Insert VirtualMemBlock{start_ptr = dev_mapped_ptr = dieBase,
                            dev_mapped_end_ptr = dieBase + backingSize,
                            rank_id = rankIdx, device_id = deviceKey,
                            phy_mem_id, owner_pid, src_type = VIR_MEM_TYPE_CCU}
  6. UPDATE Ccu SET resource_addr = dieBase WHERE id = ccuId
```

**Benefits** (all are existing capabilities, zero new code):
- On the runner side, `sim::AcquireDevPtrInNoHostProcess(ccuVa)` (`store_sim_shm_memory_common.cc:22`) directly queries by `VirtualMemBlock.start_ptr` range and hits → returns the host pointer.
- On the proxy side, `GetRealPtrByAddr(ccuVa)` (`aclrt_memory_stub.cc:129`) directly hits by `dev_mapped_ptr` range → **`LoadMicrocodeInstructionStub` no longer requires any special handling**.
- Cross-process sharing is guaranteed by `HCCL_VM_RESOURCE_ROOT` (the main process calls setenv, and the entire chain inherits it).

> **⚠ Hard constraint 1: CCU shared memory must use `MemoryManager::AllocMemByName`; it is prohibited to use `DeviceMemoryManager::AllocPhyMem/AcquirePhyMem`.**
> The latter embeds `CommPoolPolicy` (`store_sim_device_memory_manager.cc:72-90`): `kBigBlockThreshold = 200MB`, `kPoolSize = 4GB` (`include/store/store_sim_comm_pool_policy.h:19-21`). **In check-only mode, blocks with size ≥ 200MB are redirected by `ShouldRedirect` to the rank-shared `HcclCommPool`**. If CCU memory is redirected, the CCU state of all ranks aliases to the same pool → **cross-rank mutual overwrite**, and it only reproduces in check-only mode (which is the checker commonly used mode), making debugging extremely costly.
> The current headroom is sufficient but **the constraint must be hardcoded**: A5/A6 `backingSize` is both **72 MiB** (`resource_space_size`, §4.5.2), with 128 MiB headroom from the 200MB threshold.
> ⚠ This headroom **depends on two premises**; if either is broken, the risk immediately returns: ① `backingSize` takes the authoritative value rather than `max(regions ∪ ms)` (D10-supplement); ② O12 confirms WQE and raises `resource_space_size` to ≥ 200MB.
> (An early version estimated "A5 = 192 MiB based on level1 `CCU_MS_BASE=0x08000000`," with only 8 MiB from the threshold — that MS base address has been **confirmed incorrect** in §4.4.3, and the estimate is voided. R1b is downgraded from "high" to "low.")

> **⚠ Hard constraint 2: `GetRealPtrByAddr` filters by `device_id`, so it only works for "CCU addresses belonging to this rank."**
> Its query condition includes `Eq(&VirtualMemBlock::device_id, GetCurrDeviceKey())` (`aclrt_memory_stub.cc:133-138`), and the CCU row `device_id` is the **owning rank**. Therefore, resolving a **remote rank** CCU VA on the proxy side always returns nullptr. The runner side is unaffected (`AcquireDevPtrInNoHostProcess` query **does not filter by device**, `store_sim_shm_memory_common.cc:28-31`, cross-rank access works normally). If proxy cross-rank CCU reading is needed in the future (for example, online reconciliation), a "skip device filter" branch must be added for `VIR_MEM_TYPE_CCU`, or the CCU row `device_id` must be recorded as a wildcard value.

> **Key prerequisite confirmation item**: Currently `LoadMicrocodeInstructionStub` (`hccp_ccu_stub.cc:238`) calls `GetRealPtrByAddr` on `insinfo.resourceAddr`, and `0x123123123` **cannot possibly** match any `VirtualMemBlock`. This indicates that what hcomm actually passes back today is the address it obtained from its own `rtMalloc` (`resourceAddr` is only used to compute Variable/Event VAs; the instruction space goes through a separate malloc). Before landing, this assumption must be confirmed through testing (see §6 O1). If confirmed, after materialization the design should change to **have the simulator provide the instruction space**, and hcomm directly uses `ccuDieBase + 0`, thereby eliminating one malloc and one address conversion.

#### 4.5.2 Backing Strategy: Sparse vs Partitioned (Key Decision)

| Approach | Description | Memory Cost (128 ranks) | VA Cost | Complexity | Evaluation |
|---|---|---|---|---|---|
| **A. Full window backing** | One 1 GiB shared memory per die | `ftruncate` 256 GiB (tmpfs sparse; actual usage ≈ touched pages) | 256 GiB/process | Low | tmpfs sparse files do not consume physical memory, but `ShmCreate` `refCount`/`size` header fields and DB `size` all become 1GiB. Logic like `ExceedsCeiling` (`CommPoolPolicy`) that checks by size would be falsely triggered |
| **B. Region-granularity backing** | One shared memory per region (9 per die) | ≈ 9.6 MiB × 2 × 128 = 2.4 GiB | Same | High (9×128×2 files + DB rows) | File count explosion: ≈ 18k shared memory files at 1024 ranks |
| **C. Single die sparse backing (recommended)** | One shared memory per die, `size = backingSize = ccu.resource_space_size` (A5/A6 **both are** `0x4800000` = 72 MiB), `mmap` without `MAP_POPULATE` | Actual ≈ touched pages (typically < 12 MiB/die); `ftruncate` 72 MiB × 2 × 128 = 18 GiB sparse | 18 GiB/process | **Low** | ✅ Address translation unifies to `host = regionBase + (va - dieBase)`; any offset hits; no failures from "unbacked holes" |

**Recommend C**, with supplements:

- **`backingSize` takes the authoritative hcomm value `ccu.resource_space_size`; ★ do not derive from `max(regions ∪ ms end)`**:

  Based on `ccu_res_specs.h:24` `CCU_V1_V2_RESOURCE_SIZE = 72 * 1024 * 1024`, `ccu_res_specs.cc:765` `GetResourceSpaceSize()` returns this value (V3 is 128 MiB). Computed comparison:

  | Chip | `max(regions ∪ ms end)` | hcomm Authoritative Value | Difference (unmodeled but must back) |
  |---|---|---|---|
  | A6 | `0x4800000` (72 MiB) | `0x4800000` (72 MiB) | 0 — **Coincidental equality** |
  | A5 | `0x940000` (**9.25 MiB**) | `0x4800000` (72 MiB) | **62.75 MiB** |

  On A6, the two derivations happen to be the same — this is **coincidental**. On A5, they differ by 62.75 MiB. If the max derivation were used, the A5 backing would be only 9.25 MiB, while hcomm would access space above `0x940000` (for example, the SQ buffer region `[0x1000000, 0x3000000)`, see O12) → falling outside the mmap → `GetRealPtrByAddr` / `AcquireDevPtrInNoHostProcess` range query fails and returns nullptr, or `CcuDieView` typed accessor wild pointer SIGSEGV.

  Loader validation 8: `max(regions ∪ ms end) ≤ resource_space_size ≤ die_size`.

  The 72 MiB on A6 comprises: regions 1.53 MiB + unmodeled space `0x188000..0x4000000` **62.47 MiB** + MS 8 MiB. These spaces and untouched regions **do not consume physical memory** (tmpfs sparse, charged by touched pages). They only consume `ftruncate` nominal length and VA reservation.

- ★ **The semantics of `regions[]` and `resource_space_size` must be distinguished (D10 semantic supplement)**: `regions[]` describes **the regions that the simulator needs to model/decode**; `resource_space_size` is **the complete size of the resource space**. The difference is **intentional white space** (up to 62.75 MiB on A5). **It is prohibited** to reduce `resource_space_size` just because the white space is large — it must equal hcomm `CCU_V1_V2_RESOURCE_SIZE`; otherwise hcomm accesses to unmodeled regions would go out of bounds.

- ⚠ **CCU shared memory must use `MemoryManager::AllocMemByName`; it is prohibited to go through `DeviceMemoryManager::AllocPhyMem/AcquirePhyMem`** (§4.5.1 hard constraint 1). The latter carries `CommPoolPolicy`: when `size ≥ kBigBlockThreshold(200MB)` in check-only mode, it redirects to the rank-shared `HcclCommPool` → cross-rank CCU state mutual overwrite. The 72 MiB on A5/A6 has sufficient headroom from that threshold, but the constraint must be hardcoded (see R1b).

- **No `regions[*].backed` field** (original O3 is closed; see §6.2). Approach C depends on three interlocking premises: **C-1** one contiguous mmap per die covering `[0, backingSize)`; **C-2** VA→host is one subtraction `hostBase + (va - dieBase)`; **C-3** the §4.5.4 `CcuDieView` typed accessor **has no boundary checks**. If `backed:false` participates in `backingSize` computation, it breaks C-1/C-3 (wild pointer SIGSEGV). If it does not participate, it is a dead configuration. Moreover, its benefit is always 0 — regions occupy only **2.13%** of `backingSize`. PFE/JETTY/MISSION/LOOP/CHANNEL have no writers; their physical memory consumption is naturally 0 pages.

- **Lazy creation**: Only create when a (rank,die) is first touched (trigger points: level2 `CCU_U_OP_GET_BASIC_INFO` / `SET_INSTRUCTION`, level1 `HcommCcuInsCreate`, runner `EnsureCcuResource`). In 1024-rank scenarios, only the scheduled ranks are actually created.
- Provide `HCCL_VM_CCU_MEM_PREALLOC=1` to support pre-creation during the `start` phase for small topologies (to expose configuration errors early).
- Provide `HCCL_VM_CCU_MEM_DISABLE=1` to fall back to the old process-private array mode (grayscale switch, see §9).
- **If VA/page table/quota pressure is observed in future testing**, the correct convergence direction is a `backing_segments` segment model (explicitly declaring contiguous segments that need mapping, for example, A6 `[{0x0, 0x188000}, {0x4000000, 0x800000}]`, which can skip the 62.47 MiB unmodeled space), combined with `CcuDieView` holding N segments + accessor segment lookup and boundary checks (which also solves P10). **Note that it saves `ftruncate` nominal length and VA reservation, not physical memory**, so it is not done without measured evidence.

#### 4.5.3 DB Row Lifecycle Isolation (Must Be Handled)

`aclrt_memory_stub.cc` calls `PurgeFreedRowsOfDevice<PhyMemBlock/VirtualMemBlock>(deviceId)` before every `aclrtMalloc`, and `cmd_base_utils.cc:1540` also has `DeleteAll`. If the CCU region rows are mistakenly purged, the runner instantly loses all CCU memory mappings.

**Countermeasure**: Add `src_type = VIR_MEM_TYPE_CCU` (existing enumeration only has `VIR_MEM_TYPE_DEV` and others), and:
1. Relax the `src_type` filter in `AcquireDevPtrInNoHostProcess` / `GetRealPtrByAddr` to `IN (DEV, CCU)`.
2. `PurgeFreedRowsOfDevice` explicitly excludes `VIR_MEM_TYPE_CCU`.
3. The `DeleteAll` path distinguishes by mode: allow purging during topology rebuild (because Ensure will be called again); prohibit purging during single-operator re-runs.
4. The CCU row `ref_count` semantics is defined as "number of mapped processes," maintained by `AcquireMemByName`/`ReleaseMemByName` (existing mechanism).

#### 4.5.4 Runner Side Changes: Backend Replacement, Not Rewrite

Keep the `CcuResourceManager` **external API unchanged** (all 35 executors require no modification). Only replace the storage backend from arrays to shared memory views:

```cpp
// Before (ccu_resource_v1.h:42-46)
std::array<uint64_t, XN_MAX> xn_[DIE_NUM];
std::vector<char>            ms_[DIE_NUM];

// After: holds a view pointing to shared memory
struct CcuDieView {
    uint64_t dieBaseVa{0};     // Device VA (ccuDieBase)
    char*    hostBase{nullptr};// Shared memory host pointer
    uint64_t backingSize{0};
    // Typed accessors based on CcuAddressSpace region table
    uint64_t* Xn(uint16_t id)   { return reinterpret_cast<uint64_t*>(hostBase + xnOffset)   + id; }
    uint64_t* Cke(uint16_t id)  { return reinterpret_cast<uint64_t*>(hostBase + ckeOffset)  + id; }
    char*     Ms(uint16_t id)   { return hostBase + MsRegionOffset(id); }
    hcomm::CcuRep::CcuInstr* Instr(uint32_t id) { ... }
};
std::array<CcuDieView, DIE_NUM> views_;
```

Accompanying fixes (technical debt that must be addressed together; otherwise it is immediately exposed after materialization):
- **P10**: All accessors add `rankId/dieId/id` boundary checks; out-of-bounds returns failure instead of undefined behavior.
- The `GetCkeValue` `uint64_t → uint16_t` truncation (`ccu_resource_manager.cc:160-171`): Clarify Event bit width semantics (real hardware Event is a 16-bit mask, but storage is 8B/entry).
- The `GetXnAddr` `xnId * 64` incorrect stride (`:364-376`): Change to use `entry_size` from configuration, and fix the calls in `load_executor.cc:59` / `store_executor.cc:59` that pass "value" as "id."
- `DumpCcuCkeResouceInfo` uses `MS_NUM` (1536) to iterate over `cke_` (1024), causing out-of-bounds access (`:677`).
- `Init` overwrites `ccuResourceBaseAddr_` every time (`:62`, last-writer-wins): Change to per-(rank,die) storage.
- `ccuExecutorMutex` (`ccu_resource_manager.h:123`) is declared but never locked: After materialization, CCU memory becomes **cross-process shared**. The concurrency model must be explicitly defined (see §6 R2).

#### 4.5.5 Checker Side Changes

- Delete the constant block and 4 conversion functions in `ccu_all_rank_param_recorder_v3.cc:259-445`; replace with holding a `const CcuAddressSpace*`.
- `StorageManager::InitCcuInfo` (`checker/.../storage_manager.cc:757-763`) changes to fill a base address table **indexed by (rankIdx,die)** from configuration/DB (the current `std::vector<uint64_t>` only has the die dimension, which is a direct manifestation of P3). `AllRankParamRecorder::ccu_resource_base_addr_` type is upgraded accordingly.
- New capability (previously impossible): The checker can optionally **directly read the shared memory written by the runner** to perform "static graph validation + dynamic execution result" reconciliation.
- Add missing validations: `msId < ms_num`, `ckeId < cke_num`, `xnId < xn_num` (currently only LoadX/StoreX has 0..16383 checks, see §2.5). The `GenSliceFromMs` MS slice currently **bypasses** the boundary check in `CheckSingleSlice` (`task_graph_single_task_check_v3.cc:557-559`). After materialization, it should validate against the configured capacity.

#### 4.5.6 Proxy Side Changes

| Location | Change |
|---|---|
| `hccp_ccu_stub.cc:69-125` | `SetCcuV1/V2ResourceBasicInfo` → caps derived from `DeviceSpec::GetCaps()` (no longer reads yaml `caps` section); `resourceAddr` takes `Ccu.resource_addr` (real, per-rank) |
| `hccp_stub.cc:2089-2209` | `GetCcuMemInfo` → **branch by ccu_version**; region table generated from `DeviceSpec.regions` (fixes §2.3 defects 1, 2); `mem_va` uses real dieBase+offset |
| `hccp_ccu_stub.cc:400-410` | Delete `0x456000000/0x123000000` magic number discriminator; use `CcuAddressSpace::IsInCcuWindow(addr) → (rankIdx, dieId)` (this simultaneously fixes a derivative hack of P3). Note: This check currently only exists in the A5 branch (`devType == DEV_TYPE_950` gated); the change scope is correspondingly limited to the A5 path |
| `ccu_level1_res.cc:33-48` | Delete all `CCU_V1_CAPACITY`/`CCU_V2_CAPACITY`/`DIE_BASE_ADDR`/`V1_*_OFFSET`/`V2_*_OFFSET` literals; read from `DeviceSpec` instead. `GetLayout()` generated from configuration |
| `ccu_level1_res.cc:386-481` | Bump allocator remains unchanged (`SYNC_SLOT_NUM=4` changes to read `reserved_sync_slots`); `baseVa` computation uses `CcuAddressSpace::ResIdToAddr` |
| `ccu_kernel_sim.cc:66-72` | **Delete the 5th MS address space** (`CCU_MS_BASE=0x08000000`, `CCU_MS_WINDOW_CAP=64MB`, die-equal-split); use `CcuAddressSpace::MsIdToAddr` instead (fixes §2.3 defect 3) |
| `topo_ascend_cluster_parser.cc:970` | `FillCcu()` fills `resource_addr = ccuDieBase(rankIdx, dieId)` (fixes P7); simultaneously the `dieId < 2U` hardcode (`:1129`) changes to read `soc.die_num` |
| `store_dump_shm_data.cc:497-498` | Remove "placeholder" comment; write real values (fixes P8) |
| **5 `strncmp(soc_version, "Ascend9xx", 9)` locations** | `hccp_ccu_stub.cc:134,258,513`, `ccu_level1_res.cc:494-498`, `aclrt_new_stub.cc:47-52`, `hccl_task_sequential_execute_v2.cc:103-105` → all change to query `DeviceSpec.ccu_version` (single-point mapping, eliminates A5/A6 dual-branch duplication) |
| **`ParseDevType`** | `topo_cluster_ir.cc:137-152` **delete entirely** — it only recognizes exact `ascend950`/`ASCEND950`/`950`, and for real values `Ascend950DT_95A1`/`Ascend950` it **always returns `UNKNOWN`**. It also has **no 960 branch** (only 950/910B/910C/910D), and has decayed with the introduction of A6. Also delete call sites `topo_cluster_parser.cc:710`, `topo_cluster_dumper.cc:105` `devType` output, and evaluate whether the `SimDevType` enumeration and `Device.devType` field should be removed together. `DevType` is derived from `DeviceSpec` instead |
| `ccu_resource_manager.h:41-50` + `ccu_all_rank_param_recorder_v3.h:37-46` | Two duplicate `CcuComponerntType` enumerations (all members carry `_A6` suffix, with the spelling error "Componernt") → unified to `devspec::CcuResType`. Affects 8+ call sites: runner `sync_xn_wt_executor.cc:63,64,70,72,84`, `trans_mem_executor.cc:130`; checker `ccu_task_transform_instruct_a6_v3.cc:2312-2323,2366,2453-2463,2519` |

#### 4.5.7 rtMalloc Protection (P9 / Requirement 3.3)

##### 4.5.7.1 Triple Guarantee

| # | Guarantee | Mechanism | On Violation |
|---|---|---|---|
| ① | VA does not fall into this rank CCU 2 GiB | **Per-segment first-fit** (the block must fit entirely within a single user_mem segment) + head/tail dual `IsInCcuWindow` assertions | Skip to the next segment; assertion trigger = internal error |
| ② | `VA + size` does not exceed this rank 1 TiB upper limit | All segments exhausted and still cannot fit → **ERROR log + return nullptr** | ✅ Explicit error, no silent wraparound |
| ③ | Does not intrude into the next rank window | Derived from ②: `maxVA = rankBase + 1TiB − 1 < next rankBase` (`0x8000000000 + 0x10000000000 = 0x18000000000` is exactly the rank1 start, seamless and non-overlapping) | Same as ② |

★ Requirement 3.3 is **structurally guaranteed** by ① and no longer depends on any quota constant — this is the premise for deleting `alloc_guard.user_mem_limit`.

##### 4.5.7.2 ⚠ Why a "Single Global Linear Watermark" Cannot Be Used

A natural implementation concatenates the two user_mem segments into a single linear space, using a single scalar watermark + one mapping:

```
UserMemVaFromLinear(rankIdx, linear):  Subtract linear segment by segment; return rankBase + seg.offset + linear on hit
```

**It has a bug — allocation blocks would cross the CCU gap.** Precise counterexample:

```
seg0 = [0, 0xE00000000)                        # 56 GiB
cur  = 0xDFFFFFF00                               # seg0 remaining 0x100 bytes
Request alignedSize = 0x400

Linear total check: 0xDFFFFFF00 + 0x400 = 0xE000000300 ≤ userMemTotal(0xFF80000000)   → ✅ passes
VA = rankBase + 0xDFFFFFF00
Block = [rankBase+0xDFFFFFF00, rankBase+0xE000000300)
CCU window = [rankBase+0xE00000000, rankBase+0xE80000000)
→ Block tail intrudes into CCU die0 by 0x300 bytes ❌
```

Therefore `RankAddressSpace` **does not provide** `UserMemVaFromLinear`. Allocation must be performed by the caller **traversing segments** (first-fit). The waste from segment skipping = `seg.size − cur`, and since the request cannot fit, `seg.size − cur < alignedSize` → **waste is strictly less than a single request size** (≤ `CommPoolPolicy::kPoolSize` 4 GB; typically KB~MB in practice). Based on the measured peak of 0.405 GiB / seg0 56 GiB, the normal path **never skips segments; waste is 0**.

##### 4.5.7.3 `AllocVirMem` Changes (`store_sim_device_memory_manager.cc:47-62`)

> ⚠ **Existing defect discovered during implementation (O14)**: The call sites of `AllocVirMem` feed in **two different id spaces** —
> `aclrtMalloc:277` passes `GetCurrDeviceId()` (= `Context.device_id`, DB primary key, **starting from 1**),
> while `aclrtReserveMemAddress:972`, `:444`, `:1040` pass `Device.physical_id` (local id within the server, **starting from 0**,
> with the first card legitimately being 0). Both share the same window sequence → a device with `physical_id=0` and a device with `PK=1` would fall into **the same rank window**.
> This is a pre-existing problem (before the refactoring, `rankIdx = devPhyId - 1` with `devPhyId==0` causes uint64 wraparound, producing
> garbage VAs like `0xFFFFFF8000000000`, but since they are non-null, the bug was never detected).
> **S4 deliberately does not fix it** (it belongs to S2-2/O14 scope), only: ① when `devPhyId == 0`, follow the degraded path, maintaining behavior
> **bit-for-bit identical** to pre-refactoring to avoid regression; ② log one WARN identifying the problem. Test evidence: the `test_aclrt_memory_stub`
> fixture uses exactly `physical_id = 0` (`:199`); if G0 directly rejected it, 4 test cases would change from pass to fail.

```cpp
void* DeviceMemoryManager::AllocVirMem(uint64_t devPhyId, size_t size)
{
    // ── G2 size validity (★ fix latent bug B3: size==0 originally returned a non-null pointer and inserted
    //    a zero-length VirtualMemBlock with start_end_ptr == start_ptr; range queries Le&&Gt never hit) ──
    const uint64_t alignedSize = (static_cast<uint64_t>(size) + 7ULL) & ~7ULL;
    if (alignedSize == 0) { HCCL_VM_ERROR(...); return nullptr; }

    std::lock_guard<std::mutex> lock(m_virSizeLock);

    // ── G0 devPhyId == 0: existing id space inconsistency (O14), maintain old behavior + warning ──────
    const bool devKeyIsZero = (devPhyId == 0);
    if (devKeyIsZero) { HCCL_VM_WARN("...physical_id(0-based) and Device PK(1-based) share window sequence, see O14"); }

    const devspec::AddrSpace* addrSpace = devKeyIsZero ? nullptr : ResolveAddrSpace(devPhyId);
    if (addrSpace == nullptr || addrSpace->userMemSegs.empty() || addrSpace->rankStride == 0) {
        // ── Degraded path: no spec available, or devPhyId == 0 ───────────────────────
        // Fall back to pre-refactoring linear bump (VA bit-for-bit identical, including (0-1) uint64 wraparound),
        // but **newly add** rank window upper-bound check (strictly better than pre-refactoring: previously had no upper bound).
        // ⚠ This path does not skip the CCU window; under normal conditions the spec is always loaded from config/device/*.yaml
        //   (LoadAll only reads yaml, does not depend on DB).
        constexpr uint64_t kFallbackBase = 0x8000000000;      // = yaml address_space.rank_base
        constexpr uint64_t kFallbackStride = 0x10000000000;   // = yaml address_space.rank_stride
        auto& cursor = m_deviceMemCursors[devPhyId];
        if (cursor.empty()) { cursor.assign(1, 0); }
        const uint64_t used = cursor[0];
        if (used > kFallbackStride || alignedSize > kFallbackStride - used) { HCCL_VM_ERROR(...); return nullptr; }
        const uint64_t va = kFallbackBase + (devPhyId - 1) * kFallbackStride + used;
        cursor[0] = used + alignedSize;
        return reinterpret_cast<void*>(va);
    }

    const uint64_t rankIdx = devPhyId - 1;                    // Specification unchanged (D5)

    // ── G1 rank slot upper bound = GetMaxRanks() (default 1024) ──────────────────────
    const devspec::RankAddressSpace ras(*addrSpace);
    if (rankIdx >= ras.MaxRankSlots()) {
        HCCL_VM_ERROR("alloc vir mem failed: devPhyId:{} rankIdx:{} >= max rank slots {}",
                      devPhyId, rankIdx, ras->MaxRankSlots());
        return nullptr;
    }

    const uint64_t alignedSize = (static_cast<uint64_t>(size) + 7ULL) & ~7ULL;
    if (alignedSize == 0) {
        HCCL_VM_ERROR("alloc vir mem failed: devPhyId:{} invalid size {}", devPhyId, size);
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(m_virSizeLock);
    // ★ Fix latent bug B2: original member was std::map<int, uint64_t>; key is int but
    //   indexed by uint64_t devPhyId → implicit narrowing.
    auto& cursors = m_deviceMemCursors[devPhyId];
    if (cursors.size() != ras->UserMemSegCount()) { cursors.assign(ras->UserMemSegCount(), 0ULL); }

    // ── G3 segmented first-fit: block must fit entirely within a single user_mem segment (guarantees ①) ──
    const uint64_t rankBase  = ras->RankBase(rankIdx);
    const uint64_t windowTop = rankBase + ras->RankWindowSize();
    for (size_t i = 0; i < ras->UserMemSegCount(); ++i) {
        const uint64_t segOff = ras->UserMemSegOffset(i);
        const uint64_t segSize = ras->UserMemSegSize(i);
        const uint64_t cur = cursors[i];
        if (cur > segSize || alignedSize > segSize - cur) { continue; }
        const uint64_t va = rankBase + segOff + cur;

        // ── G4 invariant assertion (defensive; guarantees ①③) ───────────
        uint32_t hitRank = 0, hitDie = 0;
        if (va < rankBase || va + alignedSize > windowTop ||
            ras->IsInCcuWindow(va, hitRank, hitDie) ||
            ras->IsInCcuWindow(va + alignedSize - 1, hitRank, hitDie)) {
            HCCL_VM_ERROR("alloc vir mem INTERNAL ERROR: devPhyId:{} va={:#x} size={:#x} "
                          "violates window/ccu invariant", devPhyId, va, alignedSize);
            return nullptr;
        }
        cursors[i] = cur + alignedSize;
        HCCL_VM_INFO("dev:{} rankIdx:{} seg:{} alloc vir mem:{:#x} size:{} (seg used {:#x}/{:#x})",
                     devPhyId, rankIdx, i, va, alignedSize, cursors[i], segSize);
        return reinterpret_cast<void*>(va);
    }

    // ── G5 exhaustion = guarantees ② ─────────────
    uint64_t used = 0, cap = 0;
    for (size_t i = 0; i < ras->UserMemSegCount(); ++i) { used += cursors[i]; cap += ras->UserMemSegSize(i); }
    HCCL_VM_ERROR(
        "alloc vir mem FAILED: devPhyId:{} rankIdx:{} req:{:#x} exceeds device address budget. "
        "rank window [{:#x},{:#x}) = {} GiB; ccu reserved {} GiB; "
        "user_mem total {:#x} ({} GiB), used {:#x} ({} GiB), all {} segment(s) full",
        devPhyId, rankIdx, alignedSize, rankBase, windowTop, ras->RankWindowSize() >> 30,
        ras->CcuWindowSize() >> 30, cap, cap >> 30, used, used >> 30, ras->UserMemSegCount());
    return nullptr;
}
```

Supporting changes: Delete the `constexpr uint64_t deviceBase/deviceAddrStep` local constants within the function (replaced by spec-provided values); periodically output per-segment watermark logs (for capacity assessment, replacing the originally planned `HCCL_VM_STAT_USER_MEM_PEAK`).

##### 4.5.7.4 Zero-Regression Argument

| Argument | Data |
|---|---|
| Existing workload peak watermark | **0.405 GiB / rank** (measured from `log.txt`, 112 allocations, dev 1/9/10) |
| Distance to segment 0 boundary (56 GiB) | **138x headroom** → all existing allocations fall within segment 0 |
| VA formula within segment 0 | `rankBase + 0 + cursors[0]`, **bit-for-bit identical** to the current `deviceBase + (devPhyId-1)*deviceAddrStep + m_deviceMemSizes[devPhyId]` |
| Conclusion | Already-allocated addresses remain unchanged; 19 golden `servers_info.json` files are not affected |
| Rank ownership extraction | `(va - rank_base) / rank_stride` still correct — segment skipping only changes the within-window offset, not the window ownership. The checker unused `SIM_MEM_MASKER = 0xFFFFFF0000000000` (`sim_common.h:39`) semantics are also unaffected |
| Reverse consumer verification | Only one location in the entire repository produces `rankBase + linear`: `store_sim_device_memory_manager.cc:57`. **No** `(va - rank_base) % rank_stride` reverse computation exists. All 7 VA→host resolution points use **DB block-based** lookup rather than window-linear: `AcquireDevPtrInNoHostProcess` (`store_sim_shm_memory_common.cc:37` `offset = devPtr - virMem.start_ptr`), `GetRealPtrByAddr` (`aclrt_memory_stub.cc:152`), checker `GetSlice` (`storage_manager.cc:247,252`) |
| Single allocation upper bound | `CommPoolPolicy::kPoolSize = 4GB` + `ExceedsCeiling` remains; no giant block risk |

##### 4.5.7.5 `FreeVirMem` Intentionally Does Not Reclaim VAs (option a)

`FreeVirMem` (`:65-68`) keeps its empty implementation, but **a comment must be added explaining the rationale**; otherwise future developers will treat it as an omission:

> The free path only sets `is_freed=1` and **retains the DB row**. This is a design documented in `aclrt_memory_stub.cc:474-482` with production incident forensic evidence backing (FR1 128p / R3K4 256p: during the teardown phase, 43/48 remote address resolution misses all occurred more than 16.7s after the target row was deleted; retaining rows allows slow ranks late resolutions to still hit). The invariant is **"marked rows and active rows never coexist at the same address"**, and this invariant holds precisely because the bump allocator **never reuses addresses** — all 7 range queries (`store_sim_shm_memory_common.cc:28`, `aclrt_memory_stub.cc:97/139/1232/1321`, `device_arm/common/hccl_device_pub.cc:169/274`) **deliberately do not filter by `is_freed`**. Once VAs are reclaimed and reused, the same address would be simultaneously covered by a "freed old row" and a "new active row," making `GetOneByPred` return nondeterministic results → resolving to the wrong shared memory block.

Budget sufficiency: user_mem 1022 GiB vs measured peak 0.405 GiB (**2523x headroom**); the true first-failure point is `CommPoolPolicy::kPoolSize` (4GB) and `/dev/shm` capacity. If reclamation is needed in the future, a "quarantine period / epoch barrier" approach must be used and the above invariant must be re-verified — the cost table for four candidate approaches is in **O13**.

#### 4.5.8 Third-Party Plugin Integration (Requirement 2.3 / G6)

- **C++ plugins**: Link `device_spec`; directly use `RankAddressSpace` / `CcuAddressSpace`.
- **Python plugins (light_performance)**: Add `device_spec.py` to load the same yaml (`pyyaml` is already an existing dependency; `generate_cluster_topo.sh` uses it). Delete the constants and decoding logic ported from the checker in `ccu_graph_builder.py:221-294` / `register_model.py`.
- Provide CLI export capability (reuse the existing `src/cmd/` subcommand framework: inherit `CommandBase` + place a `CommandAutoRegister<T>` static instance in the file to automatically register with `CommandRegistry::CreateAll()`, **no registry modification needed**):
  ```
  hccl-vm device spec show [<soc>]              # Print parsed specification + 9 validation results
  hccl-vm device spec export <soc> -o x.json    # Export JSON for non-C++/Python third-party consumption
  hccl-vm device addr decode 0x18e00140008      # VA → (rank, die, region=CKE, id=1)
  hccl-vm device addr encode --rank 1 --die 0 --type CKE --id 1   # Reverse → 0x18e00140008
  ```
  `decode` is extremely valuable for investigating "where an address comes from." `encode` increases in value after user_mem changes to segmented mapping (manual VA computation becomes harder). Both have very low implementation cost (reuse `CcuAddressSpace::AddrToResId` / `ResIdToAddr`).
- **DB pathway**: Third parties can directly read the 6 `DeviceSpec*` tables in `hccl_sim.db` (§4.8), or use `hccl-vm table show DeviceSpec` to dump. They must handle "table is empty (`start` was never run)" and fall back to reading yaml.

### 4.6 Unified Call Chain

```
hcomm initialization
  → RaTlvRequest(MSG_TYPE_CCU_GET_MEM_INFO)
      → GetCcuMemInfo: DeviceSpec(soc).regions × ccuDieBase(rank,die)   ★real addresses
      → CcuMemoryService.EnsureCcuDieMemory(devKey, die)                ★real shared memory allocation
      → UPDATE Ccu.resource_addr = ccuDieBase
  → SimRaCustomChannel(CCU_U_OP_GET_BASIC_INFO)
      → baseinfo.resourceAddr = Ccu.resource_addr; caps = DeviceSpec::GetCaps()
  → HcommCcuVariableAlloc / EventAlloc (level1)
      → baseVa = CcuAddressSpace.ResIdToAddr(dieBase, XN|CKE, id)       ★real VA
  → CCU_U_OP_SET_INSTRUCTION
      → GetRealPtrByAddr(dieBase + 0) → shm host ptr                    ★direct hit, no special handling needed
  → rtCCULaunch → DB(task) → runner

runner (separate process)
  → EnsureCcuResource: Ccu.device_id/die_id → dieBase
      → AcquireDevPtrInNoHostProcess(dieBase) → same shared memory block  ★shared with hccl-vm
      → CcuDieView directly reads/writes XN/CKE/MS
  → executor->Run(): write XN[id] = v  →  lands in shared memory

hcomm / proxy / checker (any process)
  → read XN[id]: AcquireDevPtrInNoHostProcess(dieBase + xnOff + id*8)     ★sees runner writes
```

### 4.7 soc_version Configuration Convergence (Target State Defined, Phased Implementation)

#### 4.7.1 Current State: 5 Occurrence Locations, Only 1 Active

| # | Location | Current Value | Actual Downstream | Status |
|---|---|---|---|---|
| 1 | `cluster/*.yaml`<br>`server_list[].soc_version` | `Ascend950DT_95A1`<br>`Ascend960`<br>`Ascend960DT_961BY` | → `servers_info.json` → `topo_cluster_parser.cc:703-710` → `Device.socVersion` → DB `Device.soc_version` → 3 external APIs (§4.7.6) + 5 `strncmp` locations | ✅ **Only active one** |
| 2 | `server_or_pod/*.yaml`<br>top-level `soc_version` (16 files) | `Ascend950` / `Ascend960` | Only feeds one log line in `generate_server_topo.sh:300`. The `hardware_type = soc_version.upper()` fallback at `:1439` is **unreachable** (all 16/16 files have explicit `hardware_type`). Moreover, the `topo_json` dictionary (`:1441-1448`) **does not contain the `soc_version` key at all** | ❌ **Dead configuration** |
| 3 | `generate_cluster_topo.sh:629-646`<br>fallback chain | — | `:637` uses rootinfo filename regex `(ascend\d+)` (`extract_soc_version`, `:584-588`); `:643` reads `topo.json` `soc_version` → inherits from #2; **that key does not exist; dead code**. Moreover, every `server_list` entry in all 22 cluster files explicitly provides a value; the fallback chain **never triggers** | ❌ **Half-dead** |
| 4 | `topo_meta/*.yaml` top-level `soc_version` | None of the 33 files in the repository **have this field** | C++ `ParseTopoMetaYaml` (`sim_common_api.cc:173-215`) only reads `meta`/`topology`, not `soc_version`. It only exists in the `ai_skills/.../Ascend_ranktable_file_generation.md:68,83` sample and `generate_ranktable.sh:287` (default `'unknown'`) | ❌ **Documentation ghost** |
| 5 | `device/*.yaml` (new in this design) | Single value | The target state single authority; **Phase 0 loader completely ignores** (§4.7.4) | 🆕 |

> The user confusion about "not being able to tell the difference" mainly comes from #2/#3/#4 — these three **dead configurations** — they look authoritative but produce no actual effect.

**Verified key fact**: `server_topo ↔ soc_version` is **strictly 1:1** (16 pairs, no cross-over), so sinking the authority to the server_or_pod / device layer **loses no information**:

| server_or_pod file | cluster actual soc_version | That file hardware_type |
|---|---|---|
| `ascend960_server_topo_clos` / `pod_topo_clos` / `nc16_topo_clos` | `Ascend960` | `960D-clos-{8,72,16}P-topo` ✅ |
| `ascend960_server_topo_normal` / `_normal_hostdpu` / `x2_128p_..._normal` | `Ascend960DT_961BY` | **`950D-normal-...`** ❌ See R10 |
| `ascend950_server_topo_*` (10 files) | `Ascend950DT_95A1` | `950D-*` ✅ |

#### 4.7.2 Target State: Single Authority + Three-Layer Reference Chain

```
asset/cluster_model/config/
├── device/                     ★ Single soc_version authority
│   ├── ascend950.yaml            soc_version: "Ascend950DT_95A1" + full CCU specification
│   └── ascend960.yaml            soc_version: "Ascend960" (pending O8 normalization)
├── server_or_pod/              Delete soc_version, add device reference
│   └── ascend950_server_topo_normal.yaml
│         device: "ascend950"
├── cluster/                    Delete server_list[].soc_version
│   └── ascend950_cluster_4_server_hf.yaml
│         server_list:
│           - {super_pod_id: 0, id_range: [0,3], server_topo: "ascend950_server_topo_hf.yaml"}
└── topo_meta/                  ★ Required for mock-comm path (§4.8.7)
    └── 112.yaml
          device: "ascend950"   ← Replaces the soc_version ghost field in ai_skills documentation
```

Resolution chain: `cluster.server_topo` → `server_or_pod/<x>.yaml` → `.device` → `device/<y>.yaml` → `soc_version` + full CCU specification.

Reference resolution has an existing precedent: `find_server_topo()` (`generate_cluster_topo.sh:345-361`). Moreover, the `topo_bank/` and `config/super_pod/` directories **do not actually exist** (verified), so the `find_server_topo` `'bank'` branch is a dead end → reference resolution only needs to handle the yaml case.

**Incidental benefit**: After device spec is attached at the server_or_pod layer, **heterogeneous clusters (one cluster mixing different soc types) become expressible** — today `soc_version` is at the `server_list[]` entry level, but all 22 cluster files are homogeneous; that capability has never been used.

#### 4.7.3 Phased Implementation

| Phase | Content | Change Scope | When |
|---|---|---|---|
| **Phase 0 (current)** | **Three coexisting**: cluster / server_or_pod / device each retain `soc_version`; the device yaml writes this field but **the loader completely ignores it** (§4.7.4); `server_or_pod` does not add `device:` reference; matching still uses `name` longest prefix (§4.2.1) | **Topology configuration zero changes, 19 golden zero changes, external API zero changes** | S1 |
| Phase 1 | Delete #2/#3/#4 three dead configurations (16 `server_or_pod` delete field, delete `extract_soc_version` and `:637-646` dead fallback chain, clean ai_skills ghost); `server_or_pod` adds `device:` reference | 16 yaml + 2 scripts + 2 documents | After main design completes |
| Phase 2 | `cluster` deletes `soc_version`; `generate_cluster_topo.sh` retrieves value through `server_or_pod.device` → device yaml and writes to `servers_info.json` (and recommends adding `device_spec` key, O11); 19 golden files regenerated | 22 yaml + 1 script + 19 golden | After Phase 1 |
| Phase 3 | `topo_meta` adds `device:` reference to enable mock-comm (33 files + `ParseTopoMetaYaml` + `generate_ranktable.sh`); **O8 normalization decision lands** | 33 yaml + 2 code locations | After Phase 2 |

#### 4.7.4 Phase 0 Field Semantics (Decision (i))

The device yaml `soc.soc_version` field **exists but the loader completely ignores it** — it is not parsed, does not participate in matching, does not participate in validation, is not persisted to DB, and serves only for human reading; `DeviceSpec::socVersion` member is always an empty string.

⚠ **This must be explicitly documented**: It is a known "exists but inactive" field, of the **same category** as #2 in §4.7.1 (the dead `soc_version` in `server_or_pod`). Both reference implementations have a prominent comment at the top of the `soc:` section explaining this and pointing to O8; this section is the corresponding design-side record. If it is not documented in both places, future developers will mistakenly think it is active (which is exactly how #2 became dead configuration).

The Phase 0 matching authority remains `cluster/*.yaml` `soc_version` + `name` longest prefix (§4.2.1); validation 9 also scans values from the topology configuration and **does not read** the spec internal `soc_version`.

#### 4.7.5 Normalization TODO

See **O8**: A6 `Ascend960` vs `Ascend960DT_961BY` choose one (along with R10 `hardware_type` data error), to be handled after the main design completes.

#### 4.7.6 External Contracts That Must Be Maintained

`Device.soc_version` **is not an internal key**; it passes through to 3 external APIs as-is. No convergence phase may break this, unless O8 explicitly decides and accepts golden changes:

| API | Location | Description |
|---|---|---|
| `aclrtGetSocName()` | `aclrt_device_stub.cc:477-497` | **Public ACL API**, `memcpy(SocName, devRes->soc_version, ...)` returns as-is to applications |
| `halGetSocVersion()` / `halGetChipInfo()` | `ascend_hal_stub.cc:237-247`, `device_arm/level2/proxy/ascend_hal_stub.cc:122-123` | **Driver HAL API**, `chipInfo->name` returns as-is |
| `fe::PlatformInfoManager::InitRuntimePlatformInfos(device.soc_version)` | `aclrt_device_stub.cc:90` (comment `:77` "load platform_config ini") | Passed to **CANN frontend** to look up platform configuration by soc name; `HCCL_VM_ENABLE_PYTORCH=1` path (`docs/pytorch_guide.md`), **failure only WARNs → silent degradation** |

Additionally: `Device.soc_version` is a **CLI writable column** and is used as a documentation sample — `subcmd_table.cc:47` help text is `"Column to update (e.g., soc_version)"`, `cmd_table_utils.cc:600-604` has a TEXT column special case for it, `test/cmd/subcmd/subcmd_table_test.cc:123` uses it in test cases.

⇒ **The DB column is permanently retained**; only the **configuration source** is converged.

#### 4.7.7 Coordination with `ccu_resource_statistics_design.md` (A9)

The resource categories enumerated in that document (channel / instruction / Variable / Address / Event / CcuBuffer / loop / mission) and the `regions[] + ms` in this design are **the same taxonomy**, but the two documents have **zero cross-references** (verified by grep: it does not contain any occurrence of `device_spec` / `config/device` / `resource_addr`), and it hard-references the S2-to-be-deleted `recorder .cc:259-269` region constants and `MAX_LOADX_STOREX_ID_NUM(16383)`.

**Requirements**: ① The "capacity denominator" for the statistics feature (utilization = usage count / capacity) directly takes `DeviceSpec::GetCaps()`; ② The outstanding confirmation item in `ccu_resource_statistics_design.md` §3.4 ("whether the msId from `GetMSIdByAddr` is a die-global index or a bank-local index") and §4.4.3 of this document **are the same problem**, and should be resolved together in S0; ③ That document adds a dependency declaration. Otherwise, after the refactoring completes, a recurrence of "the address table is unified, but the statistics side copies another set of constants" would appear (the 5th implementation).

### 4.8 DeviceSpec DB Persistence

#### 4.8.1 Persistence Timing and Hook Point

```
hccl-vm start <cluster.yaml>
  → subcmd_start.cc:70   GenerateClusterTopo (bash+python generates JSON)
  → subcmd_start.cc:94   InitHvmEnv
  → cmd_base_utils.cc:503 InitClusterTopo
  → topo_ascend_cluster_parser.cc:303-334 → InitClusterStaticTopoData(:1010-1167)
       Already batch-inserts Server / Host / Device / Ccu / Port / EndPoint / EndPointPortMapping
       ★ New: insert DeviceSpec* 4 tables after the Device batch and before the Ccu batch
```

This also resolves the `// todo: get soc version from yaml file` hanging in `FillDevice` (`:956-968`).

#### 4.8.2 Table Structure (`Scope::Static` → RunnerDB `hccl_sim.db`)

```cpp
// include/runtime_state/sim_models.h, adjacent to Ccu(:156-162)
typedef struct {
    uint64_t id;                     // PK
    char     name[64];               // spec identifier = yaml file name ("ascend950"), UNIQUE
    uint8_t  ccu_version;            // 0=CCU_V1 1=CCU_V2 2=CCU_V3
    uint8_t  die_num;
    uint32_t spec_version;
    uint64_t rank_base, rank_stride, rank_window_size;
    uint64_t ccu_window_offset, ccu_die_size;
    uint64_t resource_space_size;    // = backingSize
    uint32_t reserved_sync_slots, count_xn_num;
    // ★ No user_mem_limit column (that field is deleted; user_mem is derived from partitions)
    // ★ No soc_version column (decision (i): Phase 0 loader does not read this key)
    // ★ No gsa_count column (expressed by the DeviceSpecRegion GSA row)
} DeviceSpec;

typedef struct { uint64_t id, spec_id; char name[32]; uint16_t type;
                 uint64_t offset, size; uint32_t entry_size, count; } DeviceSpecRegion;
typedef struct { uint64_t id, spec_id; uint32_t ccua_index; uint64_t offset, size; } DeviceSpecMsWindow;
typedef struct { uint64_t id, spec_id; uint32_t die_id; uint64_t offset; } DeviceSpecDie;
typedef struct { uint64_t id, spec_id; uint64_t offset, size; } DeviceSpecPartition;  // multi-segment user_mem + ccu
typedef struct { uint64_t id, device_id, spec_id; } DeviceSpecBinding;               // ★ see 4.8.4
```

The `db_sim_runner_ops.h` `Table(...)` list (currently 66 tables) adds 6 entries; DDL macros are already available (`Field/Version/PrimaryKey/Unique/Index/Text/Blob/Scope/AutoIncrement`, `:48-57`).

**Zero backend cost**: sqlite (`sqlite_engine.cc:984`) and mariadb (`mariadb_engine.cc:1063,1070`) both render DDL from the same metadata `RenderCreateTable`; **no need to write two sets of table creation statements**.

#### 4.8.3 Query Interface

`db_sim_runner_common.{h,cc}` adds (adjacent to `GetCcuFromDeviceByDieId`, `:768`):

```cpp
HcclVmResult GetDeviceSpecByName(const std::string& name, sim::runtime::DeviceSpec& out);
HcclVmResult GetDeviceSpecByDeviceKey(uint64_t deviceKey, sim::runtime::DeviceSpec& out);  // Through Binding
HcclVmResult GetDeviceSpecRegions(uint64_t specId, std::vector<DeviceSpecRegion>& out);
HcclVmResult GetDeviceSpecMsWindows(uint64_t specId, std::vector<DeviceSpecMsWindow>& out);
HcclVmResult GetDeviceSpecPartitions(uint64_t specId, std::vector<DeviceSpecPartition>& out);
bool LoadDeviceSpecFromDb(uint64_t deviceKey, sim::devspec::DeviceSpec& out);  // DB → in-memory object
```

#### 4.8.4 `Device` ⟷ `DeviceSpec` Association Selection

| Option | Approach | Evaluation |
|---|---|---|
| ~~soc_version join~~ | `Device.soc_version` ⟷ `DeviceSpec.soc_version` (UNIQUE) | ❌ **Excluded by decision (i)** — the loader does not read the spec `soc_version`; that column is always empty and cannot join |
| **Binding table (recommended)** | New `DeviceSpecBinding {device_id, spec_id}`, `InitClusterStaticTopoData` persists one row per device during setup (≤1024 rows) | ✅ Zero ambiguity, **does not modify the `Device` table** (avoids `Version(3)→(4)` unclear semantics), Python side can join directly |
| FK column | `Device` adds `device_spec_id`, `Version(3)→(4)` | Most semantically rigorous, but depends on O10 confirming `Table(Version(n))` change behavior (the repository has no `ALTER TABLE` migration mechanism; only `ALTER ... AUTO_INCREMENT` for reset was found) |

This phase adopts the **Binding table**; if O10 confirms Version bump is safe, it can be upgraded to an FK column in a subsequent phase and the Binding table deprecated.

#### 4.8.5 Consumers

| Consumer | Method |
|---|---|
| proxy L1/L2 | Query DB directly (already in the DB context; `GetCcuFromDeviceByDieId` is a precedent) |
| runner / checker | Prefer DB query (`LoadDeviceSpecFromDb`); fall back to `DeviceSpecRegistry` reading yaml on miss |
| light_performance | `db_reader.py` adds `DeviceSpec*` reading, reuses the `:566` `hccl_sim.db` read-only connection (that file already uses `sqlite3.connect(f"file:{path}?mode=ro", uri=True)`); delete constants ported from the checker in `ccu_graph_builder.py:221-294` / `register_model.py` |
| Third parties | Read DB (6 tables) **or** `config/device/*.yaml`; also `hccl-vm device spec show/export` CLI and `hccl-vm table show DeviceSpec` |

**★ Resolution cache (implementation-phase supplement, must-read)**: The `Db::` thin layer per contract **does not cache** (`db_sim_runner_ops.h`: each interface only delegates to one atomic interface call). Every `GetById/GetOneByPred` is one SQL statement (SQLite one prepare+step, MariaDB one round-trip), and all processes on the same machine share `data/hccl_sim.db` (WAL + busy_timeout 60s). Specification resolution also happens to fall on the CCU hot path. If each call queries the DB directly, the read count scales with the card count/channel count/allocation count (measured specification: L1 single resource allocation 18 times, single event address fetch 6 times; checker startup dual die 128 cards 385 times, 1024 cards 3073 times; A5 2 times per channel and cross-device channel results are always discarded). Therefore the convention is:

| Layer | Approach |
|---|---|
| `GetDeviceSpec` / `GetCcuResourceBaseAddr` | Process-level cache (`Device.soc_version`, `(device,die) → base address`; only caches successful results). Invalidation entry `ResetDeviceSpecCache()`, call sites: after `InitClusterStaticTopoData` commits, `hccl-vm table update Device.soc_version`, UT DB rebuild |
| `GetCurrDeviceId` | Thread-local memoize (`ctxId → device_id`, `Context.device_id` is not rewritten after row creation). Invalidation points: `SetCurrCtxTls`, `InvalidateCurrDeviceTls` (callers that bypass the former and directly delete Context rows must explicitly call, for example, `aclrtDestroyContext`) |
| checker `ResolveAllCcuBaseAddrs` | **Does not call per (device,die)** the above entry; instead performs one full table scan each of `Device` / `Ccu` + specification through the resident `DeviceSpecRegistry`, totaling 2 DB reads (independent of card count) |
| runner / store / checker address space objects | Each caches `CcuAddressSpace` per rank/device (`CcuResourceManager::AddrSpace`, `GetCcuAddrSpaceByDeviceId`, `DeviceMemoryManager::ResolveAddrSpace`) |
| A5 `ConfigChannelInfo` | First checks `locDevice.id == rmtEndPoint.device_id` before resolving die base address (does not query for cross-device channels); target state remains the pure computation `IsInCcuWindow` from §4.5.6 |

Cache validity basis: `Device.soc_version`, `Ccu.resource_addr`, `DeviceSpec*` are all `Scope::Static`, written once by `InitClusterStaticTopoData` during the start phase, with no runtime write path (the entire repository only has `ref_cnt` auto-increment and full-row deletion).

#### 4.8.6 Authority Source Constraint

**yaml is the authority; DB is a runtime artifact.** When the two are inconsistent, yaml takes precedence and the system fails fast (§4.3.2 cross-validation). Third parties that read the DB must handle the "table is empty (`start` was never run)" case and fall back to reading yaml.

#### 4.8.7 ⚠ mock-comm Path Gap

Verified: `hccl-vm mock-comm <topo_meta>` → `InitHvmCommEnv` (`cmd_base_utils.cc:523`) → `ParseRanktableAndInitCommDomain` (`topo_ascend_cluster_parser.cc:346-370`) → `ParseRanktable` (`:1265-1355`, only produces `TopoMeta` = pod/server/device id mapping) → `InitCommunicationDomain` (`:382-421`).

**This path does not go through `InitClusterStaticTopoData` at all** → in mock-comm mode, the `DeviceSpec*` tables cannot be persisted to DB (the source of `Device`/`Ccu` rows also needs to be confirmed; see O9). And mock-comm is the main entry point for checker single-operator validation.

The solution is self-consistent with the target state: `topo_meta/*.yaml` adds `device:` reference (§4.7.2), landing in Phase 3; this also cleans up the #4 documentation ghost. **Phase 0/current**: mock-comm follows the "DB miss → fall back to reading yaml" chain from §4.3.2, and functionality is not blocked.

---

## 5. Key Design Decisions and Trade-offs

| # | Decision | Alternative | Rationale |
|---|---|---|---|
| D1 | Configuration placed in `asset/cluster_model/config/device/`, YAML format | JSON; placed in `config/` root directory | Same structure as `cluster/`, `server_or_pod/`, `topo_meta/`; installation rules/test copies/golden test cases can all be reused; YAML supports comments, which specification files need extensively |
| D2 | New `device_spec` static library, **pure computation with no IO side effects** | Put the logic in `src/topo/` or `src/utils/` | Runner/checker/proxy all need to link it; `topo` depends on DB and IR, and plugins should not pull in heavy dependencies for this; a pure computation library facilitates UT and Python alignment |
| D3 | CCU memory reuses `MemoryManager` + `PhyMemBlock`/`VirtualMemBlock` | Create a separate shared memory table / use mmap anonymous sharing | Existing `AcquireDevPtrInNoHostProcess`, `GetRealPtrByAddr`, cross-process root, and refcount are all ready; **zero new IPC semantics**, lowest risk |
| D4 | Backing approach C (single die sparse, `size = max(regions ∪ MS window)`) | A (full 1GiB) / B (one file per region) / `regions[*].backed` boolean switch | See §4.5.2 table: A is falsely triggered by size-based policies, B has file count explosion, C balances translation simplicity and physical memory control. The `backed` switch was rejected — benefit is always 0 (regions occupy only 2.13% of backingSize, sparse files charge by touched pages), and it contradicts approach C premises C-1 (single contiguous mmap) / C-3 (accessor has no boundary checks), O3 is closed |
| D5 | rankIdx := `Device.physical_id - 1` | Use hcclRankId | Fully consistent with existing `AllocVirMem`, **does not change any allocated VA**; the hcclRankId approach is a breaking change with unclear benefit |
| D6 | CCU base address is **derived** from configuration, not written directly in yaml | yaml directly provides absolute addresses per rank | Rank count can reach 1024; enumeration is infeasible. The derivation formula needs only 4 constants and naturally guarantees per-rank uniqueness (solves P3) |
| D7 | Three-pathway distribution (yaml / DB / on-disk) + cross-validation | Use yaml only | Proxy stub is most natural in the DB context (`GetCcuFromDeviceByDieId` already reads that row); offline checker only has on-disk files; cross-validation detects "configuration changed but topology not regenerated" at the earliest opportunity |
| D8 | `CcuResourceManager` external API unchanged, only swap backend | Rewrite the executor layer | 35 executors + simulator are the parts this refactoring **does not need to change**; API freeze narrows the change scope to 1 class |
| D9 | Retain `HCCL_VM_CCU_MEM_DISABLE` grayscale switch | Switch directly | CCU memory materialization has a large impact scope (proxy/runner/checker three parties); a reversible grayscale path is needed |
| D10 | **device yaml is the single authority for device specifications**: address space partitioning, device base address and die offsets, CCU region layout, MS layout, capacity specification are all derived from yaml; L1/L2/runner/checker four internal layout tables are all deleted | Retain one side existing table (for example, "A5 replicates the current `GetCcuMemInfo` table") | Both macro design (device base address + die offsets) and fine-grained partitioning (CCU internal region/MS layout) share the same source from the same yaml; the A5 existing two-layout dispute changes from "which side to keep" to "all defer to yaml" (initial values verified against hcomm `ccu_res_specs` specification, completed; see §4.2.4/R5) |
| **D10-supplement** | ⚠ **D10 semantic boundary**: `ccu.regions[]` describes **the regions that the simulator needs to model/decode**; `ccu.resource_space_size` is **the complete resource space size**; the difference is **intentional white space** (A5 up to 62.75 MiB, A6 is 0). **It is prohibited** to reduce `resource_space_size` just because the white space is large | Change `resource_space_size` to be derived from `max(regions ∪ ms)` | It must equal hcomm `CCU_V1_V2_RESOURCE_SIZE` (`ccu_res_specs.h:24`); otherwise hcomm accesses to unmodeled regions (for example, SQ buffer `[0x1000000,0x3000000)`, O12) would fall outside mmap. Validation 8 only guarantees `max ≤ resource_space_size`, not equality |
| D11 | **soc_version single authority is in device yaml**, reference chain `cluster → server_or_pod → device` (three-layer homogeneous references) | Retain cluster as authority (device yaml does not declare soc_version); or device yaml lists soc_version list | Eliminates user confusion about "3 soc_version locations, cannot tell the difference"; `server_topo ↔ soc_version` is verified as **strictly 1:1** (16 pairs, no cross-over) → sinking loses no information; attaching device spec at the server_or_pod layer also makes **heterogeneous clusters expressible**; reference resolution has `find_server_topo()` (`generate_cluster_topo.sh:345-361`) precedent, and `topo_bank`/`super_pod` directories do not actually exist, only the yaml case needs handling |
| D12 | **Phased implementation**: Phase 0 (current) three coexisting, device yaml `soc_version` field exists but **the loader completely ignores it**; normalization deferred until after the main design (phases 1-3) | Converge all at once (change 22 cluster + 16 server_or_pod + 33 topo_meta + 19 golden in this phase) | One-shot convergence affects 3 external APIs (§4.7.6), 19 golden files, 2 generation scripts, ai_skills documentation, and depends on O8 A6 normalization decision (which itself depends on CANN `InitRuntimePlatformInfos` testing) — should not be coupled with the CCU main design. Phase 0 achieves **topology configuration zero changes, golden zero changes, external API zero changes** |
| D13 | `Device` ⟷ `DeviceSpec` uses **Binding table** association | `Device.soc_version` join (excluded by decision (i), that column is always empty); `Device` adds FK column (requires `Version(3)→(4)`) | Zero ambiguity, does not modify `Device` table, avoids `Table(Version(n))` change semantics being unclear (repository has no `ALTER TABLE` migration mechanism); Python side can join directly. If O10 confirms Version bump is safe, it can be upgraded to FK column later |
| D14 | **user_mem uses multi-segment `partitions` explicitly covering the full range (option A)**, `alloc_guard` deleted, available quota = `Σ user_mem segments` = 1022 GiB; allocation uses **per-segment first-fit** | Option B: delete `partitions`, `ccu.window_offset` self-describes, user_mem computed as complement by library; or retain `alloc_guard.user_mem_limit = 0xE00000000` (56 GiB) | Option A window partitioning is auditable at a glance in yaml; adding a reserved region (AIV UB and others) in the future means manually adding one partition, consistent with "do not introduce `other_reserved` field" (Q1). The only weakness of option A (possible incomplete coverage/overlap) is fully covered by validation 5. **The 56 GiB upper bound is the compensating value for when the linear bump allocator cannot skip the CCU gap**, and it becomes invalid after per-segment first-fit and would artificially reduce available quota by 94.3% |
| D15 | **GSA does not have an independent `gsa:` key**, expressed by `regions[type=="GSA"]` (if present, one region entry; if absent, do not write), `GetCaps().gsa_num` derives as "entry absent means 0" | A5 writes region + independent `gsa:{count:3072}`, A6 writes `gsa:{count:0}` | The independent key and `regions[GSA].count` are two copies of the same number (confirmed redundant on A5); and "A6 has no GSA" expressed in two forms (independent key count:0 vs no entry in regions), creating schema inconsistency |
| D16 | **WQE / SQ buffer region not modeled** (`regions` does not contain this entry) | Add a `size 32 MiB` region based on hcomm `CCU_V1_WQE_BASIC_BLOCK_OFFSET=0x1000000` | That offset is reverse-engineered from hcomm code, **not confirmed by hardware specification**; writing it into configuration would be mistaken as authoritative (prefer missing to wrong). Removal is safe: its full range `[0x1000000,0x3000000)` is fully covered by `resource_space_size = 72 MiB`; hcomm writes fall within the backing and the simulator does not interpret them. ⚠ But this safety **depends on D10-supplement** (backingSize takes the authoritative value) — if changed back to max derivation, A5 backing would be only 9.25 MiB, inevitably going out of bounds. Tracked in O12/T13 |
| D17 | **Do not reclaim released device VAs** (`FreeVirMem` keeps empty implementation), only guarantee 1022 GiB budget + watermark logs | Implement VA reclamation (free list / tail rollback) | Reclamation breaks the invariant documented in `aclrt_memory_stub.cc:474-482` with production incident forensic backing "marked rows and active rows never coexist at the same address" (FR1 128p / R3K4 256p fix), and 7 range queries **deliberately do not filter by `is_freed`**. Budget sufficiency: 1022 GiB vs measured peak 0.405 GiB = **2523x headroom**; the true first-failure point is `kPoolSize` (4GB) and `/dev/shm`. Cost table for four reclamation approaches in **O13** |

---

## 6. Risks and Open Issues

### 6.1 Risks

| ID | Risk | Level | Mitigation |
|---|---|---|---|
| R1 | **Shared memory actual/nominal size exceeds expectations**. A5/A6 `backingSize` is both **72 MiB** (`CCU_V1_V2_RESOURCE_SIZE`), 128 ranks × 2 die = **18 GiB** sparse files; when MS is heavily touched, actual usage approaches this value. `/dev/shm` defaults to 50% of memory | Medium | ① Lazy creation (only create scheduled ranks); ② `backingSize` takes `resource_space_size` (72 MiB), **must not** use full 1 GiB die window backing; ③ Provide `HCCL_VM_CCU_MEM_DISABLE`; ④ Check `/dev/shm` available space during `start` phase and warn based on `rank count × die_num × backingSize`; ⑤ If VA/page table pressure is measured, evaluate `backing_segments` segment model (§4.5.2).<br>★ An early version estimated "A5 = 192 MiB → 48 GiB (2.7x of A6)"; that estimate was based on the level1 **incorrect** `CCU_MS_BASE=0x08000000` (§4.4.3 confirmed it as incorrect), now corrected |
| R1b | **CCU shared memory, if mistakenly routed through `AllocPhyMem`, is redirected by `CommPoolPolicy`**. `kBigBlockThreshold = 200MB` (`store_sim_comm_pool_policy.h:19`); in check-only mode, `ShouldRedirect` redirects large blocks to the rank-shared `HcclCommPool` → **cross-rank CCU state mutual overwrite**, and only reproduces in check-only (checker commonly used mode), extremely hard to debug | Low (72 MiB has sufficient headroom from threshold) → but constraint must be hardcoded | CCU shared memory **must** go directly through `MemoryManager::AllocMemByName`; prohibit `DeviceMemoryManager::AllocPhyMem/AcquirePhyMem` (§4.5.1 hard constraint 1); add UT asserting that in check-only mode the CCU row `phy_mem_id` does not point to `HcclCommPool`; if O12 confirms WQE and `resource_space_size` is raised to ≥200MB, this risk immediately returns to high |
| R2 | **Cross-process concurrent writes to CCU memory**. After materialization, the runner and proxy/checker can simultaneously access the same shared memory; `ccuExecutorMutex` (`ccu_resource_manager.h:123`) is declared but never used | High | Define the concurrency model: the runner is the **sole writer** of CCU state; other processes are read-only. If checker online reconciliation is needed, use `ShmHead.lock` (`store_sim_shm_ops.h:20` already has the field) or read/write snapshots. **Must be decided at design review** |
| R3 | **DB rows mistakenly purged** (`PurgeFreedRowsOfDevice` / `DeleteAll`) | High | Add `VIR_MEM_TYPE_CCU` and explicitly exclude from all cleanup paths (§4.5.3); add UT covering "CCU rows survive after malloc" |
| R4 | **VA bit width**. `rank_base + 1024 × 1TiB = 0x4008000000000`, requires 51 bits, exceeding the 47-bit canonical user VA (< 2^47). Precise boundary: window **start** exceeds 47-bit at rank ≥ 128 (rank127 start `0x7F8000000000` < 2^47; rank128 start `0x808000000000` > 2^47); window **top** exceeds at rank ≥ 127 | Medium | Current `AllocVirMem` **already has the same problem**; not introduced by this change. Need to confirm: whether these values are ever host-side dereferenced or used for `mmap` (exploration conclusion: only used as tokens, mapped through DB). Measures: loader validation 7 outputs bit width warning; documentation states "simulated device VA is not host-dereferenceable"; if >127 rank real semantics must be supported, consider reducing `rank_stride`.<br>★ **user_mem expanded to 1022 GiB does not change this threshold**: even with the old 56 GiB upper bound, rank127 top `0x7F8000000000 + 56GiB = 0x806000000000` already exceeds 2^47; and rank ownership extraction `(va - rank_base)/rank_stride` is unaffected by segment skipping (§4.5.7.4) |
| R5 | **hcomm behavior changes after A5/A6 region layout unification**. The same A5 currently has two internal CCU layouts (L2 `GetCcuMemInfo` table vs L1 `CcuResLayout` V1, §2.3 defect 1); after unification, one side output must change | Medium | **Layout uses device yaml as the single authority (D10)**: L1/L2/runner/checker four internal tables all deleted, unified to read `DeviceSpec.regions`, no "A5 replicates current table" branch retained; S0 tests how hcomm consumes the `MSG_TYPE_CCU_GET_MEM_INFO` (`hccp_stub.cc:2220`) returned table — if only informational reporting, yaml takes full effect; if hcomm has its own built-in layout expectation (`ccu_res_specs.{h,cc}`), yaml values are calibrated against it; golden tests fix the **post-unification** `ccu_mem_rsp` byte sequence |
| R6 | **hcomm allocation failure after capacity specification unification**. If the proxy reported `xnNum` is reduced from 16384 to the runner 4096, it may trigger hcomm CCU→AICPU fallback (`ccu_northbound_call_sequence.md:224-343`) | Medium | Unify to **the real hardware value 16384**, simultaneously expand the runner array to 16384 (16384 × 8B × 2 die = 256 KiB, acceptable). The direction is "raise the runner" rather than "lower the proxy" |
| R7 | **Channel array out of bounds**. V2 reports `channelNum=1024`, runner `MAX_CCU_CHANNEL_NUM=128` | Medium | Same as R6: array width driven by configuration; `RankChannelInfo` changes from `std::array<...,128>` to a configuration-driven container |
| R8 | On-disk format change affects offline checker and historical data | Low | The `ccu0/1_resource_base_addr` field **position and type are unchanged**; only the value changes from magic number to real value; old files still parse (see §9) |
| R9 | **soc_version convergence breaks external contracts**. `Device.soc_version` passes through to 3 external APIs (`aclrtGetSocName`, `halGetSocVersion`/`halGetChipInfo`, CANN `InitRuntimePlatformInfos`), where the 3rd **only WARNs on failure → silent degradation**, and I cannot verify on this machine whether CANN accepts family-level names | Medium | Phased (D12): Phase 0 topology configuration zero changes, values zero changes; Phase 2 **must** test `InitRuntimePlatformInfos("Ascend960")` vs `("Ascend960DT_961BY")` return values first (S0-a); after Phase 2 completes, compare 19 golden files and `aclrtGetSocName` output byte-by-byte |
| R10 | **A6 topology configuration data is self-contradictory**. 3 `ascend960_*_normal*` files have `hardware_type` written as **`950D-*`** (A5), while the 3 `*_clos` files correctly have `960D-clos-*`; moreover the `soc_version` divergence (`Ascend960DT_961BY` vs `Ascend960`) **perfectly aligns** with this error → highly suspicious that `Ascend960DT_961BY` is a leftover from batch-copying the A5 template without full cleanup. `hardware_type` has real consumers: `aclrt_device_stub.cc:793-796` uses `ACL_DEV_ATTR_DEVICE_FORM_FACTOR` to look for `"POD"/"pod"` to determine form factor (through `db_sim_runner_common.cc:1024` → `Server.hardware_type`); `950D-normal-0plus8-8P-topo` does not contain POD → determined as `ACL_DEVICE_FORM_FACTOR_A_K` | Medium | **Possibly a production bug**: if these A6 configurations should be POD form factor, the form factor determination is wrong. S0-a decides together; O8 normalization corrects `hardware_type` `950D` → `960D`. Note this problem **is independent of this refactoring** and exists even without soc_version convergence |

### 6.2 Open Issues (Require Review Decision)

| ID | Issue | Impact | Recommendation |
|---|---|---|---|
| O1 | Whether the `insinfo.resourceAddr` that `LoadMicrocodeInstructionStub` passes today is the address hcomm allocated by itself, or the `resourceAddr + offset` returned by `GET_BASIC_INFO`. If the latter, it currently must return `nullptr` and fail — but test cases pass | Determines whether the instruction space can be directly provided by the simulator | Add logging and test one A5/A6 `SET_INSTRUCTION` input; decide based on the result whether the optimization at the end of §4.5.1 is feasible |
| O2 | In multi-process per-rank deployment, does the `DeviceMemoryManager` per-segment watermark (`m_deviceMemCursors`, formerly `m_deviceMemSizes`) need cross-process sharing? | Affects in-rank address uniqueness | Confirm deployment form: if one process per rank (each process has a different devPk), no sharing needed; if single process multi-rank, it is already process-internal shared, also no sharing needed |
| ~~O3~~ | ~~Whether "partial region no backing" support is needed~~ | — | **Closed (no `backed` field)**. Rationale: ① tmpfs sparse files charge by **touched pages** not `ftruncate` length; PFE/JETTY/MISSION/LOOP/CHANNEL that the simulator never reads/writes naturally consume 0 physical memory; marking false saves 0 bytes; ② On A6, 8 regions total occupy only **2.13%** of `backingSize` (1.53 MiB / 72 MiB), and `backingSize` is determined by **MS window end**, unrelated to regions; ③ If `backed` participates in `backingSize` computation, the excluded region VA falls outside mmap — the DB query path gracefully fails (returns nullptr), but the §4.5.4 `CcuDieView` typed accessor **has no boundary checks** → wild pointer SIGSEGV, directly violating approach C premises C-1/C-3; ④ Making it truly effective requires per-region segmented mmap, which degenerates to the already-rejected approach B (file count explosion + "one subtraction" translation formula invalidated). **If VA/quota pressure is observed in future testing, use the `backing_segments` segment model (which can also solve P10 boundary checks), not a boolean switch.** |
| O4 | The `Ccu` table currently has a fixed 2 rows per device (`topo_ascend_cluster_parser.cc:1125-1133` hardcodes `dieId < 2U`). If `die_num` becomes configurable in the future, the DB row count must change accordingly | Affects G1 completeness | This phase reads `soc.die_num`, but only supports 1/2; >2 reports "not yet supported" |
| O5 | `CcuResourceManager` first parameter is **deviceId** in the `EnsureCcuResource` path and **rankId** in the legacy path (`_v2.cc:44-45,123` vs `storage_manager.cc:326`). After materialization, these must be unified | Determines CCU memory index key | Unify to `deviceKey` (DB primary key), and rename the parameter in the API to eliminate ambiguity; the legacy path (`RunVirtualRuntime`) has no callers; recommend deleting concurrently |
| O6 | Does the checker need the new capability "online read runner shared memory for dynamic-static reconciliation"? | Affects R2 concurrency model | **Do not implement** this phase (keep checker reading only configuration + on-disk data); unify the address model first; reconciliation as a subsequent independent feature |
| O7 | Should `compute` / `bandwidth` sections be implemented this phase? | Affects workload | Recommend only writing schema placeholders this phase (values filled with 0 / not consumed); converge `light_performance/sim_config.json` and `TaskSchedulerDevice` in phase 2 |
| **O8** | ⚠ **TODO: A6 soc_version normalization**. Two values exist in production A6 — `Ascend960` (3 `*_clos` configurations) and `Ascend960DT_961BY` (3 `*_normal*` configurations), while one device spec can only declare one. Three options: **(a) normalize** to one value (recommended, but changes `aclrtGetSocName()` return value + 19 golden files, and requires first testing CANN `InitRuntimePlatformInfos` acceptance, see R9); **(b) create two A6 specs** + `inherits:` inheritance mechanism (CCU layout 100% identical, requires avoiding copy maintenance); **(c) list variants in spec, specified by server_or_pod** (❌ soc_version becomes two locations again, violating D11). **Along with**: R10 `hardware_type` written as `950D-*` data error corrected together | Blocks §4.7.3 phases 2/3; does not block Phase 0 (current) | **Handle after the main design (S1-S6) completes**. Phase 0 uses `name` longest prefix matching; both values can match `ascend960`; functionality is not blocked |
| **O9** | **mock-comm path** (`InitHvmCommEnv` → `ParseRanktableAndInitCommDomain` → `ParseRanktable` → `InitCommunicationDomain`) **does not go through `InitClusterStaticTopoData`**, so `DeviceSpec*` tables cannot be persisted in this mode; and the source of `Device`/`Ccu` rows in this path also needs confirmation | mock-comm is the checker single-operator validation main entry | Phase 0: covered by the §4.3.2 "DB miss → fall back to reading yaml" chain (**this is a mandatory requirement, not an optional optimization**). Phase 3: `topo_meta/*.yaml` adds `device:` reference (§4.8.7), also cleans up §4.7.1 #4 documentation ghost |
| **O10** | What is the semantics of `Table(..., Version(n))` n change? Recreate table or migrate? (Repository has no `ALTER TABLE` migration mechanism; only `mariadb_engine.cc:3691` `ALTER ... AUTO_INCREMENT` for reset was found) | Determines whether D13 can upgrade from Binding table to `Device` adding FK column | Read `sqlite_engine.cc:984` / `mariadb_engine.cc:1063` `RenderCreateTable` call chain + try changing one version and observe behavior. **This phase uses Binding table; does not depend on this conclusion** |
| **O11** | Should `servers_info.json` add a `device_spec` key? | If added, C++ does not need prefix matching (cleaner), but all 19 golden files change | Recommend adding in Phase 2. Note `ParseServersInfoJson` (`topo_cluster_parser.cc:670-724`) needs concurrent read support |
| **O12** | ⚠ **TODO: WQE / SQ buffer region location and offset not confirmed by hardware specification; not modeled this phase (D16)**.<br>**hcomm evidence**: `ccu_res_specs.h:32` `CCU_V1_WQE_BASIC_BLOCK_OFFSET = CCU_V1_CCUM_OFFSET + 0x800000 = 0x1000000`; `ccu_jetty_ctx_mgr.cc:149` `GetSqeBuffVa` → `:172` `sqeBuffVa = ccuResBaseVa + 0x1000000 + jettyCtxId × CCU_SQ_BUFFER_SIZE(256K)`; `:222` triggered by `jettyType == CcuJettyType::CCUM_CACHED_JETTY`, **not version-gated** (V1/V2 shared path). Full range `[0x1000000, 0x3000000)` = 32 MiB (128 jetty × 256K)<br>**Safety**: This range is fully covered by `resource_space_size = 72 MiB` → hcomm writes fall within the backing, simulator does not interpret, **no out-of-bounds risk**. ⚠ But this conclusion **depends on D10-supplement** (backingSize takes the authoritative value rather than max derivation) — A5 `max(regions ∪ ms)` is only 9.25 MiB; if derived by max, it inevitably goes out of bounds<br>**Next steps**: ① After confirming the offset, manually add a region at `offset 0x1000000` (`size 0x2000000`, `entry_size 0x40000`, `count 128`) and recheck validation 8; ② Confirm whether CheckerL2 `ConfigJettyInfo` (`hccp_ccu_stub.cc:467`) reaches the `CCUM_CACHED_JETTY` branch<br>**Related constants for reference**: `CCU_ONE_WQE_SIZE=64`, `CCU_WQE_NUM_PER_SQE=4` (URMA constraint: each SQE contains 4 WQEBBs), `CCU_MIN/MAX_SQ_DEPTH=16/256`, `CCU_WQEBB_RESOURCE_NUM=4096`, `CCU_START_TA_JETTY_ID=1024` (consistent with `sim_capacity_limits.h:133-147` "CCU [1024,1151]") | Does not block any phase (72 MiB backing already covers) | Adapt separately after confirmation (user decision: do not modify for now, as specific location and offset are unconfirmed) |
| **O13** | **Whether to implement device VA reclamation** (`FreeVirMem` is currently empty implementation, watermark only increases). This phase decision is **do not implement** (D17). The cost of four approaches is recorded here to prevent repeated proposals:<br>**c1** Add `Eq(is_freed, 0)` to 7 range queries → ❌ **Directly reverts FR1/R3K4 fix** (slow rank late cross-rank resolution misses again)<br>**c2** Add `generation`/`epoch` column, query takes latest generation → ❌ Semantics cannot be self-consistent ("late resolution needs old rows" directly contradicts "take latest generation"), and requires 7 query changes + add column + Version bump<br>**c3** **Quarantine reclamation**: freed ranges enter quarantine until all ranks cross a round/epoch barrier before reuse (can reuse existing `hccl_proxy_round_slot.h` `<RESOURCE_ROOT>/round_slots`) → ⚠ Feasible but is an **independent project** with workload comparable to S3<br>**c4** Tail rollback only (rollback watermark when the freed block is at the top) → ❌ After rollback, the next allocation still reuses that range, also breaking the invariant, unless the row is also hard-deleted (which reverts the fix) | VA accumulation in long-running sample sequences | **Do not implement this phase**. Budget is sufficient (1022 GiB vs measured peak 0.405 GiB = 2523x headroom); the true first-failure point is `kPoolSize` (4GB) and `/dev/shm`. If exhaustion occurs in the future, use **c3** and re-verify the §4.5.7.5 invariant |
| **O14** | ⚠ **`AllocVirMem` call sites feed two different id spaces** (discovered during S4 implementation, pre-existing before refactoring): `aclrtMalloc:277` passes `GetCurrDeviceId()` (= `Context.device_id`, DB primary key, **starting from 1**), while `aclrtReserveMemAddress:972`, `aclrt_memory_stub.cc:444`, `:1040` pass `Device.physical_id` (local id within server, **starting from 0**, first card legitimately 0, source `topo_ascend_cluster_parser.cc:959` `physical_id = deviceIn.localId`). Both share the same rank window sequence ⇒ a device with `physical_id=0` and a device with `PK=1` fall into **the same window**, addresses may overwrite each other; and `rankIdx = devPhyId - 1` with `devPhyId==0` causes uint64 wraparound, producing garbage VAs like `0xFFFFFF8000000000` (non-null so long undetected).<br>**Evidence**: `test/proxy/sqlite/aclrt_memory_stub_test.cc:199` fixture uses `physical_id = 0`; S4 initial version directly rejected `devPhyId==0`, causing 4 test cases in that suite to change from pass to fail.<br>**Same-source problem**: `src/device_arm/level1/data_op_device_stub.cc:243,267` comments already document the same kind of mispassing ("host-side TryLaunchAicpuDevProcForRank mistakenly passes GetCurrDeviceId(), which differs from GetCurrDeviceKey()..."); `docs/ranktable_rankid_device_conflict_analysis.md` also discusses rankId/deviceId specification conflicts | VA window ownership may be mismatched; directly conflicts with §4.4.4 `rankIdx := Device.physical_id - 1` definition | **S4 deliberately does not fix**, only: ① `devPhyId==0` follows degraded path, maintaining bit-for-bit identical behavior to pre-refactoring to avoid regression; ② Logs one WARN identifying the problem. Formal fix should be in **S2-2**: unify the three call sites to pass `deviceKey` (DB primary key), and assert `physical_id` is globally unique and contiguous per §4.4.4; when fixing, concurrently evaluate whether `aclrtReserveMemAddress` addresses are reverse-queried by other modules using `physical_id` windows |

---

## 7. Phased Implementation Plan

| Phase | Content | Deliverables | Dependencies | Risk |
|---|---|---|---|---|
| **S0 Pre-confirmation** | ① Test **O1** (`SET_INSTRUCTION` `insinfo.resourceAddr` source); ② **O2** (process/rank deployment form → whether watermark needs cross-process sharing); ③ **R5** (how hcomm consumes `GET_MEM_INFO` region table); ④ **R9/R10 prerequisite**: test CANN `InitRuntimePlatformInfos("Ascend960")` vs `("Ascend960DT_961BY")` return values, decide whether A6 `hardware_type` `950D-*` is a bug; ⑤ **O9** (mock-comm path `Device`/`Ccu` row source); ⑥ **O10** (`Table(Version(n))` change semantics); ⑦ **S0-6/O12** (WQE real offset + whether `ConfigJettyInfo` triggers `CCUM_CACHED_JETTY`); ⑧ Verify single-rank user_mem peak watermark on a larger sample set (measured log shows 0.405 GiB); ⑨ **A5 remaining T5**: PFE/CHANNEL/JETTY/MISSION/LOOP five offsets verified against CCU driver/hardware specification.<br>★ T6/T7 are already verified by hcomm `ccu_res_specs`, **no longer exit criteria**; the original "A5 backingSize < 200MB" hard condition is canceled with R1b downgrade (A5/A6 both are 72 MiB) | Findings memo | — | Low |
| **S1 Configuration and Model Layer** | ✅ **Two device yaml reference implementations are landed** (`asset/cluster_model/config/device/device_spec_yaml_demo_ascend950.yaml` / `_ascend960.yaml`, all 9 loader validations pass computed verification, `GetCaps()` matches `hccp_ccu_stub` V1/V2 current values 10/10 each).<br>Remaining: `src/device_spec/` library (Registry + RankAddressSpace + CcuAddressSpace + **9 validations**); CMake installation rules (§4.3.3) + 4 link integrations; CLI `device spec show/export`, `device addr decode/encode`; `test/device_spec/` UT.<br>★ yaml contains `soc.soc_version` field but **the loader completely ignores it** (decision (i), §4.7.4) | New library + UT (yaml ready) | — | Low (pure addition, does not change existing behavior) |
| **S2-1 DeviceSpec DB Persistence** | Add 6 tables (`DeviceSpec`/`Region`/`MsWindow`/`Die`/`Partition`/`Binding`, §4.8.2) + query interface (§4.8.3); `InitClusterStaticTopoData` persists to DB; `FillCcu` writes `resource_addr`; `dieId < 2U` changes to read `soc.die_num`; `cmd_table_utils.cc` table print integration | DB-queryable specification | S1 | Medium |
| **S2-2 Consumer Convergence (Read)** | Proxy L1/L2 delete literals, read `DeviceSpec`; `GetCcuMemInfo` adds version branch; runner/checker delete respective address tables, use `CcuAddressSpace`; `store_dump_shm_data` writes real values; **delete 5 `strncmp(soc_version,...)` + `ParseDevType`**; **`CcuComponerntType` → `devspec::CcuResType`** (including 8+ call sites); **delete level1 `CCU_MS_BASE=0x08000000` 5th MS address space** | 8 magic number locations cleared; 4 conversion implementations converge to 1 | S2-1 | Medium (R5/R6/R7) |
| **S3 CCU Memory Materialization** | `CcuMemoryService`; `VIR_MEM_TYPE_CCU` + cleanup path isolation; `CcuDieView` replaces arrays (**concurrently change `ccu_microcode_common_v{1,2}.h` capacity constants from `std::array` template parameters to spec-driven**, avoiding changing the same location twice); lazy creation + grayscale switch | Runner writes readable by proxy/checker | S2-2 | High (R1/R2/R3) |
| **S4 rtMalloc Protection** | `AllocVirMem` changed to **per-segment first-fit** + G0~G5 six guards (§4.5.7.3); fix latent bugs **B1** (`devPhyId==0` underflow) / **B2** (`std::map<int,...>` key narrowing) / **B3** (`size==0` returns non-null pointer); `m_deviceMemSizes` → `m_deviceMemCursors`; per-segment watermark logs. `FreeVirMem` **keeps empty implementation** and adds invariant comment (D17/O13) | Requirement 3.3 achieved (structural guarantee) | S1 | **Low** (measured peak 0.405 GiB, segment 0 has 138x headroom → zero regression, §4.5.7.4) |
| **S5 Technical Debt and Plugins** | P10 (accessor boundary checks) / P11 (trace enabled on V2 path); `GetXnAddr` `xnId*64` incorrect stride + `load_executor.cc:59`/`store_executor.cc:59` passing "value" as "id"; `DumpCcuCkeResouceInfo` using `MS_NUM` (1536) to iterate 1024-entry `cke_` out-of-bounds; `enableDump_` never assigned; delete legacy `RunVirtualRuntime` and runner `StorageManager` dead code (⚠ along with `hccl_task_thread.cc:229-255` trace channel registration); `light_performance` adds `device_spec.py`; MS flat addressing and de-interleaving unification (§4.4.3, decided together with `ccu_resource_statistics_design.md` §3.4) | — | S3 | Low |
| **S6 Specification Extension (Optional)** | `compute`/`bandwidth` persisted to DB and drives `TaskSchedulerDevice`/`ComputeDie` (production code currently **never inserts** into these two tables); converge `sim_config.json`; CCU_V3 `ccu.blocks`/`ccu.shared` sections (T12) | — | S1 | Low |
| **S7 soc_version Convergence** | §4.7.3 **phases 1-3**: delete 3 dead configurations (16 `server_or_pod` fields, `extract_soc_version` and `:637-646` dead fallback chain, ai_skills ghost) → `server_or_pod` adds `device:` reference → `cluster` deletes field + `servers_info.json` adds `device_spec` key + 19 golden regenerated → `topo_meta` adds `device:` to enable mock-comm + **O8 normalization decision lands** (along with R10 `hardware_type` correction) | Project-wide soc_version declaration locations 3 → **1** | **After main design (S1-S6) completes** | Medium (R9: 3 external APIs) |

**Recommend S1+S2 as an independently releasable milestone**: Even if S3 has not landed, S2 already eliminates "fake addresses requiring secondary conversion" (P2), "all ranks share the same base address" (P3), "4 conflicting layouts" (P4), "4 duplicate implementations" (P5), and "capacity contradictions" (P6) — five problems resolved, highest benefit/risk ratio.

**"Zero regression" proof per phase**:

| Phase | Proof Method |
|---|---|
| S1 | `hccl-vm start` behavior is byte-for-byte unchanged (pure addition, not connected to consumers); 19 golden files unchanged |
| S2-1 | `Ccu.resource_addr` changes from 0 to real value; `DeviceSpec*` 6 tables dumpable; other tables unchanged |
| S2-2 | After golden regeneration, **manually review diff** (`GetCcuMemInfo` output necessarily changes, which is the fix for §2.3 defect 1); run existing sample set on both A5 and A6 |
| S3 | New cross-process visibility test passes; `HCCL_VM_CCU_MEM_DISABLE=1` behavior reverts to S2 state |
| S4 | All existing samples fall within user_mem segment 0; allocated VAs are bit-for-bit identical to the old formula (§4.5.7.4); 19 golden files unchanged |

---

## 8. Acceptance Criteria and Test Plan

### 8.1 Acceptance Criteria

| Number | Criteria | Verification Method |
|---|---|---|
| A1 | The entire repository no longer contains `0x123123123` / `0x456456456` literals (excluding test golden files) | `grep -rn` assertion + CI gate script |
| A2 | A6 region offset literals (`0x100000`/`0x140000`/`0x4000000` and others) no longer scattered in runner/checker/proxy | Same as above |
| A3 | `ccuBase(rank,die)` is unique for any (rank,die) and falls within `[rankBase+56G, rankBase+58G)` | UT traverses rank 0..N-1 × die 0..1 assertion |
| A4 | After the runner executes `SetCKBit`, another process can read the Event bit by the same VA | New cross-process integration test (main process + runner) |
| A5 | **rtMalloc segmented mapping** (requirement 3.3): ① VAs allocated within segment 0 are **bit-for-bit identical** to the old formula `rankBase + used` (zero regression); ② When request size > segment 0 remaining, skip to segment 1, **the middle 2 GiB CCU window is never returned** (rank0: `0x8DFFFFFFF8` → `0x8E80000000`); ③ Any successful allocation `[va, va+size)` head and tail both return false through `IsInCcuWindow()`; ④ When all segments are exhausted (cumulative > 1022 GiB), return nullptr and log ERROR, log includes rank window / ccu reserved / user_mem used-total; ⑤ `devPhyId == 0` (B1) and `size == 0` (B3) both return nullptr | UT (`test/store/`) |
| A6 | `GetCcuMemInfo` returns different and configuration-consistent region tables for A5/A6 | Golden byte sequence comparison (★ this function currently has **zero test coverage**; characterization golden for "pre-change behavior" must be added first, see M3) |
| A7 | Invalid configuration causes startup failure with diagnostic information | UT covers §4.3.1 **9 validations** (including validation 3 heterogeneous window counterexample, validation 5 coverage, validation 8 `resource_space_size` upper bound) |
| A8 | `hccl-vm device addr decode <VA>` correctly outputs (rank, die, region, id); `device addr encode` is its inverse | CLI test |
| A9 | All existing checker/runner test cases pass regression | Existing UT + `install.sh` `run_test` |
| A10 | Under 128-rank topology, `/dev/shm` actual usage is measurable and within expectations (18 GiB sparse) | Stress test + `du /dev/shm/hvm_*` |
| **A11** | **soc_version declaration location count**: Phase 0 = 3 locations coexisting (cluster / server_or_pod / device) with the device side ignored by the loader; Phase 3 complete = **1 location** (device yaml) | `grep -rn "soc_version" asset/cluster_model/config/` count |
| **A12** | **At Phase 0 completion, 19 golden `servers_info.json` files are byte-for-byte unchanged** (proof of topology configuration zero changes) | `test/topo` golden full comparison |
| **A13** | `DeviceSpec*` 6 tables dumpable by `hccl-vm table show`; runner / checker / light_performance three parties read **exactly the same** specification (cross-validation with yaml passes) | Each party runs once and compares; `hccl-vm device spec export` JSON matches DB content |
| **A14** | Two device yaml `address_space` sections are **field-by-field identical** (chip-independent); `resource_space_size` identical; MS geometry parameters (`ms_size`/`ms_num`/`ccua_num`/`interleave`) identical, only `ccua_windows[].offset` differs | UT (verified with script, needs to be formalized as C++ UT) |
| **A15** | `GetCaps()` derivation results **10/10 match** with `hccp_ccu_stub.cc:69-96` (V1) / `:98-125` (V2) current values | UT (serves as cross-validation for T5/T7 correctness) |

### 8.2 Test Plan

- **UT (`test/device_spec/`, new)**
  - yaml parsing: normal / missing fields / using 32-bit type to read ≥2^32 values (should throw `TypedBadConversion`, verifying §4.2.2 parsing note) / overlapping regions / region overlapping with MS window (validation 1 combined check necessity) / interleave parameter mismatch;
  - **Validation 3 heterogeneous window counterexample**: window capacities `[256,512,512,512]` slots, summation formula passes but per-window fails → assert loader fail-fast;
  - **Validation 5 coverage**: missing segment / segment overlap / first segment offset≠0 / last segment does not reach window end / two ccu segments → each fail-fast;
  - **Validation 8**: `max(regions ∪ ms) > resource_space_size` and `resource_space_size > die_size` → fail-fast;
  - `RankAddressSpace`: requirement 3.1/3.2 six expected value exact assertions (§4.4.2 table); `userMemTotal == 0xFF80000000`; segment 0 end == CCU start, segment 1 start == CCU end, segment 1 end + 1 == rank1 start;
  - `CcuAddressSpace`: `AddrToResId` and `ResIdToAddr` mutual invertibility (random id full-region fuzz); MS interleave `id↔addr` mutual invertibility **traverse 0..1535**; boundary (id=0, id=count-1, out-of-bounds returns false not undefined behavior); `SupportsAddrDecode()` returns false for V1;
  - **`GetCaps()` cross-validation (A15)**: A5 → `hccp_ccu_stub.cc:69-96` V1 current values 10/10; A6 → `:98-125` V2 current values 10/10; including `gsa_num` "entry absent means 0" derivation (A5=3072 / A6=0);
  - **Dual file consistency (A14)**: `address_space` section field-by-field identical, `resource_space_size` identical, MS geometry parameters identical (only `ccua_windows[].offset` differs);
  - **Matching rules**: 4 real soc_version values (`Ascend950DT_95A1`/`Ascend950`/`Ascend960`/`Ascend960DT_961BY`) → correct spec; unknown value fail-fast; tied fail-fast; ★ assert spec internal `soc.soc_version` **does not participate** in matching (decision (i));
  - **Alignment test**: Feed the same yaml to both C++ and Python implementations, assert `BackingSize()`/`GetCaps()`/MS mapping outputs match (prevents the 5th implementation from drifting again).
- **UT (existing directory extensions)**
  - `test/plugin/checker/storage_manager_checker_test.cc:841-861` currently asserts `resourceBaseAddr[0] == 0x123123123`; needs to change to configuration-expected values (and upgrade to per-(rank,die) assertions);
  - ★ `test/plugin/runner/ccu_resource_manager_test.cc` currently all `Init(...)` 4th parameters are **`{}`** (empty base address list, `:59,66,73,80,92,104,116,131…`) → `ccuResourceBaseAddr_` is empty, all 4 address conversion functions return false at entry `dieId >= size()`, **function body never executes = zero coverage** (this is why the `ccu_resource_manager.cc:512,522` `// need boundary check todo` remained undetected for so long). S2-2 for the runner is **building tests from scratch**, not modifying existing tests;
  - `test/topo/sqlite/topo_golden_test.cc`: `Ccu.resource_addr` and `DeviceSpec*` 6-table persisted value golden; `:399-409` add `linkConfig("device")`;
  - `test/store/`: `AllocVirMem` segmented mapping (A5 five items) + **cross-gap counterexample** (segment 0 cursor `0xDFFFFFF00`, request `0x400` → must fall into segment 1 `rankBase+0xE80000000`, not `rankBase+0xDFFFFFF00`) + segment skip waste < request size + B1/B2/B3 three guards + `VIR_MEM_TYPE_CCU` not purged by `PurgeFreedRowsOfDevice` + check-only mode CCU row `phy_mem_id` does not point to `HcclCommPool` (R1b);
  - `test/proxy/`: ★ `GetCcuMemInfo` / `SetCcuResourceBasicInfo` **currently have zero test coverage** (verified no relevant files under `test/`); **characterization golden must be added first in S1** (recording pre-change behavior), then V1/V2 branched new golden in S2-2, making the diff reviewable and proving it is an intentional fix rather than an accidental regression (M3).
- **Integration**
  - Cross-process visibility (A4): main process Ensure → runner writes → main process reads;
  - hcomm CCU interface test cases (previously **fundamentally untestable**): `HcommCcuVariableAlloc` → launch → read back Variable;
  - A5 and A6 each run the existing sample set once;
  - **mock-comm path** (O9): under `hccl-vm mock-comm 112`, when DB has no `DeviceSpec*` rows, consumers must fall back to reading yaml and function normally.
- **Stress test**: 128 / 1024 rank shared memory actual usage and VA consumption (A10, R1, R4).
- **Gate script (CI, preventing magic number recurrence)**
  ```bash
  # Expect all 0 hits (excluding test/topo/golden and docs)
  grep -rn "0x123123123\|0x456456456" src/ include/
  grep -rn "XN_OFFSET_ADDR_A6\|XN_CKB_ADDR_A6\|XN_CCUA0_MS_ADDR_A6\|CCU_MS_BASE" src/ include/
  grep -rn 'strncmp(.*soc_version.*"Ascend9' src/
  grep -rn "findTypeByAddr\|findBaseSize\|findBaseAddr\|CcuComponerntType\|ParseDevType" src/ include/
  grep -rn "alloc_guard\|user_mem_limit\|soc_family\|soc_aliases" asset/cluster_model/config/device/
  grep -rni "wqe\|sq_buffer" asset/cluster_model/config/device/     # D16/O12: zero trace in yaml
  ```

---

## 9. Compatibility Notes

| Item | Impact | Handling |
|---|---|---|
| On-disk binary format | `ModelInfoCommInner.ccu0/1_resource_base_addr` field position/type unchanged; only value changes from magic number to real value | Old files still parse; new checker version WARNs on reading magic numbers and overrides with configuration values (transitional compatibility) |
| DB schema | **6 new `DeviceSpec*` tables** added (§4.8.2, `Scope::Static` → `hccl_sim.db`); `Ccu.resource_addr` column already exists, no change needed; new `src_type` enumeration value `VIR_MEM_TYPE_CCU` (**appended at end**, does not change existing value numbering); `Device` table **unchanged** (D13 uses Binding table association) | Tables automatically created by `Table(...)` macro metadata (sqlite/mariadb share `RenderCreateTable`), zero backend DDL; in old DBs `resource_addr = 0` is treated as "not initialized," triggering lazy creation |
| Already-allocated device VAs | `rankIdx = physical_id - 1` and `RANK_BASE`/`RANK_STRIDE` remain unchanged; user_mem expands from "56 GiB upper bound" to "1022 GiB / 2 segments" | **On-chip memory addresses completely unchanged** (measured peak 0.405 GiB all falls within segment 0; VAs within segment 0 are bit-for-bit identical to the old formula, §4.5.7.4); 19 golden files unaffected |
| `m_deviceMemSizes` → `m_deviceMemCursors` | Semantics change from "VA offset within window" to "per-segment linear watermark," and key type corrected from `int` to `uint64_t` (B2) | **Process-internal state, not persisted** → no cross-version compatibility issue; verified no external readers (only 2 locations within `AllocVirMem`) |
| Device VA reclamation | `FreeVirMem` keeps empty implementation (D17) | Consistent with current state, no behavior change; invariant and rationale in §4.5.7.5, cost of four reclamation approaches in O13 |
| hcomm interaction | `resourceAddr` / `ccu_mem_rsp.mem_va` numerical values change | hcomm side only does pass-through and offset computation, does not depend on specific values; region table unified from `DeviceSpec.regions` (D10/R5), A5 output changes with yaml initial values, S0 tests confirm hcomm consumption method |
| `soc_version` external contract | **Phase 0 zero changes** (topology configuration unchanged, values unchanged); Phase 2 may change values | 3 external APIs see §4.7.6; `Device.soc_version` DB column **permanently retained** (CLI writable, used as documentation sample); value changes must first pass R9 CANN testing |
| Grayscale rollback | — | `HCCL_VM_CCU_MEM_DISABLE=1` reverts to process-private array mode (introduced in S3, removable after S5) |
| Third parties | Secondary conversion logic for fake addresses becomes obsolete | Release notes provide `old magic number → new formula` mapping instructions, and provide `device addr decode` CLI and `device spec export` JSON to assist integration |

---

## 10. Appendix: Current Code Index (For Implementation Reference)

**Magic numbers and address tables**
- `src/proxy/level2/hccp_ccu_stub.cc:69-143` (V1/V2 caps + resourceAddr), `:201-305` (`LoadMicrocodeInstructionStub`), `:400-410` (magic number bit-shift die determination, A5-only)
- `src/proxy/level2/hccp_stub.cc:2077-2229` (`ccu_mem_info`/`GetCcuMemInfo`/`RaTlvRequest`)
- `src/proxy/level1/ccu/ccu_level1_res.cc:31-48` (constants), `:386-481` (bump alloc), `:504-578` (layout/addr/remaining)
- `src/proxy/level1/ccu/ccu_level1_res.h:38-135` (structures), `:250,293-295` (`SYNC_SLOT_NUM`/watermarks)
- `src/proxy/level1/ccu/ccu_kernel_sim.cc:60-75,1085-1113` (5th MS address space)
- `src/plugin/runner/ccu_executor/ccu_resource_manager.cc:378-531` (A6 address table + 4 conversion functions), `:45-63` (`Init`), `:134-237` (accessors, no boundary checks), `:350-376` (`GetMsAddr`/`GetXnAddr`)
- `src/plugin/checker/src/framework/task_graph_generator_v3/ccu_graph_generator_v3/ccu_all_rank_param_recorder_v3.cc:259-445` (duplicate)
- `src/plugin/runner/runner_worker/hccl_task_sequential_execute_v2.cc:35-180` (`EnsureCcuResource`)
- `src/plugin/runner/runner_utils/storage_manager.cc:309-342` (dead code path)
- `src/plugin/checker/src/utils/storage_manager.cc:757-763` (`InitCcuInfo`)
- `src/store/store_dump_shm_data.cc:454-540` (`CreateSimSynData`, `:497-498` placeholder)

**Memory and addresses**
- `src/store/store_sim_device_memory_manager.cc:47-62` (`AllocVirMem`, 1TiB window)
- `src/proxy/level2/aclrt_memory_stub.cc:129-155` (`GetRealPtrByAddr`), `:212-252` (`GetDevMappedMemPtr`), `:253-350` (`aclrtMalloc`)
- `src/store/store_sim_shm_memory_common.cc:22-64` (`AcquireDevPtrInNoHostProcess`)
- `src/store/store_sim_shm_ops.cc:91-136` (`ShmCreate`)
- `src/store/store_sim_resource_root.cc:29-42` (`HCCL_VM_RESOURCE_ROOT`)

**Specification constants**
- `include/sim_common_defs.h:34-47` (`DIE_NUM`/`CCU_*_MAX`/`MS_NUM`)
- `src/plugin/plugin_common/ccu_microcode_v1.h:20-50,847-853` (`CcuInstr` 32B)
- `src/plugin/runner/ccu_executor/ccu_microcode_common_v1.h:64-79` / `_v2.h:72-87` (`SimCcuV1`/`SimCcuV2` capacities)
- `include/sim_capacity_limits.h:21-28,133-147` (`HCCL_VM_MAX_RANKS=1024` and others)
- `include/runtime_state/sim_models.h:164-170` (`Ccu`), `:429-443` (`SimModelData`)
- `include/sim_binary_data_type_pub.h:140-196` (`ModelInfoCommInner`/`HcclVmSynData`)

**Configuration and packaging**
- `CMakeLists.txt:154-204` (`config_hcclvm_install`, add 4th install section for `config/device`)
- `asset/generate_cluster_topo.sh:100-113` (configuration directory resolution, needs `DEVICE_CONFIG_DIR`), `:345-361` (`find_server_topo`, reference resolution precedent), `:442,502` (server_list expansion), `:584-588` (`extract_soc_version`, **dead code to delete**), `:627-654` (`servers_info.json` generation + dead fallback chain)
- `asset/generate_server_topo.sh:280,300` (reads `soc_version`, logging only), `:1425-1439` (`hardware_type` derivation chain, `soc_version.upper()` fallback **unreachable**), `:1441-1448` (`topo_json` dictionary, **does not contain `soc_version` key**)
- `src/utils/sim_common_api.cc:41-78` (install root), `:173-265` (`ParseTopoMetaYaml`, YAML parsing sample, **does not read `soc_version`**)
- `src/utils/sim_yaml_config.cc` (`LoadYamlStringMap`, `HAVE_YAML_CPP` guard sample)
- `src/topo/topo_cluster_parser.cc:670-724` (`ParseServersInfoJson`, `:703-710` reads `soc_version`), `:218-309` (`ParseTopoJson`, reads `hardware_type`)
- `src/topo/topo_ascend_cluster_parser.cc:956-975` (`FillDevice`/`FillCcu`), `:1010-1167` (`InitClusterStaticTopoData`, `:1125-1133` hardcodes `dieId < 2U`), `:346-370` (`ParseRanktableAndInitCommDomain`, mock-comm path), `:382-421` (`InitCommunicationDomain`), `:1265-1355` (`ParseRanktable`)
- `src/cmd/cmd_base_utils.cc:503` (`InitClusterTopo` call site), `:523` (`InitHvmCommEnv`, mock-comm entry), `:1540` (`DeleteAll`)
- `src/cmd/subcmd/subcmd_start.cc:70,77,94` (generation → configuration directory → InitHvmEnv)
- `src/cmd/cmd_base.h:24-49` (`CommandBase` / `CommandRegistry` / `CommandAutoRegister`, integration method for new `subcmd_device`), `src/cmd/cmd_base_utils.cc:1358-1367` (`ParseCommand`)
- `include/runtime_state/db_sim_runner_ops.h:48-57` (DDL macros), `:65-…` (66 `Table(...)` list, add 6 new entries)
- `src/storage_backends/sqlite/engine/sqlite_engine.cc:984` / `mariadb/engine/mariadb_engine.cc:1063,1070` (`RenderCreateTable`, both backends share metadata)
- `test/topo/CMakeLists.txt:17-24`, `test/cmd/CMakeLists.txt:15-22`, `test/topo/sqlite/topo_golden_test.cc:374-410`
- `test/topo/golden/cluster/*/servers_info.json` (19 files, containing `soc_version`)
- `ai_skills/skills/generate-ranktable-json/references/Ascend_ranktable_file_generation.md:68,83` + `scripts/generate_ranktable.sh:287,303,451,457` (topo_meta `soc_version` **documentation ghost**)
- `ai_skills/skills/generate-cluster-json/references/ascend950_super_node_demo.yaml`, `generate-server-topo-json/references/server_topo_description.yaml` (schema documentation, phases 1-2 need synchronization)

**soc_version consumption points (S7 convergence complete list)**
- External APIs (§4.7.6, **must not break**): `src/proxy/level2/aclrt_device_stub.cc:477-497` (`aclrtGetSocName`), `:78-96` (`InitRuntimePlatformInfoFromDevice` → `fe::PlatformInfoManager::InitRuntimePlatformInfos`), `:785-800` (`ACL_DEV_ATTR_DEVICE_FORM_FACTOR` uses `hardware_type` to determine POD/A_K); `src/proxy/level2/ascend_hal_stub.cc:237-247` (`halGetSocVersion`); `src/device_arm/level2/proxy/ascend_hal_stub.cc:122-123` (`halGetChipInfo`)
- Version checks (S2-2 deletes, changes to query `DeviceSpec.ccu_version`): `src/proxy/level2/hccp_ccu_stub.cc:134,258,513`, `src/proxy/level1/ccu/ccu_level1_res.cc:494-498`, `src/proxy/level2/aclrt_new_stub.cc:47-52`, `src/plugin/runner/runner_worker/hccl_task_sequential_execute_v2.cc:103-105`
- String match tables to delete: `src/topo/topo_cluster_ir.cc:137-152` (`ParseDevType`, always returns `UNKNOWN` for real values and has no 960 branch), call sites `topo_cluster_parser.cc:710`, `topo_cluster_dumper.cc:105`
- CLI writable column (**retain**): `src/cmd/subcmd/subcmd_table.cc:47` (help sample), `src/cmd/cmd_table_utils.cc:44-50,600-604` (TEXT column special case), `test/cmd/subcmd/subcmd_table_test.cc:123`
- DB column: `include/runtime_state/sim_models.h` `Device.soc_version[32]`, `Server.hardware_type[128]`; reading `src/runtime_state/db_sim_runner_common.cc:1018-1026` (`GetHardwareTypeByDevice`)

**hcomm authoritative sources (external repository, S0/S2 verification basis)**
Paths relative to `<hcomm>/src/base_comm/resources/ccu/`:
- `ccu_device/ccu_res_specs.h:24-25` (`CCU_V1_V2_RESOURCE_SIZE = 72 MiB`, `CCU_V3_RESOURCE_SIZE = 128 MiB`), `:26-27` (`CCU_V1/V2_CCUM_OFFSET`, **CCUM and CCUA order is reversed**), `:28-31` (V3 CCUA 8 / Variable 16K / Event 2048), `:32` (`CCU_V1_WQE_BASIC_BLOCK_OFFSET = 0x1000000`, O12), `:34-46` (WQE/SQ/jetty/PFE constants), `:50-59` (INS/Address/Variable reserved sizes, per-size, `CCU_V2_RESOURCE_TOTAL_CNT_XNS_NUM = 4`), `:95-135` (`CcuResSpecInfo`, V3 private/shared partitioning), `:238` (`serveMode_` default ARMX86 "avoid using certain ccua")
- `ccu_device/ccu_res_specs.cc:353-395` (`GetXnBaseAddr`, V1/V2 branches), `:397-440` (`GetCkeBaseAddr`), `:765` (`GetResourceSpaceSize`)
- `ccu_device/ccu_block_res_strategy.cc:23` (`CCU_V1_V2_CCUA_NUM = 4`), `:253` (`avoidCcu0Flag`), `:436-438` (V1/V2 use GlobalId, V3 use RelativeId strategy)
- `ccu_device/ccu_comp/ccu_channel/ccu_jetty_ctx_mgr.cc:149-175` (`GetSqeBuffVa`), `:222` (triggered by `jettyType == CCUM_CACHED_JETTY`, **not version-gated**)
- `ccu_microcode/ccu_microcode_v1.h:28-29` (`CCU_MS_SIZE = 4096`, `CCU_MS_INTERLEAVE = 8`), `:630` (CcuV2 section `xmId` `gsaOffset[52:21]` bit field, A6 uses as address stride), `:857,859` (V1-specific GSA instruction encoders)
- `src/legacy/ascend950/unified_platform/ccu/ccu_device/ccu_res_specs_legacy.h:19-21` (A5 legacy copy, consistent with base_comm version)

**Related documents**
- `ccu_dataplane_architecture.md` (hcomm control plane/data plane layering)
- `ccu_northbound_call_sequence.md` (northbound calls and CCU→AICPU fallback)
- `docs/hccl_vm_binary_file_format.md:88-105,229-256` (syn/instr file formats)
- `docs/checker_quick_intro.md`, `docs/faq/modules/checker_faq.md` (error codes 102-113 semantics)
- `docs/light_performance/simulator_design.md:1061-1080,1534` (Python-side porting instructions and simplified declarations)
- `docs/ranktable_rankid_device_conflict_analysis.md` (rankId and deviceId specification conflict, related to §4.4.4)
- ★ `docs/ccu_resource_statistics_design.md` (CCU resource statistics feature; shares the same resource taxonomy with this design; coordination requirements see §4.7.7 / A9; its §3.4 outstanding issue is the same as §4.4.3)
- ★ `asset/cluster_model/config/device/device_spec_yaml_demo_ascend950.yaml` / `device_spec_yaml_demo_ascend960.yaml` (**landed reference implementations**, with per-field hcomm reference line numbers and 9 validation computed results)
