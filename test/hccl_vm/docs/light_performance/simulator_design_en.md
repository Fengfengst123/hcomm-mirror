# light_performance Plugin Design Document (Including Simulator Design)

> Document location: `docs/light_performance/simulator_design.md`
> Requirement source: `docs/light_performance/HCCLSIM/需求描述`
> Reference implementation: `src/plugin/checker/src/framework/task_graph_generator_v3/` (graph generation principles),
> `docs/light_performance/HCCLSIM/hccl_sim/` (simulator prototype, **authoritative baseline for the simulation module**)
> Alignment residuals and pending match list: Chapter 10
> db task / CCU instruction to JSON task type mapping table: Chapter 11

---

## 1. Requirement Background and Document Overview

### 1.1 End-to-End Requirement

Integrate light_performance (a lightweight performance simulation tool) as an independent plugin into the current project, completing the following end-to-end workflow:

```
hccl-vm workflow execution ──generates──> task sequence / instruction db files
        │
        ▼
light_performance plugin reads db files ──generates──> task graph for performance simulation
        │
        ▼
Performance simulation module performs simulation based on the task graph
        │
        ▼
Outputs performance profiling JSON file
```

### 1.2 Sub-Requirement Decomposition and Chapter Mapping

| Sub-Requirement | Content | Status | Chapter |
|--------|------|------|---------|
| Sub-Requirement 1 | Plugin engineering: command line, compilation and installation, complete plugin architecture | Command line and installation design refined (@estimate one-shot execution channel) | Chapter 3 |
| Sub-Requirement 2 | Integration with hccl-vm: db file reading, task graph generation | **Detailed design in this document** | Chapters 4 and 5 |
| Sub-Requirement 3 | CCU mode task graph generation (SQE queue + microcode instruction sequence to task graph) | **Detailed design in this document** | Chapter 6 |
| Sub-Requirement 4 | Performance simulation module (task graph parsing, performance simulation) | **Aligned and incorporated per HCCLSIM baseline** | Chapter 7 |
| Sub-Requirement 5 | Performance profiling JSON output (performance metrics, task graph) | High-level design | Chapter 8 |

> Chapter 10 records the **field gaps, type mismatches, and other conflict points** identified during this alignment (differences between the db graph generation path and the HCCLSIM simulation model) for future reference. Section 11.4 in Chapter 11 provides the mapping between v3 collected profInfo (`HcclTaskProfInfo` three fields) and JSON display types, along with the target state for the switch.

### 1.3 Scope of This Delivery

This requirement focuses exclusively on **db file reading, parsing, and the graph generation module**:

- **Input**: db files under the hccl_vm_install/data directory (format and data reference: `hccl_vm_install/data/hccl_vm_data.db`).
- **Output**: The task graph `DependencyGraph` and `SimConfig` required by the performance module (hccl_sim).
- **Reference code**:
  - Basic graph generation logic: `src/plugin/checker/src/framework/task_graph_generator_v3/task_graph_generator_v3.cc/h`;
  - CCU microcode instruction to task graph conversion: `src/plugin/checker/src/framework/task_graph_generator_v3/ccu_graph_generator_v3/`;
  - AIV mode graph generation: not considered at this time.
- **Graph generation scope (core design premise)**: The db contains **single-rank task data**, and this plugin performs **single-rank graph generation** --
  this is one of the primary differences from checker (which performs multi-rank joint graph generation). All **cross-rank task edges
  (post-wait relationships) in the original graph generation process are ignored**: only post-wait edges within a single rank are considered; cross-rank edges are not considered
  (see Sections 5.1, 5.4, and 6.6 for details).
- **Relationship with checker**: The output structure of the light_performance graph generation module is not identical to that of checker, but the basic graph generation and
  instruction conversion principles are the same; the former does not require subsequent memory conflict validation / semantic validation operations from checker, so appropriate simplifications can be made
  (see Section 5.8 and Section 6.7 for the difference comparison).

---

## 2. Plugin Overall Design

### 2.1 End-to-End Data Flow and Module Division

```
┌────────────────────────── light_performance plugin ──────────────────────────┐
│                                                                           │
│  ┌───────────┐   ┌───────────────┐   ┌───────────┐   ┌────────────────┐  │
│  │ db_reader  │──>│ graph_builder │──>│ simulator │──>│ profiling      │  │
│  │ (Ch. 4)   │   │ (Ch. 5, 6)   │   │ (Ch. 7)   │   │ (Ch. 8)        │  │
│  └───────────┘   └───────────────┘   └───────────┘   └────────────────┘  │
│   sqlite reading    SQE to Task mapping   simpy discrete event   json + trace    │
│   BLOB decoding     notify pairing        port bandwidth sharing                    │
│   operator grouping  CCU microcode expansion  on-chip memory dynamic sharing            │
│                                        eid to port resolution                       │
│  Input: data/hccl_vm_data.db (+ data/hccl_sim.db, ranktable.json)          │
│  Output: (SimConfig, DependencyGraph, TopoResolver) → makespan →            │
│        perf_profiling.json (+ graph dump: DOT/JSON/HTML)                  │
└───────────────────────────────────────────────────────────────────────────┘
```

Data contracts between modules:

```
db_reader     ──OpExecutionGroup(intermediate data model, Section 4.5)──────────>  graph_builder
graph_builder ──(SimConfig, DependencyGraph)─────────────────>  simulator   [output of this requirement]
simulator     ──result dict(tasks with start/end + makespan + ports)──>  profiling/report
```

### 2.2 Plugin Directory Structure (Source code location: `src/plugin/light_performance`)

```
src/plugin/light_performance/
├── __init__.py
├── __main__.py             # Standalone CLI entry point (Section 3.3)
├── main.py                 # @estimate plugin one-shot execution entry point (Sections 3.1/3.2)
├── manifest.json           # Plugin manifest (name=estimate; installed with package)
├── pipeline.py             # Shared pipeline: db → graph → simulation → perf_profiling.json
│                           #   (shared by run_perf_sim.py and main.py, Section 3.1)
├── db_reader.py            # Sub-Requirement 2: db reading and BLOB decoding (Chapter 4)
├── sqe_group_merge.py      # SQE batch merge preprocessing: merge UDMA data tasks within same (stream, sqeGroupId) batch into 1 entry (Section 5.1.1)
├── graph_builder.py        # Sub-Requirement 2: SQE → DependencyGraph (Chapter 5)
├── graph_dumper.py         # Task graph dump: DOT + JSON + HTML
├── ccu_graph_builder.py    # Sub-Requirement 3: CCU microcode → task graph (Chapter 6)
├── register_model.py       # CCU register/Event execution model (Section 6.4)
├── sim_config.json         # Default latency/bandwidth configuration (Section 7.2, units us/GBps; installed with package)
├── bw_profile.json         # Default bandwidth utilization configuration (installed with package)
└── simulator/              # Sub-Requirement 4: simulation module (ported from HCCLSIM, Chapter 7)
    ├── config.py           #   SimConfig (JSON units us/GBps)
    ├── task_model.py       #   Task / TaskType / TRANSFER_TYPES / INVALID
    ├── dependency_graph.py #   DependencyGraph (int stream id)
    ├── task_timing.py      #   Task latency formulas (single-point maintenance)
    ├── bandwidth_profile.py#   Bandwidth utilization table
    ├── topo_resolver.py    #   eid to ports topology resolution (ranktable.json)
    ├── input_builder.py    #   JSON task file construction + load_config
    ├── simulator.py        #   HcclSimulator / NetworkPort / _BandwidthManager
    ├── report.py           #   Text report + Chrome trace
    └── __main__.py         #   hccl_sim compatible CLI (independent of db path)
```

> The `simulator/` sub-package is ported from `docs/light_performance/HCCLSIM/hccl_sim/`, with interfaces based on the HCCLSIM baseline (Chapter 7). The `dump_graph.py` at the repository root is a one-click dump entry point
> (db to DOT/JSON/HTML, optionally with simulation timestamps).
> `manifest.json` and `main.py` (one-shot execution entry point, Chapter 3) are placed directly in this package directory;
> during installation, the entire directory becomes the `plugin/light_performance` plugin directory
> (see Section 3.1 for the installation layout and Section 3.4 for CMake rules).

### 2.3 Relationship with Existing Plugins

| Plugin | Responsibility | Relationship with light_performance |
|------|------|---------------------------|
| checker | Communication correctness validation (deadlock/memory conflict/semantics) | Reuses its graph generation principles; primary difference: checker performs **multi-rank joint graph generation**, while light_performance performs **single-rank graph generation** (no cross-rank pairing edges are created; remote synchronization tasks are retained and approximated with fixed remote latency), and does not perform validation-type post-processing |
| runner | Virtual runtime (task-by-task simulated execution of the data plane) | Also consumes opTask/ccuInstr; light_performance only reads, does not write, and does not advance isDone |
| insight | Visualization | The profiling JSON / Chrome trace from light_performance can be consumed by it (future work) |

> **Plugin tag versus directory/module name**: This plugin installs with the tag `estimate` (the host `@estimate` command channel
> and mode gate are hardcoded by tag, see Section 3.1), while the plugin directory and Python package name are both
> `light_performance` (`plugin/light_performance`) -- the tag corresponds to the "host routing identity",
> while the directory/package name corresponds to the "code organization identity".

---

## 3. Plugin Engineering Design (Sub-Requirement 1)

### 3.1 Plugin Identity and manifest.json

**The plugin tag is fixed as `estimate`** (rather than the module name light_performance): the host CLI has
hardcoded gates and path conventions for this plugin --

- The execution channel for `hccl-vm plugin run @estimate` is in `src/cmd/cmd_base_utils.cc:599-680`
  (`RunUserPlugin`) with special handling for `estimate`: only estimate mode (`sim::IsEstimateMode()`) is available,
  the plugin directory is fixed as `<installRoot>/plugin/light_performance` (tag and directory name are decoupled),
  it directly reads its `manifest.json` and synchronously executes the `entry` command (see Section 3.2 for details);
- Mode mutual exclusion: `@checker` is not available in estimate mode (`cmd_base_utils.cc:618-624`),
  `@estimate` is not available in non-estimate mode -- correctness validation and performance simulation belong to the concluding actions of two different runtime modes;
- The plugin scan/resident model (`src/plugin/hccl_plugin_manager.cc:94-169`, `src/plugin/hccl_plugin.cc`)
  matches the tag by the manifest `name` field (scans `<installRoot>/plugin`, depth <= 2).

manifest.json (both source and installation reside within the package directory: `src/plugin/light_performance/manifest.json`
to `plugin/light_performance/manifest.json`):

```json
{
    "name": "estimate",
    "version": "1.0.0",
    "entry": "python3 main.py",
    "dependency": {
        "min_core_version": "1.0.0"
    },
    "setting": {
        "db": "../../data/hccl_vm_data.db",
        "device": null,
        "iter": null,
        "sim_config": "sim_config.json",
        "bw_profile": "bw_profile.json",
        "ranktable": "",
        "out_dir": "data",
        "per_op_mode": true,
        "enable_trace": true,
        "dump_graph": false,
        "max_trace_events": 0
    }
}
```

- `name`: The plugin tag; the host routes by it (`hccl-vm plugin run @estimate`).
- `entry`: The startup command within the plugin directory. @estimate uses the one-shot synchronous execution channel (Section 3.2);
  during execution, the working directory is the plugin directory, and relative paths are resolved relative to the plugin directory.
- `setting`: Plugin-private configuration key-value pairs, read by main.py itself (following the approach of the `SettingManager` in checker
  for self-managed configuration; the Python side directly parses JSON). Path-type configurations are resolved relative to the plugin directory (that is, the package directory):
  `db` defaults to `<installRoot>/data/hccl_vm_data.db` (same source as `Loader::LoadOpTaskFile` in checker);
  `sim_config`/`bw_profile` are the latency/bandwidth and utilization configurations in the format of Section 7.2,
  defaulting to `sim_config.json`/`bw_profile.json` in the same directory as the manifest, installed with the package;
  `ranktable` defaults to `ranktable.json` in the same directory as the db;
  `device`/`iter` correspond to the CLI `--device`/`--iter`; `out_dir` is the output root directory
  (sub-directories are created under it based on the runtime timestamp); `per_op_mode` currently only supports per-operator mode (when false, a warning is issued
  and per-operator execution is used); `enable_trace`/`dump_graph`/`max_trace_events` control supplementary artifacts.

Source and installation layout (the plugin directory is the package directory, with no nesting):

```
Source = installation artifact structure (src/plugin/light_performance → hccl_vm_install/plugin/light_performance):
plugin(light_performance)/            # Plugin directory = light_performance package directory
├── manifest.json                     # Plugin manifest (name=estimate)
├── main.py                           # @estimate one-shot execution entry point (reuses pipeline.py)
├── __init__.py / __main__.py / db_reader.py / graph_builder.py / pipeline.py / ...
├── sim_config.json                   # Default latency/bandwidth (Section 7.2)
├── bw_profile.json                   # Bandwidth utilization
├── simulator/                        # Simulation sub-package (Chapter 7)
└── data/                             # Runtime output (perf_profiling.json and so on, Chapter 8;
                                      #   only produced in the installation tree; this directory does not exist in the source tree)
```

> main.py resides within the package directory but executes as a script (`python3 main.py`): it adds the **parent directory** of the package
> (`plugin/`, or `src/plugin/` in the source tree) to `sys.path` before performing
> `from light_performance.pipeline import ...` -- intra-package relative imports are unaffected,
> and the source tree and installation tree layouts are consistent.
> The `run_perf_sim.py` at the repository root (manual entry point) shares the same pipeline implementation with `main.py`:
> main.py adds "installation tree path conventions + manifest setting reading" on top of it.
> Source tree debugging still uses run_perf_sim.py (Section 3.3); in the installation tree, `hccl-vm plugin run @estimate`
> triggers main.py.

### 3.2 Invocation Chain and Process Model (@estimate One-Shot Execution)

The host side requires only one path adaptation -- the one-shot execution channel for `@estimate` is already implemented
(`cmd_base_utils.cc:599-680`); this effort changes the plugin directory in the special-case branch from `plugin/estimate`
to `plugin/light_performance` (the tag remains `estimate`; the directory is the package directory):

```
hccl-vm plugin run @estimate
    │
    ▼
RunUserPlugin("@estimate")                                  # subcmd_plugin.cc:55-57 routing
    ├─ Gate: must be in estimate mode, otherwise report error and exit              # cmd_base_utils.cc:611-617
    ├─ Read <installRoot>/plugin/light_performance/manifest.json  # Missing file or missing entry field → report error
    └─ system("cd <installRoot>/plugin/light_performance && python3 main.py")   # Synchronous blocking execution
       └─ Non-zero exit code → HCCL_VM_ERROR, command returns failure overall
```

**Client wait strategy for long-duration commands** (`StartHostClient`/`ServerListen` in `cmd_base_utils.cc`):
within a session, `hccl-vm` sends commands to the host via POSIX message queue; **the host responds only after `ParseCommand` returns (that is, after the command finishes executing)**;
after CCU CcuBuffer mode microcode expansion, the task count can reach 10^5 level (measured 192,005 tasks, software simulation approximately 54s),
far exceeding the 5s response limit for ordinary commands. To address this:

- After the client identifies `plugin run @estimate`, it adopts a long wait limit (default 30 minutes, overridable via the environment variable
  `HVM_CLIENT_RESP_TIMEOUT_MS`); during the segmented wait, it prints a
  "command is still executing on the host" message to the terminal every 30s (first at 5s), and prints a clear warning upon timeout; other commands retain the 5s limit.
- The response carries the original command string (`ACK:<cmd>`); the client drains the response queue before sending a command and only accepts responses that strictly match the current command
  -- this avoids mistakenly reading a late response from a previously timed-out command and returning prematurely (especially important when the same command is retried after interruption).
- The host prints `[HVM] @estimate performance simulation started/finished` markers to the terminal at the start and end of simulation (stdout, output along with the
  report); this mechanism ensures that **the prompt returns only after the simulation truly finishes**, and no longer produces the misleading impression that "simulation has finished".

main.py (plugin side) execution flow:

```
RunEstimate():
    setting   ← Parse the setting section of manifest.json (paths resolved relative to the plugin directory)
    db        ← setting.db (default <installRoot>/data/hccl_vm_data.db, derived as ../../data via __file__)
    config    ← setting.sim_config / bw_profile (use built-in defaults when missing, Section 7.2)
    ranktable ← ranktable.json in the same directory as db (_find_ranktable default rule; consistent with the data/ranktable.json
                written by the host InitHvmCommEnv, cmd_base_utils.cc:437-450, no configuration needed)
    groups    ← db_reader.load_op_executions(db)            # Chapter 4
    topo      ← TopoResolver(ranktable)                     # Section 7.6
    for group in groups:                                    # Per-operator (Section 5.1)
        config, dep_graph = graph_builder.build(group)      # Chapters 5 and 6
        result = HcclSimulator(config, dep_graph, topo).run()   # Chapter 7
    profiling.dump(<plugin>/light_performance/data/<timestamp>/perf_profiling.json)   # Chapter 8
```

Comparison with the checker resident process model (stdin single-line JSON command loop) and the basis for the choice:

| Dimension | checker (resident model) | estimate (one-shot model) |
|------|--------------------|----------------------|
| Startup method | fork+pipe resident during `plugin install` (`hccl_plugin.cc:74-156`) | `cd && entry` synchronous execution during `plugin run` (`cmd_base_utils.cc:655-660`) |
| Trigger protocol | stdin JSON: `{"action":"status","payload":{"status":"finish","data_id":...}}` | No protocol: entry is the start; input paths all use conventions + manifest setting |
| data_id | Carried in message (generated by the host `DumpData`) | Not passed; the output directory uses the runtime timestamp instead (Section 10.3 #13) |
| Lifecycle | Reclaimed upon stop command/pipe EOF/host exit | Process exits naturally; exit code is the execution result |
| Adaptation reason | Validation must be triggered by the host at the end of the workflow, and may run multiple times | Simulation is an offline batch process: consumes the already-persisted db, runs once and exits |

> **Evolution direction** (not implemented in this phase): If "automatic simulation triggering by the host at workflow end" is needed in the future (resident model),
> main.py can add a stdin command loop to reuse the same pipeline -- the host message channel
> (`SendMessageToPlugin`) and resident startup (`RegisterPlugin`) are both ready; at that point, only
> the @estimate special-case branch in `RunUserPlugin` needs to be switched back to the generic message path.

Data contracts (fixed conventions within the installation tree, all aligned with existing mechanisms):

| Path | Source | Description |
|------|------|------|
| `data/hccl_vm_data.db` | Written by the host during workflow execution | Main simulation input; estimate mode produces **single-rank task data** (phantom peer), naturally aligned with the single-rank graph generation premise in Sections 4.6/5.1 |
| `data/ranktable.json` | Written by `InitHvmCommEnv` | eid to port resolution (Section 7.6); `_find_ranktable` defaults to the same directory as the db, no explicit configuration needed |
| `plugin/light_performance/sim_config.json`, `bw_profile.json` | Installed with the package directory by CMake (Section 3.4) | Section 7.2 format (units us/GBps) |
| `plugin/light_performance/data/<timestamp>/` | Created at runtime by main.py | Output directory. **Must not reside under `data/`**: the host runs `remove_all("data")` at each start (`src/main.cc:32`), so the contents of the data directory do not persist across sessions |

### 3.3 Command Line Design

Host integration mode (`plugin run` is the simulation entry point, replacing manual execution of run_perf_sim.py):

```bash
# ① Start the environment in estimate mode (choose one)
hccl-vm start <topo> --estimate             # Interactive mode, PS1 displays (estimate)$>
hccl-vm run <topo> --estimate <sample command>     # One-shot mode
# (Within an already started session, switch with hccl-vm mode estimate, subcmd_mode.cc)

# ② Run workloads → produce data/hccl_vm_data.db (single-rank task list)

# ③ One-click performance simulation (only available in estimate mode, Section 3.2 gate)
hccl-vm plugin run @estimate
# Artifacts: plugin/light_performance/data/<timestamp>/perf_profiling.json (+ trace / graph dump)
```

- The one-shot execution channel does not pass through command line arguments; simulation parameters (target device, opIter filter, trace/dump switches, and so on)
  are configured via the manifest `setting` section (Section 3.1); re-run `plugin run @estimate` after changing the configuration for changes to take effect.
- `plugin install @estimate` can also be executed (through the `RegisterPlugin` resident startup channel); main.py finishing and exiting after one run is acceptable degraded behavior (Section 10.3 #14); regular usage does not require install.

Standalone debug mode (does not depend on the host, convenient for offline db analysis; already implemented):

```bash
# Simulation + report (optional trace export, task graph dump)
python3 -m light_performance \
    --db hccl_vm_install/data/hccl_vm_data.db \
    [--config sim_config.json] [--bw-profile bw_profile.json] \
    [--ranktable ranktable.json] [--device 1] [--iter 0] \
    [--trace trace.json] [--dump-dir graph_dump] \
    [--max-trace-events 0] [-v] [-s]

# One-click simulation (run_perf_sim.py, outputs perf_profiling.json; shares pipeline with main.py)
python3 run_perf_sim.py hccl_vm_install/data/hccl_vm_data.db \
    [-o perf_profiling.json] [--config ...] [--bw-profile ...] [--trace-dir ...] [--dump-dir ...]

# One-click task graph dump (DOT + JSON, optional HTML; --no-sim skips simulation)
python3 dump_graph.py hccl_vm_install/data/hccl_vm_data.db [--html] [--no-sim]
```

| Parameter | Description |
|------|------|
| `--db` | Path to hccl_vm_data.db (required) |
| `--config` | Latency/bandwidth configuration (Section 7.2 format, units us/GBps; uses built-in defaults by default) |
| `--bw-profile` | Bandwidth utilization configuration (default utilization is 100%) |
| `--jetty-num` | Overrides the UB multi-channel (jetty) count stored in the db, used for multi-jetty utilization scaling (defaults to each operator's `opDetails.ubMultiChannelNum`, see Section 7.2) |
| `--ranktable` | Topology file, used for eid to physical port resolution (defaults to `ranktable.json` in the same directory as the db; when network tasks exist but no topology is available, **simulation is skipped and only dump is performed**) |
| `--device` | Target device (deviceId) for single-rank graph generation; defaults to the first device in opDetails, only needed when the db contains multi-device data |
| `--iter` | Simulate only the operator of the specified opIter (default: all) |
| `--trace` | Export Chrome tracing JSON (for multiple operators, split into files by `_<commId>_<opIter>`) |
| `--dump-dir` | Task graph dump directory (outputs .dot/.json/.html per operator, including simulation timestamps) |
| `-v` / `-s` | Task-level details / summary only |

### 3.4 Compilation and Installation Design (hccl_vm_install/plugin/light_performance)

Add an install rule in `src/plugin/CMakeLists.txt` (following the precedent of the directory installation for insight;
Python plugins do not require compilation targets. The plugin directory is the package directory, and manifest.json/main.py/default configurations all reside
within the package directory, so a single DIRECTORY rule can carry everything):

```cmake
# estimate performance simulation plugin: plugin directory is the light_performance package directory, install entire directory
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/light_performance"
    DESTINATION "plugin"
    FILE_PERMISSIONS OWNER_READ GROUP_READ WORLD_READ
    PATTERN "__pycache__" EXCLUDE
    PATTERN "data" EXCLUDE)      # Runtime output directory is excluded from the installation package
```

Comparison with existing plugin installation methods:

| Plugin | Installation Method | Artifact |
|------|---------|------|
| checker | `install(TARGETS checker DESTINATION plugin/checker)` + `install(FILES manifest.json)` (`src/plugin/checker/CMakeLists.txt:67-82`) | Compilation artifact + manifest |
| insight | `install(DIRECTORY insight DESTINATION plugin/visualization PATTERN frontend EXCLUDE)` (`src/plugin/CMakeLists.txt:26-39`) | Python directory + manifest |
| **estimate (this effort)** | `install(DIRECTORY light_performance DESTINATION plugin)` (single rule; manifest/main.py/default configurations are all installed with the package directory) | `plugin/light_performance/`: Python package + manifest (plugin directory and package directory are unified, no nesting) |

Build and verify: After `bash build.sh` (internally `cmake && make && make install`, with the install prefix defaulting to
`hccl_vm_install/` within the repository, `CMakeLists.txt:89-92`), check the installation artifacts:

```
hccl_vm_install/plugin/light_performance/
├── manifest.json
├── main.py
├── __init__.py / __main__.py / db_reader.py / graph_builder.py / pipeline.py / ...
├── sim_config.json / bw_profile.json
└── simulator/ ...
```

- Packaging path: `build.sh --pkg` packages the entire `hccl_vm_install` into a tar archive, and the estimate plugin is distributed along with it.
- Dependencies: python3 (>=3.8, including the sqlite3 standard library) and **simpy** (discrete event simulation engine, current environment
  4.1.2, imported in `simulator/simulator.py`); the deployment environment must have them pre-installed (the insight plugin also depends on python3,
  the convention is consistent), see Section 10.3 #16.

---

## 4. db File Reading and Parsing (Sub-Requirement 2 - Upper Half, Focus of This Effort)

### 4.1 db Location and Opening

- The path is fixed as `<installRoot>/data/hccl_vm_data.db` (consistent with `Loader::LoadOpTaskFile` in checker).
- Open in read-only mode: Python `sqlite3` standard library, `mode=ro` URI, to avoid write conflicts with a running hccl-vm:
  ```python
  conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
  ```
- Auxiliary database `hccl_sim.db` (same directory): used only to resolve `opDetails.commId` to communicator identity
  (`Communicator` table, `comm_id[128] + comm_hash` as unique identifier), optional.

### 4.2 Table List and Association Model

hccl_vm_data.db is created by `OpDbOps::InitSchema()` (`src/runnerdb/modeldb/db_hccl_op_db_ops.cc:76-184`);
the tables consumed by light_performance are as follows:

| Table | Primary Key/Association | Content | Purpose |
|----|----------|------|------|
| `opDetails` | `id`; `pid` to opTask table name | One row per operator per rank: pid/deviceId/rankId/commId/opIter/syncIter/streamId(main stream)/root/rankSize/srcRank/dstRank + `opDetail` BLOB (operator parameters) + `algorithmName` (algorithm selection result, intercepted and backfilled via dual DFX hijack/log paths, see note below) + `ubMultiChannelNum` (UB multi-channel/jetty count, proxy level0 stub reads environment variable `HCCL_UB_MULTI_CHANNEL_NUM` and stores it; defaults to 1 when not configured/invalid, see Section 7.2 for multi-jetty scaling) | Operator grouping, main stream identification, operator parameters, algorithm selection results, bandwidth utilization scaling |
| `opMemInfo` | `opDetailId` → opDetails.id | Addresses and sizes for the input/output/ccl memory segments | Direction determination (Section 5.5), data volume validation |
| `opTask_P_<pid>` | `opDetailId` → opDetails.id | SQE task queue: deviceId/streamId/taskType/isDone + `opTaskMeta` BLOB | **Main input for graph generation** |
| `ccuChannels` | `(deviceId, dieId, channelId)` | CCU channels: src/dst die/device/rank + leid/reid(16B EID) + protocol + jettyNum + jettyId array | CCU channel to peer/jetty mapping (Chapter 6) |
| `ccuInstrRes` | `(deviceId, dieId)` | Microcode instruction space per die: instrCount + instrSpace BLOB (32K x 32B) | CCU microcode source (Chapter 6) |
| `ccuInstr` | `ccuInstrResId` → ccuInstrRes.id | One row per microcode delivery (startId/instrInfoSize) | Used only to detect "new microcode delivered"; decoding does not depend on it |
| `syncRecords` | -- | Host-side stream synchronization records | Not used in this effort |
| `JettyMaps` | `opDetailId` | eid mappings recorded per operator | Backup (not used when ccuChannels covers it) |

> Table structure definitions are in `include/runnerdb/sim_op_db_types.h`; writer-side semantics are in `src/runnerdb/tracedb/db_sim_op_db_ops.cc`.

> **Source of the algorithmName column (algorithm selection result interception)**: Dual-path interception, both using last-wins to backfill
> `sim::UpdateOpDetailAlgName`, serving as mutual backups and writing the same value.
>
> 1. **DFX hijack stub (primary path)**: After algorithm selection completes, the workload calls
>    `HcclDfxRegOpInfoByCommId(commName, &HcclDfxOpInfoCompat)` per operator (in the hccl repository,
>    op_common.cc), where its `algTag` carries the selected algorithm name (format
>    `{opType_commName}_{algName}_{host|device}[_{dataType}...]`, graph mode
>    `Graph_{algName}_{...}`). Level=2 mode (default) only hooks level0+level2;
>    the level1 dfx_stub is not in the process, so `src/proxy/level2/hcomm_api_stub.cc` uses
>    level2 strong symbols to preempt resolution (preload order precedes the real
>    libhcomm.so loaded by the workload through the Python plugin chain), parses the algorithm name from the algTag, backfills it, and then
>    **forwards to the real implementation via `DlsymRealWithFallback` (RTLD_NEXT + libhcomm.so SONAME fallback + dladdr
>    self-call prevention), transparently passing the return value without changing call semantics**. The structure layout is replicated from
>    hccl_res_dl.h (`static_assert` locks the algTag offset at 68); content validation
>    (commId prefix + host/device suffix) filters out op-begin registrations that lack an algorithm name and
>    provides a fallback for workload header drift. In Level=1 mode, this stub is shadowed by the level1 dfx_stub (behavior
>    same as the old version), and interception falls back to the log path.
> 2. **Log interception (backup path)**: The tool is the implementer of the workload log underlying interface (the workload
>    `DlogRecord` weak symbol resolves to `src/proxy/level2/log_stub.cc`). The level0 stub
>    `RecordOpDbInfo` first inserts the opDetails row (setting the runnerdb global `g_currOpDetailId`) and then calls
>    the real workload; the INFO-level algorithm selection log output by the internal selector (`[HcclGetAlgRes] algName=` /
>    `SelectMinCost: selected algName=` / old selector `the selected algo type is ` -- three types of markers,
>    last-match-wins overwrite for the same operator) is matched and extracted via the `DlogPrintStub` entry. Interception occurs before the message enters the tool log output
>    (immune to `log_config.yaml` levels); the proxy layer also provides an `acllogCheckDebugLevel`
>    always-pass stub (immune to workload-side log level thresholds).
>
> Old databases are automatically upgraded by `OpDbOps::InitSchema()` via
> `ALTER TABLE` to add columns; the plugin reader falls back to empty values for old databases without the columns.

**Operator grouping model**: One operator instance = rankSize opDetails rows under the same `(commId, opIter)` (one per rank),
each with an associated opMemInfo and a set of tasks (`WHERE opDetailId=?`, the table name is derived from the `pid` column of that row as `opTask_P_<pid>`,
referencing `QueryCompositeOpDetailByOpIter` in checker).

### 4.3 opTask Dynamic Tables and Multi-pid Handling

- Each task-writing process creates its own table `"opTask_P_" + getpid()`; AICPU device-side processes use `getppid()` to share the same table with the host.
- **Discover all task tables** (enumeration is mandatory; multi-pid is the norm; early sample databases already had two tables `opTask_P_477776`/`opTask_P_477780`; current samples have a single table):
  ```sql
  SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'opTask_P_%'
  ```
- Consumption order: within the same operator, ascending by `id` (= SQE delivery order, which is the basis for first-come-first-served notify pairing).

### 4.4 BLOB Decoding Specification (pack(1), Little-Endian)

All BLOBs are raw memcpy of C structures (no serialization framework); the Python side uses the `struct` module for decoding.
The following layouts have all been verified against `hccl_vm_install/data/hccl_vm_data.db` (from v5 collection onward, the sample
opTaskMeta length is 190B; historical samples were 189B/188B).

#### 4.4.1 opTaskMeta (190 Bytes)

The structure definition and writer side share: `include/sim_profiling_defs.h` (single-source definition for profiling reconciliation information)
and `src/plugin/checker/header/internal/task_meta_defs.h` (HcclTaskMetaData).

v5 collection structure (compared to the v4 189B version): `taskSrc` (1 byte) is added after `sqeGroupId` --
task source marker (`HcclTaskSrc`: 0=default SQE/aclrt native path, 1=DPU that is host CPU RoCE
`ibv_dpu_stub` collection path), used by the performance simulation DPU sequence merge pass to precisely split by source
(Section 5.1.2); taskData start offset shifts from 69 to 70.

v4 collection structure (compared to the v3 188B version): `sqeGroupId` (1 byte) is added after `jettyId` -- UDMA
SQE batch grouping marker (all WQEs within one doorbell share the same group), used by the performance simulation entry point for batch merging
(Section 5.1.1); taskData start offset shifts from 68 to 69. Group number semantics: **circularly incrementing doorbell batch number 1..255**
(wraps at 255; the collection side `device_sqe_parse_stub.cc` maintains the global `g_currSqeGroupId`,
`% 255 + 1`) -- batch numbers are unique within a 255-element window; doorbells from other operators are not visible in the current operator task list
but also consume batch numbers, so the same stream and same number within the list must belong to the same doorbell; **0 is reserved for "unmarked"**
(HcclTaskMetaData construction default); the counter never takes 0; the merge entry point treats `sqeGroupId <= 0`
as non-participating (Section 5.1.1).

v3 collection structure (compared to the v2 185B version): `HcclTaskProfInfo` sub-structure (3 consecutive bytes,
treated as an opaque whole by checker/runner) is added after `taskType`: `profTaskType` (aligned with hcomm TaskParamType value range 0~22) and
`linkType`/`transportType` (aligned with hcomm DfxLinkType value range 0~9 / DfxTransportType value range 0~4,
used for reconciliation with profiling data); taskData start offset shifts from 65 to 68.

> **Implementation status**: `db_reader.py` automatically identifies the layout by BLOB length -- `>= 190B` follows v5
> (`<BBBBQIQQIBB16s16s`, taskSrc decoded into `HcclTaskMetaData.src_type`; this branch
> **must be checked before the v4 condition**, otherwise 190B would first match the `>= 189` v4 branch, causing eid/taskData
> offset misalignment), `>= 189B` follows v4
> (`<BBBBQIQQIB16s16s`, sqeGroupId decoded into `RawTask.sqe_group_id`, src_type=0),
> `188B` follows v3 (`<BBBBQIQQI16s16s`, sqe_group_id filled with -1, not participating in merging),
> `185~187B` follows old v2 (`<BQIQQI16s16s`, profInfo filled with 0xFF=UNKNOWN) for backward compatibility,
> `< 185B` is discarded with a warning. The consumption of profInfo (`task_type` display name + link/transport)
> **is implemented** (#18, Section 11.4).

v2 collection structure (compared to the earlier 169B version): `lcEid[16]` (explicit local EID) is added to the header;
taskData members previously had `srcRankId/dstRankId` fields added.

> **rank field rollback: srcRankId/dstRankId are deprecated (current convention)**: The
> `srcRankId/dstRankId` fields previously added to taskData members have been rolled back and removed; `srcDeviceId/dstDeviceId` is used uniformly -- the collection side
> (estimate mode, `device_sqe_parse_stub.cc` and so on) writes the communicator rankId directly into
> the `srcDeviceId/dstDeviceId` fields, that is, **the deviceId field carries the rankId** (when the peer rank is unresolved,
> it is `0xFFFFFFFF`). `db_reader.py` directly consumes `src_rank_id/dst_rank_id` after unpacking
> (reads `srcDeviceId` as `srcRankId`). This is a hard switch: old BLOBs collected before the field rollback
> (dual rank/device field layout) are no longer supported.

```
Header 70B:  struct.unpack_from("<BBBBQIQQIBB16s16s", blob, 0)
┌──────┬──────┬──────────────┬────────┬───────────┬──────────────────────┐
│Offset│Length│ Field         │ Type   │ Sentinel  │ Description           │
├──────┼──────┼──────────────┼────────┼───────────┼──────────────────────┤
│ 0    │ 1    │ taskType     │ u8 enum│ --         │ Redundant with table column taskType │
│ 1~3  │ 3    │ profInfo     │ sub-struct│ 0xFF      │ HcclTaskProfInfo:      │
│      │  1   │ .profTaskType│ u8 enum│ 0xFF      │ Aligned with hcomm TaskParamType │
│      │  1   │ .linkType    │ u8 enum│ 0xFF      │ Aligned with hcomm DfxLinkType │
│      │  1   │ .transportType│u8 enum│ 0xFF      │ Aligned with hcomm DfxTransportType │
│ 4    │ 8    │ commId       │ u64    │ --         │ Communicator ID              │
│ 12   │ 4    │ rankId       │ u32    │ 0xFFFFFFFF│ Backfilled via device to rank when default │
│ 16   │ 8    │ deviceId     │ u64    │ --         │ Physical device ID            │
│ 24   │ 8    │ streamId     │ u64    │ --         │ Stream ID                 │
│ 32   │ 4    │ jettyId      │ u32    │ 0xFFFFFFFF│ Invalid = not network           │
│ 36   │ 1    │ sqeGroupId   │ u8     │ 0         │ Doorbell batch number 1..255 circularly incrementing (new in v4, Section 5.1.1); 0 = unmarked│
│ 37   │ 1    │ taskSrc      │ u8 enum│ 0         │ Task source (new in v5, Section 11.4.6): 0=default, 1=DPU(host ibv)│
│ 38   │ 16   │ rmEid[16]    │ bytes  │ All 0      │ Remote EID              │
│ 54   │ 16   │ lcEid[16]    │ bytes  │ All 0      │ Local EID (new in v2)     │
│ 70   │ 120  │ taskData     │ union  │ Interpreted by taskType, see below          │
└──────┴──────┴──────────────┴────────┴───────────┴──────────────────────┘
```

taskData union internal layout (relative offset 69):

| taskType | Member | Format String | Fields (in order) |
|----------|------|--------|------|
| 0/1 (notify) | `NotifyTask` | `<IQIBB` | srcDeviceId(carries rankId), notifyId, dstDeviceId(carries rankId), notifyCount, protocol |
| 3 (transMem) | `TransMemTask` | `<IQIQQB` | srcDeviceId(carries rankId), srcOffset, dstDeviceId(carries rankId), dstOffset, len, protocol |
| 2 (reduce) | `ReduceTask` | `<IQIQQBBBB` | srcDeviceId(carries rankId), srcOffset, dstDeviceId(carries rankId), dstOffset, dataCount(actually byte size), dataType, reduceOp, protocol, resv |
| 4 (ccu) | `CcuTask` | `<BBHHHII` + `<13Q` | dieId, missionId, timeout, instStartId, instCnt, key, argSize, args[13] |
| 5 (aiv) | `AivTask` | `<Q` | launchIdx |
| 6 (sync) | `SyncStreamTask` | `<Q` | syncIdx |
| 7 (writeWithNotify) | `WriteWithNotifyTask` | `<IQIQQQBBBBB` | srcDeviceId(carries rankId), srcOffset, dstDeviceId(carries rankId), dstOffset, len, notifyId, isReduce, dataType, reduceOp, protocol, notifyCount |

> Both `len` and `dataCount` are **byte sizes** (the reduce field name has historical reasons; it is actually used as size).
> `WriteWithNotifyTask.len` is also a byte size; `isReduce=1` indicates that the write payload is reduction data
> (`dataType`/`reduceOp` are valid in this case), mapped as a single `Write_With_Notify` task (Section 5.2).

#### 4.4.2 taskType Enumeration Full Table

`HccLTaskMetaType` (`include/sim_common_defs.h:184-193`):

| Value | Enumeration | light_performance Handling |
|----|------|----------------------|
| 0 | `NOTIFY_WAIT` | to local_wait / remote_wait (Section 5.2) |
| 1 | `NOTIFY_RECORD` | to local_record / remote_record |
| 2 | `REDUCE` | to memcpy / local_copy (reduction computation approximation, Section 5.7) |
| 3 | `MEM_CPY` | to memcpy / local_copy (determined by jettyId) |
| 4 | `CCU_GRAPH` | to CCU subgraph expansion (Chapter 6) |
| 5 | `AIV_GRAPH` | Not supported at this time; skip with warning |
| 6 | `SYNC_STREAM` | to Skip, do not generate node (Section 5.6) |
| 7 | `WRITE_WITH_NOTIFY` | to Single task mapping `Write_With_Notify` (network write + attached notify, not split, Section 5.2) |

#### 4.4.3 opDetails.opDetail (28 Bytes)

`OpDetails` (`task_meta_defs.h:141-165`): `<HHH` header + union:

```
opType u16@0 | dataType u16@2 | reduceType u16@4
opV1: count u64@6 | dataType u16@14 | reduceOpType u32@16          (total 28B)
opV2: sendCount u64@6 | sendDataType u16@14 | recvCount u64@16
      | recvDataType u16@24 | extInfo u16@26                       (AllToallV type)
```

opType is `HcclCMDType` (ALLREDUCE=2, ALLGATHER=6, REDUCE_SCATTER=7, ALLTOALL=10, and so on;
the complete enumeration is in `src/plugin/checker/header/external/hccl_types.h:162-189`), used for operator annotation in profiling reports.
The `opExtInfo` BLOB (optional) stores V operator counts/displs, AllToallV send matrices, and so on; this effort only passes it through.

V1/V2 layouts are distinguished by opType (level0 stub writer-side convention, `hccl_op_stub.cc`):
**V2 is only for AllToAll(10)/AllToAllV(8)/AllToAllVC(9)** (`BuildOpDetailsV2`, where sendCount/recvCount
are the total send/receive element counts for this rank); **AllGatherV(16)/ReduceScatterV(17) use V1 + opExtInfo**
(count = this rank-side element count). The db_reader selects the layout by opType during decoding; V2 operators decode the send and receive
side element counts and types (`send_count/recv_count/send_data_type/recv_data_type`);
`data_size/data_type` follow the send side to maintain backward compatibility with old consumers; `reduce_op` is set to 0.

#### 4.4.4 ccuChannels BLOB

| Field | Format | Description |
|------|------|------|
| `leid` / `reid` | 16B raw bytes each | Source/destination endpoint EID (`URMA_EID_LEN=16`) |
| `jettyId` | `jettyNum × u32` little-endian | Contiguous jetty number array (C structure reserves `u32[64]`; sample measurement: jettyNum=1, jettyId=(1024,)) |
| `protocol` | int column | CommProtocol: HCCS=0/TCP=1/ROCE=2/UB_CTP=3/UB_TP=4/PCIE=5/SIO=6 |

Note: The `srcRankId/dstRankId` columns are actually filled with device_id (channel establishment precedes communicator creation; see the writer side
`hccp_ccu_stub.cc:401-405` comments); the rank semantics require correction via opDetails.

#### 4.4.5 ccuInstrRes instrSpace

- `instrSpace` = the **complete microcode instruction space snapshot** for one die: `32K entries × 32B = 1 MiB`
  (`CCU_INSTRUCTION_NUM=32*1024`, `CCU_INSTRUCTION_SIZE=32`, `include/sim_common_defs.h:35-36`).
- The i-th instruction is at byte offset `i*32`; `instrCount = max(historical value, startId+instrCnt)`, **allowing gaps**.
- When reading, only the first `instrCount` entries are fetched; the CCU task `instStartId/instCnt` values are directly used as index ranges.
- The `ccuInstr` table is only for runtime awareness (runner polls for changes in `COUNT(ccuInstr)+COUNT(ccuInstrRes)`);
  light_performance decoding does not depend on this table.

### 4.5 Intermediate Data Model (db_reader Output)

```python
@dataclass
class OpExecutionGroup:          # One operator instance (single-rank perspective: only target device task data)
    comm_id: int
    op_iter: int
    op_type: int                 # HcclCMDType
    rank_size: int
    main_stream_ids: dict[int, int]      # deviceId → main stream streamId (opDetails.streamId)
    tasks: list[RawTask]         # All SQEs for this operator, globally ordered
    mem_info: dict[int, MemInfo]         # deviceId → input/output/ccl ranges
    channels: dict[tuple, ChannelInfo]   # (srcDev, srcDie, channelId) → channel info
    instr_space: dict[tuple, list[bytes]]# (deviceId, dieId) → 32B instruction list
    algorithm_name: str          # Algorithm selection result (opDetails.algorithmName, backfilled via DFX/log dual-path interception; empty string for old databases)
    ub_multi_channel_num: int    # UB multi-channel (jetty) count (opDetails.ubMultiChannelNum; old databases/zero values default to 1)
    send_count / recv_count: int # Send/receive side element counts (V2 layout operators; consistent with count for V1)
    send_data_type / recv_data_type: int # Send/receive side data types (same as above)

@dataclass
class RawTask:
    global_id: int               # Global sequence across opTask_P_* tables (sorted by table pid value + in-table id)
    op_detail_id: int
    device_id: int
    stream_id: int
    meta: HcclTaskMetaData       # 4.4.1 decoding result (dataclass form)
```

`meta` contains v2 new fields: header `lc_eid` (explicit local EID, preferred for local eid resolution in Section 5.5);
NotifyTaskData/TransMemTaskData both carry `src_rank_id/dst_rank_id`
(unpacked from the rankId value carried in the deviceId field under the estimate collection convention; see Section 4.4.1 for rank field rollback).

### 4.6 Data Integrity Validation

For each `OpExecutionGroup`:

1. **Single-rank filtering (design premise)**: The performance plugin only consumes **single-rank task data** (the sample db only contains device 1,
   which is the design norm rather than data loss). If the db contains multi-device data, filter opDetails and their tasks by the target deviceId (`--rank`, defaulting to the first device);
   `rankSize` is only passed through as metadata; it is not required that `len(opDetails rows) == rankSize`.
2. **notify peer** (no backfill):
     - Estimate collection convention (`device_sqe_parse_stub.cc`, consistent for both task types): the notify
       `srcDeviceId` (carrying rankId) = record side, `dstDeviceId` (carrying rankId) = wait side,
       both explicitly filled by the launcher; the peer rank is `0xFFFFFFFF` when unresolved.
     - The early `dstDeviceId == 0` backfill fallback has been removed along with the rank field rollback (the 0 sentinel conflicts with valid
       rank 0, and backfill is unnecessary since the launcher always explicitly fills it).
     - The notify carried by `WRITE_WITH_NOTIFY` (7) targets the **write peer** (the wait side is not in this db, per the single-rank
       simulation premise) and does not participate in local pairing; its `dstDeviceId` (carrying rankId) is the write destination.
     - Note: The runner/checker consumer side (`DecodeAndExpandTaskMeta`) splits 7 into
       [write, NOTIFY_RECORD] two sub-tasks at the decoding boundary; light_performance **does not split** -- single-task mapping
       `Write_With_Notify` (Section 5.2), ensuring a one-to-one correspondence between simulation tasks and db tasks.
3. **Decoding length validation**: `len(opTaskMeta) >= 185` (auto-adapts to v4/v3/v2 layout by length, Section 4.4.1),
   `len(opDetail) >= 28`,
   `len(instrSpace) >= instrCount*32`; discard the row and issue a warning if not satisfied
   (checker performs direct memcpy; this approach is more lenient).

### 4.7 Decoding Example (Measured hccl_vm_data.db)

Sample db (REDUCE_SCATTER, opType=7, dataType=4/FP32, count=8, rankSize=2, devType=6/950)
measured decoding results:

- `opTaskMeta` length is consistently 188B (v3 collection sample; from v4 collection onward it is 189B, Section 4.4.1); taskType distribution: 4 (CCU_GRAPH) and 6 (SYNC_STREAM) alternating.
- opDetails only has device 1 (rank 0) which means single-rank data (design norm); main stream is stream 2.
- Task sequence (per opIter main stream):

```
stream 2 (main stream):  CCU_GRAPH(die=1, mission=0, instStartId=14, instCnt=100, argSize=13)
                   SYNC_STREAM(syncIdx=n)          # Microcode delivery + stream synchronization
```

- The CCU task args match the data directory file `sqe_info_device_1_die_1_mission_0_startId_14.txt`
  item by item (args[0]=281475405602881, args[1]=281475405602945 ...);
  `args[0]/args[1]` are measured to be the inputAddr/outputAddr from opMemInfo.
- `ccuInstrRes` (dev 1, die 1) instrCount=114 = instStartId(14) + instCnt(100).
- In early 169B samples containing MEM_CPY/NOTIFY: MEM_CPY srcOffset/dstOffset can both fall within
  the opMemInfo input/output/ccl ranges (`input ⊃ output`; the ReduceScatter output is the first
  1/rankSize segment of the input); network copies with jettyId also show offsets as "local range addresses"
  (communicator memory symmetric layout means the peer buffer has the same VA as the local side), so **direction determination cannot rely on addresses alone**
  (see Section 5.5). This conclusion also applies to v2 data.

#### 4.7.1 v2 AICPU Mode Sample (opTask_P_118800)

REDUCE_SCATTER, rankSize=8, count=32768 (FP32, output=128KB), 76 tasks,
stream 2 (main stream) / 6 (scheduling stream) / 7..13 (one stream per peer). Key differences from the CCU mode sample:

- taskType distribution: NOTIFY_WAIT x 30, NOTIFY_RECORD x 30, MEM_CPY x 8 (7 network write
  jet=1..7 + 1 local), REDUCE x 7 (local), SYNC_STREAM x 1; no CCU_GRAPH, `ccuChannels` is empty.
- **Direction is entirely determined by the rank dimension** (Section 5.5 rule 1): all 7 network MEM_CPY tasks have
  `srcRank=0/dstRank=1..7` (write); the task offsets do not intersect with the opMemInfo ranges (range inference fails),
  `direction_inferred=0`.
- **notifyId is reused across ranks**: For example, `RECORD(nid=30, dst=dev2)` (wait side is on the peer, skipped) and
  local `WAIT(nid=30)` (record side is on the peer, not in this db) appear in pairs, with 7 x 2 in two rounds;
  16 locally paired pairs (nid 3/13 main to scheduling stream, 15..27 scheduling stream to peer stream, 6..12 peer stream to scheduling stream),
  14 unpaired local WAITs which are charged at `remote_wait_delay` (Section 5.4).
- lcEid/rmEid match ranktable.json verbatim (`...0040...01010001` to port `1/0`, and so on);
  all network task ports are resolvable.

### 4.8 eid to rank Index (`build_rank_index`, Peer Rank Resolution)

Peer rank resolution (Section 5.5 SQE fallback, Section 6.7 CCU primary path) shares one eid(hex) to rank_id
index, constructed with the following priority and basis:

1. **RankEidMap table (primary path)**: The `RankEidMap` table in runnerdb `hccl_sim.db` (in the same directory as hccl_vm_data.db).
   In estimate mode, the Communicator table has only one row for the local rank; the peer rank
   uses this table as the authority (`sim_models.h` comments; host precedent `GetRmtRankIdByEid`,
   `device_sqe_parse_stub.cc:889`). The row BLOB layout is consistent with `sim::RankEidMap`:
   `id(u64) + rankId(u32) + eid(16B) + pad(4B)`; the eid is in "hex string byte order",
   same source and order as `ccuChannels.leid/reid`; the Python side matches by direct `bytes.hex()` equality.
   This table is written by ranktable.json parsing (`topo_ascend_cluster_parser.cc:229`);
   in mock-comm scenarios, it is entirely overwritten along with communicator initialization, so it is authoritative.
2. **ranktable.json (fallback)**: When RankEidMap is empty/`hccl_sim.db` is missing/the table does not exist, expand
   `rank_list[].rank_id ↔ level_list[].rank_addr_list[].addr` (addr_type=EID)
   to build an isomorphic index. Both paths produce identical indexes in practice (8 ranks x 9 eids = 72 entries).

Open in read-only mode (`file:...?mode=ro` URI); degrade to fallback rather than reporting errors on read failures (including table non-existence).
Note that **`ccuChannels.srcRankId/dstRankId` cannot be directly used as communicator rank**: the writer side
(`hccp_ccu_stub.cc:401-405`) explicitly fills the global endpoint deviceId (channel establishment precedes
communicator creation; the same channel can be reused by multiple domains), which has an offset from the opDetails.rankId numbering system;
checker deprecates this field for this reason (`g_allRankChannelInfo` only stores dstDeviceId/dstDieId).

---

## 5. SQE Task Graph Generation (Sub-Requirement 2 - Lower Half, Focus of This Effort)

### 5.1 Overall Pipeline

```
OpExecutionGroup
   │
   ▼
⓪ SQE batch merge (sqe_group_merge.py, Section 5.1.1): Merge UDMA data tasks within the same (stream, sqeGroupId) batch into 1 entry
   │
   ▼
⓪' DPU sequence merge (sqe_group_merge.py, Section 5.1.2): Merge DPU large-message N-1 front blocks + tail block WRITE_WITH_NOTIFY into 1 entry
   │
   ▼
① Per-entry SQE → Task node (Section 5.2 mapping table; CCU_GRAPH forwarded to Chapter 6 for expansion)
   │     stream_id: int (SQE stream uses the original streamId directly; CCU skeleton tasks also belong to their SQE original stream, Section 6.5)
   ▼
② Stream serial chain: Group by (deviceId, streamId), ascending global_id → dep_graph.add_stream_chain
   ▼
 ③ Notify pairing: Cross-rank tasks do not participate in pairing (nodes retained); only local notify uses first-come-first-served → dep_graph.add_match (Section 5.4)
   ▼
④ Network task parameter completion: Direction inference / eid resolution (Section 5.5)
   ▼
⑤ CCU subgraph expansion (Chapter 6): CCU_GRAPH placeholder → microcode instruction interpretation → internal nodes + Event pairing
   ▼
⑥ Output assembly: (SimConfig, DependencyGraph) → HcclSimulator(config, dep, topo)
```

### 5.1.1 SQE Batch Merge (Pre-processing Before Graph Generation)

On the collection side, one UDMA doorbell (SQE) can carry multiple WQEs; tasks are still stored per WQE, and marked with
`sqeGroupId` for batch attribution (circularly incrementing doorbell batch number 1..255, wrapping at 255, never taking 0; 0 = construction
default "unmarked", Section 4.4.1). Before graph generation, split into batches by **(streamId, sqeGroupId)** -- the split key explicitly
includes the stream; merge all data tasks (MEM_CPY/REDUCE/WRITE_WITH_NOTIFY) within a batch into a single entry, aligned with
the hcomm batch transport profiling `ExecProfilingAll` caliber -- the three entry points (`pipeline.py`/
`__main__.py`/`dump_graph.py`) call it uniformly, ensuring consistent simulation and dump calibers:

- **Isolation guarantee by stream grouping**: The split key contains streamId, so **cross-stream tasks are never merged into the same batch** (one doorbell
  inherently belongs to only one stream; a single doorbell batch is not split by the key); the group number (doorbell batch number) is unique within a 255-element window
  -- doorbells from other operators are not visible in the current operator task list but also consume batch numbers, so **same stream and same number must belong to
  the same doorbell**, preventing cross-doorbell mis-merging (safe for multi-operator databases).
- **Descriptive information** (type/address/endpoint/notifyId/reduction parameters/profInfo) takes the **last** data
  task in the batch; **length takes the full batch accumulation** -- consistent field by field with `ExecProfilingAll(loc[insNum-1], rmt[insNum-1],
  totalSize, transferOp[insNum-1])`; the trace/dump display name similarly takes the op name of the last task
  (Section 11.4.1). When a batch mixes write/read directions or different dataTypes, it also follows the last task
  (a single SQE batch is typically homogeneous; mixing is a rare scenario).
- **Do not participate in merging**: notify/SDMA/synchronization tasks (notifyId is unique, pairing depends on order; local tasks
  are retained as-is and do not affect batch boundaries); unmarked/no-group-number tasks (`sqeGroupId <= 0`: 0 = unmarked,
  -1 = v3 and earlier BLOBs lack this field) are retained as-is and **cut the current batch** (batch attribution cannot be determined; conservatively
  end the current batch); UDMA non-data tasks (inline write notify) only participate in batch splitting; those interleaved within a batch are placed after the merged task
  (notify causality).
- The merged task occupies the slot of the first data task in the batch (batch head); the split key containing the stream guarantees that tasks within a batch share the
  same stream and same jetty; the relative order of the stream serial chain (Section 5.3) and notify pairing (Section 5.4) is not
  affected; local (non-UDMA) tasks do not participate in merging and do not reset the batch boundary (retained in place, not affecting batch continuity).

### 5.1.2 DPU Sequence Merge (Pre-processing Before Graph Generation)

DPU (host CPU RoCE) path large-message WriteWithNotify sequence merging, executed after 5.1.1
(`pipeline.py` unified call). Rules and semantics are described in Section 11.4.6: Scan the DPU stream order for
**src/dst dual-address contiguous** MEM_CPY chains (only data tasks with `src_type==1` of DPU origin participate;
other tasks are retained as-is without cutting the chain); the chain is terminated and merged into a single entry by an address-contiguous WRITE_WITH_NOTIFY --
**the terminating task is a natural terminator**; two independent WriteWithNotify calls cannot be cross-merged (the tail block of each call
inevitably closes the current chain); the only remaining mis-merge scenario is "an independent pure write whose address happens to adjoin the WriteWithNotify start address" -- one
theoretical scenario (semantically equivalent to a contiguous data segment + terminating notification; simulation results are identical). Chains without a terminator are not merged
(independent pure writes remain as per-entry `RDMASend`); the merged task occupies the terminator task slot (notify causality); the merged
front blocks are removed from their original positions; unmerged tasks retain their original stream order.

**Single-rank graph generation strategy (design premise, primary difference from checker)**: The db contains only single-rank task data;
the DependencyGraph describes only the tasks and dependencies of that rank. **All cross-rank pairing edges (post-wait relationships)
are not created (task nodes are retained, approximated with fixed remote latency)**:

- SQE level: RECORD where waitDev != local device (cross-rank post), WAIT where recordDev != local device (cross-rank wait)
  **retain the task node** -- graph generated in normal stream order (src != dst expresses remote, uses fixed remote latency), but **no
  pairing edges are created** -- only post-wait pairing within the single rank is considered (Section 5.4).
- CCU level: Instructions that post to a remote Event, and waits that depend on remote Event assertion **also generate task nodes**
  (Notify_Record/Notify_Wait, src=local, dst=INVALID and then fixed remote latency), but no cross-rank
  pairing edges are created (Sections 6.4/6.6).
- **Cross-rank data transport is retained** (MEMCPY with direction=write/read): They are the primary source of network
  bandwidth consumption from the perspective of this rank; only synchronization dependency edges are ignored, not the transfers themselves.
- makespan semantics: From the single-rank perspective, the operator end-to-end duration = local computation/copy + network transfers initiated by this rank
  + local synchronization waits; when the remote side becomes ready is not modeled (equivalent to assuming the remote side is immediately ready).

**Simulation granularity**: Default is per-operator (one DependencyGraph, one simulation, one record per `(commId, opIter)`),
consistent with the per-operator validation caliber of checker; `--all-in-one` optional full mode (multiple operators on the same stream are naturally serial).

### 5.2 taskType to hccl_sim Task Mapping

Aligned with the HCCLSIM new task model (`TaskType`: `NOTIFY_RECORD`/`NOTIFY_WAIT`/`REDUCE_INLINE`/
`MEMCPY`/`UB_INLINE_WRITE`/`UB_WRITE_OR_READ`/`WRITE_WITH_NOTIFY`, see Section 7.3 for details):

| db taskType | Condition | hccl_sim TaskType | Task Parameters | Latency Model |
|-------------|---------|-------------------|-----------|---------|
| NOTIFY_RECORD (1) | waitRank == local rank | `NOTIFY_RECORD` | notify_id=int, src=dst=local_rank | Fixed + trigger Event |
| NOTIFY_RECORD (1) | waitRank != local rank | `NOTIFY_RECORD` (cross-rank post: node retained, not paired) | notify_id=int, src=local_rank, dst=peer rank (INVALID if unresolved) | Fixed remote_record_delay, does not trigger Event |
| NOTIFY_WAIT (0) | recordRank == local rank | `NOTIFY_WAIT` | Same as above | Wait for Event |
| NOTIFY_WAIT (0) | recordRank != local rank | `NOTIFY_WAIT` (cross-rank wait: node retained, not paired) | notify_id=int, src=peer rank (INVALID if unresolved), dst=local_rank | Fixed remote_wait_delay |
| MEM_CPY (3) | jettyId == INVALID | `MEMCPY` | size=len, src/dst_rank=taskData endpoint rank (defaults to local if unfilled) | On-chip memory dynamic shared bandwidth |
| MEM_CPY (3) | jettyId valid, write direction | `UB_WRITE_OR_READ` | size=len, src_rank=local, dst_rank=peer rank, src/dst_eid, jetty_id, protocol, is_write=True | Port write direction bandwidth sharing |
| MEM_CPY (3) | jettyId valid, read direction | `UB_WRITE_OR_READ` | size=len, src_rank=peer rank, dst_rank=local, src/dst_eid, jetty_id, protocol, is_write=False | Port read direction bandwidth sharing |
| REDUCE (2) | jettyId == INVALID | `REDUCE_INLINE` | size=dataCount, data_type=task-level dataType enum name, src/dst_rank same as MEMCPY local branch | On-chip memory dynamic shared bandwidth |
| REDUCE (2) | jettyId valid | Same as MEM_CPY network branch (Section 5.7 approximation) | Same as above (including data_type) | Same as above |
| WRITE_WITH_NOTIFY (7) | Always write direction (local src to remote dst, no direction inference) | `WRITE_WITH_NOTIFY` (**single task, not split**) | size=len, notify_id=notifyId, src_rank=local, dst_rank=peer rank (falls back via eid to rank index if unresolved), src/dst_eid, jetty_id, protocol, is_write=True; when isReduce=1, data_type=dataType enum name | Port write direction bandwidth sharing + fixed record delay for notify after write landing (remote) |
| CCU_GRAPH (4) | -- | (Expanded, Chapter 6) | -- | -- |
| AIV_GRAPH (5) | -- | Skip | -- | Warning |
| SYNC_STREAM (6) | -- | **Skip** (host-side stream synchronization point, Section 5.6) | -- | -- |

> waitRank/recordRank are the taskData endpoint rankIds (carried by the deviceId field, src=record side,
> dst=wait side, Sections 4.4.1/4.6), used only for **determining local/cross-rank**: cross-rank notify tasks
> retain nodes during the mapping phase but do not enter pairing (Section 5.1 strategy);
> pairing occurs only among local notifies. In the new model, local/remote for notify is expressed as `src_rank == dst_rank`
> (`is_local_notify`).
> **WRITE_WITH_NOTIFY (7) single-task mapping** (one-to-one correspondence with db tasks, not split into write + NOTIFY_RECORD
> two sub-tasks): The write direction is always write (launcher convention: src local to dst peer; no Section 5.5 direction inference);
> uses port write direction bandwidth sharing; after write data lands, a notify is sent to the peer, charged at a fixed record delay
> (the peer wait is not in this db, per the single-rank simulation premise, same semantics as cross-rank record; self-write src==dst edge
> case uses local record delay); `notifyId` is retained with the task (displayed in trace/dump);
> when `isReduce=1`, only the `data_type` display is affected (same enum name mapping as the REDUCE branch).
> **Network tasks directly consume taskData for rank** (the deviceId field carries rankId; launcher convention:
> write=src local/dst peer, read=src peer/dst local): write fills `dst_rank=peer rank`,
> read fills `src_rank=peer rank`; when the peer rank is unresolved (`0xFFFFFFFF` or equal to local), it falls back to INVALID
> (Section 10.1). The direction is explicitly carried as `is_write`; the simulator no longer depends on the `src_rank==0` heuristic
> (Section 10.3 #1 already fixed; the task.txt old input path still uses the heuristic).
> The local eid prefers the SQE explicit lcEid (new in v2 collection); defaults to jettyId to ccuChannels channel
> leid derivation (Section 5.5, Section 10.1).
> The TaskType in this table is the internal latency model classification; the task display name in trace/dump has already switched to
> the profInfo.profTaskType op name (Section 11.4.1 mapping table, Section 10.3 #18 already implemented);
> the correspondence with the TaskType in this table follows the "Simulation TaskType" column in Section 11.4.1.

### 5.3 Stream Serial Chain Construction

- Group by `(deviceId, streamId)`; within each group, ascending by `global_id` (= SQE delivery order) to form a chain.
- The simulator launches one simpy process per chain for serial execution (Chapter 7) -- **the mainStart virtual
  start point and main stream special handling from checker are not needed**: simulation focuses on end-to-end makespan; each chain starts in parallel from time 0;
  the WAIT of the first task in each stream naturally forms the actual synchronization with the main stream.

### 5.4 Notify Pairing Algorithm (Local Only)

**All cross-rank post-wait edges are ignored** (Section 5.1 strategy): Before pairing, first classify by the Section 4.6 peer backfill results;
cross-rank notify tasks (RECORD where waitDev != local device, WAIT where recordDev != local device) retain nodes during the Section 5.2
mapping phase (graph generated in stream order) but do not enter pairing. Therefore, the pairing key simplifies from the triplet in checker to a single notifyId:

```
checker semantics:   pairing key = (recordDeviceId, waitDeviceId, notifyId)   ← multi-rank
This plugin:         filter cross-rank first → pairing key = notifyId                    ← single-rank
```

Consistent with `AddLocalNotifyEdges` in checker (delivery order simulation + first-come-first-served consumption,
`task_graph_generator_v3.cc:847-946`) in principle, light_performance simplifies to
**two-pass scan + first-come-first-served**:

```
Input: All local notify tasks for an operator (cross-rank nodes retained but not participating in pairing)
First pass: Scan in ascending global_id order
    RECORD → pending_records[notifyId].append(task_id)                        # Queue
    WAIT   → if pending_records[notifyId] is not empty:
                  rec = pop from front; dep_graph.add_match(wait_id, rec_id)          # First-come-first-served
             else:
                  unmatched_waits.append(wait_id)                               # Wait for a "later" record
Second pass: Handle unmatched_waits (cases where WAIT appears before RECORD)
    Replay in global_id order: WAIT is pending when unmatched; after RECORD arrives, pair with the earliest pending WAIT
    (Successfully paired RECORDs are synchronously removed from pending_records to ensure diagnostic statistics accuracy)
Cleanup: WAITs still remaining → normal (see below); RECORDs still remaining → warning, should not appear in normal data
```

Notes:

- The db id order is the delivery order, and the stream serial chain already guarantees "successor tasks on the same stream after a WAIT do not execute first",
  so the full queue simulation and deadlock detection from checker are not needed (deadlocks manifest in simulation as makespan anomalies/unexecuted tasks).
- Multiple RECORDs/WAITs for the same notifyId (for example, n11/n15 appearing repeatedly on stream 6 in early 169B samples, two rounds of main stream iterations)
  are naturally paired by the first-come-first-served rule; **notifyId reuse is allowed**.
- **Unpaired local WAIT is normal semantics**: In the v2 AICPU sample (Section 4.7), notifyIds are reused across ranks --
  local `RECORD(nid=30, dst=dev2)` (wait side is on the peer, node retained but not paired) and local `WAIT(nid=30)`
  (record side is on the peer, not in this db) appear in pairs; this WAIT is charged at a fixed `remote_wait_delay` delay during simulation
  (Section 7.6); the warning is informational only.
- Cross-rank pairs in early 169B samples: `RECORD(n18, s1→d2)` (notifies dev2) and `WAIT(n18)` (backfilled recordDev=2)
  both retain nodes; no cross-rank pairing edges are created: graph generated in stream order, `src≠dst` uses fixed remote latency;
  data transport tasks on stream 8 (MEMCPY(jet=3), and so on) are not affected.

### 5.5 Network Task Direction and eid Resolution

**Direction determination** -- network data tasks are uniformly mapped to `UB_WRITE_OR_READ`; the direction determines the port charging
direction (is_write) and rank endpoint values. The taskData endpoint rankId is carried by the deviceId field
(Section 4.4.1 rank field rollback); launcher convention (`device_sqe_parse_stub.cc`):
write = src local/dst peer, read = src peer/dst local:

| Priority | Rule | Basis |
|--------|------|------|
| 1 | srcRank == local and dstRank != local → write; dstRank == local and srcRank != local → read | Collection endpoint rankId (carried by deviceId field, Section 4.4.1); all measured samples (Section 4.7) are determined entirely by this rule, no inference needed |
| 2 | When rank is indistinguishable (equal or both unfilled), use memory range: dst falls within local output range → read; src falls within local input range → write | Derived from measured samples (Section 4.7): fails when task offsets do not intersect with opMemInfo ranges |
| 3 | Fallback: default write; diagnostic statistics count `direction_inferred` | -- |

> Network tasks explicitly carry `is_write` on the Task (Section 10.3 #1 already fixed; CCU path Section 6.7 synchronously fixed);
> the simulator selects the write/read charging direction and port query eid (`src_eid`/`dst_eid`) based on this, no longer depending on
> the `src_rank==0` heuristic; the task.txt old input path lacks this field and falls back to heuristic determination.
> `WRITE_WITH_NOTIFY` (7) **does not participate in direction determination**: Semantically always write (local src to remote dst);
> directly set `is_write=True` during graph generation.

**eid resolution** -- network tasks look up the ranktable by eid to get physical ports (`TopoResolver.get_ports_by_eid`):

```
SQE:        Local eid prefers SQE explicit lcEid (new in v2); defaults to jettyId
            ──ccuChannels(local device outbound channel)──> leid(local eid)
            Peer prefers channel reid; defaults back to SQE rmEid; when channel is missing, local eid degrades to any local channel leid
CCU microcode: channelId → channels[(dev,die,channelId)] → leid/reid + jettyId list
            When channel is missing, local eid prefers the mission SQE lcEid; degrades to any channel leid on the same die
Write direction: Simulator queries task.src_eid → ports; read direction queries task.dst_eid → ports
```

- `protocol`: Channel protocol column to HCCS/TCP/ROCE/UB_CTP/UB_TP/PCIE/SIO (CommProtocol name,
  for display only); `link_type`/`transport_type`: task profInfo (DfxLinkType/
  ProfTransportType name, Sections 11.4.2/11.4.3, uniformly populated by `_add_task`, reconciled with msprof L1).
- When the local eid cannot be resolved, issue a warning + increment `eid_unresolved` count; when network tasks exist but no ranktable is available,
  skip simulation entirely (TopoResolver throws directly for missing eids, see Section 10.3).

**Peer rank resolution** -- the peer rank (`_peer_rank`) for SQE network tasks: The taskData endpoint rankId
(carried by the deviceId field, already resolved by the launcher via `GetRmtRankIdByEid`) is preferred; when unresolved
(UINT32_MAX/local value), fall back via the eid to rank index (Section 4.8) using the peer eid; if still unavailable,
INVALID + increment `rank_unresolved` count.

### 5.6 SYNC_STREAM Handling

Host-side stream synchronization point (appears once per iteration on the sample main stream), indicating that the host waits for this stream to complete;
**does not represent a device-side task**: No task node is generated and no time is consumed (makespan is the end time of the last task in the stream chain).

- Diagnostic statistics increment `skipped_sync` count;
- Previously approximated by "unpaired local `NOTIFY_RECORD` fixed delay placeholder"; removed because it introduced spurious
  `local_record_delay` at the stream tail (and appeared in the trace as a pseudo-notify task with notify_id=invalid).

### 5.7 REDUCE Approximation

Reduction = network transfer + computation. The current model approximates it as data transport:

- Network (jettyId valid): Charge port bandwidth latency as `UB_WRITE_OR_READ`
  (direction determined per Section 5.5).
- Non-network (jettyId INVALID): Map to `REDUCE_INLINE`; charge on-chip memory bandwidth with dynamic sharing
  (`size / hbm_bandwidth`); computation latency is not separately listed at this time (can be calibrated by `dataType × dataCount` later,
  see Section 10.1 for the data_type field gap).

### 5.8 Comparison and Simplification Relative to Checker Graph Generation

| checker Step | light_performance | Reason |
|--------------|-------------------|------|
| mainStart virtual node + main stream incoming edge special handling | Removed | Simulation only needs makespan; the chain-head WAIT already expresses synchronization |
| AddLocalNotifyEdges queue simulation + deadlock detection (GRAPH_DEADLOCK) | Two-pass scan, first-come-first-served; no deadlock detection | No validation requirement; anomalies manifest as simulation results |
| AddInterRankNotifyEdges cross-rank pairing (full-graph BFS simulation) | **Completely removed**: All cross-rank post-wait pairing edges are ignored (notify task nodes retained, graph generated in stream order) | Single-rank graph generation (core difference) |
| MemSlice address resolution + memory layout table | Only range attribution for direction determination (optional) | Simulation only needs len; a precise memory model is not required |
| CheckSlaveTaskQueue slave stream structure validation | Removed | Not a validation tool |
| Memory conflict/semantic validation/synchronization conflict/reachability/single-task checks | All removed | Not a validation tool |
| CompactSyncNodes (SYNC_STREAM merging) | Skip, do not generate nodes (host-side synchronization point, not a device-side task) | Makespan is the end time of the last task in the stream chain (Section 5.6) |
| AIV subgraph expansion | Not supported (skip with warning) | Requirement explicitly states not to consider |
| CCU subgraph: queue parallelization transformation, loop merge, resource conflict validation | Queue parallelization simplified implementation (Section 7.4.1: assembly-phase skeleton/unit splitting + fork/completion set joining and attachment extra edges + `#v` virtual streams); loop merge equivalent implementation (Section 6.5); resource conflict validation removed | Parallelization eliminates queue serialization distortion (Section 7.4.1); no validation needed, node count is manageable |

### 5.9 Output Assembly

```python
def build(group: OpExecutionGroup, sim_config: SimConfig | None = None
          ) -> tuple[SimConfig, DependencyGraph]:
    dep = DependencyGraph()
    # ① SQE → Task (CCU_GRAPH stored as placeholder)
    # ② Stream chain (int stream id)
    # ③ Notify pairing (add_match)
    # ④ Direction/eid completion
    # ⑤ CCU expansion (ccu_graph_builder.expand_ccu_graph(..., local_rank, alloc_stream))
    return sim_config, dep
```

- `SimConfig` source: JSON specified by `--config` / manifest setting (Section 7.2 format, units
  us/GBps), overlaid with bandwidth utilization configuration.
- Topology: `TopoResolver` is constructed by the CLI layer (ranktable location per Section 3.3) and passed to the simulator.
- The output can be immediately submitted to `HcclSimulator(config, dep_graph, topo).run()` (returns a dict, see Section 7.1).

---

## 6. CCU Mode Task Graph Generation (Sub-Requirement 3, Focus of This Effort)

### 6.1 CCU Data Organization in the db

```
SQEs with taskType=4 (CCU_GRAPH) in opTask_P_<pid>
     └─ taskData.ccu: {dieId, missionId, timeout, instStartId, instCnt, key, argSize, args[13]}
            │  instStartId/instCnt define the instruction execution window [instStartId, instStartId+instCnt)
            │  args[13] are consumed one by one by LOAD-type instructions (cursor advances to next SQE after reading all 13 args)
            ▼
ccuInstrRes (deviceId, dieId) → instrSpace: 32K × 32B microcode instruction space
ccuChannels (deviceId, dieId, channelId) → {dstDeviceId, dstDieId, leid/reid, jettyId[]}
     └─ Channel register values → peer location and jetty
```

- SQEs are grouped by `(dieId, missionId, queue)` (single-device graph generation, deviceId is constant; queue is the SQE stream)
  into dispatch contexts (`CcuMissionKey` in checker + latest-node-append semantics,
  `task_meta_translator_v3.h:47-62`): SQEs with the same key merge into the same context only when **contiguous with the previous one** (previous
  `instCnt==13` and `instStartId` connects end to start, `IsContiguousCcuSqe`,
  `task_meta_translator_v3.cc:213-231`); otherwise **create a new context** --
  multiple dispatches of the same mission (instruction window wraparound re-delivery, typically "head SQE (13 Load
  SqeArgs) + body SQE" delivered twice in succession) are each independently expanded: each context fully replays its own instruction window
  (including the Load SqeArgs header, using **this dispatch's args** to reinitialize loop counters and other registers),
  preventing the later dispatch from inheriting stale register state from the previous one, which would cause data transport loss (Section 10.3 #20).
  - **args consumption** (`CcuGraphStateV3::GetSqe` semantics in checker): SQEs within a context are arranged in dispatch order;
    args are consumed sequentially per slot -- after reading all 13 args (reaching the last slot `sqeArgsId==12`), the cursor advances
    to the next SQE; in the typical layout, the head SQE's 13 Load SqeArgs consume the head SQE's args[0..12],
    and subsequent Load instructions within the body SQE window consume the body SQE's args.
  - **Register/Event bitmaps are globally shared across contexts** (`AllRankParamRecorder::Global()`
    semantics in checker): Waits in a later dispatch can be resolved by Event bits/residual post records left by an earlier dispatch,
    consistent with the multi-graph shared recorder behavior in checker.
  - **Dispatch contexts within the same queue advance strictly serially in dispatch order** (checker queue semantics: a single
    queue is strictly ordered by instruction; the next TaskCcuGraph on the same stream must wait for the previous one to complete); cross-queue polling
    advances jointly (Section 6.6).
- Subgraph attachment point: The position of the CCU_GRAPH SQE in the stream chain is the subgraph entry; after expansion, the placeholder is removed
  (no placeholder nodes are introduced in the task graph); the subgraph attaches to SQE predecessor/successor tasks via extra edges (Section 7.4.1).

### 6.2 Microcode Instruction Format (32B Fixed-Length)

Same source as `${HCOMM_CODE_ROOT}/src/base_comm/resources/ccu/ccu_microcode`
(reference copy in this repository: `ai_skills/skills/convert-ccu-microcode/references/ccu_microcode_v1.h`):

```
CcuInstr = CcuInstrHeader (2B) + union { V1(A5) | V2(A6) | V3 } (30B)

CcuInstrHeader bit fields (little-endian u16):
    code     : 11 bit   (header & 0x7FF)      Instruction code
    type     :  4 bit   ((header >> 11) & 0xF) Instruction category
    reserved :  1 bit

Instruction categories:  LOAD_TYPE=0x0 | CTRL_TYPE=0x1 | TRANS_TYPE=0x2 | REDUCE_TYPE=0x3
```

Chip generation to instruction set version mapping (`GetCcuInstrVersion` in checker, `ccu_task_transform_v3.cc:175-184`):
`Ascend950 → A5/V1`, `Ascend960 → A6/V2`, other devType values fall back to A5; derived from the Device soc_version
(hccl_sim.db) or `opDetails.devType` (`expand_ccu_graph` selects the `_MissionContext`/`_MissionContextA6` interpreter based on `dev_type`). **Both instruction set generations are implemented** (A6 ported from
`ccu_task_transform_instruct_a6_v3.cc` in checker, Section 11.3.3); an overview of A5/A6 instruction set differences is in Section 6.8.

### 6.3 Instruction Type to Task Mapping

#### 6.3.1 Summary Table

| Instruction Category | Representative Instructions | light_performance Behavior | Produced Task |
|----------|---------|----------------------|-----------|
| LOAD (register write) | LoadSqeArgsToXn/GSA, LoadImd, LoadX, arithmetic-logic (A6 Add/Sub/...) | Update the register model; **no nodes produced** on the data plane; A6 LOAD instructions commonly carry a `setCKE` tail field (mask non-zero → local `NOTIFY_RECORD`, `ProcessSetMask` semantics in checker); A6 Load/Store (on-chip memory to Variable data plane) does not model on-chip memory content, registers are not backfilled (only the setCKE synchronization semantics are retained, warning logged) | -- (Notify_Record when A6 setCKE is non-zero) |
| CTRL: SetCKE/SetCKBit | Set Event bits | Local → `NOTIFY_RECORD` (register Event record, with mask); remote target (cross-rank post, SyncCKE, and so on) → remote `NOTIFY_RECORD` (not paired, Section 6.6) | Notify_Record |
| CTRL: Wait/Jmp/Fence | Wait for Event bits / jump / fence | Event wait (A5 embedded waitCKE field): If bits are insufficient, pause the queue pointer; if bits are satisfied, generate `NOTIFY_WAIT` (split by contributing bits, with mask); waits depending on remote bits → remote `NOTIFY_WAIT` (not paired, fixed delay, Section 6.6). A6 independent Wait instruction (Variable value comparison spin, `TransformWaitInstr` in checker): Satisfied means pass through with no node; unsatisfied = depends on peer remote register write (SyncWtX/TransMem notify) → top-level stall resolution generates remote `NOTIFY_WAIT` (fixed `remote_wait_delay`); within loop body, directly generate node and continue. Jmp: A5 evaluates by `Xn[conditionXnId] != expectData` (`TransformJmpInstrA5` semantics in checker); A6 supports 6 condition codes (==/!=/>/>=/</<=, >=6 unconditional) + relative/absolute two modes (jumpMode, target `Xn[relTarInstrXnId]`); target must be within the current instruction window; out-of-range warns and passes through; within loop body passes through; step budget defense against backward jump infinite loops. Fence (A6 only): checker does not support it; warns and passes through (sequential order is inherently guaranteed under the queue serial model) | Notify_Wait |
| CTRL: ClearCKE | wait + clear bits | wait semantics; A6 clears the consumed wait bits per `ClearWaitMask` semantics in checker (`clearMask != 0` is a checker-unsupported scenario; warning + tolerant clearing of target bits) | Notify_Wait |
| CTRL: Loop/LoopGroup | Loop | loopCnt comes from a register; **merge expansion** (Section 6.5, falls back to full expansion on failure); A6 Loop `mode==1` is conditional expansion (`wishCKEBit == Xp.loopCtxId + 1024` match required to execute; always takes the full expansion path) | None (body instructions mapped per this table; under the merge path, data task size × total iterations) |
| TRANS (transport) | LocMem to LocMS/RmtMem/RmtMS (A5), local 4 codes + unified TransMem dmaOpCode (A6) | Map by role (Section 6.3.2) | `Memcpy`/`Ub_Write_Or_Read` |
| TRANS: Sync (cross-rank synchronization) | SyncCKE/GSA/Xn (A5), SyncXnWtx/SyncAtx (A6) | Remote assertion/notification → remote `NOTIFY_RECORD` (not paired, Section 6.6); value synchronization is a remote register write, no node created | Notify_Record (remote) |
| REDUCE | Add/Max/Min | Same as SQE REDUCE approximation (Section 5.7); CcuBuffer is a rank-local buffer, always local on-chip memory reduction | `Reduce_Inline` |

#### 6.3.2 TRANS Transport Role Determination (Consistent with `ccu_loop_merge_v3.cc:73-144` in checker)

By src/dst device ownership:

| src | dst | Role | hccl_sim Task |
|-----|-----|------|---------------|
| Local | Local | LOCAL_COPY | `Memcpy` (on-chip memory dynamic sharing) |
| Local | Remote | WRITE | `Ub_Write_Or_Read` (src_rank=local, dst_rank=peer, is_write=True) |
| Remote | Local | READ | `Ub_Write_Or_Read` (src_rank=peer, dst_rank=local, is_write=False) |

> Cross-die/card read/write bidirectional operations are uniformly mapped to `Ub_Write_Or_Read`; the direction is expressed via `is_write`/trace
> `direction` field; ports are charged independently by write/read direction; the peer rank is resolved via channel
> reid to eid to rank index (INVALID when unresolved). To preserve buffer information distinguishable in the microcode,
> CCU-converted transport tasks carry `src_buf_type`/`dst_buf_type` (mem/ms, determined by the instruction
> source/destination operand); SQE/CLI path tasks with the same name do not carry this information and output the invalid value N/A.

> Cross-rank **data transport** (WRITE/READ roles) is retained -- it is the primary source of network bandwidth consumption;
> only cross-rank **synchronization edges** (post-wait, see Sections 6.4/6.6) are ignored.

Key field sources:

- `size`: Length register (A5 `lengthXnId` direct read / A6 `xlId` via `GetXnId` -- when bit15=1,
  the register number is offset by the loop `xnIdOffset`) → look up the register model.
- Peer location: A6 channel via **register value** (the value of `Xn[GetXnId(xcId)]` is the channelId) →
  `channels[(dev,die,channelId)]` → `{dstDeviceId, dstDieId}`; A5 channel is the instruction
  immediate field. A5 CcuBuffer-type instructions: `msId / 0x8000` = rank, `msId % 0x8000` = CcuBuffer offset (CcuBuffer page is 4KB) -- **only used as
  addressing encoding**: CcuBuffer is a rank-directly-accessible local buffer (`GenSliceFromMs` +
  `AddLocalBatchReduce` in checker treats all msId values as always local); the high bits of msId are **not used as** a local/remote task type
  discriminator; cross-rank data movement is expressed by Rmt* role instructions (through channels). A6 TransMem with
  `msIdMode==1` marks the CcuBuffer end (write source / read destination is local CcuBuffer,
  `src/dst_buf_type=ms`).
- eid: Channel leid/reid (hex string); the local eid prefers the mission SQE `lcEid`
  (new in v2); defaults to the same-die channel leid as an approximation.
- A6 `TransMem` `dmaOpCode` (implemented, `TransformTransMemInstr` in checker):
  3=write, 5=write with notify (the notify value is written to a remote Variable/Event; the target type is unresolvable in single rank so it is
  approximated as remote record), 6=read, 0x1A=remote atomic count (no data plane, approximated as remote record),
  0x19 (compressed write + atomic) and others unsupported (warning, skip data plane; setCKE local
  post still executes to maintain the synchronization skeleton); `udfEnable` (network reduction) charged as ordinary UB transport
  (Section 5.7 approximation; peer reduction is not modeled); `udfType != 0` warns and skips the data plane.

### 6.4 Execution Model: Register + Event Record (register_model.py)

Python simplification of `AllRankParamRecorder` in checker (`ccu_all_rank_param_recorder_v3.h:27-110`):

```python
class CcuRegisterModel:
    xn:    dict[(dev, die, id)] -> int      # General-purpose registers (length/loop counter, and so on)
    cke:   dict[(dev, die, ckeId)] -> int   # Event bitmap (set/clear)
    gsa:   dict[(dev, die, id)] -> int      # A5 Address registers (A6 has no Address registers, only Variable/Event)
    seen_post: dict[key, list[PostRecord]]  # Event post record (for pairing)

    key = (waitDeviceId, dieId, ckeId)
    PostRecord = {record_device_id, cke_mask, remaining_mask, task_id, is_local}
```

- LOAD/arithmetic-logic instructions "execute" on this model (read/write registers); uninitialized reads → warning and treat as 0
  (checker reports error and exits; this approach handles it leniently to accommodate incomplete data). A6 arithmetic-logic instructions
  (Add/Sub/Mul/And/Or/Not/Xor/Shl/Shr/Popcnt) execute with 64-bit wraparound semantics (Mul takes the
  lower 32-bit product, shift amounts are truncated to 0x3F, arithmetic shifts interpret as signed); LoadX/StoreX
  indirect addressing target register number out of range (>16383) warns and skips the write (checker reports error and aborts).
- **A6 in-loop register number offset** (`GetXnId`/`UpdateXnId` in checker): When the register number field
  bit15=1 in the payload, within the expansion region (`curLoopIdx >= expandOffset`), offset the register number by
  `xnIdOffset × curExpandCnt`; when bit15=0, use the original number directly. The circular offset for Event/CcuBuffer IDs follows the same rule across both generations (`UpdateId`).
- **Event pairing rules** (consistent with `GenWaitNode` in checker, `ccu_task_transform_v3.cc:56-121`):
  ```
  post matches wait ⇔ post.waitDeviceId == wait.waitDeviceId
                    && post.dieId == wait.dieId
                    && post.ckeId == wait.ckeId
                    && (post.ckeMask & wait.ckeMask) != 0        # Bits have intersection
  After matching: One wait instruction aggregates into at most 1 local wait node (mask = OR of all contributing
           post bits) + 1 remote wait node (bits depending on remote assertion)
           post.remaining_mask &= ~wait.ckeMask                  # One post's bits can be consumed by multiple waits in stages
           Depleted → remove from seen_post
  Node dependencies: The aggregated wait node builds edges with all contributing posts -- the first contributing post uses
           notify pairing (add_match, Event semantics); the rest use extra edges
           (child=wait, parent=post, "wait for all parents to complete" pure timing constraint)
  ```
- **Event record only registers local-rank posts** (single-rank graph generation): Cross-rank posts generate remote
  `Notify_Record` nodes (src=local, dst=INVALID → fixed remote latency) but do not register
  `seen_post` (the assertion action occurs on the peer rank's Event); wait bits depending on remote assertion generate
  remote `Notify_Wait` nodes (fixed delay, not paired), see Section 6.6.
- When bits are not satisfied, the queue pointer pauses (waiting for posts from other queues of this rank to unlock) -- multi-queue joint advancement
  is described in Section 6.6; when multiple dispatch contexts exist for the same queue (same mission dispatched multiple times, Section 6.1),
  the later context must wait for the earlier context to complete before starting advancement.
- Final validation: `seen_post` residuals → warning and counted in diagnostic statistics (checker reports GRAPH_UNMATCHED; here degraded).

### 6.5 Queue Model and Loop Handling

- **queue**: `CcuSqeParam` groups by the SQE's stream (checker uses ccuParams[queueId]);
  each queue is one serial chain. The new task model `stream_id` is **int**:
  SQE streams use the original streamId directly; **CCU queue chains also belong to their SQE's original stream** -- whichever stream a CCU-type SQE
  is on, the expanded skeleton tasks are on that same stream (`queue` is the original stream number;
  `alloc_stream("ccu_d{die}_q{queue}")` resolves back to the original stream number via `ccu_queue_origin_stream`;
  same-name reuse merges multiple missions of the same queue); only virtual streams newly created by queue parallelization are allocated sequentially from base
  `1<<20`; logical names are retained in `stream_labels` for dump/report display.
  Exception: When the same original stream is occupied by a queue from **another die**, the latter falls back to base-allocated synthetic streams with a warning
  (chaining would serialize cross-die parallel execution); in this case, the stream kind in `stream_meta` is ccu.
  Instructions within the same queue are strictly ordered (checker semantics: `ccu_task_transform_v3.cc:189`,
  "a single queue advances strictly in instruction order; cross-queue coordination is expressed only through generated wait/post edges").
- **Assembly-phase queue parallelization splitting** (Section 7.4.1): After interpretation completes, each CCU queue chain is split into
  skeleton (wait / free set / remote synchronization nodes -- free set means record nodes that do not immediately follow a data task)
  and asynchronous data transport units:
  The skeleton stays on the queue serial chain (= SQE original stream chain, reassembled in dispatch order back to the placeholder position) to express the timing framework;
  asynchronous units are reattached to virtual streams
  `ccu_d{die}_q{queue}#v{n}` (allocated sequentially from base `1<<20`), reused across missions,
  not reused within a mission, connected to the skeleton via fork extra edges -- the serial chain only constrains the skeleton and unit-internal order,
  eliminating the "entire queue per-task serial" distortion.
- **loop expansion** (merge-first, aligned with `ccu_loop_merge_v3.cc` in checker):
  - The iteration count of `Loop`/`LoopGroup` instructions comes from registers and is known at interpretation time:
    - A5: LoopGroup parameters are in Variable (LoopNum[61:55]/RepeatNum[54:48]/NoRepeatNum[47:41])
      and Xm (gsaOffset/CcuBufferOffset/ckeOffset); Loop iteration count is in the lower 13 bits of Variable[xmId];
    - A6: LoopGroup parameters are carried by **Variable/Xm/Xp three registers** (checker
      `LoopGroupXnA6/XmA6/XpA6`) -- Variable: `loopInsCnt:10 | expandOffset:9 |
      expandCnt:9`, Xm: `ckOffset:10 | msOffset:11 | gsaOffset:32`,
      Xp: `xnIdOffset:32` (GetXnId offset, Section 6.4); Loop iteration count is in
      the lower 13 bits of `Variable[GetXnId(xmId)]`, Address stride is in the lower 32 bits of `Variable[xnId]`, context id
      is in the lower 9 bits of `Variable[xpId]`; `mode==1` is **conditional Loop** -- only expanded when the instruction
      `wishCKEBit == loopCtxId + 1024` (`A6_LOOP_BOARD_ENTRY`) is matched; otherwise
      this expansion is skipped (`ProcessLoopIns` in checker); conditional Loops always take the full expansion path
      (`CollectLoopExpandA6` in checker rejects merging).
    Each Loop within a LoopGroup first attempts the **merge path** (checker `TryProcessMergedLoopA5/A6`
    → `LoopIterationMerge` / `LoopExpandMerge` semantics); if validation fails, fall back to per-iteration full expansion.
  - **Merge precondition validation** (if any condition is not met, WARN and fall back, same semantics as checker "Skip merged-loop
    optimization"):
    1. Loop body contains only mergeable instructions:
       - A5: Simple TRANS / SYNC TRANS (SyncCKE/GSA/Xn) / REDUCE / ClearCKE
         (`CollectLoopCkeOpsA5` rules; SetCKE, LOAD, LOOP, JMP,
         TransMem with `setRmtCKEMask != 0`, ClearCKE with `clearMask != 0` are all rejected);
       - A6 (`CollectLoopCkeOpsA6` + `Collect*LoopInstrA6` rules): Local TRANS
         4 codes / TransMem (`udfType==0` and `dmaOpCode ∈ {3,6}` -- 5 write with notify
         triggers a remote post per iteration, rejected; channel must be resolvable, `IsExistRemoteDieInfo`) /
         REDUCE (TransMem with `udfEnable` for network reduction also folded per trans slot) /
         ClearCKE (`clearMask==0`); SetCKBit, LOAD, LOOP, JMP, Wait are all rejected;
    2. Wait/set masks are **balanced** within the loop body (`ConsumeProducedMask`/`ProduceMask`:
       each wait's bits must be produced by an earlier set; at sequence end, all produced bits are exhausted);
     3. No cross-expansion Event/CcuBuffer resource conflicts (`CheckResourceConflict`: Event/CcuBuffer IDs after
        ckOffset/msOffset offset across different expansions must not be reused; A6 CcuBuffer conflict checking covers
        local transport/TransMem `msIdMode` CcuBuffer end and reduction msId lists);
    4. Transport/reduction length > 0 (`ValidateLoopLen`) and expansion count × iteration count > 0;
     5. **A6 additional**: Wait Event IDs within the loop body must not be reused (`CheckA6LoopWaitCkeIdUnique`,
       hardware constraint; checker reports error and aborts, here degraded to fall back to full expansion); each expansion's iteration count/
       length/channel values must be consistent (checker: each expansion `loopCnt` equal + `InstrMerge`/
       `MemMerge` mergeability as a performance-side approximation -- register numbers after xnIdOffset offset may
       point to different registers).
  - **Merge launch** (`EmitMergedLoopInstrsV3` semantics): The loop body is executed **once** per sample expansion
    (curExpandCnt=0), slot by slot: wait → data task → set → clear wait bits;
    data task size = single-pass length × (expansion count × iteration count) (timing-equivalent folding, consistent with
    AddBatchTransMem/AddBatchReduce batch tasks in checker); Event register bitmaps for other expansions are replayed by OR/clear
    idempotency, guaranteeing post-loop register state is consistent with full expansion (only
    PostRecord task records are not duplicated). All tasks within the launch window (data tasks and folded
    notify tasks) carry `loop_cnt` (loop count after expansion = N+1 after N expansions) and
    `loop_iter_cnt` (self-loop count per single loop); total iterations = `loop_cnt` ×
    `loop_iter_cnt`, preserving pre-merge loop details for subsequent simulation use; straight-line instructions, full expansion fallback, and
    SQE/CLI path tasks have both fields as 0 (invalid; trace uniformly outputs N/A).
  - **Quantified effect** (REDUCE_SCATTER measured db, 128 expansions × 50 iterations × 13 instructions):
    Task count 192,005 → 66 (10 data tasks, several local post/wait, 35 remote synchronization nodes
    -- 21 remote posts + 14 remote waits, Section 6.6); graph generation + simulation end-to-end
    ~55 s → ~6 ms; makespan 70,113.650 us → 2,753.711 us (this section's merge eliminates
    128,000 task-level notify overheads and 64,000 small transport per-task latencies; the final value after queue parallelization
    and remote synchronization node inclusion, Section 7.4.1).
  - A6 LoopGroup configuration validation (`loopInsCnt<=512`, `expandCnt<=511`,
    `expandOffset<=511` and `expandOffset<=loopInsCnt`, `TransformLoopGroupInstr` in checker);
    if not satisfied, warn and skip the entire LoopGroup (defense against abnormal
    register values causing uncontrolled expansion); the "in-loop waitCKE must not reuse Event ID" validation is covered above in
    merge precondition validation #5 (`ccu_task_transform_instruct_a6_v3.cc:1126-1146`).

### 6.5.1 Merged Loop Pipeline Concurrency Recovery (Loop Tag + Characteristic Chain Resolution)

**Problem -- folded pseudo-serialization (merge is not timing-equivalent for notify structure)**: The merge in Section 6.5 is
timing-equivalent folding for **data tasks** (size × total iterations; total duration unchanged in bandwidth-limited scenarios), but not for Event
wait/set handshakes within the loop body. During real microcode execution, the N expansions of a LoopGroup are launched concurrently; each expansion
internally advances through M inner iterations in pipeline fashion (typically 4KB per iteration); the in-loop
`TransMemToMs → setCKE → waitCKE → TransLocMsToRmtMem / TransLocMsToLocMem`
is a **per-iteration fine-grained handshake**: the (k+1)-th iteration's on-chip memory to CcuBuffer prefetch (memcpy engine) overlaps with the k-th iteration's
network write + local copy-out (channel / memcpy engine); steady-state throughput ≈ max(phase totals).
After merge folding, per-iteration handshakes degenerate into a **single pair** of wait/record nodes; the semantics become "all front-segment data
(× N × M iterations) must complete → then the back segment can start" as a **full barrier** -- pipeline concurrency is folded into serialization.

Using the first loop template from `hccl_vm_install/data/hccl_vm_data.db` as a sample (microcode 0-10:
`TransMemToLocMs(4KB, set 0:1) → waitCKE(0:1) → 7× TransLocMsToRmtMem(4KB each,
set 0:1..0x40) → TransLocMsToLocMem(4KB, set 0x80) → waitCKE(0:ff)`; LoopGroup
128 expansions × 256 inner iterations = 32768 times × 4KB = 128MB/task):

```
Current graph (after 7.4.1 parallelization;  AA = representative of pre-loopgroup task, that is, the last node
                            in the queue skeleton before the loop launch window):
  AA -> memcpy(mem->ms) -> record(0:1)                     ← Prefetch unit (128MB, 95.95us)
  record(0:1) →[paired]→ wait(0:1) → 7× ub_write + memcpy(ms->mem)
                                    ↑ Full serialization: 8 back-segment units fork from wait(0:1),
                                      waiting via the pairing edge for the entire 128MB prefetch to complete
  Multiple { record(0:xx) / record(0x80) -> wait(0:ff) }       ← Loop-end barrier (retained)
```

- Real execution: steady-state ≈ max(prefetch 95.95, 7-port write 2526.48) us;
- Current graph: serial 95.95 + 2526.48 us.
- **Quantified** (sample db ALLGATHER, measured before and after the fix in this section): makespan **2657.52 →
  2561.54 us**. The 95.95 us serial prefetch phase is eliminated; after prefetch and local copy-out run concurrently through on-chip memory dynamic sharing,
  each takes 191.79 us (correct expression of pipeline steady state, Section 7.5); the 7-port network write at 2526.48 us becomes the sole
  critical path. Error upper bound: When two phases belong to different engines (on-chip memory prefetch vs network write) and have comparable durations,
  the loop body duration is overestimated by nearly **2×** (serial t1+t2 vs steady-state max(t1,t2)); same-engine phases
  (such as pure on-chip memory prefetch + copy-out) have serialization error absorbed by on-chip memory dynamic sharing (total unchanged);
  error accumulates linearly with the number of merged loops.

**Correctness premise -- merge validation invariants**: Merge precondition validation #2 in Section 6.5 (wait/set masks balanced within the body)
guarantees that within the merged loop body, **all wait bits are produced by earlier sets within the body and exhausted at the end** -- that is,
all Event dependencies within the merge launch window are intra-loop per-iteration pipeline handshakes; no cross-loop
true data dependencies exist (otherwise validation fails and full expansion is used). Therefore, resolving "in-body intermediate waits and their consumed
sets" does not break the loop's external synchronization: The **loop-end wait** (the last wait in the launch window, after which
no same-loop data tasks follow) and the **mission completion set** (skeleton tail + all unit tails, Section 7.4.1) continue to
carry post-loop instruction / SQE successor / same-queue next-mission queue-level serial semantics.

#### Approach (Loop Tag + Characteristic Chain Identification + Deletion and Reconnection, Iterate to Convergence)

1. **Loop tag**: `_MissionContext` maintains the current launch window's `_loop_tag`, named
   `loopgroup_{i}_loop_{j}` (i = LoopGroup instruction execution sequence number within the dispatch context, 0-based;
   j = cur_loop_idx within the group, 0-based; when LoopGroup is nested in a loop body, each is assigned by its launch window,
   not forming a hierarchy). `_emit_merged_loop` sets the tag within the launch window; `add_task` inherits by default into the new field
   `Task.loop_tag` (same pattern as `loop_cnt`). **Only the merge path sets the tag**; full expansion fallback /
   SQE / CLI / straight-line instructions have an empty string (trace outputs N/A). The tag also serves resolution scope restriction
   (prevents cross-loop / cross-mission mis-deletion) and trace/dump tracing.
2. **Characteristic chain identification** (recorded during launch + determined before assembly, avoiding reverse lookup of graph results): `_gen_waits`
   records (wait_task_id → all contributing post task_ids) into `ctx.wait_posts` upon successful pairing (the aggregated wait's match + extra all contributors, Section 6.4). The assembly phase
   (`expand_ccu_graph` assembly loop beginning, **before** skeleton/unit splitting) scans the ctx.nodes sequence; a characteristic chain match requires all of:
   - W is a locally paired `Notify_Wait` with `loop_tag == T` (no remote wait exists within the merge window --
     validation #2 guarantees all in-body wait bits are locally produced);
   - All contributing posts R* of W are `Notify_Record` with `loop_tag == T`, and each **immediately follows a same-tag data task M\*** in ctx.nodes
     (unit-tail record, that is, the folded producer task, such as TransMemToMs folded memcpy; data tasks ⊆ `_QUEUE_DATA_TYPES`,
     not limited to Ub_Write_Or_Read -- Memcpy/Reduce_Inline also exhibit this pseudo-serialization);
   - **After W (in ctx.nodes order), same-tag data tasks exist** (mid-body determination; the loop-end
     wait has no same-loop data tasks after it, naturally exempted; the next loop's data tasks have a different tag,
     so they are not misidentified);
   - R* do not immediately follow non-data nodes (free set / sync slot local assertions **do not match** -- they
     wait for real dependencies such as remote synchronization; resolution is prohibited).
3. **Deletion and reconnection** (pruning ctx.nodes, not post-assembly graph surgery):
   - **W is always deleted**: Removed from ctx.nodes and the dep task table; its pairing edges (`remove_match`)
     and aggregated extra edges (`extra_edges[W]`) are revoked -- this is the essential operation to break the barrier;
   - **R\* depends on consumers**: If R does not appear in any other wait's `wait_posts` values → delete along with it (in the sample db, set24 is only consumed by wt25, so it is directly deleted); otherwise **retain R and only
     delete W** (its retained edges are unaffected; the loop-end wait continues as normal -- timing-equivalent, semantically lossless
     conservative branch; taken when post bits are partially consumed by multiple waits);
   - **Successor unit reconnection completes automatically**: After pruning, the assembly logic re-scans ctx.nodes; units that originally forked from W
     automatically fork to the preceding skeleton node of W (that is, the pre-loopgroup task AA);
     when W is the skeleton head (no preceding skeleton), the unit becomes an entry unit, with entry dependencies expressed by SQE attachment edges
     (Section 7.4.1). **No graph surgery for edge reconnection is needed** -- fork parents / skeleton chain /
     completion set are naturally generated by existing assembly rules based on the post-pruning sequence.
   - Register / Event records require **zero changes**: Pairing consumption is already completed during launch; pruning is only a graph structure
     operation; post-loop instruction wait matching uses unchanged records (the deleted post's bits
     have been consumed by W, with no residual waiters).
4. **Iteration and statistics**: Repeat steps 2-3 until no new matches are found (handles multi-level chains in the body, such as
   prefetch→wait→middle segment→wait→end segment: after deleting the first level, the next level is exposed as a new chain head);
   `CcuBuildStats` adds `pipeline_chains_broken`; the summary line presents it as `pb=N`.

Post-fix sample task graph (compared to the beginning of this section; deleted nodes are record(0:1)/wait(0:1); the original
"wait(0:1) → ub_write" gate is taken over by AA, timing-equivalent):

```
AA -> wait(0:ff)                                        ← Loop-end barrier (skeleton)
AA -> memcpy(mem->ms)                                   ← Prefetch unit (record deleted)
AA -> Multiple { ub_write_or_read -> record(0:xx) }          ← 7-way write, concurrent with prefetch
AA -> memcpy(ms->mem) -> record(0x80)                   ← Copy-out unit, concurrent with prefetch
Multiple { record(0:xx) / record(0x80) -> wait(0:ff) }      ← Loop-end: wait for all data to complete
```

#### Implementation Details (Code Locations)

- `simulator/task_model.py`: `Task` adds a `loop_tag` slot (default empty string);
  `to_trace_args` outputs `loop_tag` (empty string → `N/A`, same caliber as `loop_cnt`);
- `ccu_graph_builder.py`:
  - `_MissionContext.__init__`: `_loop_tag = ""`, `_lg_seq = 0`, `wait_posts = {}`;
  - `_exec_loop_group`: On entry, record `lg_tag = f"loopgroup_{self._lg_seq}_loop_{
    cur_loop_idx}"` (passed through to `_emit_merged_loop`); after LoopGroup processing,
    `self._lg_seq += 1`;
  - `_emit_merged_loop`: Within the launch window, set `self._loop_tag = lg_tag` (`finally` resets,
    same pattern as `_loop_cnt`);
  - `_gen_waits`: After successful pairing, `self.wait_posts[wait_task.task_id] =
    [p.task_id for p in posts]`;
  - `expand_ccu_graph` assembly loop beginning calls `_break_pipeline_chains(ctx, dep)`
    (returns the number of deleted chains, accumulated into `stats.pipeline_chains_broken`);
- `graph_dumper.py` / `simulator/report.py`: dump `nodes[]` and trace args
  add `loop_tag`; DOT/HTML legend synchronized (optional, presented alongside loop_cnt).

Trace presentation (`export_trace`, display layer only, does not affect simulation): Tasks on a virtual slave stream that carry
`loop_tag` have their display row re-attached to the loop tag row of the owning stream (row name is the tag, such as
`loopgroup_0_loop_0`), indicating they were converted from in-loop instructions; other tasks on the same slave stream
retain their original display row -- mixed slave streams reused across missions are therefore split into the original row + tag row.
Display rows are partitioned by `(stream_id, loop_tag)`; tasks from different virtual streams do not share rows (preserving
inter-slave-stream task partitioning); main stream tasks (including loop-end waits) all retain their original row; task
`args.stream_id` retains the real stream number.


```
_break_pipeline_chains(ctx, dep):
    broken = 0; changed = True
    while changed:
        changed = False
        for W in [n for n in ctx.nodes if type == Notify_Wait and n.loop_tag]:
            posts = ctx.wait_posts.get(W.task_id);  T = W.loop_tag
            if not posts:                              continue   # Unpaired/remote
            if any R in posts: R.loop_tag != T:        continue   # Cross-loop source
            if any R in posts: R's preceding ctx.nodes node is not a same-tag data task: continue
                                                                 # Free set/sync
            if no same-tag data tasks after W:           continue   # Loop-end barrier
            for R in posts:                                        # --- Match ---
                if R in another wait's wait_posts values: continue        # Has retained consumers
                dep delete R; ctx.nodes remove R                       # Unit tail taken over by M
            dep.remove_match(W); dep.extra_edges pop W
            dep delete W; ctx.nodes remove W
            broken += 1; changed = True; break                     # Re-scan
    return broken
```

Acyclicity guarantee: Pruning only deletes nodes and edges, does not add edges; the deleted W/R are all nodes within the launch window;
fork/join edges are recalculated by the assembly phase based on the post-pruning sequence (Section 7.4.1 acyclicity argument continues to hold);
`_drop_cyclic_matches` (Section 5.4) and simulator runtime deadlock detection fallback remain unchanged.

#### Approach Evaluation (Pros and Cons) and Alternatives

**Pros**:

1. Does not change Section 6.5 merge semantics or checker alignment logic -- post-hoc resolution, risk-isolated;
2. Pruning ctx.nodes before assembly (rather than graph surgery on the assembled result): fork parents / skeleton chain /
   completion set are naturally generated by existing assembly rules, avoiding the correctness burden of "edge reconnection after node deletion"
   (post-assembly surgery would require manually reconnecting fork edges, skeleton chain, and completion set, which is error-prone);
3. Recording wait→posts pairing during launch; identification does not depend on the dependency shape of graph results -- insensitive to
   Section 7.4.1 assembly rule evolution (fork parent selection, unit splitting adjustments);
4. Loop tag restricts resolution scope (same loop, same mission), preventing cross-loop / cross-mission
   mis-deletion; the tag also feeds back into trace/dump tracing;
5. Deleting nodes rather than just deleting pairing edges: Avoids retaining a semantically dangling unpaired local wait (although its
   measured `local_wait_delay` = 30ns and has no timing difference, the trace statistics caliber is cleaner).

**Cons / Risks**:

1. The characteristic chain is a structural heuristic; identification conditions couple with launch / assembly logic and must be maintained as both evolve
   (the coupling surface has been compressed to the ctx.nodes sequence layer through "record during launch + prune before assembly");
2. `Task` model field extension requires synchronization of trace/dump/JSON presentation (Chapter 11 caliber);
3. "Full concurrency + bandwidth manager arbitration" is a pipeline **steady-state** approximation: Ignores pipeline fill/drain
   latency (4KB per iteration, sub-microsecond, negligible) and per-iteration handshake **rate** overhead
   (if Event engine throughput is the bottleneck, a dedicated rate model is needed; current folding / deletion both count as single occurrences,
   consistent with pre-fix caliber; registered as Section 10.3 #21);
4. The full expansion fallback path (`mf > 0`) does not go through this resolution -- its per-iteration wait/record is a real 4KB
   granularity handshake, but inter-iteration overlap is also serialized by the queue chain; this path has a large node count and is not
   triggered by current samples; treated as a known residual (Section 10.3 #21).

**Alternative Comparison**:

| Alternative | Approach | Pros | Cons | Conclusion |
|------|------|------|------|------|
| B. Direct judgment during launch (no tag / no identification) | `_emit_merged_loop` directly identifies mid-body waits by slot structure (data slots follow after it) and skips launching their wait/pairing | Simplest implementation, no Task field extension; judgment information (slot order) is at hand, no heuristics | Loses handshake nodes and loop tracing from trace; skipping launch changes register pairing records (loop-end wait absorbs prefetch set, timing harmless but widens behavioral difference from checker); free set / sync scenarios need additional exclusion conditions | Viable simplified variant; choose if trace presentation is not needed |
| C. Segmented pipeline modeling | Fold data tasks into K segments (8~16); retain wait/record between segments; explicitly model fill/drain | Includes pipeline fill effect, more precise | Node count × K, K value has no physical basis; under steady state, differs from full concurrency by only one head/tail segment | Over-engineering; rejected |
| D. Simulator soft dependency edges | New edge type "pipeline edge" (child starts after parent completes X%) | Graph structure unchanged | Simulator semantics become complex; X has no physical basis; and the serial chain structure (unit forks from wait) still needs graph-layer modification | Rejected |

Conclusion: Adopt the original approach (tag + characteristic chain + deletion), with the implementation location and safety
conditions strengthened per steps 2/3 above (record during launch + prune before assembly + mid-body determination + consumer conservative branch).

**Implementation status (implemented)**: `Task.loop_tag` field (task_model / trace args /
dump `nodes[]` / input_builder round trip) + `DependencyGraph.remove_task()`
(including cleanup of extra edges and notify pairing) + `_break_pipeline_chains()` (called at the assembly loop
beginning, counts `pipeline_chains_broken`, summary presents `pb=N`) + trace
loop slave stream rename presentation (`export_trace`, see above). Verification:
Sample ALLGATHER db cross-engine scenario makespan 2657.521 → 2561.541 us (56 → 54
tasks, pb=1, all 9 units fork-reconnected to the preceding skeleton node rwt22, consistent with the manual prototype item by item);
archived repository regression -- same-engine phase scenario (mv→mv all on-chip memory) makespan unchanged (zero-error scenario,
theoretical expectation); SQE-only repository pb=0 unaffected. `loop_tag` is retained as a persistent task attribute
for subsequent performance simulation consumption (such as per-loop bandwidth/rate modeling, Section 10.3 #21).

### 6.6 Graph Generation Algorithm (Pseudocode)

```
expand_ccu(device_id, die_id, sqe_params, instr_space, dep_graph, reg_model):
    Group SQEs into dispatch contexts by contiguous segments (Section 6.1: non-contiguous = new dispatch, independent window)
    Same-queue contexts are serial in dispatch order; each queue takes the first context into the active set
    while active set is not empty:
        progress = False
        for q in each queue's current context in the active set:        # Cross-queue polling
            while ptr[q] has not reached the window end:
                instr = decode(instr_space[ptr[q]])
                if instr is WAIT and reg_model.cke bits are insufficient:
                    break                              # This queue pauses; move to the next queue
                nodes = transform(instr, reg_model)   # Section 6.3 mapping; LOAD only changes registers
                for node in nodes: append to queue q's chain tail
                if instr is SET_CKE: register in reg_model.seen_post
                if instr is WAIT:     pair with seen_post → aggregate wait node (Section 6.4)
                if instr is LOOP:     merge path (Section 6.5 validation) or interpret loop body by iteration
                ptr[q] += 1; progress = True
            if ptr[q] reaches window end: current context complete; next context of the same queue enters the active set
        if not progress:
            # Stall = wait exists depending on remote Event (cross-rank, invisible in single-rank graph)
            For stalled queue: Generate remote Notify_Wait node (notify_id+mask, fixed delay),
                then replay the remaining parts of the instruction (local set / data transport), continue advancing
    SQE stream attachment (Section 7.4.1): Remove placeholder from SQE stream (no placeholder nodes introduced in the task graph);
        Entry edges (SQE predecessor → queue skeleton head) / Exit edges (mission completion set → SQE successor)
        expressed via extra edges
    Assembly splitting (Section 7.4.1): Each mission queue sequence → skeleton (stays on queue chain) + asynchronous units
        (attached to #v virtual stream, stream_id redirected to virtual stream); fork (predecessor skeleton → unit head) /
        completion set join (skeleton tail + all unit tails → SQE successor / same-queue next-mission entry)
        extra edge connection
```

- **Cross-rank post-wait edges are not created, but synchronization task nodes are retained** (single-rank graph generation, the primary difference
  from checker): `GenerateRawCcuGraphsV3` in checker (`ccu_task_transform_v3.cc:1275-1406`)
  advances multi-rank CCU graphs alternately, with cross-rank posts unlocking each other; this plugin has only single-rank instruction data:
    - Instructions that post to remote Events / remote register notifications → generate remote `Notify_Record` nodes
      (src=local, dst=INVALID → fixed `remote_record_delay`, not registered in local Event records,
      not participating in pairing):
     - A5: `rmtCKEId/locCKEMask` of SyncCKE, `setRmtCKEId/setRmtCKEMask` of SyncGSA/SyncXn,
       `setRmtCKE` of TransLocMSToRmtMS (consistent with `AddPost` trigger conditions in checker);
     - A6: TransMem `dmaOpCode=5` write-after-notify and `0x1A` remote atomic count,
       `notifyValid=1` notify of SyncXnWtx, atomic synchronization of SyncAtx -- remote target type
       (Variable/Event) is resolved by peer address mapping (`GetXnAndTypeIdByAddr` in checker); not obtainable in single rank,
       all approximated as remote record (mask takes the lower 16 bits of the notification value for trace display only).
     Value synchronization of SyncGSA/SyncXn (A5) and main write of SyncXnWtx (A6) is only a remote register write
     (checker creates nodes/writes state on the peer graph); the local side does not generate a separate task.
    - Wait bits that cannot be satisfied by local-rank posts (detected by stall detection) → the wait depends on remote → generate
      a remote `Notify_Wait` node (fixed `remote_wait_delay`, not paired), and replay the remaining parts of the
      instruction; unsatisfied waits within loop bodies also generate remote wait nodes before continuing. When partially satisfied: already-satisfied
      bits pair normally with local posts (aggregated into 1 local wait node, Section 6.4); unsatisfied bits generate one
      remote wait node. A6 independent Wait instruction (Variable value comparison) when unsatisfied uses the same mechanism: top-level stall →
      stall resolution generates remote `Notify_Wait` (task id tagged `xwt`) then passes through.
    - No nodes are created from nothing in the task graph: All nodes originate from original db data or microcode instruction conversion
      (attachment dependencies are expressed via extra edges, see Section 7.4.1).
- Local-rank post-wait pairs normally: Between multiple queues of the same device, Events across multiple dies
  are all within the local graph; edges are built per Section 6.4 rules.

### 6.7 Simplified Comparison with Checker CCU Graph Generation

**CCU task rank/eid/direction semantics**:

- **Channel-type network tasks** (write/read): Local rank takes opDetails.rankId; peer rank is looked up via
  the channel peer eid (reid) through the eid to rank index (Section 4.8); INVALID + `rank_unresolved` count if not found.
  Equivalent to `GetRemoteRankByChannel` in checker
  (`g_allRankChannelInfo`: channelId → peer), but using the communicator rank from RankEidMap
  instead of the physical dstDeviceId in checker.
- **Reduction tasks** (REDUCE microcode): CcuBuffer is a rank-local buffer (checker treats all msId values as always
  `GenSliceFromMs` + `AddLocalBatchReduce`); reduction is always a local on-chip memory operation; src/dst are both filled with
  the local rank; cross-rank data has already been charged in the preceding Rmt* role transport; reduction does not re-charge the network.
- **Local tasks** (local transport Memcpy, local reduction Reduce_Inline): src/dst are both filled with the local rank.
- **is_write**: All UB tasks produced by CCU explicitly carry it (write=True / read=False);
  the simulator does not depend on rank heuristics.
- **Local rank self-validation**: If the channel leid and the rank resolved from the index are inconsistent with opDetails.rankId,
  opDetails takes precedence and a warning is issued (eid numbering system offset protection, such as when deviceId and rankId have an offset).

| checker Step | light_performance | Reason |
|--------------|-------------------|------|
| Two-phase (raw graph → queue parallelization → reconnection) | Single-phase interpretation + assembly-phase queue parallelization splitting (Section 7.4.1: simplified port of `ApplyCcuQueueParallelizationV3`, skeleton/unit splitting + fork/join and attachment extra edges + `#v` virtual streams) | No validation needed; no need to preserve a transformable raw intermediate form; parallelization is completed once during assembly (reuse granularity and other differences per Section 7.4.1) |
| Loop merge (BatchTransMem/BatchReduce, resource conflict checking, producedMasks balancing) | **Equivalent implementation** (Section 6.5: precondition validation + folded launch + full expansion fallback; memory slice merging is not needed -- simulation only cares about data volume and timing) | Align with checker semantics to eliminate node bloat |
| SQE grouping into TaskCcuGraph (`IsContiguousCcuSqe` contiguous-merge + `CcuMissionKey` latest-node-append) | **Equivalent implementation** (Section 6.1: contiguous segment splitting into dispatch contexts, each context with independent window replay + sequential args consumption; register/Event shared across contexts, same-queue contexts serial in dispatch order) | Align with checker multi-dispatch (window wraparound re-delivery) semantics; previously merging into a single context lost the second dispatch's data transport (Section 10.3 #20 fixed) |
| GetRemoteRankByChannel (channelId → dstDeviceId/dstDieId, mapped to rank via Communicator table) | Peer eid (reid) → RankEidMap/ranktable index → communicator rank (Sections 4.8/6.7) | In estimate mode, the Communicator table has only one row for the local rank; the rankId field in ccuChannels is the endpoint deviceId and is unusable |
| AllRankParamRecorder full address space mapping (Variable/Event/PFE/CHANNEL/JETTY/MISSION/CcuBuffer ...) + on-chip memory data content + remote address to register reverse lookup (`GetXnAndTypeIdByAddr`) | Only retain Variable/Event/Address/channel four categories; A6 Load/Store on-chip memory data plane and remote target reverse lookup are not modeled (Section 6.4, `_MissionContextA6` deviation list) | Simulation only needs length and peer; data content/remote register number does not affect timing |
| A6 checker error-abort scenarios (unsupported dmaOpCode/dstType, clearType!=1, clearMask!=0, LoopGroup config illegal, LoadX/StoreX indirect number out of range, and so on) | Degraded to warning + skip data plane (synchronization semantics still execute) or skip instruction | Performance tool tolerates incomplete/abnormal data while maintaining queue skeleton integrity |
| CcuTraceInfo tracing, ccu_convert_config.json generation | Removed | Specific to validation/rewriting tools |
| CheckAllPostMatch error reporting | Degraded to warning + diagnostic statistics | Adapted for single-rank data |
| Cross-rank post unlocking other's wait (multi-rank joint advancement) | **No cross-rank edges created**: Cross-rank post/wait generates remote synchronization nodes (fixed delay); peer readiness is approximated by fixed delay | Single-rank graph generation (core difference) |

### 6.8 A5 and A6 Instruction Set Differences Overview

| Dimension | A5 (V1/Ascend950) | A6 (V2/Ascend960) |
|------|--------------------|--------------------|
| LOAD type | 7 instructions: including Address family | 18 instructions: unified Variable register + LoadX/StoreX/ClearX/Nop + arithmetic-logic (Add/Sub/Mul/And/Or/Not/Xor/Shl/Shr/Popcnt) |
| CTRL type | 5 instructions: Loop/LoopGroup/SetCKE/ClearCKE/Jmp | 7 instructions: + SetCKBit/Wait/Fence |
| TRANS type | 12 instructions: Mem to CcuBuffer all combinations + SyncCKE/GSA/Xn | 7 instructions: 4 local types + unified TransMem (dmaOpCode distinguishes) + SyncWtX/SyncAtX |
| REDUCE type | Add/Max/Min (msId list addressing, <=8 CcuBuffer) | ReduceAdd/Max/Min (same structure, no embedded waitCKE) |
| Loop register | v1.loop{xnId, startInstrId, endInstrId}, loopCnt in Variable | v2.loop{xpId(context), xnId, xmId(iteration count), mode}, mode==1 conditional expansion (Section 6.5) |
| LoopGroup parameters | Variable: LoopNum[61:55]/RepeatNum[54:48]/NoRepeatNum[47:41]; Xm: gsaOffset/CcuBufferOffset/ckeOffset | Variable: loopInsCnt:10/expandOffset:9/expandCnt:9; Xm: ckOffset:10/msOffset:11/gsaOffset:32; Xp: xnIdOffset:32 |
| Register number addressing | Direct use from payload field | `GetXnId`: when bit15=1, offset by loop xnIdOffset (Section 6.4) |
| Embedded synchronization fields | TRANS/REDUCE commonly carry waitCKE + setCKE + clearType | Local TRANS/REDUCE only carry setCKE (wait handled by independent SetCKBit/ClearCKE/Wait instructions) |
| Channel/length value source | Channel is instruction immediate field | Channel/length are both Variable register values (`Xn[xcId]`/`Xn[xlId]`) |

Both instruction set generations are implemented for graph generation (the A6 interpreter `_MissionContextA6` reuses the task launch/
Event pairing/advancement/merge launch infrastructure from A5, only overriding instruction decoding and loop organization); the A6 instruction to
task type full table is in Section 11.3.3; the established deviation list with checker is in the `_MissionContextA6`
class docstring and the Section 6.7 comparison table.

---

## 7. Performance Simulation Module (Sub-Requirement 4, Incorporated per HCCLSIM Baseline)

> This chapter describes the design of the `simulator/` sub-package (ported from `docs/light_performance/HCCLSIM/hccl_sim/`);
> interfaces are based on the HCCLSIM authoritative baseline. Differences from the old version and residual items are in Chapter 10.

### 7.1 Overview

```python
sim = HcclSimulator(config, dep_graph, topo_resolver)
result = sim.run(trace_filepath=None, max_trace_events=0)   # → dict
```

The simulator accepts three parameters:

| Parameter | Type | Content | Builder |
|------|------|------|--------|
| `config` | `SimConfig` | Latency, bandwidth, utilization configuration | Caller |
| `dep_graph` | `DependencyGraph` | Task nodes + dependency relationships | `graph_builder` (db path) / `input_builder` (JSON path) |
| `topo_resolver` | `TopoResolver` | eid to physical port resolution (ranktable.json) | Caller |

`run()` returns a result dict (no longer returns raw makespan):

```python
{
    "makespan": float,        # End-to-end duration (seconds)
    "makespan_us": float,     # (microseconds)
    "trace_file": str|None,   # Trace output path
    "trace_events": int,      # Trace event count
    "sim_time_s": float,      # Software runtime (seconds)
    "tasks": list[Task],      # All tasks (with start/end_time)
    "tasks_by_stream": dict,  # int stream id → [Task] (chain order)
    "ports": dict,            # tuple(ports) → NetworkPort
}
```

The simulator does not accept a task sequence list and does not infer dependencies -- everything is obtained from `dep_graph`.

### 7.2 Parameter One: `config: SimConfig`

**Unit convention**: Internal latency=seconds, bandwidth=B/s; **JSON configuration units latency=us, bandwidth=GBps**
(`from_dict`/`to_dict` perform automatic conversion). Port identifiers are `"die_id/port_number"` strings
(such as `"1/0"`), consistent with ranktable.json.

#### Field Definitions

| Field | Type | Default | Description |
|------|------|--------|------|
| `local_record_delay` | float | `1e-6` | Local record fixed duration; JSON value unit us |
| `remote_record_delay` | float | `1e-6` | Remote record fixed duration; JSON value unit us |
| `remote_wait_delay` | float | `1e-6` | Remote wait / unmatched local wait fixed duration; JSON value unit us |
| `port_bandwidth` | dict | `{}` | `"die/port"` → bandwidth (B/s). One eid can correspond to multiple ports, configured per port, total bandwidth summed |
| `default_port_bandwidth` | float | `53.125e9` | Default bandwidth for unconfigured ports (425 Gbps = 53.125 GBps); JSON value unit GBps |
| `port_type` | str | `"x4"` | Port type (`x2`/`x4`), authoritative source is db `Server.port_type` (server/pod topology configuration `port_type` field stored via servers_info.json). Configured bandwidth is x4 baseline; `x2` halves non-d2h port bandwidth (see "Port Type Scaling" below) |
| `hbm_bandwidth` | float | `1.4e12` | On-chip memory bandwidth, used for Reduce_Inline/Memcpy; JSON value unit GBps |
| `bw_profile_set` | BandwidthProfileSet? | `None` | Bandwidth utilization configuration set; when None, utilization is 100% (includes multi-jetty scaling step `jetty_utilization_step`, see below) |

#### Port Bandwidth Configuration Format

```json
{"port_bandwidth": {"1/0": 53.125, "0/7": 53.125}}
```

#### Port Type Scaling (`port_type`, x2/x4)

Port bandwidth values in `sim_config.json` (`<topo_type>_port_bandwidth` and
`default_port_bandwidth`) are all **x4 baseline**. The `port_type` field in server/pod topology configuration files
(`x2`/`x4`, default `x4`, illegal values cause an error) is written by
`generate_cluster_topo.sh` into the cluster-level `servers_info.json` per **leaf node** entry
(server form: one entry per server; POD form: one entry per board, with explicit
`super_pod_id`/`server_id`/`board_id` triple id, for the parser to match by id rather than by positional order;
this file is tool-specific and does not enter topo.json/rootinfo/ranktable or other hccl workload native format files),
then stored by the C++ cluster parser (`ParseServersInfoJson`) into the `Server.port_type` column
(`sim_models.h`/`db_sim_runner_ops.h`, RunnerDB v5; from v5 onward the `Server` table adds
a `board_id` column, with one `Server` row per board in POD form and `board_id` always 0 in server form).

During simulation, the `pipeline`/`__main__`/`dump_graph` entry points call
`db_reader.load_port_type(db_path, device_id)` to read the port type from the `Server`
table in `hccl_sim.db` (located by `Device.id → server_id → Server.id` to the server owning the target device;
falls back to the sole value in the entire table when not located; defaults to `x4` when the old database lacks the `port_type` column), then applies it
via `SimConfig.with_port_type()`. `get_port_bandwidth()` multiplies **non-d2h**
ports by `port_type_scale` (`x4=1.0`, `x2=0.5`), that is, x2 port bandwidth is halved;
`d2h_bandwidth` and on-chip memory bandwidth are not affected. The `--port-type` command line option can override the db value.

#### Bandwidth Utilization Configuration (`bw_profile_set`)

```
BandwidthProfileSet
├── profile_low:   BandwidthProfile  ← Used when shared port concurrency < 7
├── profile_high:  BandwidthProfile  ← Used when shared port concurrency ≥ 7
└── THRESHOLD: 7
```

Each `BandwidthProfile` is a mapping table from data size (32B~128MB) to utilization (0~1); queries round down to
the largest tier not exceeding the data size (floor matching).

JSON format:
```json
{
  "profile_low":  {"32": 0.30, "64": 0.35, "...": 0.90, "134217728": 0.95},
  "profile_high": {"32": 0.20, "64": 0.25, "...": 0.84, "134217728": 0.85},
  "single_port_utilization": 1.0,
  "jetty_utilization_step": 0.02745
}
```

> The actual key names in the file are `multi_port_utilization_low` / `multi_port_utilization_high`
> (`from_dict` also accepts the old key names `profile_low` / `profile_high`).

#### Multi-Jetty Scaling (`jetty_utilization_step`)

`multi_port_utilization_low` is calibrated for a regular configuration of **one channel, one jetty**.
When `HCCL_UB_MULTI_CHANNEL_NUM=n`, one channel has n jettys, and the concurrent transport capability scales approximately linearly;
lightweight simulation does not build a jetty-level model (jetty is only used for eid/protocol resolution, see Section 5.5), so instead
the jetty count is scaled into an overall uplift of the `profile_low` table:

```
util'(size) = min(1.0, util(size) + (n - 1) × jetty_utilization_step)
```

- `n` is taken from `opDetails.ubMultiChannelNum` (the proxy level0 stub reads the environment variable and stores it, Section 4.2),
  applied per operator; when `n ≤ 1` (not configured/invalid, falls back to 1), the original table is used directly without uplift;
- `jetty_utilization_step` range is `[0.0109, 0.044]`, defaulting to the midpoint `0.02745`,
  configured only by `bw_profile.json` (out-of-range values produce a WARNING and are clamped to the interval boundary);
- Only applies to `profile_low`: `profile_high` is the high-concurrency tier (ports are already saturated by multi-path transport, so multi-jetty
  provides no additional gain); single-port/`d2h` use constant utilization (`single_port_utilization` /
  `d2h_utilization`), and are not scaled;
- Entry point: `SimConfig.for_jetty_num(n)` → `BandwidthProfileSet.for_jetty_num(n)`
  (`bandwidth_profile.py`); the pipeline generates a scaled config copy per operator;
  debugging can use `--jetty-num` to override the db value (`run_perf_sim.py` / `python -m light_performance`).

### 7.3 Parameter Two: `dep_graph: DependencyGraph`

#### Four Layers of Content

```
DependencyGraph
├── tasks: dict[str, Task]                       ← Task nodes (with all parameters)
├── stream_chains: dict[int, list[str]]          ← Serial dependencies (int stream id)
├── notify_matches: dict[str, str]               ← Notify dependencies (wait_task_id → record_task_id)
└── extra_edges: dict[str, list[str]]            ← Extra dependency edges (child_task_id → [parent_task_id, ...])
                                                  ← Sources: CCU queue parallelization fork/join edges and SQE attachment edges (Section 7.4.1)
```

#### First Layer: Task Nodes (`Task`)

Each Task carries all parameters needed for execution (invalid values uniformly use `INVALID = 0xFFFFFFFF`,
eid is empty string, string attributes are `""`):

| Field | Type | Applicable Task Types | Simulator Purpose |
|------|------|------------|-----------|
| `task_id` | str | All | Dependency graph index |
| `task_type` | TaskType | All | Determines latency model (internal classification, Section 7.3) |
| `task_name` | str | Display use | trace/dump display name: profInfo.profTaskType op name (uniformly populated by `_add_task`, Section 11.4.1); when empty, `display_name` falls back to `task_type.value` (CCU sub-tasks/CLI input) |
| `stream_id` | **int** | All | Serial chain membership |
| `notify_id` | int? | Notify_Record/Wait/Write_With_Notify | Dependency pairing identifier (CCU path = Event id); Write_With_Notify for display only (its notify targets the peer, does not participate in pairing) |
| `mask` | int | Notify_Record/Wait | Notify bit mask (CCU path): notify_id + mask uniquely identify one notify action; wait node carries aggregated mask (OR of all contributing post consumption bits for one wait instruction, Section 6.4); SQE path and 0=unfilled |
| `src_rank` / `dst_rank` | int? | notify/transfer | Local determination (src==dst); read direction local determination |
| `size` | int | Memcpy/Reduce_Inline/Ub_*/Write_With_Notify | Compute latency + look up utilization |
| `port` | str/list? | (reserved field) | Ports are resolved in real-time by eid; db path does not fill |
| `src_eid` / `dst_eid` | str | Ub_*/Write_With_Notify | eid → ranktable → physical ports |
| `transport_type` / `link_type` | str | Display use | msprof reconciliation dimensions (profInfo transportType/linkType name, Sections 11.4.2/11.4.3, uniformly populated by `_add_task`; CCU sub-tasks do not fill) |
| `data_type` | str | Display use | trace/report annotation |
| `protocol` | str | Display use | Channel CommProtocol name (Section 11.4.2 note; empty when no channel information) |
| `jetty_id` | int? | Ub_*/Write_With_Notify | trace annotation |
| `is_write` | bool? | Network tasks | Charging direction and port query eid selection (Section 10.3 #1) |
| `start_time` / `end_time` | float | All | Filled by the simulator (-1=not executed) |

7 TaskTypes:

| TaskType | Latency Model | Description |
|----------|---------|------|
| `NOTIFY_RECORD` | Fixed latency (local also triggers Event) | Local = `is_local_notify`(src==dst); unpaired local record uses fixed latency only |
| `NOTIFY_WAIT` | Wait for Event / fixed latency | matched → wait for the Event of the record; unmatched/remote → fixed `remote_wait_delay` |
| `REDUCE_INLINE` | On-chip memory dynamic sharing | `size / hbm_bandwidth`, shares bandwidth with other on-chip memory tasks |
| `MEMCPY` | On-chip memory dynamic sharing | Same as above (on-chip memory copy, not network transfer) |
| `UB_INLINE_WRITE` | Fixed latency (same as Notify_Record: local/remote record_delay) | Notify inline write; SQE graph generation path does not produce it, only the old input path (task.txt) uses it |
| `UB_WRITE_OR_READ` | Port bandwidth sharing (write/read two directions charged independently) | Network data send/receive; `is_write` explicitly carried by graph generation (SQE network tasks in both directions uniformly map to this type) |
| `WRITE_WITH_NOTIFY` | Port write direction sharing + post-write notify fixed latency | Network write + attached notify single task, one-to-one with db `WRITE_WITH_NOTIFY` (7); notify uses fixed remote record delay (peer wait is not in this db) |

#### Second Layer: Serial Dependencies (`stream_chains`)

```
int stream id → [task_id_0, task_id_1, ..., task_id_n]
```

- List order = execution order: task[i] must complete before task[i+1] can execute
- Each stream_id launches one simpy process, executing sequentially by `yield` along the chain
- Different streams are naturally parallel
- Stream id addressing for the db path: SQE streams use the original streamId directly; **CCU queue skeleton chains belong to their SQE's original stream**
  (Section 6.5, whichever stream a CCU-type SQE is on, the expanded tasks are on that same stream; only when the original stream is occupied by a queue from another die
  is a synthetic stream allocated); virtual streams are allocated from base `1<<20`
  (Section 6.5, virtual stream logical names carry a `#v{n}` suffix); logical names are retained via `stream_labels`;
  `GraphBuilder.stream_meta` provides real/ccu/virtual three-category classification and virtual stream
  parent_sid (Section 7.4.1, pointing to the stream id of the queue it belongs to, normally the SQE original stream) on top of this, for report/trace/dump annotation

#### Third Layer: Notify Dependencies (`notify_matches`)

```
wait_task_id → record_task_id
```

Explicit 1:1 pairing: which wait waits for which record. When multiple records exist for the same notify_id, only one
pairs with the wait; unpaired records use fixed latency.

#### Fourth Layer: Extra Dependency Edges (`extra_edges`)

```
child_task_id → [parent_task_id, ...]
```

- Semantics: The child must wait for **all** parents to complete before starting (pure timing constraint, does not carry
  record/wait event semantics -- event pairing is still expressed by `notify_matches`).
- Current sources: Fork edges from CCU queue parallelization (predecessor skeleton → unit head), mission completion set
  join edges (skeleton tail + unit tails → SQE successor / same-queue next-mission entry), CCU subgraph
  attachment edges on the SQE stream (entry/exit, see Section 7.4.1), and multi-parent edges for aggregated wait nodes
  (wait → all contributing posts, Section 6.4 -- the first contributing post uses `notify_matches`, the rest use this layer).
- Writer side `add_edge` automatically deduplicates (only one edge per same child-parent pair); the simulator queries via
  `get_extra_parents` / `extra_parent_ids`.
- Acyclicity: The above edges always connect "forward" along the original instruction order (skeleton before unit, unit before join point,
  attachment before and after placeholder position, post registered before the wait instruction that consumes it); the graph generation direction guarantees acyclicity,
  and is also accounted for in the `_drop_cyclic_matches` cycle detection fallback (Section 5.4).

### 7.4 Simulator Consumption Flow

```python
def run(self, trace_filepath=None, max_trace_events=0):
    # 1. Retrieve all task nodes
    tasks = list(dep_graph.tasks.values())

    # 2. Create synchronization primitives: only matched waits create Events
    for wait_task_id in dep_graph.matched_wait_ids:
        notify_events[wait_task_id] = env.event()
    for parent_id in dep_graph.extra_parent_ids:  # All parent tasks from extra edges
        task_done_events[parent_id] = env.event() # Must be pre-built before stream process launch:
                                                  # If parent completes first and child builds Event later, it will wait forever (Section 7.4.1)

    # 3. On-chip memory bandwidth manager (Reduce_Inline + Memcpy shared, utilization always 1.0)
    hbm_mgr = _BandwidthManager(env, config.hbm_bandwidth, empty utilization table)

    # 4. Launch one simpy process per stream, executing serially along the chain
    for task in chain:
        for pid in dep_graph.get_extra_parents(task.task_id):
            yield task_done_events[pid]           # Extra edges (Section 7.4.1), start_time does not include waiting
        if NOTIFY_RECORD:
            yield timeout(calc_notify_record_delay(task, config))
            if local and matched: trigger the Event of the matched wait
        elif NOTIFY_WAIT:
            if local and matched: yield notify_events[task_id]   # Wait for Event
            else:                 yield timeout(remote_wait_delay)
        elif REDUCE_INLINE / MEMCPY:
            yield hbm_mgr.send(task_id, size)                 # On-chip memory dynamic sharing
        elif UB_INLINE_WRITE / UB_WRITE_OR_READ / WRITE_WITH_NOTIFY:
            is_write = task.is_write  # Explicit direction; None for old input path falls back to heuristic
            eid = src_eid if is_write else dst_eid
            ports = topo_resolver.get_ports_by_eid(eid)       # Throws if missing
            port = look up/create NetworkPort(ports, total bandwidth)
            yield port.send(task_id, size, is_write)          # Port bandwidth sharing
            if WRITE_WITH_NOTIFY:
                yield timeout(calc_write_with_notify_delay(task, config))
                                                           # Notify after write landing:
                                                           # Fixed record delay
                                                           # (peer wait is not in this db)
        task.end_time = env.now
        if task_id in task_done_events: task_done_events[task_id].succeed(env.now)
    # 5. makespan = max(end_time); assemble result dict (including trace export)
```

Latency formulas are unified and maintained at a single point in `task_timing.py`.

#### 7.4.1 CCU Queue Parallelization and Virtual Streams (extra edges)

**Problem** -- The CCU queue chains produced by Sections 6.5/6.6 are per-task serial streams: All
data transport, wait, and set operations in the same queue execute sequentially. On real hardware, asynchronous transport units within a queue execute concurrently;
serialization causes three distortions:

| Distortion | Mechanism |
|------|------|
| makespan overestimation | Asynchronous transport is separated by skeleton (wait/set) one by one, preventing overlap |
| Utilization tier distortion | Concurrent transport within the same queue degenerates to sequential execution; `_BandwidthManager` observes port concurrency always as 1, always querying the profile_low tier (Section 7.5) |
| Port peak concurrency statistics distortion | `peak_concurrency` is always 1, unable to reflect concurrent bandwidth sharing behavior (Section 8.2 port-level metrics) |

**checker reference** -- `ApplyCcuQueueParallelizationV3` (`ccu_task_transform_v3.cc:1052`):
After raw graph generation, a second pass transforms the graph; `CollectAsyncBoundRecordNodesV3` collects asynchronous transport nodes,
`QueueIdReuseAllocatorV3` reuses virtual queue ids by `[startNodeId, afterNode]` intervals,
fork/join edges reconnect to the skeleton.

**Simplified port** (assembly loop in `ccu_graph_builder.py`, completed in one pass during the assembly phase, no raw
intermediate form) -- for each mission's queue node sequence, perform splitting:

1. **Unit identification**: `_QUEUE_DATA_TYPES` (`Memcpy`/`Ub_Inline_Write`/
   `Ub_Write_Or_Read`/`Reduce_Inline`) data tasks + immediately following contiguous
   `NOTIFY_RECORD` chain = one **asynchronous unit**; other nodes (wait / free set /
   remote synchronization nodes) are **skeleton**, staying on the queue serial chain -- the queue chain is the skeleton itself,
   no placeholder nodes are introduced at any position.
2. **Fork edges**: `add_edge(unit head, preceding skeleton node)` -- the unit must wait for the queue to advance
   to its predecessor skeleton before starting (adjacent units share the same fork parent); units appearing before any skeleton
   (mission first node is a data task) have no skeleton predecessor; their entry dependencies are expressed by attachment edges instead.
3. **Completion set join**: Mission completion set = skeleton tail + all unit tails, joined via extra edges to
   -- the entry of the next mission on the same queue (skeleton head + pre-skeleton unit head, queue serial semantics;
   the order between skeleton tails is already covered by the queue chain serialization) and SQE stream successor tasks (exit edges,
   added during the graph_builder attachment phase).
4. **Virtual stream reattachment**: Per-queue counter/reuse pool (`vstream_seq`/
   `vstream_free`/`vstream_busy`); unit task `stream_id` is redirected to
   `ccu_d{die}_q{queue}#v{n}` (same `1<<20` base address space); unit chains are registered in
   `result.virtual_chains` and `add_stream_chain` -- the serial chain only constrains skeleton and
   unit-internal task order.
5. **SQE stream attachment** (`graph_builder._expand_ccu` consumes `result.attach`):
   CCU placeholder is removed from the SQE stream chain (no placeholder/spurious nodes are produced in the task graph);
   entry edges = SQE predecessor tasks (including immediately preceding CCU mission completion sets) → skeleton head and
   pre-skeleton unit head; exit edges = mission completion set → SQE successor tasks. All expressed via extra edges;
   the timing semantics are: The subgraph can start only after the SQE predecessor (and preceding mission completion set) completes;
   SQE successor tasks can start only after all mission completion sets of the subgraph finish.
   Empty chains are removed from the SQE stream for all-placeholder tasks.

**Event timing preservation**: Splitting does not alter `add_match` pairing -- whichever unit a record falls into, and whether a wait
stays on the skeleton, does not affect event pairing (wait crosses streams via notify_events). In the measured sample,
the trailing memcpy unit's fork parent wt28 happens to be on the accounting wait of the preceding read unit's tail set;
the cross-stream event + fork edge together guarantee the "read complete → trailing copy" original timing. **Exception**: Mid-body wait/record
pairing within a merged loop launch window is a folded pseudo-serialization barrier (per-iteration pipeline handshake
folded into a full barrier), resolved by Section 6.5.1 characteristic chain removal before assembly -- this section's splitting is based on the post-pruning
sequence, and the skeleton/unit/fork structure is not affected.

**Reuse rule differences from checker**:

| Dimension | checker (V3) | light_performance |
|------|--------------|-------------------|
| Reuse granularity | afterNode interval (join successor node) level | Mission level (all previous ones released when entering a new mission) |
| Intra-mission reuse | Allowed (DAG explicit edges, no pseudo-dependencies) | Not reused (units within the same mission are concurrent with each other; reuse on the same stream under the chain model would introduce pseudo-serialization) |
| Residual strictness | GRAPH_UNMATCHED error exit | Tolerated (performance tool, Section 6.4 degraded warning) |

Cross-mission reuse safety: Units reusing the same virtual stream are serial within that stream chain (the later mission's
unit chain follows the earlier mission's unit chain), with no timeline overlap.

**Acyclicity guarantee**: Fork (skeleton → unit head) and completion set join edges (unit tail → SQE successor /
next mission entry) always connect "forward" along the original instruction order; splitting does not change the original order;
`_drop_cyclic_matches` cycle detection also accounts for extra edges (Section 5.4) as a fallback; the simulator
also has runtime deadlock detection (reports error and exits when task `end_time<0`).

**Statistics and annotation**: `CcuBuildStats` carries parallelization (`parallel_units`/`virtual_streams`/
`fork_edges`/`join_edges` -- join_edges now counts completion set joins and SQE attachment edges), remote synchronization
(`local_posts`/`remote_posts`/`wait_matched`/`remote_waits`) and other counts;
summary line presents as `pu/vs/lp/rp/wm/rw`;
`GraphBuilder.stream_meta` classifies by logical stream name -- `s{n}` → real, contains `#` →
virtual (with parent_sid attached), others → ccu, consumed by three locations: report (`Stream count:
N (including virtual M)`, virtual stream annotation `vir_stream {n} (label)`, n numbered from 0 in allocation order,
detail collapsed when exceeding 64 streams), trace (virtual stream thread_name `vir_stream
{n} ({label})` + task args `virtual_stream: true`), dump (streams[] with
kind/virtual/parent_sid,
extra_edges list; DOT purple dashed cluster + blue dotted extra edges, HTML badge
+ dotted arcs). When stream_meta is absent (simulator standalone CLI old input), no annotation is made, for backward compatibility.

**Quantified effect** (sample db REDUCE_SCATTER iter0, 66 tasks): Queue parallelization reduces makespan from
20,404.20 to 5,149.36 us (approximately 3.96x); two subsequent corrections further reduce to **2,753.71 us**:
① Remote CcuBuffer reduction correction (Section 6.3.2, reduction is always local on-chip memory) -- the original second-phase 128MB reduction was
charged as network read at ~2526 us, corrected to on-chip memory ~96 us; ② Cross-rank synchronization node completion (Section 6.6) --
14 waits depending on remote assertion were previously skipped (read units launched early at t≈0), corrected to simulate with fixed
remote delay. 10 asynchronous units reattached to 10 virtual streams (fork edges 10, attachment/join edges generated based on
SQE neighbors and queue serial conditions) -- first phase 8 units concurrent (7 network read units each
~2526 us, on 7 independent ports `1/0`..`1/6` + 1 local memcpy ~96 us), followed by
reduction ~96 us and trailing memcpy ~96 us in serial connection. Note that single-port read peak concurrency is still
1 -- the benefit comes from **cross-port overlap**, not single-port concurrency improvement.

### 7.5 Bandwidth Sharing Mechanism (`_BandwidthManager`)

#### Generic Manager (Ports and On-Chip Memory Share the Same Model)

- **Single management process model**: When a new transfer joins, `interrupt` the management process; the management process equally divides bandwidth by the current concurrency N:
  `bw_per = total bandwidth / N`.
- **Utilization query basis is `min_remaining`**: The same group of concurrent tasks queries the
  utilization table using "minimum remaining data volume" (concurrency <7 uses profile_low, ≥7 uses profile_high), same basis within the group.
- **profile_low table is already scaled by jetty count**: Before the table lookup, the pipeline has already used
  `SimConfig.for_jetty_num(opDetails.ubMultiChannelNum)` to generate a config copy;
  when multi-jetty (`HCCL_UB_MULTI_CHANNEL_NUM>1`), the table values are overall uplifted; the manager itself does not perceive
  jetty (Section 7.2 "Multi-Jetty Scaling").
- Effective bandwidth = `(total bandwidth / N) × utilization(min_remaining, N)`; compute the earliest completion time,
  timeout settles progress; recalculate when new transfers join or existing ones complete.
- **On-chip memory dynamic sharing**: `Reduce_Inline` and `Memcpy` share one on-chip memory manager
  (utilization always 1.0) -- when multiple on-chip memory tasks are concurrent, they equally share on-chip memory bandwidth (different from the old version where each task independently used
  `size / hbm_bandwidth`, see Section 10.3).

#### Bidirectional Independence

Each `NetworkPort` contains write and read independent managers, with independent bandwidth that does not interfere:
When port bandwidth B is configured, the write direction can use B, and the read direction can also use B simultaneously.

#### Multi-Port Aggregation

When one eid corresponds to multiple physical ports, `port_key = tuple(port_ids)` serves as a logical port,
total bandwidth = sum of individual port bandwidths (`get_ports_total_bandwidth`).

### 7.6 Parameter Three: `topo_resolver: TopoResolver`

- Input `ranktable.json` (`TopoResolver(ranktable_path=...)`), providing
  **eid (16B hex string) → physical port list (`["die/port", ...]`)** resolution.
- Transport task routing: Write direction queries `task.src_eid`, read direction queries `task.dst_eid`.
- **Throws directly when eid has no matching port** (no jettyId fallback, no default port) -- therefore the db path
  skips simulation entirely when "network tasks exist but no ranktable is available" (Section 3.3); the db graph generation counts and warns about
  tasks with unresolvable eids (Section 5.5).
- Port bandwidth is looked up per resolved port individually from `SimConfig.port_bandwidth` and summed.

### 7.7 Construction Methods and CLI

All input methods uniformly construct `(config, dep_graph)`:

| Input Source | Construction Function | Returns | Status |
|--------|---------|------|------|
| **hccl-vm db (new in this effort)** | `graph_builder.build(group)` | `(config, dep_graph)` | Chapters 5 and 6, main input |
| JSON task file | `load_dep_graph(filepath)` + `load_config(...)` | `(config, dep_graph)` | Retained (hccl_sim compatible) |
| task.txt / algorithm-generated / other old input sources | -- | -- | **Removed with HCCLSIM version** (see Section 10.3) |

Construction flow is unified as:
1. Create `Task` nodes, call `dep_graph.add_task(task)`
2. Call `dep_graph.add_stream_chain(stream_id, task_ids)` in task order within the stream
3. Call `dep_graph.add_match(wait_task_id, record_task_id)` based on notify pairing results

CLI entry points:

```bash
# db input (light_performance main entry, see Section 3.3 for details)
python3 -m light_performance --db data/hccl_vm_data.db \
    --config sim_config.json --bw-profile bw_profile.json --ranktable ranktable.json -v

# One-click task graph dump (repository root directory)
python3 dump_graph.py data/hccl_vm_data.db --html

# hccl_sim compatible entry (JSON task file)
python -m simulator --tasks tasks.json --config sim_config.json \
    --bw-profile bw_profile.json --ranktable ranktable.json -v --trace trace.json
```

`--config` overrides the default latency/bandwidth for each mode (units us/GBps); `--bw-profile` independently loads
the utilization configuration; reports and traces are generated by `generate_report` / `export_trace` in `report.py`
from the `run()` result dict.

The trace file is in Chrome Trace Event format: `ts`/`dur` values are in **us** (format specification convention);
the file top level carries `"displayTimeUnit": "ns"` to specify the viewer's (chrome://tracing and so on)
Start/Wall Duration **display** unit as ns (default ms).

`export_trace` display adaptation conventions:

- **tid sequential numbering**: Thread rows are mapped to tid `0,1,2,...` in ascending stream id order (when original
  streamId 2/6..13 is used directly as tid, viewers that sort by string would place 10..13 before 2);
  the real stream id is retained in the thread_name label and `args.stream_id`.
- **Endpoint rounding**: `ts`/`end` are each rounded independently before subtraction to get `dur` -- independently rounding `ts` and `dur`
  can produce 0.001us spurious overlap; overlapping slices on the same tid are nested as sub-rows by the viewer (displayed on new lines).
- **Virtual stream annotation** (Section 7.4.1): The db path passes stream metadata via `export_trace(..., stream_meta=...)`;
  virtual stream thread_name is changed to `vir_stream {n} ({label})` (n numbered from 0 in ascending sid order, that is, allocation order,
  Eliminate the large stream numbers starting from 2^20 base; the real stream number is in
  the task `args.stream_id`); its task args append `virtual_stream: true`;
  real/ccu stream names are unchanged (the logical name of the CCU queue chain only appears in the text report line). When stream_meta
  is absent (simulator standalone CLI old input), no annotation is made, for backward compatibility.
- **Notify task args carry mask**: CCU path `Notify_Record`/`Notify_Wait`
  task args append `mask: "0x..."` (notify_id + mask uniquely identify one notify action,
  Section 7.3); SQE path and mask=0 do not output this field.
- **Operator summary first row (op_summary)**: When the pipeline calls
  `export_trace(..., op_summary=...)` per operator, each operator (when multiple operators are merged for export, each
  pid process group) reserves the `op_summary` first row with tid 1 + `thread_sort_index` 0; task row
  tid and sort_index are deferred (presentation layer only; tid avoids 0 -- some viewers such as Perfetto treat
  tid 0 as the idle thread, `thread_sort_index` is the Chrome Trace standard row order control);
  the summary event is placed at the front of the event array; triple guarantee ensures op_summary is always the first row. The first row contains one
  `X` summary slice spanning the operator's start to end (start time and duration are the slice's own ts/dur, taking
  the earliest task start to the latest end), row label   `"<op_name> | <algorithm_name>"`,
  args carry: `algorithm_name` (algorithm selection result, opDetails.algorithmName log
  interception backfill (DFX primary path, Section 4.2 note)), `data_type`/`data_count` (element count, V2 layout operators have
  send/receive side element counts and types, others have count and single-side type), `task_count` (total task count after graph generation,
  for CCU mode this is the count after microcode instruction translation to the task graph), and op_type/
  comm_id/op_iter/rank/rank_size/input_bytes/output_bytes (opMemInfo
  inputSize/outputSize, computed from workload API parameters during collection) context; summary events are not counted toward
  `max_events`.

### 7.8 Alignment Adaptation Completion Status

Differences between the old version (before modification in this repository) and the HCCLSIM baseline have been resolved in this alignment:

| # | Item | Handling |
|---|----|------|
| 1 | Task model 6 types → new 6 types (local/remote record/wait merged into Notify_*, LOCAL_COPY split into Memcpy/Reduce_Inline, network transport changed to Ub_*; subsequently expanded to 7 types with the new db WRITE_WITH_NOTIFY, Section 10.3 #19) | `task_model.py` replaced per HCCLSIM |
| 2 | stream_id str → **int**; notify_id str → **int** (INVALID=0xFFFFFFFF) | `dependency_graph.py`/graph generation side synchronized |
| 3 | Port specified by `port` int field → **eid real-time resolution** (write queries src_eid / read queries dst_eid) | `topo_resolver.py` + graph generation side fills eid |
| 4 | Unit system: JSON latency s → **us**, bandwidth B/s → **GBps**, port key int → **"die/port"** | `config.py` from_dict conversion |
| 5 | On-chip memory latency `size/bw` independent → **dynamic sharing** (Reduce_Inline+Memcpy share manager) | `_BandwidthManager` reuse |
| 6 | `run()` returns makespan → **result dict**; simulator two-param → **three-param** (+topo_resolver) | `simulator.py` |
| 7 | Latency formulas scattered → `task_timing.py` single point | Incorporated |

Unmatched residual items (including `src_rank==0` hardcoding, eid without fallback, broadcast pairing only triggering the first wait, and so on)
are tracked in Chapter 10.

---

## 8. Performance Profiling Output (Sub-Requirement 5, High-Level)

### 8.1 Output Location and Files

```
<installRoot>/plugin/light_performance/data/<timestamp>/
├── perf_profiling.json      # Main artifact: performance metrics + task graph summary
├── trace_<commId>_<opIter>.json   # (Optional) Chrome tracing, visualizable with chrome://tracing
└── light_performance.log    # Runtime log
```

- Directory naming: The `hccl-vm plugin run @estimate` one-shot execution channel does not pass data_id (Section 3.2),
  so the runtime timestamp is used instead; the standalone debug mode (run_perf_sim.py) uses the `-o` specified output path.
- Output is fixed in the plugin data directory rather than `data/`: The host clears `data/` at each start (`src/main.cc:32`);
  artifacts under the plugin directory persist across sessions; `archive.sh` archiving only covers logs/ and data/, so plugin artifacts
  are not included in automatic archiving (Section 10.3 #15).

### 8.2 Metric Definitions (Draft)

| Level | Metric | Source |
|------|------|------|
| Operator | opType/opIter/rankSize/dataSize, makespan, software simulation duration | opDetails + `sim.run()` result dict |
| Operator | Operator summary first row (algorithm_name/data_type/data_count, start/end and duration carried by slice ts/dur, Section 7.5 op_summary) | opDetails.algorithmName (DFX/log dual-path interception) + simulation task timeline |
| Operator | Total transfer volume, effective bandwidth utilization (total transfer volume / (makespan × port bandwidth)) | Summarize result["tasks"] |
| Stream | Per-stream start/end time, duration, task count + virtual stream annotation (Section 7.4.1) | result["tasks_by_stream"] + stream_meta |
| Graph | Asynchronous unit count, virtual stream count, fork edge count, completion set/attachment edge count (Section 7.4.1); local/remote post count (lp/rp), local wait paired count (wm), remote wait count (rw) | CcuBuildStats (parallel_units / virtual_streams / fork_edges / join_edges / local_posts / remote_posts / wait_matched / remote_waits, summary line `lp/rp/wm/rw`) |
| Port | Per-port peak concurrency, write/read transfer volume and average bandwidth | result["ports"] (NetworkPort) |
| Task | Per-task start/end/duration, transfer task average bandwidth | result["tasks"] (existing trace logic) |

### 8.3 perf_profiling.json Format Draft

```json
{
  "meta": {"data_id": "1234", "db": "...", "sim_config": {...}, "generated_at": "..."},
  "ops": [
    {
      "comm_id": 1, "op_iter": 0, "op_type": 7, "op_name": "REDUCE_SCATTER",
      "rank_size": 4, "data_size": 33554432,
      "makespan_us": 15230.5,
      "total_bytes": 522190080,
      "streams": [{"stream_id": 2, "label": "s2", "kind": "real", "start_us": 0.0, "end_us": 15230.5, "task_count": 30}],
      "ports": [{"ports": ["1/0"], "write_bytes": 272629760, "read_bytes": 0,
                 "peak_concurrency": 3, "avg_bw_Gbps": 143.2}],
      "graph": {"task_count": 40, "stream_count": 5, "notify_match_count": 44,
                "cross_rank_notify": 40,
                "virtual_stream_count": 10, "extra_edge_count": 20,
                "dump_files": ["graph_comm1_iter0_dev1.dot"],
                "trace_file": "trace_1_0.json"}
    }
  ]
}
```

> `cross_rank_notify` is the count of retained cross-rank notify nodes (diagnostic information): These nodes are graphed in normal stream order
> and use fixed remote delay, but do not participate in local pairing (no cross-rank record-wait edges); this count is used to detect
> data/classification anomalies (such as a local notify being misidentified as cross-rank). `kind`/`virtual_stream_count`/
> `extra_edge_count` are new fields from CCU queue parallelization (Section 7.4.1): stream classification real/ccu/virtual,
> virtual stream count, and extra edge (fork/join/attachment) count.

---

## 9. Appendix

### 9.1 Checker Graph Generation Module to light_performance Graph Generation Module Full Comparison

| Aspect | checker (C++) | light_performance (Python) |
|------|---------------|---------------------------|
| Goal | Correctness validation DAG (mainStart reachable, supports various checks) | Simulation DependencyGraph (three layers: tasks / stream_chains / notify_matches) |
| Input | db (via StorageManager/sim::Query) | Same db (sqlite3 direct read + struct decoding) |
| Graph generation scope | **Multi-rank joint graph generation** (all rank tasks in one DAG) | **Single-rank graph generation** (no cross-rank pairing edges, remote synchronization tasks retained) |
| SQE translation | TaskMetaTranslatorV3 (including MemSlice address resolution) | Section 5.2 mapping table (only takes len/jettyId/notify fields) |
| WRITE_WITH_NOTIFY consumption | Split at decoding boundary: `DecodeAndExpandTaskMeta` expands to [write, NOTIFY_RECORD] two sub-tasks | **Not split**: Single-task mapping `Write_With_Notify` (Section 5.2), one-to-one with db tasks, consistent trace/dump/statistics granularity |
| Dependency edges | Stream chain + local/cross-rank notify edges (queue simulation) | Stream chain + local-only notify first-come-first-served (two-pass scan) + extra edges (CCU fork/join/attachment, Section 7.4.1) |
| CCU | Two-phase expansion + queue parallelization + loop merge + Event record | Single-phase interpretation + assembly-phase queue parallelization (Section 7.4.1) + loop merge (Section 6.5) + Event record (Chapter 6) |
| Post-validation processing | Memory conflict/semantic/synchronization conflict/reachability/single-task checks | None |
| Output | Validation conclusions + dump | (SimConfig, DependencyGraph) → makespan → profiling json |

### 9.2 Glossary

| Term | Description |
|------|------|
| SQE | Submit Queue Element, the task unit submitted by the AICPU/SDMA side, that is, one row in the opTask table |
| WQE | Work Queue Element (UB/DB mode), not involved in this effort |
| notify | AICPU-side event synchronization primitive, record sets bit / wait waits, paired by (recordDev, waitDev, notifyId) |
| Event | CCU-side event synchronization register (bitmap), post sets bit / wait waits, paired by (waitDev, dieId, ckeId, mask) |
| jetty | Network card (communication engine) port number, determines memcpy bandwidth attribution |
| die | Chip die, CCU microcode space and Events are organized by (device, die) |
| mission | One CCU task delivery (instStartId/instCnt define the instruction window, args[13] pass parameters) |
| makespan | Task graph end-to-end duration (latest task end time) |
| Main stream / Slave stream | The operator's main stream marked by opDetails.streamId; the rest are slave streams (first WAIT, last RECORD structure) |

---

## 10. HCCLSIM Alignment Residual Items and Pending Match List

> This chapter records the **field gaps, task type mismatches, and other conflict points** confirmed during the alignment of the "db graph generation path + design document"
> with the HCCLSIM baseline. Current implementations have all been handled with "default fill invalid values / approximate mapping";
> this is documented here for future joint resolution with the collection side and simulation module.

### 10.1 Task Field Availability Table (db Graph Generation Path)

New model Task field sources and defaults in the db graph generation path (invalid value convention: int fields `INVALID=0xFFFFFFFF`,
eid empty string, other strings `""`):

| Task Field | db Source | Availability | Default/Approximate Value | Future Match Path |
|-----------|--------|--------|------------|-------------|
| `task_id` / `stream_id` | Global sequence / streamId+synthetic base | Available | -- | -- |
| `notify_id` | opTaskMeta `NotifyTask.notifyId` | Available | -- | -- |
| `size` | MEM_CPY `len` / REDUCE `dataCount` / CCU length register | Available | -- | -- |
| `jetty_id` | SQE table column / ccuChannels.jettyId array | Available | INVALID (non-network tasks) | -- |
| `link_type` | opTaskMeta `HcclTaskProfInfo.linkType` (v3, enum name mapping Section 11.4.2) | Available (v3) | "" (displays `Unknown` for v2 old BLOBs without profInfo) | -- (Section 11.4 consumed) |
| `protocol` | ccuChannels.protocol (CommProtocol enum name mapping) | Available | "" (when no channel information, trace displays `N/A`) | -- (migrated from `link_type`, Section 11.1 approach a) |
| `src_eid` (local, write) | SQE `lcEid` (new in v2 collection, Section 4.4.1) preferred; defaults to jettyId → local device outbound channel `leid` derivation | Available (v2) / approximate (fallback) | Empty → any local channel leid fallback | -- (v2 already supplemented with explicit field) |
| `dst_eid` (peer) | Channel `reid` preferred, falls back to SQE `rmEid` | Approximate | "" (tasks without channel information) | Collection side supplements with explicit peer eid |
| `src_rank` / `dst_rank` | taskData endpoint rankId (carried by deviceId field, Section 4.4.1 rank field rollback) directly consumed: write=src local/dst peer, read=src peer/dst local; CCU/unresolved falls back via eid to rank index (Section 4.8) | Available / index fallback | INVALID + `rank_unresolved` count when both explicit value and index fail to resolve | -- (Sections 5.2/5.5/6.7) |
| `transport_type` | opTaskMeta `HcclTaskProfInfo.transportType` (v3, enum name mapping Section 11.4.3) | Available (v3) | "" (displays `Unknown` for v2 old BLOBs without profInfo) | -- (Section 11.4 consumed) |
| `data_type` | REDUCE task-level `dataType` (new in v2, enum name mapping int8..bfp16); MEM_CPY task-level has no source | Partially available (v2) | REDUCE fills enum name; MEM_CPY "" | Collection side supplements MEM_CPY |
| `port` | (reserved field) | Not filled | None | Simulator resolves ports in real-time via eid (Section 7.6) |

**Instruction-level gaps (CCU microcode expansion)**: Peer rank for channel-type network tasks is already resolved by the eid to rank index
(Sections 4.8/6.7); reduction tasks are always local (CcuBuffer is a rank-local buffer, Section 6.3.2), with no peer eid
requirement. All UB tasks produced by CCU explicitly carry `is_write` (Section 6.7); direction determination does not depend on rank
values.

### 10.2 Task Type Mismatch Table

| db taskType / Old Type | New Model TaskType | Mismatch Handling |
|---------------------|----------------|-----------|
| SYNC_STREAM (taskType=6) | **No corresponding type** | Skip, do not generate node (host-side stream synchronization point, not a device-side task, Section 5.6); count `skipped_sync` |
| AIV_GRAPH (taskType=5) | None | Skip + warning (consistent between old and new models, both unsupported) |
| WRITE_WITH_NOTIFY (taskType=7, new in db) | `WRITE_WITH_NOTIFY` (new, 7th type) | Single-task mapping (**not split** into write + NOTIFY_RECORD): write always in write direction using port bandwidth sharing, attached notify at fixed record delay (Section 5.2, Section 10.3 #19 supported) |
| Old `REMOTE_RECORD` / `REMOTE_WAIT` | Already merged into `NOTIFY_RECORD` / `NOTIFY_WAIT` (src≠dst expresses remote) | Cross-rank notify in the db single-rank path generates remote nodes with `src≠dst` (Section 5.2, nodes retained without pairing) |
| Old `MEMCPY` (network transfer) | Semantics changed: new `Memcpy` = on-chip memory copy | Network transfer is now expressed by `Ub_Write_Or_Read` (write/read direction distinguished); `Ub_Inline_Write` only retains notify inline write semantics (fixed delay) |
| Old `LOCAL_COPY` | Split into `Memcpy` (pure copy) / `Reduce_Inline` (reduction) | REDUCE non-network → Reduce_Inline |
| Old `MemcpyDirection` enum (direction field) | Removed | Direction expressed by task type + rank/eid values (Section 5.5) |
| Old `sync_stream_delay` configuration item | Removed along with the type gap | Reuses local_record_delay |

### 10.3 Other Conflict Points (Future Matching)

1. ~~**`src_rank == 0` hardcoded as local** (`simulator.py` TRANSFER branch): If a read-direction task has
   the peer as rank 0 and explicitly filled, it would be misidentified as write.~~ **Fixed**: Task adds explicit
   `is_write` field (filled in the db graph generation path, None for the task.txt old path which falls back to heuristic); the simulator
   prioritizes consuming it; the peer rankId from taskData (carried by deviceId field) is also directly filled into src/dst_rank (Sections 5.2/5.5).
2. **TopoResolver has no fallback**: Throws directly when eid is not found in ranktable (no jettyId-to-port mapping,
   no default port). The db path avoids this by "skip simulation and only dump when network tasks exist but no ranktable is available"
   (Section 3.3); future work could consider building a fallback mapping from ccuChannels/JettyMaps.
3. **Broadcast pairing only triggers the first wait** (`break` on matched record): When multiple waits pair with the same
   record, subsequent waits' Events are not triggered, end_time remains -1 (tasks silently lost).
4. **`is_matched_record` linear scan**: `dependency_graph.py` performs O(R×M) traversal on matches;
   large-scale databases (tens of thousands of tasks) have performance risk; recommend maintaining a record-to-wait reverse index (also
   resolves #3).
5. **SimConfig unit system and port key changes**: JSON latency s→**us**, bandwidth B/s→**GBps**,
   port key int→**"die/port"** string; old configuration files are incompatible (sample
   `src/plugin/light_performance/sim_config.json` already synchronized in this effort).
6. **On-chip memory dynamic sharing semantics change**: Old version was per-task independent `size / hbm_bandwidth`; new version
   has Reduce_Inline+Memcpy concurrent tasks equally sharing on-chip memory bandwidth (manager with utilization always 1.0); result caliber
   differs.
7. **Utilization query basis `min_remaining`**: Concurrent tasks in the same group query the utilization tier using the minimum remaining data volume,
   different from the old version "each task queries by its own data_size" -- small tasks in the group will drag down the utilization tier of large tasks.
8. **Old input sources removed**: task.txt parsing, AllToAll generator, built-in demo have been removed from input_builder with the HCCLSIM version;
   if log replay debugging is needed, they must be restored separately.
9. **TopoResolver has no net_type filtering**: No priority rule when the same eid has multiple network type entries.
10. **Profiling JSON not implemented** (Sub-Requirement 5): Current output is task graph dump (DOT/JSON/HTML)
    + text report + Chrome trace; the perf_profiling.json format is described in the Chapter 8 draft.
11. ~~**local_rank fallback convention**: When opDetails.rankId is missing, fall back by `deviceId−1`
    (following the task.txt path convention); deviceId and rank numbering have no inherent relationship; ranktable can
    provide precision.~~ Peer rank is already resolved by the eid to rank index (RankEidMap primary path + ranktable
    fallback, Section 4.8), no longer depending on numbering conventions; local rank still uses opDetails.rankId as the authority
    (channel leid cross-validation, Section 6.7).
12. ~~**CCU queue parallelism not modeled**: CCU queue chains execute as independent synthetic streams in parallel; real hardware
    queue concurrent resource constraints are not modeled (checker side has queue parallelization transformation; this plugin simplified per Section 6.7).~~
    **Queue parallelization implemented** (Section 7.4.1: assembly-phase skeleton/unit splitting + fork/join/attachment extra edges +
    `#v` virtual streams reused across missions). Still not modeled: Real hardware queue concurrent resource upper limit --
    virtual stream count is unlimited (checker V3 is also unlimited); future work can cap by chip generation.
13. **@estimate one-shot channel does not pass data_id**: The `plugin run @estimate` special-case branch directly executes
    the entry (`cmd_base_utils.cc:626-663`) without going through the generic message path's `DumpData` (which generates
    `data_id`, currently a fixed value "1234"), so the output directory is named by the runtime timestamp (Section 8.1); if
    the resident model is adopted later, data_id semantics can be aligned.
14. **`plugin install @estimate` semantic degradation**: install goes through `RegisterPlugin` resident startup
    (fork+pipe); main.py finishes one-shot execution and exits upon stdin EOF, MonitorThread reclaims --
    behavior is harmless but inconsistent with the resident semantics of install; a true resident implementation would require a stdin command loop
    (Section 3.2 evolution direction).
15. **Plugin artifacts not included in automatic archiving**: `archive.sh` only archives `logs/` and `data/`;
    perf_profiling.json and other artifacts under `plugin/light_performance/data/` do not enter the archive directory
    (the benefit is that they are not affected by `remove_all("data")` at start); archiving requires extending archive.sh.
16. **simpy deployment dependency**: The target environment must have simpy pre-installed (not distributed with hccl_vm_install as an offline wheel);
    main.py should provide a clear error message and installation instructions when import fails at startup (Section 3.4).
17. ~~**db_reader.py not synchronized with v3 collection layout**: The C side `HcclTaskMetaData` has been extended to 188B (after taskType,
    a new `HcclTaskProfInfo` 3-byte field was added, Section 4.4.1); `db_reader.py` still decodes as 185B
    (`TASK_META_LEN=185`, `<BQIQQI16s16s`) -- the length gate `len(blob) >= 185` cannot catch
    188B BLOBs; reading databases collected with v3 would cause **entire-field misalignment** (from commId onward). Needs synchronization to
    188B/`<BBBBQIQQI16s16s` and decoding of profInfo three fields (value ranges/names per Section 11.4).~~
    **Synchronized**: `db_reader.py` auto-identifies v3 (188B/`<BBBBQIQQI16s16s`,
    profInfo three fields into `HcclTaskMetaData`) and old v2 (185B) layouts by BLOB length; see Section 4.4.1 implementation status.
18. ~~**JSON task type does not consume profInfo** (residual narrowed): trace args
    `transport_type`/`link_type` have switched to profInfo-driven, CommProtocol migrated to the new field
    `protocol` (Sections 11.4.2/11.4.3, Section 11.1 approach a implemented); **residual** is the `task_type`
    field -- currently still derived from TaskType enum + jettyId/direction conditions (Section 11.2), which has **semantic misalignment** with the collection-side
    profInfo hcomm op name (for example, `Ub_Inline_Write` bidirectionally refers to different
    tasks, Section 11.4.4); switching to `profInfo.profTaskType`-driven would enable one-to-one correspondence with msprof timeline
    op names; the target state comparison (including each value's correspondence with simulation TaskType) is in Section 11.4.1.~~
    **Implemented (all residuals closed)**: `Task` adds `task_name` display name (`task_model.py`,
    `display_name` = `task_name or task_type.value`), `graph_builder._add_task`
    uniformly injects `prof_task_type_name(profInfo.profTaskType)`; trace event name/`cat`
    (`report.py export_trace`), `args.task_type` (`to_trace_args`), verbose
    detail and dump (DOT/JSON/HTML legend) all switched to op name. Internal `TaskType` is retired as
    a latency model classification (Section 11.4.5 #3); latency/pairing/billing have zero changes. v2 old BLOBs
    (profInfo=0xFF) display `Unknown` (same caliber as link/transport); CCU microcode sub-tasks
    have no profInfo, falling back to internal names (Section 11.3). Misalignment elimination comparison is in Section 11.4.4.
19. ~~**WRITE_WITH_NOTIFY (taskType=7) decoding missing**: The `HccLTaskMetaType` IntEnum in `db_reader.py`
    only covers 0~6; `decode_op_task_meta` encountering 7 directly raises and discards
    (counted as warning).~~ **Supported (single-task mapping)**: `db_reader.py` decodes
    `WriteWithNotifyTask` (Section 4.4.1) and transparently passes as a single task; `graph_builder` maps to
    `WRITE_WITH_NOTIFY` TaskType (trace display name `Write_With_Notify`, Section 5.2),
    **one-to-one** with db tasks; latency = port write direction bandwidth sharing + post-write notify
    fixed record delay (Sections 7.3/7.4). **Split strategy not adopted** (the checker/runner side's
    "split into [write, NOTIFY_RECORD] two sub-tasks" and this plugin's early decoding-boundary split have both been
    deprecated): Splitting would fragment task granularity, making trace/dump/statistics unable to correspond one-to-one with db tasks.
    Collection-side profInfo=TASK_WRITE_(REDUCE_)WITH_NOTIFY (op name `Write_With_Notify` /
    `Reduce_With_Notify`; the hcomm source code has the latter misspelled as `Reuce_With_Notify`, see Section 11.4.5);
    both profInfo values map to the same `Write_With_Notify` task (isReduce only affects
    data_type display).
20. ~~**Multiple dispatches of the same mission SQEs merged into a single context**: CCU expansion merged all SQEs of the same
    `(dieId, missionId, queue)` into one dispatch context (single instruction window
    [first instStartId, last), single SQE args consumption) -- does not satisfy the "contiguous-only merge" constraint of `IsContiguousCcuSqe` in checker
    (`task_meta_translator_v3.cc:213-231`).
    Typical affected scenario: Instruction window wraparound re-delivery (head SQE 13 Load SqeArgs + body SQE delivered twice in succession,
    the second time with new args); under the old implementation, the second dispatch: ① window start counted by head SQE, tail
    JMP skips the entire data transport instruction segment; ② SQE args already consumed by the first dispatch, subsequent Loads get 0 values
    (Address addresses all 0); ③ Loop counter/Event residual from first dispatch state -- the second dispatch's
    Ub/Memcpy/local post are entirely lost, network data volume is only 3/10 of checker's.~~
    **Fixed (contiguous segment splitting)**: Split into multiple dispatch contexts by `instStartId`/`instCnt` continuity
    (Section 6.1); each context has independent window replay (including Load SqeArgs header) + independent sequential
    args consumption (`GetSqe` semantics); register/Event bitmaps shared across contexts (`AllRankParamRecorder`
    semantics); same-queue contexts serial in dispatch order; multi-segment task ids disambiguated with `_s{sequence number}` suffix
    (`ccu_graph_builder.expand_ccu_graph`). After fix, stream-by-stream alignment with checker single-pass graph:
    record local/remote 54/54, wait local/remote 27/36, Ub 39/39, Memcpy 6/6,
    network+local data volume 1088MB (2333 validation environment ALLGATHER dev1).
    A **wait node launch deviation** with the same root cause was also fixed: The old implementation split by "one wait node per contributing post"
    (one wait instruction could produce N nodes, inconsistent with checker `GenWaitNode`'s "at most 1 local + 1 remote node per instruction";
    in this environment, manifested as local wait 48 vs 27); now aggregated into a single node
    (mask = OR of all contributing post bits, first contributing post uses notify pairing, rest use extra edges,
    Section 6.4); after fix, local wait 27/27 aligned.
21. **Merged loop folded pseudo-serialization residual gaps**: Section 6.5.1 characteristic chain resolution covers the merge path's
    mid-body wait/record (folded pseudo-serialization barrier), but two residuals remain: ① The full expansion fallback path
    (`mf > 0`) has per-iteration wait/record as real 4KB granularity handshakes, whose inter-iteration overlap is also
    serialized by the queue chain (large node count, not triggered by current samples, not handled at this time); ② Per-iteration handshake
    **rate** overhead is not modeled -- folding and resolution both count as single occurrences (if Event engine throughput is the bottleneck, a dedicated
    rate model is needed, such as ns/occurrence × 32768 occurrences order-of-magnitude estimation).

---

## 11. db Task Type and JSON Product Task Type Mapping Table

> This chapter is a type mapping quick reference table, answering three questions:
> ① Which task types in the JSON product do the tasks in the db database (`opTask_P_<pid>` table taskType column) ultimately map to (Sections 11.1 and 11.2);
> ② The correspondence between CCU mode microcode instructions and JSON task types (Section 11.3);
> ③ The correspondence between collection-side profiling reconciliation information (`HcclTaskProfInfo`) and JSON
> display types, and the target state (Section 11.4). Implementation references: `src/plugin/light_performance/
> graph_builder.py` (SQE mapping, `_map_tasks`/`_map_transfer`),
> `src/plugin/light_performance/ccu_graph_builder.py` (CCU instruction mapping, `transform`
> series), `src/plugin/light_performance/simulator/task_model.py` (TaskType enum).
> Same source as Section 5.2 (SQE mapping + latency model) and Section 6.3 (instruction category mapping); reorganized here by
> the "db → JSON" correspondence relationship, with reverse index supplemented.
>
> **profInfo background**: From v3 collection onward, db tasks carry `HcclTaskProfInfo` in the opTaskMeta BLOB
> (Section 4.4.1, single-source definition in `include/sim_profiling_defs.h`, treated as an opaque whole by checker/runner);
> the three fields `profTaskType`/`linkType`/`transportType` have value ranges and
> names aligned one-to-one with hcomm `TaskParamType`/`DfxLinkType`/`DfxTransportType` -- that is,
> the msprof timeline op name and L1 detail linkType/transportType fields. Sections 11.1~11.3 in this chapter describe
> **the current TaskType implementation** (not yet consuming profInfo); Section 11.4 provides the complete comparison of the three fields and the switch
> target state (pending items per Section 10.3 #17~#18).

### 11.1 Task Type Carriers in JSON Products

The current implementation has three types of JSON products, all of which derive task types from the `TaskType` enum values
(string form, `task_model.py:20-26`):

| JSON Product | Format | Task Type Field | Generating Module |
|-----------|------|----------------|---------|
| `perf_profiling.json` (main artifact, Section 8.1) | Chrome Trace Event | `traceEvents[].name` / `.cat` / `.args.task_type` | `simulator/report.py` (`export_trace`, event name is the task display name: profInfo op name, Section 11.4) |
| `trace_comm<commId>_iter<opIter>.json` (optional per-operator trace) | Chrome Trace Event | Same as above | Same as above |
| Task graph dump `graph_comm<commId>_iter<opIter>_dev<dev>.json` (optional, `--dump-dir`) | `light_perf.graph_dump/2` | `nodes[].type` | `graph_dumper.py` |

> The dump JSON notify nodes carry `nodes[].mask` when the mask is non-zero (Task.mask from Section 7.3);
> the HTML view tooltip displays it synchronously; the `perf_profiling.json`/trace task args also contain
> `mask` (Section 7.7).

The complete set of task types appearing in JSON (7 types):

| TaskType Value (JSON Presentation) | Latency Model | Semantics |
|--------------------------|---------|------|
| `Notify_Record` | Fixed latency (local also triggers Event) | Notify assertion (record / Event post) |
| `Notify_Wait` | Wait for Event / fixed latency | Notify wait (wait / Event wait) |
| `Reduce_Inline` | On-chip memory dynamic sharing | Local reduction (computation approximated as transport, Section 5.7) |
| `Memcpy` | On-chip memory dynamic sharing | Local copy (on-chip memory, not network) |
| `Ub_Inline_Write` | Fixed latency (same as Notify_Record) | Notify inline write (old input path only) |
| `Ub_Write_Or_Read` | Port bandwidth sharing (write/read directions charged independently) | Network data send/receive (SQE network tasks in both directions uniformly mapped) |
| `Write_With_Notify` | Port write direction bandwidth sharing + post-write notify fixed latency | Network write + attached notify (one-to-one with db `WRITE_WITH_NOTIFY`, Section 5.2) |

> The `perf_profiling.json` structured format (meta/ops array) in Section 8.3 is a draft; the current main artifact
> is actually in Chrome tracing format (Section 10.3 #10); regardless of format, the task types are the above
> 7 TaskType values, and the mapping tables in this chapter apply uniformly to all of them.

**Type-related fields in trace args (current state and profInfo consumption)**: The `to_trace_args` method of Task
(`task_model.py`) outputs four type-related fields; all four fields have switched to profInfo-driven
(Section 10.3 #18 fully implemented, Section 11.4):

| trace args Field | Current Source | Status |
|----------------|---------|--------------------------|
| `task_type` | `profInfo.profTaskType` op name (`prof_task_type_name`, uniformly populated by `graph_builder._add_task` as `Task.task_name`; `display_name` falls back to internal TaskType name when empty) | **Implemented** (one-to-one correspondence with msprof timeline op name, Section 11.4.1; `Unknown` for v2 old BLOBs) |
| `transport_type` | `profInfo.transportType` name (`ProfTransportTypeName`, Section 11.4.3, uniformly populated by `graph_builder._add_task`) | **Implemented** |
| `link_type` | `profInfo.linkType` name (`ProfLinkTypeName`, Section 11.4.2, uniformly populated by `graph_builder._add_task`) -- **Note: this is a different concept from `protocol` (CommProtocol)** | **Implemented** (approach a: overwritten with DfxLinkType name) |
| `protocol` (new field) | Channel `CommProtocol` enum name (HCCS/TCP/ROCE/UB_CTP/UB_TP/PCIE/SIO, populated by `graph_builder._resolve_eids` / CCU `_channel_eids`) | **Implemented** (CommProtocol migrated from `link_type`; `N/A` when no channel information) |

> The `link_type` overwrite (approach a) is implemented: semantics aligned with the msprof L1 detail field name, avoiding same-name-different-meaning;
> existing CommProtocol information is not lost, moved to the `protocol` field. CCU microcode sub-tasks are simulator
> semantic products and have no profInfo (Section 11.3); their `link_type`/`transport_type` are `N/A`,
> while `protocol` retains the channel CommProtocol name.

### 11.2 SQE Path: db taskType → JSON Task Type Mapping Table

Forward mapping (db task → JSON task display name). The "JSON task type" column is the op name displayed in trace/dump
(switched to `profInfo.profTaskType` driven, Section 10.3 #18 implemented); the "internal TaskType"
(latency model classification, determining charging/pairing) retains the original Section 11.2 conditions (jettyId/direction/recordDev),
with the correspondence to op names per the "Simulation TaskType" column in Section 11.4.1:

| db taskType (opTask.taskType column) | Condition | JSON Task Type (op name) | Collection-side profInfo.profTaskType (op name) | Description |
|----------------------------------|---------|--------------|--------------------------------------|------|
| `NOTIFY_WAIT` (0) | recordDev == local device | `Notify_Wait` | `TASK_NOTIFY_WAIT` → `Notify_Wait` ✓ | Local wait: Participates in first-come-first-served pairing (Section 5.4), waits for Event |
| `NOTIFY_WAIT` (0) | recordDev != local device | `Notify_Wait` | `TASK_NOTIFY_WAIT` → `Notify_Wait` ✓ | Cross-rank wait: Node retained, not paired, src≠dst → fixed `remote_wait_delay` |
| `NOTIFY_RECORD` (1) | waitDev == local device (SDMA/local notify) | `Notify_Record` | `TASK_NOTIFY_RECORD` → `Notify_Record` ✓ | Local record: Participates in pairing, triggers Event |
| `NOTIFY_RECORD` (1) | waitDev != local device (SDMA/local notify) | `Notify_Record` | `TASK_NOTIFY_RECORD` → `Notify_Record` ✓ | Cross-rank post: Node retained, not paired, src≠dst → fixed `remote_record_delay` |
| `NOTIFY_RECORD` (1) | UBDMA WQE `inlineEn==1` (notify inline write) | `Ub_Inline_Write` | `TASK_UB_INLINE_WRITE` → `Ub_Inline_Write` ✓ | Notify inline write (name restored, Section 11.4.4); internal TaskType is still Notify_Record (fixed record delay) |
| `REDUCE` (2) | jettyId == INVALID | `Reduce_Inline` | `TASK_REDUCE_INLINE` → `Reduce_Inline` ✓ | Local reduction: On-chip memory dynamic sharing charging |
| `REDUCE` (2) | jettyId valid + write direction | `Ub_Reduce_Inline` | `TASK_UB_REDUCE_INLINE` → `Ub_Reduce_Inline` ✓ | Network reduction approximation (Section 5.7), internal TaskType=Ub_Inline_Write, port write direction charging |
| `REDUCE` (2) | jettyId valid + read direction | `Ub_Reduce_Inline` | `TASK_UB_REDUCE_INLINE` → `Ub_Reduce_Inline` ✓ | Same as above, internal TaskType=Ub_Write_Or_Read, read direction charging |
| `MEM_CPY` (3) | jettyId == INVALID (same device) | `Memcpy` | `TASK_SDMA` → `Memcpy` ✓ | Local copy: On-chip memory dynamic sharing charging; profInfo.linkType=ONCHIP |
| `MEM_CPY` (3) | jettyId == INVALID (cross-device SDMA SQE) | `Memcpy` | `TASK_SDMA` → `Memcpy` ✓ | Cross-die SDMA transport; also Memcpy, difference expressed by profInfo.linkType=PCIE / transportType=SDMA (Section 11.4) |
| `MEM_CPY` (3) | jettyId valid + write direction | `Ub_Write_Or_Read` | `TASK_UB` → `Ub_Write_Or_Read` ✓ | Network send (direction not encoded in type name, expressed by direction/is_write); internal TaskType=Ub_Inline_Write |
| `MEM_CPY` (3) | jettyId valid + read direction | `Ub_Write_Or_Read` | `TASK_UB` → `Ub_Write_Or_Read` ✓ | Network receive; internal TaskType=Ub_Write_Or_Read |
| `WRITE_WITH_NOTIFY` (7) | Always write direction (no direction inference, Section 5.5) | `Write_With_Notify` (single task, not split); `Reduce_With_Notify` when `udfFlag==1` | `TASK_WRITE_WITH_NOTIFY` → `Write_With_Notify`; `TASK_WRITE_REDUCE_WITH_NOTIFY` → `Reduce_With_Notify` when `udfFlag==1` (hcomm source code has the misspelling `Reuce_With_Notify`, Section 11.4.5) | Write+notify composite task (`WriteWithNotifyTask`, Section 4.4.1): Port write direction bandwidth sharing + post-write notify fixed record delay; both profInfo values map to the same internal WRITE_WITH_NOTIFY (latency model is consistent, display name and isReduce distinguish) |
| `CCU_GRAPH` (4) | -- | (Placeholder, does not directly produce output) | `TASK_CCU` → `Ccu` | Expanded into CCU subgraph; final products per Section 11.3 |
| `AIV_GRAPH` (5) | -- | None (skipped) | `TASK_AIV` → `AivKernel` | Warning + `skipped_aiv` count (not supported at this time) |
| `SYNC_STREAM` (6) | -- | None (skipped) | `UNKNOWN` → `Unknown` | Host-side stream synchronization point, not a device-side task (Section 5.6); no corresponding type in hcomm profiling |

> Cross-rank determination (waitDev/recordDev) per Section 4.6; network direction determination (write/read) per Section 5.5.

Reverse index (JSON task display name ← all sources, including CCU path and profInfo dimension; SQE path
display name = profInfo op name, CCU path display name = internal TaskType name, Section 11.3):

| JSON Task Display Name | SQE Source (db taskType) | Collection-side profInfo Source (op name) | CCU Source (microcode instruction, Section 11.3) |
|--------------|------------------------|------------------------------|------------------------------|
| `Notify_Record` | `NOTIFY_RECORD` (1), local/cross-rank (SDMA/local notify) | `Notify_Record` (TASK_NOTIFY_RECORD ✓) | SetCKE/SetCKBit local post; TRANS/REDUCE/LOAD instruction embedded `setCKE` field (A6 LOAD type also carries it); cross-rank assertion/notification (A5 Sync family setRmtCKE; A6 TransMem notify/0x1A atomic, SyncWtX notify, SyncAtx -- remote approximation); subgraph attachment launch-record / completion-record |
| `Notify_Wait` | `NOTIFY_WAIT` (0), local/cross-rank | `Notify_Wait` ✓ | Embedded `waitCKE` field in various instructions; ClearCKE; A6 independent Wait instruction (Variable value comparison, unsatisfied → remote wait); subgraph attachment completion-wait / launch-wait |
| `Reduce_Inline` | `REDUCE` (2), jettyId invalid | `Reduce_Inline` ✓ | REDUCE type instructions (msId list all local) |
| `Memcpy` | `MEM_CPY` (3), jettyId invalid (including cross-die SDMA) | `Memcpy` ✓ | TRANS local transport instructions (A5 4 codes: 0x0/0x2/0x5/0xA; A6 4 codes: 0x0/0x2/0x5/0x6) |
| `Ub_Inline_Write` | `NOTIFY_RECORD` (1) + `inlineEn==1` (notify inline write) | `Ub_Inline_Write` (TASK_UB_INLINE_WRITE ✓, name restored) | -- (CCU network transport uniformly maps to `Ub_Write_Or_Read`, Section 11.3.1) |
| `Ub_Write_Or_Read` | `MEM_CPY` (3) / `REDUCE` (2), jettyId valid (write/read direction) | `Ub_Write_Or_Read` (TASK_UB ✓, direction expressed by direction/is_write) | TRANS network transport instructions (A5 write role 3 codes: 0x3/0x7/0x9, read role 4 codes: 0x1/0x4/0x6/0x8; A6 unified TransMem 0x10: dmaOpCode 3/5=write, 6=read, including udfEnable network reduction approximation); REDUCE type with remote CcuBuffer |
| `Ub_Reduce_Inline` | `REDUCE` (2), jettyId valid (network reduction) | `Ub_Reduce_Inline` (TASK_UB_REDUCE_INLINE ✓) | -- |
| `Write_With_Notify` | `WRITE_WITH_NOTIFY` (7), isReduce=0 | `Write_With_Notify` (TASK_WRITE_WITH_NOTIFY ✓) | -- |
| `Reduce_With_Notify` | `WRITE_WITH_NOTIFY` (7), udfFlag==1 | `Reduce_With_Notify` (TASK_WRITE_REDUCE_WITH_NOTIFY ✓) | -- |

### 11.3 CCU Path: Microcode Instruction → JSON Task Type Mapping

The `CCU_GRAPH` (4) SQEs in the db are only placeholders (`CcuPlaceholder`, Section 6.1); the final
JSON tasks are produced by microcode interpretation from the `ccuInstrRes` instruction space (Chapter 6). **Both A5 (V1,
Ascend950, devType=6) and A6 (V2, Ascend960, devType=8) instruction set generations are implemented**
(selected by `opDetails.devType`; other devType values fall back to A5): A5 full table in Section 11.3.1,
A6 full table in Section 11.3.3.

> profInfo correspondence: The CCU placeholder SQE's `HcclTaskProfInfo` as a whole is
> `TASK_CCU`→`Ccu` (linkType/transportType are UNKNOWN; microcode has no link dimension). The expanded
> microcode sub-tasks (`Memcpy`/`Ub_Write_Or_Read`/`Notify_*`, and so on) are
> simulator semantic products with **no corresponding ProfTaskParamType**; the display name falls back to the internal TaskType name
> (established trade-off during Section 10.3 #18 implementation, Section 11.4.4); the msprof side expresses CCU tasks via
> microcode-level detail (`CcuProfilingInfo`: CCU_TASK/WAITCKE/LOOPGROUP three categories, reported with TASK_CCU)
> to express equivalent granularity information; reconciliation uses the whole SQE's `Ccu` as the alignment anchor.

#### 11.3.1 A5 Instruction Full Table (Instruction → JSON Task Type)

Microcode instruction = 2B header (type:4bit category + code:11bit instruction code, Section 6.2) + 30B payload:

| Instruction Category (type) | Instruction (code) | JSON Task Type | Description |
|----------------|------------|--------------|------|
| LOAD (0x0) | LoadSqeArgsToGSA (0x0), LoadSqeArgsToXn (0x1), LoadImdToGSA (0x2), LoadImdToXn (0x3), LoadGsaXn (0x4), LoadGsaGsa (0x5), LoadXx (0x6) | **None** (does not generate nodes) | Only reads/writes the register model (Variable/Address), providing length, loop counter, and so on for subsequent instructions |
| CTRL (0x1) | SetCKE (0x2) | `Notify_Record` | Local Event post (ckeId as notify_id, ckeMask as mask, src=dst=local); embedded `waitCKE` field can additionally produce `Notify_Wait` |
| CTRL (0x1) | ClearCKE (0x4) | `Notify_Wait` | wait semantics (embedded `waitCKE` field); the clear action itself does not produce a node |
| CTRL (0x1) | Loop (0x0) | **None** | Must be triggered by LoopGroup; standalone occurrence is abnormal, skip with warning |
| CTRL (0x1) | LoopGroup (0x1) | (No direct nodes) | Loop control: Merge/expand loop body (Section 6.5); body instructions mapped per this table; under the merge path, data task size × total iterations, and launched tasks carry loop_cnt (loop count after expansion) / loop_iter_cnt (self-loop count per single loop); total iterations = product of both fields |
| CTRL (0x1) | Jmp (0x5) | **None** | Conditional jump: `Xn[conditionXnId] != expectData` causes pc to jump to the instruction pointed by `Xn[dstInstrXnId]` (`TransformJmpInstrA5` semantics in checker, target limited to current instruction window, out-of-range warns and passes through); within loop body passes through (merge precondition validation already rejected loop bodies containing JMP); step budget defense against backward jump infinite loops |
| TRANS (0x2) | LocMemToLocMs (0x0), LocMsToLocMem (0x2), LocMsToLocMs (0x5), LocMemToLocMem (0xA) | `Memcpy` (src/dst_buf_type annotated mem/ms) | Local transport (LOCAL_COPY role, Section 6.3.2), on-chip memory dynamic sharing |
| TRANS (0x2) | LocMsToRmtMem (0x3), LocMemToRmtMem (0x9), LocMsToRmtMs (0x7), RmtMemToLocMs (0x1), RmtMsToLocMem (0x4), RmtMsToLocMs (0x6), RmtMemToLocMem (0x8) | `Ub_Write_Or_Read` (is_write distinguishes direction, src/dst_buf_type annotated mem/ms) | Cross-die/card transport (write/read roles unified type, Section 6.3.2); peer rank resolved via channel reid → eid to rank index (Sections 4.8/6.7) |
| TRANS (0x2) | SyncCKE (0xB), SyncGSA (0xC), SyncXN (0xD) | Local post portion → `Notify_Record`; remote post portion → `Notify_Record` (remote, not paired) | Remote assertion fields (`rmtCKEId/locCKEMask` of SyncCKE, `setRmtCKEId/setRmtCKEMask` of SyncGSA/SyncXN) generate remote record nodes (src=local, dst=INVALID → fixed `remote_record_delay`, not registered in local records), counted in `remote_posts` (Section 6.6); SyncGSA/SyncXN carrying a second remote post also generate one; embedded `waitCKE` can produce `Notify_Wait` |
| REDUCE (0x3) | Add (0x0), Max (0x1), Min (0x2) | `Reduce_Inline` | Reduction computation approximated as transport (Section 5.7); CcuBuffer is always a rank-local buffer (msId high bits are only addressing encoding, Section 6.3.2), local on-chip memory charging |
| (Subgraph attachment) | -- (synthesized during graph generation, not microcode) | None (does not produce task nodes) | Attachment dependencies are all expressed via extra edges (Section 7.4.1): Entry edges SQE predecessor→skeleton head/pre-skeleton unit head, exit edges mission completion set→SQE successor; no nodes are created from nothing in the task graph |

#### 11.3.2 Embedded Synchronization Fields (Cross-Instruction Common)

TRANS / REDUCE / SetCKE / Sync type instruction payloads commonly carry synchronization fields that **additionally produce**
JSON tasks alongside the main semantics:

| Payload Field | JSON Task Type | Behavior |
|---------|--------------|------|
| `waitCKEId` / `waitCKEMask` | `Notify_Wait` | When bits are already satisfied, pairs with local post records (aggregated into a single node: mask=OR of all contributing post consumption bits, first post add_match, rest extra edges, Section 10.3 #20); when bits are insufficient: top-level instruction blocks the queue (stall detection discovers it and generates a remote wait node then replays the remaining instruction parts, `remote_waits`); within a loop body, cannot block, directly generates a remote wait node and continues |
| `setCKEId` / `setCKEMask` | `Notify_Record` | Local Event post, registers seen_post record (Section 6.4) |
| `clearType` bits / SetCKE clear-bit flag | None (register clear only) | Clears bits after wait completes, does not produce a node |

> Therefore, one TRANS instruction can produce 1 data transport task (`Memcpy`/`Ub_Write_Or_Read`),
> and additionally produce several `Notify_Wait` (by bit pairing, 0..n) and 1 `Notify_Record` based on embedded fields.
> Under merged loops (Section 6.5), each slot similarly retains wait/set tasks; only data tasks are folded to size × total iterations.
>
> **A6 difference**: Embedded `waitCKE` only exists in SetCKBit/ClearCKE (CTRL type); local
> TRANS/REDUCE and TransMem/Sync types only carry the tail `setCKEId/setCKEMask` (@26/28,
> additionally producing `Notify_Record`); wait semantics are handled by the independent Wait instruction (Variable value comparison).

#### 11.3.3 A6 (Ascend960) Instruction Full Table (Instruction → JSON Task Type)

devType=960 (`DEV_TYPE_960`) corresponds to the V2/A6 instruction set, **implemented** (ported from
`ccu_task_transform_instruct_a6_v3.cc` in checker, interpreter `_MissionContextA6`). Microcode instruction
= 2B header (isomorphic to A5) + 30B V2 payload (`ccu_microcode_v1.h` CcuV2 structure,
pack(1)); register numbers generally use `GetXnId` (bit15 addressing mode, Section 6.4).

| Instruction Category (type) | Instruction (code) | JSON Task Type | Description |
|----------------|------------|--------------|------|
| LOAD (0x0) | LoadSqeArgsToX (0x1), LoadImdToX (0x2), LoadX (0x6), StoreX (0x7), ClearX (0x8), Nop (0x9), arithmetic-logic Add (0xD)/Sub (0xE)/Mul (0xF)/And (0x10)/Or (0x11)/Not (0x12)/Xor (0x13)/Shl (0x14)/Shr (0x15)/Popcnt (0x16) | **None** for data plane; `Notify_Record` additionally when `setCKEMask != 0` | Unified Variable register read/write (no Address): SQE args sequential consumption (Section 6.1), immediate loading, indirect read/write (LoadX/StoreX, oMode 0=immediate offset/1=register offset), range clear (ClearX), 64-bit wraparound arithmetic-logic (Mul takes lower 32-bit product, shift amount 0x3F truncation, Shl/Shr support logical/arithmetic/circular three forms); all except Nop carry a tail setCKE field (`ProcessSetMask` in checker, mask non-zero → local post node) |
| LOAD (0x0) | Load (0xA), Store (0xB) | **None** for data plane; setCKE additionally produces `Notify_Record` | On-chip memory ↔ Variable data plane: On-chip memory content is not modeled (checker reads/writes data via AllRankParamRecorder); registers are not backfilled, warning logged; only synchronization semantics are retained |
| CTRL (0x1) | SetCKBit (0x2) | `Notify_Record` (+ embedded `waitCKE` additionally produces `Notify_Wait`) | wait → local post → clearType==1 clears wait bits (checker only supports clearType=1, others warn and skip clear) |
| CTRL (0x1) | ClearCKE (0x4) | `Notify_Wait` (embedded `waitCKE`) | wait then clears consumed wait bits per `ClearWaitMask` semantics in checker; `clearMask != 0` is a checker-unsupported scenario (warning + tolerant clearing of target bits) |
| CTRL (0x1) | Loop (0x0) | **None** | Must be triggered by LoopGroup; standalone occurrence warns and skips. `mode==1` conditional Loop: Only expanded when `wishCKEBit == Xp.loopCtxId + 1024` matches (Section 6.5) |
| CTRL (0x1) | LoopGroup (0x1) | (No direct nodes) | Loop control: Parameters from Variable/Xm/Xp three registers (loopInsCnt:10/expandOffset:9/expandCnt:9, and so on, Section 6.5); merge/expand same as A5 (merge validation includes A6 additional rules: wait Event ID not reused, cross-expansion iteration count/length/channel consistent); illegal configuration (loopInsCnt>512, and so on) warns and skips |
| CTRL (0x1) | Jmp (0x5) | **None** | Conditional jump: conditionType 0~5 (==/!=/>/>=/</<=), >=6 unconditional; jumpMode 0=relative pc (mod 0x10000)/1=absolute instruction number, target from `Xn[relTarInstrXnId]`; limited to current instruction window, out-of-range warns and passes through; within loop body passes through; step budget defense against infinite loops |
| CTRL (0x1) | Wait (0x7) | Remote `Notify_Wait` when unsatisfied | Variable value comparison spin (conditionType same as Jmp, defaults to equal): Satisfied passes through with no node; unsatisfied = depends on peer remote register write → top-level stall resolution / within loop body directly generates remote wait node (fixed `remote_wait_delay`, task id tagged `xwt`, counted in `remote_waits`) |
| CTRL (0x1) | Fence (0x8) | **None** | Checker does not support (error abort); here warns and passes through (sequential order inherently guaranteed under queue serial model) |
| TRANS (0x2) | TransLocMemToLocMS (0x0), TransLocMSToLocMem (0x2), TransLocMSToLocMS (0x5), TransLocMemToLocMem (0x6) | `Memcpy` (src/dst_buf_type annotated mem/ms) | Local transport (length `Variable[GetXnId(xlId)]`); no embedded waitCKE, tail setCKE additionally produces `Notify_Record` |
| TRANS (0x2) | TransMem (0x10) | `Ub_Write_Or_Read` (dmaOpCode 3/5=write, 6=read); notify/atomic → remote `Notify_Record` | Unified network transport: Channel = `Variable[GetXnId(xcId)]` register value → peer rank/eid resolution same as A5 (Sections 4.8/6.7); when `msIdMode==1`, write source / read destination is local CcuBuffer (buf_type=ms); dmaOpCode 5 write-after-notify and 0x1A remote atomic count (no data plane) approximated as remote record (mask=lower 16 bits of notification value); `udfEnable` network reduction charged as ordinary UB transport (Section 5.7 approximation); 0x19 and other dmaOpCode values / `udfType != 0` warn and skip data plane (setCKE post still executes) |
| TRANS (0x2) | SyncXnWtx (0xD), SyncAtx (0xE) | Remote `Notify_Record` (approximation); SyncAtx also has setCKE → local `Notify_Record` | Cross-rank register synchronization: `notifyValid==1` notify of SyncWtX and atomic write of SyncAtx approximated as remote record (remote target type unresolvable in single rank, Section 6.6); main value synchronization is a remote register write, no node created (same caliber as A5 SyncXn); tail setCKE field of SyncWtX is not consumed by checker (local does not post); unresolvable channel warns and skips (`eid_unresolved`) |
| REDUCE (0x3) | ReduceAdd (0x0), ReduceMax (0x1), ReduceMin (0x2) | `Reduce_Inline` | Local CcuBuffer batch reduction (msId[8] list + `Variable[XnIdLength]` length); no embedded waitCKE, tail setCKE additionally produces `Notify_Record`; CcuBuffer always local on-chip memory charging (Section 6.3.2) |
| (Subgraph attachment) | -- (synthesized during graph generation, not microcode) | None (does not produce task nodes) | Same as A5 (Section 7.4.1 extra edge attachment) |

> Established deviations between A6 and checker (on-chip memory data plane, remote target type resolution, tolerant degradation of checker error scenarios,
> and so on) are summarized in the `_MissionContextA6` class docstring and Section 6.7 comparison table;
> loop merge/pipeline concurrency recovery (Sections 6.5/6.5.1), queue parallelization (Section 7.4.1), SQE grouping
> and stall resolution (Sections 6.1/6.6) and other framework mechanisms are shared across both generations.

### 11.4 HcclTaskProfInfo Three-Field Comparison (profInfo → JSON Display)

From v3 collection onward, db tasks carry `HcclTaskProfInfo` (Section 4.4.1, single-source definition
`include/sim_profiling_defs.h`). This section provides the complete comparison of the three fields' "value → enum → JSON display string"
mapping and the correspondence of each value with the simulation TaskType (latency model classification, Section 7.3);
the C++ side naming functions (`ProfTaskParamTypeName`/`ProfLinkTypeName`/
`ProfTransportTypeName`) are the sole authoritative source for display strings.

> **Implementation status**: All four fields `task_type`/`transport_type`/`link_type`/`protocol` have been implemented per
> the comparison in this section (Python side display name mappings in `db_reader.py`'s `PROF_*_NAMES`, character-by-character consistent with C++
> naming functions, Section 11.4.5; `graph_builder._add_task` uniformly populates -- `task_type`
> via `Task.task_name`/`display_name`, Section 10.3 #18 closed).

#### 11.4.1 profTaskType (ProfTaskParamType, 0~22 → msprof timeline op name)

This table supplements the correspondence of each value with the simulation TaskType (`task_model.py`, latency model classification) -- previously,
db tasks and simulation task types were matched via the db taskType column (Section 11.2); after profInfo consumption, this table's
`profTaskType → simulation TaskType` serves as the second matching dimension (conditions unchanged; the same db taskType
is routed to different TaskTypes by jettyId/direction, noted within the row):

| Value | Enum | JSON Display String (op name) | Simulation TaskType (latency model classification) | light_perf occurrence scenario |
|---|---|---|---|---|
| 0 | TASK_SDMA | `Memcpy` | `Memcpy` (jettyId invalid, on-chip memory charging) / `Ub_Inline_Write`·`Ub_Write_Or_Read` (jettyId valid, by direction, port charging) | Local copy, cross-die SDMA SQE (linkType distinguishes, Section 11.4.2) ✓ |
| 1 | TASK_RDMA | `RDMASend` | `Memcpy` (jettyId invalid, on-chip memory charging approximation) | DPU host ibv ordinary read/write (`RDMASend`, same caliber as hcomm UbMemTransport::Write/Read) ✓ |
| 2 | TASK_REDUCE_INLINE | `Reduce_Inline` | `Reduce_Inline` (jettyId invalid, on-chip memory charging) / `Ub_Inline_Write`·`Ub_Write_Or_Read` (network approximation, Section 5.7, port charging) | Local reduction ✓ |
| 3 | TASK_REDUCE_TBE | `Reduce_TBE` | -- (does not appear) | 910 generation reserved, does not appear |
| 4 | TASK_NOTIFY_RECORD | `Notify_Record` | `Notify_Record` | SDMA/local notify record ✓ |
| 5 | TASK_NOTIFY_WAIT | `Notify_Wait` | `Notify_Wait` | Notify wait ✓ |
| 6/7 | TASK_SEND_NOTIFY / TASK_SEND_PAYLOAD | `Send_Notify` / `Send_Payload` | -- (does not appear) | hcomm reserved, does not appear |
| 8 | TASK_WRITE_WITH_NOTIFY | `Write_With_Notify` | `Write_With_Notify` | WRITE_WITH_NOTIFY (7) whole task (single-task mapping, Section 10.3 #19 supported) ✓ |
| 9 | TASK_WRITE_REDUCE_WITH_NOTIFY | `Reduce_With_Notify` (tool-side corrected spelling; hcomm source code/msprof has the misspelling `Reuce_With_Notify`, Section 11.4.5) | `Write_With_Notify` (isReduce=1 reduction write payload, data_type distinguishes display) | WRITE_WITH_NOTIFY (7) isReduce=1 |
| 10 | TASK_CCU | `Ccu` | (CCU placeholder whole SQE, expanded into Chapter 6 subgraph, Section 11.3) | CCU placeholder whole SQE ✓ |
| 11/12 | TASK_AICPU_KERNEL / TASK_AICPU_REDUCE | `AicpuKernel` / `Aicpu_Reduce` | -- (not consumed) | AICPU kernel delivery (through device process, currently no collection point/not consumed) |
| 13 | TASK_AIV | `AivKernel` | -- (skipped) | AIV kernel task (skipped) |
| 14 | TASK_UB_INLINE_WRITE | `Ub_Inline_Write` | `Notify_Record` (notify fixed record delay; display name restored to op name ✓) | UBDMA `inlineEn==1` notify inline write |
| 15 | TASK_UB_REDUCE_INLINE | `Ub_Reduce_Inline` | `Ub_Inline_Write` (write) / `Ub_Write_Or_Read` (read) (network reduction approximation, port charging) | UBDMA `udfFlag==1` network reduction |
| 16 | TASK_UB | `Ub_Write_Or_Read` | `Ub_Inline_Write` (write direction) / `Ub_Write_Or_Read` (read direction) (direction expressed by direction/is_write) | UBDMA ordinary read/write WQE (both write/read directions have the same value) |
| 17~22 | TASK_DPU_KERNEL / THREAD_FENCE / CHANNEL_FENCE / INLINE_WRITE / NOTIFY_WAIT / WRITE_WITH_NOTIFY | `DpuKernel` / `Dpu_ThreadFence` / `Dpu_ChannelFence` / `Dpu_Notify_Record` / `Dpu_Notify_Wait` / `Dpu_Write_With_Notify` | `Dpu_Notify_Record`→`Notify_Record`, `Dpu_Notify_Wait`→`Notify_Wait` (cross-rank fixed delay), `Dpu_Write_With_Notify`→`Write_With_Notify` (single-task port charging, Section 11.4.6); `DpuKernel`/Fence types not produced by current collection path | Host CPU RoCE (ibv_dpu_stub) tasks ✓ (Section 11.4.6) |
| 0xFF | UNKNOWN | `Unknown` | -- (SYNC_STREAM and so on are already skipped, no nodes produced) | SYNC_STREAM, level1 channel API tasks, v2 old BLOBs and other unlabeled sources |

> `task_type` has switched to op name (Section 10.3 #18 closed); the "Simulation TaskType" column in this table is now
> an internal latency model classification index (Section 11.4.5 #3); ✓ rows have op names consistent with TaskType values.
> After SQE batch merging (Section 5.1.1), the merged task's op name and link/transport take the **last** data
> task's profInfo in the batch (description follows the last task, length takes the full batch accumulation); v3 and earlier BLOBs without
> sqeGroupId are not merged and are displayed per-task.

#### 11.4.2 linkType (ProfLinkType, 0~9 → hcomm DfxLinkType enum name)

| Value | Enum | JSON Display String | light_perf occurrence scenario |
|---|---|---|---|
| 0 | LINK_ONCHIP | `ONCHIP` | Local Memcpy/Reduce_Inline/Notify_* |
| 1 | LINK_HCCS | `HCCS` | Reserved (current collection path does not produce) |
| 2 | LINK_PCIE | `PCIE` | Cross-die SDMA SQE (same-device determined as local ONCHIP, Section 4.4.1 collection rules) |
| 3 | LINK_ROCE | `ROCE` | Host CPU RoCE (DPU, ibv_dpu_stub) tasks ✓ (Section 11.4.6) |
| 4/5/6 | LINK_SIO / HCCS_SW / STANDARD_ROCE | `SIO` / `HCCS_SW` / `STANDARD_ROCE` | Reserved |
| 7 | LINK_UB | `UB` | Network tasks (UBDMA WQE all types) |
| 8 | LINK_UBoE | `UBoE` | UBoE channel (collection side does not yet distinguish UB/UBoE, does not appear at this time) |
| 9 | LINK_RESERVED | `RESERVED` | Reserved |
| 0xFF | UNKNOWN | `Unknown` | CCU/AIV/SYNC_STREAM, level1 channel API tasks |

> **Implementation status**: The `link_type` field in trace args is populated per this table (`prof_link_type_name`,
> uniformly injected by `graph_builder._add_task`, Section 11.1 approach a); this table and the
> `CommProtocol` channel protocol (HCCS/TCP/ROCE/UB_CTP/UB_TP/PCIE/SIO) previously carried by `link_type` are **two different
> concepts**: CommProtocol describes the channel establishment protocol, while DfxLinkType describes the task execution link (profiling
> dimension). The CommProtocol name has been migrated to the `protocol` field (`N/A` when no channel information is available).

#### 11.4.3 transportType (ProfTransportType, 0~4 → msprof L1 detail transportType value range)

| Value | Enum | JSON Display String | light_perf occurrence scenario |
|---|---|---|---|
| 0 | TRANSPORT_SDMA | `SDMA` | Cross-die SDMA SQE |
| 1 | TRANSPORT_RDMA | `RDMA` | Reserved (910 generation RDMA semantics) |
| 2 | TRANSPORT_LOCAL | `LOCAL` | Local tasks (Memcpy/Reduce_Inline/Notify_*) |
| 3 | TRANSPORT_UB | `UB` | Network tasks (**latency model anchor**: port bandwidth sharing charging, Section 5.5) |
| 4 | TRANSPORT_ROCE | `ROCE` | Host CPU RoCE (DPU, ibv_dpu_stub) tasks ✓ (Section 11.4.6) |
| 0xFF | UNKNOWN | `Unknown` | Unlabeled tasks |

> **Implementation status**: The `transport_type` field in trace args is populated per this table
> (`prof_transport_type_name`, uniformly injected by `graph_builder._add_task`).
> Same-named `Memcpy` tasks' local/cross-die difference is expressed by linkType+transportType:
> ONCHIP+LOCAL (local) vs PCIE+SDMA (cross-die) -- trace args can now losslessly distinguish these two types of
> previously indistinguishable tasks in JSON.

#### 11.4.4 Ub_Inline_Write Bidirectional Semantic Misalignment and Switch Implementation Record

**Historical problem**: Before the switch, JSON's `Ub_Inline_Write` (network write direction send) and hcomm
profiling's `Ub_Inline_Write` (TASK_UB_INLINE_WRITE, notify inline write) **referred to different tasks**
and were inversely misaligned:

| Task (collection semantics) | db taskType | ProfTaskParamType (op name) | Pre-switch JSON type |
|---|---|---|---|
| UBDMA ordinary write WQE (network send, write direction) | MEM_CPY (3) | `TASK_UB` → `Ub_Write_Or_Read` | `Ub_Inline_Write` ← name misalignment |
| UBDMA network reduction (write/read direction) | REDUCE (2) | `TASK_UB_REDUCE_INLINE` → `Ub_Reduce_Inline` | `Ub_Inline_Write` / `Ub_Write_Or_Read` ← name misalignment |
| UBDMA inline write notify (`inlineEn==1`) | NOTIFY_RECORD (1) | `TASK_UB_INLINE_WRITE` → `Ub_Inline_Write` | `Notify_Record` ← name misalignment |

**Implemented (Section 10.3 #18 closed)** -- trace/dump display name switched to `profInfo.profTaskType`
op name (`Task.task_name`, uniformly injected by `graph_builder._add_task`); all three misalignments eliminated:
1. Network data tasks (TASK_UB) write/read uniformly display `Ub_Write_Or_Read`; **direction is no longer encoded
   in the type name**, expressed by the existing `direction` (write/read) and `is_write` fields in trace args
   -- latency model unchanged, still charged by port write/read direction bandwidth sharing respectively (Section 5.5).
2. The `Ub_Inline_Write` name is **restored** to notify inline write tasks (db taskType=
   NOTIFY_RECORD + inlineEn==1), one-to-one with msprof timeline.
3. Network reduction displays `Ub_Reduce_Inline` (TASK_UB_REDUCE_INLINE), preserving reduction semantic distinguishability.
4. `Notify_Record`/`Notify_Wait`/`Reduce_Inline`/`Memcpy` four types were already consistent with op names;
   the switch causes no change (`Memcpy` corresponds to TASK_SDMA).

**Implementation trade-offs (established decisions)**:
- v2 old BLOBs (without profInfo, profTaskType=0xFF) display `Unknown` -- same caliber as
  `link_type`/`transport_type` v2 behavior (strict semantics, no internal name fallback).
- CCU microcode sub-tasks (Section 11.3) have no profInfo; display name falls back to internal TaskType name -- simulator
  semantic products have no reconciliation requirement; current state maintained (network transport unified as `Ub_Write_Or_Read`, direction expressed by
  `direction`/`is_write`, Section 11.3.1).
- Internal `TaskType` enum retired as latency model classification (Section 11.4.5 #3); simulation/pairing/billing have zero changes.

#### 11.4.5 String Consistency Convention (Reconciliation Basis)

1. **Python-side display strings must be character-by-character consistent with the C++ side's three naming function outputs**
   (single source in `sim_profiling_defs.h`), including existing spellings in hcomm source code
   (`Ub_Write_Or_Read`, `Dpu_Notify_Record`, and so on) -- this is the basis for field-by-field reconciliation with msprof timeline /
   L1 detail.
   **Sole exception**: The op name for `TASK_WRITE_REDUCE_WITH_NOTIFY` is unified on the tool side as the corrected spelling
   `Reduce_With_Notify` (hcomm source code has the misspelling `Reuce_With_Notify` in two places:
   `profiling_common.h:42` and `dfx_profiling_handler_lite.cc:67`; msprof raw
   data displays the misspelled version); cross-system reconciliation encountering this string requires
   `Reuce_With_Notify ↔ Reduce_With_Notify` difference mapping.
2. trace args field-level switch (**fully implemented**, Section 10.3 #18 closed):
   `transport_type` ← transportType name, `link_type` ← linkType name
   (uniformly injected by `graph_builder._add_task`, value range tables in Sections 11.4.2/11.4.3), CommProtocol
   migrated to new field `protocol` (approach a, Section 11.1); `task_type` ← profTaskType op name
   (`Task.task_name`/`display_name`, Section 11.4.4 implementation record).
3. `TaskType` enum has been retired as internal latency model classification (`task_name` handles display duties);
   trace event names (`traceEvents[].name`) and `args.task_type` have all switched to op names;
   CCU microcode sub-tasks/CLI input have no profInfo; `display_name` falls back to internal names.

#### 11.4.6 DPU (Host CPU RoCE) Task Collection and Merging

Host DPU transport (hcomm `HostCpuRoceChannel`, pod networking cross-node path) ibv hijack tasks
(`ibv_dpu_stub.cc`) previously did not label profInfo (displayed `Unknown`), split WriteWithNotify into two
tasks, and resolved the peer back to the local side. This section records the alignment implementation for this path.

**profInfo labeling** (aligned with hcomm `host_cpu_roce_channel.cc` DFX caliber, link/transport
always ROCE/ROCE):

| ibv Operation | db taskType | profTaskType (op name) |
|---|---|---|
| Pure imm notify (len=0 WRITE_WITH_IMM / hybrid) | NOTIFY_RECORD | `TASK_DPU_INLINE_WRITE` → `Dpu_Notify_Record` |
| Write+imm (len>0 WRITE_WITH_IMM) | **WRITE_WITH_NOTIFY (7) single task** | `TASK_DPU_WRITE_WITH_NOTIFY` → `Dpu_Write_With_Notify` |
| Pure RDMA write/read (no imm) | MEM_CPY | `TASK_RDMA` → `RDMASend` (aligned with hcomm `UbMemTransport::Write/Read` precedent) |
| poll_cq pairing | NOTIFY_WAIT | `TASK_DPU_NOTIFY_WAIT` → `Dpu_Notify_Wait` |

**WriteWithNotify merged single task**: The entire operation produces only one taskType=7 task
(`WriteWithNotifyTaskData`: src/dst rank, offsets, len, notifyId, protocol),
aligned with hcomm's entire operation reporting only one `Dpu_Write_With_Notify` caliber; the `DpuPendingNotify`
pairing mechanism is unchanged. The simulation side `_map_write_with_notify` uses port write direction bandwidth sharing
(local RoCE eid → ranktable `d2h` port; falls back to `default_port_bandwidth` when `d2h` port bandwidth is not configured).

**Real peer resolution (phantom echo problem, socket timing binding)**: In estimate mode, the socket layer constructs
a phantom peer (local Send mirrored back as Recv, `hccp_ra_socket_stub.cc`); the "remote gid" returned by QP handshake is
actually the local side itself, and `RaTypicalQpModify`'s gid→IP resolution returns to the local side. Fix
(zero-byte read, relying only on API call timing):
- `RaGetSockets` resolves the real peer EndPoint from `SocketInfoT.remoteIp` (hcomm link configuration, both client/server roles
  carry it) and stores it in FdInfo (peerDeviceId/peerEid/localEid);
- `RaGetQpAttr` anchor (hccp_stub, only executed within the estimate branch) records the QP pending binding
  (`DpuRecordPendingBindQp`, `bindAttempted` prevents Modify from re-querying attr and double-binding);
- The next `RaSocketSend` in the same thread (that is, the attr packing→Send timing of hcomm ExchangeData) uses
  `BindDpuPendingQps` to bind the QP to the socket's real peer: `DpuQpInfo` adds
  `peerRankId`/`lcEid` (peer rank resolved from FdInfo's peerEid via `RankEidMap`, since the estimate
  communicator table has only the local rank; lcEid for WRITE_WITH_NOTIFY port charging).
`RaTypicalQpModify` maintains the loopback-plane semantics of `remoteDeviceId`
(required by `DpuPendingNotify` pairing; in estimate mode = local side); task metadata uniformly uses the bound fields.

**NotifyWait precise synthesis (hcomm primitive interception)**: The correct semantics are "only pending notifications inserted by the peer post
generate NotifyWait for me"; under the phantom, the peer does not run and this notification has no real source; moreover, hcomm
`NotifyWait` performs hard imm validation on the poll result (`wc.imm_data == dpuNotifyId`, immediately reporting an error on mismatch). Implementation
(`hcomm_api_stub.cc` intercepts 5 hcomm northbound primitives, real function pointer obtained via static one-time dlsym resolution,
**normal mode first line short-circuits as pure forwarding, zero overhead**):
- Post-type primitives (`HcommChannelNotifyRecordOnThread` / `HcommWriteWithNotify(±Nbi)
  OnThread`, input carries `remoteNotifyIdx`) → `InsertDpuPendingNotifyEstimate`
  records `channelSlots[channel][idx] = (imm, notifyId)` slot mapping -- under echo,
  `remoteDpuNotifyIds_ == localDpuNotifyIds_`; the post-side idx→imm mapping is the same as the wait-side
  expected mapping (same idx, same slot value);
- Wait-type primitives (`HcommChannelNotifyWaitOnThread(±WithDefaultTimeout)`, input carries
  `localNotifyIdx`) → `ConsumeDpuPendingNotifyAndEmit` precise path: Expected
  imm = `channelSlots[channel][idx]` is **precisely known**, directly synthesizes a "virtual peer post" to construct
  the wc return; imm always passes hcomm hard validation; the NotifyWait node records the local real wait event,
  and only the "notification arrival" external fact is virtualized (isomorphic to the socket echo virtualizing "peer response");
- Design constraint: Estimate mode wait consumption **only uses the precise path** -- all wait entries must be intercepted
  (single-card phantom FIFO loopback consumption is unreliable; imm mismatch/timeout when send/receive are asymmetric, old mechanism removed);
  unintercepted entries → `pollCq::miss` **ERROR** (deduplicated once per qpn+idx combination to prevent busy-loop
  log flooding) → hcomm timeout exposes the issue, based on which interception is added in `hcomm_api_stub.cc`.

**taskSrc source marker (BLOB v5)**: `HcclTaskMetaData` adds `taskSrc`
(0=default, 1=DPU); tasks launched by the DPU stub set it to 1, allowing the Python side to precisely split by source
(not relying on link semantic criteria such as transport==ROCE; future sources such as AICPU RoCE are unaffected).

**DPU sequence merge pass (Section 5.1.2, `sqe_group_merge.py::merge_dpu_write_sequences`)**:
hcomm `WriteWithNotify` splits large messages with len > maxMsgSize into N-1 ordinary RDMA_WRITE +
1 terminating WRITE_WITH_IMM (the entire operation still reports only one task); the collection side stores per-WR as N-1 MEM_CPY +
1 WRITE_WITH_NOTIFY. The pass scans DPU stream order for src/dst dual-address contiguous MEM_CPY chains,
terminated and merged into a single entry by an address-contiguous WRITE_WITH_NOTIFY (len accumulated across the full chain, description follows the terminating task including
notifyId, occupies the terminating slot); chains without a terminator are not merged (independent pure writes remain as per-entry `RDMASend`);
only `src_type==1` data tasks participate; other tasks are retained as-is without cutting the chain.

> **Verification** (222 networking 8MB all_reduce): First round (20260921) 15 `Unknown` → 0
> (`Dpu_Notify_Record` ×5 / `Dpu_Notify_Wait` ×9 / `Dpu_Write_With_Notify` ×4,
> link=ROCE), peer rank=2/4/6 (real cross-pod peers), `eid_unresolved=0`,
> WRITE_WITH_NOTIFY enters `d2h` port charging. Final version (20260922, after primitive interception):
> estimate (-np 1) makespan 166.596us, primitive interception 16 times, precise synthesis 11 times, miss=0;
> normal (-np 8) checker All Success (primitives pure forwarding, interception log 0 entries); single-machine 8p
> (118) SQE path regression unchanged.
