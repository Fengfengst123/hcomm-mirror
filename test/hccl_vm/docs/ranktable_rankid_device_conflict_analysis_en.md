# Analysis of Mapping Conflict Between rankId and Physical Device ID in Ranktable

## Problem Overview

### Symptom

In a cluster topology where multiple servers exist and some servers contain multiple devices, the hcomm service initialization phase reports a `CheckRankGraphAddrs` failure:

```
[CommunicatorImpl][CheckRankGraphAddrs]the ip address
  IpAddress[...c002:0001...] of ranktable in rank 1 is error!
[CommunicatorImpl][CheckRankGraphAddrs]the ip address
  IpAddress[...c002:000b...] of ranktable in rank 0 is error!
```

The debug logs reveal the key clue:

```
ZHF-DEBUG: peer deviceId is 0, devPhyId is 1   ← rank 1: the ranktable says device 0, the process actually runs on device 1
ZHF-DEBUG: peer deviceId is 1, devPhyId is 0   ← rank 2: the ranktable says device 1, the process actually runs on device 0
```

### Trigger Conditions

The issue triggers only when the following conditions are met:

```
At least one server exists that satisfies:
  1. The device count D_s of that server is greater than 1
  2. Other servers before it have already consumed some rankId values
     (that is, the rankId values on that server do not start from 0 in a contiguous sequence)
```

This is equivalent to: **the YAML topology configuration contains a server with a device count greater than 1, and that server is not the only server that owns ranks.**

| Configuration Type | Sample | Triggers? | Reason |
|---------|------|---------|------|
| Every server has 1 device | (1,1), (1,1,1,1) | Does not trigger | `rank % 1 = 0` always holds; sequential allocation is equivalent to modular arithmetic |
| Single server with multiple devices | (2), (4), (8) | Does not trigger | Only one server exists; `rank % D = rank`; sequential allocation is equivalent to modular arithmetic |
| Multiple servers, some with multiple devices | (1,2), (2,4) | **Triggers** | On the server with multiple devices, `rank % D_s != sequential value` |
| Multiple servers, all with multiple devices (uniform) | (2,2), (2,2,2,2), (8x8) | **Triggers** | Same as above |

---

## Key Concept Definitions

### Two Types of rankId

This document involves two distinct rankId concepts that must be strictly differentiated:

| Concept | Symbol | Definition | Source | Allocation Method |
|------|------|------|--------|---------|
| ranktable rankId | **hcclRankid** | The value of the `rank_id` field in ranktable.json | The `CreateRankTableFile` function in this tool | Incremented from 0 in superPod -> server -> device traversal order |
| MPI rankId | **mpiRankid** | The identifier that MPI assigns to each process | The MPI runtime (`MPI_Comm_rank`) | Determined by the MPI scheduling strategy; typically 0,1,2,... |

> **Implicit convention**: hcclRankid and mpiRankid are numerically equal (both increment as 0,1,2,...). This equality is the prerequisite for the entire system to function correctly.

### Device Identifier System

| Concept | Definition | Relationship in This System |
|------|------|-----------------|
| `physical_id` (phyDevId) | The physical identifier of a device; remains fixed | Defined by the cluster topology file |
| `logic_id` (logicDevId) | The logical identifier of a device; supports mapping | In this simulator, `logic_id = physical_id` (numerically equal) |
| `device_id` (DB key) | The primary key of the Device table | Assigned by the internal database auto-increment mechanism |
| `local_id` | The device logical identifier recorded in ranktable.json | Server form: set to `srcDevPhyId`; POD form: set to `boardIdx * devNumPerBoard + srcDevPhyId` (POD-global id) |
| `serverKey` | The unique identifier of a server | `GetServerKeyById(superPodId, serverId, boardId)` |

> **POD form (4-layer topology: board → pod → supernode → cluster) conventions**:
> In the table above, "server" refers to the **leaf node (host node)**. In the server form, the leaf node is the server itself;
> in the POD form, the leaf node is the **board** (each board occupies one row in the `Server` table in the database, distinguished by `board_id`). Therefore:
> - `physical_id` and the `device_id` in the ranktable are **intra-board physical ids** (for example, `[0,7]`), consistent with real hardware;
> - the `local_id` in the ranktable is a **POD-global id** (= `board_id × devices per board + intra-board physical id`, for example, the value range for 1dpod64 is `[0,63]`), consistent with the `local_id` convention in the POD-level topo.json;
> - **D (`serverMeta.size()`) in all formulas in this document = the number of devices on that board enrolled in the communicator** (for example, 8),
>   so the application side calls `aclrtSetDevice(rank % 8)` and `aclrtGetDeviceCount()` returns 8;
> - `serverKey` requires the `(superPodId, serverId, boardId)` triplet for identification.

### Involved Software Layers

```
┌─────────────────────────────────────────────────────────┐
│  Application Layer (hccl_test / sample programs)          │
│  MPI_Comm_rank -> mpiRankid                              │
│  aclrtSetDevice(mpiRankid % deviceCount)                 │
│  HcclCommInitClusterInfoConfig(ranktable, mpiRankid,...) │
└──────────────────────┬──────────────────────────────────┘
                       │ calls
┌──────────────────────▼──────────────────────────────────┐
│  hcomm Service Layer (libhccl_v2.so)                     │
│  CommunicatorImpl::CheckRankGraphAddrs()                │
│  RankGraphBuilder::BuildFromRankTable()                  │
└──────────────────────┬──────────────────────────────────┘
                       │ calls
┌──────────────────────▼──────────────────────────────────┐
│  This Project (hccl-vm simulator)                        │
│  AscendClusterTopoParser::CreateRankTableFile()           │
│  aclrtSetDevice (stub) / RaGetDevEidInfoList (stub)      │
└─────────────────────────────────────────────────────────┘
```

---

## Problem Reproduction Environment

### Topology Configuration

The test topology: 1 superPod, 2 servers (rack0 has 1 device, rack1 has 2 devices), 3 ranks in total.

```
superPod0
├── rack0 (server0): [phyDev0]              ← 1 device
└── rack1 (server1): [phyDev0, phyDev1]     ← 2 devices
```

### Process and Device Allocation (from Actual Logs)

| PID | mpiRankid | `aclrtSetDevice(rankId % D)` | logicDevId | phyDevId | Server |
|-----|-----------|------------------------------|------------|----------|-------------|
| 13311 | 0 | `0 % 1 = 0` | 0 | 0 | rack0 (1 device) |
| 13312 | 1 | `1 % 2 = 1` | 1 | 1 | rack1 (2 devices) |
| 13313 | 2 | `2 % 2 = 0` | 0 | 0 | rack1 (2 devices) |

> Log source: `log.txt:79,83,85`
> ```
> [aclrtSetDevice] Init Rank: rankId=1, deviceKey=10, logicDevId=1, phyDevId=1 totalRanks=3
> [aclrtSetDevice] Init Rank: rankId=2, deviceKey=9, logicDevId=0, phyDevId=0 totalRanks=3
> [aclrtSetDevice] Init Rank: rankId=0, deviceKey=1, logicDevId=0, phyDevId=0 totalRanks=3
> ```

The key point: rack1 has 2 devices. `mpiRankid % 2` yields 1 for rank 1 and 0 for rank 2, **which is exactly the reverse of the sequential traversal order [0,1]**.

### ranktable.json Comparison Before and After the Change

| rank_id | Old Code `rankId % D` | Old Code device_id | New Code `deviceIdx` | New Code device_id | Actual phyDevId in Application |
|---------|---------------------|-----------------|-------------------|-----------------|------------------|
| 0 | `0 % 1 = 0` | 0 | 0 | 0 | 0 |
| 1 | `1 % 2 = 1` | **1** | 1 | **0** | **1** |
| 2 | `2 % 2 = 0` | **0** | 2 | **1** | **0** |

- **Old code** (`rankId % D`): the device_id in the ranktable **matches** the actual phyDevId in the application -> validation passes
- **New code** (`deviceIdx`): the device_id values for rank 1 and rank 2 are swapped -> **conflict**

> The ranktable generated by the old code is in the archive file `hccl_vm_install/archive/20260815_094326/data/ranktable.json`
> The ranktable generated by the new code is in `hccl_vm_install/data/ranktable.json`

---

## CheckRankGraphAddrs Validation Logic Details

### Function Location and Signature

```
File: hcomm/src/legacy/ascend950/framework/communicator/communicator_impl.cc
Function: void CommunicatorImpl::CheckRankGraphAddrs() const
Lines: 1291-1338
```

### Validation Flow

The function retrieves EIDs through two separate paths and then compares whether the two sets match:

```
                      CheckRankGraphAddrs
                      /                  \
                Path A                   Path B
         (actual hardware EID)    (EID claimed by ranktable)
                |                            |
    Find entry in ranktable          rankGraph->GetPeer(myRank)
    where deviceId==devPhyId         (look up peers_ table by rank_id)
                |                            |
    RaGetDevEidInfoList                peer->GetIfaces()
    (phyId=devPhyId)                  retrieve the EID address of the interface
                |                            |
    localEidSet (EID set)         interface->GetAddr().GetEid()
                \                            /
                 \                          /
                  → localEidSet.count(interface EID) == 0 ?
                       not in set → THROW exception
```

### Path A: Actual Hardware EID

```cpp
// communicator_impl.cc:1306-1316
for (auto& rank : ranktableInfo->ranks) {
    if (rank.deviceId == devPhyId) {                    // select entry: deviceId == physical device of this process
        HRaInfo info(HrtNetworkMode::HDC, rank.deviceId);
        std::vector<HrtDevEidInfo> localEidInfos = HrtRaGetDevEidInfoList(info);  // query hardware
        for (auto& eidInfo : localEidInfos) {
            localEidSet.insert(eidInfo.ipAddress.GetEid());  // store only the EID/IP value
        }
        break;  // call only once, then exit
    }
}
```

**Key characteristics**:
- The `RaGetDevEidInfoList` interface only recognizes `phyId`; it does not recognize rankId
- The `devPhyId` passed in is the physical device on which the process actually runs
- In this path, ranktable.json is used only to "filter entries where `deviceId == devPhyId`"; the EID values come from the hardware query
- The `chipId` field is populated (`hccp_ccu_stub.cc:573`), but `CheckRankGraphAddrs` **does not read chipId**; the EID comparison matches only by the original IP value

### Path B: EID Claimed by the Ranktable

```cpp
// communicator_impl.cc:1322-1328
const auto& peer = rankGraph->GetPeer(myRank);   // look up peers_ where rank_id == myRank
const auto& interfaces = peer->GetIfaces();
for (auto& interface : interfaces) {
    // check whether the EID of the interface is in localEidSet
    if (interface->GetPos() == AddrPosition::DEVICE
        && protocols.count(LinkProtocol::PCIE) == 0
        && protocols.count(LinkProtocol::UBOE) == 0
        && localEidSet.count(interface->GetAddr().GetEid()) == 0) {
        THROW<InvalidParamsException>(...);  // EID mismatch, report error
    }
}
```

**Key characteristics**:
- `GetPeer(myRank)` uses `myRank` directly as the key to look up the `peers_` table
- The interface EID of the peer originates from the `level_list` of the entry where `rank_id == myRank` in ranktable.json
- **This path is entirely determined by the content of ranktable.json**, that is, it is affected by lines 450/451

### Retrieval of devPhyId and myRank

```cpp
// communicator_impl.cc:1256-1269 (InitCommonData)
myRank = commParams.myRank;                              // ← comes from the rank parameter = mpiRankid
devLogicId = HrtGetDevice();                              // ← the logical device already set by the current process
devPhyId = HrtGetDevicePhyIdByIndex(devLogicId);         // ← the physical device corresponding to that logical device
HCCL_INFO("ZHF-DEBUG: devLogicId is %u, devPhyId is %u", devLogicId, devPhyId);
```

---

## Source and Attribution of myRank

### Complete Propagation Chain

```
┌─ Application Layer ─────────────────────────────────────────────────────┐
│                                                                         │
│  MPI_Comm_rank(MPI_COMM_WORLD, &procRank)    // procRank=mpiRankid      │
│  uint32_t devId = (uint32_t)procRank;                                   │
│                                                                         │
│  // devId serves two purposes simultaneously:                            │
│  aclrtSetDevice(devId);                      // select device            │
│  HcclCommInitClusterInfoConfig(rtk, devId, &config, &comm);  // pass rank│
│                           ↑                                              │
│                      rank parameter = mpiRankid                          │
└──────────────────────────┬───────────────────────────────────────────────┘
                           │
┌─ hcomm Entry Point (op_base_v2.cc) ─────────────────────────────────────▼┐
│                                                                           │
│  HcclCommInitClusterInfoConfigV2(clusterInfo, uint32_t rank, ...)         │
│  // op_base_v2.cc:455                                                    │
│  → ParseJsonAndCreateComm(data, rank, ...)                                │
│  // op_base_v2.cc:501                                                    │
│  → CreateCommConfig(rank, config, comm, ranktableM)                      │
│  // op_base_v2.cc:318                                                    │
└──────────────────────────┬────────────────────────────────────────────────┘
                           │
┌─ CommParams Construction (op_base_v2.cc:146-152) ────────────────────────▼┐
│                                                                            │
│  Hccl::CommParams commParams{                                             │
│      commId,                                                              │
│      static_cast<Hccl::RankId>(rank),   // ← myRank = mpiRankid           │
│      0,                                                                   │
│      static_cast<Hccl::RankId>(rank),   // rankInParentComm also = rank   │
│      ...                                                                  │
│  };                                                                       │
└──────────────────────────┬─────────────────────────────────────────────────┘
                           │
┌─ CommunicatorImpl (communicator_impl.cc:1261) ───────────────────────────▼┐
│                                                                            │
│  void CommunicatorImpl::InitCommonData(const CommParams& params) {        │
│      myRank = commParams.myRank;    // ← final destination: mpiRankid     │
│      devLogicId = HrtGetDevice();                                         │
│      devPhyId = HrtGetDevicePhyIdByIndex(devLogicId);                     │
│  }                                                                        │
└────────────────────────────────────────────────────────────────────────────┘
```

### Conclusion

**`myRank` is mpiRankid.** It is passed in through the `rank` parameter of `HcclCommInitClusterInfoConfig`, and its origin is the `MPI_Comm_rank` call in the application.

### Construction and Lookup of the peers_ Table

```cpp
// rank_graph_builder.cc:320-328 (BuildFromRankTable)
for (const auto& rankInfo : rankTable_->ranks) {
    RankId rankId = rankInfo.rankId;           // ← hcclRankid (rank_id from ranktable.json)
    auto peer = make_shared<NetInstance::Peer>(
        rankId, rankInfo.localId, rankInfo.replacedLocalId,
        rankInfo.deviceId, ...);               // ← the deviceId of the Peer comes from ranktable.json
    peers_.emplace(rankId, peer);              // ← peers_ uses hcclRankid as the key
}
```

```cpp
// communicator_impl.cc:1322 (CheckRankGraphAddrs)
const auto& peer = rankGraph->GetPeer(myRank);  // GetPeer(peers_.at(myRank))
//        ↑ uses mpiRankid to look up the hcclRankid table
```

**The lookup succeeds** solely because hcclRankid == mpiRankid (both increment numerically as 0,1,2,...).

### Source of the Peer deviceId

```
The entry in ranktable.json where rank_id==myRank
    ↓ (BuildFromRankTable reads it)
rankInfo.deviceId
    ↓ (passed to the Peer constructor)
peer->deviceId_ = rankInfo.deviceId
    ↓ (GetDeviceId returns it)
peer->GetDeviceId() == the device_id of the entry for that rank in the ranktable
```

This is the source of `peer deviceId` in the logs. It equals the `device_id` field of the entry where `rank_id == myRank` in ranktable.json.

---

## How the Two EID Paths Relate to rankId

### Overview Diagram

```
                          mpiRankid (= myRank)
                         /                   \
                        /                     \
               ┌─ Path A ─┐              ┌─ Path B ──────────┐
               │          │              │                    │
    app calls  │ aclrtSetDevice           │ GetPeer(myRank)   │
    (app layer)│   (mpiRankid % D)        │   = peers_[myRank] │
               │          │              │   = entry in       │
               │          │              │     ranktable where │
               │          │              │     rank_id==myRank │
               │          │              │          │          │
    look up    │ (serverKey,logic_id)    │     interface EID   │
    Device     │   → device              │     (addr in        │
    table to   │   → phyDevId           │      level_list)    │
    get        │          │              │          │          │
    phyDevId   │          │              │          │          │
               │          │              │          │          │
    query      │ RaGetDevEidInfoList      │          │          │
    hardware   │   (phyDevId)            │          │          │
    EID        │          │              │          │          │
               │          │              │          │          │
               ▼          ▼              ▼          ▼          │
          localEidSet (EID set)    interface EID (single EID)   │
               \          /              \          /          │
                \        /                \        /           │
                 → compare: localEidSet.count(interface EID) ←─┘
```

### rankId -> Device Mapping Formulas for the Two Paths

| Path | rankId -> Device Mapping Formula | Formula Location | Affected by Lines 450/451 |
|------|--------------------------|-------------|---------------------|
| Path A (hardware) | `server.devices[mpiRankid % D]` | Hardcoded in the application layer (`aclrtSetDevice`) | **No**, the application is fixed |
| Path B (ranktable) | `server.devices[logicDevId]`, where logicDevId is determined by lines 450/451 | `CreateRankTableFile` | **Yes** |

### Formula Consistency Analysis

#### Old Code (line 450): `logicDevId = rankId % serverMeta.size()`

```cpp
uint32_t logicDevId = rankId % serverMeta.size();  // same formula as the application rank % D
```

| rankId | Path A: `rankId % D` -> device | Path B (line 450): `rankId % D` -> device | Consistent? |
|--------|-------------------------------|-------------------------------------|---------|
| 0 | `0 % 1 = 0` -> phyDev0 | `0 % 1 = 0` -> phyDev0 | Consistent |
| 1 | `1 % 2 = 1` -> phyDev1 | `1 % 2 = 1` -> phyDev1 | Consistent |
| 2 | `2 % 2 = 0` -> phyDev0 | `2 % 2 = 0` -> phyDev0 | Consistent |

-> Both sides have the same mapping -> EIDs match -> validation passes

#### New Code (line 451): `logicDevId = deviceIdx`

```cpp
uint32_t logicDevId = deviceIdx;  // sequential allocation 0,1,2,...; may differ from the application % D formula
```

| rankId | Path A: `rankId % D` -> device | Path B (line 451): `deviceIdx` -> device | Consistent? |
|--------|-------------------------------|-------------------------------------|---------|
| 0 | `0 % 1 = 0` -> phyDev0 | `0` -> phyDev0 | Consistent |
| 1 | `1 % 2 = 1` -> phyDev1 | `0` -> phyDev0 | **Inconsistent** |
| 2 | `2 % 2 = 0` -> phyDev0 | `1` -> phyDev1 | **Inconsistent** |

-> The devices for rank 1 and rank 2 are swapped -> EIDs do not match -> validation fails

### Detailed EID Comparison Failure Process (using rank 1 as a sample)

```
rank 1 (PID 13312):
  ┌─ Path A (actual hardware EID) ────────────────────────────────────┐
  │ app: aclrtSetDevice(1 % 2 = 1) -> logicDevId=1                   │
  │ Device table: logic_id=1 -> phyDevId=1                            │
  │ entry in ranktable where device_id==1 -> RaGetDevEidInfoList(1)   │
  │ localEidSet = {192.2.0.11, 192.2.0.18}  (real IPs of device 1)    │
  └───────────────────────────────────────────────────────────────────┘

  ┌─ Path B (EID claimed by ranktable) ──────────────────────────────┐
  │ GetPeer(myRank=1) -> peers_[1]                                   │
  │   = entry in ranktable where rank_id=1                            │
  │ new code: device_id=0 for that entry (should be 1)               │
  │ peer->GetDeviceId() = 0  (ranktable says device 0)               │
  │ interface EID = {192.2.0.1, 192.2.0.8} (IPs of device 0)         │
  └───────────────────────────────────────────────────────────────────┘

  comparison: 192.2.0.1 in {192.2.0.11, 192.2.0.18} ?
        -> not in set -> THROW "ip address ... is error!"
```

Log corroboration:
- `log.txt:1062`: `peer deviceId is 0, devPhyId is 1` (Path B yields deviceId=0 for the peer, but the process runs on devPhyId=1)
- `log.txt:1267`: `the ip address ...c002:0001... of ranktable in rank 1 is error!`

---

## Root Cause

### One-Sentence Summary

**Both EID retrieval paths use rankId as the anchor but map to a device through different formulas.** Lines 450/451 change only the formula for Path B (the ranktable side); the formula for Path A (the application side) remains fixed. When the server device count is greater than 1 and the server is not the first server, the two formulas produce different results, causing the same rankId to point to different devices on the two sides, and the EID comparison fails.

### Path A rankId Association (Indirect)

The `RaGetDevEidInfoList(devPhyId)` interface does not recognize rankId; it only recognizes phyDevId. However, the application selects `devPhyId` through `aclrtSetDevice(mpiRankid % D)`, and this step binds mpiRankid into the process:

```
mpiRankid → aclrtSetDevice(mpiRankid % D) → Device table → devPhyId → RaGetDevEidInfoList(devPhyId)
```

The ranktable.json **does not participate** in device selection in this path (it is used only to filter entries where `deviceId == devPhyId` to determine the phyId parameter).

### Path B rankId Association (Direct)

The interface EID of the peer is obtained by looking up the ranktable with `GetPeer(myRank)`, where `myRank` is mpiRankid:

```
mpiRankid(=myRank) → GetPeer(myRank) → peers_[myRank] → entry in ranktable where rank_id==myRank → interface EID
```

The ranktable.json **entirely determines** which device EID corresponds to that rankId in this path.

### Consistency Requirement

Because both paths start from mpiRankid, the rank -> device formulas on both sides must be consistent:

```
Path A formula:  device = server.devices[mpiRankid % D]     (hardcoded in the application; cannot be changed)
Path B formula:  device = server.devices[logicDevId]         (determined by lines 450/451)
```

- **Line 450** `logicDevId = rankId % serverMeta.size()` -> consistent with Path A -> passes
- **Line 451** `logicDevId = deviceIdx` -> diverges from Path A -> fails

### Complete Mapping Chain Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Consistency Requirement                           │
│                                                                         │
│  The rank → device mapping must be consistent across all levels:        │
│                                                                         │
│  1. Application aclrtSetDevice:    logicDevId = mpiRankid % D_s         │
│  2. ranktable.json device_id: must equal the phyDevId the application   │
│     actually selects                                                    │
│  3. ranktable.json local_id:  must equal the logicDevId the application │
│     actually selects                                                    │
│  4. EIDs in ranktable.json: must equal the real EIDs corresponding to   │
│     that phyDevId                                                       │
│                                                                         │
│  Any inconsistency at any level causes CheckRankGraphAddrs to report    │
│  an error                                                               │
└─────────────────────────────────────────────────────────────────────────┘

Mapping chain:

  mpiRankid (assigned by MPI)
      │
      ├──→ [Application Layer] aclrtSetDevice(mpiRankid % D)
      │         │
      │         ▼
      │    Device table (logic_id) → physical_id → devPhyId
      │         │                              │
      │         │              ┌───────────────┘
      │         │              │
      │         │              ▼
      │         │    RaGetDevEidInfoList(devPhyId)
      │         │              │
      │         │              ▼
      │         │    localEidSet (actual hardware EIDs)
      │         │
      │    (mpiRankid is simultaneously passed as the rank parameter to hcomm)
      │         │
      │         ▼
      │    myRank = mpiRankid
      │         │
      │         ▼
      │    rankGraph->GetPeer(myRank)
      │         │
      │         ▼
      │    peers_[myRank] ← entry in ranktable.json where rank_id==myRank
      │         │
      │         ▼
      │    peer interface EID (EID claimed by ranktable)
      │         │
      └─────────┘
                │
                ▼
         localEidSet.count(interface EID) == 0 ?
              │
         no match → report error
```

---

## Change Analysis

### Modified Code

```
File: src/topo/topo_ascend_cluster_parser.cc
Function: AscendClusterTopoParser::CreateRankTableFile
Lines: 449-456

Before (line 450):
    for (uint32_t deviceIdx = 0; deviceIdx < serverMeta.size(); ++deviceIdx) {
        uint32_t logicDevId = rankId % serverMeta.size();   // ← old code
        PhyDeviceId phyDevId = serverMeta[logicDevId];
        ...
        rankId++;
    }

After (line 451):
    for (uint32_t deviceIdx = 0; deviceIdx < serverMeta.size(); ++deviceIdx) {
        uint32_t logicDevId = deviceIdx;                    // ← new code
        PhyDeviceId phyDevId = serverMeta[logicDevId];
        ...
        rankId++;
    }
```

### Impact of the Change

This change alters the `device_id` and `local_id` that correspond to each `rank_id` in `ranktable.json`:

| rank_id | Old: device_idx from `rankId % D` | New: `deviceIdx` | Same? |
|---------|-----------------------------------|-----------------|-------|
| 0 (rack0) | 0 | 0 | Same |
| 1 (rack1) | 1 | 0 | **Different** |
| 2 (rack1) | 0 | 1 | **Different** |

When every server has exactly 1 device or only a single server exists, the two formulas are equivalent (`rankId % D` within the first server is the same as `deviceIdx`), and the change does not cause a problem. However, when multiple servers exist and a non-first server has multiple devices, the two formulas differ.

### Setting of device_id and local_id in BuildRankEntry

```cpp
// Server form: topo_ascend_cluster_parser.cc:413-416
json AscendClusterTopoParser::BuildRankEntry(const Server &server, int srcDevPhyId, ...)
{
    json rankEntry;
    rankEntry["device_id"] = srcDevPhyId;    // ← intra-board physical id
    rankEntry["local_id"] = srcDevPhyId;     // ← same as device_id in the Server form
    rankEntry["rank_id"] = rankId;
    ...
}

// POD form: topo_ascend_cluster_parser.cc:952-953
rankEntry["device_id"] = srcDevPhyId;                                        // ← intra-board physical id
rankEntry["local_id"] = static_cast<int>(boardIdx) * devNumPerBoard + srcDevPhyId; // ← POD-global id
```

In the Server form, `logic_id = physical_id`, so `device_id` and `local_id` have the same numerical value. In the POD form, `device_id` remains the intra-board physical id, but `local_id` is the POD-global id (`board_id × devices per board + intra-board physical id`). The `BuildRankEntry` function itself has no issue; the real problem is that the `phyDevId` passed in comes from `serverMeta[logicDevId]`, and the computation of `logicDevId` (lines 450/451) determines which physical device is assigned to that rank.

---

## Fix

### Core Fix

**Restore the formula on line 450** so that the rank -> device mapping in the ranktable is consistent with the application `aclrtSetDevice(rank % D)`:

```cpp
uint32_t logicDevId = rankId % serverMeta.size();
```

### Post-Fix Data Flow Verification

Verify with the test topology (rack0=1 device, rack1=2 devices):

```
Step 1: Application selects device (aclrtSetDevice)
  rank 0: 0 % 1 = 0 → rack0 logicDevId=0 → phyDevId=0  (IP=192.1.0.8)
  rank 1: 1 % 2 = 1 → rack1 logicDevId=1 → phyDevId=1  (IP=192.2.0.18)
  rank 2: 2 % 2 = 0 → rack1 logicDevId=0 → phyDevId=0  (IP=192.2.0.1)

Step 2: ranktable.json generation (rankId % D)
  rank 0: device_id=0, local_id=0  (phyDev0)
  rank 1: device_id=1, local_id=1  (phyDev1)
  rank 2: device_id=0, local_id=0  (phyDev0)

Step 3: CheckRankGraphAddrs validation
  rank 1: devPhyId=1 → localEidSet={192.2.0.11, 192.2.0.18}
          peer(rank_id=1) → device_id=1 → interface EID={192.2.0.18}
          192.2.0.18 in localEidSet → match ✓
  rank 2: devPhyId=0 → localEidSet={192.2.0.1, 192.2.0.8}
          peer(rank_id=2) → device_id=0 → interface EID={192.2.0.1}
          192.2.0.1 in localEidSet → match ✓
```

All mapping chains are consistent; validation passes.

---

## Key Code Location Index

### This Project (CheckerL2_8620) Code

| File | Lines | Function/Description |
|------|------|---------|
| `src/topo/topo_ascend_cluster_parser.cc` | 424-477 | `CreateRankTableFile` - generates ranktable.json |
| `src/topo/topo_ascend_cluster_parser.cc` | 449-456 | **Change point** - logicDevId computation (line 450 old / line 451 new) |
| `src/topo/topo_ascend_cluster_parser.cc` | 409-422 | `BuildRankEntry` - sets device_id/local_id/rank_id |
| `src/topo/topo_ascend_cluster_parser.cc` | 200-272 | `InitDynamicModelData` - initializes Device/Rank tables |
| `src/topo/topo_ascend_cluster_parser.cc` | 202, 234 | Comments stating that the application uses `aclrtSetDevice(rank % deviceCount)` |
| `src/proxy/level2/aclrt_device_stub.cc` | 69-128 | `aclrtSetDevice` - stub implementation for application device selection |
| `src/proxy/level2/aclrt_device_stub.cc` | 87-88 | Looks up the Device table by `(serverKey, logic_id)` |
| `src/proxy/level2/aclrt_device_stub.cc` | 110-122 | Creates a Rank table record; prints logicDevId/phyDevId |
| `src/proxy/level2/hccl_comm_stub.cc` | 83-111 | `SimGetDeviceComm` - calls `aclrtSetDevice` |
| `src/proxy/level2/hccl_comm_stub.cc` | 130-131 | Calls `SimGetDeviceComm` in `HcclCommInitAll` |
| `src/proxy/level2/hccp_ccu_stub.cc` | 528-541 | `GetAllUsedEndPoint` - looks up EndPoint by phyDevId |
| `src/proxy/level2/hccp_ccu_stub.cc` | 554-587 | `RaGetDevEidInfoList` - returns the EID list by phyId |
| `src/proxy/level2/hccp_ccu_stub.cc` | 573 | `chipId = info.phyId` (has a TODO comment, but is unrelated to this issue) |

### hcomm Service Code

| File | Lines | Function/Description |
|------|------|---------|
| `communicator_impl.cc` | 1256-1269 | `InitCommonData` - sets myRank, devLogicId, devPhyId |
| `communicator_impl.cc` | 1291-1338 | `CheckRankGraphAddrs` - EID validation logic |
| `communicator_impl.cc` | 1306-1316 | Path A: filters by `deviceId==devPhyId` and calls `RaGetDevEidInfoList` |
| `communicator_impl.cc` | 1322-1328 | Path B: `GetPeer(myRank)` retrieves the interface EID and compares it |
| `communicator_impl.cc` | 1323 | ZHF-DEBUG log: `peer deviceId` vs `devPhyId` |
| `communicator_impl.cc` | 1353-1371 | `InitRankGraph` - calls bridge to build the rankGraph |
| `communicator_impl.cc` | 1364 | `bridge->buildFromString(ranktableM, topoPath, myRank, ...)` |
| `op_base_v2.cc` | 455-514 | `HcclCommInitClusterInfoConfigV2` - entry point function |
| `op_base_v2.cc` | 131-152 | `CreateCommConfig` - constructs CommParams (myRank=rank) |
| `op_base_v2.cc` | 311-322 | `ParseJsonAndCreateComm` - parses JSON and creates the communicator |
| `rank_graph_builder.cc` | 315-353 | `BuildFromRankTable` - iterates the ranktable to build peers_ |
| `rank_graph_builder.cc` | 323-328 | `peers_.emplace(rankId, peer)` - uses hcclRankid as the key |
| `rank_graph_builder.cc` | 22, 46 | The `Build` method accepts the `myRank` parameter |

### Application Sample Code

| File | Lines | Description |
|------|------|------|
| `hcomm/examples/01_communicators/02_one_device_per_process_rank_table/main.cc` | 101-105 | MPI initialization; retrieves procRank |
| same as above | 106, 111 | `devId = procRank`, `aclrtSetDevice(devId)` |
| same as above | 130 | `HcclCommInitClusterInfoConfig(rankTableFile, devId, ...)` |

### Key Log Files

| File | Key Lines | Description |
|------|--------|------|
| `hccl_vm_install/bin/log.txt` | 79, 83, 85 | rankId/logicDevId/phyDevId printed by `aclrtSetDevice` |
| same as above | 187, 219, 299 | devLogicId/devPhyId printed by `InitCommonData` |
| same as above | 1023, 1062, 1094 | ZHF-DEBUG: `peer deviceId` vs `devPhyId` |
| same as above | 1267, 1309 | `CheckRankGraphAddrs` error messages |

### ranktable.json Files

| File | Description |
|------|------|
| `hccl_vm_install/data/ranktable.json` | Generated by the current run (new code, has the problem) |
| `hccl_vm_install/archive/20260815_094326/data/ranktable.json` | Archived file generated by the old code (correct) |
| `hccl_vm_install/archive/20260815_094636/data/ranktable.json` | Archived file from another run |

---

## Ruled-Out Items

The following factors were analyzed and confirmed **not** to be the root cause of this issue:

### chipId = info.phyId in RaGetDevEidInfoList

```cpp
// hccp_ccu_stub.cc:573
infoList[idx].chipId = info.phyId; // todo: 单server, logic id与rank id相等，但多server此处有问题。
```

This location does have a latent risk (as noted by the TODO comment), but `CheckRankGraphAddrs` **does not read the chipId field at all**. The EID comparison matches only by the original IP/EID value (`localEidSet.count(interface->GetAddr().GetEid())`). Therefore, the chipId assignment issue is unrelated to this error.

### Setting of device_id and local_id in BuildRankEntry

```cpp
// Server form: topo_ascend_cluster_parser.cc:414-415
rankEntry["device_id"] = srcDevPhyId;
rankEntry["local_id"] = srcDevPhyId;

// POD form: topo_ascend_cluster_parser.cc:952-953
rankEntry["device_id"] = srcDevPhyId;
rankEntry["local_id"] = static_cast<int>(boardIdx) * devNumPerBoard + srcDevPhyId;
```

In the Server form, `logic_id = physical_id` (numerically equal), and setting `device_id` and `local_id` to the same value is correct. In the POD form, `local_id` is the POD-global id, which differs from `device_id` (the intra-board physical id), but this does not affect the root cause of this issue. The real problem is that the `srcDevPhyId` passed in (that is, `phyDevId`) comes from `serverMeta[logicDevId]`, and the computation of `logicDevId` (lines 450/451) is the root cause.

### Whether hcomm Filters Peers by Server

`CheckRankGraphAddrs` does not iterate over all peers and does not filter by server. It only retrieves the single local peer returned by `GetPeer(myRank)` and checks the interface EID of that peer. No cross-server peer comparison issue exists.

---

## Summary

| Dimension | Content |
|------|------|
| Problem nature | The two EID retrieval paths use different rankId -> device mapping formulas |
| Path A | Actual hardware EID; devPhyId is determined by the application `aclrtSetDevice(mpiRankid % D)`; does not go through the ranktable |
| Path B | EID claimed by the ranktable; retrieved from the ranktable through `GetPeer(myRank=mpiRankid)` |
| Conflict condition | When the server device count is greater than 1 and the server is not the first server, `rankId % D != deviceIdx` |
| Root cause | Line 451 changes `rankId % D` to `deviceIdx`, breaking the consistency between Path B and Path A |
| myRank attribution | mpiRankid (comes from `MPI_Comm_rank`, passed in through the rank parameter of `HcclCommInitClusterInfoConfig`) |
| Fix method | Restore line 450: `uint32_t logicDevId = rankId % serverMeta.size();` |
