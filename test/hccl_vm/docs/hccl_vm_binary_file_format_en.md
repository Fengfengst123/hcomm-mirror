# HCCL VM Binary Stream File Format Specification

## Document Information

| Item       | Content                                |
|------------|----------------------------------------|
| Version    | v2.0                                   |
| Updated    | 2026-09-17                             |
| Use Case   | HCCL VM Simulation Data Export and Read |
| Author     | HCCL VM Team                           |

> **v2.0 Change Notes**: For the comprehensive data file (`*_hcclvm_syn_data.bin`), **the version remains 1 and the file header has no version changes**
> (this feature does not differentiate between old and new file compatibility). The `ChannelData` binary layout is **consistent with previous versions (no new fields added, 176 bytes)**,
> but the AICPU mode dump **semantics** have changed to support the multi-channel (`HCCL_UB_MULTI_CHANNEL_NUM`) feature:
> jetty precisely attributed to the (local eid, remote eid) link + **one aggregated record per eid pair**
> (`jettyNum` is the number of jetties for that eid pair, `jettyId[0..jettyNum)` carries all local jetties,
> and the reader must search for `task.jettyId` within the array, and must no longer assume `jettyNum` is always 1 or directly use `jettyId[0]`).
> Task/microcode/flag file formats remain unchanged.

---

## 1. Overview

The `DumpData` interface exports HCCL VM simulation runtime data into binary stream files, generating **3 types of files**:

| No.  | File Name Format                      | Magic Number | Purpose                                     |
|:----:|---------------------------------------|--------------|---------------------------------------------|
|  1   | `{dataId}_hcclvm_syn_data.bin`       | `0x48564D44` | Comprehensive data (model, channel, memory layout) |
|  2   | `{dataId}_hcclvm_instr_data.bin`     | `0x434D4349` | Microcode instruction data                  |
|  3   | `{dataId}_hcclvm_task_data.bin`      | `0x48565444` | Task metadata                               |

> **Note**: `{dataId}` is a unique identifier in timestamp format, e.g., `20250115_143052_3847`.

---

## 2. Common File Header

All binary files start with a **20-byte** file header.

### 2.1 File Header Structure

| Offset | Bytes | Type    | Field       | Description                    |
|:------:|:-----:|---------|-------------|--------------------------------|
|   0    |   4   | uint32  | magic       | Magic number, identifies file type |
|   4    |   2   | uint16  | version     | Version number (currently 1)   |
|   6    |   2   | uint16  | header_size | File header size (20 bytes)    |
|   8    |   4   | uint32  | flags       | Flags (reserved)               |
|   12   |   4   | uint32  | count       | Number of data entries         |
|   16   |   4   | uint32  | checksum    | Checksum (reserved)            |

### 2.2 C/C++ Structure Definition

```cpp
#pragma pack(push, 1)
struct FileHeader {
    uint32_t magic;        // Magic number
    uint16_t version;      // Version number
    uint16_t header_size;  // File header size
    uint32_t flags;        // Flags
    uint32_t count;        // Number of data entries
    uint32_t checksum;     // Checksum
};
#pragma pack(pop)
```

---

## 3. Comprehensive Data File (`*_hcclvm_syn_data.bin`)

### 3.1 Overall Structure

| No.  | Data Block                         | Size     | Description                                        |
|:----:|------------------------------------|----------|----------------------------------------------------|
|  1   | FileHeader                         | 20 Bytes | File header                                        |
|  2   | ModelInfo                          | Variable | Model information (includes ModelInfoCommInner, VDataDesTag, All2AllDataDesTag) |
|  3   | ChannelInfo (shared by CCU/AICPU)  | Variable | This data block is absent in AIV mode              |
|  4   | MemLayoutInfo                      | Variable | Memory layout information                          |

**op_expansion_mode Value Description:**

| Value | Mode   | Data Structure Used |
|:-----:|--------|---------------------|
|   0   | CCU    | ChannelInfo         |
|   1   | AICPU  | ChannelInfo (multi-channel semantics, see 3.3.3) |

### 3.2 ModelInfo Structure

#### 3.2.1 ModelInfoCommInner

**Size**: 36 bytes.

| Offset | Bytes | Type    | Field                | Description                      |
|:------:|:-----:|---------|----------------------|----------------------------------|
|   0    |   4   | uint32  | src_rank             | Source Rank ID                   |
|   4    |   4   | uint32  | dst_rank             | Destination Rank ID              |
|   8    |   4   | uint32  | root                 | Root node Rank                   |
|   12   |   4   | uint32  | rank_size            | Total number of Ranks            |
|   16   |   2   | uint16  | chip_type            | Chip type                        |
|   18   |   2   | uint16  | op_type              | Operation type                   |
|   20   |   2   | uint16  | reduce_op            | Reduce operation type            |
|   22   |   2   | uint16  | data_type            | Data type                        |
|   24   |   8   | uint64  | data_count           | Number of data elements          |
|   32   |   4   | uint32  | op_expansion_mode    | Expansion mode (see enum)        |
|   36   |   8   | uint64  | ccu0_resource_base_addr | Die0 CCU resource base address |
|   44   |   8   | uint64  | ccu1_resource_base_addr | Die1 CCU resource base address |

#### 3.2.2 VDataDesTag

**Purpose**: ReduceScatterV / AllGatherV operations.

**Size**: Variable length.

| Offset  | Bytes      | Type     | Field  | Description                    |
|:-------:|:----------:|----------|--------|--------------------------------|
|    0    |     2      | uint16   | dataType | Data type                    |
|    2    |     4      | uint32   | count  | Number of Ranks                |
|    6    | 8 x count  | uint64[] | displs | Data offset per Rank           |
|  6+8xn  | 8 x count  | uint64[] | counts | Data size per Rank             |

#### 3.2.3 All2AllDataDesTag

**Purpose**: All2All / All2AllV operations.

**Size**: Variable length.

| Offset | Bytes      | Type     | Field           | Description                      |
|:------:|:----------:|----------|-----------------|----------------------------------|
|   0    |     2      | uint16   | sendType        | Send data type                   |
|   2    |     2      | uint16   | recvType        | Receive data type                |
|   4    |     8      | uint64   | sendCount       | Send data count                  |
|   12   |     8      | uint64   | recvCount       | Receive data count               |
|   20   |     4      | uint32   | count           | Matrix size (= rankSize^2)       |
|   24   | 8 x count  | uint64[] | sendCountMatrix | Send matrix                      |

### 3.3 ChannelInfo Structure

**Applicable conditions**: `op_expansion_mode = 0` (CCU mode) or `1` (AICPU mode) -- both modes use the fixed-length ChannelData structure; not written in AIV mode.

#### 3.3.1 ChannelInfo Header

| Offset | Bytes | Type          | Field | Description           |
|:------:|:-----:|---------------|-------|-----------------------|
|   0    |   4   | uint32        | count | Number of channels    |
|   4    | Variable | ChannelData[] | data | Channel data array  |

#### 3.3.2 ChannelData

**Size**: 176 bytes (layout consistent with previous versions).

| Offset | Bytes | Type       | Field       | Description                                                      |
|:------:|:-----:|------------|-------------|------------------------------------------------------------------|
|   0    |   2   | uint16     | channelId   | CCU mode: Channel ID; AICPU mode: EndPointPair primary key (eid pair identifier) |
|   2    |   1   | uint8      | srcDieId    | Source Die ID                                                    |
|   3    |   1   | uint8      | dstDieId    | Destination Die ID                                               |
|   4    |   4   | uint32     | srcRank     | Source Rank ID                                                   |
|   8    |   4   | uint32     | dstRank     | Destination Rank ID                                              |
|   12   |  16   | uint8[16]  | leid        | Local EID                                                        |
|   28   |  16   | uint8[16]  | reid        | Remote EID                                                       |
|   44   |   2   | uint16     | protocol    | Protocol type                                                    |
|   46   |   2   | uint16     | jettyNum    | Number of Jetties                                                |
|   48   | 128   | uint32[32] | jettyId     | Jetty ID array                                                   |

#### 3.3.3 Multi-Channel Semantics (AICPU Mode)

Under the `HCCL_UB_MULTI_CHANNEL_NUM=N` feature, there are N parallel channels between the same pair of ranks (same pair of eids).
AICPU mode dump rules:

1. **One aggregated ChannelData record per eid pair**: N channels with the same `channelId` (eid pair) output **1 record**,
   `jettyNum=N`, `jettyId[0..jettyNum)` contains **all local (send-side) jetties** for that eid pair,
   ordered by **local jetty creation order** (array index is the channel index within the pair);
2. The reader searches for `task.jettyId` from the task metadata **within the `jettyId[0..jettyNum)` array**
   to determine which jetty the task uses to send from the local eid (`leid`) to the remote eid (`reid`);
3. `jettyId[0..jettyNum)` is **precisely attributed** to that eid pair (backfilled by the RA layer `RaCtxQpImport`),
   and does not contain jetties on the same local eid that go to other remote endpoints.

CCU mode maintains the existing semantics of "one record per channel, with `jettyId[]` carrying the jetty list for that channel."

### 3.4 JettyInfo Structure (Unused, Retained for Historical Reference)

`JettyData` was an independent structure for AICPU mode in early designs, **not used in the current implementation**: AICPU mode actually also writes
ChannelData (see 3.3). This section is retained only as a historical format reference; readers should not parse AICPU data as JettyData.

#### 3.4.1 JettyInfo Header

| Offset | Bytes | Type       | Field | Description          |
|:------:|:-----:|------------|-------|----------------------|
|   0    |   4   | uint32     | count | Number of Jetties    |
|   4    | Variable | JettyData[] | data | Jetty data array  |

#### 3.4.2 JettyData

**Size**: 48 bytes.

| Offset | Bytes | Type      | Field    | Description       |
|:------:|:-----:|-----------|----------|-------------------|
|   0    |   4   | uint32    | jettyId  | Jetty ID          |
|   4    |   1   | uint8     | srcDieId | Source Die ID     |
|   5    |   1   | uint8     | dstDieId | Destination Die ID|
|   6    |   2   | uint16    | protocol | Protocol type     |
|   8    |   4   | uint32    | srcRank  | Source Rank ID    |
|   12   |   4   | uint32    | dstRank  | Destination Rank ID|
|   16   |  16   | uint8[16] | leid     | Local EID         |
|   32   |  16   | uint8[16] | reid     | Remote EID        |

### 3.5 MemLayoutInfo Structure

#### 3.5.1 MemLayoutInfo Header

| Offset | Bytes | Type            | Field | Description           |
|:------:|:-----:|-----------------|-------|-----------------------|
|   0    |   4   | uint32          | count | Number of memory blocks |
|   4    | Variable | MemLayoutData[] | data | Memory layout array |

#### 3.5.2 MemLayoutData

**Size**: 32 bytes.

| Offset | Bytes | Type   | Field        | Description                      |
|:------:|:-----:|--------|--------------|----------------------------------|
|   0    |   4   | uint32 | rank_id      | Rank ID                          |
|   4    |   1   | uint8  | buffer_type  | Buffer type (see enum)           |
|   5    |   1   | uint8  | reserved     | Reserved                         |
|   6    |   8   | uint64 | start_addr   | Physical start address           |
|   14   |   8   | uint64 | size         | Block size                       |
|   22   |   8   | uint64 | global_offset| Total offset for this type       |

---

## 4. Microcode Instruction File (`*_hcclvm_instr_data.bin`)

### 4.1 Overall Structure

| No.  | Data Block                    | Size     | Description                               |
|:----:|-------------------------------|----------|-------------------------------------------|
|  1   | FileHeader                    | 20 Bytes | File header                               |
|  2   | MicrocodeInstrInner x count   | Variable | Microcode instruction data (includes Desc and Instr) |

### 4.2 MicrocodeInstrDesc

**Size**: 8 bytes.

| Offset | Bytes | Type   | Field    | Description          |
|:------:|:-----:|--------|----------|----------------------|
|   0    |   4   | uint32 | rank_id  | Rank ID              |
|   4    |   1   | uint8  | die_id   | Die ID               |
|   5    |   1   | uint8  | reserved | Reserved             |
|   6    |   2   | uint16 | count    | Number of instructions |

### 4.3 Microcode Instruction Data

Each MicrocodeInstrInner contains:

| Data Block               | Size              | Description             |
|--------------------------|-------------------|-------------------------|
| MicrocodeInstrDesc       | 8 Bytes           | Instruction descriptor  |
| CcuInstr[desc.count]     | 32 x count Bytes  | Microcode instruction array |

---

## 5. Task Metadata File (`*_hcclvm_task_data.bin`)

### 5.1 Overall Structure

| No.  | Data Block                 | Size     | Description   |
|:----:|----------------------------|----------|---------------|
|  1   | FileHeader                 | 20 Bytes | File header   |
|  2   | HcclTaskMetaData x count   | Variable | Task list     |

### 5.2 HcclTaskMetaData Structure

**Size**: Approximately 136 bytes (including union)

| Offset | Bytes | Type            | Field    | Description                          |
|:------:|:-----:|-----------------|----------|--------------------------------------|
|   0    |   1   | int8            | taskType | Task type (see enum)                 |
|   1    |   2   | uint16          | commId   | Communication domain ID              |
|   3    |   4   | uint32          | rankId   | Rank ID                              |
|   7    |   8   | uint64          | streamId | Stream ID                            |
|   15   |   4   | uint32          | jettyId  | Jetty ID                             |
|   19   | Variable | union        | taskData | Task data (parsed based on type)     |

### 5.3 taskData Union

#### 5.3.1 TransMemTask

**Applicable task types**: `MEM_CPY (3)`.

**Size**: 33 bytes.

| Offset | Bytes | Type   | Field     | Description          |
|:------:|:-----:|--------|-----------|----------------------|
|   0    |   4   | uint32 | srcRankId | Source Rank ID       |
|   4    |   8   | uint64 | srcOffset | Source offset address |
|   12   |   4   | uint32 | dstRankId | Destination Rank ID  |
|   16   |   8   | uint64 | dstOffset | Destination offset address |
|   24   |   8   | uint64 | len       | Data length          |
|   32   |   1   | uint8  | protocol  | Protocol type        |

#### 5.3.2 ReduceTask

**Applicable task types**: `REDUCE (2)`.

**Size**: 35 bytes.

| Offset | Bytes | Type   | Field     | Description              |
|:------:|:-----:|--------|-----------|--------------------------|
|   0    |   4   | uint32 | srcRankId | Source Rank ID           |
|   4    |   8   | uint64 | srcOffset | Source offset address    |
|   12   |   4   | uint32 | dstRankId | Destination Rank ID      |
|   16   |   8   | uint64 | dstOffset | Destination offset address |
|   24   |   8   | uint64 | dataCount | Number of data elements  |
|   32   |   1   | uint8  | dataType  | Data type                |
|   33   |   1   | uint8  | reduceOp  | Reduce operation type    |
|   34   |   1   | uint8  | protocol  | Protocol type            |

#### 5.3.3 NotifyTask

**Applicable task types**: `NOTIFY_WAIT (0)` / `NOTIFY_RECORD (1)`.

**Size**: 18 bytes.

| Offset | Bytes | Type   | Field       | Description          |
|:------:|:-----:|--------|-------------|----------------------|
|   0    |   4   | uint32 | srcRankId   | Source Rank ID       |
|   4    |   8   | uint64 | notifyId    | Notification ID      |
|   12   |   4   | uint32 | dstRankId   | Destination Rank ID  |
|   16   |   1   | uint8  | notifyCount | Notification count   |
|   17   |   1   | uint8  | protocol    | Protocol type        |

---

## 6. Enum Definitions

### 6.1 Task Type (HccLTaskMetaType)

| Value | Name          | Description       |
|:-----:|---------------|-------------------|
|   0   | NOTIFY_WAIT   | Wait for notification |
|   1   | NOTIFY_RECORD | Record notification |
|   2   | REDUCE        | Reduce operation  |
|   3   | MEM_CPY       | Memory copy       |
|   4   | CCU_GRAPH     | CCU graph execution |
|   5   | AIV_GRAPH     | AIV graph execution |
|   6   | EVENT_WAIT    | Event wait        |
|   7   | EVENT_RECORD  | Event record      |

### 6.2 Protocol Type (ProtocolType)

| Value | Name    | Description                      |
|:-----:|---------|----------------------------------|
|   0   | HCCS    | High-speed inter-chip communication |
|   1   | ROCE    | RDMA over Converged Ethernet     |
|   2   | PCIE    | PCIe communication               |
|   3   | SIO     | Socket I/O                       |
|   4   | UBC_CTP | UBC CTP protocol                 |
|   5   | UBC_TP  | UBC TP protocol                  |
|   6   | UB_MEM  | UB memory protocol               |

### 6.3 Buffer Type (BufferType)

| Value | Name     | Description     |
|:-----:|----------|-----------------|
|   0   | INPUT    | Input buffer    |
|   1   | OUTPUT   | Output buffer   |
|   2   | CCL      | Communication buffer |
|   3   | RESERVED | Reserved        |

### 6.4 Expansion Mode (SimOpExpansionMode)

| Value | Name                            | Description                        |
|:-----:|---------------------------------|-------------------------------------|
|   0   | SIM_OP_EXPANSION_MODE_CCU       | CCU mode, uses ChannelInfo          |
|   1   | SIM_OP_EXPANSION_MODE_AICPU     | AICPU mode, uses ChannelInfo        |

---

## 7. C/C++ Read Example

### 7.1 Structure Definitions

```cpp
#include <cstdint>
#include <vector>
#include <cstring>

#pragma pack(push, 1)

// ============================================================================
// File Header
// ============================================================================
struct FileHeader {
    uint32_t magic;
    uint16_t version;
    uint16_t header_size;
    uint32_t flags;
    uint32_t count;
    uint32_t checksum;
};

// ============================================================================
// Model Information
// ============================================================================
struct ModelInfoCommInner {
    uint32_t src_rank;
    uint32_t dst_rank;
    uint32_t root;
    uint32_t rank_size;
    uint16_t chip_type;
    uint16_t op_type;
    uint16_t reduce_op;
    uint16_t data_type;
    uint64_t data_count;
    uint32_t op_expansion_mode;
    uint64_t ccu0_resource_base_addr;
    uint64_t ccu1_resource_base_addr;
};

// ============================================================================
// Channel Data (shared by CCU / AICPU modes)
// ============================================================================
struct ChannelData {
    uint16_t channelId;      // CCU: channel ID; AICPU: EndPointPair primary key (eid pair identifier)
    uint8_t  srcDieId;
    uint8_t  dstDieId;
    uint32_t srcRank;
    uint32_t dstRank;
    uint8_t  leid[16];
    uint8_t  reid[16];
    uint16_t protocol;
    uint16_t jettyNum;       // AICPU: number of local jetties for this eid pair (may be >1 for multi-channel); CCU: number of jetties for this channel
    uint32_t jettyId[32];    // AICPU: all local (send) jetties for this eid pair, ordered by creation order, task.jettyId searched within array
};

// ============================================================================
// Jetty Data (unused historical structure; AICPU actually uses ChannelData, see above)
// ============================================================================
struct JettyData {
    uint32_t jettyId;
    uint8_t  srcDieId;
    uint8_t  dstDieId;
    uint16_t protocol;
    uint32_t srcRank;
    uint32_t dstRank;
    uint8_t  leid[16];
    uint8_t  reid[16];
};

// ============================================================================
// Memory Layout
// ============================================================================
struct MemLayoutData {
    uint32_t rank_id;
    uint8_t  buffer_type;
    uint8_t  reserved;
    uint64_t start_addr;
    uint64_t size;
    uint64_t global_offset;
};

// ============================================================================
// Task Type Enum
// ============================================================================
enum class HccLTaskMetaType : int8_t {
    NOTIFY_WAIT   = 0,
    NOTIFY_RECORD = 1,
    REDUCE        = 2,
    MEM_CPY       = 3,
    CCU_GRAPH     = 4,
    AIV_GRAPH     = 5,
    EVENT_WAIT    = 6,
    EVENT_RECORD  = 7
};

// ============================================================================
// Task Data Union
// ============================================================================
struct TransMemTask {
    uint32_t srcRankId;
    uint64_t srcOffset;
    uint32_t dstRankId;
    uint64_t dstOffset;
    uint64_t len;
    uint8_t  protocol;
};

struct ReduceTask {
    uint32_t srcRankId;
    uint64_t srcOffset;
    uint32_t dstRankId;
    uint64_t dstOffset;
    uint64_t dataCount;
    uint8_t  dataType;
    uint8_t  reduceOp;
    uint8_t  protocol;
};

struct NotifyTask {
    uint32_t srcRankId;
    uint64_t notifyId;
    uint32_t dstRankId;
    uint8_t  notifyCount;
    uint8_t  protocol;
};

// ============================================================================
// Task Metadata
// ============================================================================
struct HcclTaskMetaData {
    HccLTaskMetaType taskType;
    uint16_t commId;
    uint32_t rankId;
    uint64_t streamId;
    uint32_t jettyId;
    union {
        TransMemTask transMem;
        ReduceTask   reduce;
        NotifyTask   notify;
    } taskData;
};

#pragma pack(pop)
```

### 7.2 Reading Comprehensive Data File Example

```cpp
#include <cstdio>
#include <iostream>

// Magic number definitions
constexpr uint32_t HCCLVM_SYN_FILE_MAGIC = 0x48564D44;

bool ReadSynthesisData(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }

    // ===== Step 1: Read file header =====
    FileHeader header;
    if (fread(&header, sizeof(FileHeader), 1, fp) != 1) {
        std::cerr << "Failed to read header" << std::endl;
        fclose(fp);
        return false;
    }

    // Verify magic number
    if (header.magic != HCCLVM_SYN_FILE_MAGIC) {
        std::cerr << "Invalid magic number: 0x" << std::hex << header.magic << std::endl;
        fclose(fp);
        return false;
    }

    std::cout << "File version: " << header.version << std::endl;
    std::cout << "Data count: " << header.count << std::endl;

    // ===== Step 2: Read model information =====
    ModelInfoCommInner modelComm;
    fread(&modelComm, sizeof(ModelInfoCommInner), 1, fp);
    
    std::cout << "\n[Model Info]" << std::endl;
    std::cout << "  Rank Size: " << modelComm.rank_size << std::endl;
    std::cout << "  Op Type: " << modelComm.op_type << std::endl;
    std::cout << "  Data Count: " << modelComm.data_count << std::endl;
    std::cout << "  Expansion Mode: " << modelComm.op_expansion_mode << std::endl;

    // ===== Step 3: Read VDataDesTag =====
    uint16_t vDataType;
    uint32_t vDataCount;
    fread(&vDataType, sizeof(uint16_t), 1, fp);
    fread(&vDataCount, sizeof(uint32_t), 1, fp);
    
    if (vDataCount > 0) {
        std::vector<uint64_t> displs(vDataCount);
        std::vector<uint64_t> counts(vDataCount);
        fread(displs.data(), sizeof(uint64_t), vDataCount, fp);
        fread(counts.data(), sizeof(uint64_t), vDataCount, fp);
        std::cout << "\n[VDataDes] Count: " << vDataCount << std::endl;
    }

    // ===== Step 4: Read All2AllDataDesTag =====
    uint16_t sendType, recvType;
    uint64_t sendCount, recvCount;
    uint32_t matrixCount;
    
    fread(&sendType, sizeof(uint16_t), 1, fp);
    fread(&recvType, sizeof(uint16_t), 1, fp);
    fread(&sendCount, sizeof(uint64_t), 1, fp);
    fread(&recvCount, sizeof(uint64_t), 1, fp);
    fread(&matrixCount, sizeof(uint32_t), 1, fp);
    
    if (matrixCount > 0) {
        std::vector<uint64_t> sendCountMatrix(matrixCount);
        fread(sendCountMatrix.data(), sizeof(uint64_t), matrixCount, fp);
        std::cout << "\n[All2All] Matrix Count: " << matrixCount << std::endl;
    }

    // ===== Step 5: Read ChannelInfo (both CCU and AICPU modes use fixed-length ChannelData structure; absent in AIV mode) =====
    if (modelComm.op_expansion_mode != 2) {
        uint32_t channelCount;
        fread(&channelCount, sizeof(uint32_t), 1, fp);
        
        std::vector<ChannelData> channels(channelCount);
        fread(channels.data(), sizeof(ChannelData), channelCount, fp);
        
        std::cout << "\n[ChannelInfo] Count: " << channelCount << std::endl;
        for (size_t i = 0; i < channels.size() && i < 3; ++i) {
            std::cout << "  Channel[" << i << "]: ID=" << channels[i].channelId
                      << ", srcRank=" << channels[i].srcRank
                      << ", dstRank=" << channels[i].dstRank
                      << ", jettyNum=" << channels[i].jettyNum;
            for (uint16_t j = 0; j < channels[i].jettyNum; ++j) {
                std::cout << ", jetty[" << j << "]=" << channels[i].jettyId[j];
            }
            std::cout << std::endl;
        }
    }

    // ===== Step 6: Read memory layout =====
    uint32_t memCount;
    fread(&memCount, sizeof(uint32_t), 1, fp);
    
    std::vector<MemLayoutData> memLayouts(memCount);
    fread(memLayouts.data(), sizeof(MemLayoutData), memCount, fp);
    
    std::cout << "\n[MemLayout] Count: " << memCount << std::endl;
    for (size_t i = 0; i < memLayouts.size() && i < 3; ++i) {
        std::cout << "  Mem[" << i << "]: rank=" << memLayouts[i].rank_id
                  << ", type=" << (int)memLayouts[i].buffer_type
                  << ", size=" << memLayouts[i].size << std::endl;
    }

    fclose(fp);
    std::cout << "\nRead synthesis data success!" << std::endl;
    return true;
}
```

### 7.3 Reading Task Metadata File Example

```cpp
constexpr uint32_t HCCLVM_TASK_FILE_MAGIC = 0x48565444;

bool ReadTaskMetaData(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }

    // ===== Step 1: Read file header =====
    FileHeader header;
    fread(&header, sizeof(FileHeader), 1, fp);

    if (header.magic != HCCLVM_TASK_FILE_MAGIC) {
        std::cerr << "Invalid magic number" << std::endl;
        fclose(fp);
        return false;
    }

    std::cout << "Task count: " << header.count << std::endl;

    // ===== Step 2: Read all tasks =====
    std::vector<HcclTaskMetaData> tasks(header.count);
    fread(tasks.data(), sizeof(HcclTaskMetaData), header.count, fp);

    // ===== Step 3: Parse and print tasks =====
    std::cout << "\n[Task List]" << std::endl;
    for (size_t i = 0; i < tasks.size(); ++i) {
        const auto& task = tasks[i];
        std::cout << "Task[" << i << "] ";
        
        switch (task.taskType) {
            case HccLTaskMetaType::MEM_CPY:
                std::cout << "MEM_CPY: "
                          << "src=" << task.taskData.transMem.srcRankId
                          << ", dst=" << task.taskData.transMem.dstRankId
                          << ", len=" << task.taskData.transMem.len;
                break;
                
            case HccLTaskMetaType::REDUCE:
                std::cout << "REDUCE: "
                          << "src=" << task.taskData.reduce.srcRankId
                          << ", dst=" << task.taskData.reduce.dstRankId
                          << ", count=" << task.taskData.reduce.dataCount;
                break;
                
            case HccLTaskMetaType::NOTIFY_WAIT:
                std::cout << "NOTIFY_WAIT: "
                          << "notifyId=" << task.taskData.notify.notifyId;
                break;
                
            case HccLTaskMetaType::NOTIFY_RECORD:
                std::cout << "NOTIFY_RECORD: "
                          << "notifyId=" << task.taskData.notify.notifyId;
                break;
                
            default:
                std::cout << "UNKNOWN: type=" << static_cast<int>(task.taskType);
                break;
        }
        std::cout << std::endl;
    }

    fclose(fp);
    std::cout << "\nRead task meta data success!" << std::endl;
    return true;
}
```

---

## 8. Notes

| No.  | Note                | Description                                                         |
|:----:|---------------------|---------------------------------------------------------------------|
|  1   | Byte Order          | All multi-byte fields use **Little-Endian** byte order              |
|  2   | Memory Alignment    | Structures use `#pragma pack(1)` for **1-byte alignment**           |
|  3   | Variable-Length Fields | Read the `count` field first, then read the corresponding number of data entries based on the count value |
|  4   | Mode Detection      | Both CCU and AICPU modes read ChannelInfo (fixed-length ChannelData structure); this section is absent in AIV mode; JettyData is an unused historical structure |
|  5   | Union Parsing        | The `taskData` in task metadata must be parsed with the correct structure based on `taskType` |
|  6   | Magic Number Verification | Always verify the magic number when reading a file to ensure the correct file type |

---

## Appendix A: Magic Number Quick Reference

| File Type      | Magic Number | ASCII  | Description               |
|----------------|--------------|--------|---------------------------|
| Comprehensive Data | `0x48564D44` | "HVMD" | Hccl VM Data             |
| Microcode Instructions | `0x434D4349` | "CMCI" | CCU Microcode Instruction |
| Task Metadata  | `0x48565444` | "HVTM" | Hccl VM Task Metadata     |

---

## Appendix B: File Reading Flow

**Step 1**: Open the binary file.

**Step 2**: Read FileHeader (20 Bytes)

**Step 3**: Verify magic number

- Magic number mismatch -> Return error, close file.
- Magic number matches -> Continue.

**Step 4**: Read data content based on count.

**Step 5**: Close the file.

---

## Appendix C: Contact Information

For any questions, please contact HCCL VM Team.

---

## End of Document
