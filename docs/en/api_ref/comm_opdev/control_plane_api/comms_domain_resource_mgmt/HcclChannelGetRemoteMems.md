# HcclChannelGetRemoteMems

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:25:14.880Z pushedAt=2026-09-29T11:40:10.075Z -->

## Supported Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Not supported
<!-- end id3 -->
<!-- npu="910" id4 -->
- Atlas training products: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas inference products: Not supported
<!-- end id5 -->

## Description

Obtains the remote memory information exchanged in a communication channel. The returned remote memory list contains all remote memory (including the default **HcclBuffer** of the communicator), not only the memory registered by users.

## Function Prototype

```c
HcclResult HcclChannelGetRemoteMems(HcclComm comm, ChannelHandle channel, uint32_t *memNum, CommMem **remoteMems, char ***memTags)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator handle.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| channel | Input | Communication channel handle.<br>For details about the ChannelHandle type, see [ChannelHandle](../../datatype_definition/ChannelHandle.md). |
| memNum | Output | Number of memories. |
| remoteMems | Output | Remote memory list, which contains all remote memory (including the default **HcclBuffer** of the communicator).<br>For details about the CommMem type, see [CommMem](../../datatype_definition/CommMem.md). |
| memTags | Output | List of remote memory strings. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

None

## Example

```c
uint32_t channelNum = 1;
std::vector<HcclChannelDesc> channelDesc(channelNum);
HcclChannelDescInit(channelDesc.data(), channelNum);

// Omit the channelDesc configuration.

HcclComm comm;
CommEngine engine = CommEngine::COMM_ENGINE_AIV;
std::vector<ChannelHandle> channels(channelNum);
HcclChannelAcquire(comm, engine, channelDesc.data(), channelNum, channels.data());

uint32_t memNum = 0;
CommMem* remoteMems = nullptr;
char** memTags = nullptr;
HcclChannelGetRemoteMems(comm, channels[0], &memNum, &remoteMems, &memTags);
```
