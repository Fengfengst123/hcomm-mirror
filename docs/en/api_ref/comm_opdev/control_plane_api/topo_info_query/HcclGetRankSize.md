# HcclGetRankSize

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:39:59.211Z pushedAt=2026-09-30T02:41:15.653Z -->

## Supported Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Supported
<!-- end id3 -->
<!-- npu="910" id4 -->
- Atlas training products: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas inference products: Not supported
<!-- end id5 -->

## Description

Queries the number of ranks in the specified communicator.

## Function Prototype

```c
HcclResult HcclGetRankSize(HcclComm comm, uint32_t *rankSize)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator where the collective communication operation is performed.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| rankSize | Output | Number of ranks in the communicator. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

- **HcclComm** must be obtained through the communicator creation API and remain within its valid lifecycle. Passing an invalid pointer as the communicator input parameter is not allowed.

## Example

Take a communicator with 4 servers and 8 devices as an example. The total number of ranks is 32:

```c
uint32_t rankSize;
HcclComm comm;
HcclGetRankSize(comm, &rankSize);
// rankSize=32
```
