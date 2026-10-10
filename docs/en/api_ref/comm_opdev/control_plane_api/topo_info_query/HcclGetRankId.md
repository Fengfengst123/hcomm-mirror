# HcclGetRankId

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:39:51.460Z pushedAt=2026-09-30T02:40:08.405Z -->

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

Obtains the rank ID of a device in a specified communicator.

## Function Prototype

```c
HcclResult HcclGetRankId(HcclComm comm, uint32_t *rank)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator where the collective communication operation is performed.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| rank | Output | Output address pointer of the rank ID in the collective communicator. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

None

## Example

```c
uint32_t rank;
HcclComm comm;
HcclGetRankId(comm, &rank);
```
