# HcclGetRankId

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:24:05.210Z pushedAt=2026-09-29T01:15:54.588Z -->

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
<!-- npu="310p" id4 -->
- Atlas inference products: Supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas training products: Supported
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
| comm | Input | Communicator in which the collective communication operation is performed.<br>For the definition of the HcclComm type, see [HcclComm](./data_type_definition/HcclComm.md). |
| rank | Output | Output address pointer of the rank ID in the specified communicator. |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

None

## Example

```c
// Initialize the communicator.
HcclComm comm;
// Obtain the rank ID of the current device.
uint32_t rank;
HcclGetRankId(comm, &rank);
```
