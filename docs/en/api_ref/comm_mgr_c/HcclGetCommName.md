# HcclGetCommName

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:22:14.919Z pushedAt=2026-09-29T01:11:16.594Z -->

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

Obtains the name of the communicator where the specified communication operation is performed.

## Function Prototype

```c
HcclResult HcclGetCommName(HcclComm comm, char* commName)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| **comm** | Input | Communicator where the collective communication operation is performed.<br>For the definition of the HcclComm type, see [HcclComm](./data_type_definition/HcclComm.md). |
| **commName** | Output | Obtained communicator name.<br>Type: char*, with a maximum length of 128. |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

None

## Example

```c
// Initialize the communicator.
HcclComm comm;
// Query the communicator name.
char commName[128] = {0};
HcclResult ret = HcclGetCommName(comm, commName);
// Handle errors.
if (ret != HCCL_SUCCESS) {
}
```
