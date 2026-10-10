# HcclCommResume

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:13:45.470Z pushedAt=2026-09-28T10:44:10.548Z -->

> [!NOTE] Note
> This API is reserved and may change in the future. It is not intended for developer use.

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
- Atlas inference products: Not supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas training products: Not supported
<!-- end id5 -->

## Description

Restores the state of a communicator.

If a developer suspends a communicator by calling [HcclCommSuspend](HcclCommSuspend.md) or the **aclrtDeviceTaskAbort** API provided by ACL, this API must be called after fault recovery to restore the communicator to the normal state.

## Function Prototype

```c
HcclResult HcclCommResume(HcclComm comm)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator to be restored from the suspended state to the normal state.<br>For the definition of the HcclComm type, see [HcclComm](./data_type_definition/HcclComm.md). |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

- Before calling this API, call the **aclrtDeviceTaskAbort** API provided by ACL to stop task execution on the current device.
- Before calling this API to restore the communicator state, perform a cluster synchronization operation.

## Example

```c
uint32_t rankSize = 8;
uint32_t deviceId = 0;
// Generate the rank identifier information of the root node.
HcclRootInfo rootInfo;
HCCLCHECK(HcclGetRootInfo(&rootInfo));
// Initialize the communicator.
HcclComm hcclComm;
HCCLCHECK(HcclCommInitRootInfo(rankSize, &rootInfo, deviceId, &hcclComm));
// Assume that the communicator has been suspended by the HcclCommSuspend API or the aclrtDeviceTaskAbort API provided by acl. Restore the communicator.
HCCLCHECK(HcclCommResume(hcclComm));
// Destroy the communicator.
HCCLCHECK(HcclCommDestroy(hcclComm));
```
