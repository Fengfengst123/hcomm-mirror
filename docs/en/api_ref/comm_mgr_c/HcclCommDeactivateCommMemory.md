# HcclCommDeactivateCommMemory

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:07:55.505Z pushedAt=2026-09-28T08:53:47.187Z -->

> [!NOTE] Note
> This API is for trial use and may be changed later. It cannot be used in production environments.

## Supported Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Not supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Not supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- Atlas inference products: Not supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas training products: Not supported
<!-- end id5 -->

## Description

Deactivates the activated virtual memory. After deactivation, if the address is used for collective communication again, the zero-copy feature cannot be enabled.

## Function Prototype

```c
HcclResult HcclCommDeactivateCommMemory(HcclComm comm, void *virPtr)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | HCCL communicator. It is recommended to use the largest communicator in the server, that is, the communicator covering the maximum number of devices. |
| virPtr | Input | Start address of the virtual address to be deactivated, that is, the virtual memory address specified by the **virPtr** parameter of the [HcclCommActivateCommMemory](HcclCommActivateCommMemory.md) API.<br>Note that the specified virtual memory must have been successfully activated, and only the entire address block can be deactivated. |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

None

## Example

```c
// Assume that virPtr is the virtual memory address activated through HcclCommActivateCommMemory.

// Deactivate the activated virtual memory.
HcclCommDeactivateCommMemory(hcclComm, virPtr);

// The address can no longer be used for zero-copy communication.
// To reuse it, call HcclCommActivateCommMemory again to activate it.
```
