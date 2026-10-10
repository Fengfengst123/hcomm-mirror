# HcclCommUnsetMemoryRange

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:17:05.378Z pushedAt=2026-09-28T11:32:11.206Z -->

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

Notifies the HCCL communicator to stop using the reserved virtual memory.

## Function Prototype

```c
HcclResult HcclCommUnsetMemoryRange(HcclComm comm, void *baseVirPtr)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | HCCL communicator. It is recommended to use the largest communicator in the server, that is, the communicator covering the maximum number of devices. |
| baseVirPtr | Input | Base address of the reserved virtual memory.<br>The specified base address must have been successfully set by [HcclCommSetMemoryRange](HcclCommSetMemoryRange.md); otherwise, an error is reported. |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

If activated memory still exists in this virtual address space, this API fails to execute.

## Example

```c
// Assume that the virtual memory range has been set by HcclCommSetMemoryRange.
// baseVirPtr is the base address of the virtual memory passed to HcclCommSetMemoryRange.

// Ensure that all activated memory within this address range has been deactivated.
// HcclCommDeactivateCommMemory(hcclComm, activatedVirPtr);

// Cancel the reserved virtual memory.
HcclCommUnsetMemoryRange(hcclComm, baseVirPtr);
```
