# HcclCommMemReg

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:26:58.045Z pushedAt=2026-09-29T11:54:44.085Z -->

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

Registers the allocated memory with a communicator and obtains the corresponding registration handle.

## Function Prototype

```c
HcclResult HcclCommMemReg(HcclComm comm, const char *memTag, const CommMem *mem, HcclMemHandle *memHandle)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator handle.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| memTag | Input | Memory string tag. The maximum character length is **HCCL_RES_TAG_MAX_LEN**.<br>const uint32_t HCCL_RES_TAG_MAX_LEN = 255; |
| mem | Input | Memory information. For details about the CommMem type, see [CommMem](../../datatype_definition/CommMem.md). |
| memHandle | Output | Memory handle.<br>The HcclMemHandle type is defined as follows:<br>typedef void *HcclMemHandle; |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

- Within a communicator, only one block of memory can be registered for the same **memTag**.
- Within a communicator, registering the same **memTag** repeatedly returns the **HCCL_E_PARA** error and does not reuse the existing registered memory handle.
- Within a communicator, different **memTags** can be mapped to overlapping or identical memory regions.
- When `mem->type` is `COMM_MEM_TYPE_CCU`, it indicates that CCU resource space memory is registered, which is supported only by Ascend 950PR&950DT products. The registration process for CCU memory is the same as that for the device. For details, see [CommMemType](../../datatype_definition/CommMemType.md).

## Example

```c
HcclComm comm; // Communicator handle. Omitted here.
const char* memTag = "memTag"; // Memory tag.
void *deviceBuffer = nullptr; // Address of the allocated device memory (allocated through APIs such as aclrtMalloc).
CommMem memInfo; // Memory information.
memInfo.addr = deviceBuffer; // Configure the address of the allocated memory.
memInfo.size = 1024; // Configure the size of the allocated memory.
memInfo.type = COMM_MEM_TYPE_DEVICE; // Configure the type of the allocated memory.
HcclMemHandle memHandle; // Memory handle.
HcclCommMemReg(comm, memTag, &memInfo, &memHandle);
```
