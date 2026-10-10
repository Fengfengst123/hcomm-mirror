# HcclEngineCtxCreate

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:30:09.611Z pushedAt=2026-09-30T01:47:28.146Z -->

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

Creates a communication engine context with a specific tag for a specified communicator and communication engine.

A communication engine context is a block of memory that can be used by the data plane of the communication engine to store information such as resource handles or parameters required for executing operators. Once created, it can be obtained and used repeatedly. By specifying a communicator and a communication engine type, a communication engine tag can index a communication engine context.

## Function Prototype

```c
HcclResult HcclEngineCtxCreate(HcclComm comm, const char *ctxTag, CommEngine engine, uint64_t size, void **ctx)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator handle.<br>For the definition of the HcclComm type, see [HcclComm](../../../comm_mgr_c/data_type_definition/HcclComm.md). |
| ctxTag | Input | Communication engine context tag. The maximum character length is **HCCL_RES_TAG_MAX_LEN**.<br>const uint32_t HCCL_RES_TAG_MAX_LEN = 255; |
| engine | Input | Communication engine type.<br>For the definition of CommEngine, see [CommEngine](../../datatype_definition/CommEngine.md). |
| size | Input | Size of the ctx memory, in bytes.<br>**size** cannot be **0**. |
| ctx | Output | Communication engine context. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

None

## Example

```c
// Communicator handle
HcclComm comm;
// Create a communication engine context of 16B.
uint64_t size = 16;
void *ctx = nullptr;
string ctxTag = "ctxTag";
CommEngine engine = COMM_ENGINE_CPU_TS;
HcclResult ret = HcclEngineCtxCreate(comm, ctxTag, engine, size, &ctx);
if (ret != HCCL_SUCCESS) {
    // Error handling
}
```
