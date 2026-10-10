# HcclEngineCtxGet

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:31:10.122Z pushedAt=2026-09-30T01:54:07.880Z -->

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

Specifies a communicator and a communication engine, and obtains the corresponding communication engine context by using the communication engine context tag.

## Function Prototype

```c
HcclResult HcclEngineCtxGet(HcclComm comm, const char *ctxTag, CommEngine engine, void **ctx, uint64_t *size)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator handle.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| ctxTag | Input | Communication engine context tag. The maximum character length is **HCCL_RES_TAG_MAX_LEN**.<br>const uint32_t HCCL_RES_TAG_MAX_LEN = 255; |
| engine | Input | Communication engine type. |
| ctx | Output | Communication engine context handle. |
| size | Output | Memory size corresponding to the communication engine context. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

None

## Example

```c
HcclComm comm;
uint64_t size = 0;
void *ctx = nullptr;
const char *ctxTag = "ctxTag";
CommEngine engine = CommEngine::COMM_ENGINE_CPU_TS;
HcclResult ret = HcclEngineCtxGet(comm, ctxTag, engine, &ctx, &size);
```
