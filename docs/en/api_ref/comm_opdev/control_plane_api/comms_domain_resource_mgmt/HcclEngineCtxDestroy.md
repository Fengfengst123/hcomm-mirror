# HcclEngineCtxDestroy

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:30:43.528Z pushedAt=2026-09-30T01:51:59.834Z -->

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

Destroys the communication engine context corresponding to a specified communicator and communication engine using a specific tag.

## Function Prototype

```c
HcclResult HcclEngineCtxDestroy(HcclComm comm, const char *ctxTag, CommEngine engine)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator handle.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| ctxTag | Input | Communication engine context tag (maximum character length: **HCCL_RES_TAG_MAX_LEN**). |
| engine | Input | Communication engine type. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

None

## Example

```c
HcclComm comm;
uint64_t size = 16;
void *ctx = nullptr;
string ctxTag = "ctxTag";
CommEngine engine = CommEngine::COMM_ENGINE_CPU_TS;
HcclResult ret = HcclEngineCtxCreate(comm, ctxTag, engine, size, &ctx);
ret = HcclEngineCtxDestroy(comm, ctxTag, engine);
```
