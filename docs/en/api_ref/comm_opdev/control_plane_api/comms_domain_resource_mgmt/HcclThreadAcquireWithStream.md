# HcclThreadAcquireWithStream

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:36:52.366Z pushedAt=2026-09-30T02:30:49.150Z -->

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

Obtains a communication thread based on the communicator and Runtime stream handle, and allocates a specified number of synchronization resources (Notify) for the communication thread. For related concepts, see [Communication Operator Development Guide - Concurrency Model](../../../../comm_op_dev_guide/prog_models_concepts/concurrency_model.md).

## Function Prototype

```c
HcclResult HcclThreadAcquireWithStream(HcclComm comm, CommEngine engine, aclrtStream stream, uint32_t notifyNum, ThreadHandle *thread)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator handle.<br>For the definition of the HcclComm type, see [HcclComm](../../../comm_mgr_c/data_type_definition/HcclComm.md). |
| engine | Input | Communication engine type.<br>For the definition of the CommEngine type, see [CommEngine](../../datatype_definition/CommEngine.md). |
| stream | Input | Stream handle. |
| notifyNum | Input | Number of synchronization signals. |
| thread | Output | Thread handle.<br>For the definition of the ThreadHandle type, see [ThreadHandle](../../datatype_definition/ThreadHandle.md). |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

This API supports only the **COMM_ENGINE_CPU**, **COMM_ENGINE_CPU_TS**, and **COMM_ENGINE_CCU** communication engines.

## Example

```c
// Communicator handle.
HcclComm comm;
// Create a runtime stream.
aclrtStream stream;
aclrtCreateStream(&stream);
// Create threads and allocate Notify resources for the two threads.
ThreadHandle thread;
HcclResult ret = HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, stream, 2, &thread);
if (ret != HCCL_SUCCESS) {
    // Error handling.
}

// Data plane operations.
// ...

// Stream synchronization.
aclrtSynchronizeStream(stream);
```
