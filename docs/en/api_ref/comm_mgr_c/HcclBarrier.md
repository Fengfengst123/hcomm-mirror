# HcclBarrier

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:06:30.322Z pushedAt=2026-09-28T08:46:55.375Z -->

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
- Atlas training products: Supported
<!-- end id5 -->

## Description

Blocks the streams of all ranks in the specified communicator until all ranks have dispatched and executed this operation.

## Function Prototype

```c
HcclResult HcclBarrier(HcclComm comm, aclrtStream stream)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator where the collective communication operation is performed.<br>For the definition of the HcclComm type, see [HcclComm](./data_type_definition/HcclComm.md). |
| stream | Input | Stream used by this rank. |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

None

## Example

```c
HcclComm comm;
aclrtStream stream;
aclrtCreateStream(&stream);

// Submit the communication task to this stream, for example, HcclAllReduce.
// ...

// Block until all ranks have performed the Barrier operation.
HcclBarrier(comm, stream);
```
