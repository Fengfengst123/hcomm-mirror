# HcommWriteReduceOnThread

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T08:35:31.908Z pushedAt=2026-10-08T11:19:35.004Z -->

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

Writes data to the specified memory on the channel, performs the reduceOp operation on the memory data of length **count*sizeof(dataType)** in **src** and the memory data of the same length pointed to by **dst**, and outputs the result to **dst**. The API caller is the node where **src** resides.

## Function Prototype

```c
int32_t HcommWriteReduceOnThread(ThreadHandle thread, ChannelHandle channel, void *dst, const void *src, uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | -------------------------------------- |
| thread | Input | Communication thread handle, which is the thread obtained through the [HcclThreadAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclThreadAcquire.md) API.<br>For the definition of the ThreadHandle type, see [ThreadHandle](../../../datatype_definition/ThreadHandle.md). |
| channel | Input | Communication channel handle, which is the channel obtained through the [HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md) API.<br>For the definition of the ChannelHandle type, see [ChannelHandle](../../../datatype_definition/ChannelHandle.md). |
| dst | Output | Destination memory address, which is the memory obtained through [HcclGetHcclBuffer](../../../control_plane_api/comms_domain_resource_mgmt/HcclGetHcclBuffer.md) or [HcclChannelGetHcclBuffer](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelGetHcclBuffer.md). |
| src | Input | Source memory address, which is the memory obtained through [HcclGetHcclBuffer](../../../control_plane_api/comms_domain_resource_mgmt/HcclGetHcclBuffer.md) or [HcclChannelGetHcclBuffer](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelGetHcclBuffer.md). |
| count | Input | Number of elements. |
| dataType | Input | Data type.<br>For the definition of the HcommDataType type, see [HcommDataType](../../../datatype_definition/HcommDataType.md).<br>Different product models support different data types. For details, see [dataType Description](#datatype-description).|
| reduceOp | Input | Reduce operation type. Supported types: sum, max, min.<br>For the definition of the HcommReduceOp type, see [HcommReduceOp](../../../datatype_definition/HcommReduceOp.md). |

### dataType Description

<!-- npu="950" id6 -->
- For the Ascend 950PR&950DT products, the supported data types are **int8**, **int16**, **int32**, **uint8**, **uint16**, **uint32**, **fp16**, **fp32**, and **bfp16**.
<!-- end id6 -->
<!-- npu="A3" id7 -->
- For the Atlas A3 products, the supported data types are **int8**, **int16**, **int32**, **float16**, **float32**, and **bfp16**.
<!-- end id7 -->
<!-- npu="910b" id8 -->
- For the Atlas A2 products, the supported data types are **int8**, **int16**, **int32**, **float16**, **float32**, and **bfp16**.
<!-- end id8 -->

## Return Value

**int32_t**: This API returns **0** on success and a non-zero value on failure.

## Constraints

None

## Example

```c
// Allocate communication thread resources.
CommEngine engine = CommEngine::COMM_ENGINE_CPU_TS; // Used by the Atlas A3 products.
CommEngine engine = CommEngine::COMM_ENGINE_AICPU_TS; // Used by the Ascend 950PR&950DT products.
uint32_t threadNum = 1;
uint32_t notifyNumPerThread = 1;
ThreadHandle thread;
HcclComm comm;
HcclThreadAcquire(comm, engine, threadNum, notifyNumPerThread, &thread);

// Allocate communication channel resources.
uint32_t channelNum = 1;
HcclChannelDesc channelDesc;
HcclChannelDescInit(&channelDesc, channelNum);
ChannelHandle channel;
HcclChannelAcquire(comm, engine, &channelDesc, channelNum, &channel);

// Obtain the local communication memory information.
void * localBuffer;
uint64_t localBufferSize;
HcclGetHcclBuffer(comm, &localBuffer, &localBufferSize);
// Obtain the remote communication memory information.
void * remoteBuffer;
uint64_t remoteBufferSize;
HcclChannelGetHcclBuffer(comm, channel, &remoteBuffer, &remoteBufferSize);
uint64_t count = std::min(localBufferSize/sizeof(uint64_t), remoteBufferSize/sizeof(uint64_t));

// Perform reduce on the local and remote memory data, and write the result to the remote memory.
HcommWriteReduceOnThread(thread, channel, remoteBuffer, localBuffer, count, HCOMM_DATA_TYPE_INT32, HCOMM_REDUCE_SUM);
```
