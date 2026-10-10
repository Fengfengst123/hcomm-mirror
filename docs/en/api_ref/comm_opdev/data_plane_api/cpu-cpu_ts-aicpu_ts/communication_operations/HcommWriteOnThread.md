# HcommWriteOnThread

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T08:34:34.671Z pushedAt=2026-10-08T02:23:39.620Z -->

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

Writes data to the specified memory on a channel, copying the memory data of length **len** from **src** to the memory area of the same length pointed to by **dst**. The API caller is the node where **src** resides. This API is asynchronous.

## Function Prototype

```c
int32_t HcommWriteOnThread(ThreadHandle thread, ChannelHandle channel, void *dst, const void *src, uint64_t len)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| thread | Input | Communication thread handle, which is the thread obtained through the [HcclThreadAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclThreadAcquire.md) API.<br>For the definition of the ThreadHandle type, see [ThreadHandle](../../../datatype_definition/ThreadHandle.md). |
| channel | Input | Communication channel handle, which is the channel obtained through the [HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md) API.<br>For the definition of the ChannelHandle type, see [ChannelHandle](../../../datatype_definition/ChannelHandle.md). |
| dst | Output | Destination memory address, which is the memory obtained through [HcclGetHcclBuffer](../../../control_plane_api/comms_domain_resource_mgmt/HcclGetHcclBuffer.md) or [HcclChannelGetHcclBuffer](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelGetHcclBuffer.md). |
| src | Input | Source memory address, which is the memory obtained through [HcclGetHcclBuffer](../../../control_plane_api/comms_domain_resource_mgmt/HcclGetHcclBuffer.md) or [HcclChannelGetHcclBuffer](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelGetHcclBuffer.md). |
| len | Input | Data length (in bytes). |

## Return Value

**int32_t**: This API returns **0** on success and a non-zero value on failure.

<!-- npu="950" id6 -->
## Constraints

For Ascend 950PR&950DT products, only the UB_CTP and UBoE communication protocols are supported.
<!-- end id6 -->

## Example

```c
// Allocate communication thread resources.
CommEngine engine = CommEngine::COMM_ENGINE_CPU_TS; // Used by Atlas A3 products.
CommEngine engine = CommEngine::COMM_ENGINE_AICPU_TS; // Used by the Ascend 950PR&950DT product series.
uint32_t threadNum = 1;
uint32_t notifyNumPerThread = 1;
ThreadHandle thread;
HcclComm comm;
HcclThreadAcquire(comm, engine, threadNum, notifyNumPerThread, &thread);

// Allocate communication channel resources.
HcclChannelDesc channelDesc;
HcclChannelDescInit(&channelDesc, channelNum);
uint32_t channelNum = 1;
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
uint64_t len = std::min(localBufferSize, remoteBufferSize);

// Write the local memory content to the remote memory.
HcommWriteOnThread(thread, channel, remoteBuffer, localBuffer, len);
```
