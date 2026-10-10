# HcommLocalCopyOnThread

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T08:40:26.543Z pushedAt=2026-10-08T02:58:22.551Z -->

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

Provides local memory copy. It copies the memory data of length **len** pointed to by **src** to the memory of the same length pointed to by **dst**.

## Function Prototype

```c
int32_t HcommLocalCopyOnThread(ThreadHandle thread, void *dst, const void *src, uint64_t len)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| thread | Input | Communication thread handle, which is the thread obtained through the [HcclThreadAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclThreadAcquire.md) API.<br>For the definition of the ThreadHandle type, see [ThreadHandle](../../../datatype_definition/ThreadHandle.md). |
| dst | Output | Destination memory address, which is device memory. |
| src | Input | Source memory address, which is device memory. |
| len | Input | Data length (in bytes). |

## Return Value

**int32_t**: This API returns **0** on success and a non-zero value on failure.

## Constraints

**dst** and **src** are allocated device memory.

<!-- npu="950" id6 -->
On Ascend 950PR&950DT products, this API can be called only in AICPU_TS mode on the device side.
<!-- end id6 -->

## Example

```c
HcclComm comm;
CommEngine engine = COMM_ENGINE_CPU_TS;
aclrtStream stream;
aclrtCreateStream(&stream);
ThreadHandle thread;
HcclResult result = HcclThreadAcquireWithStream(comm, engine, stream, 2, &thread);

// Allocate device memory.
uint64_t memSize = 256;
s32 policy = static_cast<int>(ACL_MEM_TYPE_HIGH_BAND_WIDTH) | static_cast<int>(ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMallocAttrValue moduleIdValue;
moduleIdValue.moduleId = HCCL;
aclrtMallocAttribute attrs{.attr = ACL_RT_MEM_ATTR_MODULE_ID, .value = moduleIdValue};
aclrtMallocConfig cfg{.attrs = &attrs, .numAttrs = 1};

void* inputMem;
void* outputMem;
aclrtMallocWithCfg(&inputMem, memSize, static_cast<aclrtMemMallocPolicy>(policy), &cfg);
aclrtMallocWithCfg(&outputMem, memSize, static_cast<aclrtMemMallocPolicy>(policy), &cfg);
// Perform D2D copy.
HcommLocalCopyOnThread(thread, outputMem, inputMem, memSize);
```

On Ascend 950PR&950DT products, this function must be compiled for use on the device side:

```c
HcclComm comm;
CommEngine engine = COMM_ENGINE_AICPU_TS;

void *cclBufferAddr = nullptr;
uint64_t cclBufferSize = 0;
HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize);
ThreadHandle thread;
HcclThreadAcquire(comm, engine, 1, 1, &thread);

// Allocate other resources.
// Copy parameters and launch the kernel.

// Orchestrate the algorithm on the device side.
uint64_t len = 256;
void *src = param.userIn;
void *dst = param.cclBuf;
// Perform the D2D copy.
HcommLocalCopyOnThread(thread, dst, src, len);
```
