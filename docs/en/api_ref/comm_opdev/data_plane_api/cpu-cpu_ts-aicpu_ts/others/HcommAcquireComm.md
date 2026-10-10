# HcommAcquireComm

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T08:44:09.613Z pushedAt=2026-10-08T03:14:41.303Z -->

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

Obtains the corresponding communicator based on the passed **commId** and locks the communicator to prevent it from being obtained repeatedly.

## Function Prototype

```c
int32_t HcommAcquireComm(const char* commId)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| commId | Input | Communicator ID. |

## Return Value

**int32_t**: This API returns **0** on success and a non-zero value on failure.

## Constraints

1. This API can be called only on the device side in AI CPU mode.
2. **HcommAcquireComm** and **HcommReleaseComm** correspond to the lock and unlock actions respectively, and must be called in pairs. The API internally intercepts repeated lock scenarios to prevent the same communicator from being occupied by multiple threads simultaneously.

## Example

This function must be compiled for use on the device side:

```c
// Kernel function executed on the AI CPU.
extern "C" unsigned int HcclAicpuKernel(const char* commId)
{
    // Lock the communicator to prevent it from being used concurrently.
    if (HcommAcquireComm(commId) != HCCL_SUCCESS) {
        return 1;
    }

    // Perform task orchestration.
    // ...

    // Release the communicator.
    if (HcommReleaseComm(commId) != HCCL_SUCCESS) {
        return 1;
    }
    return 0;
}
```
