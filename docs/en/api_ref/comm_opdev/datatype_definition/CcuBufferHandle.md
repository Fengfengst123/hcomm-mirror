# CcuBufferHandle

<!-- md-trans-meta sourceCommit=2b912e7d2b105735579276a63a51364890c56788 translatedAt=2026-09-28T08:48:55.886Z pushedAt=2026-10-08T03:45:50.651Z -->

## Description

Handle type of CCU Buffer (MS Buffer), used to identify the MS Buffer resource created through [CcuBuffer](../data_plane_api/ccu/resource_allocation_operation/CcuBuffer.md) within the CCU kernel. MS Buffer is an on-chip high-speed scratchpad area in the CCU die, used to transfer data between on-chip memory and the remote end.

## Prototype

```c
typedef uint64_t CcuBufferHandle;
```
