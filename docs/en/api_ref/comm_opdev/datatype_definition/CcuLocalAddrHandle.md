# CcuLocalAddrHandle

<!-- md-trans-meta sourceCommit=2b912e7d2b105735579276a63a51364890c56788 translatedAt=2026-09-28T08:50:32.409Z pushedAt=2026-10-08T03:49:49.955Z -->

## Description

Handle type of the CCU LocalAddr (local address), used to identify the local HBM address resource created in the CCU kernel through the [LocalAddr](../data_plane_api/ccu/resource_allocation_operation/LocalAddr.md) resource. This handle is used as the local address parameter in CCU data plane APIs (such as **LocalCopy**, **LocalReduce**, **Read**, and **Write**).

## Prototype

```c
typedef uint64_t CcuLocalAddrHandle;
```
