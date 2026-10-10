# HcommAclrtNotifyRecordOnThread

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T08:39:27.759Z pushedAt=2026-10-08T02:54:06.696Z -->

## Supported Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Not supported
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

Sends a synchronization signal based on the Notify created through the ACL API. It must be used in pairs with **HcommAclrtNotifyWaitOnThread**.

## Function Prototype

```c
int32_t HcommAclrtNotifyRecordOnThread(ThreadHandle thread, uint64_t dstNotifyId)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| thread | Input | Thread handle, which is the thread obtained through the [HcclThreadAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclThreadAcquire.md) API.<br>For the definition of the ThreadHandle type, see [ThreadHandle](../../../datatype_definition/ThreadHandle.md). |
| dstNotifyId | Input | Synchronization signal ID, which is the **notifyId** obtained through the **aclrtGetNotifyId** API. |

## Return Value

**int32_t**: This API returns **0** on success and a non-zero value on failure.

## Constraints

None

## Example

```c
HcclComm comm;
CommEngine engine = COMM_ENGINE_CPU_TS;
aclrtStream streams[2];
ThreadHandle threads[2];
// Allocate two streams, each with two Notify resources.
aclrtCreateStream(&streams[0]);
aclrtCreateStream(&streams[1]);
HcclResult result = HcclThreadAcquireWithStream(comm, engine, streams[0], 2, &threads[0]);
result = HcclThreadAcquireWithStream(comm, engine, streams[1], 2, &threads[1]);
aclrtNotify notify;
uint64_t notifyId;
aclrtCreateNotify(&(notify), ACL_NOTIFY_DEFAULT);
aclrtGetNotifyId(notify, &(notifyId));
// Send the synchronization signal.
HcommAclrtNotifyRecordOnThread(threads[0], notifyId);
// Wait for the synchronization signal.
uint32_t timeout = 1;
HcommAclrtNotifyWaitOnThread(threads[1], notifyId, timeout);
```
