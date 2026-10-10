# HcclTeamDestroy

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:34:16.526Z pushedAt=2026-09-30T02:08:32.464Z -->

## Supported Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Not supported
<!-- end id3 -->
<!-- npu="910" id4 -->
- Atlas training products: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas inference products: Not supported
<!-- end id5 -->

## Description

Destroys a team and releases its resources. During destruction, the local memory of the team's **syncMem** is released and the memory handle of **syncMem** is deregistered.

## Function Prototype

```c
HcclResult HcclTeamDestroy(HcommTeamHandle team)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| team | Input | Handle of the team to be destroyed, which cannot be **NULL**. It can be created by [HcclTeamCreate](HcclTeamCreate.md). |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

1. During destruction, the local memory of the team's **syncMem** is released, the **CommRegMem** memory handle of **syncMem** is deregistered, and the registration entries are cleared.

2. The team handle cannot be used after it is destroyed.

## Example

```c
HcclResult ret = HcclTeamDestroy(team);
if (ret != HCCL_SUCCESS) {
    printf("HcclTeamDestroy failed, ret = %d\n", ret);
}
```
