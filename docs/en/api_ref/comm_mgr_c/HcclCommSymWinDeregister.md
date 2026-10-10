# HcclCommSymWinDeregister

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:15:09.247Z pushedAt=2026-09-28T10:53:23.285Z -->

## Supported Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Not supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- Atlas inference products: Not supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas training products: Not supported
<!-- end id5 -->

## Description

Deregisters a registered symmetric memory window and releases the symmetric memory window resources. This API does not release the memory allocated by the user. The user still needs to release the corresponding memory in the same way as it was allocated.

<!-- npu="950" id6 -->
- For Ascend 950PR&950DT products, this API supports the URMA scenario and the UB Memory scenario.
<!-- end id6 -->
<!-- npu="A3" id7 -->
- For Atlas A3 products, this API supports the HCCS link communication scenario.
<!-- end id7 -->

## Function Prototype

```c
HcclResult HcclCommSymWinDeregister(HcclCommSymWindow winHandle)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| winHandle | Input | Handle of the registered symmetric window.<br>For the definition of the HcclCommSymWindow type, see [HcclCommSymWindow](./data_type_definition/HcclCommSymWindow.md). |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

- This API must be used together with [HcclCommSymWinRegister](HcclCommSymWinRegister.md).
- The supported scope is the same as that of [HcclCommSymWinRegister](HcclCommSymWinRegister.md).
- Ensure that all ranks in the communicator call this API at the same time to release the symmetric window resources.

## Example

See [Example](HcclCommSymWinRegister.md#example).
