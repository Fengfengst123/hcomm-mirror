# HcclGetHeterogMode

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:39:42.923Z pushedAt=2026-09-30T02:38:28.596Z -->

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

Obtains the heterogeneous networking mode of a given communicator.

## Function Prototype

```c
HcclResult HcclGetHeterogMode(HcclComm comm, HcclHeterogMode *mode)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator where the collective communication operation is performed.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| mode | Output | Heterogeneous mode.<br>For details about the HcclHeterogMode type, see [HcclHeterogMode](../../datatype_definition/HcclHeterogMode.md). |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

None

## Example

```c
HcclHeterogMode mode;
HcclResult ret = HcclGetHeterogMode(comm, &mode);
if (ret == HCCL_SUCCESS) {
    switch (mode) {
        case HCCL_HETEROG_MODE_HOMOGENEOUS:
            printf("Homogeneous networking \n");
            break;
        case HCCL_HETEROG_MODE_MIX_A2_A3:
            printf("A2/A3 heterogeneous networking \n");
            break;
        default:
            printf("Unknown networking mode \n");
            break;
    }
}
```
