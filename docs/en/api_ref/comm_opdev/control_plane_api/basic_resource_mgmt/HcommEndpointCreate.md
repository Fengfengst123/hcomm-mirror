# HcommEndpointCreate

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:56:58.830Z pushedAt=2026-09-29T06:02:13.065Z -->

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

Creates an endpoint for a communication device.

## Function Prototype

```c
HcommResult HcommEndpointCreate(const EndpointDesc *endpoint, EndpointHandle *endpointHandle)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| endpoint | Input | Endpoint initialization configuration information.<br>Mandatory members: **protocol**, **commAddr**, and **loc**.<br>For details about the EndpointDesc type, see [EndpointDesc](../../datatype_definition/EndpointDesc.md). |
| endpointHandle | Output | Returned endpoint handle.<br>For details about the EndpointHandle type, see [EndpointHandle](../../datatype_definition/EndpointHandle.md). |

## Return Value

**HcommResult**: The API returns **0** on success and other values on failure.

## Constraints

 - When the endpoint is on the host side, the supported protocols are RoCE and UB_CTP.
 - When the endpoint is on the device side, the supported protocols are RoCE, UB_CTP, UB_MEM, PCIe, UBoE, HCCS, and UB_RTP. Among them, UB_RTP is supported only by Ascend 950PR&950DT products.

## Example

```c
struct in_addr ipAddr;
inet_pton(AF_INET, "192.168.1.100", &ipAddr);
const EndpointDesc endpointDesc = {
    .protocol = COMM_PROTOCOL_UB_RTP,
    .commAddr = {
        .type = COMM_ADDR_TYPE_IP_V4,
        .addr = ipAddr
    },
    .loc = {
        .locType = ENDPOINT_LOC_TYPE_DEVICE,
        .device = {
            .devPhyId = 0,
            .superDevId = 0,
            .serverIdx = 0,
            .superPodIdx = 0
        }
    },
    .raws = {0}
};
EndpointHandle endpointHandle = nullptr;
HcommResult result = HcommEndpointCreate(&endpointDesc, &endpointHandle);
```
