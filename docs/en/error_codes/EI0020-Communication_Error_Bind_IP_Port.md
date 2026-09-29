# EI0020 Communication_Error_Bind_IP_Port

## Symptom

The following is error format. The placeholder %s indicates the error cause.

```text
Failed to enable listening for the NPU network adapter socket. Reason: %s
```

Error example:

```text
Failed to enable listening for the NPU network adapter socket. Reason: The IP address 192.1.3.198 add port 16666 have already been bound.
```

## Solution

1. Check whether the single-card multi-process scenario is used (multiple service processes are started on the same device). If yes, handle the issue based on how the communication domain is initialized:

   - If the communication domain is initialized using root info: configure the environment variable HCCL_NPU_SOCKET_PORT_RANGE to specify a port range.

   - If the communication domain is initialized using a rank table file: HCCL_NPU_SOCKET_PORT_RANGE does not take effect in this mode. Configure different device_port/host_port values for the rank with the same device_ip in the rank table file used by each process (the default value 16666 is used when the fields are not configured).
