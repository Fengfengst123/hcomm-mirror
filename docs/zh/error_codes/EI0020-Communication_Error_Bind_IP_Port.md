# EI0020 Communication_Error_Bind_IP_Port

## 错误信息

报错格式如下，占位符%s表示报错原因：

```text
Failed to enable listening for the NPU network adapter socket. Reason: %s
```

报错示例如下：

```text
Failed to enable listening for the NPU network adapter socket. Reason: The IP address 192.1.3.198 add port 16666 have already been bound.
```

## 解决方法

1. 确认是否为单卡多进程场景（同一Device上拉起多个业务进程）。若是，根据通信域初始化方式分别处理：

   - rootInfo方式初始化通信域：请配置环境变量HCCL_NPU_SOCKET_PORT_RANGE指定端口范围。

   - rank table文件方式初始化通信域：该方式下HCCL_NPU_SOCKET_PORT_RANGE不生效，请为每个进程使用的rank table文件中同一device_ip的rank配置互不相同的device_port/host_port（字段缺省时默认使用16666，单卡多进程场景下会冲突）。
