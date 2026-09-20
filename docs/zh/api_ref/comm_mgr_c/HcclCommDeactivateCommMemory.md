# HcclCommDeactivateCommMemory

> [!NOTE]说明
> 本接口为试用接口，后续可能存在变更，暂不支持应用于生产环境。

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：不支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：不支持
<!-- end id3 -->
<!-- npu="310p" id4 -->
- Atlas推理系列产品：不支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas训练系列产品：不支持
<!-- end id5 -->

## 功能说明

将已经激活的虚拟内存反激活，反激活后如果再使用该地址进行集合通信，将无法开启零拷贝功能。

## 函数原型

```c
HcclResult HcclCommDeactivateCommMemory(HcclComm comm, void *virPtr)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | HCCL通信域，建议使用Server内最大的通信域，即覆盖最大卡数的通信域。 |
| virPtr | 输入 | 需要反激活的虚拟地址的起始地址，即[HcclCommActivateCommMemory](HcclCommActivateCommMemory.md)接口“virPtr”参数指定的虚拟内存地址。<br>需要注意，指定的虚拟内存必须是已成功激活的内存，且仅支持整个地址块反激活。 |

## 返回值

[HcclResult](./data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

- comm必须为有效的通信域句柄（由通信域创建接口获得，调用期间保持有效）。禁止传入空指针、野指针或已销毁的句柄。

## 调用示例

```c
// 假设virPtr为已通过HcclCommActivateCommMemory激活的虚拟内存地址

// 反激活已激活的虚拟内存
HcclCommDeactivateCommMemory(hcclComm, virPtr);

// 后续该地址无法再用于零拷贝通信
// 如需重新使用，需再次调用HcclCommActivateCommMemory激活
```
