# HcclCommSymWinDeregister

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
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

将已注册的对称内存窗口解除注册，释放对称内存窗口资源。该接口不释放用户申请的内存，用户仍需按照内存申请方式释放对应内存。

<!-- npu="950" id6 -->
- 针对Ascend 950PR&950DT系列产品，本接口支持URMA场景和UB Memory场景。
<!-- end id6 -->
<!-- npu="A3" id7 -->
- 针对Atlas A3系列产品，本接口支持HCCS链路通信场景。
<!-- end id7 -->

## 函数原型

```c
HcclResult HcclCommSymWinDeregister(HcclCommSymWindow winHandle)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| winHandle | 输入 | 注册过的对称窗口句柄。<br>HcclCommSymWindow类型的定义请参见[HcclCommSymWindow](./data_type_definition/HcclCommSymWindow.md)。 |

## 返回值

[HcclResult](./data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

- 该接口需要配套[HcclCommSymWinRegister](HcclCommSymWinRegister.md)使用。
- 支持范围与[HcclCommSymWinRegister](HcclCommSymWinRegister.md)一致。
- 确保通信域中的所有rank同时调用该接口释放对称窗口资源。
- 本接口不支持多线程并发调用。调用方须保证不同线程对本接口的调用串行执行。

## 调用示例

请参见[调用示例](HcclCommSymWinRegister.md#调用示例)。
