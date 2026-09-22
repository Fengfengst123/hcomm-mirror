# EI0001 Config_Error_Invalid_Environment_Variable

## 错误信息

报错格式如下，占位符%s的含义依次为环境变量值、环境变量名、报错原因：

```text
Value %s for environment variable %s is invalid. Expected value: %s.
```

报错示例如下：

```text
Value 2147483648 for environment variable HCCL_EXEC_TIMEOUT is invalid. Expected value: a number greater than or equal to 0s and less than or equal to 2147483647s.
```

<!-- npu="950" id1 -->
## plog日志定位

上述错误信息为ERROR MESSAGE的错误码上报格式。针对Ascend 950PR&950DT系列产品，配套plog日志使用以下检索关键字，正文包含环境变量名、配置值及错误原因：

| EI错误码 | 错误标题 | 一级关键字 | 二级关键字 | 三级关键字 |
| --- | --- | --- | --- | --- |
| EI0001 | Config_Error_Invalid_Environment_Variable环境变量配置异常 | InitGroupStage | EnvConfig | - |

可使用`grep -F '[InitGroupStage][EnvConfig]'`检索plog。例如，设置`HCCL_ENTRY_LOG_ENABLE=-1`时，日志正文包含以下信息，表示该环境变量仅支持0或1：

```text
[InitGroupStage][EnvConfig] ... Env config "HCCL_ENTRY_LOG_ENABLE" value "-1" is invalid. ... Should be 0 or 1
```

<!-- end id1 -->

## 解决方法

环境变量配置无效，请根据错误提示重新配置环境变量。
