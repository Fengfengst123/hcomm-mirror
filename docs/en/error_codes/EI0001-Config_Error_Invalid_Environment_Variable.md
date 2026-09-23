# EI0001 Config_Error_Invalid_Environment_Variable

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: environment variable value, environment variable name, expected value.

```text
Value %s for environment variable %s is invalid. Expected value: %s.
```

Error example:

```text
Value 2147483648 for environment variable HCCL_EXEC_TIMEOUT is invalid. Expected value: a number greater than or equal to 0s and less than or equal to 2147483647s.
```

<!-- npu="950" id1 -->
## Locating the fault in plog

The error format above is used for ERROR MESSAGE reporting. For Ascend 950PR&950DT products, the corresponding plog contains the environment variable name, configured value, and reason, with the following search keywords:

| Error code | Error title | Stage keyword | Scenario keyword | Engine keyword |
| --- | --- | --- | --- | --- |
| EI0001 | Config_Error_Invalid_Environment_Variable | InitGroupStage | EnvConfig | - |

Use `grep -F '[InitGroupStage][EnvConfig]'` to search plog. For example, setting `HCCL_ENTRY_LOG_ENABLE=-1` produces a message containing the following information, indicating that only 0 or 1 is supported:

```text
[InitGroupStage][EnvConfig] ... Env config "HCCL_ENTRY_LOG_ENABLE" value "-1" is invalid. ... Should be 0 or 1
```

<!-- end id1 -->

## Solution

The environment variable configuration is invalid. Please reconfigure the environment variable as prompted in the error message.
