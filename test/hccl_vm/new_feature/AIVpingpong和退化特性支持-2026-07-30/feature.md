# AIV 小数据量 ping-pong 与大数据量退化适配

| 特性 | HCCL 实现 | 模拟器 适配 | 当前边界 |
| --- | --- | --- | --- |
| AIV 小数据量 ping-pong | `len * sizeof(T) <= 512 KiB` 时启用，根据 tag 奇偶交替使用 `PONG + FLAG2` / `PING + FLAG1`，并省略尾部 `BarrierAll()`。当前明确用于 **AllGather、等长 AllToAll、ReduceScatter LocalTree/LocalTreeCoreCtrl**。 | 按 HCCL 布局建模完整的 65 MiB `aivCommInfo`，包含两组数据区和标志区；`Record/WaitFlag` 跟随当前 `gmOutOffset`，清零时能回溯到 `FLAG1` 起始地址。 | 模拟器将 tag 固定为 `1`，当前实际覆盖 **PONG + FLAG2** 分支。 |
| AIV 大数据量退化 | 普通 `AIV` 模式下，选择器数据量 `>= 8 MiB * rankSize`，或数据量 `> cclBufferSize * 16` 时 AIV 返回 `NOT_MATCH`；随后继续选择 AICPU，匹配后将执行模式改为 `AICPU_TS`。 | 支持当前算子展开模式从 `AIV` 更新为 `AICPU`，再启动/复用 AICPU device 进程执行，适配真实退化流程。 | 该大数据阈值当前见于 **AllReduce、AllGather、ReduceScatter、Broadcast、Reduce、Scatter、等长 AllToAll**；**AllToAllV 当前无此数据量退化判断**。`AIV_ONLY` 会跳过 `8 MiB * rankSize` 判断，且 AIV 不匹配时不会退化到 AICPU；`cclBufferSize * 16` 限制仍生效。 |
