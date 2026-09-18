# 20260730 Ascend 950 模拟器 算子支持概览

## 1. 模拟器支持算子及 HCCL Engine 约束

| 模拟器 支持算子 | 模拟器 支持范围/约束 | hccl_test 支持算子 | AICPU_TS engine | CCU_MS engine | CCU_SCHED engine | AIV_ONLY engine |
| --- | --- | --- | --- | --- | --- | --- |
| `AllReduce` | 支持 | 支持 | 支持 | 支持 | 支持 | 支持 |
| `AllGather` | 支持 | 支持 | 支持 | 支持 | 支持 | 支持 |
| `ReduceScatter` | 支持 | 支持 | 支持 | 支持 | 支持 | 支持 |
| `Broadcast` | 支持 | 支持 | 支持 | 支持 | 支持 | 支持 |
| `Reduce` | 支持 | 支持 | 支持 | 支持 | 支持 | 支持 |
| `Scatter` | 支持 | 支持 | 支持 | 不支持 | 支持 | 支持 |
| `AlltoAll` | 支持 | 支持 | 支持 | 不支持 | 支持 | 支持 |
| `AlltoAllV` | 支持；收发 buffer 均须按 peer rank 顺序连续存放，`sdispls/rdispls` 等于对应 counts 的前缀和，不支持空洞。 | 支持 | 支持 | 不支持 | 支持 | 支持 |
| `AlltoAllVC` | 支持；收发 buffer 均须按 peer rank 顺序连续存放。 | 支持 | 支持 | 不支持 | 支持 | 不支持 |
| `AllGatherV` | 支持；接收 buffer 按 source rank 顺序连续存放，`recvDispls` 等于 `recvCounts` 的前缀和，不支持空洞。 | 支持 | 支持 | 不支持 | 支持 | 不支持 |
| `ReduceScatterV` | 支持；各 rank 输入分片连续存放，`sendDispls` 等于 `sendCounts` 的前缀和；每个 rank 的输出连续。 | 支持 | 支持 | 不支持 | 支持 | 不支持 |
| `Send` | 支持；目的 rank 必须合法且不能是本 rank，非零 count 的 buffer 不能为空；校验对端输出从偏移 0 连续存放、仅来自本 rank 的 INPUT、源/目的偏移一致，且总字节数完全匹配。 | 不支持 | 支持 | 不支持 | 不支持 | 支持 |
| `Recv` | 支持；源 rank 必须合法且不能是本 rank，非零 count 的 buffer 不能为空；校验本 rank 输出从偏移 0 连续存放、仅来自指定源 rank 的 INPUT、源/目的偏移一致，且总字节数完全匹配。 | 不支持 | 支持 | 不支持 | 不支持 | 支持 |
| `BatchSendRecv` | 仅支持环形一发一收：rank 数至少为 2，每 rank 恰好两个 item（一个 SEND、一个 RECV），向下一 rank 发送并从上一 rank 接收；所有 rank 的 count/data type 一致，非零 count 的收发 buffer 有效；输出须连续、大小完全匹配，且仅来自上一 rank 的 INPUT 同偏移数据。 | 不支持 | 支持 | 不支持 | 不支持 | 不支持 |

> 模拟器已支持当前HCCL仓中HCCL engine和hccltest同时支持的算子。
> P2P算子(send/recv, batchSendRecv) 算子将随着hccltest后续的具体实现跟进适配
