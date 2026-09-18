# 20260730 Ascend 950 Simulator Operator Support Overview

## 1. Simulator Supported Operators and HCCL Engine Constraints

| Simulator Supported Operators | Simulator Support Scope/Constraints | hccl_test Supported Operators | AICPU_TS engine | CCU_MS engine | CCU_SCHED engine | AIV_ONLY engine |
| --- | --- | --- | --- | --- | --- | --- |
| `AllReduce` | Supported | Supported | Supported | Supported | Supported | Supported |
| `AllGather` | Supported | Supported | Supported | Supported | Supported | Supported |
| `ReduceScatter` | Supported | Supported | Supported | Supported | Supported | Supported |
| `Broadcast` | Supported | Supported | Supported | Supported | Supported | Supported |
| `Reduce` | Supported | Supported | Supported | Supported | Supported | Supported |
| `Scatter` | Supported | Supported | Supported | Not supported | Supported | Supported |
| `AlltoAll` | Supported | Supported | Supported | Not supported | Supported | Supported |
| `AlltoAllV` | Supported; send/recv buffers must be stored consecutively in peer rank order, `sdispls/rdispls` must equal the prefix sum of the corresponding counts, no gaps supported. | Supported | Supported | Not supported | Supported | Supported |
| `AlltoAllVC` | Supported; send/recv buffers must be stored consecutively in peer rank order. | Supported | Supported | Not supported | Supported | Not supported |
| `AllGatherV` | Supported; recv buffer stored consecutively in source rank order, `recvDispls` equals the prefix sum of `recvCounts`, no gaps supported. | Supported | Supported | Not supported | Supported | Not supported |
| `ReduceScatterV` | Supported; each rank's input shards stored consecutively, `sendDispls` equals the prefix sum of `sendCounts`; each rank's output is contiguous. | Supported | Supported | Not supported | Supported | Not supported |
| `Send` | Supported; destination rank must be valid and cannot be the current rank, non-zero count buffers cannot be empty; verifies peer output stored contiguously from offset 0, sourced only from this rank's INPUT, source/destination offsets match, and total byte count matches exactly. | Not supported | Supported | Not supported | Not supported | Supported |
| `Recv` | Supported; source rank must be valid and cannot be the current rank, non-zero count buffers cannot be empty; verifies this rank's output stored contiguously from offset 0, sourced only from the specified source rank's INPUT, source/destination offsets match, and total byte count matches exactly. | Not supported | Supported | Not supported | Not supported | Supported |
| `BatchSendRecv` | Only supports ring-pattern one-send-one-recv: at least 2 ranks, each rank has exactly two items (one SEND, one RECV), sending to the next rank and receiving from the previous rank; all ranks have the same count/data type, non-zero count send/recv buffers must be valid; output must be contiguous, sizes must match exactly, and sourced only from the previous rank's INPUT at the same offset. | Not supported | Supported | Not supported | Not supported | Not supported |

> The simulator now supports all operators that are jointly supported by HCCL engines and hccl_test in the current HCCL repository.
> P2P operators (send/recv, batchSendRecv) will be adapted as hccl_test provides concrete implementations in the future.
