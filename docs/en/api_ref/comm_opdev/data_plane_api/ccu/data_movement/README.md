# Introduction

<!-- md-trans-meta sourceCommit=53fdf106ef330fac6b558d13103b73026d3055b6 translatedAt=2026-09-28T07:54:24.109Z pushedAt=2026-09-30T04:01:02.605Z -->

This section provides the data movement APIs in the CCU kernel for moving bytes between the local HBM, the local MS Buffer, and the remote HBM of another rank, with an optional reduction operation performed during the movement.

All data movement APIs are asynchronous. When the hardware completes a movement, it automatically sets bit `mask` of `event` to 1, and the downstream waits for the completion signal through `EventWait`. Based on the data path, the APIs are divided into the following two categories:

| Type | Applicable Scenario | API |
| --- | --- | --- |
| Local operation | Copy and reduction between local HBM↔local HBM or local HBM↔local MS Buffer | [LocalCopy](LocalCopy.md), [LocalReduce](LocalReduce.md) |
| Cross-rank operation | Read and write data between the local and remote HBMs through a channel, with optional reduction | [Read](Read.md), [ReadReduce](ReadReduce.md), [Write](Write.md), [WriteReduce](WriteReduce.md) |

Cross-rank operations require that all `ChannelHandle` belong to the same die, which is uniformly validated by the framework.

## API List

- [LocalCopy](LocalCopy.md)
- [LocalReduce](LocalReduce.md)
- [Read](Read.md)
- [ReadReduce](ReadReduce.md)
- [Write](Write.md)
- [WriteReduce](WriteReduce.md)
