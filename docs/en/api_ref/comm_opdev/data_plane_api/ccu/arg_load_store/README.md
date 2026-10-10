# Introduction

<!-- md-trans-meta sourceCommit=53fdf106ef330fac6b558d13103b73026d3055b6 translatedAt=2026-09-28T07:49:23.383Z pushedAt=2026-09-30T03:33:21.379Z -->

This section provides the APIs for passing scalar values between variables and external data sources in the CCU kernel, used to declare the source or destination of runtime data during the registration phase.

A variable is only a scalar placeholder during the registration phase and carries no value itself. The APIs in this section are responsible for exchanging data with the outside before/after kernel execution. External sources fall into the following two categories:

| Source/Destination | When the address is determined | API |
| --- | --- | --- |
| `taskArgs[]` (injected by the host at each launch) | At runtime launch | [LoadArg](LoadArg.md) |
| HBM immediate address (a fixed constant at registration) | Registration phase | [Load](Load.md) (overload 1/2), [Store](Store.md) (overload 1/2) |
| HBM indirect address (determined by variables at runtime) | Runtime | [Load](Load.md) (overload 3/4), [Store](Store.md) (overload 3/4) |

Both `Load` and `Store` automatically select the immediate addressing or indirect addressing path based on the type of the first parameter (`uint64_t` or `Variable`). Each path internally supports two granularities: a single variable and a batch of `Array<Variable>`.

## API List

- [LoadArg](LoadArg.md)
- [Load](Load.md)
- [Store](Store.md)
