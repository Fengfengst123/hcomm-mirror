# HcclConfig

<!-- md-trans-meta sourceCommit=2b912e7d2b105735579276a63a51364890c56788 translatedAt=2026-09-28T06:35:47.323Z pushedAt=2026-09-28T08:21:08.556Z -->

## Description

Defines the configuration related to collective communication.

## Prototype

```c
typedef enum {
    HCCL_DETERMINISTIC = 0, /* 0: non-deterministic, 1: deterministic, 2: strict(order-preserving) */
    HCCL_CONFIG_RESERVED
} HcclConfig;
```

## Parameters

- **HCCL_DETERMINISTIC**: Whether to enable deterministic computation.

  - **0**: Disables deterministic computation.
  - **1**: Enables deterministic computation.
  - **2**: Enables the order preservation function (supported only by Atlas A2 training products/Atlas A2 inference products).

- **HCCL_CONFIG_RESERVED**: Reserved parameter.
