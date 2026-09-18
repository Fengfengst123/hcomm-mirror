/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 日志染色: 模块 tag (须在 include sim_log.h 之前)
#define HCCL_VM_MODULE "HCOMM_STUB"

#include "hccl/hccl_types.h"
#include "hccl_proxy_common.h"
#include "sim_log.h"
#include <cstdint>
#include <dlfcn.h>

#ifdef __cplusplus
extern "C" {
#endif

HcclResult HcclDevMemAcquire(HcclComm comm, const char* memTag, uint64_t* size, void** addr, bool* newCreated)
{
    HCCL_VM_INFO(
        "[HcclDevMemAcquire] comm={:p}, memTag={}, size_ptr={:p}, "
        "addr_ptr={:p}, newCreated_ptr={:p}",
        static_cast<void*>(comm), (memTag != nullptr) ? memTag : "nullptr", static_cast<void*>(size),
        static_cast<void*>(addr), static_cast<void*>(newCreated));

    using RealFunc = HcclResult (*)(HcclComm comm, const char* memTag, uint64_t* size, void** addr, bool* newCreated);

    const auto realFn = reinterpret_cast<RealFunc>(dlsym(RTLD_NEXT, __func__));
    if (realFn == nullptr) {
        const char* err = dlerror();
        HCCL_VM_ERROR("HcclDevMemAcquire dlsym failed, err={}", err ? err : "unknown");
        return HCCL_E_INTERNAL;
    }

    // 转发调用真实底层函数
    HcclResult ret = realFn(comm, memTag, size, addr, newCreated);
    if (ret != HCCL_SUCCESS) {
        HCCL_VM_ERROR("HcclDevMemAcquire realFn failed, ret={}", static_cast<int>(ret));
        return ret;
    }

    // 仅在真实函数成功时，将*addr(host进程可访问地址)转化为device进程可访问地址
    *addr = sim::GetDevMapperAddrByHostPtr(*addr);
    if (*addr == nullptr) {
        return HCCL_E_PTR;
    }

    return ret;
}

#ifdef __cplusplus
}
#endif // __cplusplus
