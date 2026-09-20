/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_EXPERIMENTAL_ROCE_MEM_H
#define HCOMM_EXPERIMENTAL_ROCE_MEM_H

#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>
#include <string>
#include "reged_mem_mgr.h"
#include "rma_buffer_mgr.h"
#include "buffer_key.h"
#include "local_rdma_rma_buffer_v2.h"
#include "remote_rma_buffer.h"
#include "exchange_rdma_buffer_dto.h"

namespace hcomm_experimental {

/**
 * @note 远端内存管理key：EndpointDesc标识网卡，pid标识该网卡上的远端进程。
 */
struct RemoteRdmaMemKey {
    EndpointDesc endpointDesc{};
    uint32_t pid{0};
};

inline bool operator==(const RemoteRdmaMemKey& lhs, const RemoteRdmaMemKey& rhs) noexcept
{
    return (lhs.pid == rhs.pid) && (std::memcmp(&lhs.endpointDesc, &rhs.endpointDesc, sizeof(EndpointDesc)) == 0);
}

struct RemoteRdmaMemKeyHash {
    size_t operator()(const RemoteRdmaMemKey& key) const noexcept
    {
        // FNV-1a（64位）对EndpointDesc字节序列与pid做hash
        const uint8_t* p = reinterpret_cast<const uint8_t*>(&key);
        size_t h
            = sizeof(size_t) == 8 ? static_cast<size_t>(14695981039346656037ull) : static_cast<size_t>(2166136261u);
        const size_t prime
            = sizeof(size_t) == 8 ? static_cast<size_t>(1099511628211ull) : static_cast<size_t>(16777619u);
        for (size_t i = 0; i < sizeof(key.endpointDesc); ++i) {
            h ^= static_cast<size_t>(p[i]);
            h *= prime;
        }
        h ^= static_cast<size_t>(key.pid);
        h *= prime;
        return h;
    }
};

/**
 * @note 职责：用于通信设备EndPoint的注册内存信息管理，支持基于RmaBufferMgr类的重叠内存的检测报错等。
 */
class RoceRegedMemMgr : public RegedMemMgr {
public:
    using LocalRdmaRmaBufferMgr
        = hcomm::RmaBufferMgr<Hccl::BufferKey<uintptr_t, u64>, std::shared_ptr<Hccl::LocalRdmaRmaBuffer>>;
    using RemoteRdmaRmaBufferMgr
        = hcomm::RmaBufferMgr<Hccl::BufferKey<uintptr_t, u64>, std::shared_ptr<Hccl::RemoteRdmaRmaBuffer>>;

    RoceRegedMemMgr();
    ~RoceRegedMemMgr() = default;

    HcclResult RegisterMemory(HcommMem mem, const char* memTag, void** memHandle) override;
    HcclResult UnregisterMemory(void* memHandle) override;
    HcclResult
    MemoryExport(const EndpointDesc endpointDesc, void* memHandle, void** memDesc, uint32_t* memDescLen) override;
    HcclResult MemoryImport(const void* memDesc, uint32_t descLen, HcommMem* outMem) override;
    HcclResult MemoryUnimport(const void* memDesc, uint32_t descLen) override;
    HcclResult GetAllMemHandles(void** memHandles, uint32_t* memHandleNum) override;
    HcclResult GetMemDesc(const EndpointDesc endpointDesc, Hccl::LocalRdmaRmaBuffer* localRdmaRmaBuffer);
    HcclResult GetParamsFromMemDesc(
        const void* memDesc, uint32_t descLen, EndpointDesc& endpointDesc, Hccl::ExchangeRdmaBufferDto& dto,
        uint32_t& pid);

private:
    std::unique_ptr<LocalRdmaRmaBufferMgr> localRdmaRmaBufferMgr_{};
    std::vector<std::shared_ptr<Hccl::LocalRdmaRmaBuffer>> allRegisteredBuffers_;
    std::unordered_map<RemoteRdmaMemKey, std::unique_ptr<RemoteRdmaRmaBufferMgr>, RemoteRdmaMemKeyHash>
        remoteRdmaRmaBufferMgrs_;
    mutable std::mutex memMtx_;
};
} // namespace hcomm_experimental

#endif // HCOMM_EXPERIMENTAL_ROCE_MEM_H
