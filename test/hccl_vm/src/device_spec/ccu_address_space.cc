/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "device_spec/ccu_address_space.h"

#include <limits>

#include "sim_log.h"

namespace sim {
namespace devspec {

    CcuAddressSpace::CcuAddressSpace(const DeviceSpec& spec) : m_spec(spec) {}

    const RegionSpec* CcuAddressSpace::FindRegion(CcuResType type) const { return m_spec.FindRegion(type); }

    const MsWindow* CcuAddressSpace::FindMsWindow(uint64_t offset) const
    {
        for (const auto& w : m_spec.ms.windows) {
            if (offset >= w.offset && offset < w.offset + w.size) {
                return &w;
            }
        }
        return nullptr;
    }

    CcuResType CcuAddressSpace::TypeByOffset(uint64_t offset) const
    {
        for (const auto& r : m_spec.regions) {
            if (offset >= r.offset && offset < r.offset + r.size) {
                return r.type;
            }
        }
        if (FindMsWindow(offset) != nullptr) {
            return CcuResType::MS;
        }
        return CcuResType::UNKNOWN;
    }

    uint64_t CcuAddressSpace::RegionOffset(CcuResType type, bool& ok) const
    {
        const RegionSpec* r = FindRegion(type);
        ok = (r != nullptr);
        return ok ? r->offset : 0;
    }

    uint32_t CcuAddressSpace::EntrySize(CcuResType type, bool& ok) const
    {
        const RegionSpec* r = FindRegion(type);
        ok = (r != nullptr && r->entrySize != 0);
        return ok ? r->entrySize : 0;
    }

    uint32_t CcuAddressSpace::RegionCount(CcuResType type, bool& ok) const
    {
        const RegionSpec* r = FindRegion(type);
        ok = (r != nullptr);
        return ok ? r->count : 0;
    }

    bool CcuAddressSpace::AddrToResId(uint64_t dieBase, uint64_t addr, CcuResType& type, uint64_t& id) const
    {
        type = CcuResType::UNKNOWN;
        id = 0;
        if (dieBase == 0 || addr < dieBase) {
            HCCL_VM_ERROR("AddrToResId: addr {:#x} 小于 dieBase {:#x}", addr, dieBase);
            return false;
        }
        const uint64_t off = addr - dieBase;

        // MS 区优先判定：它不在 regions 表内
        if (FindMsWindow(off) != nullptr) {
            uint32_t msId = 0;
            if (!MsAddrToId(dieBase, addr, msId)) {
                return false;
            }
            type = CcuResType::MS;
            id = msId;
            return true;
        }

        for (const auto& r : m_spec.regions) {
            if (off < r.offset || off >= r.offset + r.size) {
                continue;
            }
            if (r.entrySize == 0) {
                // 不透明块（不按 id 编址）：只报类型，id 取块内字节偏移
                type = r.type;
                id = off - r.offset;
                return true;
            }
            const uint64_t local = off - r.offset;
            if (local % r.entrySize != 0) {
                HCCL_VM_ERROR(
                    "AddrToResId: addr {:#x} 未按 region \"{}\" 的 entry_size {} 对齐", addr, r.name.c_str(),
                    r.entrySize);
                return false;
            }
            const uint64_t rid = local / r.entrySize;
            // ★ 上界校验：修掉 ccu_resource_manager.cc:512,522 的 "需要判断界限 todo"
            if (rid >= r.count) {
                HCCL_VM_ERROR(
                    "AddrToResId: addr {:#x} 换算出 region \"{}\" 的 id {} ≥ count {}", addr, r.name.c_str(), rid,
                    r.count);
                return false;
            }
            type = r.type;
            id = rid;
            return true;
        }

        HCCL_VM_ERROR("AddrToResId: addr {:#x}（dieBase {:#x}，偏移 {:#x}）不落在任何已建模区域内", addr, dieBase, off);
        return false;
    }

    bool CcuAddressSpace::ResIdToAddr(uint64_t dieBase, CcuResType type, uint64_t id, uint64_t& addr) const
    {
        addr = 0;
        if (dieBase == 0) {
            HCCL_VM_ERROR("ResIdToAddr: dieBase 为 0");
            return false;
        }
        if (type == CcuResType::MS) {
            uint32_t msId = 0;
            if (id > std::numeric_limits<uint32_t>::max()) {
                return false;
            }
            msId = static_cast<uint32_t>(id);
            return MsIdToAddr(dieBase, msId, addr);
        }
        const RegionSpec* r = FindRegion(type);
        if (r == nullptr) {
            HCCL_VM_ERROR(
                "ResIdToAddr: 该 spec 无 \"{}\" 区域（A6 无 GSA 属正常，调用方应先判 HasGsa）",
                CcuResTypeToString(type));
            return false;
        }
        if (r->entrySize == 0) {
            HCCL_VM_ERROR("ResIdToAddr: region \"{}\" 是不透明块（entry_size=0），不支持 id 编址", r->name.c_str());
            return false;
        }
        if (id >= r->count) {
            HCCL_VM_ERROR("ResIdToAddr: region \"{}\" 的 id {} ≥ count {}", r->name.c_str(), id, r->count);
            return false;
        }
        const uint64_t off = r->offset + id * r->entrySize;
        if (off > std::numeric_limits<uint64_t>::max() - dieBase) {
            HCCL_VM_ERROR("ResIdToAddr overflow: dieBase={:#x} off={:#x}", dieBase, off);
            return false;
        }
        addr = dieBase + off;
        return true;
    }

    bool CcuAddressSpace::AddrToIdOfType(uint64_t dieBase, CcuResType type, uint64_t addr, uint64_t& id) const
    {
        id = 0;
        if (dieBase == 0 || addr < dieBase) {
            HCCL_VM_ERROR("AddrToIdOfType: addr {:#x} 小于 dieBase {:#x}", addr, dieBase);
            return false;
        }
        const RegionSpec* r = FindRegion(type);
        if (r == nullptr || r->entrySize == 0) {
            HCCL_VM_ERROR("AddrToIdOfType: 无 \"{}\" 区域或其 entry_size 为 0", CcuResTypeToString(type));
            return false;
        }
        const uint64_t off = addr - dieBase;
        if (off < r->offset || off >= r->offset + r->size) {
            HCCL_VM_ERROR(
                "AddrToIdOfType: addr {:#x}（偏移 {:#x}）不在 region \"{}\" [{:#x}, {:#x}) 内", addr, off,
                r->name.c_str(), r->offset, r->offset + r->size);
            return false;
        }
        const uint64_t local = off - r->offset;
        if (local % r->entrySize != 0) {
            HCCL_VM_ERROR("AddrToIdOfType: addr {:#x} 未按 entry_size {} 对齐", addr, r->entrySize);
            return false;
        }
        const uint64_t rid = local / r->entrySize;
        if (rid >= r->count) {
            HCCL_VM_ERROR("AddrToIdOfType: region \"{}\" 换算 id {} ≥ count {}", r->name.c_str(), rid, r->count);
            return false;
        }
        id = rid;
        return true;
    }

    bool CcuAddressSpace::MsAddrToId(uint64_t dieBase, uint64_t addr, uint32_t& msId) const
    {
        msId = 0;
        const MsSpec& m = m_spec.ms;
        if (dieBase == 0 || addr < dieBase || m.msSize == 0 || m.ccuaNum == 0 || m.interleave == 0) {
            return false;
        }
        const uint64_t off = addr - dieBase;
        const MsWindow* w = FindMsWindow(off);
        if (w == nullptr) {
            HCCL_VM_ERROR("MsAddrToId: addr {:#x}（偏移 {:#x}）不在任何 MS 窗口内", addr, off);
            return false;
        }
        const uint64_t local = off - w->offset;
        if (local % m.msSize != 0) {
            HCCL_VM_ERROR("MsAddrToId: addr {:#x} 未按 ms_size {} 对齐", addr, m.msSize);
            return false;
        }
        const uint64_t localIdx = local / m.msSize;
        const uint64_t groupTotal = static_cast<uint64_t>(m.interleave) * m.ccuaNum;
        const uint64_t id = (localIdx / m.interleave) * groupTotal + static_cast<uint64_t>(w->index) * m.interleave
                            + (localIdx % m.interleave);
        if (id >= m.msNum) {
            HCCL_VM_ERROR("MsAddrToId: addr {:#x} 换算出 msId {} ≥ ms_num {}", addr, id, m.msNum);
            return false;
        }
        msId = static_cast<uint32_t>(id);
        return true;
    }

    bool CcuAddressSpace::MsIdToAddr(uint64_t dieBase, uint32_t msId, uint64_t& addr) const
    {
        addr = 0;
        const MsSpec& m = m_spec.ms;
        if (dieBase == 0 || m.msSize == 0 || m.ccuaNum == 0 || m.interleave == 0) {
            return false;
        }
        if (msId >= m.msNum) {
            HCCL_VM_ERROR("MsIdToAddr: msId {} ≥ ms_num {}", msId, m.msNum);
            return false;
        }
        const uint64_t groupTotal = static_cast<uint64_t>(m.interleave) * m.ccuaNum;
        const uint64_t groupIdx = msId / groupTotal;
        const uint64_t ccuaIdx = (msId % groupTotal) / m.interleave;
        const uint64_t posInGroup = msId % m.interleave;
        const uint64_t localIdx = groupIdx * m.interleave + posInGroup;
        if (ccuaIdx >= m.windows.size()) {
            HCCL_VM_ERROR("MsIdToAddr: msId {} 换算出 ccuaIndex {} ≥ 窗口数 {}", msId, ccuaIdx, m.windows.size());
            return false;
        }
        const MsWindow& w = m.windows[ccuaIdx];
        const uint64_t off = w.offset + localIdx * m.msSize;
        if (off >= w.offset + w.size) {
            HCCL_VM_ERROR(
                "MsIdToAddr: msId {} 换算偏移 {:#x} 超出窗口 [{:#x}, {:#x})", msId, off, w.offset, w.offset + w.size);
            return false;
        }
        if (off > std::numeric_limits<uint64_t>::max() - dieBase) {
            return false;
        }
        addr = dieBase + off;
        return true;
    }

    bool CcuAddressSpace::IsInResourceSpace(uint64_t dieBase, uint64_t addr) const
    {
        if (dieBase == 0 || addr < dieBase || m_spec.resourceSpaceSize == 0) {
            return false;
        }
        return (addr - dieBase) < m_spec.resourceSpaceSize;
    }

} // namespace devspec
} // namespace sim
