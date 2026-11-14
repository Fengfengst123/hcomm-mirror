/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_DEVICE_SPEC_CCU_ADDRESS_SPACE_H
#define SIM_DEVICE_SPEC_CCU_ADDRESS_SPACE_H

#include "device_spec/device_spec_types.h"

namespace sim {
namespace devspec {

    /**
     * @brief CCU die 内的 地址 ↔ 资源 id 转换（纯计算，无 IO、无副作用）。
     *
     * 本类是【唯一】实现，取代此前并存的 4 份：
     *   - runner  ccu_resource_manager.cc:378-531（常量块 + findBaseAddr/findTypeByAddr/findBaseSize
     *             + GetMSIdByAddr/GetXnAndTypeIdByAddr/GetXnIdByAddr/GetAddrByXnId）
     *   - checker ccu_all_rank_param_recorder_v3.cc:259-445（同名同体的复制品）
     *   - proxy L1 ccu_level1_res.cc:504-578（只有正向 id→addr）
     *   - light_performance ccu_graph_builder.py / register_model.py（Python 重写）
     *
     * ★ 与旧实现的差异：全部方法带完整边界检查（id < count、addr 落在区域内），
     *   修掉 ccu_resource_manager.cc:512,522 的 "需要判断界限 todo"——
     *   旧 runner 侧无上界校验而 checker 侧有，两份实现健壮性不一致。
     */
    class CcuAddressSpace {
    public:
        explicit CcuAddressSpace(const DeviceSpec& spec);

        const DeviceSpec& Spec() const { return m_spec; }

        /// 偏移（相对 ccuDieBase）→ 区域类型。未命中任何区域返回 CcuResType::UNKNOWN。
        /// 取代 findTypeByAddr。
        CcuResType TypeByOffset(uint64_t offset) const;

        /// 区域的起始偏移（相对 ccuDieBase）。未命中返回 0 且 ok 置 false。取代 findBaseAddr。
        uint64_t RegionOffset(CcuResType type, bool& ok) const;
        /// 单个资源的字节步进。未命中返回 0 且 ok 置 false。取代 findBaseSize。
        uint32_t EntrySize(CcuResType type, bool& ok) const;
        /// 区域容量（可分配的最大 id 数）。未命中返回 0 且 ok 置 false。
        uint32_t RegionCount(CcuResType type, bool& ok) const;

        /**
         * @brief 设备地址 → (区域类型, 资源 id)。
         * @param dieBase 该 (rank, die) 的 CCU 基址，由 RankAddressSpace::CcuDieBase 得到
         * @param addr    指令中携带的资源地址
         * @param type    [out] 命中的区域类型
         * @param id      [out] 资源 id
         * @return addr 不落在任何已建模区域内、或落在区域内但超出 count 上界时返回 false
         *
         * 取代 runner/checker 的 GetXnAndTypeIdByAddr。
         */
        bool AddrToResId(uint64_t dieBase, uint64_t addr, CcuResType& type, uint64_t& id) const;

        /**
         * @brief (区域类型, 资源 id) → 设备地址。
         * @return id ≥ 区域 count、或该类型无对应区域时返回 false
         *
         * 取代 runner/checker 的 GetAddrByXnId。
         */
        bool ResIdToAddr(uint64_t dieBase, CcuResType type, uint64_t id, uint64_t& addr) const;

        /**
         * @brief 已知类型时的 地址 → id（不做类型推断）。
         * @return addr 不在该类型区域内、或换算出的 id ≥ count 时返回 false
         *
         * 取代 runner/checker 的 GetXnIdByAddr。
         */
        bool AddrToIdOfType(uint64_t dieBase, CcuResType type, uint64_t addr, uint64_t& id) const;

        /// MS 设备地址 → msId（含 4-bank × interleave 去交织）。取代 GetMSIdByAddr。
        bool MsAddrToId(uint64_t dieBase, uint64_t addr, uint32_t& msId) const;
        /// msId → MS 设备地址（交织正向）。旧实现缺失此方向，此处补齐。
        bool MsIdToAddr(uint64_t dieBase, uint32_t msId, uint64_t& addr) const;

        /// addr 是否落在该 die 的 CCU 资源空间内（[dieBase, dieBase + resourceSpaceSize)）。
        bool IsInResourceSpace(uint64_t dieBase, uint64_t addr) const;

    private:
        const RegionSpec* FindRegion(CcuResType type) const;
        const MsWindow* FindMsWindow(uint64_t offset) const;

        const DeviceSpec& m_spec;
    };

} // namespace devspec
} // namespace sim

#endif // SIM_DEVICE_SPEC_CCU_ADDRESS_SPACE_H
