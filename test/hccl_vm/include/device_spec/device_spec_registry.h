/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_DEVICE_SPEC_REGISTRY_H
#define SIM_DEVICE_SPEC_REGISTRY_H

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "device_spec/device_spec_types.h"

namespace sim {
namespace devspec {

    class RankAddressSpace;
    class CcuAddressSpace;

    /**
     * @brief device 规格注册表（进程级单例，加载后只读）。
     *
     * 数据源：asset/cluster_model/config/device/*.yaml，
     * 安装后经 CMake config_hcclvm_install 落到 <install_root>/config/device/。
     *
     * ★ 阶段 0 匹配规则（设计文档 §4.2.1 / §4.7.4 决策 (i)）：
     *   由 spec 的 name（= 文件名）对拓扑配置里的 soc_version 做【大小写不敏感最长前缀】匹配。
     *   yaml 内的 soc.soc_version 字段【完全不解析】——不参与匹配、不参与校验、不落库。
     *
     * 线程安全：LoadAll/LoadFromFile 需在读之前完成（start 阶段单线程）；
     * 加载完成后所有查询均为只读，可并发调用。
     */
    class DeviceSpecRegistry {
    public:
        static DeviceSpecRegistry& GetInstance();

        DeviceSpecRegistry(const DeviceSpecRegistry&) = delete;
        DeviceSpecRegistry& operator=(const DeviceSpecRegistry&) = delete;

        /**
         * @brief 扫描 <installRoot>/config/device/*.yaml 并全部加载 + 校验。
         * @param dirOverride 非空时改用该目录（UT 用）；为空时走 InstallPath::ResolveToInstallRoot
         * @return 全部成功返回 true；任一文件解析/校验失败即 fail-fast 返回 false，
         *         失败原因见 LastError()。已加载的内容会被清空（不留下半成品状态）。
         */
        bool LoadAll(const std::string& dirOverride = "");

        /// 加载单个 yaml 文件（UT / 第三方工具用）。同样执行全部校验。
        bool LoadFromFile(const std::string& absPath);

        /// 清空（UT 用）。
        void Reset();

        /// 按 spec 名（= yaml 文件名去掉 .yaml）精确查找。未命中返回 nullptr。
        const DeviceSpec* FindByName(const std::string& specName) const;

        /**
         * @brief 按拓扑配置里的 soc_version 查找 spec（阶段 0 的唯一匹配入口）。
         *
         * 规则：lower(socVersion).startswith(lower(spec.name))；多份命中取 name 最长者；
         * 最长仍并列 → 返回 nullptr 并置 LastError（fail-fast，不做静默 fallback）。
         */
        const DeviceSpec* FindBySocVersion(const std::string& socVersion) const;

        /// 全部已加载的 spec（按 name 升序）。
        std::vector<const DeviceSpec*> All() const;

        /**
         * @brief 若所有已加载 spec 的 address_space 完全一致，返回该公共地址空间；否则返回 nullptr。
         *
         * 用途：`DeviceMemoryManager::AllocVirMem` 在无法通过 DB 把 devPhyId 映射到具体
         * soc_version 时（例如 DB 尚未就绪）的兜底。A5/A6 的 address_space 按设计逐字段
         * 相同（芯片无关，验收 A14 有 UT 断言），故此时选哪份 spec 都不影响地址计算。
         * 一旦未来出现 address_space 不同的芯片，本函数返回 nullptr，调用方须自行降级。
         */
        const AddrSpace* CommonAddrSpace() const;

        size_t Size() const;
        std::string LastError() const;

        /**
         * @brief 该 spec 的 rank 地址空间计算器。
         * @return spec 为 nullptr 时返回 nullptr。返回对象轻量（仅持一个引用），可按值使用。
         */
        std::unique_ptr<RankAddressSpace> MakeRankAddressSpace(const DeviceSpec* spec) const;
        /// 该 spec 的 CCU 地址空间计算器。spec 为 nullptr 时返回 nullptr。
        std::unique_ptr<CcuAddressSpace> MakeCcuAddressSpace(const DeviceSpec* spec) const;

        /**
         * @brief 校验 9：扫描 cluster/*.yaml + server_or_pod/*.yaml 中出现的全部 soc_version，
         *        断言每个值都能唯一命中一份已加载的 spec，且每份 spec 至少被一个值命中。
         * @param configRoot <install_root>/config；为空时用 InstallPath 推导
         * @param violations [out] 违规描述列表（未命中 / 并列命中 / spec 未被使用）
         * @return 无违规返回 true
         *
         * 注：本校验只读拓扑 yaml 的 soc_version 字面量，不读 device yaml 的 soc.soc_version
         *     （决策 (i)）。
         */
        bool CheckSocVersionCoverage(const std::string& configRoot, std::vector<std::string>& violations) const;

    private:
        DeviceSpecRegistry() = default;
        ~DeviceSpecRegistry() = default;

        /// key = spec.name（小写化，用于大小写不敏感查找）
        std::map<std::string, std::shared_ptr<DeviceSpec>> m_specs;
        mutable std::mutex m_mutex;
        /// const 查询（FindBySocVersion / CheckSocVersionCoverage）失败时需回填原因，故为 mutable
        mutable std::string m_lastError;
    };

} // namespace devspec
} // namespace sim

#endif // SIM_DEVICE_SPEC_REGISTRY_H
