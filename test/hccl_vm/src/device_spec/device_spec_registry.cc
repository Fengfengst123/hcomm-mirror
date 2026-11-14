/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "device_spec/device_spec_registry.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <dirent.h>
#include <set>
#include <sys/stat.h>

#include "device_spec_yaml_loader.h"
#include "device_spec/ccu_address_space.h"
#include "device_spec/rank_address_space.h"
#include "sim_common_api.h"
#include "sim_log.h"

#if !defined(NO_YAML_CONFIG) && defined(HAVE_YAML_CPP)
#include "yaml-cpp/yaml.h"
#endif

namespace sim {
namespace devspec {
    namespace {
        constexpr const char* DEVICE_CONFIG_REL_DIR = "config/device";
        constexpr const char* YAML_SUFFIX = ".yaml";

        std::string Lower(const std::string& s)
        {
            std::string r = s;
            std::transform(r.begin(), r.end(), r.begin(), [](unsigned char c) {
                return static_cast<char>(tolower(c));
            });
            return r;
        }

        bool IsRegularFile(const std::string& path)
        {
            struct stat st {};
            return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
        }

        bool EndsWith(const std::string& s, const std::string& suffix)
        {
            return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
        }

        /// 去掉 .yaml 后缀，得到文件名主干。
        std::string StemOf(const std::string& fileName)
        {
            if (!EndsWith(fileName, YAML_SUFFIX)) {
                return fileName;
            }
            return fileName.substr(0, fileName.size() - strlen(YAML_SUFFIX));
        }

        /**
         * 文件名与 spec.name 的一致性检查。
         *
         * 匹配规则以 name 为准（大小写不敏感最长前缀），故文件名必须能体现出 name，
         * 否则会出现"文件名说 A、内容说 B"的割裂。允许文件名带前缀（如参考实现的
         * device_spec_yaml_demo_ascend950.yaml），但前缀与 name 之间必须有 '_' 或 '-' 分隔，
         * 防止 xascend950.yaml 这类误匹配。正式交付名为 ascend950.yaml，同样满足。
         */
        bool FileNameMatchesSpecName(const std::string& stem, const std::string& name)
        {
            const std::string s = Lower(stem);
            const std::string n = Lower(name);
            if (n.empty()) {
                return false;
            }
            if (s == n) {
                return true;
            }
            if (s.size() <= n.size() || !EndsWith(s, n)) {
                return false;
            }
            const char sep = s[s.size() - n.size() - 1];
            return sep == '_' || sep == '-';
        }

        /// 列出目录下全部 *.yaml 的绝对路径（按文件名升序，保证加载顺序确定）。
        bool ListYamlFiles(const std::string& dir, std::vector<std::string>& out, std::string& err)
        {
            out.clear();
            DIR* dp = ::opendir(dir.c_str());
            if (dp == nullptr) {
                err = "无法打开 device 配置目录: " + dir;
                return false;
            }
            std::vector<std::string> names;
            struct dirent* ent = nullptr;
            while ((ent = ::readdir(dp)) != nullptr) {
                const std::string n = ent->d_name;
                if (n == "." || n == "..") {
                    continue;
                }
                if (!EndsWith(n, YAML_SUFFIX)) {
                    continue;
                }
                const std::string full = dir + "/" + n;
                if (IsRegularFile(full)) {
                    names.push_back(full);
                }
            }
            ::closedir(dp);
            std::sort(names.begin(), names.end());
            out = names;
            return true;
        }

#if !defined(NO_YAML_CONFIG) && defined(HAVE_YAML_CPP)
        /// 从拓扑配置 yaml 中递归收集所有 soc_version 字面量（校验 9 用）。
        void CollectSocVersions(const YAML::Node& node, std::vector<std::string>& out)
        {
            if (!node) {
                return;
            }
            if (node.IsMap()) {
                for (auto it = node.begin(); it != node.end(); ++it) {
                    const std::string key = it->first.as<std::string>("");
                    if (key == "soc_version" && it->second.IsScalar()) {
                        const std::string v = it->second.as<std::string>("");
                        if (!v.empty()) {
                            out.push_back(v);
                        }
                    } else {
                        CollectSocVersions(it->second, out);
                    }
                }
                return;
            }
            if (node.IsSequence()) {
                for (const auto& item : node) {
                    CollectSocVersions(item, out);
                }
            }
        }
#endif
    } // namespace

    DeviceSpecRegistry& DeviceSpecRegistry::GetInstance()
    {
        static DeviceSpecRegistry instance;
        return instance;
    }

    bool DeviceSpecRegistry::LoadFromFile(const std::string& absPath)
    {
        DeviceSpec spec;
        std::string err;
        if (!LoadDeviceSpecYaml(absPath, spec, err)) {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_lastError = err;
            HCCL_VM_ERROR("load device spec failed: {}", err);
            return false;
        }
        std::lock_guard<std::mutex> lock(m_mutex);
        const std::string key = Lower(spec.name);
        if (m_specs.count(key) != 0) {
            m_lastError = "device spec 名重复: " + spec.name + "（已加载自 " + m_specs[key]->sourcePath
                          + "，又尝试加载 " + absPath + "）";
            HCCL_VM_ERROR("{}", m_lastError);
            return false;
        }
        // 文件名必须能体现 name 字段，否则匹配规则（name 最长前缀）会与文件名脱节
        const std::string stem = StemOf(absPath.substr(absPath.find_last_of('/') + 1));
        if (!FileNameMatchesSpecName(stem, spec.name)) {
            m_lastError = "device spec 的 name(\"" + spec.name + "\") 与文件名(\"" + stem
                          + "\") 不一致；匹配规则以 name 为准，文件名须等于 name 或以 _/- 分隔的前缀形式包含它";
            HCCL_VM_ERROR("{}", m_lastError);
            return false;
        }
        m_specs[key] = std::make_shared<DeviceSpec>(std::move(spec));
        m_lastError.clear();
        return true;
    }

    bool DeviceSpecRegistry::LoadAll(const std::string& dirOverride)
    {
        const std::string dir
            = dirOverride.empty() ? InstallPath::ResolveToInstallRoot(DEVICE_CONFIG_REL_DIR) : dirOverride;
        std::vector<std::string> files;
        std::string err;
        if (!ListYamlFiles(dir, files, err)) {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_lastError = err;
            HCCL_VM_ERROR("{}", err);
            return false;
        }
        if (files.empty()) {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_lastError = "device 配置目录为空: " + dir + "（期望至少一份 ascend*.yaml）";
            HCCL_VM_ERROR("{}", m_lastError);
            return false;
        }

        // 先加载到临时容器，全部成功后再原子替换，避免留下半成品状态
        std::map<std::string, std::shared_ptr<DeviceSpec>> loaded;
        for (const auto& f : files) {
            DeviceSpec spec;
            if (!LoadDeviceSpecYaml(f, spec, err)) {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_lastError = err;
                HCCL_VM_ERROR("load device spec failed: {}", err);
                return false;
            }
            const std::string key = Lower(spec.name);
            const std::string stem = StemOf(f.substr(f.find_last_of('/') + 1));
            if (!FileNameMatchesSpecName(stem, spec.name)) {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_lastError = f + ": name(\"" + spec.name + "\") 与文件名(\"" + stem
                              + "\") 不一致；文件名须等于 name 或以 _/- 分隔的前缀形式包含它";
                HCCL_VM_ERROR("{}", m_lastError);
                return false;
            }
            if (loaded.count(key) != 0) {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_lastError = "device spec 名重复: " + spec.name;
                HCCL_VM_ERROR("{}", m_lastError);
                return false;
            }
            loaded[key] = std::make_shared<DeviceSpec>(std::move(spec));
        }

        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_specs = std::move(loaded);
            m_lastError.clear();
        }
        HCCL_VM_INFO("device spec registry loaded {} spec(s) from {}", m_specs.size(), dir);
        return true;
    }

    void DeviceSpecRegistry::Reset()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_specs.clear();
        m_lastError.clear();
    }

    const DeviceSpec* DeviceSpecRegistry::FindByName(const std::string& specName) const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_specs.find(Lower(specName));
        return (it == m_specs.end()) ? nullptr : it->second.get();
    }

    const DeviceSpec* DeviceSpecRegistry::FindBySocVersion(const std::string& socVersion) const
    {
        if (socVersion.empty()) {
            m_lastError = "FindBySocVersion: soc_version 为空";
            return nullptr;
        }
        const std::string lowerSoc = Lower(socVersion);
        std::lock_guard<std::mutex> lock(m_mutex);

        // 大小写不敏感最长前缀匹配（设计文档 §4.2.1）
        // ★ 不使用 yaml 内的 soc.soc_version 字段（决策 (i)，加载器完全不解析该 key）
        size_t bestLen = 0;
        std::vector<const DeviceSpec*> best;
        for (const auto& kv : m_specs) {
            const std::string& name = kv.first; // 已小写化
            if (name.empty() || lowerSoc.compare(0, name.size(), name) != 0) {
                continue;
            }
            if (name.size() > bestLen) {
                bestLen = name.size();
                best.clear();
                best.push_back(kv.second.get());
            } else if (name.size() == bestLen) {
                best.push_back(kv.second.get());
            }
        }

        if (best.empty()) {
            std::string candidates;
            for (const auto& kv : m_specs) {
                if (!candidates.empty()) {
                    candidates += ", ";
                }
                candidates += kv.second->name;
            }
            m_lastError = "soc_version \"" + socVersion + "\" 未命中任何 device spec；可用 spec: [" + candidates + "]";
            HCCL_VM_ERROR("{}", m_lastError);
            return nullptr;
        }
        if (best.size() > 1) {
            std::string names;
            for (const auto* s : best) {
                if (!names.empty()) {
                    names += ", ";
                }
                names += s->name;
            }
            m_lastError = "soc_version \"" + socVersion + "\" 命中多份等长 device spec: [" + names
                          + "]，无法确定唯一匹配（fail-fast，不做静默 fallback）";
            HCCL_VM_ERROR("{}", m_lastError);
            return nullptr;
        }
        return best.front();
    }

    std::vector<const DeviceSpec*> DeviceSpecRegistry::All() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<const DeviceSpec*> out;
        out.reserve(m_specs.size());
        for (const auto& kv : m_specs) { // std::map 已按 key（小写 name）升序
            out.push_back(kv.second.get());
        }
        return out;
    }

    const AddrSpace* DeviceSpecRegistry::CommonAddrSpace() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_specs.empty()) {
            return nullptr;
        }
        const AddrSpace* ref = nullptr;
        for (const auto& kv : m_specs) {
            const AddrSpace& a = kv.second->addr;
            if (ref == nullptr) {
                ref = &a;
                continue;
            }
            // 逐字段比较（含派生量），任一不同即无公共地址空间
            if (ref->rankBase != a.rankBase || ref->rankStride != a.rankStride
                || ref->rankWindowSize != a.rankWindowSize || ref->ccuWindowOffset != a.ccuWindowOffset
                || ref->ccuWindowSize != a.ccuWindowSize || ref->ccuDieSize != a.ccuDieSize
                || ref->ccuDieOffsets != a.ccuDieOffsets || ref->userMemTotal != a.userMemTotal
                || ref->userMemSegs.size() != a.userMemSegs.size() || ref->partitions.size() != a.partitions.size()) {
                return nullptr;
            }
            for (size_t i = 0; i < a.userMemSegs.size(); ++i) {
                if (ref->userMemSegs[i].offset != a.userMemSegs[i].offset
                    || ref->userMemSegs[i].size != a.userMemSegs[i].size) {
                    return nullptr;
                }
            }
        }
        return ref;
    }

    size_t DeviceSpecRegistry::Size() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_specs.size();
    }

    std::string DeviceSpecRegistry::LastError() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_lastError;
    }

    std::unique_ptr<RankAddressSpace> DeviceSpecRegistry::MakeRankAddressSpace(const DeviceSpec* spec) const
    {
        if (spec == nullptr) {
            return nullptr;
        }
        return std::unique_ptr<RankAddressSpace>(new RankAddressSpace(spec->addr));
    }

    std::unique_ptr<CcuAddressSpace> DeviceSpecRegistry::MakeCcuAddressSpace(const DeviceSpec* spec) const
    {
        if (spec == nullptr) {
            return nullptr;
        }
        return std::unique_ptr<CcuAddressSpace>(new CcuAddressSpace(*spec));
    }

    bool DeviceSpecRegistry::CheckSocVersionCoverage(
        const std::string& configRoot, std::vector<std::string>& violations) const
    {
        violations.clear();
#if !defined(NO_YAML_CONFIG) && defined(HAVE_YAML_CPP)
        const std::string root = configRoot.empty() ? InstallPath::ResolveToInstallRoot("config") : configRoot;
        const std::string subDirs[] = {"cluster", "server_or_pod"};

        // 1) 收集拓扑配置里出现的全部 soc_version
        std::set<std::string> socValues;
        for (const auto& sub : subDirs) {
            const std::string dir = root + "/" + sub;
            std::vector<std::string> files;
            std::string err;
            if (!ListYamlFiles(dir, files, err)) {
                HCCL_VM_WARN("校验9: 跳过不存在的目录 {}", dir);
                continue;
            }
            for (const auto& f : files) {
                try {
                    const YAML::Node doc = YAML::LoadFile(f);
                    std::vector<std::string> found;
                    CollectSocVersions(doc, found);
                    for (const auto& v : found) {
                        socValues.insert(v);
                    }
                } catch (const std::exception& e) {
                    violations.push_back(f + ": YAML 解析失败: " + e.what());
                }
            }
        }

        if (socValues.empty()) {
            violations.push_back(
                "校验9: 在 " + root + "/{cluster,server_or_pod} 下未找到任何 soc_version，无法校验覆盖性");
            return false;
        }

        // 2) 正向：每个 soc_version 必须唯一命中一份 spec
        std::set<std::string> usedSpecs;
        for (const auto& soc : socValues) {
            const DeviceSpec* spec = FindBySocVersion(soc);
            if (spec == nullptr) {
                violations.push_back("校验9: soc_version \"" + soc + "\" 未唯一命中任何 device spec —— " + LastError());
                continue;
            }
            usedSpecs.insert(Lower(spec->name));
        }

        // 3) 反向：每份 spec 至少被一个 soc_version 命中（防 spec 腐化）
        for (const auto* spec : All()) {
            if (usedSpecs.count(Lower(spec->name)) == 0) {
                violations.push_back(
                    "校验9: device spec \"" + spec->name
                    + "\" 未被任何拓扑配置的 soc_version 命中（疑似废弃或命名不匹配）");
            }
        }

        if (!violations.empty()) {
            for (const auto& v : violations) {
                HCCL_VM_ERROR("{}", v);
            }
            return false;
        }
        HCCL_VM_INFO("校验9 通过: {} 个 soc_version 值唯一命中 {} 份 device spec", socValues.size(), usedSpecs.size());
        return true;
#else
        (void)configRoot;
        violations.push_back("校验9: 编译时未启用 yaml-cpp，无法执行 soc_version 覆盖性校验");
        return false;
#endif
    }

} // namespace devspec
} // namespace sim
