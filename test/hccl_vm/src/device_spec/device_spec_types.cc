/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "device_spec/device_spec_types.h"

#include <algorithm>

namespace sim {
namespace devspec {
    namespace {
        struct ResTypeNamePair {
            CcuResType type;
            const char* name;
        };

        /// yaml 的 regions[].type 取值闭集。未知取值由加载器 fail-fast（设计文档 §4.3.1）。
        const ResTypeNamePair RES_TYPE_NAMES[] = {
            {CcuResType::INSTR, "INSTR"},     {CcuResType::CCUM, "CCUM"},   {CcuResType::GSA, "GSA"},
            {CcuResType::XN, "XN"},           {CcuResType::CKE, "CKE"},     {CcuResType::PFE, "PFE"},
            {CcuResType::CHANNEL, "CHANNEL"}, {CcuResType::JETTY, "JETTY"}, {CcuResType::MISSION, "MISSION"},
            {CcuResType::LOOP, "LOOP"},       {CcuResType::WQE, "WQE"},     {CcuResType::MS, "MS"},
        };

        struct VersionNamePair {
            CcuVersion ver;
            const char* name;
        };

        const VersionNamePair VERSION_NAMES[] = {
            {CcuVersion::CCU_V1, "CCU_V1"},
            {CcuVersion::CCU_V2, "CCU_V2"},
            {CcuVersion::CCU_V3, "CCU_V3"},
        };
    } // namespace

    const char* CcuResTypeToString(CcuResType type)
    {
        for (const auto& item : RES_TYPE_NAMES) {
            if (item.type == type) {
                return item.name;
            }
        }
        return "UNKNOWN";
    }

    bool CcuResTypeFromString(const std::string& str, CcuResType& out)
    {
        for (const auto& item : RES_TYPE_NAMES) {
            if (str == item.name) {
                out = item.type;
                return true;
            }
        }
        out = CcuResType::UNKNOWN;
        return false;
    }

    const char* CcuVersionToString(CcuVersion ver)
    {
        for (const auto& item : VERSION_NAMES) {
            if (item.ver == ver) {
                return item.name;
            }
        }
        return "CCU_INVALID";
    }

    bool CcuVersionFromString(const std::string& str, CcuVersion& out)
    {
        for (const auto& item : VERSION_NAMES) {
            if (str == item.name) {
                out = item.ver;
                return true;
            }
        }
        out = CcuVersion::CCU_INVALID;
        return false;
    }

    const RegionSpec* DeviceSpec::FindRegion(CcuResType type) const
    {
        if (type == CcuResType::UNKNOWN) {
            return nullptr;
        }
        auto it = std::find_if(regions.begin(), regions.end(), [type](const RegionSpec& r) {
            return r.type == type;
        });
        return it == regions.end() ? nullptr : &(*it);
    }

    CcuCaps DeviceSpec::GetCaps() const
    {
        // 全部由 regions / ms 推导，yaml 中不设 caps 段（设计文档 §4.2.3）。
        // 推导结果必须与 hccp_ccu_stub.cc:69-96(V1) / :98-125(V2) 的现值逐项一致（验收 A15）。
        CcuCaps caps{};
        auto countOf = [this](CcuResType type) -> uint32_t {
            const RegionSpec* r = FindRegion(type);
            return (r == nullptr) ? 0U : r->count;
        };
        caps.instructionNum = countOf(CcuResType::INSTR);
        caps.missionNum = countOf(CcuResType::MISSION);
        caps.loopEngineNum = countOf(CcuResType::LOOP);
        caps.gsaNum = countOf(CcuResType::GSA); // A6 无该条目 → 0
        caps.xnNum = countOf(CcuResType::XN);
        caps.ckeNum = countOf(CcuResType::CKE);
        caps.channelNum = countOf(CcuResType::CHANNEL);
        caps.jettyNum = countOf(CcuResType::JETTY);
        caps.pfeNum = countOf(CcuResType::PFE);
        caps.msNum = ms.msNum;
        return caps;
    }

    uint64_t DeviceSpec::MaxModelledEnd() const
    {
        uint64_t end = 0;
        for (const auto& r : regions) {
            end = std::max(end, r.offset + r.size);
        }
        for (const auto& w : ms.windows) {
            end = std::max(end, w.offset + w.size);
        }
        return end;
    }

} // namespace devspec
} // namespace sim
