/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_DEVICE_SPEC_YAML_LOADER_H
#define SIM_DEVICE_SPEC_YAML_LOADER_H

#include <string>

#include "device_spec/device_spec_types.h"

namespace sim {
namespace devspec {

    /**
     * @brief 解析并校验单个 device spec yaml。
     *
     * @param absPath yaml 绝对路径
     * @param out     [out] 解析结果；失败时内容未定义
     * @param err     [out] 失败原因（含完整 key 路径，便于定位）
     * @return 解析 + 全部校验通过返回 true
     *
     * 校验清单见设计文档 §4.3.1（1~5、7~9；6 已取消）。校验 7（VA 位宽）与
     * 校验 9（soc_version 覆盖）依赖运行时环境/其它配置文件，不在本函数内执行，
     * 分别由 DeviceSpecRegistry::CheckVaBitWidth / CheckSocVersionCoverage 承担。
     */
    bool LoadDeviceSpecYaml(const std::string& absPath, DeviceSpec& out, std::string& err);

} // namespace devspec
} // namespace sim

#endif // SIM_DEVICE_SPEC_YAML_LOADER_H
