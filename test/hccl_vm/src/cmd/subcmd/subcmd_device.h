/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_VM_DEVICE_COMMAND_H
#define HCCL_VM_DEVICE_COMMAND_H

#include <cstdint>
#include <string>

#include "cmd_base.h"

namespace HcclSim {

/**
 * @brief `hccl-vm device` 子命令：device 规格查询与地址编解码。
 *
 * 设计文档 §4.5.8。用途：
 *   - spec show   : 打印解析后的规格 + 加载器校验结果（排查"配置为什么没生效"）
 *   - spec export : 导出 JSON，供非 C++/Python 的第三方消费
 *   - addr decode : 设备 VA → (rank, die, region, id)，排查"这个地址从哪来"
 *   - addr encode : (rank, die, region, id) → 设备 VA，逆运算
 *                   （user_mem 改为分段映射后人工算 VA 变难，此命令价值上升）
 */
class DeviceCommand : public CommandBase {
public:
    static std::string StaticName() { return "device"; }
    void Setup(CLI::App& app) override;

private:
    bool EnsureLoaded();
    void ShowSpec();
    void ExportSpec();
    void DecodeAddr();
    void EncodeAddr();

    std::string m_specDir; // --dir：覆盖 config/device 目录（开发期用）
    std::string m_soc;     // spec show [soc] / spec export <soc>
    std::string m_outFile; // spec export -o
    std::string m_addrStr; // addr decode <VA>
    std::string m_resType; // addr encode --type
    uint64_t m_resId{0};   // addr encode --id
    uint32_t m_rank{0};    // addr encode --rank / decode 结果展示
    uint32_t m_die{0};     // addr encode --die
};

} // namespace HcclSim

#endif // HCCL_VM_DEVICE_COMMAND_H
