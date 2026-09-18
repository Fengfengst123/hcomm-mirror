/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "scatter_semantics_checker.h"

#include <map>

#include "base.h"
#include "check_utils.h"
#include "sim_log.h"
#include "utils/dump/dump_json_utils.h"
#include "utils/error_codes.h"

namespace HcclSim {
HcclResult TaskCheckScatterSemantics(
    std::map<DeviceId, RankMemorySemantics>& allRankMemSemantics, u64 dataSize, DeviceId rootDeviceId,
    const std::vector<DeviceId>& rankToDevice)
{
    u32 rankSize = allRankMemSemantics.size();
    if (allRankMemSemantics.size() != rankSize) {
        HCCL_VM_ERROR(
            "{} Scatter rank set size mismatch: expected {}, actual {}.",
            MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), rankSize, allRankMemSemantics.size());
        return HcclResult::HCCL_E_PARA;
    }

    for (auto& [deviceId, mem] : allRankMemSemantics) {
        const RankId deviceRank = FindRankIndexByDeviceId(rankToDevice, deviceId);
        u64 totalSize = 0;
        for (auto& [eleAddr, ele] : mem[BufferType::OUTPUT]) {
            const u64 rangeEnd = ele.startAddr + ele.size;
            if (ele.startAddr != totalSize) {
                HCCL_VM_ERROR(
                    "{} Scatter output for device {} should continue at "
                    "0x{:x}, "
                    "but the next actual range starts at 0x{:x} (actual range: "
                    "[0x{:x},0x{:x}))."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), deviceId, totalSize, ele.startAddr,
                    ele.startAddr, rangeEnd, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.size() != 1) {
                HCCL_VM_ERROR(
                    "{} Scatter output range [0x{:x},0x{:x}) for device {} "
                    "should "
                    "come from exactly one source, but it actually comes from "
                    "{} sources."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_REDUCE_ERROR), ele.startAddr, rangeEnd, deviceId,
                    ele.srcBufs.size(), ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            const auto& srcBuf = *ele.srcBufs.begin();
            if (srcBuf.deviceId != rootDeviceId) {
                HCCL_VM_ERROR(
                    "{} Scatter output range [0x{:x},0x{:x}) for device {} "
                    "should come "
                    "from root {}, but it actually comes from device {}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, deviceId,
                    rootDeviceId, srcBuf.deviceId, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (srcBuf.bufType != BufferType::INPUT) {
                HCCL_VM_ERROR(
                    "{} Scatter output range [0x{:x},0x{:x}) for device {} "
                    "should come "
                    "from INPUT, but it actually comes from device {} with "
                    "buffer type {}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, deviceId,
                    srcBuf.deviceId, BufferTypeToString(srcBuf.bufType), ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (srcBuf.srcAddr != deviceRank * dataSize + totalSize) {
                HCCL_VM_ERROR(
                    "{} Scatter output range [0x{:x},0x{:x}) for device {} "
                    "should come "
                    "from root {} at source address 0x{:x}, but it actually "
                    "comes from address 0x{:x}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, deviceId,
                    srcBuf.deviceId, deviceRank * dataSize + totalSize, srcBuf.srcAddr, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }
            totalSize += ele.size;
        }
        if (totalSize != dataSize) {
            HCCL_VM_ERROR(
                "{} Scatter output for device {} ends too early. The "
                "checker has "
                "validated 0x{:x} bytes in total, but the expected "
                "size is 0x{:x}.",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), deviceId, totalSize, dataSize);
            return HcclResult::HCCL_E_PARA;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}
} // namespace HcclSim
