/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * for the full text of the License. Description: AllGatherV语义校验实现类
 */

#include "allgather_v_semantics_checker.h"

#include <map>

#include "checker_def.h"
#include "sim_log.h"
#include "utils/dump/dump_json_utils.h"
#include "utils/error_codes.h"

namespace HcclSim {
HcclResult TaskCheckAllGatherVSemantics(
    std::map<DeviceId, RankMemorySemantics>& allRankMemSemantics, VDataDesTagInner& vDataDes,
    const std::vector<DeviceId>& rankToDevice)
{
    u32 rankSize = allRankMemSemantics.size();
    if (allRankMemSemantics.size() != rankSize) {
        HCCL_VM_ERROR(
            "{} AllGatherV rank set size mismatch: expected {}, actual {}.",
            MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), rankSize, allRankMemSemantics.size());
        return HcclResult::HCCL_E_PARA;
    }

    u64 outputSize = 0;
    // AllGatherV 输入不等长
    for (u32 i = 0; i < rankSize; i++) {
        u64 curCounts = vDataDes.counts[i];
        u64 curLength = curCounts * CHECK_SIZE_TABLE[vDataDes.dataType];
        outputSize += curLength;
    }

    for (auto& [deviceId, mem] : allRankMemSemantics) {
        u64 totalSize = 0;
        uint32_t curRankIdx = 0;
        u64 curDataSize = 0;
        for (auto& [eleAddr, ele] : mem[BufferType::OUTPUT]) {
            while (curRankIdx < rankSize && vDataDes.counts[curRankIdx] == 0) {
                ++curRankIdx;
            }
            if (curRankIdx >= rankSize) {
                HCCL_VM_ERROR(
                    "{} AllGatherV output for device {} contains unexpected "
                    "data after all "
                    "expected source ranks have been consumed."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SIZE_ERROR), deviceId, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }
            const u64 inputSize = vDataDes.counts[curRankIdx] * CHECK_SIZE_TABLE[vDataDes.dataType];
            const u64 rangeEnd = ele.startAddr + ele.size;

            if (ele.startAddr != totalSize) {
                HCCL_VM_ERROR(
                    "{} AllGatherV output for device {} should continue at "
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
                    "{} AllGatherV output range [0x{:x},0x{:x}) for device {} "
                    "should come "
                    "from exactly one source, but it actually comes from {} "
                    "sources."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_REDUCE_ERROR), ele.startAddr, rangeEnd, deviceId,
                    ele.srcBufs.size(), ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            const auto& srcBuf = *ele.srcBufs.begin();

            if (srcBuf.deviceId != rankToDevice[curRankIdx]) {
                HCCL_VM_ERROR(
                    "{} AllGatherV output range [0x{:x},0x{:x}) for device {} "
                    "should come from "
                    "source index {}, but it actually comes from device {}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, deviceId,
                    curRankIdx, srcBuf.deviceId, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (srcBuf.bufType != BufferType::INPUT) {
                HCCL_VM_ERROR(
                    "{} AllGatherV output range [0x{:x},0x{:x}) for device {} "
                    "should come from "
                    "INPUT, but it actually comes from device {} with buffer "
                    "type {}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, deviceId,
                    srcBuf.deviceId, BufferTypeToString(srcBuf.bufType), ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (srcBuf.srcAddr != curDataSize) {
                HCCL_VM_ERROR(
                    "{} AllGatherV output range [0x{:x},0x{:x}) for device {} "
                    "should come from "
                    "source index {} at source address 0x{:x}, but the actual "
                    "source address is 0x{:x}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, deviceId,
                    srcBuf.deviceId, curDataSize, srcBuf.srcAddr, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            curDataSize += ele.size;

            if (curDataSize == inputSize) {
                curDataSize = 0;
                curRankIdx++;
            } else if (curDataSize > inputSize) {
                HCCL_VM_ERROR(
                    "{} AllGatherV data collected from source index {} for "
                    "device {} becomes larger "
                    "than expected after outputRange [0x{:x},0x{:x}). The "
                    "accumulated size is 0x{:x}, "
                    "but the expected size from this source is 0x{:x}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SIZE_ERROR), curRankIdx, deviceId, ele.startAddr,
                    rangeEnd, curDataSize, inputSize, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }
            totalSize += ele.size;
        }
        if (totalSize != outputSize) {
            HCCL_VM_ERROR(
                "{} AllGatherV output for device {} ends too early: "
                "the checker validated "
                "0x{:x} bytes in total, but the expected total "
                "result size is 0x{:x} bytes.",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), deviceId, totalSize, outputSize);
            return HcclResult::HCCL_E_PARA;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}
} // namespace HcclSim
