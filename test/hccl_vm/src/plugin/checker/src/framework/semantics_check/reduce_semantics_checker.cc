/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_semantics_checker.h"

#include <map>

#include "base.h"
#include "check_utils.h"
#include "sim_log.h"
#include "utils/dump/dump_json_utils.h"
#include "utils/error_codes.h"

namespace HcclSim {
HcclResult TaskCheckReduceSemantics(
    std::map<DeviceId, RankMemorySemantics>& allRankMemSemantics, u64 dataSize, HcclReduceOp reduceType,
    DeviceId rootDeviceId, const std::vector<DeviceId>& rankToDevice)
{
    u32 rankSize = allRankMemSemantics.size();
    if (rankSize == 0) {
        return HCCL_SUCCESS;
    }
    if (allRankMemSemantics.size() != rankSize) {
        HCCL_VM_ERROR(
            "{} Reduce rank set size mismatch: expected {}, actual {}.",
            MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), rankSize, allRankMemSemantics.size());
        return HcclResult::HCCL_E_PARA;
    }

    bool foundRoot = false;
    for (auto& [deviceId, mem] : allRankMemSemantics) {
        if (deviceId != rootDeviceId) {
            continue;
        }
        foundRoot = true;

        if (dataSize == 0) {
            break;
        }

        u64 totalSize = 0;
        for (auto& [eleAddr, ele] : mem[BufferType::OUTPUT]) {
            const u64 rangeEnd = ele.startAddr + ele.size;
            if (ele.startAddr != totalSize) {
                HCCL_VM_ERROR(
                    "{} Reduce output for root {} should continue at 0x{:x}, "
                    "but the next actual range starts at 0x{:x} (actual range: "
                    "[0x{:x},0x{:x}))."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), rootDeviceId, totalSize, ele.startAddr,
                    ele.startAddr, rangeEnd, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.size() > 1 && ele.reduceType != reduceType) {
                HCCL_VM_ERROR(
                    "{} Reduce result range [0x{:x},0x{:x}) for root {} was "
                    "reduced with mode {}, "
                    "but the operator expects reduce mode {}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_REDUCE_ERROR), ele.startAddr, rangeEnd, rootDeviceId,
                    DumpReduceOpToString(ele.reduceType), DumpReduceOpToString(reduceType), ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.size() != rankSize) {
                HCCL_VM_ERROR(
                    "{} Reduce result range [0x{:x},0x{:x}) for root {} should "
                    "combine inputs "
                    "from {} source ranks, but it actually combines {}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_REDUCE_ERROR), ele.startAddr, rangeEnd, rootDeviceId,
                    rankSize, ele.srcBufs.size(), ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (!CoversAllRankDevices(ele.srcBufs, rankToDevice)) {
                HCCL_VM_ERROR(
                    "{} Reduce result range [0x{:x},0x{:x}) for root {} "
                    "expected one "
                    "source from each device {} in rank order, but got sources "
                    "from devices {}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, rootDeviceId,
                    DeviceIdListToString(rankToDevice), SrcBufDeviceListToString(ele.srcBufs), ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            for (auto& srcBuf : ele.srcBufs) {
                if (srcBuf.bufType != BufferType::INPUT) {
                    HCCL_VM_ERROR(
                        "{} Reduce result range [0x{:x},0x{:x}) for root {} "
                        "should come from INPUT, "
                        "but source device {} actually provides buffer type {}."
                        "\nCurrent result range detail:\n{}",
                        MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, rootDeviceId,
                        srcBuf.deviceId, BufferTypeToString(srcBuf.bufType), ele.Describe());
                    return HcclResult::HCCL_E_PARA;
                }

                if (srcBuf.srcAddr != totalSize) {
                    HCCL_VM_ERROR(
                        "{} Reduce result range [0x{:x},0x{:x}) for root {} "
                        "should read "
                        "source device {} at 0x{:x}, but the actual source "
                        "address is 0x{:x}."
                        "\nCurrent result range detail:\n{}",
                        MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, rootDeviceId,
                        srcBuf.deviceId, totalSize, srcBuf.srcAddr, ele.Describe());
                    return HcclResult::HCCL_E_PARA;
                }
            }
            totalSize += ele.size;
        }
        if (totalSize != dataSize) {
            HCCL_VM_ERROR(
                "{} Reduce result for root {} ends at 0x{:x} bytes, "
                "but the expected total size is 0x{:x}.",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), rootDeviceId, totalSize, dataSize);
            return HcclResult::HCCL_E_PARA;
        }
    }

    if (!foundRoot && dataSize > 0) {
        HCCL_VM_ERROR(
            "{} Reduce produced no result data for root rank {}, but "
            "this rank is expected to hold "
            "a full reduced result of 0x{:x} bytes from all {} "
            "participating ranks.",
            MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), rootDeviceId, dataSize, rankSize);
        return HcclResult::HCCL_E_PARA;
    }

    return HcclResult::HCCL_SUCCESS;
}
} // namespace HcclSim
