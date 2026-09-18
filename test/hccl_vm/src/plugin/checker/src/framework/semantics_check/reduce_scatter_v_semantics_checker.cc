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
 * for the full text of the License. Description: ReduceScatterV语义校验实现类
 */

#include "reduce_scatter_v_semantics_checker.h"

#include <map>

#include "base.h"
#include "checker_def.h"
#include "sim_log.h"
#include "utils/dump/dump_json_utils.h"
#include "utils/error_codes.h"

namespace HcclSim {
HcclResult TaskCheckReduceScatterVSemantics(
    std::map<DeviceId, RankMemorySemantics>& allRankMemSemantics, HcclReduceOp reduceType, VDataDesTagInner& vDataDes,
    const std::vector<DeviceId>& rankToDevice)
{
    u32 rankSize = allRankMemSemantics.size();
    if (allRankMemSemantics.size() != rankSize) {
        HCCL_VM_ERROR(
            "{} ReduceScatterV rank set size mismatch: expected {}, actual {}.",
            MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), rankSize, allRankMemSemantics.size());
        return HcclResult::HCCL_E_PARA;
    }

    u64 outputSize = 0;

    for (auto& [deviceId, mem] : allRankMemSemantics) {
        const RankId deviceRank = FindRankIndexByDeviceId(rankToDevice, deviceId);
        u64 totalSize = 0;
        for (auto& [eleAddr, ele] : mem[BufferType::OUTPUT]) {
            const u64 rangeEnd = ele.startAddr + ele.size;
            if (ele.startAddr != totalSize) {
                HCCL_VM_ERROR(
                    "{} ReduceScatterV output for device {} should continue at "
                    "0x{:x}, "
                    "but the next actual range starts at 0x{:x} (actual range: "
                    "[0x{:x},0x{:x}))."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), deviceId, totalSize, ele.startAddr,
                    ele.startAddr, rangeEnd, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.size() > 1 && ele.reduceType != reduceType) {
                HCCL_VM_ERROR(
                    "{} ReduceScatterV output range [0x{:x},0x{:x}) for device "
                    "{} uses reduce "
                    "mode {}, but the operator is set to reduce mode {}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_REDUCE_ERROR), ele.startAddr, rangeEnd, deviceId,
                    DumpReduceOpToString(ele.reduceType), DumpReduceOpToString(reduceType), ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.size() != rankSize) {
                HCCL_VM_ERROR(
                    "{} ReduceScatterV output range [0x{:x},0x{:x}) for device "
                    "{} expected "
                    "{} source ranks but got {}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_REDUCE_ERROR), ele.startAddr, rangeEnd, deviceId,
                    rankSize, ele.srcBufs.size(), ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }
            if (!CoversAllRankDevices(ele.srcBufs, rankToDevice)) {
                HCCL_VM_ERROR(
                    "{} ReduceScatterV output range [0x{:x},0x{:x}) for device "
                    "{} expected one "
                    "source from each device {} in rank order, but got sources "
                    "from devices {}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, deviceId,
                    DeviceIdListToString(rankToDevice), SrcBufDeviceListToString(ele.srcBufs), ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            for (auto& srcBuf : ele.srcBufs) {
                if (srcBuf.bufType != BufferType::INPUT) {
                    HCCL_VM_ERROR(
                        "{} ReduceScatterV output range [0x{:x},0x{:x}) for "
                        "device {} comes "
                        "from device {} with buffer type {}, but it should "
                        "come from INPUT."
                        "\nCurrent result range detail:\n{}",
                        MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, deviceId,
                        srcBuf.deviceId, BufferTypeToString(srcBuf.bufType), ele.Describe());
                    return HcclResult::HCCL_E_PARA;
                }

                const u64 outputOffset = vDataDes.displs[deviceRank] * CHECK_SIZE_TABLE[vDataDes.dataType];
                if (srcBuf.srcAddr != outputOffset + totalSize) {
                    HCCL_VM_ERROR(
                        "{} ReduceScatterV output range [0x{:x},0x{:x}) for "
                        "device {} should "
                        "come from device {}.INPUT at 0x{:x}, but it actually "
                        "comes from device {}.INPUT "
                        "at 0x{:x}.\nCurrent result range detail:\n{}",
                        MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, deviceId,
                        srcBuf.deviceId, outputOffset + totalSize, srcBuf.deviceId, srcBuf.srcAddr, ele.Describe());
                    return HcclResult::HCCL_E_PARA;
                }
            }
            totalSize += ele.size;
        }
        outputSize = vDataDes.counts[deviceRank] * CHECK_SIZE_TABLE[vDataDes.dataType];

        if (totalSize != outputSize) {
            HCCL_VM_ERROR(
                "{} ReduceScatterV output for device {} ends too "
                "early: the checker "
                "validated 0x{:x} bytes in total, but the expected "
                "result size is 0x{:x}.",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), deviceId, totalSize, outputSize);
            return HcclResult::HCCL_E_PARA;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}
} // namespace HcclSim
