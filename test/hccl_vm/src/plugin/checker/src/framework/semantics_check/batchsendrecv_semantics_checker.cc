/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "batchsendrecv_semantics_checker.h"

#include <map>

#include "base.h"
#include "check_utils.h"
#include "sim_log.h"
#include "utils/dump/dump_json_utils.h"
#include "utils/error_codes.h"

namespace HcclSim {
HcclResult TaskCheckBatchSendRecvSemantics(
    std::map<DeviceId, RankMemorySemantics>& allRankMemSemantics, u32 expectedRankSize, u64 dataSize,
    const std::vector<DeviceId>& rankToDevice)
{
    if (expectedRankSize == 0 || allRankMemSemantics.size() != expectedRankSize) {
        HCCL_VM_ERROR(
            "{} BatchSendRecv rank set size mismatch: expected {}, actual {}.",
            MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), expectedRankSize, allRankMemSemantics.size());
        return HcclResult::HCCL_E_PARA;
    }

    for (auto& [deviceId, mem] : allRankMemSemantics) {
        const RankId deviceRank = FindRankIndexByDeviceId(rankToDevice, deviceId);
        u64 totalSize = 0;
        uint32_t curRankIdx = 0;
        u64 curDataSize = 0;
        for (auto& [eleAddr, ele] : mem[BufferType::OUTPUT]) {
            const u64 rangeEnd = ele.startAddr + ele.size;
            if (ele.startAddr != totalSize) {
                HCCL_VM_ERROR(
                    "{} BatchSendRecv output for device {} should continue at "
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
                    "{} BatchSendRecv output range [0x{:x},0x{:x}) for device "
                    "{} should "
                    "come from exactly one source, but it actually comes from "
                    "{} sources."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_REDUCE_ERROR), ele.startAddr, rangeEnd, deviceId,
                    ele.srcBufs.size(), ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            const auto& srcBuf = *ele.srcBufs.begin();
            if (srcBuf.deviceId != rankToDevice[curRankIdx]) {
                HCCL_VM_ERROR(
                    "{} BatchSendRecv output range [0x{:x},0x{:x}) for device "
                    "{} should come "
                    "from source index {}, but it actually comes from device "
                    "{}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, deviceId,
                    curRankIdx, srcBuf.deviceId, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (srcBuf.bufType != BufferType::INPUT) {
                HCCL_VM_ERROR(
                    "{} BatchSendRecv output range [0x{:x},0x{:x}) for device "
                    "{} should come "
                    "from source{}.INPUT, but it actually comes from "
                    "device{}.{}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, deviceId,
                    curRankIdx, srcBuf.deviceId, BufferTypeToString(srcBuf.bufType), ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (srcBuf.srcAddr != dataSize * deviceRank + curDataSize) {
                HCCL_VM_ERROR(
                    "{} BatchSendRecv output range [0x{:x},0x{:x}) for device "
                    "{} should take "
                    "data from source index {} at input address 0x{:x}, but it "
                    "actually takes data "
                    "from device {} at input address 0x{:x}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, deviceId,
                    curRankIdx, dataSize * deviceRank + curDataSize, srcBuf.deviceId, srcBuf.srcAddr, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }
            curDataSize += ele.size;
            if (curDataSize == dataSize) {
                curDataSize = 0;
                curRankIdx++;
            } else if (curDataSize > dataSize) {
                HCCL_VM_ERROR(
                    "{} BatchSendRecv data collected from source index {} for "
                    "device {} becomes "
                    "larger than expected after outputRange [0x{:x},0x{:x}). "
                    "The accumulated size is "
                    "0x{:x}, but the expected size from this source is 0x{:x}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SIZE_ERROR), curRankIdx, deviceId, ele.startAddr,
                    rangeEnd, curDataSize, dataSize, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }
            totalSize += ele.size;
        }
        // 如果curRankIdx等于expectedRankSize，表示已经接受到其他所有rank的数据
        if (curRankIdx != expectedRankSize) {
            HCCL_VM_ERROR(
                "{} BatchSendRecv output for device {} ends too "
                "early. The checker has "
                "validated 0x{:x} bytes in total, but the expected "
                "total size is 0x{:x}.",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), deviceId, totalSize, dataSize * expectedRankSize);
            return HcclResult::HCCL_E_PARA;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskCheckBatchSendRecvRingSemantics(
    std::map<DeviceId, RankMemorySemantics>& allRankMemSemantics, u32 expectedRankSize, u64 dataSize,
    const std::vector<DeviceId>& rankToDevice)
{
    if (expectedRankSize < 2 || allRankMemSemantics.size() != expectedRankSize) {
        HCCL_VM_ERROR(
            "{} BatchSendRecv ring rank set size mismatch: expected "
            "{}, actual {}.",
            MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), expectedRankSize, allRankMemSemantics.size());
        return HcclResult::HCCL_E_PARA;
    }

    for (auto& [deviceId, mem] : allRankMemSemantics) {
        const RankId deviceRank = FindRankIndexByDeviceId(rankToDevice, deviceId);
        const uint32_t expectedSrcRank = (deviceRank + expectedRankSize - 1U) % expectedRankSize;
        const auto outputIt = mem.find(BufferType::OUTPUT);
        if (dataSize == 0) {
            if (outputIt != mem.end()) {
                for (const auto& output : outputIt->second) {
                    if (output.second.size != 0) {
                        HCCL_VM_ERROR(
                            "{} BatchSendRecv ring device {} should "
                            "have an empty output, but range "
                            "[0x{:x},0x{:x}) is present.",
                            MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SIZE_ERROR), deviceId, output.second.startAddr,
                            output.second.startAddr + output.second.size);
                        return HcclResult::HCCL_E_PARA;
                    }
                }
            }
            continue;
        }
        if (outputIt == mem.end() || outputIt->second.empty()) {
            HCCL_VM_ERROR(
                "{} BatchSendRecv ring output is missing for device "
                "{}, expected size 0x{:x}.",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), deviceId, dataSize);
            return HcclResult::HCCL_E_PARA;
        }

        u64 totalSize = 0;
        for (const auto& outputEntry : outputIt->second) {
            const auto& output = outputEntry.second;
            const u64 rangeEnd = output.startAddr + output.size;
            if (rangeEnd < output.startAddr || output.startAddr != totalSize || output.size > dataSize - totalSize) {
                HCCL_VM_ERROR(
                    "{} BatchSendRecv ring output for device {} is not a "
                    "contiguous range of size "
                    "0x{:x}; next actual range is [0x{:x},0x{:x}).\nCurrent "
                    "result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SIZE_ERROR), deviceId, dataSize, output.startAddr,
                    rangeEnd, output.Describe());
                return HcclResult::HCCL_E_PARA;
            }
            if (output.srcBufs.size() != 1) {
                HCCL_VM_ERROR(
                    "{} BatchSendRecv ring output range [0x{:x},0x{:x}) for "
                    "device {} should "
                    "come from exactly one source, but it comes from {} "
                    "sources.\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_REDUCE_ERROR), output.startAddr, rangeEnd, deviceId,
                    output.srcBufs.size(), output.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            const auto& srcBuf = *output.srcBufs.begin();
            if (srcBuf.deviceId != rankToDevice[expectedSrcRank] || srcBuf.bufType != BufferType::INPUT
                || srcBuf.srcAddr != output.startAddr) {
                HCCL_VM_ERROR(
                    "{} BatchSendRecv ring output range [0x{:x},0x{:x}) for "
                    "device {} should "
                    "come from source index {}.INPUT at address 0x{:x}, but it "
                    "comes from device{}.{} at address "
                    "0x{:x}.\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), output.startAddr, rangeEnd, deviceId,
                    expectedSrcRank, output.startAddr, srcBuf.deviceId, BufferTypeToString(srcBuf.bufType),
                    srcBuf.srcAddr, output.Describe());
                return HcclResult::HCCL_E_PARA;
            }
            totalSize += output.size;
        }
        if (totalSize != dataSize) {
            HCCL_VM_ERROR(
                "{} BatchSendRecv ring output for device {} has size "
                "0x{:x}, expected 0x{:x}.",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), deviceId, totalSize, dataSize);
            return HcclResult::HCCL_E_PARA;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}
} // namespace HcclSim
