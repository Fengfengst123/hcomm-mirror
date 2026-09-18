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
 * for the full text of the License. Description: All2All语义校验函数实现
 * Author: yinding
 * Create: 2024-09-30
 */

#include "all2all_semantics_checker.h"

#include <map>
#include <vector>

#include "checker_def.h"
#include "sim_log.h"
#include "utils/dump/dump_json_utils.h"
#include "utils/error_codes.h"

namespace HcclSim {
void GenSendOffsetForOneRank(
    All2AllDataDesTagInner& all2AllDataDes, u32 rankSize, u32 targetRankIdx, std::vector<u64>& sendOffsets)
{
    for (u32 srcRankIdx = 0; srcRankIdx < rankSize; srcRankIdx++) {
        u64 offset = 0;
        for (u32 dstRankIdx = 0; dstRankIdx < targetRankIdx; dstRankIdx++) {
            u64 count = all2AllDataDes.sendCountMatrix[srcRankIdx * rankSize + dstRankIdx];
            u64 size = count * CHECK_SIZE_TABLE[all2AllDataDes.recvType];
            offset += size;
        }
        sendOffsets.push_back(offset);
    }
    return;
}

HcclResult TaskCheckAll2AllSemantics(
    std::map<DeviceId, RankMemorySemantics>& allRankMemSemantics, All2AllDataDesTagInner& all2AllDataDes,
    const std::vector<DeviceId>& rankToDevice)
{
    u32 rankSize = allRankMemSemantics.size();
    if (allRankMemSemantics.size() != rankSize) {
        HCCL_VM_ERROR(
            "{} AllToAll rank set size mismatch: expected {}, actual {}.",
            MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), rankSize, allRankMemSemantics.size());
        return HcclResult::HCCL_E_PARA;
    }

    for (auto& [deviceId, mem] : allRankMemSemantics) {
        const RankId deviceRank = FindRankIndexByDeviceId(rankToDevice, deviceId);
        std::vector<u64> sendOffsets;
        GenSendOffsetForOneRank(all2AllDataDes, rankSize, deviceRank, sendOffsets);

        u64 totalSize = 0;
        uint32_t curRankIdx = 0;
        u64 curDataSize = 0;

        // curRankIdx向当前device发送的数据量是0 直接跳过
        while (curRankIdx < rankSize && all2AllDataDes.sendCountMatrix[curRankIdx * rankSize + deviceRank] == 0) {
            curRankIdx++;
        }

        for (auto& [eleAddr, ele] : mem[BufferType::OUTPUT]) {
            const u64 rangeEnd = ele.startAddr + ele.size;
            if (ele.startAddr != totalSize) {
                HCCL_VM_ERROR(
                    "{} AllToAll output for device {} should continue at "
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
                    "{} AllToAll output range [0x{:x},0x{:x}) for device {} "
                    "should "
                    "come from exactly one source, but it actually comes from "
                    "{} sources."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_REDUCE_ERROR), ele.startAddr, rangeEnd, deviceId,
                    ele.srcBufs.size(), ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (curRankIdx >= rankSize) {
                HCCL_VM_ERROR(
                    "{} AllToAll expected output for device {} has already "
                    "ended, "
                    "but outputRange [0x{:x},0x{:x}) is still "
                    "present.\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::CHECKER_RUNTIME_ERROR), deviceId, ele.startAddr, rangeEnd,
                    ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            const auto& srcBuf = *ele.srcBufs.begin();
            const u64 expectedSrcAddr = sendOffsets[curRankIdx] + curDataSize;
            const u64 expectedSrcEnd = expectedSrcAddr + ele.size;
            const u64 actualSrcAddr = srcBuf.srcAddr;
            const u64 actualSrcEnd = actualSrcAddr + ele.size;

            if (srcBuf.deviceId != rankToDevice[curRankIdx]) {
                HCCL_VM_ERROR(
                    "{} AllToAll output range [0x{:x},0x{:x}) should come from "
                    "rank{}.INPUT[0x{:x},0x{:x}), but it actually comes from "
                    "device{}.{}[0x{:x},0x{:x})."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, curRankIdx,
                    expectedSrcAddr, expectedSrcEnd, srcBuf.deviceId, BufferTypeToString(srcBuf.bufType), actualSrcAddr,
                    actualSrcEnd, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (srcBuf.bufType != BufferType::INPUT) {
                HCCL_VM_ERROR(
                    "{} AllToAll output range [0x{:x},0x{:x}) should come from "
                    "rank{}.INPUT[0x{:x},0x{:x}), but it actually comes from "
                    "device{}.{}[0x{:x},0x{:x})."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, curRankIdx,
                    expectedSrcAddr, expectedSrcEnd, srcBuf.deviceId, BufferTypeToString(srcBuf.bufType), actualSrcAddr,
                    actualSrcEnd, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (srcBuf.srcAddr != expectedSrcAddr) {
                HCCL_VM_ERROR(
                    "{} AllToAll output range [0x{:x},0x{:x}) should come from "
                    "rank{}.INPUT[0x{:x},0x{:x}), but it actually comes from "
                    "device{}.INPUT[0x{:x},0x{:x})."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, curRankIdx,
                    expectedSrcAddr, expectedSrcEnd, srcBuf.deviceId, actualSrcAddr, actualSrcEnd, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            curDataSize += ele.size;
            u64 recvCountFromCurRank = all2AllDataDes.sendCountMatrix[curRankIdx * rankSize + deviceRank];
            u64 recvDataSizeFromCurRank = recvCountFromCurRank * CHECK_SIZE_TABLE[all2AllDataDes.recvType];
            if (curDataSize == recvDataSizeFromCurRank) {
                curDataSize = 0;
                curRankIdx++;
                while (curRankIdx < rankSize
                       && all2AllDataDes.sendCountMatrix[curRankIdx * rankSize + deviceRank] == 0) {
                    // 跳过后续所有发送量为0的rank
                    curRankIdx++;
                }
            } else if (curDataSize > recvDataSizeFromCurRank) {
                HCCL_VM_ERROR(
                    "{} AllToAll data collected from rank{}.INPUT for device "
                    "{} becomes larger "
                    "than expected after outputRange [0x{:x},0x{:x}). The "
                    "accumulated size is 0x{:x}, "
                    "but the expected size from this source rank is "
                    "0x{:x}.\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SIZE_ERROR), curRankIdx, deviceId, ele.startAddr,
                    rangeEnd, curDataSize, recvDataSizeFromCurRank, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }
            totalSize += ele.size;
        }
        // 如果curRankIdx等于rankSize，表示已经接受到其他所有rank的数据
        if (curRankIdx != rankSize) {
            const u64 expectedDataSizeFromCurRank = all2AllDataDes.sendCountMatrix[curRankIdx * rankSize + deviceRank]
                                                    * CHECK_SIZE_TABLE[all2AllDataDes.recvType];
            const u64 missingSizeFromCurRank = expectedDataSizeFromCurRank - curDataSize;
            HCCL_VM_ERROR(
                "{} AllToAll output for device {} ends too early. The checker "
                "has "
                "validated 0x{:x} bytes in total, but data from rank{} is "
                "still incomplete: "
                "0x{:x} bytes are missing (received 0x{:x}, expected 0x{:x}).",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), deviceId, totalSize, curRankIdx,
                missingSizeFromCurRank, curDataSize, expectedDataSizeFromCurRank);
            return HcclResult::HCCL_E_PARA;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}
} // namespace HcclSim
