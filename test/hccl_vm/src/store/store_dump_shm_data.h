/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DUMP_FLAG_DATA_H
#define DUMP_FLAG_DATA_H

#include "operation_data/operation_data_types.h"
#include "rt_external_kernel.h"
#include "sim_binary_data_type_pub.h"
#include "sim_common_defs.h"
#include <cstdint>
#include <map>
#include <vector>

namespace HcclSim {
std::string GenDataId();
HcclVmResult DumpDataToFile(const std::string& dataId);

HcclVmResult DumpHcclVmFlagData(HcclSim::HcclVmFlagData& flagData);
HcclVmResult GetHcclVmFlagData(HcclSim::HcclVmFlagData& waitFlag);
HcclVmResult DumpHcclVmSynthesisData(const std::string& dataId, const sim::operation::OpExecutionKey& key);
HcclVmResult DumpHcclVmInstrData(const std::string& dataId);
HcclVmResult DumpHcclVmTask(const std::string& dataId, const sim::operation::OpExecutionKey& key);
HcclVmResult CreateChannelInfo(HcclVmSynData& hvmSynData);
HcclVmResult CreateJettyInfo(HcclVmSynData& hvmSynData);
HcclVmResult CreateMemoryInfo(HcclVmSynData& hvmSynData, const sim::operation::OpMemInfoTab& memInfo, uint32_t rankId);
HcclVmResult CreateSimTaskMetaData(
    HcclVmTaskMetaData& hvmTaskMetaData,
    const std::map<uint32_t, std::vector<sim::operation::CompositeOpDetail>>& compositeDataMap);
HcclVmResult CreateSimSynData(HcclVmSynData& hvmSynData, const sim::operation::OpExecutionKey& key);
HcclVmResult
GenCaModelCcuInstr(hcomm::CcuRep::CcuInstr* instrData, uint32_t instrCnt, uint32_t deviceId, uint32_t dieId);
HcclVmResult GenCaModelCcuToml(rtCcuTaskInfo_t* taskInfo, uint32_t deviceId, uint32_t streamId);
} // namespace HcclSim
#endif
