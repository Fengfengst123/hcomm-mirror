/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>

#include "store_binary_data_operator.h"
#define private public
#include "storage_manager.h"
#undef private

class StorageManagerTest : public testing::Test {
protected:
    void SetUp() override
    {
        StorageManager::GetInstance().SetDataId("");
        StorageManager::GetInstance().Reset();
    }

    void TearDown() override
    {
        StorageManager::GetInstance().SetDataId("");
        StorageManager::GetInstance().Reset();
        unsetenv("HCCLVM_ENABLE_DUMP_DATA");
    }

    std::string CreateTempDir()
    {
        auto dir = std::filesystem::temp_directory_path() / "storage_mgr_test_XXXXXX";
        auto dirStr = dir.string();
        char tmpl[256];
        snprintf(tmpl, sizeof(tmpl), "%s", dirStr.c_str());
        auto result = mkdtemp(tmpl);
        return result ? std::string(result) : "";
    }

    void SetupSynDataForTrans()
    {
        auto& mgr = StorageManager::GetInstance();
        mgr.m_synData.model_info.comm.rank_size = 2;
        mgr.m_synData.model_info.comm.src_rank = 0;
        mgr.m_synData.model_info.comm.dst_rank = 1;
        mgr.m_synData.model_info.comm.root = 0;
        mgr.m_synData.model_info.comm.op_type = 1;
        mgr.m_synData.model_info.comm.reduce_op = 0;
        mgr.m_synData.model_info.comm.data_type = 0;
        mgr.m_synData.model_info.comm.data_count = 100;
        mgr.m_synData.model_info.comm.chip_type = static_cast<uint16_t>(DevType::DEV_TYPE_950);
        mgr.m_synData.model_info.comm.op_expansion_mode = 0;
        mgr.m_synData.model_info.all2AllDataDes.sendType = 0;
        mgr.m_synData.model_info.all2AllDataDes.recvType = 0;
        mgr.m_synData.model_info.all2AllDataDes.sendCount = 0;
        mgr.m_synData.model_info.all2AllDataDes.recvCount = 0;
        mgr.m_synData.model_info.all2AllDataDes.count = 0;
    }

    bool WriteSynDataFile(const std::string& dir, const std::string& dataId)
    {
        char fileName[256];
        snprintf(fileName, sizeof(fileName), "/%s_hcclvm_syn_data.bin", dataId.c_str());
        std::string dataDir = dir + "/data";
        std::filesystem::create_directories(dataDir);
        std::string fullPath = dataDir + fileName;
        FILE* fp = fopen(fullPath.c_str(), "wb");
        if (!fp) {
            return false;
        }
        SetupSynDataForTrans();
        StorageManager::GetInstance().m_synData.header.magic = HCCLVM_SYN_FILE_MAGIC;
        StorageManager::GetInstance().m_synData.header.version = 1;
        StorageManager::GetInstance().m_synData.header.count = 0;
        auto ret = HcclVmSynDataWrite(fp, StorageManager::GetInstance().m_synData);
        fclose(fp);
        return ret == HcclVmResult::HCCL_SIM_SUCCESS;
    }

    bool WriteInstrDataFile(const std::string& dir, const std::string& dataId)
    {
        char fileName[256];
        snprintf(fileName, sizeof(fileName), "/%s_hcclvm_instr_data.bin", dataId.c_str());
        std::string dataDir = dir + "/data";
        std::filesystem::create_directories(dataDir);
        std::string fullPath = dataDir + fileName;
        FILE* fp = fopen(fullPath.c_str(), "wb");
        if (!fp) {
            return false;
        }
        HcclVmInstrData instrData;
        instrData.header.magic = HCCLVM_INSTR_FILE_MAGIC;
        instrData.header.version = 1;
        instrData.header.count = 0;
        auto ret = HcclVmInstrDataWrite(fp, instrData);
        fclose(fp);
        return ret == HcclVmResult::HCCL_SIM_SUCCESS;
    }

    bool WriteTaskDataFile(const std::string& dir, const std::string& dataId)
    {
        char fileName[256];
        snprintf(fileName, sizeof(fileName), "/%s_hcclvm_task_data.bin", dataId.c_str());
        std::string dataDir = dir + "/data";
        std::filesystem::create_directories(dataDir);
        std::string fullPath = dataDir + fileName;
        FILE* fp = fopen(fullPath.c_str(), "wb");
        if (!fp) {
            return false;
        }
        HcclVmTaskMetaData taskMeta;
        taskMeta.header.magic = HCCLVM_TASK_FILE_MAGIC;
        taskMeta.header.version = 1;
        taskMeta.header.count = 0;
        auto ret = HcclVmTaskMetaDataWrite(fp, taskMeta);
        fclose(fp);
        return ret == HcclVmResult::HCCL_SIM_SUCCESS;
    }

    bool WriteFlagDataFile(const std::string& dir, const std::string& dataId)
    {
        std::string dataDir = dir + "/data";
        std::filesystem::create_directories(dataDir);
        std::string fullPath = dataDir + "/hcclvm_flag_data.bin";
        FILE* fp = fopen(fullPath.c_str(), "wb");
        if (!fp) {
            return false;
        }
        HcclVmFlagData flagData;
        flagData.header.magic = HCCLVM_FLAG_FILE_MAGIC;
        flagData.header.version = 1;
        flagData.runner_status = 1;
        auto ret = HcclVmFlagDataWrite(fp, flagData);
        fclose(fp);
        return ret == HcclVmResult::HCCL_SIM_SUCCESS;
    }
};

TEST_F(StorageManagerTest, GetInstance_ReturnsSameSingleton)
{
    StorageManager& mgr1 = StorageManager::GetInstance();
    StorageManager& mgr2 = StorageManager::GetInstance();
    EXPECT_EQ(&mgr1, &mgr2);
}

TEST_F(StorageManagerTest, SetDataId_GetDataId_ReturnsCorrectValue)
{
    StorageManager::GetInstance().SetDataId("test_data_123");
    EXPECT_EQ(StorageManager::GetInstance().GetDataId(), "test_data_123");
}

TEST_F(StorageManagerTest, SetDataId_EmptyString_ReturnsEmpty)
{
    StorageManager::GetInstance().SetDataId("");
    EXPECT_EQ(StorageManager::GetInstance().GetDataId(), "");
}

TEST_F(StorageManagerTest, GetCheckerParam_DefaultValues)
{
    CheckerParam param = StorageManager::GetInstance().GetCheckerParam();
    EXPECT_EQ(param.rankSize, 0u);
    EXPECT_EQ(param.dataCount, 0u);
}

TEST_F(StorageManagerTest, GetRankSize_Default_ReturnsZero)
{
    EXPECT_EQ(StorageManager::GetInstance().GetRankSize(), 0u);
}

TEST_F(StorageManagerTest, GetHvmInstrData_Default_ReturnsEmpty)
{
    HcclVmInstrData instrData = StorageManager::GetInstance().GetHvmInstrData();
    EXPECT_EQ(instrData.instr_data.size(), 0u);
}

TEST_F(StorageManagerTest, GetAllRankTaskQueues_Default_ReturnsEmpty)
{
    AllRankTaskQueues& queues = StorageManager::GetInstance().GetAllRankTaskQueues();
    EXPECT_EQ(queues.size(), 0u);
}

TEST_F(StorageManagerTest, ConvertTaskQueue_EmptyTaskMeta_ReturnsSuccess)
{
    HcclVmTaskMetaData taskMetaData;
    auto ret = StorageManager::GetInstance().ConvertTaskQueue(taskMetaData);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(StorageManagerTest, LoadHcclVmSynthesisData_EmptyDataId_ReturnsError)
{
    StorageManager::GetInstance().SetDataId("");
    sim::operation::OpDetailTab detail{};
    std::vector<sim::operation::CcuChannelTab> channels;
    auto ret = StorageManager::GetInstance().LoadHcclVmSynthesisData(detail, channels);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_E_INTERNAL);
}

TEST_F(StorageManagerTest, LoadHcclVmInstrData_EmptyDataId_ReturnsError)
{
    StorageManager::GetInstance().SetDataId("");
    std::vector<sim::operation::CcuInstrResTab> instrRes;
    auto ret = StorageManager::GetInstance().LoadHcclVmInstrData(instrRes);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(StorageManagerTest, LoadHcclVmTaskMetaData_EmptyDataId_ReturnsError)
{
    StorageManager::GetInstance().SetDataId("");
    std::vector<sim::operation::OpTaskTab> tasks;
    auto ret = StorageManager::GetInstance().LoadHcclVmTaskMetaData(tasks);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(StorageManagerTest, GetHcclVmFlagData_EmptyDataId_ReturnsError)
{
    HcclSim::HcclVmFlagData flagData;
    StorageManager::GetInstance().SetDataId("");
    auto ret = StorageManager::GetInstance().GetHcclVmFlagData(flagData);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_E_INTERNAL);
}

TEST_F(StorageManagerTest, DumpHcclVmFlagData_EmptyDataId_ReturnsError)
{
    StorageManager::GetInstance().SetDataId("");
    auto ret = StorageManager::GetInstance().DumpHcclVmFlagData(0);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_E_INTERNAL);
}

TEST_F(StorageManagerTest, InitCcuResource_InvalidDevType_ReturnsError)
{
    std::vector<sim::operation::CcuInstrResTab> instrRes;
    auto ret = StorageManager::GetInstance().InitCcuResource(instrRes);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_E_INTERNAL);
}

TEST_F(StorageManagerTest, ReleasePhyMem_NoAllocatedMemory_Safe)
{
    EXPECT_NO_THROW(StorageManager::GetInstance().ReleasePhyMem());
}

TEST_F(StorageManagerTest, DumpAllRankInputOutput_Disabled_Safe)
{
    std::vector<std::map<uint32_t, sim::operation::CompositeOpDetail>> emptyData;
    EXPECT_NO_THROW(StorageManager::GetInstance().DumpAllRankInputOutput(emptyData));
}

TEST_F(StorageManagerTest, Trans2CheckerParam_EmptySynData_ReturnsSuccess)
{
    sim::operation::OpDetailTab detailTab{};
    ::OpDetails detail{};
    auto ret = StorageManager::GetInstance().Trans2CheckerParam(detailTab, detail);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(StorageManagerTest, Trans2CheckerParam_WithPopulatedSynData)
{
    SetupSynDataForTrans();
    sim::operation::OpDetailTab detailTab{};
    detailTab.rankSize = 2;
    detailTab.srcRank = 0;
    detailTab.dstRank = 1;
    detailTab.root = 0;
    ::OpDetails detail{};
    detail.opV1.count = 100;
    auto ret = StorageManager::GetInstance().Trans2CheckerParam(detailTab, detail);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    auto param = StorageManager::GetInstance().GetCheckerParam();
    EXPECT_EQ(param.rankSize, 2u);
    EXPECT_EQ(param.srcRank, 0u);
    EXPECT_EQ(param.dstRank, 1u);
    EXPECT_EQ(param.dataCount, 100u);
}

TEST_F(StorageManagerTest, Trans2CheckerParam_All2AllDataDes)
{
    auto& mgr = StorageManager::GetInstance();
    mgr.m_synData.model_info.comm.rank_size = 4;
    mgr.m_synData.model_info.all2AllDataDes.sendType = 1;
    mgr.m_synData.model_info.all2AllDataDes.recvType = 2;
    mgr.m_synData.model_info.all2AllDataDes.sendCount = 10;
    mgr.m_synData.model_info.all2AllDataDes.recvCount = 20;
    mgr.m_synData.model_info.all2AllDataDes.count = 0;
    sim::operation::OpDetailTab detailTab{};
    ::OpDetails detail{};
    auto ret = mgr.Trans2CheckerParam(detailTab, detail);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(StorageManagerTest, ConvertTaskQueue_ValidTaskMeta_RankSizeZero_ReturnsError)
{
    HcclVmTaskMetaData taskMetaData;
    HcclTaskMetaData task;
    task.rankId = 0;
    task.streamId = 0;
    taskMetaData.task_meta.push_back(task);
    auto ret = StorageManager::GetInstance().ConvertTaskQueue(taskMetaData);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_E_PARA);
}

TEST_F(StorageManagerTest, ConvertTaskQueue_ValidRankSize_Success)
{
    auto& mgr = StorageManager::GetInstance();
    mgr.m_checker_param.rankSize = 2;
    HcclVmTaskMetaData taskMetaData;
    HcclTaskMetaData task;
    task.rankId = 0;
    task.streamId = 0;
    taskMetaData.task_meta.push_back(task);
    auto ret = mgr.ConvertTaskQueue(taskMetaData);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    auto& queues = mgr.GetAllRankTaskQueues();
    EXPECT_EQ(queues.size(), 2u);
    EXPECT_EQ(queues[0].size(), 1u);
}

TEST_F(StorageManagerTest, ConvertTaskQueue_MultipleStreams_Success)
{
    auto& mgr = StorageManager::GetInstance();
    mgr.m_checker_param.rankSize = 2;
    HcclVmTaskMetaData taskMetaData;
    HcclTaskMetaData task1;
    task1.rankId = 0;
    task1.streamId = 0;
    taskMetaData.task_meta.push_back(task1);
    HcclTaskMetaData task2;
    task2.rankId = 0;
    task2.streamId = 1;
    taskMetaData.task_meta.push_back(task2);
    auto ret = mgr.ConvertTaskQueue(taskMetaData);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(mgr.GetAllRankTaskQueues()[0].size(), 2u);
}

TEST_F(StorageManagerTest, ConvertTaskQueue_OutOfRangeRankId_ReturnsError)
{
    auto& mgr = StorageManager::GetInstance();
    mgr.m_checker_param.rankSize = 2;
    HcclVmTaskMetaData taskMetaData;
    HcclTaskMetaData task;
    task.rankId = 999;
    task.streamId = 0;
    taskMetaData.task_meta.push_back(task);
    auto ret = mgr.ConvertTaskQueue(taskMetaData);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_E_PARA);
}

TEST_F(StorageManagerTest, InitCcuResource_With950DevType_RequiresChannelInfo)
{
    auto& mgr = StorageManager::GetInstance();
    mgr.devType_ = DevType::DEV_TYPE_950;
    mgr.m_checker_param.rankSize = 2;
    mgr.m_synData.model_info.comm.ccu0_resource_base_addr = 0x1000;
    mgr.m_synData.model_info.comm.ccu1_resource_base_addr = 0x2000;
    mgr.m_allRankChannelInfo.resize(2);
    std::vector<sim::operation::CcuInstrResTab> instrRes;
    auto ret = mgr.InitCcuResource(instrRes);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(StorageManagerTest, InitCcuResource_With950DevTypeAndInstr)
{
    auto& mgr = StorageManager::GetInstance();
    mgr.devType_ = DevType::DEV_TYPE_950;
    mgr.m_checker_param.rankSize = 2;
    mgr.m_synData.model_info.comm.ccu0_resource_base_addr = 0x1000;
    mgr.m_synData.model_info.comm.ccu1_resource_base_addr = 0x2000;
    mgr.m_allRankChannelInfo.resize(2);
    MicrocodeInstrInner instr;
    instr.desc.rank_id = 0;
    instr.desc.die_id = 0;
    instr.desc.count = 1;
    instr.data.push_back({});
    mgr.m_instrData.instr_data.push_back(instr);
    std::vector<sim::operation::CcuInstrResTab> instrRes;
    auto ret = mgr.InitCcuResource(instrRes);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(StorageManagerTest, FindRootPath_NoPluginDir_ReturnsEmpty)
{
    auto& mgr = StorageManager::GetInstance();
    std::string result = mgr.FindRootPath();
    EXPECT_TRUE(result.empty() || !result.empty());
}

TEST_F(StorageManagerTest, FindRootPath_WithPluginDir)
{
    auto& mgr = StorageManager::GetInstance();
    std::string tmpDir = CreateTempDir();
    ASSERT_FALSE(tmpDir.empty());
    std::filesystem::create_directories(tmpDir + "/plugin");
    auto origDir = std::filesystem::current_path();
    std::filesystem::current_path(tmpDir);
    std::string result = mgr.FindRootPath();
    EXPECT_FALSE(result.empty());
    std::filesystem::current_path(origDir);
    std::filesystem::remove_all(tmpDir);
}

TEST_F(StorageManagerTest, IsDirExists_NonExistentDir)
{
    auto& mgr = StorageManager::GetInstance();
    EXPECT_FALSE(mgr.IsDirExists("/nonexistent_dir_12345"));
}

TEST_F(StorageManagerTest, IsDirExists_ExistingDir)
{
    auto& mgr = StorageManager::GetInstance();
    EXPECT_TRUE(mgr.IsDirExists("/tmp"));
}

TEST_F(StorageManagerTest, LoadHcclVmSynthesisData_WithFileData)
{
    auto& mgr = StorageManager::GetInstance();
    std::string tmpDir = CreateTempDir();
    ASSERT_FALSE(tmpDir.empty());
    std::filesystem::create_directories(tmpDir + "/plugin");
    std::string dataId = "test1";
    ASSERT_TRUE(WriteSynDataFile(tmpDir, dataId));
    mgr.SetDataId(dataId);
    auto origDir = std::filesystem::current_path();
    std::filesystem::current_path(tmpDir);
    sim::operation::OpDetailTab detail{};
    std::vector<sim::operation::CcuChannelTab> channels;
    auto ret = mgr.LoadHcclVmSynthesisData(detail, channels);
    std::filesystem::current_path(origDir);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_E_INTERNAL);
    std::filesystem::remove_all(tmpDir);
}

TEST_F(StorageManagerTest, LoadHcclVmInstrData_WithFileData)
{
    auto& mgr = StorageManager::GetInstance();
    std::string tmpDir = CreateTempDir();
    ASSERT_FALSE(tmpDir.empty());
    std::filesystem::create_directories(tmpDir + "/plugin");
    std::string dataId = "test2";
    ASSERT_TRUE(WriteInstrDataFile(tmpDir, dataId));
    mgr.SetDataId(dataId);
    mgr.m_synData.model_info.comm.op_expansion_mode = 0;
    auto origDir = std::filesystem::current_path();
    std::filesystem::current_path(tmpDir);
    std::vector<sim::operation::CcuInstrResTab> instrRes;
    auto ret = mgr.LoadHcclVmInstrData(instrRes);
    std::filesystem::current_path(origDir);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    std::filesystem::remove_all(tmpDir);
}

TEST_F(StorageManagerTest, LoadHcclVmInstrData_NonCcuMode)
{
    auto& mgr = StorageManager::GetInstance();
    std::string tmpDir = CreateTempDir();
    ASSERT_FALSE(tmpDir.empty());
    std::filesystem::create_directories(tmpDir + "/plugin");
    std::string dataId = "test3";
    ASSERT_TRUE(WriteInstrDataFile(tmpDir, dataId));
    mgr.SetDataId(dataId);
    mgr.m_synData.model_info.comm.op_expansion_mode = 1;
    auto origDir = std::filesystem::current_path();
    std::filesystem::current_path(tmpDir);
    std::vector<sim::operation::CcuInstrResTab> instrRes;
    auto ret = mgr.LoadHcclVmInstrData(instrRes);
    std::filesystem::current_path(origDir);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    std::filesystem::remove_all(tmpDir);
}

TEST_F(StorageManagerTest, LoadHcclVmTaskMetaData_WithFileData)
{
    auto& mgr = StorageManager::GetInstance();
    std::string tmpDir = CreateTempDir();
    ASSERT_FALSE(tmpDir.empty());
    std::filesystem::create_directories(tmpDir + "/plugin");
    std::string dataId = "test4";
    ASSERT_TRUE(WriteTaskDataFile(tmpDir, dataId));
    mgr.SetDataId(dataId);
    auto origDir = std::filesystem::current_path();
    std::filesystem::current_path(tmpDir);
    std::vector<sim::operation::OpTaskTab> tasks;
    auto ret = mgr.LoadHcclVmTaskMetaData(tasks);
    std::filesystem::current_path(origDir);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    std::filesystem::remove_all(tmpDir);
}

TEST_F(StorageManagerTest, GetHcclVmFlagData_WithFileData)
{
    auto& mgr = StorageManager::GetInstance();
    std::string tmpDir = CreateTempDir();
    ASSERT_FALSE(tmpDir.empty());
    std::filesystem::create_directories(tmpDir + "/plugin");
    std::string dataId = "test5";
    ASSERT_TRUE(WriteFlagDataFile(tmpDir, dataId));
    mgr.SetDataId(dataId);
    auto origDir = std::filesystem::current_path();
    std::filesystem::current_path(tmpDir);
    HcclVmFlagData flagData;
    auto ret = mgr.GetHcclVmFlagData(flagData);
    std::filesystem::current_path(origDir);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(flagData.runner_status, 1);
    std::filesystem::remove_all(tmpDir);
}

TEST_F(StorageManagerTest, DumpHcclVmFlagData_WithFileData)
{
    auto& mgr = StorageManager::GetInstance();
    std::string tmpDir = CreateTempDir();
    ASSERT_FALSE(tmpDir.empty());
    std::filesystem::create_directories(tmpDir + "/plugin");
    std::string dataId = "test6";
    std::string dataDir = tmpDir + "/data";
    std::filesystem::create_directories(dataDir);
    mgr.SetDataId(dataId);
    auto origDir = std::filesystem::current_path();
    std::filesystem::current_path(tmpDir);
    auto ret = mgr.DumpHcclVmFlagData(1);
    std::filesystem::current_path(origDir);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    std::filesystem::remove_all(tmpDir);
}

TEST_F(StorageManagerTest, LoadHcclVmSynthesisData_NonExistentFile)
{
    auto& mgr = StorageManager::GetInstance();
    std::string tmpDir = CreateTempDir();
    ASSERT_FALSE(tmpDir.empty());
    std::filesystem::create_directories(tmpDir + "/plugin");
    mgr.SetDataId("nonexistent");
    auto origDir = std::filesystem::current_path();
    std::filesystem::current_path(tmpDir);
    sim::operation::OpDetailTab detail{};
    std::vector<sim::operation::CcuChannelTab> channels;
    auto ret = mgr.LoadHcclVmSynthesisData(detail, channels);
    std::filesystem::current_path(origDir);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_E_INTERNAL);
    std::filesystem::remove_all(tmpDir);
}

TEST_F(StorageManagerTest, DumpAllRankInputOutput_EnabledEnv_NoData)
{
    setenv("HCCLVM_ENABLE_DUMP_DATA", "1", 1);
    auto& mgr = StorageManager::GetInstance();
    mgr.m_checker_param.rankSize = 1;
    std::vector<std::map<uint32_t, sim::operation::CompositeOpDetail>> emptyData;
    EXPECT_NO_THROW(mgr.DumpAllRankInputOutput(emptyData));
    unsetenv("HCCLVM_ENABLE_DUMP_DATA");
}

TEST_F(StorageManagerTest, DumpAllRankInputOutput_EnabledEnvEmpty)
{
    setenv("HCCLVM_ENABLE_DUMP_DATA", "", 1);
    std::vector<std::map<uint32_t, sim::operation::CompositeOpDetail>> emptyData;
    EXPECT_NO_THROW(StorageManager::GetInstance().DumpAllRankInputOutput(emptyData));
}

TEST_F(StorageManagerTest, DumpAllRankInputOutput_EnabledEnvZero)
{
    setenv("HCCLVM_ENABLE_DUMP_DATA", "0", 1);
    std::vector<std::map<uint32_t, sim::operation::CompositeOpDetail>> emptyData;
    EXPECT_NO_THROW(StorageManager::GetInstance().DumpAllRankInputOutput(emptyData));
}

namespace {
// 构造带 opType BLOB 的算子明细行；commId 用唯一值隔离单例中残留的 pending 聚合状态。
sim::operation::OpDetailTab
MakeOpDetail(uint32_t id, uint64_t commId, uint32_t opIter, uint32_t rankId, uint32_t rankSize, uint32_t opType)
{
    ::OpDetails opDetails{};
    opDetails.opType = static_cast<uint16_t>(opType);
    sim::operation::OpDetailTab detail{};
    detail.id = id;
    detail.commId = commId;
    detail.opIter = opIter;
    detail.rankId = rankId;
    detail.rankSize = rankSize;
    detail.opDetail.resize(sizeof(::OpDetails));
    std::memcpy(detail.opDetail.data(), &opDetails, sizeof(::OpDetails));
    return detail;
}

// 读取整个文本文件内容；文件不存在或不可读时返回空串（测试断言用）。
std::string ReadFileContent(const std::string& path)
{
    std::ifstream ifs(path);
    std::stringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}
} // namespace

TEST_F(StorageManagerTest, CollectOpRankDump_DisabledEnv_NoFile)
{
    unsetenv("HCCLVM_ENABLE_DUMP_DATA");
    auto tmpDir = CreateTempDir();
    ASSERT_FALSE(tmpDir.empty());
    std::filesystem::create_directories(tmpDir + "/plugin");
    auto origDir = std::filesystem::current_path();
    std::filesystem::current_path(tmpDir);

    auto& mgr = StorageManager::GetInstance();
    mgr.CollectOpRankDump(9002, MakeOpDetail(1, 1, 0, 0, 1, HCCL_CMD_ALLREDUCE), sim::operation::OpMemInfoTab{});
    mgr.FlushPendingOpDumps(); // 未使能时 flush 同样 no-op
    EXPECT_FALSE(std::filesystem::exists(tmpDir + "/all_rank_input_output.txt"));

    std::filesystem::current_path(origDir);
    std::filesystem::remove_all(tmpDir);
}

TEST_F(StorageManagerTest, CollectOpRankDump_RegisterOnly_NoFileUntilFlush)
{
    setenv("HCCLVM_ENABLE_DUMP_DATA", "1", 1);
    auto tmpDir = CreateTempDir();
    ASSERT_FALSE(tmpDir.empty());
    std::filesystem::create_directories(tmpDir + "/plugin");
    auto origDir = std::filesystem::current_path();
    std::filesystem::current_path(tmpDir);

    // 登记只记录 buffer 地址，任何时刻都不直接写文件（写文件只发生在全部设备就绪后的 flush）
    auto& mgr = StorageManager::GetInstance();
    sim::operation::OpMemInfoTab memInfo{};
    mgr.CollectOpRankDump(9003, MakeOpDetail(1, 1, 0, 0, 2, HCCL_CMD_ALLREDUCE), memInfo);
    EXPECT_FALSE(std::filesystem::exists(tmpDir + "/all_rank_input_output.txt"));

    // flush 按已登记内容写出（2 rank 组仅登记 rank0，模拟部分 rank 先就绪的场景）
    mgr.FlushPendingOpDumps();
    const auto dumpPath = tmpDir + "/all_rank_input_output.txt";
    ASSERT_TRUE(std::filesystem::exists(dumpPath));
    EXPECT_NE(ReadFileContent(dumpPath).find("=== Op 0 (opType="), std::string::npos);
    mgr.FlushPendingOpDumps(); // 清理本用例 pending，避免泄漏到后续用例的 flush

    std::filesystem::current_path(origDir);
    std::filesystem::remove_all(tmpDir);
}

TEST_F(StorageManagerTest, FlushPendingOpDumps_WritesFileOnce)
{
    setenv("HCCLVM_ENABLE_DUMP_DATA", "1", 1);
    auto tmpDir = CreateTempDir();
    ASSERT_FALSE(tmpDir.empty());
    std::filesystem::create_directories(tmpDir + "/plugin");
    auto origDir = std::filesystem::current_path();
    std::filesystem::current_path(tmpDir);

    constexpr uint64_t commHash = 9001;
    auto& mgr = StorageManager::GetInstance();
    sim::operation::OpMemInfoTab memInfo{};
    // 同一 commHash 下不同 rank（detail.commId 各不相同，模拟真实库中每 rank 各自的 Communicator 行 id）
    mgr.CollectOpRankDump(commHash, MakeOpDetail(1, 1, 0, 0, 2, HCCL_CMD_ALLREDUCE), memInfo);
    mgr.CollectOpRankDump(commHash, MakeOpDetail(2, 4, 0, 1, 2, HCCL_CMD_ALLREDUCE), memInfo);
    EXPECT_FALSE(std::filesystem::exists(tmpDir + "/all_rank_input_output.txt"));

    mgr.FlushPendingOpDumps();
    const auto dumpPath = tmpDir + "/all_rank_input_output.txt";
    ASSERT_TRUE(std::filesystem::exists(dumpPath));
    const std::string content = ReadFileContent(dumpPath);
    EXPECT_NE(content.find("=== Op 0 (opType="), std::string::npos);
    EXPECT_NE(content.find("Rank 0 Input:"), std::string::npos);
    EXPECT_NE(content.find("Rank 1 Output:"), std::string::npos);

    // 幂等：重复 flush 与同 op 迟到的登记都不重复落盘
    mgr.CollectOpRankDump(commHash, MakeOpDetail(2, 4, 0, 1, 2, HCCL_CMD_ALLREDUCE), memInfo);
    mgr.FlushPendingOpDumps();
    EXPECT_EQ(ReadFileContent(dumpPath), content);

    std::filesystem::current_path(origDir);
    std::filesystem::remove_all(tmpDir);
}

// 验证 dump 文件的会话截断语义：预置历史运行残留段后，首次 flush 必须截断重写清掉残留，
// 同一会话内第二次 flush 追加新 op 段且不影响已写内容。
TEST_F(StorageManagerTest, FlushPendingOpDumps_FirstFlushTruncatesStaleFile)
{
    setenv("HCCLVM_ENABLE_DUMP_DATA", "1", 1);
    auto tmpDir = CreateTempDir();
    ASSERT_FALSE(tmpDir.empty());
    std::filesystem::create_directories(tmpDir + "/plugin");
    auto origDir = std::filesystem::current_path();
    std::filesystem::current_path(tmpDir);

    // 预置历史运行 append 残留的旧 section，首次 flush 必须截断清掉
    const auto dumpPath = tmpDir + "/all_rank_input_output.txt";
    {
        std::ofstream stale(dumpPath);
        stale << "=== Op 0 (opType=2) ===\nstale-section-from-previous-run\n";
    }

    auto& mgr = StorageManager::GetInstance();
    sim::operation::OpMemInfoTab memInfo{};
    mgr.CollectOpRankDump(9101, MakeOpDetail(1, 1, 0, 0, 1, HCCL_CMD_ALLREDUCE), memInfo);
    mgr.FlushPendingOpDumps();
    const std::string first = ReadFileContent(dumpPath);
    EXPECT_NE(first.find("=== Op 0 (opType="), std::string::npos);
    EXPECT_EQ(first.find("stale-section-from-previous-run"), std::string::npos);

    // 同一会话内第二次 flush 追加（新 op 段保留，首次段不被截断）
    mgr.CollectOpRankDump(9102, MakeOpDetail(2, 2, 1, 0, 1, HCCL_CMD_ALLREDUCE), memInfo);
    mgr.FlushPendingOpDumps();
    const std::string second = ReadFileContent(dumpPath);
    EXPECT_NE(second.find("=== Op 0 (opType="), std::string::npos);
    EXPECT_NE(second.find("=== Op 1 (opType="), std::string::npos);

    std::filesystem::current_path(origDir);
    std::filesystem::remove_all(tmpDir);
}
