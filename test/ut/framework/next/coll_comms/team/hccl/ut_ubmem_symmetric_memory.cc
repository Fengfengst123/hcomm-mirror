/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "mockcpp/mockcpp.hpp"
#include "coll_comm.h"

#define private public
#include "ubmem_symmetric_memory.h"
#include "ubmem_symmetric_memory_agent.h"
#undef private
#include "hcomm_team.h"
#include "hcomm_team_mgr.h"

namespace hccl {
namespace {

    constexpr size_t TEST_GRANULARITY = 0x1000U;
    constexpr uint64_t TEST_STRIDE = 0x4000U;
    constexpr uintptr_t TEST_WINDOW_BASE = 0x100000U;
    constexpr uintptr_t TEST_DEVICE_WINDOW = 0x200000U;

    std::vector<std::vector<uint8_t>> g_recvFrames;
    size_t g_recvFrameIndex = 0U;
    uint32_t g_mapMemCallCount = 0U;
    std::vector<void*> g_boundBaseVas;
    std::vector<size_t> g_boundUserSizes;
    std::vector<CommMem> g_boundFirstMemberMems;

    struct SymmetricVaInitMockData {
        size_t freeHbmSize{std::numeric_limits<size_t>::max()};
        size_t totalHbmSize{std::numeric_limits<size_t>::max()};
        int32_t deviceId{0};
        size_t granularity{TEST_GRANULARITY};
        void* heapBase{reinterpret_cast<void*>(TEST_WINDOW_BASE + 2U * TEST_STRIDE)};
    };

    void MockSymmetricVaInit(SymmetricVaInitMockData& data)
    {
        MOCKER_CPP(aclrtGetMemInfo)
            .expects(once())
            .with(
                ACL_HBM_MEM_HUGE, outBoundP(&data.freeHbmSize, sizeof(data.freeHbmSize)),
                outBoundP(&data.totalHbmSize, sizeof(data.totalHbmSize)))
            .will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtGetDevice)
            .expects(once())
            .with(outBoundP(&data.deviceId, sizeof(data.deviceId)))
            .will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtMemGetAllocationGranularity)
            .expects(once())
            .with(
                mockcpp::any(), ACL_RT_MEM_ALLOC_GRANULARITY_RECOMMENDED,
                outBoundP(&data.granularity, sizeof(data.granularity)))
            .will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtReserveMemAddressNoUCMemory)
            .expects(once())
            .with(
                outBoundP(&data.heapBase, sizeof(data.heapBase)), mockcpp::any(), mockcpp::any(), mockcpp::any(),
                mockcpp::any())
            .will(returnValue(ACL_SUCCESS));
    }

    HcommResult StubBindUbSymmetricWindow(
        HcclCommSymWindow handle, HcommTeamHandle lsaTeam, uint32_t netLayer, const CommMem* memberMems,
        uint32_t memberNum, void* baseVa, size_t stride, size_t userSize)
    {
        (void)handle;
        (void)lsaTeam;
        (void)netLayer;
        (void)stride;
        if (memberMems == nullptr || memberNum == 0U) {
            return HCOMM_E_PTR;
        }
        g_boundBaseVas.emplace_back(baseVa);
        g_boundUserSizes.emplace_back(userSize);
        g_boundFirstMemberMems.emplace_back(memberMems[0]);
        return HCOMM_SUCCESS;
    }

    aclError StubMapMemFailOnSecondCall(
        void* virtualAddress, size_t size, size_t offset, aclrtDrvMemHandle handle, uint64_t flags)
    {
        (void)virtualAddress;
        (void)size;
        (void)offset;
        (void)handle;
        (void)flags;
        ++g_mapMemCallCount;
        return g_mapMemCallCount == 1U ? ACL_SUCCESS : ACL_ERROR_RT_PARAM_INVALID;
    }

    HcclResult StubSocketSend(SocketHandler socketHandle, void* sendBuffer, uint64_t sendSize, uint64_t* sentSize)
    {
        (void)socketHandle;
        (void)sendBuffer;
        *sentSize = sendSize;
        return HCCL_SUCCESS;
    }

    HcclResult StubSocketRecv(SocketHandler socketHandle, void* recvBuffer, uint64_t recvSize, uint64_t* receivedSize)
    {
        (void)socketHandle;
        if (g_recvFrameIndex >= g_recvFrames.size() || g_recvFrames[g_recvFrameIndex].size() != recvSize) {
            return HCCL_E_INTERNAL;
        }
        const auto& frame = g_recvFrames[g_recvFrameIndex++];
        if (memcpy_s(recvBuffer, recvSize, frame.data(), frame.size()) != EOK) {
            return HCCL_E_MEMORY;
        }
        *receivedSize = recvSize;
        return HCCL_SUCCESS;
    }

    std::unique_ptr<UbMemSymmetricMemory> MakeSymmetricMemory(const std::vector<uint32_t>& worldRankIds = {0U, 2U})
    {
        return std::make_unique<UbMemSymmetricMemory>(nullptr, nullptr, 0U, worldRankIds);
    }

    class TestUbMemSymmetricMemoryWithMock : public testing::Test {
    protected:
        void TearDown() override
        {
            GlobalMockObject::verify();
            g_recvFrames.clear();
            g_recvFrameIndex = 0U;
            g_mapMemCallCount = 0U;
            g_boundBaseVas.clear();
            g_boundUserSizes.clear();
            g_boundFirstMemberMems.clear();
        }
    };

    TEST(TestUbMemSymmetricMemory, Ut_ValidateRegisterRange_When_RangeVaries_Expect_CorrectResult)
    {
        auto memory = MakeSymmetricMemory();

        EXPECT_EQ(memory->ValidateRegisterRange(nullptr, TEST_GRANULARITY), HCCL_E_PTR);
        EXPECT_EQ(memory->ValidateRegisterRange(reinterpret_cast<void*>(TEST_WINDOW_BASE), 0U), HCCL_E_PARA);
        EXPECT_EQ(
            memory->ValidateRegisterRange(reinterpret_cast<void*>(std::numeric_limits<uintptr_t>::max() - 1U), 4U),
            HCCL_E_PARA);
        EXPECT_EQ(
            memory->ValidateRegisterRange(reinterpret_cast<void*>(TEST_WINDOW_BASE), TEST_GRANULARITY), HCCL_SUCCESS);
    }

    TEST_F(TestUbMemSymmetricMemoryWithMock, Ut_InitSymmetricVa_When_StrideExceedsHbm_Expect_ParaError)
    {
        CollComm collComm(nullptr, 0U, "ut_stride_hbm", ManagerCallbacks{}, CollCommInitMode::simpleMode);
        const std::vector<uint32_t> ranks{0U};
        UbMemSymmetricMemory memory(&collComm, reinterpret_cast<HcommTeamHandle>(TEST_WINDOW_BASE), 0U, ranks);
        size_t freeHbmSize = 0U;
        size_t totalHbmSize = 1U;
        MOCKER_CPP(aclrtGetMemInfo)
            .expects(once())
            .with(
                ACL_HBM_MEM_HUGE, outBoundP(&freeHbmSize, sizeof(freeHbmSize)),
                outBoundP(&totalHbmSize, sizeof(totalHbmSize)))
            .will(returnValue(ACL_SUCCESS));

        EXPECT_EQ(memory.InitSymmetricVa(), HCCL_E_PARA);
        EXPECT_EQ(memory.heapBase_, nullptr);
    }

    TEST(TestUbMemSymmetricMemory, Ut_ValidateMappingRange_When_RangeVaries_Expect_CorrectResult)
    {
        auto memory = MakeSymmetricMemory();
        memory->heapBase_ = reinterpret_cast<void*>(TEST_WINDOW_BASE);
        memory->stride_ = TEST_STRIDE;
        memory->granularity_ = TEST_GRANULARITY;

        EXPECT_EQ(memory->ValidateMappingRange(TEST_GRANULARITY, 2U * TEST_GRANULARITY), HCCL_SUCCESS);
        EXPECT_EQ(memory->ValidateMappingRange(1U, TEST_GRANULARITY), HCCL_E_PARA);
        EXPECT_EQ(memory->ValidateMappingRange(TEST_GRANULARITY, TEST_GRANULARITY - 1U), HCCL_E_PARA);
        EXPECT_EQ(memory->ValidateMappingRange(TEST_STRIDE, TEST_GRANULARITY), HCCL_E_PARA);

        memory->heapBase_ = nullptr;
    }

    TEST(TestUbMemSymmetricMemory, Ut_ImportLocalHandle_When_SelfMember_Expect_BorrowedHandle)
    {
        auto memory = MakeSymmetricMemory();
        memory->selfMember_ = 1U;
        UbMemSymmetricMemory::PaMappingInfo mapping;
        mapping.paHandle = reinterpret_cast<aclrtDrvMemHandle>(TEST_WINDOW_BASE);
        aclrtDrvMemHandle importedHandle = nullptr;
        bool ownsHandle = true;

        EXPECT_EQ(memory->ImportMemberHandle(1U, {}, mapping, importedHandle, ownsHandle), HCCL_SUCCESS);
        EXPECT_EQ(importedHandle, mapping.paHandle);
        EXPECT_FALSE(ownsHandle);
    }

    TEST_F(TestUbMemSymmetricMemoryWithMock, Ut_ImportMemberHandle_When_RemoteMember_Expect_OwnedImportedHandle)
    {
        auto memory = MakeSymmetricMemory();
        memory->selfMember_ = 0U;
        UbMemSymmetricMemory::PaMappingInfo mapping;
        std::vector<uint8_t> shareableDesc(sizeof(aclrtMemFabricHandle), 1U);
        aclrtDrvMemHandle expectedHandle = reinterpret_cast<aclrtDrvMemHandle>(TEST_WINDOW_BASE);
        aclrtDrvMemHandle importedHandle = nullptr;
        bool ownsHandle = false;

        MOCKER_CPP(aclrtMemImportFromShareableHandleV2)
            .expects(once())
            .with(
                mockcpp::any(), ACL_MEM_SHARE_HANDLE_TYPE_FABRIC, 0U,
                outBoundP(&expectedHandle, sizeof(expectedHandle)))
            .will(returnValue(ACL_SUCCESS));

        EXPECT_EQ(memory->ImportMemberHandle(1U, shareableDesc, mapping, importedHandle, ownsHandle), HCCL_SUCCESS);
        EXPECT_EQ(importedHandle, expectedHandle);
        EXPECT_TRUE(ownsHandle);
    }

    TEST_F(TestUbMemSymmetricMemoryWithMock, Ut_MapAllMembers_When_PartialMapFails_Expect_ImmediateCleanup)
    {
        auto memory = MakeSymmetricMemory();
        memory->heapBase_ = reinterpret_cast<void*>(TEST_WINDOW_BASE + TEST_STRIDE);
        memory->stride_ = TEST_STRIDE;
        memory->granularity_ = TEST_GRANULARITY;
        memory->selfMember_ = 0U;
        memory->lsaTeamSize_ = 2U;
        UbMemSymmetricMemory::PaMappingInfo mapping;
        mapping.baseVaSize = TEST_GRANULARITY;
        mapping.paHandle = reinterpret_cast<aclrtDrvMemHandle>(TEST_WINDOW_BASE);
        std::vector<std::vector<uint8_t>> shareableDescs(2U, std::vector<uint8_t>(sizeof(aclrtMemFabricHandle), 1U));
        aclrtDrvMemHandle remoteHandle = reinterpret_cast<aclrtDrvMemHandle>(TEST_WINDOW_BASE + TEST_GRANULARITY);
        std::vector<CommMem> memberMems;

        MOCKER_CPP(aclrtMemImportFromShareableHandleV2)
            .expects(once())
            .with(mockcpp::any(), ACL_MEM_SHARE_HANDLE_TYPE_FABRIC, 0U, outBoundP(&remoteHandle, sizeof(remoteHandle)))
            .will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtMapMem).stubs().will(invoke(StubMapMemFailOnSecondCall));
        MOCKER_CPP(aclrtUnmapMem).expects(once()).will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtFreePhysical).expects(once()).with(remoteHandle).will(returnValue(ACL_SUCCESS));

        EXPECT_EQ(memory->MapAllMembers(0U, shareableDescs, mapping, memberMems), HCCL_E_RUNTIME);
        EXPECT_EQ(g_mapMemCallCount, 2U);
        EXPECT_EQ(memory->activeMappingCount_, 0U);
        EXPECT_TRUE(mapping.peerMappings.empty());
        EXPECT_EQ(mapping.paHandle, reinterpret_cast<aclrtDrvMemHandle>(TEST_WINDOW_BASE));

        memory->heapBase_ = nullptr;
    }

    TEST(TestUbMemSymmetricMemoryAgent, Ut_Init_When_SingleMember_Expect_ParaError)
    {
        const std::vector<uint32_t> ranks{3U};
        UbMemSymmetricMemoryAgent agent(nullptr, 0, 3U, ranks, 0U, "ut_single");

        EXPECT_EQ(agent.Init(), HCCL_E_PARA);
    }

    TEST_F(TestUbMemSymmetricMemoryWithMock, Ut_ExchangeInfo_When_TwoMembers_Expect_MemberOrderedData)
    {
        const std::vector<uint32_t> ranks{0U, 2U};
        UbMemSymmetricMemoryAgent agent(nullptr, 0, 0U, ranks, 0U, "ut_multi");
        agent.neighborLinksChecked_ = true;
        agent.isExchangeInfo_ = true;
        agent.leftSocket_ = reinterpret_cast<SocketHandler>(TEST_WINDOW_BASE);
        agent.rightSocket_ = agent.leftSocket_;
        uint32_t localValue = 3U;
        uint32_t remoteValue = 5U;
        UbmemPacket remotePacket{};
        remotePacket.type = UbmemPacketType::DATA;
        remotePacket.memberId = 1U;
        ASSERT_EQ(memcpy_s(remotePacket.data, sizeof(remotePacket.data), &remoteValue, sizeof(remoteValue)), EOK);
        const auto* packetBegin = reinterpret_cast<const uint8_t*>(&remotePacket);
        g_recvFrames = {std::vector<uint8_t>(packetBegin, packetBegin + sizeof(remotePacket))};
        MOCKER(SocketSendNb).stubs().will(invoke(StubSocketSend));
        MOCKER(SocketRecvNb).stubs().will(invoke(StubSocketRecv));
        MOCKER(hrtSetDevice).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER(hrtResetDevice).stubs().will(returnValue(HCCL_SUCCESS));
        ASSERT_EQ(agent.InitRecvThread(), HCCL_SUCCESS);

        std::vector<uint32_t> output(ranks.size(), 0U);
        EXPECT_EQ(agent.ExchangeInfo(&localValue, output.data(), sizeof(localValue)), HCCL_SUCCESS);
        EXPECT_EQ(output[0], localValue);
        EXPECT_EQ(output[1], remoteValue);
        EXPECT_EQ(g_recvFrameIndex, g_recvFrames.size());

        agent.leftSocket_ = nullptr;
        agent.rightSocket_ = nullptr;
    }

    TEST_F(TestUbMemSymmetricMemoryWithMock, Ut_RegisterWindow_When_PostExchangeMapFails_Expect_Cleanup)
    {
        CollComm owner(nullptr, 0U, "ut_cleanup", ManagerCallbacks{}, CollCommInitMode::simpleMode);
        const std::vector<uint32_t> ranks{0U, 2U};
        UbMemSymmetricMemory memory(&owner, reinterpret_cast<HcommTeamHandle>(TEST_WINDOW_BASE), 0U, ranks);
        memory.agent_ = std::make_unique<UbMemSymmetricMemoryAgent>(nullptr, 0, 0U, ranks, 0U, "ut_cleanup");
        memory.agent_->neighborLinksChecked_ = true;
        memory.agent_->isExchangeInfo_ = true;
        memory.agent_->leftSocket_ = reinterpret_cast<SocketHandler>(TEST_WINDOW_BASE);
        memory.agent_->rightSocket_ = memory.agent_->leftSocket_;
        SymmetricVaInitMockData vaInitData;
        MockSymmetricVaInit(vaInitData);
        ASSERT_EQ(memory.InitSymmetricVa(), HCCL_SUCCESS);

        void* baseUserVa = reinterpret_cast<void*>(TEST_WINDOW_BASE);
        size_t baseVaSize = TEST_GRANULARITY;
        aclrtDrvMemHandle localHandle = reinterpret_cast<aclrtDrvMemHandle>(TEST_WINDOW_BASE + TEST_GRANULARITY);
        int32_t localPid = 1;
        int32_t remotePid = 2;

        UbmemPacket remotePidPacket{};
        remotePidPacket.type = UbmemPacketType::DATA;
        remotePidPacket.memberId = 1U;
        ASSERT_EQ(memcpy_s(remotePidPacket.data, sizeof(remotePidPacket.data), &remotePid, sizeof(remotePid)), EOK);
        UbmemShareableInfo remoteInfo{};
        remoteInfo.size = baseVaSize;
        UbmemPacket remoteInfoPacket{};
        remoteInfoPacket.type = UbmemPacketType::DATA;
        remoteInfoPacket.memberId = 1U;
        ASSERT_EQ(memcpy_s(remoteInfoPacket.data, sizeof(remoteInfoPacket.data), &remoteInfo, sizeof(remoteInfo)), EOK);
        const auto* pidPacketBegin = reinterpret_cast<const uint8_t*>(&remotePidPacket);
        const auto* infoPacketBegin = reinterpret_cast<const uint8_t*>(&remoteInfoPacket);
        g_recvFrames
            = {std::vector<uint8_t>(pidPacketBegin, pidPacketBegin + sizeof(remotePidPacket)),
               std::vector<uint8_t>(infoPacketBegin, infoPacketBegin + sizeof(remoteInfoPacket))};

        MOCKER(SocketSendNb).expects(exactly(2)).will(invoke(StubSocketSend));
        MOCKER(SocketRecvNb).expects(exactly(2)).will(invoke(StubSocketRecv));
        MOCKER(hrtSetDevice).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER(hrtResetDevice).stubs().will(returnValue(HCCL_SUCCESS));
        ASSERT_EQ(memory.agent_->InitRecvThread(), HCCL_SUCCESS);
        MOCKER_CPP(aclrtMemGetAddressRange)
            .expects(once())
            .with(baseUserVa, outBoundP(&baseUserVa, sizeof(baseUserVa)), outBoundP(&baseVaSize, sizeof(baseVaSize)))
            .will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtMemRetainAllocationHandle)
            .expects(once())
            .with(baseUserVa, outBoundP(&localHandle, sizeof(localHandle)))
            .will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtMemExportToShareableHandleV2).expects(once()).will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtDeviceGetBareTgid)
            .stubs()
            .with(outBoundP(&localPid, sizeof(localPid)))
            .will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtMemSetPidToShareableHandleV2).expects(once()).will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtMapMem).expects(once()).will(returnValue(ACL_ERROR_RT_PARAM_INVALID));
        MOCKER_CPP(aclrtReleaseMemAddress).expects(once()).will(returnValue(ACL_SUCCESS));

        HcclCommSymWindow window = reinterpret_cast<void*>(TEST_DEVICE_WINDOW);
        HcclComm comm = reinterpret_cast<HcclComm>(TEST_WINDOW_BASE + 3U * TEST_STRIDE);
        EXPECT_EQ(memory.RegisterWindow(baseUserVa, baseVaSize, window, comm), HCCL_E_RUNTIME);
        EXPECT_TRUE(memory.windowsByHandle_.empty());
        ASSERT_EQ(memory.paMappingMap_.size(), 1U);
        EXPECT_EQ(memory.paMappingMap_.begin()->second->refCount, 0U);
        EXPECT_TRUE(memory.paMappingMap_.begin()->second->peerMappings.empty());
        EXPECT_FALSE(memory.paMappingMap_.begin()->second->vaOffsetReserved);
        EXPECT_EQ(g_recvFrameIndex, g_recvFrames.size());
        memory.agent_->leftSocket_ = nullptr;
        memory.agent_->rightSocket_ = nullptr;
    }

    TEST_F(TestUbMemSymmetricMemoryWithMock, Ut_RegisterAndFinalize_When_TwoMembers_Expect_LifecycleComplete)
    {
        CollComm collComm(nullptr, 0U, "ut_multi_lifecycle", ManagerCallbacks{}, CollCommInitMode::simpleMode);
        const std::vector<uint32_t> ranks{0U, 2U};
        UbMemSymmetricMemory memory(&collComm, reinterpret_cast<HcommTeamHandle>(TEST_WINDOW_BASE), 0U, ranks);
        memory.agent_ = std::make_unique<UbMemSymmetricMemoryAgent>(nullptr, 0, 0U, ranks, 0U, "ut_multi_lifecycle");
        memory.agent_->neighborLinksChecked_ = true;
        memory.agent_->isExchangeInfo_ = true;
        memory.agent_->leftSocket_ = reinterpret_cast<SocketHandler>(TEST_WINDOW_BASE);
        memory.agent_->rightSocket_ = memory.agent_->leftSocket_;
        SymmetricVaInitMockData vaInitData;
        MockSymmetricVaInit(vaInitData);
        ASSERT_EQ(memory.InitSymmetricVa(), HCCL_SUCCESS);

        void* baseUserVa = reinterpret_cast<void*>(TEST_WINDOW_BASE);
        size_t baseVaSize = TEST_GRANULARITY;
        aclrtDrvMemHandle localHandle = reinterpret_cast<aclrtDrvMemHandle>(TEST_WINDOW_BASE + TEST_GRANULARITY);
        aclrtDrvMemHandle remoteHandle = reinterpret_cast<aclrtDrvMemHandle>(TEST_WINDOW_BASE + 2U * TEST_GRANULARITY);
        int32_t localPid = 1;
        int32_t remotePid = 2;

        UbmemPacket remotePidPacket{};
        remotePidPacket.type = UbmemPacketType::DATA;
        remotePidPacket.memberId = 1U;
        ASSERT_EQ(memcpy_s(remotePidPacket.data, sizeof(remotePidPacket.data), &remotePid, sizeof(remotePid)), EOK);
        UbmemShareableInfo remoteInfo{};
        remoteInfo.offset = 0U;
        remoteInfo.size = baseVaSize;
        UbmemPacket remoteInfoPacket{};
        remoteInfoPacket.type = UbmemPacketType::DATA;
        remoteInfoPacket.memberId = 1U;
        ASSERT_EQ(memcpy_s(remoteInfoPacket.data, sizeof(remoteInfoPacket.data), &remoteInfo, sizeof(remoteInfo)), EOK);
        const auto* pidPacketBegin = reinterpret_cast<const uint8_t*>(&remotePidPacket);
        const auto* infoPacketBegin = reinterpret_cast<const uint8_t*>(&remoteInfoPacket);
        g_recvFrames
            = {std::vector<uint8_t>(pidPacketBegin, pidPacketBegin + sizeof(remotePidPacket)),
               std::vector<uint8_t>(infoPacketBegin, infoPacketBegin + sizeof(remoteInfoPacket))};

        MOCKER(SocketSendNb).expects(exactly(2)).will(invoke(StubSocketSend));
        MOCKER(SocketRecvNb).expects(exactly(2)).will(invoke(StubSocketRecv));
        MOCKER(hrtSetDevice).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER(hrtResetDevice).stubs().will(returnValue(HCCL_SUCCESS));
        ASSERT_EQ(memory.agent_->InitRecvThread(), HCCL_SUCCESS);
        MOCKER_CPP(aclrtMemGetAddressRange)
            .expects(once())
            .with(baseUserVa, outBoundP(&baseUserVa, sizeof(baseUserVa)), outBoundP(&baseVaSize, sizeof(baseVaSize)))
            .will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtMemRetainAllocationHandle)
            .expects(once())
            .with(baseUserVa, outBoundP(&localHandle, sizeof(localHandle)))
            .will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtMemExportToShareableHandleV2).expects(once()).will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtDeviceGetBareTgid)
            .expects(once())
            .with(outBoundP(&localPid, sizeof(localPid)))
            .will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtMemSetPidToShareableHandleV2).expects(once()).will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtMemImportFromShareableHandleV2)
            .expects(once())
            .with(mockcpp::any(), ACL_MEM_SHARE_HANDLE_TYPE_FABRIC, 0U, outBoundP(&remoteHandle, sizeof(remoteHandle)))
            .will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtMapMem).expects(exactly(2)).will(returnValue(ACL_SUCCESS));
        MOCKER(HcommTeamBindUbSymmetricWindow)
            .expects(once())
            .will(returnValue(static_cast<HcommResult>(HCOMM_SUCCESS)));
        MOCKER_CPP(aclrtUnmapMem).expects(exactly(2)).will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtFreePhysical).expects(once()).with(remoteHandle).will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtReleaseMemAddress).expects(once()).will(returnValue(ACL_SUCCESS));

        HcclCommSymWindow window = reinterpret_cast<void*>(TEST_DEVICE_WINDOW);
        HcclComm comm = reinterpret_cast<HcclComm>(TEST_WINDOW_BASE + 3U * TEST_STRIDE);
        ASSERT_EQ(memory.RegisterWindow(baseUserVa, baseVaSize, window, comm), HCCL_SUCCESS);
        ASSERT_EQ(memory.paMappingMap_.size(), 1U);
        const auto& paMapping = memory.paMappingMap_.begin()->second;
        ASSERT_NE(paMapping, nullptr);
        ASSERT_EQ(paMapping->memberMems.size(), ranks.size());
        ASSERT_EQ(paMapping->peerMappings.size(), ranks.size());
        EXPECT_FALSE(paMapping->peerMappings[0].ownsHandle);
        EXPECT_TRUE(paMapping->peerMappings[1].ownsHandle);
        EXPECT_EQ(paMapping->memberMems[0].addr, memory.heapBase_);
        EXPECT_EQ(
            paMapping->memberMems[1].addr,
            static_cast<void*>(static_cast<uint8_t*>(memory.heapBase_) + memory.stride_));
        EXPECT_EQ(paMapping->memberMems[0].size, baseVaSize);
        EXPECT_EQ(paMapping->memberMems[1].size, baseVaSize);
        EXPECT_EQ(memory.activeMappingCount_, ranks.size());

        memory.agent_->leftSocket_ = nullptr;
        memory.agent_->rightSocket_ = nullptr;
        memory.Finalize();
        EXPECT_TRUE(memory.finalized_);
        EXPECT_TRUE(memory.windowsByAddress_.empty());
        EXPECT_TRUE(memory.windowsByHandle_.empty());
        EXPECT_TRUE(memory.paMappingMap_.empty());
        EXPECT_EQ(memory.activeMappingCount_, 0U);
        EXPECT_EQ(memory.heapBase_, nullptr);
        EXPECT_EQ(g_recvFrameIndex, g_recvFrames.size());
        HcclComm ownerComm = nullptr;
        EXPECT_EQ(GetHcommWindowComm(window, ownerComm), HCCL_E_NOT_FOUND);
    }

    TEST_F(TestUbMemSymmetricMemoryWithMock, Ut_RegisterOverlapAndReregister_When_SameAllocation_Expect_ReusePaMapping)
    {
        CollComm owner(nullptr, 0U, "ut_overlap", ManagerCallbacks{}, CollCommInitMode::simpleMode);
        const std::vector<uint32_t> ranks{0U, 2U};
        UbMemSymmetricMemory memory(&owner, reinterpret_cast<HcommTeamHandle>(TEST_WINDOW_BASE), 0U, ranks);
        memory.agent_ = std::make_unique<UbMemSymmetricMemoryAgent>(nullptr, 0, 0U, ranks, 0U, "ut_overlap");
        memory.agent_->neighborLinksChecked_ = true;
        memory.agent_->isExchangeInfo_ = true;
        memory.agent_->leftSocket_ = reinterpret_cast<SocketHandler>(TEST_WINDOW_BASE);
        memory.agent_->rightSocket_ = memory.agent_->leftSocket_;
        SymmetricVaInitMockData vaInitData;
        MockSymmetricVaInit(vaInitData);
        ASSERT_EQ(memory.InitSymmetricVa(), HCCL_SUCCESS);

        void* baseUserVa = reinterpret_cast<void*>(TEST_WINDOW_BASE);
        size_t baseVaSize = TEST_GRANULARITY;
        aclrtDrvMemHandle localHandle = reinterpret_cast<aclrtDrvMemHandle>(TEST_WINDOW_BASE + TEST_GRANULARITY);
        aclrtDrvMemHandle remoteHandle = reinterpret_cast<aclrtDrvMemHandle>(TEST_WINDOW_BASE + 2U * TEST_GRANULARITY);
        int32_t localPid = 1;
        int32_t remotePid = 2;
        UbmemPacket remotePidPacket{};
        remotePidPacket.type = UbmemPacketType::DATA;
        remotePidPacket.memberId = 1U;
        ASSERT_EQ(memcpy_s(remotePidPacket.data, sizeof(remotePidPacket.data), &remotePid, sizeof(remotePid)), EOK);
        UbmemShareableInfo remoteInfo{};
        remoteInfo.size = baseVaSize;
        UbmemPacket remoteInfoPacket{};
        remoteInfoPacket.type = UbmemPacketType::DATA;
        remoteInfoPacket.memberId = 1U;
        ASSERT_EQ(memcpy_s(remoteInfoPacket.data, sizeof(remoteInfoPacket.data), &remoteInfo, sizeof(remoteInfo)), EOK);
        const auto* pidPacketBegin = reinterpret_cast<const uint8_t*>(&remotePidPacket);
        const auto* infoPacketBegin = reinterpret_cast<const uint8_t*>(&remoteInfoPacket);
        g_recvFrames
            = {std::vector<uint8_t>(pidPacketBegin, pidPacketBegin + sizeof(remotePidPacket)),
               std::vector<uint8_t>(infoPacketBegin, infoPacketBegin + sizeof(remoteInfoPacket)),
               std::vector<uint8_t>(infoPacketBegin, infoPacketBegin + sizeof(remoteInfoPacket)),
               std::vector<uint8_t>(infoPacketBegin, infoPacketBegin + sizeof(remoteInfoPacket)),
               std::vector<uint8_t>(infoPacketBegin, infoPacketBegin + sizeof(remoteInfoPacket))};
        MOCKER(SocketSendNb).expects(exactly(5)).will(invoke(StubSocketSend));
        MOCKER(SocketRecvNb).expects(exactly(5)).will(invoke(StubSocketRecv));
        MOCKER(hrtSetDevice).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER(hrtResetDevice).stubs().will(returnValue(HCCL_SUCCESS));
        ASSERT_EQ(memory.agent_->InitRecvThread(), HCCL_SUCCESS);
        MOCKER_CPP(aclrtMemGetAddressRange)
            .expects(exactly(4))
            .with(
                mockcpp::any(), outBoundP(&baseUserVa, sizeof(baseUserVa)), outBoundP(&baseVaSize, sizeof(baseVaSize)))
            .will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtMemRetainAllocationHandle)
            .expects(exactly(4))
            .with(baseUserVa, outBoundP(&localHandle, sizeof(localHandle)))
            .will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtMemExportToShareableHandleV2).expects(once()).will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtDeviceGetBareTgid)
            .stubs()
            .with(outBoundP(&localPid, sizeof(localPid)))
            .will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtMemSetPidToShareableHandleV2).expects(once()).will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtMemImportFromShareableHandleV2)
            .expects(exactly(2))
            .with(mockcpp::any(), ACL_MEM_SHARE_HANDLE_TYPE_FABRIC, 0U, outBoundP(&remoteHandle, sizeof(remoteHandle)))
            .will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtMapMem).expects(exactly(4)).will(returnValue(ACL_SUCCESS));
        MOCKER(HcommTeamBindUbSymmetricWindow).expects(exactly(4)).will(invoke(StubBindUbSymmetricWindow));
        MOCKER_CPP(aclrtUnmapMem).expects(exactly(4)).will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtFreePhysical).expects(exactly(2)).with(remoteHandle).will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtReleaseMemAddress).expects(once()).will(returnValue(ACL_SUCCESS));

        HcclComm comm = reinterpret_cast<HcclComm>(TEST_WINDOW_BASE + 3U * TEST_STRIDE);
        HcclCommSymWindow firstWindow = reinterpret_cast<void*>(TEST_DEVICE_WINDOW);
        HcclCommSymWindow overlapWindow = reinterpret_cast<void*>(TEST_DEVICE_WINDOW + TEST_GRANULARITY);
        HcclCommSymWindow duplicateWindow = reinterpret_cast<void*>(TEST_DEVICE_WINDOW + 2U * TEST_GRANULARITY);
        HcclCommSymWindow reregisteredWindow = reinterpret_cast<void*>(TEST_DEVICE_WINDOW + 3U * TEST_GRANULARITY);
        ASSERT_EQ(memory.RegisterWindow(baseUserVa, baseVaSize, firstWindow, comm), HCCL_SUCCESS);
        ASSERT_EQ(
            memory.RegisterWindow(
                reinterpret_cast<void*>(TEST_WINDOW_BASE + 0x80U), baseVaSize - 0x80U, overlapWindow, comm),
            HCCL_SUCCESS);
        ASSERT_EQ(memory.RegisterWindow(baseUserVa, baseVaSize, duplicateWindow, comm), HCCL_SUCCESS);
        ASSERT_EQ(memory.paMappingMap_.size(), 1U);
        EXPECT_EQ(memory.paMappingMap_.begin()->second->refCount, 3U);
        auto overlapRecordIter = memory.windowsByHandle_.find(overlapWindow);
        ASSERT_NE(overlapRecordIter, memory.windowsByHandle_.end());
        ASSERT_NE(overlapRecordIter->second, nullptr);
        EXPECT_EQ(overlapRecordIter->second->userVa, reinterpret_cast<void*>(TEST_WINDOW_BASE + 0x80U));
        EXPECT_EQ(overlapRecordIter->second->userSize, baseVaSize - 0x80U);
        EXPECT_EQ(overlapRecordIter->second->offsetInBaseVa, 0x80U);
        ASSERT_EQ(g_boundBaseVas.size(), 3U);
        ASSERT_EQ(g_boundUserSizes.size(), 3U);
        ASSERT_EQ(g_boundFirstMemberMems.size(), 3U);
        void* mappedBase = memory.heapBase_;
        EXPECT_EQ(g_boundBaseVas[0], mappedBase);
        EXPECT_EQ(g_boundBaseVas[1], static_cast<void*>(static_cast<uint8_t*>(mappedBase) + 0x80U));
        EXPECT_EQ(g_boundBaseVas[2], mappedBase);
        EXPECT_EQ(g_boundUserSizes[0], baseVaSize);
        EXPECT_EQ(g_boundUserSizes[1], baseVaSize - 0x80U);
        EXPECT_EQ(g_boundUserSizes[2], baseVaSize);
        EXPECT_EQ(g_boundFirstMemberMems[1].addr, g_boundBaseVas[1]);
        EXPECT_EQ(g_boundFirstMemberMems[1].size, baseVaSize - 0x80U);

        EXPECT_EQ(memory.DeregisterWindow(duplicateWindow), HCCL_SUCCESS);
        ASSERT_EQ(memory.paMappingMap_.size(), 1U);
        EXPECT_EQ(memory.paMappingMap_.begin()->second->refCount, 2U);
        EXPECT_EQ(memory.activeMappingCount_, ranks.size());
        EXPECT_EQ(memory.DeregisterWindow(overlapWindow), HCCL_SUCCESS);
        ASSERT_EQ(memory.paMappingMap_.size(), 1U);
        EXPECT_EQ(memory.paMappingMap_.begin()->second->refCount, 1U);
        EXPECT_EQ(memory.activeMappingCount_, ranks.size());
        EXPECT_EQ(memory.DeregisterWindow(firstWindow), HCCL_SUCCESS);
        ASSERT_EQ(memory.paMappingMap_.size(), 1U);
        EXPECT_EQ(memory.paMappingMap_.begin()->second->refCount, 0U);
        EXPECT_TRUE(memory.paMappingMap_.begin()->second->peerMappings.empty());
        EXPECT_TRUE(memory.paMappingMap_.begin()->second->shareableHandleReady);
        EXPECT_TRUE(memory.paMappingMap_.begin()->second->memberAccessGranted);
        EXPECT_FALSE(memory.paMappingMap_.begin()->second->vaOffsetReserved);
        EXPECT_EQ(memory.activeMappingCount_, 0U);

        ASSERT_EQ(memory.RegisterWindow(baseUserVa, baseVaSize, reregisteredWindow, comm), HCCL_SUCCESS);
        ASSERT_EQ(memory.paMappingMap_.size(), 1U);
        EXPECT_EQ(memory.paMappingMap_.begin()->second->refCount, 1U);
        EXPECT_TRUE(memory.paMappingMap_.begin()->second->vaOffsetReserved);
        EXPECT_EQ(memory.activeMappingCount_, ranks.size());
        EXPECT_EQ(memory.DeregisterWindow(reregisteredWindow), HCCL_SUCCESS);
        EXPECT_EQ(memory.paMappingMap_.begin()->second->refCount, 0U);
        EXPECT_EQ(memory.activeMappingCount_, 0U);
        EraseHcommWindowOwner(firstWindow);
        EraseHcommWindowOwner(overlapWindow);
        EraseHcommWindowOwner(duplicateWindow);
        EraseHcommWindowOwner(reregisteredWindow);
        EXPECT_EQ(g_recvFrameIndex, g_recvFrames.size());
        memory.agent_->leftSocket_ = nullptr;
        memory.agent_->rightSocket_ = nullptr;
    }

} // namespace
} // namespace hccl
