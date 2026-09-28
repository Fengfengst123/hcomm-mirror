/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>
#include "ub_mem_endpoint.h"
#include "hcomm_res.h"
#include "hcomm_c_adpt.h"
#include "ip_address.h"
#include "hccp.h"
#include "buffer.h"
#include "network_api_exception.h"
#include "endpoint.h"
#include "adapter_rts.h"

class AicpuUbRtpEndpointTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AicpuUbRtpEndpointTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "AicpuUbRtpEndpointTest tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in AicpuUbRtpEndpointTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        std::cout << "A Test case in AicpuUbRtpEndpointTest TearDown" << std::endl;
    }

    void CreateEndpointDesc(EndpointDesc& endpointDesc, const std::string& ip = "1.0.0.0")
    {
        Hccl::IpAddress localIp(ip);
        endpointDesc.protocol = COMM_PROTOCOL_UB_RTP;
        endpointDesc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
        endpointDesc.commAddr.addr = localIp.GetBinaryAddress().addr;
        endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    }

    HcommResult CreateEndpoint(EndpointHandle& endpointHandle, const std::string& ip = "1.0.0.0")
    {
        EndpointDesc endpointDesc;
        CreateEndpointDesc(endpointDesc, ip);
        return HcommEndpointCreate(&endpointDesc, &endpointHandle);
    }

    CommMem CreateCommMem(void* addr, size_t size, CommMemType type)
    {
        CommMem mem;
        mem.type = type;
        mem.size = size;
        mem.addr = addr;
        return mem;
    }
};

static HcclResult stub_hrtGetDeviceType_950(DevType& devType)
{
    devType = DevType::DEV_TYPE_950;
    return HCCL_SUCCESS;
}

static HcclResult stub_hrtGetDeviceType_910B(DevType& devType)
{
    devType = DevType::DEV_TYPE_910B;
    return HCCL_SUCCESS;
}

TEST_F(AicpuUbRtpEndpointTest, Ut_HcommEndpointCreate_When_UbRtp_Device_Expect_Return_SUCCESS)
{
    MOCKER(hrtGetDeviceType).stubs().will(invoke(stub_hrtGetDeviceType_950));

    EndpointDesc endpointDesc;
    CreateEndpointDesc(endpointDesc);
    EndpointHandle endpointHandle;

    HcommResult ret = HcommEndpointCreate(&endpointDesc, &endpointHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(AicpuUbRtpEndpointTest, Ut_HcommEndpointCreate_When_UbRtp_On_910B_Expect_Return_NOT_SUPPORT)
{
    MOCKER(hrtGetDeviceType).stubs().will(invoke(stub_hrtGetDeviceType_910B));

    EndpointDesc endpointDesc;
    CreateEndpointDesc(endpointDesc);
    EndpointHandle endpointHandle;

    HcommResult ret = HcommEndpointCreate(&endpointDesc, &endpointHandle);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(AicpuUbRtpEndpointTest, Ut_HcommEndpointCreate_When_UbCtp_On_910B_Expect_Return_NOT_SUPPORT)
{
    MOCKER(hrtGetDeviceType).stubs().will(invoke(stub_hrtGetDeviceType_910B));

    EndpointDesc endpointDesc;
    CreateEndpointDesc(endpointDesc);
    endpointDesc.protocol = COMM_PROTOCOL_UB_CTP;
    EndpointHandle endpointHandle;

    HcommResult ret = HcommEndpointCreate(&endpointDesc, &endpointHandle);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(AicpuUbRtpEndpointTest, Ut_HcommEndpointCreate_When_Hccs_On_950_Expect_Return_NOT_SUPPORT)
{
    MOCKER(hrtGetDeviceType).stubs().will(invoke(stub_hrtGetDeviceType_950));

    EndpointDesc endpointDesc;
    CreateEndpointDesc(endpointDesc);
    endpointDesc.protocol = COMM_PROTOCOL_HCCS;
    EndpointHandle endpointHandle;

    HcommResult ret = HcommEndpointCreate(&endpointDesc, &endpointHandle);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(AicpuUbRtpEndpointTest, Ut_HcommEndpointCreate_When_Roce_Device_On_950_Expect_Return_NOT_SUPPORT)
{
    MOCKER(hrtGetDeviceType).stubs().will(invoke(stub_hrtGetDeviceType_950));

    EndpointDesc endpointDesc;
    CreateEndpointDesc(endpointDesc);
    endpointDesc.protocol = COMM_PROTOCOL_ROCE;
    EndpointHandle endpointHandle;

    HcommResult ret = HcommEndpointCreate(&endpointDesc, &endpointHandle);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(AicpuUbRtpEndpointTest, Ut_HcommEndpointCreate_When_Roce_Host_On_910B_Expect_Return_NOT_SUPPORT)
{
    MOCKER(hrtGetDeviceType).stubs().will(invoke(stub_hrtGetDeviceType_910B));

    EndpointDesc endpointDesc;
    CreateEndpointDesc(endpointDesc);
    endpointDesc.protocol = COMM_PROTOCOL_ROCE;
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    EndpointHandle endpointHandle;

    HcommResult ret = HcommEndpointCreate(&endpointDesc, &endpointHandle);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(AicpuUbRtpEndpointTest, Ut_HcommEndpointCreate_When_UbCtp_Host_On_910B_Expect_Return_NOT_SUPPORT)
{
    MOCKER(hrtGetDeviceType).stubs().will(invoke(stub_hrtGetDeviceType_910B));

    EndpointDesc endpointDesc;
    CreateEndpointDesc(endpointDesc);
    endpointDesc.protocol = COMM_PROTOCOL_UB_CTP;
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    EndpointHandle endpointHandle;

    HcommResult ret = HcommEndpointCreate(&endpointDesc, &endpointHandle);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(AicpuUbRtpEndpointTest, Ut_HcommEndpointCreate_When_UbRtp_Host_Expect_Return_ERROR)
{
    EndpointDesc endpointDesc;
    CreateEndpointDesc(endpointDesc);
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    EndpointHandle endpointHandle;

    HcommResult ret = HcommEndpointCreate(&endpointDesc, &endpointHandle);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(AicpuUbRtpEndpointTest, Ut_When_Register_Memory_NORMAL_Expect_Return_SUCCESS)
{
    MOCKER(hrtGetDeviceType).stubs().will(invoke(stub_hrtGetDeviceType_950));

    EndpointHandle endpointHandle;
    HcommResult ret = CreateEndpoint(endpointHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    CommMem mem = CreateCommMem((void*)0x01, 10, COMM_MEM_TYPE_DEVICE);
    void* memHandle;

    ret = HcommMemReg(endpointHandle, "memTag", &mem, &memHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcommMemUnreg(endpointHandle, memHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}
