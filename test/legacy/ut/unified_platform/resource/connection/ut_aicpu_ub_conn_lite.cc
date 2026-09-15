/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#define private public
#include "ub_conn_lite.h"
#include "reduce_in.h"
#include "stream_lite.h"
#include "ascend_hal.h"
#include "rtsq_base.h"
#include "sqe.h"
#include "internal_exception.h"
#undef private

using namespace Hccl;

class AicpuUbConnLiteTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "UbConnLite tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "UbConnLite tests tear down." << std::endl; }

    virtual void SetUp()
    {
        MOCKER_CPP(&RtsqBase::QuerySqBaseAddr)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(reinterpret_cast<u64>(&mockSq)));
        MOCKER_CPP(&RtsqBase::QuerySqDepth)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(static_cast<u32>(AC_SQE_MAX_CNT)));
        MOCKER_CPP(&RtsqBase::QuerySqStatusByType).stubs().with(mockcpp::any()).will(returnValue(static_cast<u32>(0)));
        MOCKER_CPP(&RtsqBase::ConfigSqStatusByType).stubs();
        std::cout << "A Test case in UbConnLite SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in UbConnLite TearDown" << std::endl;
    }

    u8 mockSq[AC_SQE_SIZE * AC_SQE_MAX_CNT]{0};
};

TEST_F(AicpuUbConnLiteTest, test_RmaConnLite)
{
    RmaConnLite rdma(1);
    EXPECT_EQ(1, rdma.GetQpVa());

    UbJettyLiteId id(1, 1, 1);
    UbJettyLiteAttr attr(1, 1, 1, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;

    RmaConnLite ub(id, attr, rmtEid);

    std::cout << ub.Describe() << std::endl;

    EXPECT_EQ(1, ub.GetUbJettyLiteId().GetDieId());
    EXPECT_EQ(1, ub.GetUbJettyLiteAttr().dbAddr_);
    EXPECT_EQ(1, ub.GetRmtEid().raw[0]);
}

TEST_F(AicpuUbConnLiteTest, test_UBConnLite_construct_by_unique_id)
{
    std::vector<char> data(57, 0);
    UbConnLite ubConn(data);
}

TEST_F(AicpuUbConnLiteTest, test_UBConnLite_Read)
{
    UbJettyLiteId id(1, 1, 1);
    u8 sqVa[1024];
    u64 sqVAaddr = reinterpret_cast<u64>(sqVa);
    UbJettyLiteAttr attr(1, sqVAaddr, 1, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);
    UbConnLite ubConn(id, attr, rmtEid);

    EXPECT_EQ(1, ubConn.GetUbJettyLiteId().GetDieId());
    EXPECT_EQ(1, ubConn.GetUbJettyLiteAttr().dbAddr_);
    EXPECT_EQ(1, ubConn.GetRmtEid().raw[0]);

    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    RmaBufSliceLite loc(0x1111, 64, 1, 1);
    RmtRmaBufSliceLite rmt(0x2222, 64, 1, 1, 1, UINT32_MAX);
    RmtRmaBufSliceLite notify(0x2222, 64, 1, 1, 1, UINT32_MAX);
    u64 notifyData(0x000000000000ffff);
    SqeConfigLite cfg;
    ConnLiteOperationOut out;
    u8 data[512];
    out.pi = 0;
    out.data = data;
    UdmaSqeWrite* sqe = (UdmaSqeWrite*)out.data;
    out.dataSize = 64;
    rmt.GetTokenValue();
    EXPECT_NO_THROW(ubConn.Read(loc, rmt, cfg, stream, out));
}

TEST_F(AicpuUbConnLiteTest, test_UBConnLite_Read_Slice)
{
    UbJettyLiteId id(1, 1, 1);
    u8 sqVa[1024];
    u64 sqVAaddr = reinterpret_cast<u64>(sqVa);
    UbJettyLiteAttr attr(1, sqVAaddr, 4, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;

    UbConnLite ubConn(id, attr, rmtEid);

    EXPECT_EQ(1, ubConn.GetUbJettyLiteId().GetDieId());
    EXPECT_EQ(1, ubConn.GetUbJettyLiteAttr().dbAddr_);
    EXPECT_EQ(1, ubConn.GetRmtEid().raw[0]);

    RmaBufSliceLite loc(0x1111, 257 * 1024 * 1024, 1, 1);
    RmtRmaBufSliceLite rmt(0x2222, 257 * 1024 * 1024, 1, 1, 1, UINT32_MAX);
    RmtRmaBufSliceLite notify(0x2222, 64, 1, 1, 1, UINT32_MAX);
    u64 notifyData(0x000000000000ffff);
    SqeConfigLite cfg;
    ConnLiteOperationOut out;
    u8 data[512];
    out.pi = 0;
    out.data = data;
    UdmaSqeWrite* sqe = (UdmaSqeWrite*)out.data;
    out.dataSize = 64;
    rmt.GetTokenValue();
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    EXPECT_NO_THROW(ubConn.Read(loc, rmt, cfg, stream, out));
}

TEST_F(AicpuUbConnLiteTest, test_UBConnLite_ReadReduce_Slice)
{
    UbJettyLiteId id(1, 1, 1);
    u8 sqVa[1024];
    u64 sqVAaddr = reinterpret_cast<u64>(sqVa);
    UbJettyLiteAttr attr(1, sqVAaddr, 4, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;

    UbConnLite ubConn(id, attr, rmtEid);

    EXPECT_EQ(1, ubConn.GetUbJettyLiteId().GetDieId());
    EXPECT_EQ(1, ubConn.GetUbJettyLiteAttr().dbAddr_);
    EXPECT_EQ(1, ubConn.GetRmtEid().raw[0]);

    RmaBufSliceLite loc(0x1111, 257 * 1024 * 1024, 1, 1);
    RmtRmaBufSliceLite rmt(0x2222, 257 * 1024 * 1024, 1, 1, 1, UINT32_MAX);
    RmtRmaBufSliceLite notify(0x2222, 64, 1, 1, 1, UINT32_MAX);
    u64 notifyData(1);
    SqeConfigLite cfg;
    ConnLiteOperationOut out;
    u8 data[512];
    out.pi = 0;
    out.data = data;
    UdmaSqeWrite* sqe = (UdmaSqeWrite*)out.data;
    out.dataSize = 64;
    rmt.GetTokenValue();
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);
    ReduceIn reduceIn(DataType::INT8, ReduceOp::SUM);
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    EXPECT_NO_THROW(ubConn.ReadReduce(reduceIn, loc, rmt, stream, cfg, out));
}

TEST_F(AicpuUbConnLiteTest, test_UBConnLite_ReadReduce)
{
    UbJettyLiteId id(1, 1, 1);
    u8 sqVa[1024];
    u64 sqVAaddr = reinterpret_cast<u64>(sqVa);
    UbJettyLiteAttr attr(1, sqVAaddr, 1, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;

    UbConnLite ubConn(id, attr, rmtEid);

    EXPECT_EQ(1, ubConn.GetUbJettyLiteId().GetDieId());
    EXPECT_EQ(1, ubConn.GetUbJettyLiteAttr().dbAddr_);
    EXPECT_EQ(1, ubConn.GetRmtEid().raw[0]);

    RmaBufSliceLite loc(0x1111, 64, 1, 1);
    RmtRmaBufSliceLite rmt(0x2222, 64, 1, 1, 1, UINT32_MAX);
    RmtRmaBufSliceLite notify(0x2222, 64, 1, 1, 1, UINT32_MAX);
    u64 notifyData(1);
    SqeConfigLite cfg;
    ConnLiteOperationOut out;
    u8 data[512];
    out.pi = 0;
    out.data = data;
    UdmaSqeWrite* sqe = (UdmaSqeWrite*)out.data;
    out.dataSize = 64;
    rmt.GetTokenValue();
    ReduceIn reduceIn(DataType::INT8, ReduceOp::SUM);
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    EXPECT_NO_THROW(ubConn.ReadReduce(reduceIn, loc, rmt, stream, cfg, out));
}

TEST_F(AicpuUbConnLiteTest, test_UBConnLite_Write)
{
    UbJettyLiteId id(1, 1, 1);
    u8 sqVa[1024];
    u64 sqVAaddr = reinterpret_cast<u64>(sqVa);
    UbJettyLiteAttr attr(1, sqVAaddr, 1, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;

    UbConnLite ubConn(id, attr, rmtEid);

    EXPECT_EQ(1, ubConn.GetUbJettyLiteId().GetDieId());
    EXPECT_EQ(1, ubConn.GetUbJettyLiteAttr().dbAddr_);
    EXPECT_EQ(1, ubConn.GetRmtEid().raw[0]);

    RmaBufSliceLite loc(0x1111, 64, 1, 1);
    RmtRmaBufSliceLite rmt(0x2222, 64, 1, 1, 1, UINT32_MAX);
    RmtRmaBufSliceLite notify(0x2222, 64, 1, 1, 1, UINT32_MAX);
    u64 notifyData(0x000000000000ffff);
    SqeConfigLite cfg;
    ConnLiteOperationOut out;
    u8 data[512];
    out.pi = 0;
    out.data = data;
    UdmaSqeWrite* sqe = (UdmaSqeWrite*)out.data;
    out.dataSize = 64;
    rmt.GetTokenValue();
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    EXPECT_NO_THROW(ubConn.Write(loc, rmt, cfg, stream, out));
}

TEST_F(AicpuUbConnLiteTest, test_UBConnLite_Write_Slice)
{
    UbJettyLiteId id(1, 1, 1);
    u8 sqVa[1024];
    u64 sqVAaddr = reinterpret_cast<u64>(sqVa);
    UbJettyLiteAttr attr(1, sqVAaddr, 4, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;

    UbConnLite ubConn(id, attr, rmtEid);

    EXPECT_EQ(1, ubConn.GetUbJettyLiteId().GetDieId());
    EXPECT_EQ(1, ubConn.GetUbJettyLiteAttr().dbAddr_);
    EXPECT_EQ(1, ubConn.GetRmtEid().raw[0]);

    RmaBufSliceLite loc(0x1111, 257 * 1024 * 1024, 1, 1);
    RmtRmaBufSliceLite rmt(0x2222, 257 * 1024 * 1024, 1, 1, 1, UINT32_MAX);
    RmtRmaBufSliceLite notify(0x2222, 64, 1, 1, 1, UINT32_MAX);
    u64 notifyData(0x000000000000ffff);
    SqeConfigLite cfg;
    ConnLiteOperationOut out;
    u8 data[512];
    out.pi = 0;
    out.data = data;
    UdmaSqeWrite* sqe = (UdmaSqeWrite*)out.data;
    out.dataSize = 64;
    rmt.GetTokenValue();
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    EXPECT_NO_THROW(ubConn.Write(loc, rmt, cfg, stream, out));
}

TEST_F(AicpuUbConnLiteTest, test_UBConnLite_WriteReduceWithNotify)
{
    UbJettyLiteId id(1, 1, 1);
    u8 sqVa[1024];
    u64 sqVAaddr = reinterpret_cast<u64>(sqVa);
    UbJettyLiteAttr attr(1, sqVAaddr, 4096, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;

    UbConnLite ubConn(id, attr, rmtEid);

    std::cout << ubConn.Describe() << std::endl;

    EXPECT_EQ(1, ubConn.GetUbJettyLiteId().GetDieId());
    EXPECT_EQ(1, ubConn.GetUbJettyLiteAttr().dbAddr_);
    EXPECT_EQ(1, ubConn.GetRmtEid().raw[0]);

    RmaBufSliceLite loc(0x1111, 64, 1, 1);
    RmtRmaBufSliceLite rmt(0x2222, 64, 1, 1, 1, UINT32_MAX);
    RmtRmaBufSliceLite notify(0x2222, 64, 1, 1, 1, UINT32_MAX);
    u64 notifyData(1);
    SqeConfigLite cfg;
    ConnLiteOperationOut out;
    u8 data[512];
    out.pi = 0;
    out.data = data;
    UdmaSqeWrite* sqe = (UdmaSqeWrite*)out.data;
    out.dataSize = 64;
    rmt.GetTokenValue();
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);

    MOCKER(memset_s).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER(memcpy_s).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));

    EXPECT_NO_THROW(
        ubConn.WriteReduceWithNotify(DataType::INT8, ReduceOp::SUM, loc, rmt, cfg, stream, out, notify, notifyData));
    EXPECT_NO_THROW(
        ubConn.WriteReduceWithNotify(DataType::INT16, ReduceOp::MAX, loc, rmt, cfg, stream, out, notify, notifyData));
    EXPECT_NO_THROW(
        ubConn.WriteReduceWithNotify(DataType::INT32, ReduceOp::MIN, loc, rmt, cfg, stream, out, notify, notifyData));
    EXPECT_NO_THROW(
        ubConn.WriteReduceWithNotify(DataType::FP16, ReduceOp::SUM, loc, rmt, cfg, stream, out, notify, notifyData));
    EXPECT_NO_THROW(
        ubConn.WriteReduceWithNotify(DataType::FP32, ReduceOp::MAX, loc, rmt, cfg, stream, out, notify, notifyData));
    EXPECT_NO_THROW(
        ubConn.WriteReduceWithNotify(DataType::UINT8, ReduceOp::MAX, loc, rmt, cfg, stream, out, notify, notifyData));
    EXPECT_NO_THROW(
        ubConn.WriteReduceWithNotify(DataType::UINT32, ReduceOp::MAX, loc, rmt, cfg, stream, out, notify, notifyData));
}

TEST_F(AicpuUbConnLiteTest, test_UBConnLite_WriteReduceWithNotify_Slice)
{
    UbJettyLiteId id(1, 1, 1);
    u8 sqVa[1024];
    u64 sqVAaddr = reinterpret_cast<u64>(sqVa);
    UbJettyLiteAttr attr(1, sqVAaddr, 4096, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;

    UbConnLite ubConn(id, attr, rmtEid);

    std::cout << ubConn.Describe() << std::endl;

    EXPECT_EQ(1, ubConn.GetUbJettyLiteId().GetDieId());
    EXPECT_EQ(1, ubConn.GetUbJettyLiteAttr().dbAddr_);
    EXPECT_EQ(1, ubConn.GetRmtEid().raw[0]);

    RmaBufSliceLite loc(0x1111, 257 * 1024 * 1024, 1, 1);
    RmtRmaBufSliceLite rmt(0x2222, 257 * 1024 * 1024, 1, 1, 1, UINT32_MAX);
    RmtRmaBufSliceLite notify(0x2222, 64, 1, 1, 1, UINT32_MAX);
    u64 notifyData(1);
    SqeConfigLite cfg;
    ConnLiteOperationOut out;
    u8 data[512];
    out.pi = 0;
    out.data = data;
    UdmaSqeWrite* sqe = (UdmaSqeWrite*)out.data;
    out.dataSize = 64;
    rmt.GetTokenValue();
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);

    MOCKER(memset_s).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER(memcpy_s).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    EXPECT_NO_THROW(
        ubConn.WriteReduceWithNotify(DataType::INT8, ReduceOp::SUM, loc, rmt, cfg, stream, out, notify, notifyData));
    EXPECT_NO_THROW(
        ubConn.WriteReduceWithNotify(DataType::INT16, ReduceOp::MAX, loc, rmt, cfg, stream, out, notify, notifyData));
    EXPECT_NO_THROW(
        ubConn.WriteReduceWithNotify(DataType::INT32, ReduceOp::MIN, loc, rmt, cfg, stream, out, notify, notifyData));
    EXPECT_NO_THROW(
        ubConn.WriteReduceWithNotify(DataType::FP16, ReduceOp::SUM, loc, rmt, cfg, stream, out, notify, notifyData));
    EXPECT_NO_THROW(
        ubConn.WriteReduceWithNotify(DataType::FP32, ReduceOp::MAX, loc, rmt, cfg, stream, out, notify, notifyData));
}

TEST_F(AicpuUbConnLiteTest, test_UBConnLite_WriteWithNotify)
{
    UbJettyLiteId id(1, 1, 1);
    u8 sqVa[1024];
    u64 sqVAaddr = reinterpret_cast<u64>(sqVa);
    UbJettyLiteAttr attr(1, sqVAaddr, 4096, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;

    UbConnLite ubConn(id, attr, rmtEid);

    EXPECT_EQ(1, ubConn.GetUbJettyLiteId().GetDieId());
    EXPECT_EQ(1, ubConn.GetUbJettyLiteAttr().dbAddr_);
    EXPECT_EQ(1, ubConn.GetRmtEid().raw[0]);

    RmaBufSliceLite loc(0x1111, 64, 1, 1);
    RmtRmaBufSliceLite rmt(0x2222, 64, 1, 1, 1, UINT32_MAX);
    RmtRmaBufSliceLite notify(0x2222, 64, 1, 1, 1, UINT32_MAX);
    u64 notifyData(1);
    SqeConfigLite cfg;
    ConnLiteOperationOut out;
    u8 data[512];
    out.pi = 0;
    out.data = data;
    UdmaSqeWrite* sqe = (UdmaSqeWrite*)out.data;
    out.dataSize = 64;
    rmt.GetTokenValue();
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);

    MOCKER(memset_s).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER(memcpy_s).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    EXPECT_NO_THROW(ubConn.WriteWithNotify(loc, rmt, cfg, out, notify, stream, notifyData));
}

TEST_F(AicpuUbConnLiteTest, test_UBConnLite_WriteWithNotify_Detour)
{
    UbJettyLiteId id(1, 1, 1);
    u8 sqVa[1024];
    u64 sqVAaddr = reinterpret_cast<u64>(sqVa);
    UbJettyLiteAttr attr(1, sqVAaddr, 3, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;

    UbConnLite ubConn(id, attr, rmtEid);

    EXPECT_EQ(1, ubConn.GetUbJettyLiteId().GetDieId());
    EXPECT_EQ(1, ubConn.GetUbJettyLiteAttr().dbAddr_);
    EXPECT_EQ(1, ubConn.GetRmtEid().raw[0]);

    RmaBufSliceLite loc(0x1111, 64, 1, 1);
    RmtRmaBufSliceLite rmt(0x2222, 64, 1, 1, 1, UINT32_MAX);
    RmtRmaBufSliceLite notify(0x2222, 64, 1, 1, 1, UINT32_MAX);
    u64 notifyData(1);
    SqeConfigLite cfg;
    ConnLiteOperationOut out;
    u8 data[512];
    out.pi = 0;
    out.data = data;
    UdmaSqeWrite* sqe = (UdmaSqeWrite*)out.data;
    out.dataSize = 64;
    rmt.GetTokenValue();
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);
    ubConn.ci = 0;
    EXPECT_NO_THROW(ubConn.Write(loc, rmt, cfg, stream, out));
    EXPECT_NO_THROW(ubConn.Write(loc, rmt, cfg, stream, out));

    ubConn.ci = 1;
    MOCKER(memset_s).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER(memcpy_s).stubs().with(mockcpp::any()).will(returnValue(0));
    EXPECT_NO_THROW(ubConn.WriteWithNotify(loc, rmt, cfg, out, notify, stream, notifyData));
    EXPECT_EQ(1, ubConn.piDetourCount);
    EXPECT_EQ(0, ubConn.ciDetourCount);

    ubConn.ci = 4;
    ubConn.ciDetourCount = 1;
    EXPECT_NO_THROW(ubConn.WriteWithNotify(loc, rmt, cfg, out, notify, stream, notifyData));
    EXPECT_EQ(2, ubConn.piDetourCount);
}

TEST_F(AicpuUbConnLiteTest, test_UBConnLite_WriteWithNotify_Throw)
{
    UbJettyLiteId id(1, 1, 1);
    u8 sqVa[1024];
    u64 sqVAaddr = reinterpret_cast<u64>(sqVa);
    UbJettyLiteAttr attr(1, sqVAaddr, 3, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;

    UbConnLite ubConn(id, attr, rmtEid);

    EXPECT_EQ(1, ubConn.GetUbJettyLiteId().GetDieId());
    EXPECT_EQ(1, ubConn.GetUbJettyLiteAttr().dbAddr_);
    EXPECT_EQ(1, ubConn.GetRmtEid().raw[0]);

    RmaBufSliceLite loc(0x1111, 64, 1, 1);
    RmtRmaBufSliceLite rmt(0x2222, 64, 1, 1, 1, UINT32_MAX);
    RmtRmaBufSliceLite notify(0x2222, 64, 1, 1, 1, UINT32_MAX);
    u64 notifyData(1);
    SqeConfigLite cfg;
    ConnLiteOperationOut out;
    u8 data[512];
    out.pi = 0;
    out.data = data;
    UdmaSqeWrite* sqe = (UdmaSqeWrite*)out.data;
    out.dataSize = 64;
    rmt.GetTokenValue();
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);
    ubConn.ci = 0;
    MOCKER(memset_s).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER(memcpy_s).stubs().with(mockcpp::any()).will(returnValue(0));
    EXPECT_THROW(ubConn.WriteWithNotify(loc, rmt, cfg, out, notify, stream, notifyData), InternalException);

    ubConn.ci = 2;
    MOCKER(memset_s).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER(memcpy_s).stubs().with(mockcpp::any()).will(returnValue(1));
    EXPECT_THROW(ubConn.WriteWithNotify(loc, rmt, cfg, out, notify, stream, notifyData), InternalException);
    EXPECT_EQ(1, ubConn.piDetourCount);
}

TEST_F(AicpuUbConnLiteTest, test_UBConnLite_WriteWithNotify_Slice)
{
    UbJettyLiteId id(1, 1, 1);
    u8 sqVa[1024];
    u64 sqVAaddr = reinterpret_cast<u64>(sqVa);
    UbJettyLiteAttr attr(1, sqVAaddr, 4096, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;

    UbConnLite ubConn(id, attr, rmtEid);

    EXPECT_EQ(1, ubConn.GetUbJettyLiteId().GetDieId());
    EXPECT_EQ(1, ubConn.GetUbJettyLiteAttr().dbAddr_);
    EXPECT_EQ(1, ubConn.GetRmtEid().raw[0]);

    RmaBufSliceLite loc(0x1111, 257 * 1024 * 1024, 1, 1);
    RmtRmaBufSliceLite rmt(0x2222, 257 * 1024 * 1024, 1, 1, 1, UINT32_MAX);
    RmtRmaBufSliceLite notify(0x2222, 64, 1, 1, 1, UINT32_MAX);
    u64 notifyData(1);
    SqeConfigLite cfg;
    ConnLiteOperationOut out;
    u8 data[512];
    out.pi = 0;
    out.data = data;
    UdmaSqeWrite* sqe = (UdmaSqeWrite*)out.data;
    out.dataSize = 64;
    rmt.GetTokenValue();
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);
    MOCKER(memset_s).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER(memcpy_s).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    EXPECT_NO_THROW(ubConn.WriteWithNotify(loc, rmt, cfg, out, notify, stream, notifyData));
}

TEST_F(AicpuUbConnLiteTest, test_UBConnLite_InlineWrite)
{
    UbJettyLiteId id(1, 1, 1);
    u8 sqVa[1024];
    u64 sqVAaddr = reinterpret_cast<u64>(sqVa);
    UbJettyLiteAttr attr(1, sqVAaddr, 1, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;

    UbConnLite ubConn(id, attr, rmtEid);

    EXPECT_EQ(1, ubConn.GetUbJettyLiteId().GetDieId());
    EXPECT_EQ(1, ubConn.GetUbJettyLiteAttr().dbAddr_);
    EXPECT_EQ(1, ubConn.GetRmtEid().raw[0]);

    RmaBufSliceLite loc(0x1111, 64, 1, 1);
    RmtRmaBufSliceLite rmt(0x2222, 64, 1, 1, 1, UINT32_MAX);
    RmtRmaBufSliceLite notify(0x2222, 64, 1, 1, 1, UINT32_MAX);
    u64 notifyData(1);
    SqeConfigLite cfg;
    ConnLiteOperationOut out;
    u8 data[512];
    out.pi = 0;
    out.data = data;
    UdmaSqeWrite* sqe = (UdmaSqeWrite*)out.data;
    out.dataSize = 64;
    rmt.GetTokenValue();
    u8 write_data(1);
    u16 write_dsize(1);
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    EXPECT_NO_THROW(ubConn.InlineWrite(&write_data, write_dsize, rmt, cfg, stream, out));
}

TEST_F(AicpuUbConnLiteTest, test_UBConnLite_WriteReducee)
{
    UbJettyLiteId id(1, 1, 1);
    u8 sqVa[1024];
    u64 sqVAaddr = reinterpret_cast<u64>(sqVa);
    UbJettyLiteAttr attr(1, sqVAaddr, 1, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;

    UbConnLite ubConn(id, attr, rmtEid);

    EXPECT_EQ(1, ubConn.GetUbJettyLiteId().GetDieId());
    EXPECT_EQ(1, ubConn.GetUbJettyLiteAttr().dbAddr_);
    EXPECT_EQ(1, ubConn.GetRmtEid().raw[0]);

    RmaBufSliceLite loc(0x1111, 64, 1, 1);
    RmtRmaBufSliceLite rmt(0x2222, 64, 1, 1, 1, UINT32_MAX);
    RmtRmaBufSliceLite notify(0x2222, 64, 1, 1, 1, UINT32_MAX);
    u64 notifyData(1);
    SqeConfigLite cfg;
    ConnLiteOperationOut out;
    u8 data[512];
    out.pi = 0;
    out.data = data;
    UdmaSqeWrite* sqe = (UdmaSqeWrite*)out.data;
    out.dataSize = 64;
    rmt.GetTokenValue();
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    EXPECT_NO_THROW(ubConn.WriteReduce(DataType::INT8, ReduceOp::SUM, loc, stream, rmt, cfg, out));
}

TEST_F(AicpuUbConnLiteTest, test_UBConnLite_WriteReduce_Slice)
{
    UbJettyLiteId id(1, 1, 1);
    u8 sqVa[1024];
    u64 sqVAaddr = reinterpret_cast<u64>(sqVa);
    UbJettyLiteAttr attr(1, sqVAaddr, 4, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;

    UbConnLite ubConn(id, attr, rmtEid);

    EXPECT_EQ(1, ubConn.GetUbJettyLiteId().GetDieId());
    EXPECT_EQ(1, ubConn.GetUbJettyLiteAttr().dbAddr_);
    EXPECT_EQ(1, ubConn.GetRmtEid().raw[0]);

    RmaBufSliceLite loc(0x1111, 257 * 1024 * 1024, 1, 1);
    RmtRmaBufSliceLite rmt(0x2222, 257 * 1024 * 1024, 1, 1, 1, UINT32_MAX);
    RmtRmaBufSliceLite notify(0x2222, 64, 1, 1, 1, UINT32_MAX);
    u64 notifyData(1);
    SqeConfigLite cfg;
    ConnLiteOperationOut out;
    u8 data[512];
    out.pi = 0;
    out.data = data;
    UdmaSqeWrite* sqe = (UdmaSqeWrite*)out.data;
    out.dataSize = 64;
    rmt.GetTokenValue();
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    EXPECT_NO_THROW(ubConn.WriteReduce(DataType::INT8, ReduceOp::SUM, loc, stream, rmt, cfg, out));
}

TEST_F(AicpuUbConnLiteTest, test_UBConnLite_BatchOneSidedRead)
{
    UbJettyLiteId id(1, 1, 1);
    u8 sqVa[1024];
    u64 sqVAaddr = reinterpret_cast<u64>(sqVa);
    UbJettyLiteAttr attr(1, sqVAaddr, 4, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;

    UbConnLite ubConn(id, attr, rmtEid);

    u32 dataSize = 400 * 1024 * 1024;
    RmaBufSliceLite loc(0x1111, dataSize, 1, 1);
    RmtRmaBufSliceLite rmt(0x2222, dataSize, 1, 1, 1, UINT32_MAX);
    SqeConfigLite cfg;
    ConnLiteOperationOut out;
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    ubConn.BatchOneSidedRead({loc}, {rmt}, cfg, stream, out);
}

TEST_F(AicpuUbConnLiteTest, test_UBConnLite_BatchOneSidedWrite)
{
    UbJettyLiteId id(1, 1, 1);
    u8 sqVa[1024];
    u64 sqVAaddr = reinterpret_cast<u64>(sqVa);
    UbJettyLiteAttr attr(1, sqVAaddr, 4, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;

    UbConnLite ubConn(id, attr, rmtEid);

    u32 dataSize = 400 * 1024 * 1024;
    RmaBufSliceLite loc(0x1111, dataSize, 1, 1);
    RmtRmaBufSliceLite rmt(0x2222, dataSize, 1, 1, 1, UINT32_MAX);
    SqeConfigLite cfg;
    ConnLiteOperationOut out;
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    ubConn.BatchOneSidedWrite({loc}, {rmt}, cfg, stream, out);
}

/* ---------- GetInflight / UpdateCi / CheckOverflow (反压新增接口) ---------- */

TEST_F(AicpuUbConnLiteTest, GetInflight_NoWrap_ExpectDiff)
{
    UbJettyLiteId id(1, 1, 1);
    UbJettyLiteAttr attr(1, 1, 8, 1, false);
    Eid rmtEid;
    UbConnLite ubConn(id, attr, rmtEid);

    ubConn.pi = 20;
    ubConn.ci = 5;
    EXPECT_EQ(15u, ubConn.GetInflight());
}

TEST_F(AicpuUbConnLiteTest, GetInflight_Wrap_ExpectDiff)
{
    UbJettyLiteId id(1, 1, 1);
    UbJettyLiteAttr attr(1, 1, 8, 1, false);
    Eid rmtEid;
    UbConnLite ubConn(id, attr, rmtEid);

    ubConn.pi = 5;
    ubConn.ci = 65530;
    EXPECT_EQ(11u, ubConn.GetInflight());
}

TEST_F(AicpuUbConnLiteTest, GetInflight_Equal_ExpectZero)
{
    UbJettyLiteId id(1, 1, 1);
    UbJettyLiteAttr attr(1, 1, 8, 1, false);
    Eid rmtEid;
    UbConnLite ubConn(id, attr, rmtEid);

    ubConn.pi = 100;
    ubConn.ci = 100;
    EXPECT_EQ(0u, ubConn.GetInflight());
}

/* ---------- UpdateCi ---------- */

TEST_F(AicpuUbConnLiteTest, UpdateCi_Sequential_ExpectCiAdvanced)
{
    UbJettyLiteId id(1, 1, 1);
    UbJettyLiteAttr attr(1, 1, 8, 1, false);
    Eid rmtEid;
    UbConnLite ubConn(id, attr, rmtEid);

    std::vector<std::pair<u16, u16>> slots = {{0, 10}, {1, 20}};
    ubConn.UpdateCi(slots.data(), slots.size());
    EXPECT_EQ(20u, ubConn.ci);
}

TEST_F(AicpuUbConnLiteTest, UpdateCi_PartialSequence_ExpectLastConsumedCi)
{
    UbJettyLiteId id(1, 1, 1);
    UbJettyLiteAttr attr(1, 1, 8, 1, false);
    Eid rmtEid;
    UbConnLite ubConn(id, attr, rmtEid);

    std::vector<std::pair<u16, u16>> slots = {{1, 20}, {3, 40}};
    ubConn.UpdateCi(slots.data(), slots.size());
    EXPECT_EQ(0u, ubConn.ci);

    std::vector<std::pair<u16, u16>> slots2 = {{0, 10}};
    ubConn.UpdateCi(slots2.data(), slots2.size());
    EXPECT_EQ(20u, ubConn.ci);
}

TEST_F(AicpuUbConnLiteTest, UpdateCi_Wraparound_ExpectCorrectCi)
{
    UbJettyLiteId id(1, 1, 1);
    UbJettyLiteAttr attr(1, 1, 8, 1, false);
    Eid rmtEid;
    UbConnLite ubConn(id, attr, rmtEid);

    std::vector<std::pair<u16, u16>> slots = {{0, 65530}, {1, 65535}};
    ubConn.UpdateCi(slots.data(), slots.size());
    EXPECT_EQ(65535u, ubConn.ci);
}

TEST_F(AicpuUbConnLiteTest, UpdateCi_EmptyInput_ExpectNoChange)
{
    UbJettyLiteId id(1, 1, 1);
    UbJettyLiteAttr attr(1, 1, 8, 1, false);
    Eid rmtEid;
    UbConnLite ubConn(id, attr, rmtEid);

    ubConn.UpdateCi(nullptr, 0);
    EXPECT_EQ(0u, ubConn.ci);
}

TEST_F(AicpuUbConnLiteTest, UpdateCi_DuplicateSeqIdx_ExpectLastValue)
{
    UbJettyLiteId id(1, 1, 1);
    UbJettyLiteAttr attr(1, 1, 8, 1, false);
    Eid rmtEid;
    UbConnLite ubConn(id, attr, rmtEid);

    std::vector<std::pair<u16, u16>> slots = {{0, 10}, {0, 99}};
    ubConn.UpdateCi(slots.data(), slots.size());
    EXPECT_EQ(10u, ubConn.ci);
}

/* ---------- CheckOverflow (cache version: u32) ---------- */

TEST_F(AicpuUbConnLiteTest, CheckOverflow_Cache_NoOverflow_ExpectFalse)
{
    UbJettyLiteId id(1, 1, 1);
    UbJettyLiteAttr attr(1, 1, 8, 1, false);
    Eid rmtEid;
    UbConnLite ubConn(id, attr, rmtEid);

    ubConn.sqDepth_ = 8;
    ubConn.pi = 5;
    ubConn.ci = 2;
    EXPECT_FALSE(ubConn.CheckOverflow(static_cast<u32>(3)));
}

TEST_F(AicpuUbConnLiteTest, CheckOverflow_Cache_Overflow_ExpectTrue)
{
    UbJettyLiteId id(1, 1, 1);
    UbJettyLiteAttr attr(1, 1, 8, 1, false);
    Eid rmtEid;
    UbConnLite ubConn(id, attr, rmtEid);

    ubConn.sqDepth_ = 8;
    ubConn.pi = 5;
    ubConn.ci = 2;
    EXPECT_TRUE(ubConn.CheckOverflow(static_cast<u32>(6)));
}

/* ---------- CheckOverflow (transport version: u64, bool, bool) ---------- */

TEST_F(AicpuUbConnLiteTest, CheckOverflow_Transport_Read_NoOverflow_ExpectFalse)
{
    UbJettyLiteId id(1, 1, 1);
    UbJettyLiteAttr attr(1, 1, 8, 1, false);
    Eid rmtEid;
    UbConnLite ubConn(id, attr, rmtEid);

    ubConn.sqDepth_ = 8;
    ubConn.pi = 5;
    ubConn.ci = 2;
    ubConn.maxReadSize = 1024;
    ubConn.maxWriteSize = 1024;
    EXPECT_FALSE(ubConn.CheckOverflow(static_cast<u64>(2048), true, false));
}

TEST_F(AicpuUbConnLiteTest, CheckOverflow_Transport_WriteWithNotify_Overflow_ExpectTrue)
{
    UbJettyLiteId id(1, 1, 1);
    UbJettyLiteAttr attr(1, 1, 8, 1, false);
    Eid rmtEid;
    UbConnLite ubConn(id, attr, rmtEid);

    ubConn.sqDepth_ = 8;
    ubConn.pi = 5;
    ubConn.ci = 2;
    ubConn.maxReadSize = 1024;
    ubConn.maxWriteSize = 1024;
    EXPECT_TRUE(ubConn.CheckOverflow(static_cast<u64>(8192), false, true));
}

/* ---------- CalcWqeCount ---------- */

TEST_F(AicpuUbConnLiteTest, CalcWqeCount_Read_ExactMultiple_ExpectNoRounding)
{
    UbJettyLiteId id(1, 1, 1);
    UbJettyLiteAttr attr(1, 1, 8, 1, false);
    Eid rmtEid;
    UbConnLite ubConn(id, attr, rmtEid);

    ubConn.maxReadSize = 1024;
    ubConn.maxWriteSize = 1024;
    EXPECT_EQ(2u, ubConn.CalcWqeCount(2048, true, false));
}

TEST_F(AicpuUbConnLiteTest, CalcWqeCount_Read_PartialSlice_ExpectCeil)
{
    UbJettyLiteId id(1, 1, 1);
    UbJettyLiteAttr attr(1, 1, 8, 1, false);
    Eid rmtEid;
    UbConnLite ubConn(id, attr, rmtEid);

    ubConn.maxReadSize = 1024;
    ubConn.maxWriteSize = 1024;
    EXPECT_EQ(3u, ubConn.CalcWqeCount(2049, true, false));
}

TEST_F(AicpuUbConnLiteTest, CalcWqeCount_WriteWithNotify_ExpectExtraOne)
{
    UbJettyLiteId id(1, 1, 1);
    UbJettyLiteAttr attr(1, 1, 8, 1, false);
    Eid rmtEid;
    UbConnLite ubConn(id, attr, rmtEid);

    ubConn.maxReadSize = 1024;
    ubConn.maxWriteSize = 1024;
    EXPECT_EQ(3u, ubConn.CalcWqeCount(2048, false, true));
}
