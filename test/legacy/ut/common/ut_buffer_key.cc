/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <stdint.h>
#include <limits>
#include "buffer_key.h"

using U64Key = hcomm::BufferKey<uint64_t, uint64_t>;
using U32Key = hcomm::BufferKey<uint32_t, uint32_t>;

static const uint64_t U64_MAX = std::numeric_limits<uint64_t>::max();
static const uint32_t U32_MAX = std::numeric_limits<uint32_t>::max();

class BufferKeyTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "BufferKeyTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "BufferKeyTest TearDown" << std::endl; }

    virtual void SetUp() {}

    virtual void TearDown() {}
};

/* ---------------- 基本访问接口 ---------------- */

TEST_F(BufferKeyTest, Ut_Constructor_When_Normal_Expect_AddrSizeAccessible)
{
    U64Key key(100, 200);
    EXPECT_EQ(key.Addr(), static_cast<uint64_t>(100));
    EXPECT_EQ(key.Size(), static_cast<uint64_t>(200));
}

TEST_F(BufferKeyTest, Ut_Equal_When_SameAddrAndSize_Expect_Equal)
{
    U64Key key(100, 200);
    U64Key same(100, 200);
    U64Key diffAddr(101, 200);
    U64Key diffSize(100, 201);
    EXPECT_TRUE(key == same);
    EXPECT_FALSE(key != same);
    EXPECT_TRUE(key != diffAddr);
    EXPECT_TRUE(key != diffSize);
}

/* ---------------- IsSubset ---------------- */

TEST_F(BufferKeyTest, Ut_IsSubset_When_FullyContained_Expect_True)
{
    U64Key outer(0, 100);
    U64Key inner(10, 20); // [10, 30) 完全位于 [0, 100) 内
    EXPECT_TRUE(inner.IsSubset(outer));
}

TEST_F(BufferKeyTest, Ut_IsSubset_When_EndAlignedWithOtherEnd_Expect_True)
{
    U64Key outer(0, 100);
    U64Key inner(10, 90); // [10, 100) 结束地址与外层重合
    EXPECT_TRUE(inner.IsSubset(outer));
}

TEST_F(BufferKeyTest, Ut_IsSubset_When_PrefixOfOther_Expect_True)
{
    U64Key outer(10, 100);
    U64Key inner(10, 50); // 起始地址相同，size 更小
    EXPECT_TRUE(inner.IsSubset(outer));
}

TEST_F(BufferKeyTest, Ut_IsSubset_When_SameKey_Expect_False)
{
    U64Key key(10, 20); // 不含等于情况
    EXPECT_FALSE(key.IsSubset(key));
}

TEST_F(BufferKeyTest, Ut_IsSubset_When_StartBeforeOther_Expect_False)
{
    U64Key outer(10, 100);
    U64Key inner(0, 20); // 起始地址小于外层
    EXPECT_FALSE(inner.IsSubset(outer));
}

TEST_F(BufferKeyTest, Ut_IsSubset_When_EndBeyondOther_Expect_False)
{
    U64Key outer(0, 10);
    U64Key inner(5, 10); // [5, 15) 结束地址超出 [0, 10)
    EXPECT_FALSE(inner.IsSubset(outer));
}

TEST_F(BufferKeyTest, Ut_IsSubset_When_SizeLargerThanOther_Expect_False)
{
    U64Key outer(10, 10);
    U64Key inner(10, 20); // 先校验 size：size 大于外层必不可能为子集
    EXPECT_FALSE(inner.IsSubset(outer));
}

TEST_F(BufferKeyTest, Ut_IsSubset_When_HighAddrContainedWithoutWrap_Expect_True)
{
    // 高地址区间的正常包含关系不受防溢出改写影响
    U64Key outer(U64_MAX - 0x50, 0x40);
    U64Key inner(U64_MAX - 0x40, 0x20);
    EXPECT_TRUE(inner.IsSubset(outer));
}

TEST_F(BufferKeyTest, Ut_IsSubset_When_AddrPlusSizeWraparound_Expect_False)
{
    // this 的 addr + size 回绕（数学结束地址越过类型上限），不可能位于低地址区间内。
    // 旧实现 addr_ + size_ 回绕后变小，会误判为 true。
    U64Key wrapped(U64_MAX - 15, 0x20);
    U64Key low(0, 0x100);
    EXPECT_FALSE(wrapped.IsSubset(low));
}

TEST_F(BufferKeyTest, Ut_IsSubset_When_OtherAddrPlusSizeWraparound_Expect_False)
{
    // other 的 addr + size 回绕，低地址区间同样不可能为其子集
    U64Key wrapped(U64_MAX - 15, 0x20);
    U64Key low(0, 0x100);
    EXPECT_FALSE(low.IsSubset(wrapped));
}

/* ---------------- IsSuperset ---------------- */

TEST_F(BufferKeyTest, Ut_IsSuperset_When_FullyContains_Expect_True)
{
    U64Key outer(0, 100);
    U64Key inner(10, 20);
    EXPECT_TRUE(outer.IsSuperset(inner));
}

TEST_F(BufferKeyTest, Ut_IsSuperset_When_PrefixContainOther_Expect_True)
{
    U64Key outer(10, 100);
    U64Key inner(10, 50); // 起始地址相同，size 更大
    EXPECT_TRUE(outer.IsSuperset(inner));
}

TEST_F(BufferKeyTest, Ut_IsSuperset_When_SameKey_Expect_False)
{
    U64Key key(10, 20); // 不含等于情况
    EXPECT_FALSE(key.IsSuperset(key));
}

TEST_F(BufferKeyTest, Ut_IsSuperset_When_StartAfterOther_Expect_False)
{
    U64Key outer(10, 100);
    U64Key inner(0, 20); // 起始地址大于内层
    EXPECT_FALSE(outer.IsSuperset(inner));
}

TEST_F(BufferKeyTest, Ut_IsSuperset_When_EndBeforeOther_Expect_False)
{
    U64Key outer(0, 50);
    U64Key inner(40, 20); // [40, 60) 结束地址超出 [0, 50)
    EXPECT_FALSE(outer.IsSuperset(inner));
}

TEST_F(BufferKeyTest, Ut_IsSuperset_When_SizeSmallerThanOther_Expect_False)
{
    U64Key outer(0, 10);
    U64Key inner(5, 20); // 先校验 size：size 小于内层必不可能为超集
    EXPECT_FALSE(outer.IsSuperset(inner));
}

TEST_F(BufferKeyTest, Ut_IsSuperset_When_HighAddrContainsWithoutWrap_Expect_True)
{
    U64Key outer(U64_MAX - 0x50, 0x40);
    U64Key inner(U64_MAX - 0x40, 0x20);
    EXPECT_TRUE(outer.IsSuperset(inner));
}

TEST_F(BufferKeyTest, Ut_IsSuperset_When_AddrPlusSizeWraparound_Expect_False)
{
    // other 的 addr + size 回绕后变小，低地址区间不可能包含高地址区间。
    // 旧实现 other.addr_ + other.size_ 回绕后变小，会误判为 true。
    U64Key low(0, 0x100);
    U64Key wrapped(U64_MAX - 15, 0x20);
    EXPECT_FALSE(low.IsSuperset(wrapped));
}

TEST_F(BufferKeyTest, Ut_IsSuperset_When_OtherAddrPlusSizeWraparound_Expect_False)
{
    U64Key low(0, 0x100);
    U64Key wrapped(U64_MAX - 15, 0x20);
    EXPECT_FALSE(wrapped.IsSuperset(low));
}

/* ---------------- IsIntersect ---------------- */

TEST_F(BufferKeyTest, Ut_IsIntersect_When_PartialOverlap_Expect_True)
{
    U64Key left(0, 10);
    U64Key right(5, 10); // [5, 15) 与 [0, 10) 重叠
    EXPECT_TRUE(left.IsIntersect(right));
    EXPECT_TRUE(right.IsIntersect(left));
}

TEST_F(BufferKeyTest, Ut_IsIntersect_When_ContainsOther_Expect_True)
{
    U64Key outer(0, 100);
    U64Key inner(10, 20);
    EXPECT_TRUE(outer.IsIntersect(inner));
    EXPECT_TRUE(inner.IsIntersect(outer));
}

TEST_F(BufferKeyTest, Ut_IsIntersect_When_SameKey_Expect_True)
{
    U64Key key(5, 10);
    EXPECT_TRUE(key.IsIntersect(key));
}

TEST_F(BufferKeyTest, Ut_IsIntersect_When_NoOverlap_Expect_False)
{
    U64Key left(0, 10);
    U64Key right(20, 10);
    EXPECT_FALSE(left.IsIntersect(right));
    EXPECT_FALSE(right.IsIntersect(left));
}

TEST_F(BufferKeyTest, Ut_IsIntersect_When_EndTouchOtherStart_Expect_False)
{
    // 半开区间端点相接：[0, 10) 与 [10, 15) 不相交
    U64Key left(0, 10);
    U64Key right(10, 5);
    EXPECT_FALSE(left.IsIntersect(right));
    EXPECT_FALSE(right.IsIntersect(left));
}

TEST_F(BufferKeyTest, Ut_IsIntersect_When_HighAddrOverlapWraparound_Expect_True)
{
    // this 的 addr + size 回绕，但与高位不回绕区间存在实际重叠。
    // 旧实现 addr_ + size_ 回绕后变小，会漏判为 false。
    U64Key wrapped(U64_MAX - 15, 0x20); // [MAX-15, MAX+17)，与 [MAX-20, MAX-10) 重叠
    U64Key high(U64_MAX - 20, 0x10);
    EXPECT_TRUE(wrapped.IsIntersect(high));
    EXPECT_TRUE(high.IsIntersect(wrapped));
}

TEST_F(BufferKeyTest, Ut_IsIntersect_When_HighAddrSameStartDiffSize_Expect_True)
{
    // 同一起始地址、size 不同：较短区间包含于较长区间（长区间跨回绕）
    U64Key wrapped(U64_MAX - 15, 0x20);
    U64Key high(U64_MAX - 15, 0xF);
    EXPECT_TRUE(wrapped.IsIntersect(high));
    EXPECT_TRUE(high.IsIntersect(wrapped));
}

TEST_F(BufferKeyTest, Ut_IsIntersect_When_HighVsLow_Expect_False)
{
    U64Key wrapped(U64_MAX - 15, 0x20);
    U64Key low(0, 0x100);
    EXPECT_FALSE(wrapped.IsIntersect(low));
    EXPECT_FALSE(low.IsIntersect(wrapped));
}

/* ---------------- IsDisjoint ---------------- */

TEST_F(BufferKeyTest, Ut_IsDisjoint_When_SelfBeforeOther_Expect_True)
{
    U64Key left(0, 10); // [0, 10) 完全在 [10, 15) 之前
    U64Key right(10, 5);
    EXPECT_TRUE(left.IsDisjoint(right));
}

TEST_F(BufferKeyTest, Ut_IsDisjoint_When_SelfAfterOther_Expect_True)
{
    U64Key left(0, 10);
    U64Key right(20, 5); // [20, 25) 完全在 [0, 10) 之后
    EXPECT_TRUE(right.IsDisjoint(left));
}

TEST_F(BufferKeyTest, Ut_IsDisjoint_When_EndTouchOtherStart_Expect_True)
{
    // 端点相接视为不相交
    U64Key left(0, 10);
    U64Key right(10, 5);
    EXPECT_TRUE(left.IsDisjoint(right));
    EXPECT_TRUE(right.IsDisjoint(left));
}

TEST_F(BufferKeyTest, Ut_IsDisjoint_When_Overlap_Expect_False)
{
    U64Key left(0, 10);
    U64Key right(5, 10);
    EXPECT_FALSE(left.IsDisjoint(right));
    EXPECT_FALSE(right.IsDisjoint(left));
}

TEST_F(BufferKeyTest, Ut_IsDisjoint_When_HighAddrOverlapWraparound_Expect_False)
{
    // this 的 addr + size 回绕，与高位不回绕区间存在实际重叠，应判定为相交。
    // 旧实现 addr_ + size_ 回绕后变小，<= other.addr_ 误判为 true。
    U64Key wrapped(U64_MAX - 15, 0x20);
    U64Key high(U64_MAX - 20, 0x10);
    EXPECT_FALSE(wrapped.IsDisjoint(high));
    EXPECT_FALSE(high.IsDisjoint(wrapped));
}

TEST_F(BufferKeyTest, Ut_IsDisjoint_When_HighVsLow_Expect_True)
{
    U64Key wrapped(U64_MAX - 15, 0x20);
    U64Key low(0, 0x100);
    EXPECT_TRUE(wrapped.IsDisjoint(low));
    EXPECT_TRUE(low.IsDisjoint(wrapped));
}

/* ---------------- 32 位实例化 ---------------- */

TEST_F(BufferKeyTest, Ut_U32_Relation_When_Normal_Expect_Correct)
{
    U32Key outer(0, 100);
    U32Key inner(10, 20);
    EXPECT_TRUE(inner.IsSubset(outer));
    EXPECT_TRUE(outer.IsSuperset(inner));
    EXPECT_TRUE(inner.IsIntersect(outer));

    U32Key left(0, 10);
    U32Key right(10, 5);
    EXPECT_FALSE(left.IsIntersect(right));
    EXPECT_TRUE(left.IsDisjoint(right));
}

TEST_F(BufferKeyTest, Ut_U32_Relation_When_AddrPlusSizeWraparound_Expect_Correct)
{
    // 32 位类型同样存在 addr + size 回绕场景
    U32Key wrapped(U32_MAX - 9, 0x20);
    U32Key low(0, 0x100);
    EXPECT_FALSE(wrapped.IsSubset(low));
    EXPECT_FALSE(low.IsSuperset(wrapped));
    EXPECT_FALSE(wrapped.IsIntersect(low));
    EXPECT_TRUE(wrapped.IsDisjoint(low));
}
