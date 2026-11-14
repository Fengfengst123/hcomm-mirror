# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(src_to_test_list
    main.cc
    ut_root_group_strategy_grouping_utest.cc
    ut_rank_info_detect_scalable_utest.cc
)

add_executable(hccl_utest_rank_info_detect_scalable ${src_to_test_list})

target_include_directories(hccl_utest_rank_info_detect_scalable PRIVATE
    ${UT_COMMON_INCLUDE_LIST}
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_info_detect
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/rank_table_info
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/common
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/common
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/common/types
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/entrance
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/resource_manager/socket
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/socket
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/external_system
)

target_compile_definitions(hccl_utest_rank_info_detect_scalable PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
    google=ascend_private
    -DOPEN_HCCL_TEST
    OPEN_BUILD_PROJECT
)

target_compile_options(hccl_utest_rank_info_detect_scalable PRIVATE
    -U_FORTIFY_SOURCE
    -fno-var-tracking-assignments
    -Os
    -O0 -g --coverage -fprofile-arcs -ftest-coverage -fno-access-control
)

target_link_libraries(hccl_utest_rank_info_detect_scalable PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    $<BUILD_INTERFACE:rdma_core_headers>
    -pthread
    hccl_llt
    -lgcov
)

add_test(NAME rank_info_detect_scalable_utest COMMAND hccl_utest_rank_info_detect_scalable)
