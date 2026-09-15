# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(hccl_ut_base_comm_orion_adpt_utils_list
    main.cc
    ut_orion_adpt_utils.cc
)

add_executable(hccl_ut_base_comm_orion_adpt_utils ${hccl_ut_base_comm_orion_adpt_utils_list})

target_include_directories(hccl_ut_base_comm_orion_adpt_utils PRIVATE
    ${UT_COMMON_INCLUDE_LIST}
    # dlog 桩头文件(log.h 依赖 <dlog_pub.h>)
    ${HCCL_CODE_ROOT}/test/stub/depends/pkg_inc/base
    ${HCCL_CODE_ROOT}/src/base_comm/common
    ${HCCL_CODE_ROOT}/src/base_comm/primitives/api_c_adpt
    # orion_adpt_utils.h 传递依赖 hcomm_adapter_hccp.h (HrtUbJfcMode)
    ${HCCL_CODE_ROOT}/src/base_comm/resources/southbound_adpt
)

target_compile_definitions(hccl_ut_base_comm_orion_adpt_utils PRIVATE
    HCOMM_CODE_ROOT="${HCCL_CODE_ROOT}"
)

target_compile_options(hccl_ut_base_comm_orion_adpt_utils PRIVATE
    -U_FORTIFY_SOURCE
    -O0 -g --coverage -fprofile-arcs -ftest-coverage -fno-access-control
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>
)

target_link_libraries(hccl_ut_base_comm_orion_adpt_utils PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    $<BUILD_INTERFACE:rdma_core_headers>
    -pthread
    hccl_llt
    -Wl,--whole-archive
    -Wl,--no-whole-archive
    -lgcov
    -lrt
    -ldl
)

add_dependencies(hccl_ut_base_comm_orion_adpt_utils hccl_llt)

enable_testing()
add_test(NAME hccl_ut_base_comm_orion_adpt_utils COMMAND hccl_ut_base_comm_orion_adpt_utils)
