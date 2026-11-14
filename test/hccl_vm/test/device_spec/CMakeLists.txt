# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -D_GLIBCXX_USE_CXX11_ABI=0")

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/device_spec
    ${CMAKE_SOURCE_DIR}/third_party/yaml-cpp/include
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime
)

link_libraries(
    gtest_main
    gtest
)

# 把 device 配置拷到测试可执行文件同级的 config/device/，与安装布局一致，
# 使 UT 可用相对路径（或 InstallPath 向上查找）定位。
file(COPY "${CMAKE_SOURCE_DIR}/asset/cluster_model/config/device"
     DESTINATION "${CMAKE_BINARY_DIR}/output/bin/config")
file(COPY "${CMAKE_SOURCE_DIR}/asset/cluster_model/config/cluster"
     DESTINATION "${CMAKE_BINARY_DIR}/output/bin/config")
file(COPY "${CMAKE_SOURCE_DIR}/asset/cluster_model/config/server_or_pod"
     DESTINATION "${CMAKE_BINARY_DIR}/output/bin/config")
file(COPY "${CMAKE_SOURCE_DIR}/config/log_config.yaml"
     DESTINATION "${CMAKE_BINARY_DIR}/output/bin/config")

add_executable(test_device_spec
    device_spec_loader_test.cc
    rank_address_space_test.cc
    ccu_address_space_test.cc
    device_spec_registry_test.cc
)
target_link_libraries(test_device_spec PRIVATE pthread device_spec log yaml-cpp::yaml-cpp)
target_compile_definitions(test_device_spec PRIVATE
    HCCLVM_DEVICE_SPEC_TEST_DIR="${CMAKE_SOURCE_DIR}/asset/cluster_model/config/device"
    HCCLVM_DEVICE_CONFIG_ROOT="${CMAKE_SOURCE_DIR}/asset/cluster_model/config"
)
set_target_properties(test_device_spec PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

if(TARGET third_party_yaml_cpp)
    add_dependencies(test_device_spec third_party_yaml_cpp)
endif()

add_test(NAME test_device_spec COMMAND test_device_spec)
