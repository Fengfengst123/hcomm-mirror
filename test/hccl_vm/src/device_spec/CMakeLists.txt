# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# device_spec: device 规格的唯一权威解析与地址计算库（设计文档 §4.3.1 / D2）。
# 纯计算、无 IO 副作用（除加载 yaml），故 hccl-vm 主进程 / proxy / runner / checker 均可安全链接。
add_library(device_spec STATIC "")

target_sources(device_spec PRIVATE
    device_spec_types.cc
    device_spec_yaml_loader.cc
    device_spec_registry.cc
    rank_address_space.cc
    ccu_address_space.cc
)

# 头文件对外暴露（include/device_spec/*.h），消费方用 device_spec/xxx.h 引用
target_include_directories(device_spec PUBLIC
    ${HCCL_VM_INCLUDE}
)

target_include_directories(device_spec PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HCCL_VM_SRC}
    ${THRID_PARTY_DIR}
    ${THRID_PARTY_DIR}/yaml-cpp/include
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/include
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime
)

hcclvm_apply_common_flags(device_spec)

# yaml-cpp 用于解析 device yaml；HAVE_YAML_CPP 由 log 库 PUBLIC 传递，此处显式再声明一次以防
# 消费方未链接 log 时宏丢失。
target_compile_definitions(device_spec PUBLIC HAVE_YAML_CPP)

target_link_libraries(device_spec PUBLIC log)
target_link_libraries(device_spec PRIVATE yaml-cpp::yaml-cpp)

if(TARGET third_party_yaml_cpp)
    add_dependencies(device_spec third_party_yaml_cpp)
endif()

# 日志染色: 模块 tag
target_compile_definitions(device_spec PRIVATE HCCL_VM_MODULE="DEVSPEC")
