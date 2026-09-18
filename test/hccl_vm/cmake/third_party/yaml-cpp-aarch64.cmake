# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 交叉编译 aarch64 版本的 yaml-cpp 静态库, 供 device_arm (bin/device) 链接使用。
# 复用 host 版 (yaml-cpp.cmake) 的同一份源码包 third_party/yaml-cpp-0.8.0.tar.gz,
# 但使用 CANN HCC 工具链 (cmake/toolchains/aarch64-cann-hcc.cmake) 交叉编译,
# 产物安装到 third_party/yaml-cpp-aarch64/, 与 host 版 (third_party/yaml-cpp/) 物理隔离。
#
# 调用前提: 顶层 CMakeLists.txt 的 BUILD_DEVICE_ARM 分支已校验 CANN_HCC_ROOT 并可定位 toolchain file。

include_guard(GLOBAL)

# 装载点与源码包 (与 host 版 yaml-cpp.cmake 保持同一份 tarball, 离线可用)
set(YAMLCPP_AARCH64_FILE "yaml-cpp-0.8.0.tar.gz")
set(YAMLCPP_AARCH64_PKG_PATH ${CMAKE_SOURCE_DIR}/third_party/${YAMLCPP_AARCH64_FILE})
set(YAMLCPP_AARCH64_INSTALL_PATH ${CMAKE_SOURCE_DIR}/third_party/yaml-cpp-aarch64)
set(YAMLCPP_AARCH64_INCLUDE_DIR ${YAMLCPP_AARCH64_INSTALL_PATH}/include)

# 优先使用 lib64, 兼容 lib
set(YAMLCPP_AARCH64_LIB_DIR "")
if(EXISTS "${YAMLCPP_AARCH64_INSTALL_PATH}/lib64/libyaml-cpp.a")
    set(YAMLCPP_AARCH64_LIB_DIR "${YAMLCPP_AARCH64_INSTALL_PATH}/lib64")
elseif(EXISTS "${YAMLCPP_AARCH64_INSTALL_PATH}/lib/libyaml-cpp.a")
    set(YAMLCPP_AARCH64_LIB_DIR "${YAMLCPP_AARCH64_INSTALL_PATH}/lib")
endif()

message(STATUS "[ThirdParty][aarch64] YAMLCPP_AARCH64_INSTALL_PATH=${YAMLCPP_AARCH64_INSTALL_PATH}")
message(STATUS "[ThirdParty][aarch64] YAMLCPP_AARCH64_LIB_DIR=${YAMLCPP_AARCH64_LIB_DIR}")

# 增量缓存: 头文件 + 静态库均已存在则跳过交叉编译
if(YAMLCPP_AARCH64_LIB_DIR AND EXISTS "${YAMLCPP_AARCH64_INCLUDE_DIR}/yaml-cpp/yaml.h")
    message(STATUS "[ThirdParty][aarch64] yaml-cpp (aarch64) found in ${YAMLCPP_AARCH64_INSTALL_PATH}")
else()
    # 与 host 版 yaml-cpp.cmake 对齐: 本地 tarball 缺失时回退在线下载 (下载在 build 阶段执行)
    if(EXISTS ${YAMLCPP_AARCH64_PKG_PATH})
        message(STATUS "[ThirdParty][aarch64] Found local yaml-cpp package: ${YAMLCPP_AARCH64_PKG_PATH}")
        set(YAMLCPP_AARCH64_PROJECT_URL ${YAMLCPP_AARCH64_PKG_PATH})
    else()
        if(NOT DEFINED YAMLCPP_URL)
            set(YAMLCPP_URL "https://raw.gitcode.com/src-openeuler/yaml-cpp/blobs/d1ead4fff417073b9cdbf98b8b55eb0efc00b0ba/yaml-cpp-0.8.0.tar.gz")
        endif()
        message(STATUS "[ThirdParty][aarch64] Downloading yaml-cpp from ${YAMLCPP_URL}")
        set(YAMLCPP_AARCH64_PROJECT_URL ${YAMLCPP_URL})
    endif()
    message(STATUS "[ThirdParty][aarch64] yaml-cpp (aarch64) not found, will cross-build from source")

    # 编译选项与 host 版对齐 (_GLIBCXX_USE_CXX11_ABI=0 必须, 与 device CMakeLists 一致)
    set(YAMLCPP_AARCH64_CXXFLAGS "-D_GLIBCXX_USE_CXX11_ABI=0 -O2 -D_FORTIFY_SOURCE=2 -fPIC -fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack")
    set(YAMLCPP_AARCH64_CFLAGS   "-D_GLIBCXX_USE_CXX11_ABI=0 -O2 -D_FORTIFY_SOURCE=2 -fPIC -fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack")

    set(YAMLCPP_AARCH64_TOOLCHAIN_FILE ${CMAKE_SOURCE_DIR}/cmake/toolchains/aarch64-cann-hcc.cmake)

    set(YAMLCPP_AARCH64_OPTS
        -DCMAKE_TOOLCHAIN_FILE=${YAMLCPP_AARCH64_TOOLCHAIN_FILE}
        -DCANN_HCC_ROOT=${CANN_HCC_ROOT}
        -DCMAKE_CXX_FLAGS=${YAMLCPP_AARCH64_CXXFLAGS}
        -DCMAKE_C_FLAGS=${YAMLCPP_AARCH64_CFLAGS}
        -DCMAKE_INSTALL_PREFIX=${YAMLCPP_AARCH64_INSTALL_PATH}
        -DCMAKE_INSTALL_LIBDIR=lib64
        -DCMAKE_CXX_STANDARD=17
        -DYAML_CPP_BUILD_TESTS=OFF
        -DYAML_CPP_BUILD_TOOLS=OFF
        -DYAML_CPP_BUILD_CONTRIB=OFF
        -DBUILD_SHARED_LIBS=OFF
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    )

    # DOWNLOAD_EXTRACT_TIMESTAMP 为 CMake 3.24+ 引入的选项(配套策略 CMP0135),
    # 旧版本(如3.22)不识别该关键字, 会将其与TRUE误解析为URL的值, 导致报错:
    # "At least one entry of URL is a path (invalid in a list)", 故仅在高版本传入。
    # (新旧版本的默认行为一致: 提取时更新源码时间戳)
    set(YAMLCPP_AARCH64_DOWNLOAD_OPTS "")
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        list(APPEND YAMLCPP_AARCH64_DOWNLOAD_OPTS DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    endif()

    include(ExternalProject)
    ExternalProject_Add(third_party_yaml_cpp_aarch64
        URL ${YAMLCPP_AARCH64_PROJECT_URL}
        ${YAMLCPP_AARCH64_DOWNLOAD_OPTS}
        DOWNLOAD_DIR ${CMAKE_SOURCE_DIR}/third_party
        DOWNLOAD_NO_PROGRESS TRUE
        CONFIGURE_COMMAND ${CMAKE_COMMAND} ${YAMLCPP_AARCH64_OPTS} <SOURCE_DIR>
        BUILD_COMMAND ${CMAKE_MAKE_PROGRAM}
        INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install
        EXCLUDE_FROM_ALL TRUE
    )
    set(YAMLCPP_AARCH64_LIB_DIR "${YAMLCPP_AARCH64_INSTALL_PATH}/lib64")
endif()

if(NOT EXISTS ${YAMLCPP_AARCH64_INCLUDE_DIR})
    file(MAKE_DIRECTORY "${YAMLCPP_AARCH64_INCLUDE_DIR}")
endif()

# 暴露给 device_arm 子工程使用的安装路径 (经顶层 CMakeLists 的 CMAKE_ARGS 透传)
set(YAMLCPP_AARCH64_INSTALL_PATH ${YAMLCPP_AARCH64_INSTALL_PATH} CACHE INTERNAL "aarch64 yaml-cpp install path")
set(YAMLCPP_AARCH64_INCLUDE_DIR ${YAMLCPP_AARCH64_INCLUDE_DIR} CACHE INTERNAL "aarch64 yaml-cpp include dir")
set(YAMLCPP_AARCH64_LIB_DIR ${YAMLCPP_AARCH64_LIB_DIR} CACHE INTERNAL "aarch64 yaml-cpp lib dir")
