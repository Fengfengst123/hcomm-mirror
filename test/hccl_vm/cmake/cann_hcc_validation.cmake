# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

include_guard(GLOBAL)

function(hcclvm_validate_cann_hcc hcc_root result_var missing_path_var)
    set(_cann_hcc_target aarch64-target-linux-gnu)
    set(_cann_hcc_bin "${hcc_root}/bin")
    set(_cann_hcc_sysroot "${hcc_root}/sysroot")
    set(_cann_hcc_required_paths
        "${_cann_hcc_bin}/${_cann_hcc_target}-gcc"
        "${_cann_hcc_bin}/${_cann_hcc_target}-g++"
        "${_cann_hcc_bin}/${_cann_hcc_target}-ar"
        "${_cann_hcc_bin}/${_cann_hcc_target}-ranlib"
        "${_cann_hcc_bin}/${_cann_hcc_target}-strip"
        "${_cann_hcc_bin}/${_cann_hcc_target}-ld"
        "${_cann_hcc_bin}/${_cann_hcc_target}-nm"
        "${_cann_hcc_bin}/${_cann_hcc_target}-objcopy"
        "${_cann_hcc_sysroot}/lib64/ld-linux-aarch64.so.1"
        "${_cann_hcc_sysroot}/lib64/libc.so.6"
        "${_cann_hcc_sysroot}/usr/lib64/libstdc++.so.6"
        "${_cann_hcc_sysroot}/usr/lib64/libgcc_s.so.1"
    )

    set(_cann_hcc_is_complete TRUE)
    set(_cann_hcc_missing_path "")
    foreach(_required_path IN LISTS _cann_hcc_required_paths)
        if(NOT EXISTS "${_required_path}")
            set(_cann_hcc_is_complete FALSE)
            set(_cann_hcc_missing_path "${_required_path}")
            break()
        endif()
    endforeach()

    set(${result_var} "${_cann_hcc_is_complete}" PARENT_SCOPE)
    set(${missing_path_var} "${_cann_hcc_missing_path}" PARENT_SCOPE)
endfunction()
