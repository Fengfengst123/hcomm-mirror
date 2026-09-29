# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

find_cann_package(securec MODULE REQUIRED)

set(NETCO_IDL ${HCCL_CODE_ROOT}/src/base_comm/resources/hccp/netco/idl)

# Compile the real registration, dispatch, timer and storage code as C. Unused DC
# slice/job paths are discarded, so this test needs neither a device nor a network.
add_executable(netco_callback_utest
    tc_netco_callbacks.c
    ${NETCO_IDL}/src/data_sync/dc/bkf_dc.c
    ${NETCO_IDL}/src/data_sync/dc/bkf_dc_table_type.c
    ${NETCO_IDL}/src/libbackfin/system_base/bkf_sys_log.c
    ${NETCO_IDL}/src/libbackfin/system_base/bkf_tmr.c
    ${NETCO_IDL}/src/libbackfin/debug_tools/bkf_disp.c
    ${NETCO_IDL}/src/libbackfin/debug_tools/bkf_log.c
    ${NETCO_IDL}/src/libbackfin/utils/bkf_mem.c
    ${NETCO_IDL}/src/libbackfin/utils/bkf_str.c
    ${NETCO_IDL}/src/libbackfin/utils/bkf_comm.c
    ${NETCO_IDL}/src/libbackfin/utils/bkf_bas_type_mthd.c
    ${NETCO_IDL}/src/libbackfin/utils/bkf_bufr.c
    ${NETCO_IDL}/src/bdep/src/v_stringlib.c
    ${NETCO_IDL}/src/bdep/src/vos_systime.c
    ${NETCO_IDL}/src/bdep/avl_dopra/src/v_avl3/src/v_avl3.c
    ${NETCO_IDL}/src/bdep/avl_dopra/src/v_avlbase/src/v_avlbase.c
    ${NETCO_IDL}/src/bdep/avl_dopra/src/v_avlbase/src/v_avlpub.c
)
set_property(TARGET netco_callback_utest PROPERTY C_STANDARD 11)
target_include_directories(netco_callback_utest PRIVATE
    ${NETCO_IDL}/src/data_sync/dc
    ${NETCO_IDL}/src/libbackfin/include
    ${NETCO_IDL}/src/bdep/include
    ${NETCO_IDL}/src/bdep/include/avl_dopra
    ${NETCO_IDL}/src/bdep/avl_dopra/src/v_avlbase/include
    ${NETCO_IDL}/interface/include/dc
    ${NETCO_IDL}/interface/include/debug_tools
    ${NETCO_IDL}/interface/include/design_pattern
    ${NETCO_IDL}/interface/include/system_base
    ${NETCO_IDL}/interface/include/utils
)
target_compile_options(netco_callback_utest PRIVATE
    -ffunction-sections -fdata-sections -Werror=cast-function-type
)
target_link_options(netco_callback_utest PRIVATE -Wl,--gc-sections)
target_link_libraries(netco_callback_utest PRIVATE c_sec pthread)
add_test(NAME netco_callback_utest COMMAND netco_callback_utest)
