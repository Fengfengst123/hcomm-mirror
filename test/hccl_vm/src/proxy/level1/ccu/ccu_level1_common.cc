/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

/**
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * the full text of the License. Description: CCU Level1
 * 公共工具函数实现——仅放置"不依赖 TLS / 不依赖引擎内部状态"
 *              的纯函数。依赖引擎状态的函数（PrimKindName / NormalizeSimTask
 * 等） 实现仍在 ccu_kernel_sim.cc 中。
 *              注意：本文件当前暂未编入构建（与 ccu_kernel_sim.cc
 * 存在同名符号）， 待后续从 ccu_kernel_sim.cc 迁出公共实现后启用。 Create:
 * 2026-09-15
 */

#define HCCL_VM_MODULE "CCU_L1_COMMON"

#include "ccu_level1_common.h"

#include "sim_log.h"

namespace HcclSim {
namespace CcuSim {

/*==================== 句柄 tag 校验（constexpr 定义已在头文件中）
 * ====================*/
/* CheckHandleTag 为 constexpr inline，无需 .cc 实现。 */

} // namespace CcuSim
} // namespace HcclSim
