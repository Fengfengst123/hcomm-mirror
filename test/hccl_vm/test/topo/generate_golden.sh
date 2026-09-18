#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 重新生成集群拓扑 golden 基线 (test/topo/golden/)
# 用法: bash test/topo/generate_golden.sh [build目录, 默认 ./build]
# 说明: 以当前代码与脚本的输出为基准, 重建 golden/cluster 与 golden/ranktable 下的基线文件

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
BUILD_DIR="${1:-${REPO_ROOT}/build}"

if [ ! -x "${BUILD_DIR}/output/bin/test_topo_golden" ]; then
    echo "[错误] 未找到 ${BUILD_DIR}/output/bin/test_topo_golden, 请先构建(BUILD_TESTS=ON)" >&2
    exit 1
fi

echo "[信息] 重新生成 golden 基线到 ${SCRIPT_DIR}/golden ..."
TOPO_UT_UPDATE_GOLDEN=1 "${BUILD_DIR}/output/bin/test_topo_golden"

echo "[信息] golden 基线已更新: ${SCRIPT_DIR}/golden"
