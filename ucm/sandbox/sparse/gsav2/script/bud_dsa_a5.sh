#!/bin/bash
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../../.." && pwd)
OPS_DIR="${REPO_ROOT}/ucm/sandbox/sparse/gsav2/ops"

echo "[GSA-A5] 仓库根目录: ${REPO_ROOT}"
cd "${REPO_ROOT}"
echo "当前工作目录是：$PWD"

export PIP_INDEX_URL="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
export PLATFORM="ascend" ENABLE_SPARSE=true
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/local/Ascend/ascend-toolkit/latest/$(uname -i)-linux/devlib"
pip install -v -e . --no-build-isolation

echo "[GSA-A5] ===== 编译并安装 GSA v2 A5 算子 ====="
echo "[GSA-A5] 公共 kernel 头文件由 build_ops.sh 从容器内 CANN 源码复制"
cd "${OPS_DIR}"
bash build.sh a5 install

cd "${REPO_ROOT}"
echo "[GSA-A5] ===== Done ====="
