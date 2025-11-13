#!/bin/bash
set -e
set -u

echo "$(date): 开始启动 vLLM 服务..."

# ================================
# 🔧 基础参数配置
# ================================
export PYTHONHASHSEED=123456
export VLLM_TORCH_PROFILER_DIR="/home/externals/suanfabu/lihao/profiling_results"

# 模型配置
MODEL_PATH="/home/models/DeepSeek-R1-Distill-Qwen-32B"
MODEL_NAME="DeepSeek-R1-Distill-Qwen-32B"

# vLLM 服务配置
MAX_MODEL_LEN=131000
TENSOR_PARALLEL_SIZE=2
GPU_MEMORY_UTILIZATION=0.87
SERVER_PORT=8090
BLOCK_SIZE=128
ENABLE_PREFIX_CACHING="--no-enable-prefix-caching"  # 可置为空 ""

# ================================
# 🧠 Unified Cache (UCM) 配置
# ================================
UCM_STORE_PATH="/home/externals/suanfabu/data"
KV_CONNECTOR="UnifiedCacheConnectorV1"
KV_MODULE_PATH="ucm.integration.vllm.uc_connector"
KV_ROLE="kv_both"
UCM_CONNECTOR_NAME="UcmNfsStore"
TRANSFER_STREAM_NUMBER=16
UCM_SPARSE_CONFIG_GSA="{}"

# ================================
# ⚙️ 环境初始化
# ================================
mkdir -p "$UCM_STORE_PATH"
rm -rf "$UCM_STORE_PATH"/*

# 检查模型目录是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型目录不存在！ ($MODEL_PATH)"
    exit 1
fi

# ================================
# 🧾 构建 KV 传输配置 JSON
# ================================
KV_TRANSFER_CONFIG=$(cat <<EOF
{
    "kv_connector": "$KV_CONNECTOR",
    "kv_connector_module_path": "$KV_MODULE_PATH",
    "kv_role": "$KV_ROLE",
    "kv_connector_extra_config": {
        "ucm_connector_name": "$UCM_CONNECTOR_NAME",
        "ucm_connector_config": {
            "storage_backends": "$UCM_STORE_PATH",
            "transferStreamNumber": $TRANSFER_STREAM_NUMBER
        },
        "ucm_sparse_config": {
            "GSA": $UCM_SPARSE_CONFIG_GSA
        }
    }
}
EOF
)

# ================================
# 🚀 启动 vLLM 服务
# ================================
echo "$(date): 正在启动 vLLM，模型：$MODEL_NAME"
vllm serve "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --max-model-len "$MAX_MODEL_LEN" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --trust-remote-code \
    --port "$SERVER_PORT" \
    --block-size "$BLOCK_SIZE" \
    $ENABLE_PREFIX_CACHING \
    --kv-transfer-config "$KV_TRANSFER_CONFIG"
