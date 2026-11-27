# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(kv_quant_sparse_flash_attention_v2_depends
    attention/common
    attention/sparse_flash_attention
    attention/kv_quant_sparse_flash_attention
    common
    CACHE INTERNAL "Dependencies for kv_quant_sparse_flash_attention_v2"
)

# The full prototype is exposed as aclnnInnerKvQuantSparseFlashAttentionV2.
# Manual V2/V3 wrappers preserve the published V2 ABI and opt into actual_sparse_lengths through V3.
target_sources(op_host_aclnnInner PRIVATE
    kv_quant_sparse_flash_attention_v2_def.cpp
)

target_sources(optiling PRIVATE
    kv_quant_sparse_flash_attention_v2_tiling_register.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../kv_quant_sparse_flash_attention/op_host/kv_quant_sparse_flash_attention_tiling.cpp
)

target_sources(opsproto PRIVATE
    kv_quant_sparse_flash_attention_v2_infershape.cpp
)

target_sources(opapi PRIVATE
    op_api/aclnn_kv_quant_sparse_flash_attention_v2.cpp
    op_api/aclnn_kv_quant_sparse_flash_attention_v3.cpp
)

install(FILES
    op_api/aclnn_kv_quant_sparse_flash_attention_v2.h
    op_api/aclnn_kv_quant_sparse_flash_attention_v3.h
    DESTINATION ${ACLNN_INC_INSTALL_DIR}/aclnnop
)

target_include_directories(optiling PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../kv_quant_sparse_flash_attention/op_host
)

add_ops_compile_options(
    OP_NAME KvQuantSparseFlashAttentionV2
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -Werror
            -mllvm -cce-vf-remove-membar=false
            -mllvm -cce-aicore-hoist-movemask=false
)
