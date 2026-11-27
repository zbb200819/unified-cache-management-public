/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_KV_QUANT_SPARSE_FLASH_ATTENTION_V3_H
#define ACLNN_KV_QUANT_SPARSE_FLASH_ATTENTION_V3_H

#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

// V3 adds the optional per-request actual_sparse_lengths input to the V2 interface.
__attribute__((visibility("default"))) aclnnStatus aclnnKvQuantSparseFlashAttentionV3GetWorkspaceSize(
    const aclTensor* query, const aclTensor* key, const aclTensor* value, const aclTensor* sparseIndices,
    const aclTensor* keyDequantScaleOptional, const aclTensor* valueDequantScaleOptional,
    const aclTensor* blockTableOptional, const aclTensor* actualSeqLengthsQueryOptional,
    const aclTensor* actualSeqLengthsKvOptional, const aclTensor* sinksOptional,
    const aclTensor* actualSparseLengthsOptional, double scaleValue, int64_t keyQuantMode, int64_t valueQuantMode,
    int64_t sparseBlockSizeOptional, char* layoutQueryOptional, char* layoutKvOptional, int64_t sparseModeOptional,
    int64_t preTokensOptional, int64_t nextTokensOptional, int64_t attentionModeOptional,
    int64_t quantScaleRepoModeOptional, int64_t tileSizeOptional, int64_t ropeHeadDimOptional,
    bool returnSoftmaxLseOptional, const aclTensor* attentionOut, const aclTensor* softmaxMax,
    const aclTensor* softmaxSum, uint64_t* workspaceSize, aclOpExecutor** executor);

__attribute__((visibility("default"))) aclnnStatus aclnnKvQuantSparseFlashAttentionV3(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_KV_QUANT_SPARSE_FLASH_ATTENTION_V3_H
