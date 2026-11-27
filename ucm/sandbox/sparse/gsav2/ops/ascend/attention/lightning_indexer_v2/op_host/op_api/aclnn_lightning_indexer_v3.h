/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_LIGHTNING_INDEXER_V3_H
#define ACLNN_LIGHTNING_INDEXER_V3_H

#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief candidate（两级 TopK source 模式）接口：在 V2 原型基础上新增
 * candidate_topk_blocks / candidate_block_size 属性与 candidate_block_indices /
 * candidate_block_length 输出。candidate_topk_blocks=-1 时关闭 candidate，行为与
 * aclnnLightningIndexerV2 一致；有效值 (0,2048] 且为 64 的倍数时开启 source 模式。
 * candidate 特性仅 ascend910b/ascend910_93 支持。
 * @domain aclnn_ops_infer
 */
__attribute__((visibility("default"))) aclnnStatus aclnnLightningIndexerV3GetWorkspaceSize(
    const aclTensor* q, const aclTensor* k, const aclTensor* w, const aclTensor* cuSeqlensQOptional,
    const aclTensor* cuSeqlensKOptional, const aclTensor* sequsedQOptional, const aclTensor* sequsedKOptional,
    const aclTensor* cmpResidualKOptional, const aclTensor* blockTableOptional,
    const aclTensor* outputIdxOffsetOptional, const aclTensor* metadataOptional, int64_t topk, int64_t maxSeqlenQ,
    char* layoutQOptional, char* layoutKOptional, int64_t maskMode, int64_t cmpRatio, int64_t returnValue,
    int64_t candidateTopkBlocks, int64_t candidateBlockSize, const aclTensor* sparseIndices,
    const aclTensor* sparseValues, const aclTensor* candidateBlockIndices, const aclTensor* candidateBlockLength,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnLightningIndexerV3 is used to perform calculations.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnLightningIndexerV3(void* workspace, uint64_t workspaceSize,
                                                                           aclOpExecutor* executor,
                                                                           const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_LIGHTNING_INDEXER_V3_H
