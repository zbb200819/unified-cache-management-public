/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_LIGHTNING_INDEXER_V2_H
#define ACLNN_LIGHTNING_INDEXER_V2_H

#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 兼容接口：签名冻结为已发布的 V2 原型（无 candidate 参数，2 输出）。
 * candidate（两级 TopK）能力由 aclnnLightningIndexerV3 承接。
 * 符号/参数顺序/类型与已发布版本一致。
 * @domain aclnn_ops_infer
 */
__attribute__((visibility("default"))) aclnnStatus aclnnLightningIndexerV2GetWorkspaceSize(
    const aclTensor* q, const aclTensor* k, const aclTensor* w, const aclTensor* cuSeqlensQOptional,
    const aclTensor* cuSeqlensKOptional, const aclTensor* sequsedQOptional, const aclTensor* sequsedKOptional,
    const aclTensor* cmpResidualKOptional, const aclTensor* blockTableOptional,
    const aclTensor* outputIdxOffsetOptional, const aclTensor* metadataOptional, int64_t topk, int64_t maxSeqlenQ,
    char* layoutQOptional, char* layoutKOptional, int64_t maskMode, int64_t cmpRatio, int64_t returnValue,
    const aclTensor* sparseIndices, const aclTensor* sparseValues, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnLightningIndexerV2 is used to perform calculations.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnLightningIndexerV2(void* workspace, uint64_t workspaceSize,
                                                                           aclOpExecutor* executor,
                                                                           const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_LIGHTNING_INDEXER_V2_H
