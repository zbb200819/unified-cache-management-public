/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file lightning_indexer_v2_infershape.cpp
 * \brief
 */
#include "base/err_msg.h"
#include "log/log.h"
#include <vector>
#include <string>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"

using namespace ge;

namespace ops {
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t ATTR_TOPK_INDEX = 0;
constexpr uint32_t ATTR_QUERY_LAYOUT_INDEX = 2;
constexpr uint32_t ATTR_KEY_LAYOUT_INDEX = 3;
constexpr uint32_t ATTR_RETURN_VALUE_INDEX = 6;
constexpr uint32_t ATTR_CANDIDATE_TOPK_BLOCKS_INDEX = 7;
constexpr int64_t CANDIDATE_TOPK_BLOCKS_OFF = -1;

static ge::graphStatus InferShapeLightningIndexerV2(gert::InferShapeContext* context)
{
    OP_CHECK_IF(
        context == nullptr,
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{"LightningIndexerV2", "InferShape/InferDataType context is unavailable; check "
                                                            "the operator registration and context construction."})),
        return ge::GRAPH_FAILED);
    const gert::Shape* queryShape = context->GetInputShape(QUERY_INDEX);
    OP_CHECK_IF(queryShape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(
                    (context->GetNodeName() == nullptr ? "LightningIndexerV2" : context->GetNodeName()), "q"),
                return ge::GRAPH_FAILED);
    const gert::Shape* keyShape = context->GetInputShape(KEY_INDEX);
    OP_CHECK_IF(keyShape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(
                    (context->GetNodeName() == nullptr ? "LightningIndexerV2" : context->GetNodeName()), "k"),
                return ge::GRAPH_FAILED);

    gert::Shape* sparseIndicesShape = context->GetOutputShape(0);
    OP_CHECK_IF(sparseIndicesShape == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1014", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{
                        (context->GetNodeName() == nullptr ? "LightningIndexerV2" : context->GetNodeName()),
                        "InferShape framework object sparseIndicesShape is unavailable; check the operator "
                        "registration and context construction."})),
                return ge::GRAPH_FAILED);
    gert::Shape* sparseValuesShape = context->GetOutputShape(1);
    OP_CHECK_IF(sparseValuesShape == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1014", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{
                        (context->GetNodeName() == nullptr ? "LightningIndexerV2" : context->GetNodeName()),
                        "InferShape framework object sparseValuesShape is unavailable; check the operator registration "
                        "and context construction."})),
                return ge::GRAPH_FAILED);
    // candidate (two-level topk)：source 输出 + 预留 block_length（恒空）
    gert::Shape* candidateBlockIndicesShape = context->GetOutputShape(2);
    OP_CHECK_IF(candidateBlockIndicesShape == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1014", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{
                        (context->GetNodeName() == nullptr ? "LightningIndexerV2" : context->GetNodeName()),
                        "InferShape framework object candidateBlockIndicesShape is unavailable; check the operator "
                        "registration and context construction."})),
                return ge::GRAPH_FAILED);
    gert::Shape* candidateBlockLengthShape = context->GetOutputShape(3);
    OP_CHECK_IF(candidateBlockLengthShape == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1014", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{
                        (context->GetNodeName() == nullptr ? "LightningIndexerV2" : context->GetNodeName()),
                        "InferShape framework object candidateBlockLengthShape is unavailable; check the operator "
                        "registration and context construction."})),
                return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1014", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{
                        (context->GetNodeName() == nullptr ? "LightningIndexerV2" : context->GetNodeName()),
                        "InferShape framework object attrs is unavailable; check the operator registration and context "
                        "construction."})),
                return ge::GRAPH_FAILED);
    const char* inputLayoutQueryPtr = attrs->GetAttrPointer<char>(ATTR_QUERY_LAYOUT_INDEX);
    OP_CHECK_IF(inputLayoutQueryPtr == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1014", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{
                        (context->GetNodeName() == nullptr ? "LightningIndexerV2" : context->GetNodeName()),
                        "InferShape: registered optional attribute layout_q is unavailable; check context construction "
                        "and default-attribute materialization."})),
                return ge::GRAPH_FAILED);
    const char* inputLayoutKeyPtr = attrs->GetAttrPointer<char>(ATTR_KEY_LAYOUT_INDEX);
    OP_CHECK_IF(inputLayoutKeyPtr == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1014", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{
                        (context->GetNodeName() == nullptr ? "LightningIndexerV2" : context->GetNodeName()),
                        "InferShape: registered optional attribute layout_k is unavailable; check context construction "
                        "and default-attribute materialization."})),
                return ge::GRAPH_FAILED);
    const int64_t* seleced_count = attrs->GetInt(ATTR_TOPK_INDEX);
    OP_CHECK_IF(seleced_count == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    (context->GetNodeName() == nullptr ? "LightningIndexerV2" : context->GetNodeName()), "topk",
                    "Required attribute topk is unavailable."),
                return ge::GRAPH_FAILED);
    std::string inputLayoutQueryPtrStr = std::string(inputLayoutQueryPtr);
    std::string inputLayoutKeyPtrStr = std::string(inputLayoutKeyPtr);
    OP_CHECK_IF(
        inputLayoutQueryPtrStr != "TND" && inputLayoutQueryPtrStr != "BSND",
        OP_LOGE_FOR_INVALID_VALUE("LightningIndexerV2", "layout_q", inputLayoutQueryPtrStr.c_str(), "TND or BSND"),
        return ge::GRAPH_FAILED);

    sparseIndicesShape->SetDimNum(queryShape->GetDimNum());
    if (inputLayoutQueryPtrStr == "BSND") {
        OP_CHECK_IF(queryShape->GetDimNum() != 4,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("LightningIndexerV2", "q",
                                                             std::to_string(queryShape->GetDimNum()).c_str(),
                                                             "When in BSND case, the shape dim of q must be 4"),
                    return ge::GRAPH_FAILED);
        sparseIndicesShape->SetDim(0, queryShape->GetDim(0)); // 0:Dim B
        sparseIndicesShape->SetDim(1, queryShape->GetDim(1)); // 1:Dim S
        sparseIndicesShape->SetDim(2, keyShape->GetDim(2));   // 2:Dim N
        sparseIndicesShape->SetDim(3, *seleced_count);        // 3:Dim K
    } else {
        OP_CHECK_IF(queryShape->GetDimNum() != 3,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("LightningIndexerV2", "q",
                                                             std::to_string(queryShape->GetDimNum()).c_str(),
                                                             "When in TND case, the shape dim of q must be 3"),
                    return ge::GRAPH_FAILED);
        sparseIndicesShape->SetDim(0, queryShape->GetDim(0));            // 0:Dim T
        int32_t nDimIndex = (inputLayoutKeyPtrStr == "PA_BBND") ? 2 : 1; // 2:Key Dim N
        sparseIndicesShape->SetDim(1, keyShape->GetDim(nDimIndex));      // 1:Dim N
        sparseIndicesShape->SetDim(2, *seleced_count);                   // 2:Dim K
    }

    // GE Int 属性按 int64_t 读取（仓惯例），不做小端 bool* type-punning
    const int64_t* return_value = attrs->GetAttrPointer<int64_t>(ATTR_RETURN_VALUE_INDEX);
    bool returnValueFlag = (return_value != nullptr) ? (*return_value != 0) : false;
    if (returnValueFlag) {
        *sparseValuesShape = *sparseIndicesShape;
    } else {
        sparseValuesShape->SetDimNum(1);
        sparseValuesShape->SetDim(0, 0);
    }

    // candidate_block_indices: candidate_topk_blocks != -1(source) 时输出，复制 sparse_indices shape、
    // 末维改 candidate_topk_blocks（BSND 4 维 / TND 3 维双布局）；off 时输出 (0,)
    const int64_t* candidate_topk_blocks = attrs->GetAttrPointer<int64_t>(ATTR_CANDIDATE_TOPK_BLOCKS_INDEX);
    int64_t candBlocks = (candidate_topk_blocks != nullptr) ? *candidate_topk_blocks : CANDIDATE_TOPK_BLOCKS_OFF;
    if (candBlocks > 0) {
        *candidateBlockIndicesShape = *sparseIndicesShape;
        candidateBlockIndicesShape->SetDim(candidateBlockIndicesShape->GetDimNum() - 1, candBlocks);
    } else {
        candidateBlockIndicesShape->SetDimNum(1);
        candidateBlockIndicesShape->SetDim(0, 0);
    }
    // candidate_block_length: 预留接口，恒空
    candidateBlockLengthShape->SetDimNum(1);
    candidateBlockLengthShape->SetDim(0, 0);
    OP_LOGI(context->GetNodeName(), "LightningIndexerV2 InferShape end.");

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeLightningIndexerV2(gert::InferDataTypeContext* context)
{
    OP_CHECK_IF(
        context == nullptr,
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{"LightningIndexerV2", "InferShape/InferDataType context is unavailable; check "
                                                            "the operator registration and context construction."})),
        return ge::GRAPH_FAILED);
    OP_LOGI(context->GetNodeName(), "Enter LightningIndexerV2 InferDataType impl.");
    // default set q's dtype as fia's output type
    ge::DataType outputType = ge::DT_INT32;
    // attention_out, outidx:0
    context->SetOutputDataType(0, outputType);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    context->SetOutputDataType(2, ge::DT_INT32); // candidate_block_indices:2
    context->SetOutputDataType(3, ge::DT_INT32); // candidate_block_length:3
    OP_LOGI(context->GetNodeName(), "LightningIndexerV2 InferDataType end.");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(LightningIndexerV2)
    .InferShape(InferShapeLightningIndexerV2)
    .InferDataType(InferDataTypeLightningIndexerV2);
} // namespace ops
