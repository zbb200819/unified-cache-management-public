# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set(quant_lightning_indexer_v2_depends
    attention/lightning_indexer_v2
    CACHE INTERNAL "Dependencies for quant_lightning_indexer_v2"
)

target_sources(op_host_aclnnInner PRIVATE
    quant_lightning_indexer_v2_def.cpp
)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../lightning_indexer_v2/op_host/checkers/checker_sources.cmake)
add_lightning_indexer_v2_common_checker_sources(optiling)
target_sources(optiling PRIVATE
    quant_lightning_indexer_v2_tiling.cpp
    quant_lightning_indexer_v2_tiling_info_parser.cpp
    checkers/qliv2_checker.cpp
    checkers/qliv2_variant_checker.cpp
)

target_sources(opsproto PRIVATE
    quant_lightning_indexer_v2_infershape.cpp
)

target_sources(opapi PRIVATE
    ../op_api/aclnn_quant_lightning_indexer_v2.cpp
    ../op_api/aclnn_quant_lightning_indexer_v3.cpp
)

target_include_directories(op_host_aclnnInner PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
)
target_include_directories(optiling PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
)
target_include_directories(opsproto PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
)

# metadata 布局头使用 inline constexpr，与源工程的 GNU C++17 host 编译环境保持一致
target_compile_options(optiling PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-std=gnu++1z>
)

install(FILES
    ../op_api/aclnn_quant_lightning_indexer_v2.h
    ../op_api/aclnn_quant_lightning_indexer_v3.h
    DESTINATION ${ACLNN_INC_INSTALL_DIR}/aclnnop
)

add_ops_compile_options(
    OP_NAME QuantLightningIndexerV2
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -Werror
            -mllvm -cce-vf-remove-membar=false
            -mllvm -cce-aicore-hoist-movemask=false
)
