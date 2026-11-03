/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * */
#ifndef ATB_KVCACHE_PREFETCH_H
#define ATB_KVCACHE_PREFETCH_H
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <mutex>
#include <omp.h>
#include <pybind11/numpy.h>
#include <queue>
#include <sstream>
#include <stdarg.h>
#include <stdexcept>
#include <stdio.h>
#include <string>
#include <thread>
#include <torch/torch.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "ucmstore_v1.h"

namespace py = pybind11;
namespace ucmprefetch {
using BlockId = std::array<std::byte, 16>; /* 16-byte block hash */

class __attribute__((visibility("hidden"))) UCMKVPrefetch {
private:
    UC::StoreV1* mStoreV1_{nullptr};
    uint32_t mLayerNum = 78;
    // [I19] UCMLoadSSD 的上一次异步 Load 的未收尾 handle:
    // TaskWrapper::tasks_ 的条目只在 Wait() 中 erase,丢弃 handle 即每调用
    // 泄漏一条;收尾动作顺延到下一次调用(见 UCMLoadSSD)。
    UC::Detail::TaskHandle mPendingLoadHandle{0};
    bool mHasPendingLoad{false};

    enum class PrefetchStatus : int32_t {
        GSA_PREFETCH_DONE = 0,
        GSA_STORE_EMPTY,
        GSA_BLOCKID_EMPTY,
        GSA_HANDLE_ERROR,
        GSA_LOOKUP_ERROR
    };

public:
    UCMKVPrefetch() {}

    ~UCMKVPrefetch() {}

    void StoreInit(uintptr_t storePtr, uint32_t layerNum)
    {
        mStoreV1_ = reinterpret_cast<UC::StoreV1*>(storePtr);
        if (mStoreV1_ == nullptr) { throw std::runtime_error("storeV1 is null"); }
        mLayerNum = layerNum;
        // [I19] 新 store 实例:上一次的 pending handle 已失效,不再收尾。
        mHasPendingLoad = false;
    }

    int32_t UCMLoadSSD(const std::vector<UC::Detail::BlockId>& blockIds)
    {
        if (mStoreV1_ == nullptr) { return static_cast<int32_t>(PrefetchStatus::GSA_STORE_EMPTY); }

        if (blockIds.empty()) { return static_cast<int32_t>(PrefetchStatus::GSA_BLOCKID_EMPTY); }
        // [I19] 收尾上一次调用的 load handle:TaskWrapper::tasks_ 的条目
        // (TaskPtr + WaiterPtr + map 节点)只在 Wait() 中 erase——此前 handle
        // 被直接丢弃,每次调用泄漏一条,长跑缓慢增长。本调用为异步语义,
        // 收尾动作顺延到下一次调用(通常隔一步,任务大概率已完成):先
        // Check,完成则 Wait(立即返回并清理);未完成也 Wait(store 侧
        // timeoutMs_ + Cancel 兜底,条目必然被 erase,只放弃该次预取)。
        if (mHasPendingLoad) {
            auto checkResult = mStoreV1_->Check(mPendingLoadHandle);
            if (!checkResult || !checkResult.Value()) {
                std::fprintf(stderr,
                             "[GSA] UCMLoadSSD: previous async load (handle=%llu) still "
                             "in flight, draining it via Wait before submitting a new one\n",
                             static_cast<unsigned long long>(mPendingLoadHandle));
            }
            (void)mStoreV1_->Wait(mPendingLoadHandle);
            mHasPendingLoad = false;
        }
        UC::Detail::TaskDesc task;
        task.reserve(blockIds.size() * mLayerNum);

        for (const auto& blockId : blockIds) {
            for (uint32_t layerID = 0; layerID < mLayerNum; layerID++) {
                task.emplace_back(UC::Detail::Shard{blockId, layerID, {}});
            }
        }
        auto loadHandle = mStoreV1_->Load(std::move(task));
        if (!loadHandle) { return static_cast<int32_t>(PrefetchStatus::GSA_HANDLE_ERROR); }
        mPendingLoadHandle = loadHandle.Value();
        mHasPendingLoad = true;
        return static_cast<int32_t>(PrefetchStatus::GSA_PREFETCH_DONE);
    }

    std::vector<uint8_t> UCMLookup(const std::vector<UC::Detail::BlockId>& blockIds)
    {
        if (mStoreV1_ == nullptr) { return {}; }

        if (blockIds.empty()) { return {}; }

        auto lookupResult = mStoreV1_->Lookup(blockIds.data(), blockIds.size());
        if (!lookupResult) { return {}; }

        auto result = lookupResult.Value();
        if (result.size() != blockIds.size()) { return {}; }
        return result;
    }
};

}  // namespace ucmprefetch

#endif