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
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <kvcache_log.h>
#include <map>
#include <mutex>
#include <new>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <utility>
#include <vector>
#include "kvcache_pre.h"
#include "kvcache_pre_detail.h"
#include "securec.h"

#define H2D_INFO_SIZE 6

namespace ucmprefetch {

void GSAPrefetchEngineC::KVCacheInplaceScatter(std::string type, int layerID, aclrtStream stream,
                                               aclrtContext context)
{
    mLogger.log(LogLevel::DEBUG,
                "Decode step: %u, |KVCache Prefetch| Start KVCacheInplaceScatter "
                "type: %s, load num: %d \n",
                mDecodeStep, type.c_str(), mLoadNum[layerID]);

    int lastDim = 0;
    int streamIdx = layerID % mStreamNum;
    torch::Tensor hbmBufferND;
    aclTensor* dstTensor;

    if (type == "K") {
        lastDim = mKDim;
        hbmBufferND = mKBufferDevice[streamIdx].slice(0, 0, mLoadNum[layerID]);
        dstTensor = mKCacheAcl[layerID];
    } else {
        lastDim = mVDim;
        hbmBufferND = mVBufferDevice[streamIdx].slice(0, 0, mLoadNum[layerID]);
        dstTensor = mVCacheAcl[layerID];
    }

    auto ret = aclrtSetCurrentContext(context);
    if (ret != ACL_SUCCESS) {
        const char* errMsg = aclGetRecentErrMsg();
        mLogger.log(LogLevel::ERROR,
                    "|KVCache Prefetch| aclrtSetCurrentContext Error: %d, Error: %s \n", ret,
                    errMsg ? errMsg : "Unknown error");
    }

    torch::Tensor dstIndexND = mLoadIndexDevice[streamIdx]
                                   .slice(0, 0, mLoadNum[layerID])
                                   .view({-1, 1})
                                   .expand({mLoadNum[layerID], lastDim});
    aclTensor* dstIndexTensor;
    aclTensor* dstHBMBufferTensor;
    PrepareScatterAclTensor(type, streamIdx, dstIndexND, hbmBufferND, dstIndexTensor,
                            dstHBMBufferTensor);

    RunInplaceScatter(type, layerID, streamIdx, dstTensor, dstIndexTensor, dstHBMBufferTensor,
                      stream);

    mLogger.log(LogLevel::DEBUG,
                "Decode step: %u, |KVCache Prefetch| End KVCacheInplaceScatter "
                "type: %s \n",
                mDecodeStep, type.c_str());
}

void GSAPrefetchEngineC::PrepareScatterAclTensor(const std::string& type, int streamIdx,
                                                 torch::Tensor& dstIndexND,
                                                 torch::Tensor& hbmBufferND,
                                                 aclTensor*& dstIndexTensor,
                                                 aclTensor*& dstHBMBufferTensor)
{
    if (type == "K") {
        if (mKDstIndexAcl[streamIdx] != nullptr) {
            ReleaseAclTensor(mKDstIndexAcl[streamIdx]);
            mKDstIndexAcl[streamIdx] = nullptr;
        }
        if (mKHBMBufferAcl[streamIdx] != nullptr) {
            ReleaseAclTensor(mKHBMBufferAcl[streamIdx]);
            mKHBMBufferAcl[streamIdx] = nullptr;
        }

        mKDstIndexAcl[streamIdx] = AtTensorToAclTensorShared(dstIndexND);
        mKHBMBufferAcl[streamIdx] = AtTensorToAclTensorShared(hbmBufferND);
        dstIndexTensor = mKDstIndexAcl[streamIdx];
        dstHBMBufferTensor = mKHBMBufferAcl[streamIdx];
    } else {
        if (mVDstIndexAcl[streamIdx] != nullptr) {
            ReleaseAclTensor(mVDstIndexAcl[streamIdx]);
            mVDstIndexAcl[streamIdx] = nullptr;
        }
        if (mVHBMBufferAcl[streamIdx] != nullptr) {
            ReleaseAclTensor(mVHBMBufferAcl[streamIdx]);
            mVHBMBufferAcl[streamIdx] = nullptr;
        }

        mVDstIndexAcl[streamIdx] = AtTensorToAclTensorShared(dstIndexND);
        mVHBMBufferAcl[streamIdx] = AtTensorToAclTensorShared(hbmBufferND);
        dstIndexTensor = mVDstIndexAcl[streamIdx];
        dstHBMBufferTensor = mVHBMBufferAcl[streamIdx];
    }
}

void* GSAPrefetchEngineC::GetScatterWorkspace(const std::string& type, int layerID, int streamIdx,
                                              uint64_t workspaceSize)
{
    uint64_t& maxWorkspaceSize =
        type == "K" ? mKMaxWorkspaceSize[streamIdx] : mVMaxWorkspaceSize[streamIdx];

    void*& workspaceAddr = type == "K" ? mKWorkspaceAddr[streamIdx] : mVWorkspaceAddr[streamIdx];

    if (workspaceSize > maxWorkspaceSize) {
        if (maxWorkspaceSize > 0) {
            auto ret = aclrtFree(workspaceAddr);
            if (ret != ACL_SUCCESS) {
                const char* errMsg = aclGetRecentErrMsg();
                mLogger.log(LogLevel::ERROR, "|KVCache Prefetch| aclrtFree Error: %d, Error: %s\n",
                            ret, errMsg ? errMsg : "Unknown error");
            }
        }

        auto ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            const char* errMsg = aclGetRecentErrMsg();
            mLogger.log(LogLevel::ERROR, "|KVCache Prefetch| aclrtMalloc Error: %d, Erroe: %s \n",
                        ret, errMsg ? errMsg : "Unknown error");
            KVPackAbort(mKVPackControl);
        } else {
            mLogger.log(LogLevel::DEBUG,
                        "|KVCache Prefetch| aclrtMalloc success for layer: %d, "
                        "stype: %s\n",
                        layerID, type.c_str());
        }

        maxWorkspaceSize = workspaceSize;
    }

    return workspaceAddr;
}

void GSAPrefetchEngineC::RunInplaceScatter(const std::string& type, int layerID, int streamIdx,
                                           aclTensor* dstTensor, aclTensor* dstIndexTensor,
                                           aclTensor* dstHBMBufferTensor, aclrtStream stream)
{
    int64_t dim = 0;
    int64_t reduce = 0;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto ret = aclnnInplaceScatterGetWorkspaceSize(
        dstTensor, dim, dstIndexTensor, dstHBMBufferTensor, reduce, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        const char* errMsg = aclGetRecentErrMsg();
        mLogger.log(LogLevel::ERROR,
                    "|KVCache Prefetch| InplaceScatterGetWorkspaceSize "
                    "Error: %d, Error: %s\n",
                    ret, errMsg ? errMsg : "Unknown error");
        KVPackAbort(mKVPackControl);
    }

    void* workspaceAddr = GetScatterWorkspace(type, layerID, streamIdx, workspaceSize);

    ret = aclnnInplaceScatter(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        const char* errMsg = aclGetRecentErrMsg();
        mLogger.log(LogLevel::ERROR,
                    "|KVCache Prefetch|  aclnnGather Kcache Error: %d, Error: %s\n", ret,
                    errMsg ? errMsg : "Unknown error");
        KVPackAbort(mKVPackControl);
    }
}

int GSAPrefetchEngineC::CallPrefetchProcess(const TaskInfo& taskInfo, ThreadInfo& threadInfo)
{
    std::lock_guard<std::mutex> lock(mMutexUseToken);
    auto start = std::chrono::high_resolution_clock::now();
    uint32_t oneRunBsLen = taskInfo.reqIdList.size();
    int ret = 0;

    // 判别式：预取结果准备失败（已打日志、结果清零、版本照常递增）时记录返回码，
    // 后续层循环仍需完成以维持跨 rank 协议完整（空结果天然走零加载路径）。
    if (PreparePrefetchResults(taskInfo) != 0) { ret = -1; }

    mLogger.log(LogLevel::INFO,
                "Decode step: %u, |KVCache Prefetch| "
                "all layer average hit rate: %f for bs: %d\n",
                mDecodeStep, average2d(mAvgHitRate), oneRunBsLen);
    for (size_t i = 0; i < mAvgHitRate.size(); ++i) { mAvgHitRate[i].clear(); }
    auto managerEnd = std::chrono::high_resolution_clock::now();
    auto managerDuration =
        std::chrono::duration_cast<std::chrono::microseconds>(managerEnd - start);
    PrefetchTiming timing;

    for (int layerID = 0; layerID < mLayerNum; layerID++) {
        MergeAndCopyPrefetchLayer(taskInfo, threadInfo, layerID, timing);
    }
    // 收尾同步失败打日志（Synchronize 内部已记录）并上抛返回码，进程不终止。
    if (Synchronize(threadInfo.context_, threadInfo.stream_, mStreamNum, mLogger) != 0) {
        ret = -1;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    mLogger.log(LogLevel::INFO,
                "Decode step: %u, |KVCache Prefetch| Finish async pretch, "
                "Synchronize cost: %lu, manager time:%lu, "
                "cpu merge time: %lu, memcpy time: %lu, "
                "sync time: %lu, h2d time: %lu, ops time: %lu\n",
                mDecodeStep, duration.count(), managerDuration.count(), timing.cpuMove,
                timing.memcpyUs, timing.sync, timing.h2d, timing.ops);
    return ret;
}

int GSAPrefetchEngineC::PreparePrefetchResults(const TaskInfo& taskInfo)
{
    uint32_t oneRunBsLen = taskInfo.reqIdList.size();
    PrefetchShmLayout layout{};
    layout.versionSize = 4;
    layout.countSize = 4;
    layout.maxLoadNum = mMaxLoadNum;
    layout.missTokenSize = layout.countSize + layout.maxLoadNum * sizeof(int);
    layout.loadSlotSize = layout.countSize + layout.maxLoadNum * sizeof(int);
    layout.totalSize = layout.missTokenSize + layout.loadSlotSize + mSpaTopkLen * sizeof(int);
    uint32_t* versionPtr = static_cast<uint32_t*>(mPrefetchResultShm);
    if (mRank == 0) {
        return PublishPrefetchResultsRank0(taskInfo, oneRunBsLen, layout, versionPtr);
    }
    return SyncPrefetchResultsFromRank0(taskInfo, layout, versionPtr);
}

int GSAPrefetchEngineC::PublishPrefetchResultsRank0(const TaskInfo& taskInfo,
                                                    uint32_t oneRunBsLen,
                                                    const PrefetchShmLayout& layout,
                                                    uint32_t* versionPtr)
{
    /* 提前识别：进入并行域前完成张量合法性判别并一次性提取只读视图，
     * OMP 域内不再触碰任何 torch API（消除 data_ptr dtype 抛异常、下标
     * 越界抛 IndexError 与张量切片堆分配三类异常/终止源）。 */
    PrefetchTopkView topkView;
    if (AcquirePrefetchTopkBase(taskInfo, oneRunBsLen, topkView) != 0) {
        ZeroFillAllPrefetchResults(oneRunBsLen, layout, versionPtr);
        return -1;
    }
    int failedCount = 0;
#pragma omp parallel for num_threads(mThreadNum) proc_bind(master) reduction(+ : failedCount)
    for (int layerID = 0; layerID < mTopkIndexerNum; layerID++) {
        for (size_t i = 0; i < oneRunBsLen; i++) {
            if (ProcessPrefetchRequest(taskInfo, topkView, i, layerID, layout) != 0) {
                failedCount += 1;
            }
        }
    }
    __atomic_store_n(versionPtr, mPrefetchResultVersion.fetch_add(1) + 1, __ATOMIC_RELEASE);
    mLogger.log(LogLevel::INFO, "Rank %d synced upload version: %u, %u\n", mRank,
                mPrefetchResultVersion.load(), __atomic_load_n(versionPtr, __ATOMIC_ACQUIRE));
    if (failedCount > 0) {
        mLogger.log(LogLevel::ERROR,
                    "Decode step: %u, |KVCache Prefetch| %d (req,layer) results skipped "
                    "this step\n",
                    mDecodeStep, failedCount);
        return -1;
    }
    return 0;
}

int GSAPrefetchEngineC::AcquirePrefetchTopkBase(const TaskInfo& taskInfo, uint32_t oneRunBsLen,
                                                PrefetchTopkView& topkView)
{
    const torch::Tensor& topkList = taskInfo.topkList;
    const bool topkDefined = topkList.defined();
    const int64_t topkDim = topkDefined ? topkList.dim() : 0;
    /* mMaxBs 同时界定 shm 结果区容量（mPrefetchResultShmSize），
     * 越界批次必须在写 shm 前拒绝。 */
    const bool topkOk =
        topkDefined && topkDim >= 2 &&
        static_cast<int64_t>(topkList.size(0)) >= static_cast<int64_t>(mTopkIndexerNum) &&
        static_cast<int64_t>(topkList.size(1)) >= static_cast<int64_t>(oneRunBsLen) &&
        taskInfo.topkLenList.size() >= static_cast<size_t>(oneRunBsLen) &&
        static_cast<uint64_t>(oneRunBsLen) <= static_cast<uint64_t>(mMaxBs) &&
        topkList.scalar_type() == torch::kInt;
    if (!topkOk) {
        mLogger.log(LogLevel::ERROR,
                    "Decode step: %u, |KVCache Prefetch| invalid topkList tensor: "
                    "defined=%d dim=%lld size0=%lld size1=%lld dtype=%d bsLen=%u maxBs=%zu "
                    "topkIndexerNum=%d\n",
                    mDecodeStep, topkDefined ? 1 : 0, static_cast<long long>(topkDim),
                    topkDim >= 1 ? static_cast<long long>(topkList.size(0)) : -1LL,
                    topkDim >= 2 ? static_cast<long long>(topkList.size(1)) : -1LL,
                    topkDefined ? static_cast<int>(topkList.scalar_type()) : -1, oneRunBsLen,
                    mMaxBs, mTopkIndexerNum);
        return -1;
    }
    topkView.base = topkList.data_ptr<int>();
    topkView.stride0 = topkList.stride(0);
    topkView.stride1 = topkList.stride(1);
    topkView.numel = topkList.numel();
    return 0;
}

void GSAPrefetchEngineC::ZeroFillAllPrefetchResults(uint32_t oneRunBsLen,
                                                    const PrefetchShmLayout& layout,
                                                    uint32_t* versionPtr)
{
    std::cerr << "|KVCache Prefetch| [ERROR] PreparePrefetchResults: invalid topkList, "
                 "results zero-filled for this step\n";
    if (mPrefetchResultShm == nullptr) {
        return;
    }
    /* 结果区整体清零 = 每请求零 miss，消费侧按空结果降级；
     * 版本照常递增，保住 rank!=0 的等待协议（否则对端超时 abort）。 */
    std::memset(static_cast<uint8_t*>(mPrefetchResultShm) + layout.versionSize, 0,
                static_cast<size_t>(oneRunBsLen) * static_cast<size_t>(mTopkIndexerNum) *
                    layout.totalSize);
    __atomic_store_n(versionPtr, mPrefetchResultVersion.fetch_add(1) + 1, __ATOMIC_RELEASE);
}

uint64_t ResolvePrefetchSyncTimeoutUs()
{
    const char* envVal = std::getenv("UCM_PREFETCH_SYNC_TIMEOUT_MS");
    if (envVal == nullptr || envVal[0] == '\0') {
        return kPrefetchSyncDefaultTimeoutMs * 1000ULL;
    }
    char* parseEnd = nullptr;
    const long parsedMs = std::strtol(envVal, &parseEnd, 10);
    if (parseEnd == envVal || parsedMs <= 0) {
        std::cerr << "|KVCache Prefetch| [WARN] invalid UCM_PREFETCH_SYNC_TIMEOUT_MS='"
                  << envVal << "', fallback to " << kPrefetchSyncDefaultTimeoutMs << "ms\n";
        return kPrefetchSyncDefaultTimeoutMs * 1000ULL;
    }
    return static_cast<uint64_t>(parsedMs) * 1000ULL;
}

int GSAPrefetchEngineC::SyncPrefetchResultsFromRank0(const TaskInfo& taskInfo,
                                                     const PrefetchShmLayout& layout,
                                                     uint32_t* versionPtr)
{
    /* L-9：超时默认 30s（UCM_PREFETCH_SYNC_TIMEOUT_MS 可覆盖，100us 粒度轮询），
     * 超时判定同时复核版本号，消除"恰在边界发布成功仍被判超时"的误杀竞态；
     * fatal 对端提前置错时立即退出，不等超时。 */
    const uint64_t timeoutUs = ResolvePrefetchSyncTimeoutUs();
    uint32_t expectedVersion = mPrefetchResultVersion.load() + 1;
    uint64_t waitedUs = 0;
    while (__atomic_load_n(versionPtr, __ATOMIC_ACQUIRE) != expectedVersion &&
           waitedUs < timeoutUs) {
        if (mKVPackControl->fatalError.load(std::memory_order_acquire) != 0) {
            KVPackAbort(mKVPackControl);
        }
        usleep(100);
        waitedUs += 100;
    }
    if (waitedUs >= timeoutUs &&
        __atomic_load_n(versionPtr, __ATOMIC_ACQUIRE) != expectedVersion) {
        mLogger.log(LogLevel::ERROR,
                    "Rank %d timeout(%llu ms) waiting for prefetch result, version: %u, "
                    "expected: %u\n",
                    mRank, static_cast<unsigned long long>(timeoutUs / 1000),
                    __atomic_load_n(versionPtr, __ATOMIC_ACQUIRE), expectedVersion);
        KVPackAbort(mKVPackControl);
    }
    ReadPrefetchResults(taskInfo, layout);
    mPrefetchResultVersion.store(expectedVersion);
    mLogger.log(LogLevel::INFO, "Rank %d synced prefetch results, version: %u\n", mRank,
                __atomic_load_n(versionPtr, __ATOMIC_ACQUIRE));
    return 0;
}

int GSAPrefetchEngineC::ProcessPrefetchRequest(const TaskInfo& taskInfo,
                                               const PrefetchTopkView& topkView, size_t i,
                                               int layerID, const PrefetchShmLayout& layout)
{
    std::string oneReqID = taskInfo.reqIdList[i];
    int oneTopkLen = taskInfo.topkLenList[i];
    if (oneTopkLen <= 0) {
        mLogger.log(LogLevel::ERROR,
                    "Decode step: %u, |KVCache Prefetch| topk len is zero: "
                    "%d for req_id: %s, rank: %d, taskID: %d, i: %zu\n",
                    mDecodeStep, oneTopkLen, oneReqID.c_str(), mRank, taskInfo.taskID, i);
        // 清零槽位避免消费侧读到上一 step 的陈旧结果（零 miss = 该请求本步无加载）。
        ClearPrefetchResultSlot(i, layerID, layout);
        return 0;
    }
    /* 平坦读边界：topkLen 以外的长度越界一律拒绝（防 OOB 读与脏数据扩散）。 */
    const int64_t topkOffset = static_cast<int64_t>(layerID) * topkView.stride0 +
                               static_cast<int64_t>(i) * topkView.stride1;
    if (topkView.base == nullptr ||
        topkOffset + static_cast<int64_t>(oneTopkLen) > topkView.numel) {
        mLogger.log(LogLevel::ERROR,
                    "Decode step: %u, |KVCache Prefetch| topk view out of range: "
                    "req_id: %s, layerID: %d, i: %zu, offset: %lld, topkLen: %d, numel: %lld\n",
                    mDecodeStep, oneReqID.c_str(), layerID, i, static_cast<long long>(topkOffset),
                    oneTopkLen, static_cast<long long>(topkView.numel));
        ClearPrefetchResultSlot(i, layerID, layout);
        return -1;
    }
    const int* topkPtr = topkView.base + topkOffset;
    std::vector<int> oneTopkList(topkPtr, topkPtr + oneTopkLen);
    std::vector<int> useBlockTable;
    if (!RunHbmTopkLookup(oneReqID, oneTopkList, layerID, static_cast<int>(i), oneTopkLen,
                          useBlockTable)) {
        ClearPrefetchResultSlot(i, layerID, layout);
        return -1;
    }
    /* 与 init 路径（ProcessTokensMapTopk 尾部同判）一致——useBlockTable 尺寸
     * 与 mSpaTopkLen 不一致属可恢复场景（短/年轻请求 sparse 窗口不足、LRU 冷
     * slot 补齐失败），清槽跳过、消费侧按零 miss 降级。调用方
     * PublishPrefetchResultsRank0 按 failedCount 汇总记账、版本照常递增，
     * rank 间等待协议完整。 */
    if (static_cast<uint32_t>(useBlockTable.size()) != mSpaTopkLen) {
        mLogger.log(LogLevel::ERROR,
                    "Decode step: %u, |KVCache Prefetch| useBlockTable size mismatch, skip: "
                    "req_id: %s, layerID: %d, mSpaTopkLen: %d, useBlockTableSize: %zu\n",
                    mDecodeStep, oneReqID.c_str(), layerID, mSpaTopkLen, useBlockTable.size());
        ClearPrefetchResultSlot(i, layerID, layout);
        return -1;
    }
    /* mAllTopkIndicesTensor 读侧与写侧（SetTopkIndices，Python 线程）统一
     * mUpdateReqMutex 协议；本调用点位于 OMP 并行域，锁缩到仅包住 find+memcpy
     * 的最小临界区。 */
    int copyRet = 0;
    {
        UReqLockGuard lock(mUpdateReqMutex, UReqLockStatsAt(UReqLockSite::ProcessPrefetchRequest));
        copyRet = CopyUseBlockTableToIndicesTensor(oneReqID, layerID, useBlockTable);
    }
    if (copyRet != 0) {
        ClearPrefetchResultSlot(i, layerID, layout);
        return -1;
    }
    WritePrefetchResult(oneReqID, i, layerID, useBlockTable, layout);
    return 0;
}

HBMManager* GSAPrefetchEngineC::FindReqLayerManager(const std::string& reqID, int layerID)
{
    /* find 判别：杜绝 map operator[] 在 OMP 失败路径上的并发插入竞态；
     * managers 与 miss/loadSlot 容器在 SetTokensMap 中同步扩容，命中即全部就绪。 */
    auto managersIt = mHBMMangers.find(reqID);
    if (managersIt == mHBMMangers.end() ||
        static_cast<size_t>(layerID) >= managersIt->second.size() ||
        !managersIt->second[static_cast<size_t>(layerID)]) {
        return nullptr;
    }
    return managersIt->second[static_cast<size_t>(layerID)].get();
}

bool GSAPrefetchEngineC::RunHbmTopkLookup(const std::string& oneReqID,
                                           std::vector<int>& oneTopkList, int layerID,
                                           int bsIndex, int oneTopkLen,
                                           std::vector<int>& useBlockTable)
{
    HBMManager* manager = FindReqLayerManager(oneReqID, layerID);
    if (manager == nullptr) {
        mLogger.log(LogLevel::ERROR,
                    "Decode step: %u, |KVCache Prefetch| HBM manager is not initialized: "
                    "req_id: %s, layerID: %d, rank: %d, bsIndex: %d\n",
                    mDecodeStep, oneReqID.c_str(), layerID, mRank, bsIndex);
        return false;
    }
    mMissGlobalToken[oneReqID][layerID].clear();
    mLoadSlot[oneReqID][layerID].clear();
    if (mOpenIndexerOffload && static_cast<size_t>(bsIndex) < mIsIndexerOffload.size() &&
        mIsIndexerOffload[static_cast<size_t>(bsIndex)]) {
        TokenIndexMapInter(oneTopkList, bsIndex, layerID, oneTopkLen, oneReqID);
    }
    manager->processTopkWithDebug(oneTopkList, useBlockTable, mMissGlobalToken[oneReqID][layerID],
                                  mLoadSlot[oneReqID][layerID], mLogger);
    if (mMissGlobalToken[oneReqID][layerID].size() != mLoadSlot[oneReqID][layerID].size() ||
        mMissGlobalToken[oneReqID][layerID].size() > static_cast<size_t>(mMaxLoadNum)) {
        KVPackAbort(mKVPackControl);
    }
    mAvgHitRate[layerID].emplace_back(manager->mHitRate);
    const float avgHitRate =
        mAvgHitRate[layerID].empty() ? 0.0f : average(mAvgHitRate[layerID]);
    mLogger.log(LogLevel::DEBUG,
                "Decode step: %u, |KVCache Prefetch| expand topk len: "
                "%lu and hit rate %f and average hit rate: %f, for layer: %d req_id: %s,  \n",
                mDecodeStep, oneTopkList.size(), manager->mHitRate, avgHitRate, layerID,
                oneReqID.c_str());
    return true;
}

void GSAPrefetchEngineC::WritePrefetchResult(const std::string& oneReqID, size_t i, int layerID,
                                             const std::vector<int>& useBlockTable,
                                             const PrefetchShmLayout& layout)
{
    size_t reqLayerOffset = (i * mTopkIndexerNum + layerID) * layout.totalSize;
    uint8_t* shmBase =
        static_cast<uint8_t*>(mPrefetchResultShm) + layout.versionSize + reqLayerOffset;
    uint32_t missTokenCount =
        std::min((size_t)mMissGlobalToken[oneReqID][layerID].size(), (size_t)layout.maxLoadNum);
    std::memcpy(shmBase, &missTokenCount, layout.countSize);

    if (missTokenCount > 0) {
        std::memcpy(shmBase + layout.countSize, mMissGlobalToken[oneReqID][layerID].data(),
                    missTokenCount * sizeof(int));
    }

    uint32_t loadSlotCount =
        std::min((size_t)mLoadSlot[oneReqID][layerID].size(), (size_t)layout.maxLoadNum);
    std::memcpy(shmBase + layout.missTokenSize, &loadSlotCount, layout.countSize);
    if (loadSlotCount > 0) {
        std::memcpy(shmBase + layout.missTokenSize + layout.countSize,
                    mLoadSlot[oneReqID][layerID].data(), loadSlotCount * sizeof(int));
    }
    std::memcpy(shmBase + layout.missTokenSize + layout.loadSlotSize, useBlockTable.data(),
                useBlockTable.size() * sizeof(int));
}

void GSAPrefetchEngineC::ClearPrefetchResultSlot(size_t i, int layerID,
                                                 const PrefetchShmLayout& layout)
{
    if (mPrefetchResultShm == nullptr) { return; }
    const size_t slotOffset = layout.versionSize +
                              (i * static_cast<size_t>(mTopkIndexerNum) +
                               static_cast<size_t>(layerID)) *
                                  layout.totalSize;
    std::memset(static_cast<uint8_t*>(mPrefetchResultShm) + slotOffset, 0, layout.totalSize);
}

void GSAPrefetchEngineC::ReadPrefetchResults(const TaskInfo& taskInfo,
                                             const PrefetchShmLayout& layout)
{
    uint32_t oneRunBsLen = taskInfo.reqIdList.size();
    for (int layerID = 0; layerID < mTopkIndexerNum; layerID++) {
        for (size_t i = 0; i < oneRunBsLen; i++) {
            if (taskInfo.topkLenList[i] <= 0) { continue; }
            std::string oneReqID = taskInfo.reqIdList[i];
            size_t reqLayerOffset = (i * mTopkIndexerNum + layerID) * layout.totalSize;
            uint8_t* shmBase =
                static_cast<uint8_t*>(mPrefetchResultShm) + layout.versionSize + reqLayerOffset;
            uint32_t missTokenCount = 0;
            memcpy(&missTokenCount, shmBase, layout.countSize);
            mMissGlobalToken[oneReqID][layerID].clear();
            if (missTokenCount > 0) {
                mMissGlobalToken[oneReqID][layerID].resize(missTokenCount);
                memcpy(mMissGlobalToken[oneReqID][layerID].data(), shmBase + layout.countSize,
                       missTokenCount * sizeof(int));
            }
            uint32_t loadSlotCount = 0;
            memcpy(&loadSlotCount, shmBase + layout.missTokenSize, layout.countSize);
            mLoadSlot[oneReqID][layerID].clear();
            if (loadSlotCount > 0) {
                mLoadSlot[oneReqID][layerID].resize(loadSlotCount);
                memcpy(mLoadSlot[oneReqID][layerID].data(),
                       shmBase + layout.missTokenSize + layout.countSize,
                       loadSlotCount * sizeof(int));
            }
            std::vector<int> useBlockTable(mSpaTopkLen);
            memcpy(useBlockTable.data(), shmBase + layout.missTokenSize + layout.loadSlotSize,
                   mSpaTopkLen * sizeof(int));
            /* 读侧统一 mUpdateReqMutex 协议（写侧 SetTopkIndices、erase 侧
             * DelReqIDRun 均持该锁），并复用 CopyUseBlockTableToIndicesTensor
             * 的 find+四重判别替代 operator[]——未注册 key 的默认插入既是并发
             * 写的结构级 UB，也是注册不变式被破坏时 torch select 抛
             * c10::Error 的确定抛点。判别失败清空本 (req,layer) 的 miss/slot，
             * 消费侧按零加载降级。 */
            int copyRet = 0;
            {
                UReqLockGuard lock(mUpdateReqMutex, UReqLockStatsAt(UReqLockSite::ReadPrefetchResults));
                copyRet = CopyUseBlockTableToIndicesTensor(oneReqID, layerID, useBlockTable);
            }
            if (copyRet != 0) {
                mMissGlobalToken[oneReqID][layerID].clear();
                mLoadSlot[oneReqID][layerID].clear();
            }
        }
    }
}

void GSAPrefetchEngineC::MergeAndCopyPrefetchLayer(const TaskInfo& taskInfo, ThreadInfo& threadInfo,
                                                   int layerID, PrefetchTiming& timing)
{
    uint32_t oneRunBsLen = taskInfo.reqIdList.size();
    mLoadNum[layerID] = 0;
    int streamIdx = layerID % mStreamNum;
    const uint64_t epoch = (uint64_t(mPrefetchResultVersion.load()) << 32) | uint64_t(layerID + 1);
    if (BeginKVPackSlot(streamIdx, epoch) != 0) { KVPackAbort(mKVPackControl); }
    std::vector<uint64_t> hbmSlotIndex;
    auto cpuStart = std::chrono::high_resolution_clock::now();
    /* Whole-batch merge: one LoadOnToken task per leg for all valid requests
     * of this layer (backend IO of all requests overlaps; the load runs on a
     * worker thread in parallel with the engine-side non-UCM token copies).
     * Shared staging capacity is still guarded per request (cumulative
     * mLoadNum vs mKVBufferLen / mLoadIndexBufferLen in ValidateUCMMergeRequest). */
    std::vector<uint32_t> requestIndices;
    requestIndices.reserve(oneRunBsLen);
    for (size_t i = 0; i < oneRunBsLen; i++) {
        if (taskInfo.topkLenList[i] > 0) {
            requestIndices.push_back(static_cast<uint32_t>(i));
        }
    }
    UCMTensorMergeBatch(taskInfo, requestIndices, layerID, streamIdx, hbmSlotIndex);
    if (FinishKVPack(streamIdx, epoch) != 0) { KVPackAbort(mKVPackControl); }
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    timing.cpuMove +=
        std::chrono::duration_cast<std::chrono::microseconds>(cpuEnd - cpuStart).count();
    CopyPrefetchToH2DBuffers(streamIdx, layerID);
    ValidateAndPublishHbmSlotIndex(streamIdx, layerID, hbmSlotIndex);
    auto memcpyEnd = std::chrono::high_resolution_clock::now();
    timing.memcpyUs +=
        std::chrono::duration_cast<std::chrono::microseconds>(memcpyEnd - cpuEnd).count();
    if (Synchronize(threadInfo.context_, static_cast<aclrtStream*>(threadInfo.stream_) + streamIdx,
                    1, mLogger) != 0) {
        KVPackAbort(mKVPackControl);
    }
    mLogger.log(LogLevel::DEBUG,
                "Decode step: %u, |KVCache Prefetch| H2D size for all request %d for layer: %d\n",
                mDecodeStep, mLoadNum[layerID], layerID);
    auto syncEnd = std::chrono::high_resolution_clock::now();
    timing.sync +=
        std::chrono::duration_cast<std::chrono::microseconds>(syncEnd - memcpyEnd).count();
    if (mLoadNum[layerID] == 0) {
        MarkKVH2DDone(streamIdx, epoch);
        return;
    }
    TransferAndScatterPrefetchLayer(threadInfo, layerID, streamIdx, timing, syncEnd);
    MarkKVH2DDone(streamIdx, epoch);
}

void GSAPrefetchEngineC::CopyPrefetchToH2DBuffers(int streamIdx, int layerID)
{
    if (mLoadNum[layerID] != 0) {
        std::memcpy(mH2DPtrInfo[streamIdx][0],
                    reinterpret_cast<const void*>(mSharedKBufferHostPtr[streamIdx]),
                    static_cast<size_t>(mLoadNum[layerID]) * mKVSizeBytesToken);
        if (!mISSPAC8) {
            std::memcpy(mH2DPtrInfo[streamIdx][1],
                        reinterpret_cast<const void*>(mSharedVBufferHostPtr[streamIdx]),
                        static_cast<size_t>(mLoadNum[layerID]) * mKVSizeBytesTokenRe);
        }
    }
}

void GSAPrefetchEngineC::ValidateAndPublishHbmSlotIndex(int streamIdx, int layerID,
                                                        std::vector<uint64_t>& hbmSlotIndex)
{
    uint32_t indexNum = hbmSlotIndex.size();
    if ((uint32_t)mLoadNum[layerID] != hbmSlotIndex.size()) {
        mLogger.log(LogLevel::ERROR,
                    "[ERROR] hbmSlotIndex is not equer to mLoadNum[layerID] !!! layer_id = %d, "
                    "hbmSlotIndex.size() = %d, mLoadNum[layerID] = %d\n",
                    layerID, hbmSlotIndex.size(), mLoadNum[layerID]);
        KVPackAbort(mKVPackControl);
    }
    if (hbmSlotIndex.size() > (uint32_t)mLoadIndexBufferLen) {
        mLogger.log(LogLevel::ERROR,
                    "[ERROR] copy is greater than bound !!! layer_id %d, "
                    "hbmSlotIndex.size()  %d, mLoadIndexBufferLen %d\n",
                    layerID, hbmSlotIndex.size(), mLoadIndexBufferLen);
        KVPackAbort(mKVPackControl);
    }
    std::memcpy(mLoadIndexHost[streamIdx].data_ptr(), hbmSlotIndex.data(),
                indexNum * sizeof(int64_t));
}

void GSAPrefetchEngineC::TransferAndScatterPrefetchLayer(ThreadInfo& threadInfo, int layerID,
                                                         int streamIdx, PrefetchTiming& timing,
                                                         PrefetchTimePoint syncEnd)
{
    size_t kSizeBytes = mLoadNum[layerID] * mKVSizeBytesToken;
    size_t vSizeBytes = mLoadNum[layerID] * mKVSizeBytesTokenRe;
    size_t indexSizeBytes = mLoadNum[layerID] * sizeof(uint64_t);
    int32_t ret = CopyTrans(static_cast<void**>(&mH2DPtrInfo[streamIdx][0]),
                            static_cast<void**>(&mH2DPtrInfo[streamIdx][2]), kSizeBytes, 1, "h2d",
                            static_cast<aclrtStream*>(threadInfo.stream_) + streamIdx);
    if (ret == -1) {
        mLogger.log(LogLevel::ERROR, "[ERROR] mKBufferDevice CopyTrans H2D ERROR !!!\n");
        KVPackAbort(mKVPackControl);
    }
    if (!mISSPAC8) {
        ret = CopyTrans(static_cast<void**>(&mH2DPtrInfo[streamIdx][1]),
                        static_cast<void**>(&mH2DPtrInfo[streamIdx][3]), vSizeBytes, 1, "h2d",
                        static_cast<aclrtStream*>(threadInfo.stream_) + streamIdx);
        if (ret == -1) {
            mLogger.log(LogLevel::ERROR, "[ERROR] mVBufferDevice CopyTrans H2D ERROR !!!\n");
            KVPackAbort(mKVPackControl);
        }
    }
    ret = CopyTrans(static_cast<void**>(&mH2DPtrInfo[streamIdx][4]),
                    static_cast<void**>(&mH2DPtrInfo[streamIdx][5]), indexSizeBytes, 1, "h2d",
                    static_cast<aclrtStream*>(threadInfo.stream_) + streamIdx);
    if (ret == -1) {
        mLogger.log(LogLevel::ERROR, "[ERROR] mLoadIndexDevice CopyTrans H2D ERROR !!!\n");
        KVPackAbort(mKVPackControl);
    }
    if (Synchronize(threadInfo.context_, static_cast<aclrtStream*>(threadInfo.stream_) + streamIdx,
                    1, mLogger) != 0) {
        KVPackAbort(mKVPackControl);
    }
    auto h2dEnd = std::chrono::high_resolution_clock::now();
    timing.h2d += std::chrono::duration_cast<std::chrono::microseconds>(h2dEnd - syncEnd).count();
    KVCacheInplaceScatter("K", layerID, static_cast<aclrtStream*>(threadInfo.stream_)[streamIdx],
                          *static_cast<aclrtContext*>(threadInfo.context_));
    if (!mISSPAC8) {
        KVCacheInplaceScatter("V", layerID,
                              static_cast<aclrtStream*>(threadInfo.stream_)[streamIdx],
                              *static_cast<aclrtContext*>(threadInfo.context_));
    }
    auto opsEnd = std::chrono::high_resolution_clock::now();
    timing.ops += std::chrono::duration_cast<std::chrono::microseconds>(opsEnd - h2dEnd).count();
}

int GSAPrefetchEngineC::IndexerLoadProcess(const TaskInfo& taskInfo, ThreadInfo& threadInfo)
{
    if (!mOpenIndexerOffload) { return 0; }
    UReqLockGuard lock(mUpdateReqMutex, UReqLockStatsAt(UReqLockSite::IndexerLoadProcess));
    auto start = std::chrono::high_resolution_clock::now();
    uint32_t runBsLen = taskInfo.reqIdList.size();
    int ret = 0;
    allNeedLoadBlock.clear();
    allMissIdxs.clear();
    for (size_t i = 0; i < runBsLen; i++) {
        if (mDocsTables.find(taskInfo.reqIdList[i]) == mDocsTables.end() ||
            taskInfo.topkLenList[i] <= 0) {
            mLogger.log(LogLevel::ERROR,
                        "Decode step: %u, |KVCache Prefetch| topk len is zero: %d\n", mDecodeStep,
                        taskInfo.topkLenList[i]);
            continue;
        }
        allMissIdxs.insert({taskInfo.reqIdList[i], std::vector<std::vector<int>>(mLayerNum)});
        allNeedLoadBlock.insert({taskInfo.reqIdList[i], std::vector<std::vector<int>>(mLayerNum)});
        // 判别式：单请求 topk 张量非法（已打日志）时记失败返回码并跳过，其余请求继续。
        if (RunOneBsPrefetch(taskInfo.reqIdList[i], taskInfo.topkLenList[i], i,
                             taskInfo.indexerTopkList[i]) != 0) {
            ret = -1;
            continue;
        }
    }
    UCMIndexerLoadAsync(threadInfo.context_, threadInfo.stream_);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    mLogger.log(LogLevel::INFO,
                "Decode step: %u, |KVCache Prefetch| Finish indexer load cost: %lu\n", mDecodeStep,
                duration.count());
    return ret;
}

void GSAPrefetchEngineC::DealLessPrefetchBlock(std::unordered_set<int>& hitBlocks,
                                               std::map<int, int>& hitBlocksIdx, std::string reqID,
                                               int layerID, int topkLen)
{
    if (!mOpenIndexerOffload) { return; }
    for (auto it = mDocsTables[reqID][layerID].rbegin(); it != mDocsTables[reqID][layerID].rend();
         ++it) {
        if (hitBlocks.find(it->second) == hitBlocks.end() && hitBlocks.size() < (uint32_t)topkLen) {
            hitBlocks.insert(it->second);
            hitBlocksIdx.insert(std::make_pair(it->first, it->second));
        }
    }
    mLogger.log(LogLevel::DEBUG,
                "Decode step: %u, |KVCache Prefetch| hitBlocks after deal len: %d\n", mDecodeStep,
                hitBlocks.size());
}

int GSAPrefetchEngineC::RunOneBsPrefetch(std::string reqID, int topkLen, int topkIndex,
                                         const torch::Tensor& topkList)
{
    if (!mOpenIndexerOffload) { return 0; }
    /* 提前识别：进入并行域前完成张量判别并一次性提取只读视图（含读取
     * 范围校验），OMP 域内不再触碰任何 torch API（消除 data_ptr dtype
     * 抛异常、下标越界抛 IndexError 与张量切片堆分配三类异常/终止源）。 */
    PrefetchTopkView topkView;
    if (AcquireIndexerTopkBase(topkList, reqID, topkLen, topkView) != 0) {
        return -1;
    }
#pragma omp parallel for num_threads(4) proc_bind(master)
    for (int i = 0; i < mLayerNum; i++) {
        std::unordered_set<int> hitBlocks;
        std::map<int, int> hitBlocksIdx;
        std::vector<int> missIdxs;
        PrefetchReqInfo oneBsInfo;
        oneBsInfo.topkLen = topkLen;
        oneBsInfo.reqID = reqID;
        oneBsInfo.layerID = i;
        const int* topk_ptr = topkView.base + static_cast<int64_t>(i) * topkView.stride0;
        std::vector<int> oneTopkList(topk_ptr, topk_ptr + topkLen);
        GetHitAndMissBlock(oneBsInfo, oneTopkList, hitBlocks, hitBlocksIdx, missIdxs);
        if (missIdxs.size() != 0 && hitBlocksIdx.size() < (uint32_t)topkLen) {
            RunPrefetchH2D(oneBsInfo, hitBlocks, hitBlocksIdx, missIdxs);
        }
        if (hitBlocksIdx.size() != (uint32_t)topkLen) {
            mLogger.log(LogLevel::DEBUG,
                        "Decode step: %u, "
                        "|KVCache Prefetch| successIndex size is error!!! "
                        "mPrefetchIdx[reqID][layerID] size: %lu mDocsTables[reqID][i] size: %lu, "
                        "hitBlocksIdx size: %lu, "
                        "mBlocksMap[reqID][layerID]: %lu, "
                        "missIdxs size: %lu\n",
                        mDecodeStep, mPrefetchIdx[reqID][i].size(), mDocsTables[reqID][i].size(),
                        hitBlocksIdx.size(), mBlocksMap[reqID][i].size(), missIdxs.size());
            DealLessPrefetchBlock(hitBlocks, hitBlocksIdx, reqID, i, topkLen);
        }
        FinishOneLayerPrefetch(reqID, i, hitBlocksIdx, missIdxs);
    }
    return 0;
}

int GSAPrefetchEngineC::AcquireIndexerTopkBase(const torch::Tensor& topkList,
                                               const std::string& reqID, int topkLen,
                                               PrefetchTopkView& topkView)
{
    const bool topkDefined = topkList.defined();
    const int64_t topkDim = topkDefined ? topkList.dim() : 0;
    bool topkOk = topkDefined && topkDim >= 1 &&
                  static_cast<int64_t>(topkList.size(0)) >= static_cast<int64_t>(mLayerNum) &&
                  topkList.scalar_type() == torch::kInt;
    if (topkOk) {
        /* 平坦读边界：末层末行读取范围不得超过张量元素数（防 OOB 读）。 */
        const int64_t stride0 = topkList.stride(0);
        topkOk = topkLen >= 0 && static_cast<int64_t>(mLayerNum) - 1 >= 0 &&
                 (static_cast<int64_t>(mLayerNum) - 1) * stride0 +
                         static_cast<int64_t>(topkLen) <=
                     topkList.numel();
    }
    if (!topkOk) {
        mLogger.log(LogLevel::ERROR,
                    "Decode step: %u, |KVCache Prefetch| invalid indexer topkList tensor: "
                    "reqID=%s defined=%d dim=%lld size0=%lld dtype=%d layerNum=%d topkLen=%d\n",
                    mDecodeStep, reqID.c_str(), topkDefined ? 1 : 0, static_cast<long long>(topkDim),
                    topkDim >= 1 ? static_cast<long long>(topkList.size(0)) : -1LL,
                    topkDefined ? static_cast<int>(topkList.scalar_type()) : -1, mLayerNum,
                    topkLen);
        std::cerr << "|KVCache Prefetch| [ERROR] RunOneBsPrefetch: invalid indexer topkList, "
                     "layer run skipped, reqID="
                  << reqID << std::endl;
        return -1;
    }
    topkView.base = topkList.data_ptr<int>();
    topkView.stride0 = topkList.stride(0);
    topkView.numel = topkList.numel();
    return 0;
}

void GSAPrefetchEngineC::FinishOneLayerPrefetch(const std::string& reqID, int layerID,
                                                std::map<int, int>& hitBlocksIdx,
                                                std::vector<int>& missIdxs)
{
    int successIndex = 0;
    std::vector<int> blockTableResult;
    std::vector<std::pair<int, int>> sortResult(hitBlocksIdx.size());
    for (auto it = hitBlocksIdx.begin(); it != hitBlocksIdx.end(); it++) {
        sortResult[successIndex] = std::make_pair(it->first, it->second);
        successIndex += 1;
    }
    std::sort(sortResult.begin(), sortResult.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    int32_t pairIdx = 0;
    for (auto onePair : sortResult) {
        blockTableResult.emplace_back(onePair.second);
        mPrefetchBlockTableIdx[reqID][layerID][pairIdx] = onePair.first;
        pairIdx += 1;
    }
    LoadKVToHBM(allNeedLoadBlock[reqID][layerID], missIdxs, layerID, reqID);
    std::memcpy(mIndexerLoadResultT[reqID][layerID].data_ptr(), blockTableResult.data(),
                successIndex * sizeof(int));
    mPrefetchIdx[reqID][layerID].clear();
    for (auto it = mDocsTables[reqID][layerID].begin(); it != mDocsTables[reqID][layerID].end();
         it++) {
        if (hitBlocksIdx.find(it->first) != hitBlocksIdx.end()) {
            continue;
        } else {
            mPrefetchIdx[reqID][layerID].push_back(it->first);
        }
    }
}

void GSAPrefetchEngineC::GetHitAndMissBlock(PrefetchReqInfo oneBsInfo, std::vector<int>& topkList,
                                            std::unordered_set<int>& hitBlocks,
                                            std::map<int, int>& hitBlocksIdx,
                                            std::vector<int>& missIdxs)
{
    if (!mOpenIndexerOffload) { return; }
    int topkLen = oneBsInfo.topkLen;
    int layerID = oneBsInfo.layerID;
    std::string reqID = oneBsInfo.reqID;

    for (int j = 0; j < topkLen; j++) {
        int32_t item = topkList[j];
        if (mDocsTables[reqID][layerID].find(item) != mDocsTables[reqID][layerID].end()) {
            int blockID = mDocsTables[reqID][layerID][item];
            hitBlocks.insert(blockID);
            hitBlocksIdx.insert(std::make_pair(item, blockID));
        } else {
            missIdxs.push_back(item);
        }
    }
    if ((hitBlocks.size() + missIdxs.size()) != (uint32_t)topkLen) {
        mLogger.log(LogLevel::ERROR,
                    "|KVCache Prefetch| Decode step: %u, Rank: %d, reqID: %s, layer: %d, hit size: "
                    "%lu, miss size: %lu , topkLen: %d, not equal error\n",
                    mDecodeStep, mRank, reqID.c_str(), layerID, hitBlocks.size(), missIdxs.size(),
                    topkLen);
    }
}

void GSAPrefetchEngineC::RunPrefetchH2D(PrefetchReqInfo oneBsInfo,
                                        std::unordered_set<int>& hitBlocks,
                                        std::map<int, int>& hitBlocksIdx,
                                        std::vector<int>& missIdxs)
{
    if (!mOpenIndexerOffload) { return; }
    int layerID = oneBsInfo.layerID;
    std::string reqID = oneBsInfo.reqID;

    int oneFreeBlockLen = mPrefetchIdx[reqID][layerID].size();
    std::vector<int> oneFreeBlockTable;

    uint32_t index = 0;
    std::vector<int> onePrefetchIdx = mPrefetchIdx[reqID][layerID];
    int oneFreeBlockIndex = 0;
    while (oneFreeBlockIndex < oneFreeBlockLen && index < missIdxs.size()) {
        int oneFreeBlockID = mDocsTables[reqID][layerID][onePrefetchIdx[oneFreeBlockIndex]];
        if (hitBlocks.find(oneFreeBlockID) != hitBlocks.end()) {
            oneFreeBlockIndex += 1;
            continue;
        } else {
            oneFreeBlockTable.push_back(oneFreeBlockID);
            hitBlocks.insert(oneFreeBlockID);
            hitBlocksIdx.insert(std::make_pair(missIdxs[index], oneFreeBlockID));
            index += 1;
            oneFreeBlockIndex += 1;
        }
    }
    uint32_t loadLen = oneFreeBlockTable.size();
    missIdxs.erase(missIdxs.begin() + loadLen, missIdxs.end());
    allNeedLoadBlock[reqID][layerID] = oneFreeBlockTable;
    allMissIdxs[reqID][layerID] = missIdxs;
}

void GSAPrefetchEngineC::LoadKVToHBM(std::vector<int> loadNPUBlockIDs, std::vector<int> missIdxs,
                                     int layerID, std::string reqID)
{
    for (size_t i = 0; i < loadNPUBlockIDs.size(); i++) {
        int oriIdx = mBlocksMap[reqID][layerID][loadNPUBlockIDs[i]];
        mBlocksMap[reqID][layerID][loadNPUBlockIDs[i]] = missIdxs[i];
        mDocsTables[reqID][layerID].erase(oriIdx);
        mDocsTables[reqID][layerID][missIdxs[i]] = loadNPUBlockIDs[i];
    }
}

void GSAPrefetchEngineC::IndexerInitLoad(std::vector<std::vector<int>>& blockTable,
                                         std::vector<std::vector<int>>& slabIdx, std::string reqID)
{
    if (!mOpenIndexerOffload) { return; }
    /* 本函数子树读 mAllBlocksHash（find 于 IndexerInitLoad /
     * ProcessIndexerInitLoadLayer）与 mLastKVBlockAddress（LoadNonUcmIndexerBlock 一带，
     * 引用跨函数全程持有），调用点（execute_finished 在 _submit_prefetch_task 之后）
     * 与 prefetch worker 清理段 erase（DelReqIDRun，持 mUpdateReqMutex）并发，读者
     * 补锁对称。一次性 init 路径，非解码热路径。 */
    UReqLockGuard lock(mUpdateReqMutex, UReqLockStatsAt(UReqLockSite::IndexerInitLoad));

    mLogger.log(LogLevel::INFO, "Decode step: %u, |KVCache Prefetch| Start IndexerInitLoad\n",
                mDecodeStep);
    auto start = std::chrono::high_resolution_clock::now();
    int number = blockTable[0].size();

    auto ret = aclrtSetCurrentContext(*(static_cast<aclrtContext*>(mContext)));
    if (ret != ACL_SUCCESS) {
        const char* errMsg = aclGetRecentErrMsg();
        mLogger.log(LogLevel::ERROR,
                    "|KVCache Prefetch|  aclrtSetCurrentContext Error:"
                    " %d, Error: %s\n",
                    ret, errMsg ? errMsg : "Unknown error");
        return;
    }

    auto hashIt = mAllBlocksHash.find(reqID);
    if (hashIt == mAllBlocksHash.end() || hashIt->second.empty()) {
        mLogger.log(LogLevel::ERROR, "[ERROR] IndexerInitLoad no hash, reqID=%s\n", reqID.c_str());
        return;
    }

    for (int layerID = 0; layerID < mLayerNum; layerID++) {
        if (!ProcessIndexerInitLoadLayer(blockTable, slabIdx, reqID, layerID, number)) { return; }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    mLogger.log(LogLevel::INFO,
                "Decode step: %u, |KVCache Prefetch| End IndexerInitLoad  load cost: %lu\n",
                mDecodeStep, static_cast<unsigned long>(duration.count()));
}

bool GSAPrefetchEngineC::ProcessIndexerInitLoadLayer(std::vector<std::vector<int>>& blockTable,
                                                     std::vector<std::vector<int>>& slabIdx,
                                                     const std::string& reqID, int layerID,
                                                     int number)
{
    if (static_cast<size_t>(layerID) >= mIndexerPtr.size() ||
        static_cast<size_t>(layerID) >= blockTable.size() ||
        static_cast<size_t>(layerID) >= slabIdx.size()) {
        mLogger.log(LogLevel::ERROR,
                    "[ERROR] IndexerInitLoad layerID out of range: reqID=%s layerID=%d "
                    "indexerPtrSize=%zu blockTableSize=%zu slabIdxSize=%zu\n",
                    reqID.c_str(), layerID, mIndexerPtr.size(), blockTable.size(),
                    slabIdx.size());
        return false;
    }
    auto* indexerBaseAddr = reinterpret_cast<std::uint8_t*>(mIndexerPtr[layerID]);
    if (indexerBaseAddr == nullptr) {
        mLogger.log(LogLevel::ERROR,
                    "[ERROR] IndexerInitLoad indexer base address is nullptr, layerID=%d\n",
                    layerID);
        return false;
    }
    auto hashIt = mAllBlocksHash.find(reqID);
    if (hashIt == mAllBlocksHash.end()) {
        mLogger.log(LogLevel::ERROR,
                    "[ERROR] IndexerInitLoad reqID not found in mAllBlocksHash: "
                    "reqID=%s layerID=%d\n",
                    reqID.c_str(), layerID);
        return false;
    }
    UC::Detail::TaskDesc task;
    task.reserve(number);
    task.brief = "UCMIndexerInitLoad";
    if (!BuildIndexerInitLoadShards(blockTable, slabIdx, reqID, layerID, number, indexerBaseAddr,
                                    hashIt->second, task)) {
        return false;
    }
    return SubmitIndexerInitLoadTask(task, reqID, layerID);
}

bool GSAPrefetchEngineC::BuildIndexerInitLoadShards(std::vector<std::vector<int>>& blockTable,
                                                    std::vector<std::vector<int>>& slabIdx,
                                                    const std::string& reqID, int layerID,
                                                    int number, std::uint8_t* indexerBaseAddr,
                                                    const std::vector<BlockId>& hashVec,
                                                    UC::Detail::TaskDesc& task)
{
    size_t shardIndex = static_cast<size_t>(layerID);
    for (int i = 0; i < number; i++) {
        if (static_cast<size_t>(i) >= blockTable[layerID].size() ||
            static_cast<size_t>(i) >= slabIdx[layerID].size()) {
            mLogger.log(LogLevel::ERROR,
                        "[ERROR] IndexerInitLoad row too short: reqID=%s layerID=%d i=%d "
                        "blockRow=%zu slabRow=%zu\n",
                        reqID.c_str(), layerID, i, blockTable[layerID].size(), slabIdx[layerID].size());
            continue;
        }
        int deviceBlockId = blockTable[layerID][i];
        int blockIdx = slabIdx[layerID][i];
        if (deviceBlockId < 0 || deviceBlockId >= mDeviceIndexerLen) {
            mLogger.log(LogLevel::ERROR,
                        "[ERROR] IndexerInitLoad deviceBlockId out of range, "
                        "reqID=%s layerID=%d deviceBlockId=%d mDeviceIndexerLen=%d\n",
                        reqID.c_str(), layerID, deviceBlockId, mDeviceIndexerLen);
            continue;
        }
        auto* indexerAddr =
            indexerBaseAddr + static_cast<size_t>(deviceBlockId) * mIndexerSizeBytesBlock;
        if (blockIdx >= static_cast<int>(hashVec.size())) {
            if (!LoadNonUcmIndexerBlock(reqID, layerID, blockIdx, deviceBlockId, indexerAddr)) {
                return false;
            }
            continue;
        }
        if (blockIdx < 0) {
            mLogger.log(LogLevel::ERROR,
                        "[ERROR] IndexerInitLoad blockIdx out of range, "
                        "reqID=%s layerID=%d blockIdx=%d hashNum=%zu\n",
                        reqID.c_str(), layerID, blockIdx, hashVec.size());
            continue;
        }
        /* One address per store tensor slot: the store's H2D path reads addrs[i]
         * for every slot, so pad to the full count with the indexer address at
         * kIndexerTensorIndex (ghost slots stay nullptr). */
        std::vector<void*> tensorAddrs(kStoreTensorCount, nullptr);
        tensorAddrs[kIndexerTensorIndex] = indexerAddr;
        task.emplace_back(UC::Detail::Shard{hashVec[static_cast<size_t>(blockIdx)], shardIndex,
                                            std::move(tensorAddrs)});
    }
    return true;
}

bool GSAPrefetchEngineC::SubmitIndexerInitLoadTask(UC::Detail::TaskDesc& task,
                                                   const std::string& reqID, int layerID)
{
    if (task.empty()) { return true; }
    if (!mStoreV1_) {
        mLogger.log(LogLevel::ERROR,
                    "[ERROR] IndexerInitLoad mStoreV1_ is nullptr, reqID=%s layerID=%d\n",
                    reqID.c_str(), layerID);
        return false;
    }
    const size_t loadNumber = task.size();
    auto loadResult = mStoreV1_->Load(std::move(task));
    if (!loadResult) {
        const std::string error = loadResult.Error().ToString();
        mLogger.log(LogLevel::ERROR,
                    "[ERROR] IndexerInitLoad Load failed, "
                    "reqID=%s layerID=%d loadNumber=%zu status=%s\n",
                    reqID.c_str(), layerID, loadNumber, error.c_str());
        return false;
    }
    auto waitStatus = mStoreV1_->Wait(loadResult.Value());
    if (waitStatus.Failure()) {
        const std::string error = waitStatus.ToString();
        mLogger.log(LogLevel::ERROR,
                    "[ERROR] IndexerInitLoad Wait failed, "
                    "reqID=%s layerID=%d loadNumber=%zu status=%s\n",
                    reqID.c_str(), layerID, loadNumber, error.c_str());
        return false;
    }
    return true;
}

bool GSAPrefetchEngineC::LoadNonUcmIndexerBlock(const std::string& reqID, int layerID, int blockIdx,
                                                int deviceBlockId, std::uint8_t* indexerAddr)
{
    auto lastIt = mLastKVBlockAddress.find(reqID);
    if (lastIt == mLastKVBlockAddress.end()) {
        mLogger.log(LogLevel::ERROR, "|IndexerInitLoad| IndexerInitLoad no non-UCM block map \n");
        return false;
    }

    auto blockAddressIt = lastIt->second.find(blockIdx);
    if (blockAddressIt == lastIt->second.end()) {
        mLogger.log(LogLevel::ERROR,
                    "|IndexerInitLoad| IndexerInitLoad non-UCM block not found \n");
        return false;
    }

    const auto& blockAddress = blockAddressIt->second;
    const auto* indexerBase = reinterpret_cast<const std::uint8_t*>(blockAddress.indexerAddr);
    const void* lastIndexerAddr = static_cast<const void*>(
        indexerBase + static_cast<size_t>(layerID) * mIndexerSizeBytesBlock);

    auto copyRet = aclrtMemcpy(indexerAddr, mIndexerSizeBytesBlock, lastIndexerAddr,
                               mIndexerSizeBytesBlock, ACL_MEMCPY_HOST_TO_DEVICE);
    if (copyRet != ACL_SUCCESS) {
        const char* errMsg = aclGetRecentErrMsg();
        mLogger.log(LogLevel::ERROR,
                    "[ERROR] IndexerInitLoad last block memcpy failed, "
                    "reqID=%s layerID=%d blockIdx=%d deviceBlockId=%d "
                    "ret=%d error=%s \n",
                    reqID.c_str(), layerID, blockIdx, deviceBlockId, copyRet,
                    errMsg ? errMsg : "Unknown error");
        return false;
    }

    return true;
}

void GSAPrefetchEngineC::UCMIndexerLoadAsync(void* context, void* stream)
{
    (void)stream;
    mLogger.log(LogLevel::INFO, "Decode step: %u, |KVCache Prefetch| Start UCMIndexerLoadAsync\n",
                mDecodeStep);
    auto start = std::chrono::high_resolution_clock::now();

    if (context == nullptr) {
        mLogger.log(LogLevel::ERROR, "[ERROR] UCMIndexerLoadAsync context is nullptr\n");
        return;
    }

    auto ret = aclrtSetCurrentContext(*(static_cast<aclrtContext*>(context)));
    if (ret != ACL_SUCCESS) {
        const char* errMsg = aclGetRecentErrMsg();
        mLogger.log(LogLevel::ERROR,
                    "|KVCache Prefetch| aclrtSetCurrentContext Error: %d, Error: %s\n", ret,
                    errMsg ? errMsg : "Unknown error");
        return;
    }

    size_t totalNumber = 0;
    for (int layerID = 0; layerID < mLayerNum; layerID++) {
        if (!LoadIndexerLayer(layerID, totalNumber)) { return; }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    mLogger.log(LogLevel::INFO,
                "Decode step: %u, |KVCache Prefetch| End UCMIndexerLoadAsync "
                "blockNum=%zu load cost: %lu us\n",
                mDecodeStep, totalNumber, static_cast<unsigned long>(duration.count()));
}

bool GSAPrefetchEngineC::LoadIndexerLayer(int layerID, size_t& totalNumber)
{
    auto* indexerBaseAddr = reinterpret_cast<std::uint8_t*>(mIndexerPtr[layerID]);
    if (indexerBaseAddr == nullptr) {
        mLogger.log(LogLevel::ERROR,
                    "[ERROR] UCMIndexerLoadAsync indexer base address is nullptr, "
                    "layerID=%d\n",
                    layerID);
        return false;
    }
    for (auto it = allNeedLoadBlock.begin(); it != allNeedLoadBlock.end(); ++it) {
        const std::string& reqID = it->first;
        const auto& hbmBlockIDs = it->second[layerID];
        if (!LoadIndexerRequest(reqID, hbmBlockIDs, layerID, indexerBaseAddr, totalNumber)) {
            return false;
        }
    }
    return true;
}

bool GSAPrefetchEngineC::LoadIndexerRequest(const std::string& reqID,
                                            const std::vector<int>& hbmBlockIDs, int layerID,
                                            std::uint8_t* indexerBaseAddr, size_t& totalNumber)
{
    int number = static_cast<int>(hbmBlockIDs.size());
    if (number == 0) { return true; }
    auto hashIt = mAllBlocksHash.find(reqID);
    if (hashIt == mAllBlocksHash.end()) {
        mLogger.log(LogLevel::ERROR,
                    "[ERROR] UCMIndexerLoadAsync no hash, "
                    "reqID=%s layerID=%d\n",
                    reqID.c_str(), layerID);
        return false;
    }
    auto missIt = allMissIdxs.find(reqID);
    if (missIt == allMissIdxs.end()) {
        mLogger.log(LogLevel::ERROR,
                    "[ERROR] UCMIndexerLoadAsync no miss indexes, "
                    "reqID=%s layerID=%d\n",
                    reqID.c_str(), layerID);
        return false;
    }
    const auto& hashVec = hashIt->second;
    const auto& missBlockIdxs = missIt->second[layerID];
    if (hbmBlockIDs.size() != missBlockIdxs.size()) {
        mLogger.log(LogLevel::ERROR,
                    "[ERROR] UCMIndexerLoadAsync size mismatch, "
                    "reqID=%s layerID=%d hbmNum=%zu missNum=%zu\n",
                    reqID.c_str(), layerID, hbmBlockIDs.size(), missBlockIdxs.size());
        return false;
    }

    UC::Detail::TaskDesc task;
    task.reserve(number);
    task.brief = "UCMIndexerLoad";
    if (!BuildIndexerLoadTask(reqID, layerID, indexerBaseAddr, hbmBlockIDs, missBlockIdxs, hashVec,
                              task, totalNumber)) {
        return false;
    }

    if (task.empty()) { return true; }
    return LoadIndexerTask(reqID, layerID, task, totalNumber);
}

bool GSAPrefetchEngineC::BuildIndexerLoadTask(const std::string& reqID, int layerID,
                                              std::uint8_t* indexerBaseAddr,
                                              const std::vector<int>& hbmBlockIDs,
                                              const std::vector<int>& missBlockIdxs,
                                              const std::vector<BlockId>& hashVec,
                                              UC::Detail::TaskDesc& task, size_t& totalNumber)
{
    const int lastUCMBlockIdx = static_cast<int>(hashVec.size());
    size_t shardIndex = static_cast<size_t>(layerID);

    for (size_t i = 0; i < hbmBlockIDs.size(); i++) {
        int hbmBlockID = hbmBlockIDs[i];
        int missBlockIdx = missBlockIdxs[i];
        void* indexerAddr = static_cast<void*>(indexerBaseAddr + static_cast<size_t>(hbmBlockID) *
                                                                     mIndexerSizeBytesBlock);

        if (missBlockIdx >= lastUCMBlockIdx) {
            if (!CopyNonUCMIndexerBlock(reqID, layerID, missBlockIdx, hbmBlockID, indexerAddr,
                                        totalNumber)) {
                return false;
            }
            continue;
        }

        if (missBlockIdx < 0 || static_cast<size_t>(missBlockIdx) >= hashVec.size()) {
            mLogger.log(LogLevel::ERROR,
                        "[ERROR] UCMIndexerLoadAsync missBlockIdx out of range, "
                        "reqID=%s layerID=%d missBlockIdx=%d, hashNum=%zu\n",
                        reqID.c_str(), layerID, missBlockIdx, hashVec.size());
            continue;
        }
        /* Same slot-padded addrs contract as the init load: the store's H2D
         * path indexes addrs[i] for every tensorSizes_ slot. */
        std::vector<void*> tensorAddrs(kStoreTensorCount, nullptr);
        tensorAddrs[kIndexerTensorIndex] = indexerAddr;
        task.emplace_back(UC::Detail::Shard{hashVec[missBlockIdx], shardIndex,
                                            std::move(tensorAddrs)});
    }
    return true;
}

bool GSAPrefetchEngineC::CopyNonUCMIndexerBlock(const std::string& reqID, int layerID,
                                                int missBlockIdx, int hbmBlockID, void* indexerAddr,
                                                size_t& totalNumber)
{
    auto lastIt = mLastKVBlockAddress.find(reqID);
    if (lastIt == mLastKVBlockAddress.end()) {
        mLogger.log(LogLevel::ERROR, "|IndexerInitLoad| IndexerInitLoad no non-UCM block map \n");
        return false;
    }
    auto blockAddressIt = lastIt->second.find(missBlockIdx);
    if (blockAddressIt == lastIt->second.end()) {
        mLogger.log(LogLevel::ERROR, "|IndexerInitLoad| IndexerInitLoad no non-UCM block map \n");
        return false;
    }
    const auto& blockAddress = blockAddressIt->second;
    const auto* indexerBase = reinterpret_cast<const std::uint8_t*>(blockAddress.indexerAddr);
    const void* lastIndexerAddr = static_cast<const void*>(
        indexerBase + static_cast<size_t>(layerID) * mIndexerSizeBytesBlock);
    auto copyRet = aclrtMemcpy(indexerAddr, mIndexerSizeBytesBlock, lastIndexerAddr,
                               mIndexerSizeBytesBlock, ACL_MEMCPY_HOST_TO_DEVICE);
    if (copyRet != ACL_SUCCESS) {
        const char* errMsg = aclGetRecentErrMsg();
        mLogger.log(LogLevel::ERROR,
                    "[ERROR] UCMIndexerLoadAsync last block memcpy failed, "
                    "reqID=%s layerID=%d missBlockIdx=%d hbmBlockID=%d "
                    "ret=%d error=%s\n",
                    reqID.c_str(), layerID, missBlockIdx, hbmBlockID, copyRet,
                    errMsg ? errMsg : "Unknown error");
        return false;
    }
    totalNumber++;
    return true;
}

bool GSAPrefetchEngineC::LoadIndexerTask(const std::string& reqID, int layerID,
                                         UC::Detail::TaskDesc& task, size_t& totalNumber)
{
    if (!mStoreV1_) {
        mLogger.log(LogLevel::ERROR,
                    "[ERROR] UCMIndexerLoadAsync mStoreV1_ is nullptr, reqID=%s layerID=%d\n",
                    reqID.c_str(), layerID);
        return false;
    }
    const size_t loadNumber = task.size();
    auto loadResult = mStoreV1_->Load(std::move(task));
    if (!loadResult) {
        const std::string error = loadResult.Error().ToString();
        mLogger.log(LogLevel::ERROR,
                    "[ERROR] UCMIndexerLoadAsync Load failed, "
                    "reqID=%s layerID=%d loadNumber=%zu status=%s\n",
                    reqID.c_str(), layerID, loadNumber, error.c_str());
        return false;
    }
    auto waitStatus = mStoreV1_->Wait(loadResult.Value());
    if (waitStatus.Failure()) {
        const std::string error = waitStatus.ToString();
        mLogger.log(LogLevel::ERROR,
                    "[ERROR] UCMIndexerLoadAsync Wait failed, "
                    "reqID=%s layerID=%d loadNumber=%zu status=%s\n",
                    reqID.c_str(), layerID, loadNumber, error.c_str());
        return false;
    }
    totalNumber += loadNumber;
    return true;
}

void GSAPrefetchEngineC::SetKvTensor(torch::Tensor kCachesTensor, torch::Tensor vCachesTensor,
                                     torch::Tensor lastChunkKBuffer, torch::Tensor lastChunkvBuffer,
                                     int layerID)
{
    mLogger.log(LogLevel::DEBUG,
                "Decode step: %u, |KVCache Prefetch| Start SetKvTensor layerID: %u\n", mDecodeStep,
                layerID);

    // init token an block tensor size
    if (mKVHBMBlockLen == 0) { mKVHBMBlockLen = kCachesTensor.sizes()[0]; }
    mKcacheTensor[layerID] = kCachesTensor;
    mKcachePtr[layerID] = reinterpret_cast<uint64_t>(kCachesTensor.data_ptr());
    at::Tensor kCachesTensorND = kCachesTensor.view({-1, mKDim});
    mKCacheAcl[layerID] = AtTensorToAclTensorShared(kCachesTensorND);
    mLastChunkNum = mSparseLen / mBlockSize;
    mLastCHunkKBuffer[layerID] = lastChunkKBuffer;
    mLastCHunkKPtr[layerID] = reinterpret_cast<uint64_t>(mLastCHunkKBuffer[layerID].data_ptr());
    if (!mISSPAC8) {
        mVcacheTensor[layerID] = vCachesTensor;
        mVcachePtr[layerID] = reinterpret_cast<uint64_t>(vCachesTensor.data_ptr());
        at::Tensor vCachesTensorND = vCachesTensor.view({-1, mVDim});
        mVCacheAcl[layerID] = AtTensorToAclTensorShared(vCachesTensorND);
        mLastCHunkVBuffer[layerID] = lastChunkvBuffer;
        mLastCHunkVPtr[layerID] = reinterpret_cast<uint64_t>(mLastCHunkVBuffer[layerID].data_ptr());
    }
    mLogger.log(LogLevel::INFO, "Decode step: %u, |KVCache Prefetch| SetKvTensor for layer: %d\n",
                mDecodeStep, layerID);
}

void GSAPrefetchEngineC::SetKvTensorIndex(int layerID, torch::Tensor indexCache)
{
    mLogger.log(LogLevel::DEBUG,
                "Decode step: %u, |KVCache Prefetch| Start SetKvTensorIndex layerID: %u\n",
                mDecodeStep, layerID);
    if (!mOpenIndexerOffload) { return; }
    if (mDeviceIndexerLen == 0) {
        mDeviceIndexerLen = indexCache.sizes()[0];
        mIndexerSizeBytesBlock = indexCache[0].element_size() * indexCache[0].numel();
        mLogger.log(LogLevel::INFO,
                    "Decode step: %u, "
                    "|KVCache Prefetch| init indexer kvcache, mDeviceIndexerLen: %d, "
                    "mIndexerSizeBytesBlock: %d\n",
                    mDecodeStep, mDeviceIndexerLen, mIndexerSizeBytesBlock);
    }
    mIndexerTensor[layerID] = indexCache;
    mIndexerPtr[layerID] = reinterpret_cast<uint64_t>(indexCache.data_ptr());
    mLogger.log(LogLevel::INFO,
                "Decode step: %u, |KVCache Prefetch| SetKvTensorIndex for layer: %d\n", mDecodeStep,
                layerID);
}

enum class H2DInfoIndex {
    KBufferHost = 0,
    VBufferHost,
    KBufferDevice,
    VBufferDevice,
    LoadIndexHost,
    LoadIndexDevice
};

void GSAPrefetchEngineC::SetKvBuffer(torch::Tensor kBufferHost, torch::Tensor vBufferHost,
                                     torch::Tensor kBufferDevice, torch::Tensor vBufferDevice,
                                     torch::Tensor loadIndexHost, torch::Tensor loadIndexDevice)
{
    mLogger.log(LogLevel::DEBUG, "Decode step: %u, |KVCache Prefetch| Start SetKvBuffer\n",
                mDecodeStep);
    mKBufferHost = kBufferHost;
    mKBufferDevice = kBufferDevice;
    mLoadIndexHost = loadIndexHost;
    mLoadIndexDevice = loadIndexDevice;
    if (!mISSPAC8) {
        mVBufferHost = vBufferHost;
        mVBufferDevice = vBufferDevice;
    }
    if (mKVBufferLen == 0) {
        mKVBufferLen = mKBufferHost[0].sizes()[0];
        mLoadIndexBufferLen = mLoadIndexHost[0].sizes()[0];
    }
    mH2DPtrInfo.resize(mStreamNum);
    for (int i = 0; i < mStreamNum; i++) {
        mH2DPtrInfo[i].resize(H2D_INFO_SIZE);
        if (!mISSPAC8) {
            mVBufferHostPtr[i] = reinterpret_cast<uint64_t>(mVBufferHost[i].data_ptr());
            mH2DPtrInfo[i][static_cast<size_t>(H2DInfoIndex::VBufferHost)] =
                mVBufferHost[i].data_ptr();
            mH2DPtrInfo[i][static_cast<size_t>(H2DInfoIndex::VBufferDevice)] =
                mVBufferDevice[i].data_ptr();
        }
        mKBufferHostPtr[i] = reinterpret_cast<uint64_t>(mKBufferHost[i].data_ptr());
        mH2DPtrInfo[i][static_cast<size_t>(H2DInfoIndex::KBufferHost)] = mKBufferHost[i].data_ptr();
        mH2DPtrInfo[i][static_cast<size_t>(H2DInfoIndex::KBufferDevice)] =
            mKBufferDevice[i].data_ptr();
        mH2DPtrInfo[i][static_cast<size_t>(H2DInfoIndex::LoadIndexHost)] =
            mLoadIndexHost[i].data_ptr();
        mH2DPtrInfo[i][static_cast<size_t>(H2DInfoIndex::LoadIndexDevice)] =
            mLoadIndexDevice[i].data_ptr();
    }
    if (mKVPackControl == nullptr && InitKVPackControl(mKVPackDpRank) != 0) {
        mLogger.log(LogLevel::ERROR, "|KVPack| shared staging initialization failed\n");
        std::abort();
    }
    mLogger.log(LogLevel::DEBUG, "Decode step: %u, |KVCache Prefetch| SetKvBuffer\n", mDecodeStep);
}

void GSAPrefetchEngineC::SubmitTask(std::vector<int>& topkLensInput, std::vector<int>& bsIndexInput,
                                    std::vector<std::string>& reqIDList, torch::Tensor& topkLists,
                                    std::vector<bool>& isIndexerOffload, std::vector<int>& seqLens,
                                    uint32_t stepTime)
{
    mLogger.log(LogLevel::INFO,
                "Decode step: %u, |KVCache Prefetch| start SubmitTask batch size: %lu\n",
                mDecodeStep, reqIDList.size());
    auto start = std::chrono::high_resolution_clock::now();
    mDecodeStep = stepTime;
    /* UReqLock 埋点 dump 触发点（提交路径每步必经）：UCM_PREFETCH_LOCK_PROFILE
     * 开启时按 mUReqLockDumpIntervalSteps 周期输出争用画像（exchange 清零，
     * 输出为距上次 dump 的区间值）。 */
    if (mUReqLockProfileEnabled && static_cast<uint64_t>(mDecodeStep) >= mUReqLockNextDumpStep) {
        DumpUReqLockStats();
        mUReqLockNextDumpStep = static_cast<uint64_t>(mDecodeStep) + mUReqLockDumpIntervalSteps;
    }
    // 提前识别：set_kv_buffer 未完成（KVPack 控制块为空）时提交任务会在 worker 路径
    // 空指针崩溃，这里直接打错误日志并跳过本步提交（running 标志保持未置位）。
    if (mKVPackControl == nullptr) {
        mLogger.log(LogLevel::ERROR,
                    "Decode step: %u, |KVCache Prefetch| set_kv_buffer not done yet, "
                    "skip prefetch task\n",
                    mDecodeStep);
        std::cerr << "|KVCache Prefetch| [ERROR] set_kv_buffer not done, prefetch task skipped\n";
        return;
    }
    TaskInfo inputInfo(totalTaskNum, -1, topkLists, topkLensInput, bsIndexInput, reqIDList);
    mIsPrefetchRunning.store(true, std::memory_order_release);
    mIsIndexerOffload = isIndexerOffload;
    /* 提交期对 mRealTimeIdx/mUseBlockTableIdx/mSeqLen 的读写与 prefetch
     * worker 清理段 erase（DelReqIDRun，持 mUpdateReqMutex）及 SubmitTaskIndexer
     * 的 mUseBlockTableIdx 拷贝并发（gsa_index 路径 Python 采样状态后不等待 worker
     * 空闲即提交），对称加锁；锁内仅容器操作，无设备调用。
     * DEBUG 日志移出临界区（Logger 命中级别时含落盘 µs 级开销），
     * 尺寸锁内取样、锁外打印。 */
    struct SubmitTaskDebugSize {
        std::string reqId;
        uint32_t realTimeIdxSize;
        uint32_t useBlockTableIdxSize;
    };
    std::vector<SubmitTaskDebugSize> debugSizes;
    {
        UReqLockGuard lock(mUpdateReqMutex, UReqLockStatsAt(UReqLockSite::SubmitTask));
        for (uint32_t i = 0; i < reqIDList.size(); i++) {
            if (mOpenIndexerOffload && mIsIndexerOffload[i]) {
                mRealTimeIdx[reqIDList[i]].clear();
                mRealTimeIdx[reqIDList[i]] = mUseBlockTableIdx[reqIDList[i]];
                mSeqLen[reqIDList[i]] = seqLens[i];
                debugSizes.push_back({reqIDList[i],
                                      static_cast<uint32_t>(mRealTimeIdx[reqIDList[i]][0].size()),
                                      static_cast<uint32_t>(
                                          mUseBlockTableIdx[reqIDList[i]][0].size())});
            }
        }
    }
    for (size_t k = 0; k < debugSizes.size(); ++k) {
        mLogger.log(LogLevel::DEBUG,
                    "Decode step: %u, "
                    "|KVCache Prefetch| SubmitTask mRealTimeIdx: %u, mUseBlockTableIdx: %u "
                    "for req_id: %s\n",
                    mDecodeStep, debugSizes[k].realTimeIdxSize,
                    debugSizes[k].useBlockTableIdxSize, debugSizes[k].reqId.c_str());
    }
    mTaskManger->SubmitTask(inputInfo, MultiBSThreadFun);
    totalTaskNum += 1;
    auto endLoadPre = std::chrono::high_resolution_clock::now();
    auto durationLoadPre =
        std::chrono::duration_cast<std::chrono::microseconds>(endLoadPre - start);
    mLogger.log(
        LogLevel::INFO,
        "Decode step: %u, |KVCache Prefetch| finish SubmitTask batch size: %lu, cost: %lu\n",
        mDecodeStep, reqIDList.size(), durationLoadPre.count());
}

void GSAPrefetchEngineC::SubmitTaskIndexer(std::vector<int>& topkLensInput,
                                           std::vector<std::string>& reqIDList,
                                           std::vector<torch::Tensor>& topkLists, uint32_t stepTime)
{
    if (!mOpenIndexerOffload) { return; }
    mLogger.log(LogLevel::INFO,
                "Decode step: %u, |KVCache Prefetch| start SubmitTaskIndexer batch size: %lu\n",
                mDecodeStep, reqIDList.size());
    auto start = std::chrono::high_resolution_clock::now();
    /* mPrefetchBlockTableIdx 唯一写者为 indexer worker 的
     * FinishOneLayerPrefetch（IndexerLoadProcess 全程持 mUpdateReqMutex），读侧
     * 对称加锁；is_indexer_load 门控保证提交时 indexer worker 空闲，锁无竞争。 */
    UReqLockGuard lock(mUpdateReqMutex, UReqLockStatsAt(UReqLockSite::SubmitTaskIndexer));
    for (uint32_t i = 0; i < reqIDList.size(); i++) {
        for (int layerID = 0; layerID < mLayerNum; layerID++) {
            std::memcpy(mUseBlockTableIdx[reqIDList[i]][layerID].data(),
                        mPrefetchBlockTableIdx[reqIDList[i]][layerID].data(),
                        topkLensInput[i] * sizeof(int));
        }
    }
    mDecodeStep = stepTime;
    TaskInfo inputInfo;
    inputInfo.indexerTopkList = topkLists;
    inputInfo.topkLenList = topkLensInput;
    inputInfo.reqIdList = reqIDList;
    mIsIndexerLoadRunning.store(true, std::memory_order_release);
    mTaskManger->SubmitTask(inputInfo, MultiBSThreadFunIndexer);
    totalTaskNum += 1;
    auto endLoadPre = std::chrono::high_resolution_clock::now();
    auto durationLoadPre =
        std::chrono::duration_cast<std::chrono::microseconds>(endLoadPre - start);
    mLogger.log(
        LogLevel::INFO,
        "Decode step: %u, |KVCache Prefetch| finish SubmitTaskIndexer batch size: %lu, cost: %lu\n",
        mDecodeStep, reqIDList.size(), durationLoadPre.count());
}

void GSAPrefetchEngineC::SetTopkIndices(std::string reqID, torch::Tensor& topIndices)
{
    mLogger.log(LogLevel::DEBUG,
                "Decode step: %u, |KVCache Prefetch| Start SetTopkIndices reqID: %s\n", mDecodeStep,
                reqID.c_str());
    /* mAllTopkIndicesTensor 全部读写统一 mUpdateReqMutex 协议（读者：worker 树
     * ReadPrefetchResults / CopyUseBlockTableToIndicesTensor；erase 侧：DelReqIDRun）。
     * Python 侧在等待 worker 空闲之前调用本接口（execute_begin 前置段
     * _process_scheduled_req），新 req 首个 decode step 的 map 插入/重平衡必然与
     * worker 并发遍历交叠，不能靠 Python 序列化避免，锁不可省。锁收窄到仅包
     * map 赋值（百 ns 级），End 日志移出临界区——Logger 命中级别时含自身互斥
     * 与 localtime/strftime/vsnprintf/落盘（µs 级），不应计入持锁时长。 */
    {
        UReqLockGuard lock(mUpdateReqMutex, UReqLockStatsAt(UReqLockSite::SetTopkIndices));
        mAllTopkIndicesTensor[reqID] = topIndices;
    }
    mLogger.log(LogLevel::DEBUG,
                "Decode step: %u, |KVCache Prefetch| End SetTopkIndices reqID: %s\n", mDecodeStep,
                reqID.c_str());
}

void GSAPrefetchEngineC::CopyPreSwapTokens(uintptr_t useTokenBase, uintptr_t spaTokenBase,
                                           const std::vector<int>& swapRange,
                                           const std::vector<int>& needAppend, int sparseLen,
                                           int spaMaxBs, int mtpToken, int changeLen)
{
#pragma omp parallel for num_threads(8) proc_bind(master)
    for (int layerID = 0; layerID < mTopkIndexerNum; layerID++) {
        void* usePtr =
            reinterpret_cast<void*>(useTokenBase + sparseLen * layerID * sizeof(int32_t));
        for (int index = swapRange[0]; index < swapRange[1]; index++) {
            void* spaCpuPtr = reinterpret_cast<void*>(
                spaTokenBase + (size_t)(layerID * spaMaxBs + index) * sparseLen * sizeof(int32_t));
            std::memcpy(spaCpuPtr, usePtr, sparseLen * sizeof(int32_t));
            std::memcpy(reinterpret_cast<uint32_t*>(spaCpuPtr) + sparseLen - changeLen + mtpToken -
                            1 - index % mtpToken,
                        needAppend.data(),
                        (changeLen - mtpToken + 1 + index % mtpToken) * sizeof(int));
        }
    }
}

void GSAPrefetchEngineC::PreSwapTokens(std::vector<uintptr_t>& useTokenPtrs,
                                       std::vector<int32_t>& seqLens,
                                       std::vector<std::vector<int>>& swapIndex,
                                       std::vector<uintptr_t>& spaTokensPtr,
                                       std::vector<std::string>& reqIDs, int sparseLen,
                                       int spaMaxBs, int mtpToken)
{
    mLogger.log(LogLevel::DEBUG,
                "Decode step: %u, |KVCache Prefetch|"
                " Start PreSwapTokens, sparseLen: %d, spaMaxBs: %d, mtpToken: %d\n",
                mDecodeStep, sparseLen, spaMaxBs, mtpToken);
    auto start = std::chrono::high_resolution_clock::now();
    auto ret = aclrtSetCurrentContext(*(static_cast<aclrtContext*>(mContext)));
    if (ret != ACL_SUCCESS) {
        const char* errMsg = aclGetRecentErrMsg();
        mLogger.log(LogLevel::ERROR,
                    "|KVCache Prefetch| aclrtSetCurrentContext Error: %d, Error: %s\n", ret,
                    errMsg ? errMsg : "Unknown error");
    }
    uint64_t max_tokens_time = 0;
    uint64_t cpu_copy_time = 0;
    for (size_t bs = 0; bs < useTokenPtrs.size(); bs++) {
        auto start_max = std::chrono::high_resolution_clock::now();
        const int32_t* data = reinterpret_cast<const int32_t*>(useTokenPtrs[bs]);
        int maxID = *std::max_element(data, data + sparseLen * mTopkIndexerNum);
        int start = maxID + 1;
        int end = seqLens[bs] + mtpToken - 1;
        int change_len = end - start;
        std::vector<int> needAppend;
        if (change_len <= mtpToken - 1) {
            mLogger.log(LogLevel::ERROR,
                        "[ERROR] PreSwapTokens change len < 0 !!!!!! start %d, end %d\n", start,
                        end);
            change_len = mtpToken;
        }
        mLogger.log(
            LogLevel::DEBUG,
            "Decode step: %u, |KVCache Prefetch| Start PreSwapTokens, change_len: %d, maxID: %d\n",
            mDecodeStep, change_len, maxID);
        needAppend.resize(change_len);
        std::iota(needAppend.begin(), needAppend.end(), start);
        auto max_end = std::chrono::high_resolution_clock::now();
        max_tokens_time +=
            std::chrono::duration_cast<std::chrono::microseconds>(max_end - start_max).count();
        CopyPreSwapTokens(useTokenPtrs[bs], spaTokensPtr[0], swapIndex[bs], needAppend, sparseLen,
                          spaMaxBs, mtpToken, change_len);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        cpu_copy_time +=
            std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - max_end).count();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    mLogger.log(LogLevel::DEBUG,
                "Decode step: %u, |KVCache Prefetch| End PreSwapTokens, "
                "bs: %lu total time: %lu max_tokens_time: %lu cpu_copy_time: %lu\n",
                mDecodeStep, useTokenPtrs.size(), duration.count(), max_tokens_time, cpu_copy_time);
}

void GSAPrefetchEngineC::DelBlocksMap(std::string reqID)
{
    mLogger.log(LogLevel::DEBUG,
                "Decode step: %u, |KVCache Prefetch| Start DelBlocksMap, reqID: %s\n", mDecodeStep,
                reqID.c_str());
    {
        UReqLockGuard lock(mUpdateReqMutex, UReqLockStatsAt(UReqLockSite::DelBlocksMap));
        mDelSeqIds.insert(reqID);
        // Only run the deletion when neither prefetch nor indexer load is running,
        // so that the shared per-request maps are not erased mid-update.
        if (!GetPrefetchStatus() && !GetIndexerStatus()) { DelReqIDRun(); }
    }
    mLogger.log(LogLevel::DEBUG,
                "Decode step: %u, |KVCache Prefetch| End DelBlocksMap, reqID: %s\n", mDecodeStep,
                reqID.c_str());
}

void GSAPrefetchEngineC::DelReqIDRun()
{
    mLogger.log(LogLevel::DEBUG, "Decode step: %u, |KVCache Prefetch| Start DelReqIDRun\n",
                mDecodeStep);
    for (auto it = mDelSeqIds.begin(); it != mDelSeqIds.end(); it++) {
        if (mSparseSeqLen.find(*it) == mSparseSeqLen.end()) {
            mLogger.log(LogLevel::INFO,
                        "Decode step: %u, |KVCache Prefetch| req: %s not in mSparseSeqLen\n",
                        mDecodeStep, (*it).c_str());
            continue;
        } else {
            mAllTopkIndicesTensor.erase(*it);
            mSparseSeqLen.erase(*it);
            mAllReqIdSlots.erase(*it);
            mPromptLen.erase(*it);
            mHBMMangers.erase(*it);
            mMissGlobalToken.erase(*it);
            mLoadSlot.erase(*it);
            mReqUseBlockTable.erase(*it);
            /* L-4 修复：SetBlocksMap 写入的两兄弟成员一并清理，
             * 否则每完成请求泄漏 16B×块数（BlockId/LastKVBlockAddress 条目）。 */
            mAllBlocksHash.erase(*it);
            mLastKVBlockAddress.erase(*it);
            mBlocksMap.erase(*it);
            mDocsTables.erase(*it);
            mPrefetchIdx.erase(*it);
            mIndexerLoadResultT.erase(*it);
            mPrefetchBlockTableIdx.erase(*it);
            mUseBlockTableIdx.erase(*it);
            mRealTimeIdx.erase(*it);
            mSeqLen.erase(*it);
            allNeedLoadBlock.erase(*it);
            allMissIdxs.erase(*it);
            mLogger.log(LogLevel::INFO,
                        "Decode step: %u, |KVCache Prefetch| delete info of req: %s\n", mDecodeStep,
                        (*it).c_str());
        }
    }
    mDelSeqIds.clear();
    mLogger.log(LogLevel::DEBUG, "Decode step: %u, |KVCache Prefetch| End DelReqIDRun\n",
                mDecodeStep);
}

bool GSAPrefetchEngineC::GetPrefetchStatus()
{
    return mIsPrefetchRunning.load(std::memory_order_acquire);
}

bool GSAPrefetchEngineC::GetIndexerStatus()
{
    return mIsIndexerLoadRunning.load(std::memory_order_acquire);
}
}  // namespace ucmprefetch
