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
#include "kvcache_pre.h"
#include <kvcache_log.h>
#include <sched.h>
#include <stdint.h>
#define MAX_LOAD_NUM 2000u

namespace ucmprefetch {
void MutliBSThreadFun(void* args)
{
    GSAPrefetchEngineC* engine = static_cast<GSAPrefetchEngineC*>(args);
    int ret = engine->CallPrefetchProcessFun();
    engine->mMutex.lock();
    engine->DelReqIDRun();
    engine->mMutex.unlock();
    if (ret == 0) { engine->SetPrefetchStatus(true); }
}

void RunQueuePolling(void* args)
{
    GSAPrefetchEngineC* engine = static_cast<GSAPrefetchEngineC*>(args);
    engine->QueuePolling();
}

void CallPrefetchBSLayer(void* args, PrefetchInfoLayer oneBsInfo)
{
    GSAPrefetchEngineC* engine = static_cast<GSAPrefetchEngineC*>(args);
    engine->RunPrefetchBSLayer(oneBsInfo);
}

GSAPrefetchEngineC::GSAPrefetchEngineC(torch::Tensor& loadSuccessBlocks,
                                       torch::Tensor& successTableLen, bool useMla, bool isLog,
                                       int rank)
    : mLogger("./log/kvcache_pre_log.txt", LogLevel::INFO, isLog)
{
    mLoadSuccessBlocks = loadSuccessBlocks;
    mLayerNum = mLoadSuccessBlocks.sizes()[0];
    mMaxBs = mLoadSuccessBlocks.sizes()[1];
    mMaxTopkLen = mLoadSuccessBlocks.sizes()[2];
    mSuccessTableLen = successTableLen;
    mIsLog = isLog;
    mBsIndexList = (int*)malloc(sizeof(int) * mMaxBs);
    mTopkLenList = (int*)malloc(sizeof(int) * mMaxBs);
    mIsPrefetchDone = true;
    mThreadPool = ThreadPool::GetInst();
    mUseMla = useMla;
    mRank = rank;
    cudaStream_t stream_;
    auto err = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        mLogger.log(LogLevel::ERROR, "GSAPrefetchEngineC create stream error %s\n",
                    cudaGetErrorString(err));
    }
    mLogger.log(LogLevel::INFO, "GSAPrefetchEngineC create stream success\n");
    mGSATransBackend = TransBackend::GetInst(stream_);
    mLogger.log(LogLevel::INFO, "GSAPrefetchEngineC create stream success\n");
    if (mRank != 0) {
        mLogger.SetLevel(LogLevel::WARNING);
        mIsLog = false;
    }
    mLogger.log(LogLevel::INFO, "GSAPrefetchEngineC Init mLayerNum %d mMaxBs %u, mUseMla %d\n",
                mLayerNum, mMaxBs, mUseMla);
    mThreadPool->Enqueue(RunQueuePolling, this);
}

void GSAPrefetchEngineC::CheckInputIndex(uint32_t maxLen, uint32_t index)
{
    if (index >= maxLen) {
        mLogger.log(LogLevel::ERROR,
                    "Decode step: %u, |KVCache Prefetch| index error! index: %u, maxLen: %u\n",
                    mDecodeStep, index, maxLen);
        std::abort();
    }
}

GSAPrefetchEngineC::~GSAPrefetchEngineC()
{
    free(mBsIndexList);
    free(mTopkLenList);
    mIsPrefetchRunning = false;
}

void GSAPrefetchEngineC::QueuePolling()
{
    while (mIsPrefetchRunning) {
        if (!mLoadQueue.empty()) {
            PrefetchInfoLayer oneBsInfo;
            oneBsInfo = mLoadQueue.pop();
            mThreadPool->Enqueue(CallPrefetchBSLayer, this, oneBsInfo);
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(5));
        }
    }
}

void GSAPrefetchEngineC::AddPrefetchTask(uint32_t layerID, std::vector<std::string>& reqIDList,
                                         std::vector<std::vector<int32_t>>& topkList,
                                         std::vector<int>& bsIndexList,
                                         std::vector<int>& topkLenList)
{
    PrefetchInfoLayer oneBsInfo;
    oneBsInfo.layerID = layerID;
    oneBsInfo.reqIDList = reqIDList;
    oneBsInfo.topkList = topkList;
    oneBsInfo.bsIndexList = bsIndexList;
    oneBsInfo.topkLenList = topkLenList;
    mLoadQueue.push(oneBsInfo);
}

void GSAPrefetchEngineC::RunPrefetchBSLayer(PrefetchInfoLayer oneBsInfo)
{
    auto start = std::chrono::high_resolution_clock::now();
    std::map<std::string, std::vector<int>> batchLoadBlockLayer;
    std::map<std::string, std::vector<int>> batchMissIDLayer;
    for (size_t i = 0; i < oneBsInfo.reqIDList.size(); i++) {
        if (oneBsInfo.topkLenList[i] == 0) {
            continue;
        } else {
            std::vector<int> oneLoadBlockIDs;
            std::vector<int> oneMissIdxs;
            RunOneBsPrefetchLayer(oneBsInfo.reqIDList[i], oneBsInfo.bsIndexList[i],
                                  oneBsInfo.layerID, oneLoadBlockIDs, oneMissIdxs,
                                  oneBsInfo.topkList[i]);
            batchLoadBlockLayer[oneBsInfo.reqIDList[i]] = oneLoadBlockIDs;
            batchMissIDLayer[oneBsInfo.reqIDList[i]] = oneMissIdxs;
        }
    }
    TransKVCacheLayer(oneBsInfo.layerID, batchLoadBlockLayer, batchMissIDLayer);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    mLogger.log(LogLevel::INFO,
                "Decode step: %u, |KVCache Prefetch| Finish async pretch cost: %lu\n", mDecodeStep,
                duration.count());
}

void GSAPrefetchEngineC::SetBlocksMap(std::string reqID, std::vector<int>& blockTableList,
                                      std::vector<int>& remainIdx, std::vector<int>& prefetchIdx,
                                      int maxIdx, std::vector<int>& slots)
{
    if (mBlocksMap.find(reqID) != mBlocksMap.end()) {
        mBlocksMap[reqID].clear();
        mDocsTables[reqID].clear();
        mPrefetchIdx[reqID].clear();
        mAllReqIdSlots[reqID].clear();
    }
    mAllReqIdSlots[reqID] = slots;
    for (int i = 0; i < mLayerNum; i++) {
        std::map<int, int> oneDocTable;
        std::map<int, int> oneBlockMap;
        std::vector<int> onePrefetchIdx;
        for (auto idx : remainIdx) {
            oneDocTable[idx] = blockTableList[idx];
            oneBlockMap[blockTableList[idx]] = idx;
        }
        for (auto idx : prefetchIdx) {
            oneDocTable[idx] = blockTableList[idx];
            oneBlockMap[blockTableList[idx]] = idx;
            onePrefetchIdx.push_back(idx);
        }
        mDocsTables[reqID].push_back(oneDocTable);
        mBlocksMap[reqID].push_back(oneBlockMap);
        mPrefetchIdx[reqID].push_back(onePrefetchIdx);
    }
    mPromptLen[reqID] = maxIdx;
    PrintMap(reqID, 0);
}

void GSAPrefetchEngineC::SetBlocksMapMultiLayer(std::string reqID,
                                                std::vector<std::map<int, int>>& remainMap,
                                                std::vector<std::map<int, int>>& prefetchMap,
                                                int maxIdx, std::vector<int>& slots)
{
    if (mBlocksMap.find(reqID) != mBlocksMap.end()) {
        mBlocksMap[reqID].clear();
        mDocsTables[reqID].clear();
        mPrefetchIdx[reqID].clear();
        mAllReqIdSlots[reqID].clear();
    }
    mAllReqIdSlots[reqID] = slots;
    for (int i = 0; i < mLayerNum; i++) {
        std::map<int, int> oneDocTable;
        std::map<int, int> oneBlockMap;
        std::vector<int> onePrefetchIdx;
        for (auto it = remainMap[i].begin(); it != remainMap[i].end(); it++) {
            oneDocTable[it->first] = it->second;
            oneBlockMap[it->second] = it->first;
        }
        for (auto it = prefetchMap[i].begin(); it != prefetchMap[i].end(); it++) {
            oneDocTable[it->first] = it->second;
            oneBlockMap[it->second] = it->first;
            onePrefetchIdx.push_back(it->first);
        }
        mDocsTables[reqID].push_back(oneDocTable);
        mBlocksMap[reqID].push_back(oneBlockMap);
        mPrefetchIdx[reqID].push_back(onePrefetchIdx);
    }
    mPromptLen[reqID] = maxIdx;
}

void GSAPrefetchEngineC::AddBlocksMap(std::string reqID, int idx, int blockID)
{
    if (mBlocksMap.find(reqID) == mBlocksMap.end()) {
        for (int i = 0; i < mLayerNum; ++i) {
            std::map<int, int> oneDocTable;
            std::map<int, int> oneBlockMap;
            oneDocTable[idx] = blockID;
            oneBlockMap[blockID] = idx;
            mDocsTables[reqID].push_back(oneDocTable);
            mBlocksMap[reqID].push_back(oneBlockMap);
        }
    } else {
        for (int i = 0; i < mLayerNum; i++) {
            mDocsTables[reqID][i][idx] = blockID;
            mBlocksMap[reqID][i][blockID] = idx;
        }
    }
}

void GSAPrefetchEngineC::DelBlocksMap(std::string reqID)
{
    mMutex.lock();
    mDelSeqIds.insert(reqID);
    if (mIsPrefetchDone) { DelReqIDRun(); }
    mMutex.unlock();
}

void GSAPrefetchEngineC::DelReqIDRun()
{
    for (auto it = mDelSeqIds.begin(); it != mDelSeqIds.end(); it++) {
        if (mBlocksMap.find(*it) == mBlocksMap.end()) {
            continue;
        } else {
            mBlocksMap.erase(*it);
            mDocsTables.erase(*it);
            mPromptLen.erase(*it);
            mPrefetchIdx.erase(*it);
            mAllReqIdSlots.erase(*it);
            std::cout << "Del reqID: " << *it << std::endl;
        }
        if (mPromptLen.find(*it) == mPromptLen.end()) {
            continue;
        } else {
            mPromptLen.erase(*it);
        }
    }
    mDelSeqIds.clear();
}

void GSAPrefetchEngineC::PrintMap(std::string reqID, int i)
{
    std::ostringstream oss;
    oss << "Decode step: " << mDecodeStep << " Rnak: " << mRank << " reqID: " << reqID
        << " layerID: " << i << "mDocsTables";
    for (auto it : mDocsTables[reqID][i]) { oss << "(" << it.first << ", " << it.second << ")"; }
    oss << "------\n";
    mLogger.log(LogLevel::INFO, oss.str().c_str());
    oss.str("");
    oss << "Decode step: " << mDecodeStep << " Rnak: " << mRank << " reqID: " << reqID
        << " layerID: " << i << "mBlocksMap";
    for (auto it : mBlocksMap[reqID][i]) { oss << "(" << it.first << ", " << it.second << ")"; }
    oss << "------\n";
    mLogger.log(LogLevel::INFO, oss.str().c_str());
    oss.str("");
}

void GSAPrefetchEngineC::PrintVector(std::vector<int>& vec, int layerID, std::string reqID,
                                     std::string name)
{
    std::ostringstream oss;
    oss << "Decode step: " << mDecodeStep << " Rnak: " << mRank << " reqID: " << reqID
        << " layerID: " << layerID << " " << name << ": ";
    for (auto it : vec) { oss << it << " "; }
    oss << "------\n";
    mLogger.log(LogLevel::DEBUG, oss.str().c_str());
    oss.str("");
}

void GSAPrefetchEngineC::GetHitAndMissBlock(PrefetchReqInfo oneBsInfo,
                                            std::unordered_set<int>& hitBlocks,
                                            std::map<int, int>& hitBlocksIdx,
                                            std::vector<int>& missIdxs)
{
    int topkLen = oneBsInfo.topkLen;
    int layerID = oneBsInfo.layerID;
    std::string reqID = oneBsInfo.reqID;
    int topkIndex = oneBsInfo.topkIndex;

    std::ostringstream oss;
    oss << "Decode step: " << mDecodeStep << " Rnak: " << mRank << " reqID: " << reqID
        << " layerID: " << layerID << " topk len: " << topkLen << " topk: ";
    std::vector<int32_t> topkItem32(topkLen);
    std::vector<int64_t> topkItem64(topkLen);
    bool isInt32 = (mUseTopkIdxs.scalar_type() == torch::kInt32);
    if (isInt32) {
        std::memcpy(topkItem32.data(),
                    mUseTopkIdxs[layerID][topkIndex].data_ptr<int32_t>(),
                    topkLen * sizeof(int32_t));
    } else {
        std::memcpy(topkItem64.data(),
                    mUseTopkIdxs[layerID][topkIndex].data_ptr<int64_t>(),
                    topkLen * sizeof(int64_t));
    }
    for (int j = 0; j < topkLen; j++) {
        int64_t item = 0;
        if (isInt32) {
            item = topkItem32[j];
        } else {
            item = topkItem64[j];
        }
        oss << item << " ";
        if (mDocsTables[reqID][layerID].find(item) != mDocsTables[reqID][layerID].end()) {
            int blockID = mDocsTables[reqID][layerID][item];
            hitBlocks.insert(blockID);
            hitBlocksIdx.insert(std::make_pair(item, blockID));
        } else {
            missIdxs.push_back(item);
        }
    }
    oss << "------\n";
    mLogger.log(LogLevel::DEBUG, oss.str().c_str());
    oss.str("");
    if ((hitBlocks.size() + missIdxs.size()) != (uint32_t)topkLen) {
        mLogger.log(LogLevel::ERROR,
                    "|KVCache Prefetch| Decode step: %u, Rank: %d, reqID: %s, layer: %d, hit size: "
                    "%lu, miss size: %lu , topkLen: %d, not equal error\n",
                    mDecodeStep, mRank, reqID, layerID, hitBlocks.size(), missIdxs.size(), topkLen);
        PrintMap(reqID, layerID);
    }
}

void GSAPrefetchEngineC::RunPrefetchH2D(PrefetchReqInfo oneBsInfo,
                                        std::unordered_set<int>& hitBlocks,
                                        std::map<int, int>& hitBlocksIdx,
                                        std::vector<int>& missIdxs)
{
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
    LoadKVToHBM(oneFreeBlockTable, missIdxs, layerID, reqID);
}

void GSAPrefetchEngineC::RunOneBsPrefetch(std::string reqID, int topkLen, int bsIndex,
                                          int topkIndex)
{
    std::vector<int> costtime = {0, 0, 0, 0};
#pragma omp parallel for num_threads(16) proc_bind(master)
    for (int i = 0; i < mLayerNum; i++) {
        mLoadSuccessBlocks[i][bsIndex].fill_(0);
        std::unordered_set<int> hitBlocks;
        std::map<int, int> hitBlocksIdx;
        std::vector<int> missIdxs;
        PrefetchReqInfo oneBsInfo;
        oneBsInfo.topkLen = topkLen;
        oneBsInfo.reqID = reqID;
        oneBsInfo.topkIndex = topkIndex;
        oneBsInfo.bsIndex = bsIndex;
        oneBsInfo.layerID = i;
        auto start = std::chrono::high_resolution_clock::now();
        GetHitAndMissBlock(oneBsInfo, hitBlocks, hitBlocksIdx, missIdxs);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        costtime[0] += duration.count();
        start = std::chrono::high_resolution_clock::now();
        if (missIdxs.size() != 0) { RunPrefetchH2D(oneBsInfo, hitBlocks, hitBlocksIdx, missIdxs); }
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        costtime[1] += duration.count();

        start = std::chrono::high_resolution_clock::now();
        std::vector<int32_t> successBlock;
        int successIndex = 0;
        for (auto it = hitBlocksIdx.begin(); it != hitBlocksIdx.end(); it++) {
            // mLoadSuccessBlocks[i][bsIndex][successIndex] = it->second;
            successBlock.push_back(it->second);
            successIndex += 1;
        }
        std::memcpy(
            mLoadSuccessBlocks[i][bsIndex].data_ptr<int32_t>(),
            successBlock.data(),
            successBlock.size() * sizeof(int32_t));
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        costtime[2] += duration.count();

        start = std::chrono::high_resolution_clock::now();
        mPrefetchIdx[reqID][i].clear();
        for (auto it = mDocsTables[reqID][i].begin(); it != mDocsTables[reqID][i].end(); it++) {
            if (it->first >= mPromptLen[reqID]) { break; }
            if (hitBlocksIdx.find(it->first) != hitBlocksIdx.end()) {
                continue;
            } else {
                mPrefetchIdx[reqID][i].push_back(it->first);
            }
        }
        mSuccessTableLen[i][bsIndex] = (int)(hitBlocks.size());
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        costtime[3] += duration.count();
    }
    mLogger.log(LogLevel::INFO,
                "Decode step: %u, |KVCache Prefetch| Finish async pretch reqID: %s, bsIndex: %d, "
                "cost time(us): GetHitAndMiss %d, PrefetchH2D %d, Update block table info %d, "
                "Update prefetch block info %d\n",
                mDecodeStep, reqID.c_str(), bsIndex, costtime[0], costtime[1], costtime[2],
                costtime[3]);
}

void GSAPrefetchEngineC::RunOneBsPrefetchLayer(std::string reqID, int bsIndex, int layerID,
                                               std::vector<int>& oneFreeBlockTable,
                                               std::vector<int>& missIdxs,
                                               std::vector<int>& topkList)
{
    mLoadSuccessBlocks[layerID][bsIndex].fill_(0);
    std::unordered_set<int> hitBlocks;
    std::map<int, int> hitBlocksIdx;
    int32_t topkLen = topkList.size();

    // GetHitAndMissBlock
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
                    mDecodeStep, mRank, reqID, layerID, hitBlocks.size(), missIdxs.size(), topkLen);
        PrintMap(reqID, layerID);
        PrintVector(topkList, layerID, reqID, "topkList");
    }

    // RunPrefetchH2D
    if (missIdxs.size() != 0) {
        std::vector<int> onePrefetchIdx = mPrefetchIdx[reqID][layerID];
        int index = 0;
        for (size_t z = 0; z < onePrefetchIdx.size(); z++) {
            int oneFreeBlockID = mDocsTables[reqID][layerID][onePrefetchIdx[z]];
            if (hitBlocks.find(oneFreeBlockID) == hitBlocks.end()) {
                oneFreeBlockTable.push_back(oneFreeBlockID);
                hitBlocks.insert(oneFreeBlockID);
                hitBlocksIdx.insert(std::make_pair(missIdxs[index], oneFreeBlockID));
                index += 1;
                if (oneFreeBlockTable.size() == missIdxs.size()) { break; }
            }
        }
        missIdxs.erase(missIdxs.begin() + oneFreeBlockTable.size(), missIdxs.end());
        LoadKVToHBM(oneFreeBlockTable, missIdxs, layerID, reqID);
    }

    // Update success blocks and prefetch idx
    int successIndex = 0;
    for (auto it = hitBlocksIdx.begin(); it != hitBlocksIdx.end(); it++) {
        mLoadSuccessBlocks[layerID][bsIndex][successIndex] = it->second;
        successIndex += 1;
    }
    mPrefetchIdx[reqID][layerID].clear();
    for (auto it = mDocsTables[reqID][layerID].begin(); it != mDocsTables[reqID][layerID].end();
         it++) {
        if (it->first >= mPromptLen[reqID]) { break; }
        if (hitBlocksIdx.find(it->first) != hitBlocksIdx.end()) {
            continue;
        } else {
            mPrefetchIdx[reqID][layerID].push_back(it->first);
        }
    }
    mSuccessTableLen[layerID][bsIndex] = (int)(hitBlocks.size());
}

void GSAPrefetchEngineC::TransKVCache()
{
    std::ostringstream oss;
    for (int layerID = 0; layerID < mLayerNum; layerID++) {
        oss << "Decode step: " << mDecodeStep << " Rnak: " << mRank << " layerID: " << layerID
            << " load info(reqID-blockID-slot-ptr): ";
        std::vector<uint64_t> hostKPtr;
        std::vector<uint64_t> deviceKPtr;
        std::vector<uint64_t> hostVPtr;
        std::vector<uint64_t> deviceVPtr;
        for (auto& it : allNeedLoadBlock) {
            std::string reqID = it.first;
            std::vector<int>& loadBlockIDs = allNeedLoadBlock[reqID][layerID];
            std::vector<int>& oneMissIDs = allMissIdxs[reqID][layerID];
            if (loadBlockIDs.size() == 0) { continue; }
            for (uint32_t i = 0; i < loadBlockIDs.size(); i++) {
                deviceKPtr.push_back(mKcachePtr[layerID] + mKVSzieBytes * loadBlockIDs[i]);
                hostKPtr.push_back(mSlabKcachePtr[layerID] +
                                   mKVSzieBytes * mAllReqIdSlots[reqID][oneMissIDs[i]]);
                if (!mUseMla) {
                    deviceVPtr.push_back(mVcachePtr[layerID] + mKVSzieBytes * loadBlockIDs[i]);
                    hostVPtr.push_back(mSlabVcachePtr[layerID] +
                                       mKVSzieBytes * mAllReqIdSlots[reqID][oneMissIDs[i]]);
                }
                oss << "(" << reqID << "-" << loadBlockIDs[i] << "-"
                    << mAllReqIdSlots[reqID][oneMissIDs[i]] << "-" << hostKPtr.back() << "-"
                    << deviceKPtr.back() << ") ";
            }
        }
        oss << "------\n";
        mLogger.log(LogLevel::DEBUG, oss.str().c_str());
        oss.str("");
        mLogger.log(
            LogLevel::DEBUG,
            "|KVCache Prefetch| Decode step: %u, Rank: %d, layerID: %d, H2D kvcache size: %lu\n",
            mDecodeStep, mRank, layerID, deviceKPtr.size());
        if (deviceKPtr.size() == 0) { continue; }
        uint32_t loadNum = deviceKPtr.size();
        if (deviceKPtr.size() > MAX_LOAD_NUM) {
            mLogger.log(LogLevel::WARNING,
                        "|KVCache Prefetch| Decode step: %u, Rank: %d, layerID: %d, load size: %lu "
                        "exceed max load num: %u\n",
                        mDecodeStep, mRank, layerID, deviceKPtr.size(), MAX_LOAD_NUM);
            loadNum = MAX_LOAD_NUM;
        }
        std::memcpy(mDeviceKPtrTensorCpu[0].data_ptr(), &deviceKPtr[0], sizeof(uint64_t) * loadNum);
        std::memcpy(mDeviceKPtrTensorCpu[1].data_ptr(), &hostKPtr[0], sizeof(uint64_t) * loadNum);
        if (!mUseMla) {
            std::memcpy(mDeviceKPtrTensorCpu[2].data_ptr(), &deviceVPtr[0],
                        sizeof(uint64_t) * loadNum);
            std::memcpy(mDeviceKPtrTensorCpu[3].data_ptr(), &hostVPtr[0],
                        sizeof(uint64_t) * loadNum);
        }
        mDeviceKPtrTensor = mDeviceKPtrTensorCpu.to(mDeviceKPtrTensor.device());
        auto ret = mGSATransBackend->copy_trans(
            static_cast<void**>(mDeviceKPtrTensor[1].data_ptr()),
            static_cast<void**>(mDeviceKPtrTensor[0].data_ptr()), mKVSzieBytes, deviceKPtr.size());
        if (ret != 0) {
            mLogger.log(LogLevel::ERROR,
                        "|KVCache Prefetch| Decode step: %u, Rank: %d, layerID: %d, H2D kvcache "
                        "transfer error\n",
                        mDecodeStep, mRank, layerID);
        } else {
            mLogger.log(LogLevel::DEBUG,
                        "|KVCache Prefetch| Decode step: %u, Rank: %d, layerID: %d, H2D kvcache "
                        "transfer success\n",
                        mDecodeStep, mRank, layerID);
        }
        if (!mUseMla) {
            mGSATransBackend->copy_trans(static_cast<void**>(mDeviceKPtrTensor[3].data_ptr()),
                                         static_cast<void**>(mDeviceKPtrTensor[2].data_ptr()),
                                         mKVSzieBytes, deviceKPtr.size());
            mLogger.log(LogLevel::DEBUG, "|KVCache Prefetch| transfer vcache done\n");
        }
    }
}

void GSAPrefetchEngineC::TransKVCacheLayer(int layerID,
                                           std::map<std::string, std::vector<int>>& batchLoadBlock,
                                           std::map<std::string, std::vector<int>>& batchMissIdxs)
{
    std::vector<uint64_t> hostKPtr;
    std::vector<uint64_t> deviceKPtr;
    std::vector<uint64_t> hostVPtr;
    std::vector<uint64_t> deviceVPtr;
    for (auto& it : batchLoadBlock) {
        std::string reqID = it.first;
        std::vector<int>& loadBlockIDs = batchLoadBlock[reqID];
        std::vector<int>& oneMissIDs = batchMissIdxs[reqID];
        if (loadBlockIDs.size() == 0) { continue; }
        for (uint32_t i = 0; i < loadBlockIDs.size(); i++) {
            deviceKPtr.push_back(mKcachePtr[layerID] + mKVSzieBytes * loadBlockIDs[i]);
            hostKPtr.push_back(mSlabKcachePtr[layerID] +
                               mKVSzieBytes * mAllReqIdSlots[reqID][oneMissIDs[i]]);
            if (!mUseMla) {
                deviceVPtr.push_back(mVcachePtr[layerID] + mKVSzieBytes * loadBlockIDs[i]);
                hostVPtr.push_back(mSlabVcachePtr[layerID] +
                                   mKVSzieBytes * mAllReqIdSlots[reqID][oneMissIDs[i]]);
            }
        }
    }
    if (deviceKPtr.size() == 0) { return; }
    uint32_t loadNum = deviceKPtr.size();
    auto device = mKvCaches[0].device();
    auto optionsCpu =
        torch::TensorOptions().dtype(torch::kUInt64).device("cpu").pinned_memory(true);
    torch::Tensor transPtrTensorCpu = torch::empty({4, loadNum}, optionsCpu);

    std::memcpy(transPtrTensorCpu[0].data_ptr(), &deviceKPtr[0], sizeof(uint64_t) * loadNum);
    std::memcpy(transPtrTensorCpu[1].data_ptr(), &hostKPtr[0], sizeof(uint64_t) * loadNum);
    if (!mUseMla) {
        std::memcpy(transPtrTensorCpu[2].data_ptr(), &deviceVPtr[0], sizeof(uint64_t) * loadNum);
        std::memcpy(transPtrTensorCpu[3].data_ptr(), &hostVPtr[0], sizeof(uint64_t) * loadNum);
    }
    torch::Tensor transPtrTensor = transPtrTensorCpu.to(device);
    auto ret = mGSATransBackend->copy_trans(static_cast<void**>(transPtrTensor[1].data_ptr()),
                                            static_cast<void**>(transPtrTensor[0].data_ptr()),
                                            mKVSzieBytes, deviceKPtr.size());
    if (ret != 0) {
        mLogger.log(LogLevel::ERROR,
                    "|KVCache Prefetch| Decode step: %u, Rank: %d, layerID: %d, copy_trans error\n",
                    mDecodeStep, mRank, layerID);
    }
    if (!mUseMla) {
        mGSATransBackend->copy_trans(static_cast<void**>(transPtrTensor[3].data_ptr()),
                                     static_cast<void**>(transPtrTensor[2].data_ptr()),
                                     mKVSzieBytes, deviceKPtr.size());
    }
    ret = mGSATransBackend->synchronize();
    if (ret != 0) {
        mLogger.log(
            LogLevel::ERROR,
            "|KVCache Prefetch| Decode step: %u, Rank: %d, layerID: %d, synchronize error\n",
            mDecodeStep, mRank, layerID);
    }
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

void GSAPrefetchEngineC::SetKvCache(std::vector<torch::Tensor>& kvCaches,
                                    std::vector<uint64_t>& kCachesPtr,
                                    std::vector<uint64_t>& vCachesPtr,
                                    std::vector<uint64_t>& slabKCachesPtr,
                                    std::vector<uint64_t>& slabVCachesPtr, bool isNewTrans)
{
    if (mTensorElemSize != 0) { return; }
    mTensorElemSize = kvCaches[0].element_size();
    if (mUseMla) {
        mKVSzieBytes = kvCaches[0].element_size() * kvCaches[0][0].numel();
    } else {
        mKVSzieBytes = kvCaches[0].element_size() * kvCaches[0][0][0].numel();
    }
    mLogger.log(
        LogLevel::INFO,
        "Decode step: %u, |KVCache Prefetch| init kvcache mKVSzieBytes: %u, mTensorElemSize "
        "%u,\n",
        mDecodeStep, mKVSzieBytes, mTensorElemSize);
    mKcachePtr = kCachesPtr;
    mVcachePtr = vCachesPtr;
    mSlabKcachePtr = slabKCachesPtr;
    mSlabVcachePtr = slabVCachesPtr;
    std::ostringstream oss;
    oss << "Decode step: " << mDecodeStep << " Rnak: " << mRank
        << " kvcache ptr (layerID-kptr-kslabptr-vptr-vslabptr): ";
    for (int i = 0; i < mLayerNum; i++) {
        oss << "(" << i << "-" << mKcachePtr[i] << "-" << mSlabKcachePtr[i] << "-" << mVcachePtr[i]
            << "-" << mSlabVcachePtr[i] << ") ";
    }
    oss << "------\n";
    mLogger.log(LogLevel::INFO, oss.str().c_str());
    oss.str("");
    mKvCaches = kvCaches;
    mIsNewTrans = isNewTrans;
    auto device = mKvCaches[0].device();
    auto options = torch::TensorOptions().dtype(torch::kUInt64).device(device);
    mDeviceKPtrTensor = torch::empty({4, MAX_LOAD_NUM}, options);
    auto optionsCpu =
        torch::TensorOptions().dtype(torch::kUInt64).device("cpu").pinned_memory(true);
    mDeviceKPtrTensorCpu = torch::empty({4, MAX_LOAD_NUM}, optionsCpu);
}

void GSAPrefetchEngineC::RunAsyncPrefetchBsTrans(std::vector<std::string>& reqIDsInput,
                                                 std::vector<int>& topkLensInput,
                                                 std::vector<int>& bsIndexInput)
{
    mLogger.log(LogLevel::INFO,
                "Decode step: %u, |KVCache Prefetch| start async pretch batch size: %lu\n",
                mDecodeStep, reqIDsInput.size());
    mRunBsLen = reqIDsInput.size();
    if (mRunBsLen > mMaxBs) {
        mLogger.log(LogLevel::ERROR,
                    "Decode step: %u, |KVCache Prefetch| mRunBsLen %u, maxBs: %d\n", mDecodeStep,
                    mRunBsLen, mMaxBs);
        std::abort();
    }
    mReqIdList.clear();
    mReqIdList.assign(reqIDsInput.begin(), reqIDsInput.end());
    memcpy(mTopkLenList, topkLensInput.data(), sizeof(int) * mRunBsLen);
    memcpy(mBsIndexList, bsIndexInput.data(), sizeof(int) * mRunBsLen);
    mMutex.lock();
    mIsPrefetchDone = false;
    mMutex.unlock();
    mThreadPool->Enqueue(MutliBSThreadFun, this);
}

void GSAPrefetchEngineC::SetBlockTableInfo(torch::Tensor& blockTables, torch::Tensor& blockLengths,
                                           torch::Tensor& inputTopkBuf, int step)
{
    mLoadSuccessBlocks = blockTables;
    mSuccessTableLen = blockLengths;
    mUseTopkIdxs = inputTopkBuf.clone();
    mDecodeStep = step;
}

int GSAPrefetchEngineC::CallPrefetchProcessFun()
{
    auto start = std::chrono::high_resolution_clock::now();
    allNeedLoadBlock.clear();
    allMissIdxs.clear();
    for (size_t i = 0; i < mRunBsLen; i++) {
        if (mDocsTables.find(mReqIdList[i]) == mDocsTables.end() || mTopkLenList[i] <= 0) {
            mLogger.log(LogLevel::ERROR,
                        "Decode step: %u, |KVCache Prefetch| topk len is zero: %d\n", mDecodeStep,
                        mTopkLenList[i]);
            continue;
        }
        allMissIdxs.insert({mReqIdList[i], std::vector<std::vector<int>>(mLayerNum)});
        allNeedLoadBlock.insert({mReqIdList[i], std::vector<std::vector<int>>(mLayerNum)});
        RunOneBsPrefetch(mReqIdList[i], mTopkLenList[i], mBsIndexList[i], i);
    }
    auto begin = std::chrono::high_resolution_clock::now();
    TransKVCache();
    auto ret = mGSATransBackend->synchronize();
    if (ret != 0) {
        mLogger.log(LogLevel::ERROR,
                    "|KVCache Prefetch| Decode step: %u, Rank: %d, synchronize "
                    "error\n",
                    mDecodeStep, mRank);
    } else {
        mLogger.log(LogLevel::DEBUG,
                    "|KVCache Prefetch| Decode step: %u, Rank: %d, H2D kvcache "
                    "transfer success\n",
                    mDecodeStep, mRank);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    mLogger.log(
        LogLevel::INFO,
        "Decode step: %u, |KVCache Prefetch| Finish async pretch cost: %lu, KV load cost: %lu\n",
        mDecodeStep, duration.count(), duration1.count());
    return 0;
}

bool GSAPrefetchEngineC::GetPrefetchStatus() { return mIsPrefetchDone; }

void GSAPrefetchEngineC::SetPrefetchStatus(bool flag)
{
    mMutex.lock();
    mIsPrefetchDone = flag;
    mMutex.unlock();
}

void GSAPrefetchEngineC::SetModelRunningStatus(bool flag) { mStopPrefetch = flag; }

std::map<std::string, std::vector<std::vector<int>>> GSAPrefetchEngineC::ObtainLoadBlocks()
{
    return allNeedLoadBlock;
}

std::map<std::string, std::vector<std::vector<int>>> GSAPrefetchEngineC::ObtainMissIdxs()
{
    return allMissIdxs;
}

std::map<std::string, std::vector<std::map<int, int>>> GSAPrefetchEngineC::ObtainBlocksMap()
{
    return mBlocksMap;
}

std::map<std::string, std::vector<std::map<int, int>>> GSAPrefetchEngineC::ObtainDocsMap()
{
    return mDocsTables;
}
}  // namespace ucmprefetch
