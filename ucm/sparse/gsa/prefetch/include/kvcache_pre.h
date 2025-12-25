#ifndef ATB_KV_CACHE_PRE_H
#define ATB_KV_CACHE_PRE_H
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <future>
#include <iostream>
#include <kvcache_log.h>
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
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "../../../../store/ucmstore.h"
#include "kvcache_safe_queue.h"
#include "kvcache_thread.h"
#include "kvcache_trans.h"

namespace py = pybind11;

namespace ucmprefetch {
typedef struct {
    int topkLen;
    std::string reqID;
    int layerID;
    int topkIndex;
    int bsIndex;
} PrefetchReqInfo;

void MutliBSThreadFun(void* args);

void RunQueuePolling(void* args);

void CallPrefetchBSLayer(void* args, PrefetchInfoLayer oneBsInfo);

class __attribute__((visibility("hidden"))) GSAPrefetchEngineC {
private:
    std::map<std::string, std::vector<std::map<int, int>>> mDocsTables;
    std::map<std::string, std::vector<std::map<int, int>>> mBlocksMap;
    torch::Tensor mLoadSuccessBlocks;
    torch::Tensor mSuccessTableLen;
    torch::Tensor mUseTopkIdxs;
    int mLayerNum;
    int mRank = -1;
    uint32_t mMaxBs = 30;
    std::vector<std::string> mReqIdList;
    int* mTopkLenList = NULL;
    int* mBsIndexList = NULL;
    uint32_t mRunBsLen = 0;
    bool mIsLog = false;
    bool mIsPrefetchDone = true;
    bool mUseMla = false;
    Logger mLogger;
    ThreadPool* mThreadPool;
    uint32_t mDecodeStep = 0;
    uint32_t mMaxTopkLen = 0;
    uint32_t mMaxBlocksLen = 0;
    std::unordered_set<std::string> mDelSeqIds;
    std::map<std::string, std::vector<std::vector<int>>> allNeedLoadBlock;
    std::map<std::string, std::vector<std::vector<int>>> allMissIdxs;
    std::map<std::string, int> mPromptLen;
    UC::CCStore<>* mStore = nullptr;
    std::vector<torch::Tensor> mKvCaches;
    uint32_t mTensorElemSize = 0;
    uint32_t mKVSzieBytes = 0;
    std::map<std::string, std::vector<std::vector<int>>> mPrefetchIdx;
    TransBackend* mGSATransBackend;
    std::vector<uint64_t> mKcachePtr;
    std::vector<uint64_t> mVcachePtr;
    std::vector<uint64_t> mSlabKcachePtr;
    std::vector<uint64_t> mSlabVcachePtr;
    bool mIsNewTrans = false;
    std::map<std::string, std::vector<int>> mAllReqIdSlots;
    torch::Tensor mDeviceKPtrTensor;
    torch::Tensor mDeviceKPtrTensorCpu;
    KvCacheSafeQueue mLoadQueue;
    bool mIsPrefetchRunning = true;

public:
    std::mutex mMutex;
    bool mStopPrefetch = false;

private:
    void LoadKVToHBM(std::vector<int> loadNPUBlockIDs, std::vector<int> missIdxs, int layerID,
                     std::string reqID);

    void TransKVCache();

    void GetHitAndMissBlock(PrefetchReqInfo oneBsInfo, std::unordered_set<int>& hitBlocks,
                            std::map<int, int>& hitBlocksIdx, std::vector<int>& missIdxs);

    void RunPrefetchH2D(PrefetchReqInfo oneBsInfo, std::unordered_set<int>& hitBlocks,
                        std::map<int, int>& hitBlocksIdx, std::vector<int>& missIdxs);

    void RunOneBsPrefetch(std::string reqID, int topkLen, int bsIndex, int topkIndex);

public:
    ~GSAPrefetchEngineC();

    GSAPrefetchEngineC(torch::Tensor& loadSuccessBlocks, torch::Tensor& successTableLen,
                       bool useMla, bool isLog, int rank);

    void SetBlocksMap(std::string reqID, std::vector<int>& blockTableList,
                      std::vector<int>& remainIdx, std::vector<int>& prefetchIdx, int maxIdx,
                      std::vector<int>& slots);

    void SetBlocksMapMultiLayer(std::string reqID, std::vector<std::map<int, int>>& remainMap,
                                std::vector<std::map<int, int>>& prefetchMap, int maxIdx,
                                std::vector<int>& slots);

    void SetKvCache(std::vector<torch::Tensor>& kvCaches, std::vector<uint64_t>& kCachesPtr,
                    std::vector<uint64_t>& vCachesPtr, std::vector<uint64_t>& slabKCachesPtr,
                    std::vector<uint64_t>& slabVCachesPtr, bool isNewTrans);

    void CheckInputIndex(uint32_t maxLen, uint32_t index);

    void AddBlocksMap(std::string reqID, int idx, int blockID);

    void DelBlocksMap(std::string reqID);

    void DelReqIDRun();

    void SetBlockTableInfo(torch::Tensor& blockTables, torch::Tensor& blockLengths,
                           torch::Tensor& inputTopkBuf, int step);

    void RunAsyncPrefetchBsTrans(std::vector<std::string>& reqIDsInput,
                                 std::vector<int>& topkLensInput, std::vector<int>& bsIndexInput);

    void AddPrefetchTask(uint32_t layerID, std::vector<std::string>& reqIDList,
                         std::vector<std::vector<int32_t>>& topkList, std::vector<int>& bsIndexList,
                         std::vector<int>& topkLenList);

    void RunPrefetchBSLayer(PrefetchInfoLayer oneBsInfo);

    void RunOneBsPrefetchLayer(std::string reqID, int bsIndex, int layerID,
                               std::vector<int>& oneFreeBlockTable, std::vector<int>& missIdxs,
                               std::vector<int>& topkList);

    void TransKVCacheLayer(int layerID, std::map<std::string, std::vector<int>>& batchLoadBlock,
                           std::map<std::string, std::vector<int>>& batchMissIdxs);

    void QueuePolling();

    int CallPrefetchProcessFun();

    void PrintMap(std::string reqID, int i);

    void PrintVector(std::vector<int>& vec, int layerID, std::string reqID, std::string name);

    bool GetPrefetchStatus();

    void SetPrefetchStatus(bool flag);

    void SetModelRunningStatus(bool flag);

    std::map<std::string, std::vector<std::vector<int>>> ObtainLoadBlocks();

    std::map<std::string, std::vector<std::vector<int>>> ObtainMissIdxs();

    std::map<std::string, std::vector<std::map<int, int>>> ObtainBlocksMap();

    std::map<std::string, std::vector<std::map<int, int>>> ObtainDocsMap();
};

}  // namespace ucmprefetch

#endif
