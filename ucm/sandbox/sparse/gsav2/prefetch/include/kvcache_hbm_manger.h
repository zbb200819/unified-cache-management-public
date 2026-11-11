/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */
#ifndef GSA_HBM_MANAGER_H
#define GSA_HBM_MANAGER_H

#include <algorithm>
#include <cassert>
#include <chrono>
#include <climits>
#include <cstdint>
#include <functional>
#include <iostream>
#include <kvcache_log.h>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#ifdef DEBUG
#define HBM_ASSERT(f) assert(f)
#else
#define HBM_ASSERT(f) ((void)0)
#endif
// ================= HBM 管理器 =================
class HBMManager {
public:
    HBMManager() : mInitialized(false) {}

    void init(int dramTensorCount, int hbmCapacity, int hbmImmutableSize, int topkSize,
              Logger& mLogger, bool preloadHbm = true, int maxTotalSlots = 32768)
    {
        HBM_ASSERT(dramTensorCount > 0);
        HBM_ASSERT(hbmCapacity > 0);
        HBM_ASSERT(hbmImmutableSize > 0);
        HBM_ASSERT(topkSize > 0);
        HBM_ASSERT(hbmImmutableSize <= hbmCapacity);
        HBM_ASSERT(topkSize <= hbmCapacity);
        mDramTensorCount = dramTensorCount;
        mHbmCapacity = hbmCapacity;
        mHbmImmutableSize = hbmImmutableSize;
        mTopkSize = topkSize;
        mMaxTotalSlots = (maxTotalSlots > hbmCapacity) ? maxTotalSlots : hbmCapacity;
        mPinnedSlotCount = 0;
        mLogger.log(
            LogLevel::DEBUG,
            "|HBM Manager| Init mDramTensorCount: %u, mHbmCapacity: %d, mHbmImmutableSize: %d, "
            "mTopkSize: %d, mMaxTotalSlots: %d \n",
            mDramTensorCount, mHbmCapacity, mHbmImmutableSize, mTopkSize, mMaxTotalSlots);
        mGlobalToSlot.assign(mDramTensorCount, -1);
        // mSlotToGlobal 仅覆盖常规 slot [0, mHbmCapacity-1]
        // pinned slot 的 gid 映射由公式推导: gid = slot - mHbmCapacity + mDramTensorCount
        mSlotToGlobal.assign(mHbmCapacity, -1);
        mLruPrev.assign(mMaxTotalSlots, -1);
        mLruNext.assign(mMaxTotalSlots, -1);
        mIsPinned.assign(mMaxTotalSlots, false);
        mAddedSlotFlag.assign(mMaxTotalSlots, 0);
        // 去重标记数组：覆盖两段连续 gid 域——常规 [0, mDramTensorCount) 与
        // pinned [mDramTensorCount, mDramTensorCount + mMaxTotalSlots - mHbmCapacity)
        // （pinned gid = slot - mHbmCapacity + mDramTensorCount）。
        mSeenGidFlag.assign(mDramTensorCount + mMaxTotalSlots - mHbmCapacity, 0);
        mSeenGidGeneration = 0;
        // 初始化常规 slot 映射
        if (preloadHbm) {
            // local slot [0, mHbmCapacity-1] 映射为 global 组合:
            //   slot [0, min(256, mDramTensorCount-1)]  -> global [0, ...] 顺序
            //   slot [min(256, mDramTensorCount), mHbmCapacity-1] -> global [mDramTensorCount-1 ...
            //   ...] 倒序
            // 守卫 mGlobalToSlot 下标 < mDramTensorCount、gid >= 0，避免短 prompt
            // (mDramTensorCount <= 256) 时堆缓冲区溢出。
            int slotIdx = 0;
            for (; slotIdx <= 256 && slotIdx < mHbmCapacity && slotIdx < mDramTensorCount;
                 ++slotIdx) {
                mGlobalToSlot[slotIdx] = slotIdx;
                mSlotToGlobal[slotIdx] = slotIdx;
            }
            /* loop1 已恒等映射 gid [0, identityCount)。loop2 倒序回填
             * 剩余 slot 时必须止步于该区间——若在 mHbmCapacity >
             * mDramTensorCount（短 prompt 常见）时回填 gid >= 0 的已映射高位
             * gid，恒等 slot 沦为幽灵（mSlotToGlobal[k]=k 但 mGlobalToSlot[k]
             * 指向逆序重映射的高位 slot），validateMapping 必败、LRU 驱逐误杀
             * 正身映射（多余 DRAM 重载）、buildTopkOutput 冷填不排除幽灵导致
             * 同一 global 双槽 gather（softmax 权重双计）。被排除的剩余 slot
             * 保持 -1（空位，冷填按需启用）。 */
            const int identityCount = slotIdx;
            for (int gid = mDramTensorCount - 1;
                 slotIdx < mHbmCapacity && gid >= identityCount; --gid, ++slotIdx) {
                mGlobalToSlot[gid] = slotIdx;
                mSlotToGlobal[slotIdx] = gid;
            }
        }
        // 构建 LRU 链表：仅常规 slots (0 <-> 1 <-> ... <-> mHbmCapacity-1)
        // pinned slots 在 processTopkWithDebug 中按需动态加入
        mLruHead = 0;
        mLruTail = mHbmCapacity - 1;
        for (int i = 0; i < mHbmCapacity; ++i) {
            mLruPrev[i] = (i == 0) ? -1 : i - 1;
            mLruNext[i] = (i == mHbmCapacity - 1) ? -1 : i + 1;
        }
        mAddedSlotGeneration = 0;
        mInitialized = true;
    }

    /* miss 换入与回填：victim 自 LRU 尾部驱逐（跳过 pinned）。missGlobalIds 与
     * missIndices 平行入栈（missIndices[j] 为第 j 个 miss 的去重后序号），回填
     * 直接取 missGlobalIds[j]，无需 deduped 序列二次查表。 */
    void processTopkMisses(std::vector<int>& topkSlotIds,
                           std::vector<int>& missGlobalIds, std::vector<int>& missSlots,
                           const std::vector<int>& missIndices, Logger& mLogger)
    {
        int missCount = missGlobalIds.size();
        int loadedCount = 0;
        if (missCount > 0) {
            for (int i = 0; i < missCount; ++i) {
                int victimSlot = mLruTail;
                while (victimSlot != -1 && mIsPinned[victimSlot]) {
                    victimSlot = mLruPrev[victimSlot];
                }
                if (victimSlot == -1) {
                    mLogger.log(LogLevel::ERROR,
                                "|HBM Manager| No evictable slot for miss gid: %d, "
                                "remaining %d misses dropped\n",
                                missGlobalIds[i], missCount - i);
                    continue;
                }
                int oldGid = mSlotToGlobal[victimSlot];
                if (oldGid >= 0 && oldGid < mDramTensorCount) {
                    mGlobalToSlot[oldGid] = -1;
                } else if (oldGid == -1) {
                    mLogger.log(LogLevel::WARNING, "|HBM Manager| Evicting unallocated slot: %d\n",
                                victimSlot);
                } else {
                    mLogger.log(
                        LogLevel::ERROR,
                        "|HBM Manager| Evicting slot with out-of-range gid: %d in slot: %d\n",
                        oldGid, victimSlot);
                }
                int newGid = missGlobalIds[i];
                mSlotToGlobal[victimSlot] = newGid;
                mGlobalToSlot[newGid] = victimSlot;
                missSlots.push_back(victimSlot);
                loadedCount++;
                removeFromList(victimSlot);
                insertToHead(victimSlot);
            }
            missGlobalIds.resize(loadedCount);
            for (int j = 0; j < loadedCount; ++j) {
                int slot = mGlobalToSlot[missGlobalIds[j]];
                topkSlotIds[missIndices[j]] = slot;
                if (slot >= 0 && slot < mMaxTotalSlots) {
                    mAddedSlotFlag[slot] = mAddedSlotGeneration;
                }
            }
        }
    }

    void buildTopkOutput(const std::vector<int>& topkSlotIds, int dedupedSize,
                         std::vector<int>& output)
    {
        output.reserve(mHbmImmutableSize);
        int realTopkSlotNum = 0;
        for (int i = 0; i < dedupedSize; ++i) {
            if (topkSlotIds[i] < 0) { continue; }
            output.push_back(topkSlotIds[i]);
            realTopkSlotNum += 1;
        }
        int slotsNeeded = mHbmImmutableSize - realTopkSlotNum;
        int count = 0;
        int curr = mLruHead;
        while (curr != -1 && count < slotsNeeded) {
            if (mAddedSlotFlag[curr] != mAddedSlotGeneration) {
                output.push_back(curr);
                mAddedSlotFlag[curr] = mAddedSlotGeneration;
                count++;
            }
            curr = mLruNext[curr];
        }
    }

    /* topk 查找主路径：去重 + hit/miss 分类 + miss 换入 + 输出组装。
     *
     * 去重：代际标记数组（mSeenGidFlag/mSeenGidGeneration）替代 unordered_set，
     * 覆盖常规 gid 域与 pinned gid 公式域两段连续区间；域外 gid（负值/超界
     * pinned）不参与去重，直通各自异常分支（日志条数随重复次数变化，可接受；
     * 超界 pinned 的原始 slot 值照写，行为不变）。代际推进到 255 时全量清零回 1，
     * 陈旧标记不可能与新代际别名命中。
     *
     * 去重与分类合流单循环：dedupedCount 即去重后序号（与原 topkSlotIds 下标
     * 语义一致，负 gid 占据序号但不出槽，保持原采样点），miss 的 gid 与序号
     * 平行入栈供 processTopkMisses 免查表回填。scratch 复用类成员：
     * mScratchSlotIds 同尺寸 assign 免分配、mScratchMissIndices clear 保留
     * capacity；manager 为 per-(req,layer) 独占对象，无跨线程共享。 */
    void processTopkWithDebug(const std::vector<int>& topkIds, std::vector<int>& output,
                              std::vector<int>& missGlobalIds, std::vector<int>& missSlots,
                              Logger& mLogger)
    {
        HBM_ASSERT(mInitialized);
        mTopkSize = topkIds.size();
        missGlobalIds.reserve(mTopkSize);
        missSlots.reserve(mTopkSize);
        mAddedSlotGeneration = (mAddedSlotGeneration == INT_MAX) ? 1 : mAddedSlotGeneration + 1;
        if (mSeenGidGeneration == 255) {
            std::fill(mSeenGidFlag.begin(), mSeenGidFlag.end(), 0);
            mSeenGidGeneration = 1;
        } else {
            ++mSeenGidGeneration;
        }
        const int gidBound = static_cast<int>(mSeenGidFlag.size());
        mScratchSlotIds.assign(mTopkSize, -1);
        mScratchMissIndices.clear();
        const int thresahold = 2048;
        const int thresaholdSub = 2047;
        int dedupedCount = 0;
        for (int i = 0; i < mTopkSize; ++i) {
            int gid = topkIds[i];
            if (gid >= 0 && gid < gidBound) {
                if (mSeenGidFlag[gid] == mSeenGidGeneration) {
                    mLogger.log(LogLevel::ERROR,
                                "|HBM Manager| Duplicate gid in topkIds: %d at index: %d, "
                                "skipping\n",
                                gid, i);
                    continue;
                }
                mSeenGidFlag[gid] = mSeenGidGeneration;
            }
            if (gid >= mDramTensorCount) {
                int transformedSlot = gid - mDramTensorCount + mHbmCapacity;
                if (transformedSlot >= mMaxTotalSlots) {
                    mLogger.log(LogLevel::ERROR,
                                "|HBM Manager| Pinned slot %d exceeds maxTotalSlots %d, gid: %d\n",
                                transformedSlot, mMaxTotalSlots, gid);
                    mScratchSlotIds[dedupedCount] = transformedSlot;
                    ++dedupedCount;
                    continue;
                }
                if (!mIsPinned[transformedSlot]) {
                    mIsPinned[transformedSlot] = true;
                    insertToHead(transformedSlot);
                    mPinnedSlotCount++;
                } else {
                    moveToHead(transformedSlot);
                }
                mScratchSlotIds[dedupedCount] = transformedSlot;
                mAddedSlotFlag[transformedSlot] = mAddedSlotGeneration;
                ++dedupedCount;
                continue;
            }
            if (gid < 0) {
                mLogger.log(LogLevel::ERROR, "|KVCache Prefetch| Topk is abnormal value: %d \n",
                            gid);
                ++dedupedCount;
                continue;
            }
            int slot = mGlobalToSlot[gid];
            if (slot != -1) {
                moveToHead(slot);
                mScratchSlotIds[dedupedCount] = slot;
                mAddedSlotFlag[slot] = mAddedSlotGeneration;
            } else {
                missGlobalIds.push_back(gid);
                mScratchMissIndices.push_back(dedupedCount);
                if ((int)mScratchMissIndices.size() == thresahold) { break; }
            }
            if (dedupedCount == thresaholdSub) {
                mHitRate = 1 - mScratchMissIndices.size() * 1.0 / thresahold;
            }
            ++dedupedCount;
        }
        processTopkMisses(mScratchSlotIds, missGlobalIds, missSlots, mScratchMissIndices, mLogger);
        buildTopkOutput(mScratchSlotIds, dedupedCount, output);
    }

    int getSlotForGlobal(int gid) const
    {
        if (gid >= mDramTensorCount) { return gid - mDramTensorCount + mHbmCapacity; }
        if (gid >= 0 && gid < mDramTensorCount) { return mGlobalToSlot[gid]; }
        return -1;
    }

    // pinned slot -> gid: 由公式推导，gid = slot - mHbmCapacity + mDramTensorCount
    int getGidForSlot(int slot) const
    {
        if (slot >= 0 && slot < mHbmCapacity) { return mSlotToGlobal[slot]; }
        if (slot >= mHbmCapacity && slot < mMaxTotalSlots && mIsPinned[slot]) {
            return slot - mHbmCapacity + mDramTensorCount;
        }
        return -1;
    }

    bool isInitialized() const { return mInitialized; }
    int getTopkSize() const { return mTopkSize; }
    int getHbmCapacity() const { return mHbmCapacity; }
    int getHbmImmutableSize() const { return mHbmImmutableSize; }
    int getDramTensorCount() const { return mDramTensorCount; }
    int getPinnedSlotCount() const { return mPinnedSlotCount; }
    int getMaxTotalSlots() const { return mMaxTotalSlots; }

    bool validateLruChain() const
    {
        int expectedCount = mHbmCapacity + mPinnedSlotCount;
        int curr = mLruHead;
        int count = 0;
        int safetyLimit = mMaxTotalSlots + 10;

        while (curr != -1 && count < safetyLimit) {
            count++;
            curr = mLruNext[curr];
        }

        if (count >= safetyLimit || count != expectedCount) return false;

        curr = mLruTail;
        count = 0;
        while (curr != -1 && count < safetyLimit) {
            count++;
            curr = mLruPrev[curr];
        }

        return count == expectedCount;
    }

    bool validateMapping() const
    {
        // 检查常规 slot 映射
        for (int slot = 0; slot < mHbmCapacity; ++slot) {
            int gid = mSlotToGlobal[slot];
            if (gid == -1) continue;
            if (gid < -1 || gid >= mDramTensorCount) return false;
            if (mGlobalToSlot[gid] != slot) return false;
        }
        // 检查已初始化的 pinned slot 映射 (gid 由公式推导)
        for (int slot = mHbmCapacity; slot < mMaxTotalSlots; ++slot) {
            if (!mIsPinned[slot]) continue;
            int expectedGid = slot - mHbmCapacity + mDramTensorCount;
            // pinned slot 的 gid 必须与公式一致
            if (expectedGid < mDramTensorCount) return false;
        }
        return true;
    }

    // mSlotToGlobal 仅覆盖常规 slot [0, mHbmCapacity-1]
    // LastChunkMove 等外部代码仅遍历此范围
    std::vector<int> mSlotToGlobal;
    std::vector<int> mGlobalToSlot;
    float mHitRate;

private:
    int mDramTensorCount;
    int mHbmCapacity;
    int mHbmImmutableSize;
    int mTopkSize;
    int mPinnedSlotCount;
    int mMaxTotalSlots;

    std::vector<int> mLruPrev;
    std::vector<int> mLruNext;
    std::vector<bool> mIsPinned;
    std::vector<int> mAddedSlotFlag;
    // 去重标记数组（代际模式；与 mAddedSlotFlag 的 INT_MAX 翻转不同：uint8 空间
    // 小，代际到 255 时全量清零回 1，杜绝 8bit 空间的陈旧别名命中）
    std::vector<uint8_t> mSeenGidFlag;
    // 每调用 scratch：同尺寸 assign / clear 复用，免逐步堆分配
    std::vector<int> mScratchSlotIds;
    std::vector<int> mScratchMissIndices;

    int mLruHead;
    int mLruTail;
    int mAddedSlotGeneration;
    int mSeenGidGeneration;
    bool mInitialized;

    inline void removeFromList(int slot)
    {
        int p = mLruPrev[slot];
        int n = mLruNext[slot];
        if (p != -1)
            mLruNext[p] = n;
        else
            mLruHead = n;
        if (n != -1)
            mLruPrev[n] = p;
        else
            mLruTail = p;
    }

    inline void insertToHead(int slot)
    {
        mLruPrev[slot] = -1;
        mLruNext[slot] = mLruHead;
        if (mLruHead != -1) mLruPrev[mLruHead] = slot;
        mLruHead = slot;
        if (mLruTail == -1) mLruTail = slot;
    }

    inline void moveToHead(int slot)
    {
        if (slot == mLruHead) return;
        removeFromList(slot);
        insertToHead(slot);
    }
};

#endif  // GSA_HBM_MANAGER_H