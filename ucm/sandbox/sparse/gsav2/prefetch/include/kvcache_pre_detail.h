/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */
#ifndef ATB_KV_CACHE_PRE_DETAIL_H
#define ATB_KV_CACHE_PRE_DETAIL_H

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>
#include "ucmstore_v1.h"

namespace ucmprefetch {

class GSAPrefetchEngineC;

// 典型 CPU cache line 大小（字节），用于隔离跨进程共享结构的伪共享。
constexpr size_t kKVCacheLineSize = 64;

// KVPack 跨进程共享布局（详见 kvcache_pre_detail.h 的使用者）。
struct alignas(kKVCacheLineSize) KVPackRankState {
    std::atomic<uint64_t> packedEpoch{0};
    std::atomic<uint64_t> h2dDoneEpoch{0};
};

struct alignas(kKVCacheLineSize) SharedKVPackHeader {
    std::atomic<uint32_t> initialized{0};
    std::atomic<int32_t> fatalError{0};
};

static_assert(std::atomic<uint64_t>::is_always_lock_free,
              "KV pack requires lock-free uint64_t atomics");
static_assert(std::atomic<uint32_t>::is_always_lock_free,
              "KV pack requires lock-free uint32_t atomics");
static_assert(std::atomic<int32_t>::is_always_lock_free,
              "KV pack requires lock-free int32_t atomics");

inline int KVPackFail(SharedKVPackHeader* header)
{
    header->fatalError.store(-1, std::memory_order_release);
    return -1;
}

// Coverage flush helper: flushes gcov data before abort (for death-test coverage).
// Declared without weak so the linker pulls __gcov_dump from libgcov.a; the
// -lgcov link flag is added by run_tests.sh only in COVERAGE mode.

#if defined(UCM_PREFETCH_COVERAGE)
extern "C" void __gcov_dump(void);
#endif

[[noreturn]] inline void KVPackAbort(SharedKVPackHeader* header)
{
    if (header != nullptr) { header->fatalError.store(-1, std::memory_order_release); }
#if defined(UCM_PREFETCH_COVERAGE)
    // Flush coverage data before abort so death-test lines are counted.
    __gcov_dump();
#endif
    std::abort();
}

// [H1] 带上下文的终止:abort 前先把 reason 直写 stderr——write(2) 绕过一切
// 用户态缓冲,进程 stdio 缓冲不 flush 也不会丢;mLogger 只写文件不写 stderr,
// 裸 abort 点原先死亡时现场两处(日志文件/stderr)都没有死因。fatalError
// 置位语义与 KVPackAbort 一致(跨 rank 可见)。原 KVPackAbort 保留不动,
// 已有前置日志的调用点零改动。
[[noreturn]] inline void KVPackAbortWithReason(SharedKVPackHeader* header, const char* reason)
{
    if (reason != nullptr) {
        const size_t len = std::strlen(reason);
        if (len > 0) {
            ssize_t written = ::write(STDERR_FILENO, reason, len);
            (void)written;
            written = ::write(STDERR_FILENO, "\n", 1);
            (void)written;
        }
    }
    if (header != nullptr) { header->fatalError.store(-1, std::memory_order_release); }
#if defined(UCM_PREFETCH_COVERAGE)
    // Flush coverage data before abort so death-test lines are counted.
    __gcov_dump();
#endif
    std::abort();
}

struct LastKVBlockAddress {
    std::uintptr_t kAddr{0};
    std::uintptr_t vAddr{0};
    std::uintptr_t indexerAddr{0};
    int indexerBlockIdx{-1};
};

struct UCMMergeBlockContext {
    const std::vector<size_t>* tokenIndices;
    size_t viewIndex;
    size_t tokenBase;
    bool isnoUCMblock;
    int blockIdx;
    const std::byte* kSrc;
    const std::byte* vSrc;
    bool owned;
};

struct UCMMergeTimeStatus {
    uint64_t preLoadUs = 0;
    uint64_t loadToHostUs = 0;
    uint64_t preMemcpyUs = 0;
    uint64_t memcpyUs = 0;
};

// address[i] 是 task[i] 对应 shard 的起始地址，布局仍为 [K block][V block]。
// 对象不允许复制或移动；析构时通知 owner 释放资源。
struct UCMMergeHostAddress {
    explicit UCMMergeHostAddress(GSAPrefetchEngineC& owner) : owner(owner) {}
    ~UCMMergeHostAddress();
    UCMMergeHostAddress(const UCMMergeHostAddress&) = delete;
    UCMMergeHostAddress& operator=(const UCMMergeHostAddress&) = delete;
    UCMMergeHostAddress(UCMMergeHostAddress&&) = delete;
    UCMMergeHostAddress& operator=(UCMMergeHostAddress&&) = delete;
    GSAPrefetchEngineC& owner;
    bool hasHostTask = false;
    std::vector<const std::byte*> addresses;
};

struct LastChunkSourceBlock {
    const std::byte* kBase{nullptr};
    const std::byte* vBase{nullptr};
};

using PrefetchTimePoint = std::chrono::high_resolution_clock::time_point;

struct PrefetchShmLayout {
    uint32_t versionSize;
    uint32_t countSize;
    uint32_t maxLoadNum;
    size_t missTokenSize;
    size_t loadSlotSize;
    size_t totalSize;
};

struct PrefetchTiming {
    uint64_t cpuMove = 0;
    uint64_t memcpy = 0;
    uint64_t sync = 0;
    uint64_t h2d = 0;
    uint64_t ops = 0;
};

struct TokenCopyTask {
    const std::byte* kSrc;
    const std::byte* vSrc;
    size_t dstOffset;
    uint64_t hbmSlot;
    bool owned;
};

}  // namespace ucmprefetch

#endif
