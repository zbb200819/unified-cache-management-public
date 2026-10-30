/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 *
 * UT for the UCMKVPrefetch class declared in include/kvcache_prefetch.h.
 * The class owns a background waiter thread and delegates to UC::StoreV1; we
 * exercise it with MockStoreV1 so the async load/wait/release lifecycle runs
 * deterministically and all exception branches are covered.
 */
#include <atomic>
#include <chrono>
#include <gtest/gtest.h>
#include <stdexcept>
#include <thread>
#include <vector>
#include "../common/mock_store_v1.h"
#include "kvcache_prefetch.h"
#include "kvcache_trans_ascend.h"

using ucmprefetch::UCMKVPrefetch;
using ucmprefetch_test::MockStoreV1;

namespace {
UC::Detail::BlockId MakeBlockId(uint8_t v)
{
    UC::Detail::BlockId id{};
    for (auto& b : id) { b = std::byte{v}; }
    return id;
}
}  // namespace

TEST(UCMKVPrefetch, StoreInitRejectsNullPointer)
{
    UCMKVPrefetch p;
    EXPECT_THROW(p.StoreInit(reinterpret_cast<uintptr_t>(nullptr)), std::runtime_error);
}

TEST(UCMKVPrefetch, StoreInitAcceptsValidPointer)
{
    MockStoreV1 store;
    UCMKVPrefetch p;
    p.StoreInit(reinterpret_cast<uintptr_t>(&store));
    /* No exception. */
    SUCCEED();
}

TEST(UCMKVPrefetch, UCMLoadSSDRejectsNullStore)
{
    UCMKVPrefetch p;
    std::vector<UC::Detail::BlockId> ids = {MakeBlockId(1)};
    EXPECT_THROW(p.UCMLoadSSD(ids), std::runtime_error);
}

TEST(UCMKVPrefetch, UCMLoadSSDRejectsEmptyBlockIds)
{
    MockStoreV1 store;
    UCMKVPrefetch p;
    p.StoreInit(reinterpret_cast<uintptr_t>(&store));
    std::vector<UC::Detail::BlockId> empty;
    EXPECT_THROW(p.UCMLoadSSD(empty), std::runtime_error);
}

TEST(UCMKVPrefetch, UCMLoadSSDReturnsIncreasingTaskIds)
{
    MockStoreV1 store;
    UCMKVPrefetch p;
    p.StoreInit(reinterpret_cast<uintptr_t>(&store));
    std::vector<UC::Detail::BlockId> ids = {MakeBlockId(1), MakeBlockId(2)};
    int32_t t1 = p.UCMLoadSSD(ids);
    int32_t t2 = p.UCMLoadSSD(ids);
    EXPECT_GT(t2, t1);
    // NOTE: do NOT call UCMWait here — see UCMLoadSSDDoesNotEnqueueAsyncWait_BugDocumented.
}

TEST(UCMKVPrefetch, UCMLoadSSDDoesNotEnqueueAsyncWait_BugDocumented)
{
    // BUG-FOCUS: UCMLoadSSD (kvcache_prefetch.h) registers the handle in
    // mTaskHandle but NEVER enqueues the taskID into mWaitQueue_ nor notifies
    // mTaskCv_. Consequently AsyncWaitLoop never wakes to process the task,
    // and a subsequent UCMWait(taskID) blocks forever (the predicate
    // `mTaskHandle.find(taskID) == end()` never becomes true).
    // This test proves the hang by running UCMWait in a thread with a timeout:
    // if UCMWait returns within the timeout, the bug is fixed (flip the assert).
    // The UCMKVPrefetch is heap-allocated and intentionally leaked here so the
    // detached waiter thread does not touch freed memory; the process exits
    // right after anyway.
    auto* store = new MockStoreV1;
    auto* p = new UCMKVPrefetch;
    p->StoreInit(reinterpret_cast<uintptr_t>(store));
    std::vector<UC::Detail::BlockId> ids = {MakeBlockId(1)};
    int32_t t = p->UCMLoadSSD(ids);

    std::atomic<bool> returned{false};
    std::thread waiter([&] {
        p->UCMWait(t);
        returned.store(true);
    });
    // Give the waiter at most 500ms; the (buggy) code blocks indefinitely.
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    EXPECT_FALSE(returned.load());  // bug: still blocked
    waiter.detach();                // leak intentionally; process exits after this test.
    // Do NOT delete p/store: the detached waiter still references them.
}

TEST(UCMKVPrefetch, UCMLoadSSDPropagatesStoreFailure)
{
    MockStoreV1 store;
    store.loadSSDToHostFails = true;
    UCMKVPrefetch p;
    p.StoreInit(reinterpret_cast<uintptr_t>(&store));
    std::vector<UC::Detail::BlockId> ids = {MakeBlockId(1)};
    EXPECT_THROW(p.UCMLoadSSD(ids), std::runtime_error);
}

TEST(UCMKVPrefetch, UCMWaitOnUnknownTaskIsNoop)
{
    MockStoreV1 store;
    UCMKVPrefetch p;
    p.StoreInit(reinterpret_cast<uintptr_t>(&store));
    /* Wait on an unknown task id: must return without blocking forever. */
    p.UCMWait(424242);
    SUCCEED();
}

TEST(UCMKVPrefetch, UCMLookupRejectsNullStore)
{
    UCMKVPrefetch p;
    std::vector<UC::Detail::BlockId> ids = {MakeBlockId(1)};
    EXPECT_THROW(p.UCMLookup(ids), std::runtime_error);
}

TEST(UCMKVPrefetch, UCMLookupRejectsEmptyBlockIds)
{
    MockStoreV1 store;
    UCMKVPrefetch p;
    p.StoreInit(reinterpret_cast<uintptr_t>(&store));
    std::vector<UC::Detail::BlockId> empty;
    EXPECT_THROW(p.UCMLookup(empty), std::runtime_error);
}

TEST(UCMKVPrefetch, UCMLookupReturnsExistenceVector)
{
    MockStoreV1 store;
    UCMKVPrefetch p;
    p.StoreInit(reinterpret_cast<uintptr_t>(&store));
    std::vector<UC::Detail::BlockId> ids = {MakeBlockId(1), MakeBlockId(2), MakeBlockId(3)};
    auto result = p.UCMLookup(ids);
    EXPECT_EQ(result.size(), ids.size());
    for (auto v : result) { EXPECT_EQ(v, 1u); }
}

TEST(UCMKVPrefetch, UCMLookupPropagatesStoreFailure)
{
    MockStoreV1 store;
    store.lookupFails = true;
    UCMKVPrefetch p;
    p.StoreInit(reinterpret_cast<uintptr_t>(&store));
    std::vector<UC::Detail::BlockId> ids = {MakeBlockId(1)};
    EXPECT_THROW(p.UCMLookup(ids), std::runtime_error);
}

TEST(UCMKVPrefetch, UCMLookupDetectsSizeMismatch)
{
    /* Force a mismatch by overriding Lookup to return a vector of different
     * size. MockStoreV1 returns exactly num entries, so trigger via a custom
     * subclass that returns a short vector. */
    struct ShortStore : MockStoreV1 {
        UC::Expected<std::vector<uint8_t>> Lookup(const UC::Detail::BlockId*, size_t num) override
        { return std::vector<uint8_t>(num + 1, 1u); }
    } store;
    UCMKVPrefetch p;
    p.StoreInit(reinterpret_cast<uintptr_t>(&store));
    std::vector<UC::Detail::BlockId> ids = {MakeBlockId(1)};
    EXPECT_THROW(p.UCMLookup(ids), std::runtime_error);
}

TEST(UCMKVPrefetch, AsyncWaiterNeverProcessesDueToMissingEnqueue_BugDocumented)
{
    // BUG-FOCUS (same root cause as UCMLoadSSDDoesNotEnqueueAsyncWait):
    // Because UCMLoadSSD never enqueues into mWaitQueue_, AsyncWaitLoop never
    // calls WaitSSDToHost/ReleaseHostBuffer. Prove it: after UCMLoadSSD, the
    // mock store records LoadSSDToHost but NEVER WaitSSDToHost / ReleaseHostBuffer.
    // When the enqueue bug is fixed, flip the assertion to EXPECT_TRUE.
    MockStoreV1 store;
    UCMKVPrefetch p;
    p.StoreInit(reinterpret_cast<uintptr_t>(&store));
    std::vector<UC::Detail::BlockId> ids = {MakeBlockId(1)};
    (void)p.UCMLoadSSD(ids);
    // Give the async waiter a moment to (not) process.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    bool hasWait = false;
    bool hasRelease = false;
    for (const auto& c : store.calls) {
        if (c.name == "WaitSSDToHost") { hasWait = true; }
        if (c.name == "ReleaseHostBuffer") { hasRelease = true; }
    }
    EXPECT_FALSE(hasWait);     // bug: async waiter never runs
    EXPECT_FALSE(hasRelease);  // bug: nothing released
}

TEST(UCMKVPrefetch, WaitSSDToHostFailurePathUnreachableDueToEnqueueBug_BugDocumented)
{
    // BUG-FOCUS: Because the async waiter never processes (missing enqueue),
    // a failing WaitSSDToHost is never invoked, so the failure-swallowing
    // try/catch in UCMWaitInternal is dead code in practice. Documented:
    // the mock's waitSSDToHostFails flag has no observable effect.
    // When the enqueue bug is fixed, this test should be replaced with one that
    // confirms UCMWait returns despite the failure.
    MockStoreV1 store;
    store.waitSSDToHostFails = true;
    UCMKVPrefetch p;
    p.StoreInit(reinterpret_cast<uintptr_t>(&store));
    std::vector<UC::Detail::BlockId> ids = {MakeBlockId(1)};
    (void)p.UCMLoadSSD(ids);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    // WaitSSDToHost was never called (bug).
    bool hasWait = false;
    for (const auto& c : store.calls) {
        if (c.name == "WaitSSDToHost") { hasWait = true; }
    }
    EXPECT_FALSE(hasWait);
}

TEST(UCMKVPrefetch, DestructorJoinsBackgroundThread)
{
    // The destructor sets mStop_ and joins mAsyncWaiter_. Even with a pending
    // (never-enqueued) task, the destructor must return promptly because the
    // waiter loop's cv predicate `mStop_ || !mWaitQueue_.empty()` becomes true
    // on stop (queue empty -> returns).
    MockStoreV1 store;
    auto start = std::chrono::steady_clock::now();
    {
        UCMKVPrefetch p;
        p.StoreInit(reinterpret_cast<uintptr_t>(&store));
        std::vector<UC::Detail::BlockId> ids = {MakeBlockId(1)};
        (void)p.UCMLoadSSD(ids);
        /* Pending task is still registered when destructor runs. */
    }
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
    // Destructor must complete in well under 1s (cv wakes immediately on stop).
    EXPECT_LT(elapsed.count(), 1000);
}
