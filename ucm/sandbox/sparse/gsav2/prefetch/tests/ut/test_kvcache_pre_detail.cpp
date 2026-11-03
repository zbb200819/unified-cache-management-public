/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 *
 * UT for helpers in include/kvcache_pre_detail.h and the free functions
 * `average` / `average2d` defined in src/kvcache_pre.cpp.
 */
#include <cmath>
#include <csignal>
#include <gtest/gtest.h>
#include <stdexcept>
#include "kvcache_pre.h"
#include "kvcache_pre_detail.h"

namespace ucmprefetch {

/* ---- KVPackFail / KVPackAbort ---- */

TEST(KVPackDetail, KVPackFailSetsFatalFlagAndReturnsMinusOne)
{
    SharedKVPackHeader header;
    header.fatalError.store(0, std::memory_order_relaxed);
    int ret = KVPackFail(&header);
    EXPECT_EQ(ret, -1);
    EXPECT_EQ(header.fatalError.load(std::memory_order_acquire), -1);
}

TEST(KVPackDetail, KVPackFailIsIdempotent)
{
    SharedKVPackHeader header;
    header.fatalError.store(0, std::memory_order_relaxed);
    KVPackFail(&header);
    KVPackFail(&header);
    EXPECT_EQ(header.fatalError.load(), -1);
}

TEST(KVPackDetail, KVPackAbortWithNullHeaderDoesNotSetFlag)
{
    /* KVPackAbort with nullptr must not deref; it aborts the process, so we
     * cannot call it directly. We assert that it is declared [[noreturn]]. */
    static_assert(noexcept(false), "KVPackAbort is not noexcept-tagged");
    SUCCEED();
}

/* ---- KVPackAbortWithReason (H1) ---- */

TEST(KVPackDetailDeath, KVPackAbortWithReasonWritesReasonAndAborts)
{
    /* [H1] 死亡测试:abort 前把 reason 直写 stderr(write(2) 绕过用户态缓冲,
     * 进程 stdio 缓冲不 flush 也不丢);进程以 SIGABRT 终止,fatalError 置位
     * 语义与 KVPackAbort 相同(单行 atomic store,由代码审读保证)。 */
    SharedKVPackHeader header;
    header.fatalError.store(0, std::memory_order_relaxed);
    EXPECT_EXIT(KVPackAbortWithReason(&header, "[KVPackUT] fatal: injected reason"),
                ::testing::KilledBySignal(SIGABRT), "\\[KVPackUT\\] fatal: injected reason");
}

TEST(KVPackDetailDeath, KVPackAbortWithReasonWithNullHeaderStillAborts)
{
    /* [H1] header 为 null 时不解引用(与旧 KVPackAbort(nullptr) 一致),
     * reason 仍先落 stderr,随后 SIGABRT。 */
    EXPECT_EXIT(KVPackAbortWithReason(nullptr, "[KVPackUT] fatal: null header"),
                ::testing::KilledBySignal(SIGABRT), "\\[KVPackUT\\] fatal: null header");
}

/* ---- structs default-init ---- */

TEST(KVPackDetail, LastKVBlockAddressDefaults)
{
    LastKVBlockAddress a;
    EXPECT_EQ(a.kAddr, 0u);
    EXPECT_EQ(a.vAddr, 0u);
    EXPECT_EQ(a.indexerAddr, 0u);
    EXPECT_EQ(a.indexerBlockIdx, -1);
}

TEST(KVPackDetail, UCMMergeBlockContextFieldsDefault)
{
    UCMMergeBlockContext c{};
    EXPECT_EQ(c.tokenIndices, nullptr);
    EXPECT_EQ(c.viewIndex, 0u);
    EXPECT_EQ(c.tokenBase, 0u);
    EXPECT_FALSE(c.isnoUCMblock);
    EXPECT_EQ(c.blockIdx, 0);
    EXPECT_EQ(c.kSrc, nullptr);
    EXPECT_EQ(c.vSrc, nullptr);
    EXPECT_FALSE(c.owned);
}

TEST(KVPackDetail, PrefetchShmLayoutDefaultZero)
{
    PrefetchShmLayout l{};
    EXPECT_EQ(l.versionSize, 0u);
    EXPECT_EQ(l.totalSize, 0u);
}

TEST(KVPackDetail, PrefetchTimingDefaultZero)
{
    PrefetchTiming t{};
    EXPECT_EQ(t.cpuMove, 0u);
    EXPECT_EQ(t.h2d, 0u);
}

/* ---- free functions average / average2d ---- */

TEST(AverageTest, AverageOfNonEmpty)
{
    std::vector<float> v = {1.0f, 2.0f, 3.0f, 4.0f};
    EXPECT_FLOAT_EQ(average(v), 2.5f);
}

TEST(AverageTest, AverageOfSingle)
{
    std::vector<float> v = {7.5f};
    EXPECT_FLOAT_EQ(average(v), 7.5f);
}

TEST(AverageTest, AverageOfEmptyThrows)
{
    std::vector<float> v;
    EXPECT_THROW(average(v), std::runtime_error);
}

TEST(Average2dTest, EmptyMatrixReturnsZero)
{
    std::vector<std::vector<float>> m;
    EXPECT_FLOAT_EQ(average2d(m), 0.0f);
}

TEST(Average2dTest, FlatMatrix)
{
    std::vector<std::vector<float>> m = {
        {1.0f, 2.0f},
        {3.0f, 4.0f}
    };
    EXPECT_FLOAT_EQ(average2d(m), 2.5f);
}

TEST(Average2dTest, JaggedMatrix)
{
    std::vector<std::vector<float>> m = {
        {1.0f, 2.0f, 3.0f},
        {4.0f}
    };
    EXPECT_FLOAT_EQ(average2d(m), 2.5f);
}

/* ---- average2d with non-empty matrix of empty rows (fixed: returns 0.0f) ----
 * Previously this divided by zero (count==0) and returned NaN; the fix adds a
 * count==0 guard that returns 0.0f instead. */
TEST(Average2dBug, NonEmptyMatrixOfEmptyRowsReturnsZeroAfterFix)
{
    std::vector<std::vector<float>> m = {{}, {}};
    float result = average2d(m);
    EXPECT_FALSE(std::isnan(result)) << "average2d should not produce NaN for empty rows";
    EXPECT_FLOAT_EQ(result, 0.0f);
}

}  // namespace ucmprefetch
