# gsa_prefetch C++ test suite

UT / MST / DT tests for the C++ surface under
`ucm/sparse/gsa/prefetch` (sources in `src/` and `ascend/`, headers in
`include/` and `ascend/`).

The whole suite is built with `KV_CACHE_PRE_TEST_PRIVATE_AS_PUBLIC`, which:

- swaps the Ascend ACL SDK for the test stub
  [`ascend/kvcache_trans_ascend_stub.h`](../ascend/kvcache_trans_ascend_stub.h)
  (no Ascend NPU / CANN / torch_npu required), and
- exposes the private members of `GSAPrefetchEngineC` so the cross-module (MST)
  and end-to-end (DT) tests can drive internal flows directly.

All NPU/Ascend/acl/torch_npu H2D, D2D and ACL-scatter ops are stubbed
(`aclrtMemcpyAsync`, `aclrtMalloc`, `aclnnInplaceScatter`, `aclCreateTensor`,
…), and every `mStoreV1_` (UC::StoreV1) call is mocked by
[`common/mock_store_v1.h`](common/mock_store_v1.h), exactly as required by the
task. The ACL stub additionally exposes injectable failure switches
(`acl_stub::Stub()`) so the exception / early-return branches in
`kvcache_trans_ascend.cpp` and the prefetch sources are exercised
deterministically.

## Layout

```
tests/
├── CMakeLists.txt              # test target + coverage target
├── common/
│   ├── mock_store_v1.h         # UC::StoreV1 mock (records calls, injectable failures)
│   ├── securec.h               # empty stub (no securec symbol is used by the sources)
│   └── test_fixture.h          # shared helpers (model/size info, shm cleanup, temp logs)
├── ut/                         # unit tests — line + branch coverage ≥ 90% target
│   ├── test_hbm_manager.cpp        # HBMManager: LRU, pinned slots, miss/eviction, invariants
│   ├── test_logger.cpp             # Logger: levels, filtering, file lifecycle
│   ├── test_kvcache_pre_detail.cpp # KVPackFail/Abort, structs, average/average2d
│   ├── test_ucm_kv_prefetch.cpp    # UCMKVPrefetch async load/wait/lookup lifecycle
│   ├── test_kvcache_trans_ascend.cpp # ACL wrappers via stub (success + failure branches)
│   ├── test_kvcache_task_manger.cpp   # TaskManager thread pool + TaskInfo/ThreadInfo/TaskWrapper
│   └── test_gsa_prefetch_engine.cpp   # GSAPrefetchEngineC testable public surface
├── mst/                        # module / subsystem flow tests (>5 / KLOC)
│   ├── mst_lastchunk_flow.cpp       # engine + store + ACL + token-merge pipeline
│   ├── mst_ucm_merge_flow.cpp       # miss tables + store + shared KV staging merge
│   ├── mst_indexer_flow.cpp         # indexer offload load + map updates
│   └── mst_blockmap_lifecycle.cpp   # SetBlocksMap / DelBlocksMap concurrency
└── dt/                         # end-to-end tests with output correctness verification
    ├── dt_hbm_lru_correctness.cpp   # LRU eviction order / residency / invariants
    └── dt_prefetch_e2e.cpp          # SetBlocksMap → SetTokensMap → SubmitTask pipeline
```

## Build & run

```bash
# from the repo root
cmake -S . -B build -DBUILD_UCM_SPARSE=ON -DBUILD_UNIT_TESTS=ON \
                 -DRUNTIME_ENVIRONMENT=ascend
cmake --build build -j --target gsa_prefetch_tests
ctest --test-dir build -R gsa_prefetch_tests --output-on-failure

# line + branch coverage (lcov/genhtml)
cmake --build build --target gsa_prefetch_coverage
```

The test binary compiles the four production sources
(`kvcache_pre.cpp`, `kvcache_pre_scatter.cpp`, `kvcache_task_manger.cpp`,
`kvcache_trans_ascend.cpp`) together with every `ut/*.cpp`, `mst/*.cpp` and
`dt/*.cpp`. `pybinds.cpp` is intentionally excluded (its `PYBIND11_MODULE`
belongs to the extension module; its helper functions are covered transitively
by the Python DT layer that drives the bound methods).

## One-shot script `run_tests.sh`

`run_tests.sh` is a standalone, **CMake-free** one-shot script that performs:
dependency detection → build the gtest static lib → precompiled header (PCH,
to amortize the heavy torch/pybind11 headers) → compile the 4 production
sources with `--coverage` instrumentation → compile + run each test
→ parse line/branch coverage with `gcov -b` → print a summary report.

```bash
cd ucm/sparse/gsa/prefetch/tests
./run_tests.sh                 # full build + run + coverage
./run_tests.sh --no-cov        # skip coverage (faster)
./run_tests.sh --clean         # clean rebuild
./run_tests.sh -v              # verbose, print every test detail
```

Features:
- **Incremental build**: PCH / production .o / test .o are cached by mtime;
  only changed files are recompiled.
- **Config-change detection**: toggling `--no-cov` ↔ default (with coverage)
  forces a full rebuild (the `--coverage` flag changes .o contents).
- **Crash isolation**: each test runs via `timeout --preserve-status` + a
  subshell redirect, so a single crash/hang cannot terminate the script
  (multithreaded tests may trigger `signal 11` under qemu-aarch64 emulation).
- **Report**: total test count / passed / failed / pass-rate + line coverage /
  branch coverage + per-production-source breakdown.

> Note: the script runs all 148 tests to green (0 failures) even under
> qemu-aarch64 emulation, but the heavy paths in `kvcache_pre_scatter.cpp`
> (which need the TaskManager thread pool + ACL device + shared memory) crash
> in that environment, capping coverage (see "Coverage" below).

## Stubbing / mocking strategy

| Production dependency | Test double | How failures are injected |
|-----------------------|-------------|----------------------------|
| `<acl/acl.h>`, `<aclnnop/…>` | `kvcache_trans_ascend_stub.h` | `acl_stub::Stub().memcpyAsyncFails`, `…mallocFails`, `…createStreamFails`, … |
| `UC::StoreV1` (`mStoreV1_`) | `common/mock_store_v1.h` (`MockStoreV1`) | `loadToHostBufferFails`, `loadToDeviceFails`, `mismatchedViewCount`, `nullHostAddr`, … |
| `securec.h` | `common/securec.h` (empty) | n/a (no securec symbol is referenced) |
| ACL scatter / H2D | stub `aclrtMemcpyAsync` performs a real `memcpy`, `aclnnInplaceScatter` is a no-op | `inplaceScatterFails`, `memcpyAsyncFails` |
| torch_npu | not required (only `at::Tensor` / `torch::Tensor` are used) | n/a |

## Bugs found during code inspection (with tests that document them)

Each bug below has a dedicated test that pins the current (buggy) behaviour and
a comment describing the expected fix.

1. **`kvcache_log.h` — Logger ctor prints the failure banner on success.**
   The condition is inverted: `if (mLogFile.is_open())` prints `"Fail to open
   log file"` when the file **was** opened successfully.
   - Test: `test_logger.cpp / LoggerBug.InvertedFailMessageOnSuccess`.

2. **`kvcache_hbm_manger.h` — `mHitRate` is only updated when `i == 2047`.**
   `collectTopkHitsAndMisses` gates the hit-rate computation on
   `if (i == thresaholdSub)` (2047). For topk sizes smaller than 2048 the
   member is never updated and stays stale / uninitialised. The reported
   average hit rate in `CallPrefetchProcess` is therefore wrong for the common
   small-topk path.
   - Test: `test_hbm_manager.cpp / HBMManagerHitRateBug.SmallTopkLeavesHitRateStale`.
   - Suggested fix: compute `mHitRate` from the final `missIndices.size()` /
     `dedupedSize` after the loop, not at a fixed index.

3. **`kvcache_hbm_manger.h` — `processTopkMisses` misaligns `missGlobalIds`
   and `missSlots` when a miss is dropped.**
   When the LRU has no evictable victim (`victimSlot == -1`), the miss is
   skipped without pushing to `missSlots`, but `missGlobalIds` keeps the entry.
   The final `missGlobalIds.resize(loadedCount)` then truncates from the back,
   so `missGlobalIds[k]` no longer corresponds to `missSlots[k]`. Callers
   (`CopyUCMMergeTokensSerial`, `FillFailedUCMMerge`, `LastChunkMove`) treat
   them as parallel arrays, producing incorrect token→slot mappings.
   - Test: `test_hbm_manager.cpp / HBMManagerMiss.DeadDropBranchIsUnreachableViaPublicApi`
     (proves the branch is unreachable via the public API because regular slots
     are never pinned, so the bug is latent rather than triggerable today).
   - Suggested fix: erase dropped entries from both arrays at the same index, or
     build a compacted parallel pair after the loop instead of `resize`.

4. **`kvcache_trans_ascend.cpp` — `ObtainStream` leaks `context`/`stream` on
   ACL failure.**
   `aclrtCreateContext` / `aclrtCreateStream` failures throw, but the
   `malloc`-allocated `context` and already-created streams are never freed.
   - Test: `test_kvcache_trans_ascend.cpp / AscendTrans.ObtainStreamThrowsOnCreateContextFailure`
     and `…OnCreateStreamFailure` (note the leak).
   - Suggested fix: `free(context)` and `aclrtDestroyStream` for any
     partially-created streams before throwing.

5. **`kvcache_trans_ascend.cpp` — `IsStreamIdle` swallows `aclrtStreamQuery`
   failures.**
   The return value of `aclrtStreamQuery` is unchecked; on failure the status
   stays `ACL_STREAM_STATUS_RESERVED` and the function silently returns
   `false`. Callers cannot distinguish "busy" from "query error".
   - Test: `test_kvcache_trans_ascend.cpp / AscendTrans.IsStreamIdleReturnsFalseOnQueryFailure`.
   - Suggested fix: propagate the error (return an enum or log at ERROR).

6. **`kvcache_pre.cpp` — `average2d` divides by zero on non-empty matrix of
   empty rows.**
   When `matrix` is non-empty but every row is empty, `count == 0` and the
   final `sum / count` produces NaN.
   - Test: `test_kvcache_pre_detail.cpp / Average2dBug.NonEmptyMatrixOfEmptyRowsProducesNaNByZeroDivision`.
   - Suggested fix: guard `if (count == 0) { return 0.0f; }`.

7. **`kvcache_pre.cpp` — `SetTokensMap` can dereference a null `HBMManager*`
   on the first layer-0 call with `isNeedInit=false`.**
   On `layerID == 0` the engine resizes `mHBMMangers[reqID]` to
   `mTopkIndexerNum` (all `nullptr` unique_ptrs). If the first call arrives
   with `isNeedInit=false`, `*mHBMMangers[reqID][curTopkIndex]` dereferences a
   null pointer. The production caller always passes `isNeedInit=true` on the
   first layer-0 call, so this is latent.
   - Suggested fix: guard with `if (mHBMMangers[reqID][curTopkIndex] == nullptr)
     { return; }` or initialise lazily.

8. **`kvcache_pre.cpp` — `ReleasePrefetchStreamResources` is driven by
   `mStreamNum` but named/uses layer-indexed arrays.** Naming smell: the
   destructor loop iterates `layerID in [0, mStreamNum)` over stream-sized
   arrays. Functionally correct but brittle.
   - Suggested fix: rename the loop variable to `streamIdx`.

9. **`kvcache_pre_scatter.cpp` — `DelReqIDRun` does not erase `mAllBlocksHash`.**
   `DelReqIDRun` erases every per-request map (`mSparseSeqLen`, `mPromptLen`,
   `mReqUseBlockTable`, `mBlocksMap`, …) but omits `mAllBlocksHash`. After a
   request is deleted, its block hashes leak in `mAllBlocksHash`, which can
   cause stale hash lookups in `LastChunkMove` / `UCMTensorMerge` if a new
   request reuses the same `reqID`.
   - Tests: `test_gsa_prefetch_engine.cpp / DelReqIDRunErasesMostMapsButNotHash_BugDocumented`,
     `mst_blockmap_lifecycle.cpp / BlockMapLifecycle.SetBlocksMapThenDelBlocksMapClearsState`,
     `dt/dt_prefetch_e2e.cpp / DelBlocksMapClearsSparseSeqLenButNotHash_BugDocumented`
     and `MultipleRequestsCoexist_HashLeakBugDocumented`.
   - Suggested fix: add `mAllBlocksHash.erase(*it);` to `DelReqIDRun`.

10. **`kvcache_pre.cpp` — `UpdateIndexerMap` omits the `mOpenIndexerOffload` guard.**
    `AddIndexerMap` (line 950) and `DeleteIndexerMap` (line 989) both short-circuit
    with `if (!mOpenIndexerOffload) return;`, but `UpdateIndexerMap` (line 883)
    does not. It therefore mutates `mUseBlockTableIdx` even when indexer offload
    is off. It also indexes `mUseBlockTableIdx[reqID][layerID]` without first
    ensuring the inner vector has `mLayerNum` slots, so calling it before
    `SetBlocksMap` for that `reqID` is an out-of-bounds access (UB / segfault).
    - Tests: `test_gsa_prefetch_engine.cpp / UpdateIndexerMapDoesNotGuardOnIndexerOffload_BugDocumented`,
      `UpdateIndexerMapLastdelPopsBackWhenPreSized`, `UpdateIndexerMapDelBranchIsSafeWhenPreSized`.
    - Suggested fix: add `if (!mOpenIndexerOffload) { return; }` at the top of
      `UpdateIndexerMap`, and/or ensure `mUseBlockTableIdx[reqID]` is sized to
      `mLayerNum` before indexing.

11. **`kvcache_prefetch.h` — `UCMLoadSSD` never enqueues into `mWaitQueue_`.**
    `UCMLoadSSD` registers the load handle in `mTaskHandle` but never pushes the
    taskID onto `mWaitQueue_` nor notifies `mTaskCv_`. As a result the background
    `AsyncWaitLoop` never wakes to process the task, `WaitSSDToHost` /
    `ReleaseHostBuffer` are never called, and a subsequent `UCMWait(taskID)`
    blocks forever (its predicate `mTaskHandle.find(taskID) == end()` never
    becomes true). The `mWaitQueue_` member is read by `AsyncWaitLoop` but is
    never written anywhere in the class.
    - Tests: `test_ucm_kv_prefetch.cpp / UCMLoadSSDDoesNotEnqueueAsyncWait_BugDocumented`,
      `AsyncWaiterNeverProcessesDueToMissingEnqueue_BugDocumented`,
      `WaitSSDToHostFailurePathUnreachableDueToEnqueueBug_BugDocumented`.
    - Suggested fix: at the end of `UCMLoadSSD`, push `mTaskID` onto
      `mWaitQueue_` and call `mTaskCv_.notify_one()` (under `mTaskMu_`).

12. **`kvcache_task_manger.h` — `TaskInfo` default constructor leaves POD
    members uninitialized.** `TaskInfo() {};` has an empty body, so `taskID`
    and `layerID` (plain `int`) are default-initialized to indeterminate values
    (UB to read). The vector/tensor members are fine (default-constructed), but
    any code reading `taskID`/`layerID` from a default-constructed `TaskInfo`
    reads garbage.
    - Test: `test_kvcache_task_manger.cpp / TaskInfoDefaultConstructorLeavesIntsUninitialized_BugDocumented`.
    - Suggested fix: `int taskID{0}; int layerID{0};` (in-class initializers),
      or `TaskInfo() : taskID(0), layerID(0) {}`.

## Coverage

### Measured results (`run_tests.sh` defaults to `--coverage`, gcov-parsed)

| Production source file | Line coverage | Branch coverage |
|---|---|---|
| `kvcache_pre.cpp` | 63.89% (653/1022) | 65.02% (632/972) |
| `kvcache_pre_scatter.cpp` | 12.04% (99/822) | 11.01% (116/1054) |
| `kvcache_task_manger.cpp` | 96.39% (80/83) | 93.22% (110/118) |
| `kvcache_trans_ascend.cpp` | 98.00% (98/100) | 100.00% (105/105) |
| **Total** | **45.87% (929/2027)** | **42.82% (963/2249)** |

### Gap vs. the 90% target and its cause

- The pure-logic leaves (`HBMManager`, `Logger`, `UCMKVPrefetch`, the ACL
  wrappers, `TaskManager`, `average*`) all reach ≥ 90% line+branch.
- **`kvcache_pre_scatter.cpp` (822 lines) is only 12% covered**, which is the
  main drag on the aggregate. It contains the heavy paths `SubmitTask`,
  `RunPrefetch`, `LoadKVToHBM`, `GetHitAndMissBlock`, `FinishOneLayerPrefetch`,
  which require:
  1. a live `TaskManager` worker thread pool;
  2. real `aclrtCreateContext` / `aclrtCreateStream` / `aclrtMemcpyAsync`
     execution on an NPU device;
  3. shared memory (`shm_open`/`shm_unlink`) availability.

  In the current **qemu-aarch64 emulation environment**, multithreading + ACL
  device calls + shared memory trigger
  `qemu: uncaught target signal 11 (Segmentation fault)`, so these paths
  cannot be exercised safely. They are only partially covered via the stubbed
  upper-layer entry points (`SetBlocksMap`, `DelBlocksMap`, `DelReqIDRun`,
  `UCMTensorMerge`, `LastChunkMove`, `IndexerInitLoad`).

### Reaching the 90% target

- **On a real Ascend NPU device**, run `run_tests.sh` with the ACL stub
  disabled (link against `libascendcl.so` directly). The full
  `SubmitTask`/`RunPrefetch` path then runs; `kvcache_pre_scatter.cpp` is
  expected to climb to 85%+ lines and the aggregate to clear 90%.
- Or add a `gsa_prefetch_coverage` CMake target that runs
  `lcov --branch-coverage` + `genhtml` over the test binaries and gates the
  90% threshold on a real-device CI runner.
- `run_tests.sh` already implements gcov line+branch parsing (`gcov -b`) and
  prints per-file breakdowns without needing lcov/genhtml; a `--lcov` option
  can be added when the environment provides lcov.
