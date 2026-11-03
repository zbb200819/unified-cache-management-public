# GSA/UCM 第三轮代码级风险审阅(专项:gsav2)

- **分析对象**:`ucm/sandbox/sparse/gsav2`(gsa.py、prefetch/ 全部 C++/Python 源码、共享内存模块、Ascend 传输层),含关联的 `ucm/store/detail/template/task_wrapper.h`(只读参考)
- **分析日期**:2026-11-03
- **审阅方法**:三路并行——人工通读支撑文件(TaskManager/HBMManager/传输层/pybinds/共享内存模块)+ 两路全文深扫(gsa.py 约 2929 行;kvcache_pre.cpp + kvcache_pre_scatter.cpp 约 3800 行);全部行号为**当前 HEAD(`b85c3b1`)实测行号**
- **与既有清单的关系**:A~G 类(21 项)与 H 类(2 项)均已修复,本轮**不重复报告**已修复项;本轮发现全部为**新识别**风险,编号 I1~I22
- **严重程度定义**(同主清单):
  - **P0**:确定性崩溃或确定性内存踩写,可复现触发条件明确
  - **P1**:条件性崩溃 / 静默数据损坏 / 竞态踩写,特定场景触发
  - **P2**:行为回归 / 资源泄漏 / 脆弱隐式约定,当前不崩但易被后续改动或长跑引爆

---

## 总览表

| 编号 | 标题 | 严重度 | 触发条件 | 主要位置 | 状态 |
|---|---|---|---|---|---|
| I1 | 双锁域无公共互斥:DelReqIDRun erase 与 Python 写路径并发 | **P0** | 稳态每步存在窗口 | pre:293-295 / scatter:1421-1429 | ✅ 已修复 |
| I2 | mIsPrefetchRunning 生命周期缺陷:背靠背任务的假空闲窗口 | **P0** | 任务积压 + 恰逢 DelBlocksMap | scatter:1217 / pre:296 / task_manger:79-90 | ✅ 已修复 |
| I3 | preempt 路径先 free 后 drain,与正常路径顺序相反 | P1 | 抢占时预取任务在飞 + 块被重分配 | gsa.py:1779-1786 | 待修复 |
| I4 | cal_indexer_topk_blocks 覆写与 indexer 线程共享的 numpy 视图 | P1 | IS_INDEXER_OFFLOAD + 加载跨步 | gsa.py:2133-2135 / 2208-2210 | 待修复 |
| I5 | "仅日志不拦截"链:脏 token → 越界 slot → 设备索引越界 | P1 | mRealTimeIdx 过期 / topk 脏数据 | pre:900-915 + hbm_manger:101-108 | 待修复 |
| I6 | TokenIndexMapInter 无锁读 mRealTimeIdx/mSeqLen(C1 只修写侧) | P1 | 预取任务运行中 SubmitTask/DelReqIDRun 并发 | pre:892-917 vs scatter:1226-1228 | 待修复 |
| I7 | mAllTopkIndicesTensor 跨锁域:worker memcpy vs SetTopkIndices 替换 | P1 | 新请求接入时预取在飞 | scatter:342 vs 1290-1291 | 待修复 |
| I8 | mAllBlocksHash/mLastKVBlockAddress 跨锁域:find/at vs 重赋值 | P1 | SetBlocksMap 与预取并发 | pre:1383/1666 vs 649-656 | 待修复 |
| I9 | SetKvTensor/SetKvBuffer 无锁替换张量 + aclTensor 覆盖泄漏 | P1 | KV cache 重分配时任务在飞 | scatter:1114-1137 | 待修复 |
| I10 | 预取结果 shm 偏移无 i < mMaxBs 校验 → OOB 写共享内存 | P1 | batch 超配置 mMaxBs | scatter:352 / pre:531 | 待修复 |
| I11 | 增块路径写 mPrefetchBlockTableIdx/mIndexerLoadResultT 无容量校验 | P1 | 活跃块数增长超初始容量 | scatter:667 / 671-672 | 待修复 |
| I12 | Get* 访问器无锁 + operator[] 插入 + 空指针解引用 | P1 | 任务在飞或请求已删时 Python 查询 | scatter:1498-1523 | 待修复 |
| I13 | get_model_input 静默 del 旧 stat:池 slot 永久泄漏 + C++ map 残留 | P2 | 同 req_id 重调度残留旧条目 | gsa.py:459-461 | 待修复 |
| I14 | set_modelrunning_status(True) 无 try/finally,异常路径闸门永久卡死 | P2 | begin→finished 之间任何异常 | gsa.py:1834 / 2334 | 待修复 |
| I15 | MTP + rec_result=None 时陈旧 topk 静默提交 | P2 | spec>1 且调用方未传 rec_result | gsa.py:2389-2412 | 待修复 |
| I16 | _update_dsa_block_table 稀疏/稠密索引口径不一致 | P2 | IS_INDEXER_OFFLOAD + decode 增块 | gsa.py:1178 vs 1182 | 待修复 |
| I17 | 缓冲尺寸与守卫上界脱节;indexer_last_q_index 无越界守卫 | P2 | token 数贴边界 / 大 chunk indexer req | gsa.py:786/797/826/1624 | 待修复 |
| I18 | slot 不足仅打日志后 fail-continue,半初始化状态继续推理 | P2 | slab 分配与 seq_len 失配 | gsa.py:1269-1283 | 待修复 |
| I19 | UCMLoadSSD 丢弃 handle,TaskWrapper::tasks_ 每次调用永久泄漏 | P2 | 每次调用即泄漏(长跑累积) | kvcache_prefetch.h:94-96 | 待修复 |
| I20 | ObtainStream malloc 后 throw 泄漏 + worker 线程外异常 terminate | P2 | 启动期 ACL 初始化失败 | trans_ascend.cpp:57-75 | 待修复 |
| I21 | GetAclDataType 无 float8_e4m3fn 映射(c8 模式) | P2(待验证) | A5 + enable_sparse_sfa_c8 | trans_ascend.cpp:138-152 | 待验证 |
| I22 | 杂项:空输入 UB / 累积泄漏 / vector\<bool\> data race / 32 位截断 / 输入无界 | P2 | 各自条件 | 见分项 | 待修复 |

---

# 一、P0:结构性并发缺陷

> **✅ I1 + I2 联动修复记录(P0 批次)**

- **修复内容**(锁序统一为 **mMutexUseToken → mUpdateReqMutex**,与 worker 临界区内 `CallPrefetchProcess→IsReqDeleted` 的既有嵌套方向一致,经审计全仓无反向嵌套):
  - **I1**:`DelBlocksMap`(scatter)的立即删除分支改为持**双锁**(token→req)执行 `DelReqIDRun`;worker 收尾路径同样收编——新增 `GSAPrefetchEngineC::finishPrefetchTask()`:双锁内执行 `DelReqIDRun`,随后复位 `mIsPrefetchRunning`、递减 `mPrefetchTaskCount`。`MultiBSThreadFun` 的收尾三行散落代码改为统一调用 `finishPrefetchTask()`(含异常路径:`CallPrefetchProcess` 抛异常时先 `finishPrefetchTask()` 再 rethrow,保持 TaskManager 的 FAILED 语义,且标志/计数必然复位,Python drain 不挂死);
  - **I2**:新增 `std::atomic<int> mPrefetchTaskCount`(public 成员):`SubmitTask` 在**入队前** `fetch_add(1)`(紧贴入队,缩小异常泄漏窗口);`MultiBSThreadFun` 入口置 `mIsPrefetchRunning=true`(替代原 SubmitTask 入队时置位),经 `finishPrefetchTask` 递减计数;`GetPrefetchStatus()` 语义改为"执行中 **或** 排队计数>0";`DelBlocksMap` 空闲判定改为"排队计数==0 且 !执行中 且 !indexer 在跑"三条件;
  - 死锁论证:`mMutexUseToken` 全仓仅两处获取(worker `CallPrefetchProcess`、Python `SetTokensMap`),二者临界区内仅嵌套获取 `mUpdateReqMutex`(worker 经 `IsReqDeleted`),方向一致 token→req;`DelBlocksMap`/`finishPrefetchTask` 按 token→req 双锁,无反向路径 → 无锁环;
  - 同族缺口(I7/I8/I12 等 worker 读 vs Python 单锁写的部分)**未**随本修复自动消解(worker 写/读路径本身仍持单锁),按批次计划在第三批处理;
  - **测试**:新增 4 个 UT(排队计数语义 `PrefetchStatusReflectsQueuedTaskCount`、排队中延迟删除 `DelBlocksMapDefersWhileTaskQueued`、收尾函数行为 `FinishPrefetchTaskRunsDeferredDeletionAndResetsCount`、原有原子标志测试保持);既有 `DelBlocksMapQueuesDeletionWhenPrefetchRunning`/`DelBlocksMapRunsImmediatelyWhenIdle`/mst 并发双删测试语义兼容。
- **验证**:C++ 无法本地编译,新旧 UT 需随构建跑;TSAN 下高频 SubmitTask+DelBlocksMap 打窗口(原为堆损坏)。

## I1【P0】双锁域无公共互斥:DelReqIDRun 的 map erase 与 Python 写路径并发

**机制**:引擎有两把互不相交的锁——`mMutexUseToken`(Python 主线程的 SetTokensMap 等入口 / 预取 worker 的 CallPrefetchProcess)与 `mUpdateReqMutex`(indexer 路径、SetBlocksMap/SetIndexerMap、DelBlocksMap、DelReqIDRun)。**DelReqIDRun 要 erase 的 17 个 per-request map,与 Python 侧持另一把锁写入的 map 大量重叠**:

```
worker:  CallPrefetchProcess(持 mMutexUseToken) → 释放
         → DelReqIDRun(持 mUpdateReqMutex)→ erase mHBMMangers/mMissGlobalToken/mLoadSlot/...
Python: SetTokensMap(持 mMutexUseToken)→ 写 mHBMMangers[reqID].resize()/mAllReqIdSlots[reqID].assign()/...
```

`CallPrefetchProcess` 释放 `mMutexUseToken` 的**瞬间**,阻塞在该锁上的 Python `SetTokensMap` 立即获锁写入同一批 map——与 worker 紧接着的 `DelReqIDRun()` erase 并发。std::map / std::unordered_map 的并发 erase 与写 = **堆损坏/崩溃**。该窗口在稳态下每一步都存在,只是 erase 通常发生在请求确实被删时。

**同族缺口**(同一根因的不同面,严重度 P1,详见 I6~I9、I12):
- `mAllTopkIndicesTensor`:worker(omp 区内,scatter:342)memcpy vs `SetTopkIndices`(持 mUpdateReqMutex,scatter:1290)整体赋值替换——注释声称互斥,实际预取线程并不取该锁;
- `mAllBlocksHash` / `mLastKVBlockAddress`:worker `find/at`(pre:1383/:1666,unordered_map 插入引发 rehash)vs `SetBlocksMap`(pre:649-656)整体重赋值;
- `Get*` 访问器(GetTokenBlockMap 等,scatter:1498-1523)**完全无锁**;
- `SetKvTensor/SetKvBuffer`(scatter:1114-1137)无锁替换张量与指针表。

**修复方向**:① `DelReqIDRun` 统一改为持双锁(`mMutexUseToken` + `mUpdateReqMutex`,注意全局锁序声明避免死锁);或 ② 删除收敛到 worker 侧(任务收尾在仍持 `mMutexUseToken` 的临界区内执行,或引入 generation 计数延迟删除);或 ③ 最彻底:per-request map 改 `shared_ptr` 快照 + 请求级读写锁。Python 侧入口(SetTokensMap/SetBlocksMap/SetTopkIndices/SetKvTensor)必须与 worker 侧触碰的容器建立**同一把锁**。

## I2【P0】mIsPrefetchRunning 生命周期缺陷:背靠背排队任务的"假空闲"窗口

**机制**:标志置 true **仅**发生在 `SubmitTask` 入队时刻(scatter:1217);TaskManager worker 取任务**不置位**(kvcache_task_manger.cpp:79-90 取任务不碰标志);`MultiBSThreadFun` 每任务结束置 false(pre:296)。

```
SubmitTask(步 N)   → 入队任务 A,置 true
worker             → 执行 A → 置 false
SubmitTask(步 N+1) → 入队任务 B(此时 A 可能仍在跑,标志已是 false 或即将 false)
worker             → 取走 B 立即执行(B 全程标志为 false,无人再置 true)
```

任务 B 执行期间(持 `mMutexUseToken`,对 `mHBMMangers/mMissGlobalToken/mLoadSlot/...` 做 operator[]/遍历),Python 侧 `DelBlocksMap`(scatter:1427)检查 `!GetPrefetchStatus() && !GetIndexerStatus()` → 两个都 false → 在 **Python 线程**直接 `DelReqIDRun()` erase 同一批 map → **与 B 并发 erase/read = 堆损坏**。

**放大因素**:B1 互斥闸(`mStopPrefetch` 自旋,scatter:202)拉长任务耗时、加剧队列积压,窗口随之放大。`mIsIndexerLoadRunning` 同模式(scatter:1273 / pre:304),但其 DelReqIDRun 与 IndexerLoadProcess 都持 `mUpdateReqMutex` 可串行化,独立危害小。

**修复方向**:标志改由 TaskManager 在**任务真正被 worker 取走时**置位(或改为队列深度计数器 `in-flight + queued`);`DelBlocksMap` 的立即删除分支改为仅在"队列空且 worker 空闲"时执行,否则只置墓碑延迟到 worker 收尾删除(与 I1 的修复联动)。

---

# 二、P1:条件性崩溃 / 静默数据损坏

## I3【P1】preempt 释放路径先 free 后 drain,与正常结束路径顺序相反

**位置**:gsa.py:1779-1786(`build_gsa_metadata` 循环内)。

`_release_preempted_gsa_stats`(1802-1812)**先**执行池 free + `update_slot_num()`(向 scheduler 进程发布新的空闲 slot 数),**然后**才 `while get_prefetch_status()` drain、最后 `del_finish_meta`(置 B2 墓碑)。而正常结束路径 `request_finished_in_worker`(2509-2521)是**先 drain 再 free**。

**后果**:drain 窗口内——① C++ 预取线程仍持有该请求 map 的裸指针(k/v refs 指向已 free 的 dump slab 区)且可继续写;② vLLM scheduler 已在同一调度步内回收并可能**重分配**被抢占请求的 device 块(prefix-cache 复用不重写内容),in-flight DMA 仍会 scatter 进这些块 → **跨请求 KV 污染**。墓碑只阻止"新任务组装",不回收已发射的 DMA;`SetModelRunningStatus` 仅置原子位,worker 只在循环间隙检查,不等待在飞 DMA。

**触发条件**:请求被抢占时上一步提交的预取任务仍在执行(任务时延 > 步间窗口)+ 被释放块被重分配。

**修复方向**:对齐正常路径顺序——先 drain(含 indexer 状态)→ 再 free/发布 → 再墓碑。

## I4【P1】cal_indexer_topk_blocks 覆写与 C++ indexer 线程共享的 numpy 视图

**位置**:gsa.py:2133-2135(覆写)vs 2208-2210(提交共享视图)。

`set_indexer_topk_blocks_decode` 提交的 `topk_lists` 是 `req_meta.indexer_topk_blocks_cpu[:, :prompt_sparse_len]` 的 numpy 视图(pybind 传 buffer 与 numpy **共享内存**),C++ 侧 `inputInfo.indexerTopkList = topkLists`(scatter:1270)交给 indexer 工作线程**异步**读取(587-588/629 行 `data_ptr<int>()`)。下一步的 `cal_indexer_topk_blocks` 在 `execute_finished` 中(2334 行已 `set_modelrunning_status(False)` **恢复**线程后)直接覆写同一 numpy 缓冲——覆写前**没有任何 `get_indexer_status()` 等待**;唯一的 `is_indexer_load` 检查(2199 行)只 gate 新任务提交,且发生在覆写之后。

**后果**:撕读 → indexer 加载错误块(静默错预取,无报错)。

**触发条件**:`IS_INDEXER_OFFLOAD=True`(当前 utils.py:90 默认 False,现网默认不触发)+ indexer 加载任务时延跨步 + 该请求下一步仍是 `need_indexer_sparse_reqs` 成员。

**修复方向**:覆写前等待 `get_indexer_status()` 为 false(或提交时 deep-copy,或双缓冲轮换)。

## I5【P1】"仅日志不拦截"链:脏 token → 越界 slot → 设备索引越界

**链条**(三道防线全部只打 ERROR 日志、不拦截不清洗):

1. **pre:900-906 `TokenIndexMapInter` 越界读**:`revIdx >= mRealTimeIdx[reqID][layerID].size()` 仅打日志,`:906` **照常执行** `mRealTimeIdx[reqID][layerID][revIdx]`(std::vector operator[] 无检查)→ OOB 读 UB,拿到垃圾 idx 污染 oneTopkList;
2. **pre:908-915 脏 token 不清洗**:转换后 `oneTopkList[j] < 0 || >= mSeqLen[reqID]` 仅打日志,脏 token 继续进 topk 列表(是 1 的直接上游与下游放大器);
3. **hbm_manger.h:101-108 越界 pinned slot 仍写入**:`transformedSlot >= mMaxTotalSlots` 仅打日志,`:107` 仍 `topkSlotIds[i] = transformedSlot` → 经 `buildTopkOutput`(只挡负值,:202)→ `WritePrefetchResult` → shm → **全 rank** 的 `mAllTopkIndicesTensor`(scatter:342 memcpy)→ 设备 gather/scatter 索引越界。

**修复方向**:三处改为 fail-loud(抛异常/abort)或 fail-safe(跳过该 token 并计数);最低成本:①② 在越界时 `continue` 跳过该 token,③ 越界时写 -1(下游已有负值过滤)。

## I6【P1】TokenIndexMapInter 无锁读 mRealTimeIdx/mSeqLen(C1 修复的读侧缺口)

C1 修复只覆盖了写侧(SubmitTask 持 mUpdateReqMutex 拷贝,scatter:1226-1228)。读侧在预取 worker(仅持 mMutexUseToken)经 `ProcessPrefetchRequest`→`TokenIndexMapInter`(omp 并行区内)读 `mRealTimeIdx[reqID][layerID]` 与 `mSeqLen[reqID]`,与写方(SubmitTask)、删方(DelReqIDRun,持 mUpdateReqMutex)均无公共锁 → 并发 std::map 读写 UB。修复方向:读侧同样持 mUpdateReqMutex(临界区仅为 vector 读取,开销可接受),或随 I1 统一锁域。

## I7【P1】mAllTopkIndicesTensor 跨锁域

worker(omp 并行区内,scatter:342)`memcpy(mAllTopkIndicesTensor[oneReqID][layerID].data_ptr(), ...)` vs `SetTopkIndices`(Python 线程,持 mUpdateReqMutex,scatter:1290-1291)`mAllTopkIndicesTensor[reqID] = topIndices` 整体赋值。注释声称锁防"主预取线程的写入",**实际预取线程并不取 mUpdateReqMutex**。新请求接入发生在预取任务运行期:map 插入与 worker 的 operator[](omp 区内多线程)并发;tensor 赋值替换使旧 Tensor 引用计数归零释放,与 worker 正在 memcpy 的旧 data_ptr 并发 → UAF。

## I8【P1】mAllBlocksHash / mLastKVBlockAddress 跨锁域

worker 链路(持 mMutexUseToken)`mLastKVBlockAddress.find(reqID)`(pre:1383)、`mAllBlocksHash.at(reqID)`(pre:1666)vs Python `SetBlocksMap/SetLastIndexerBlock`(持 mUpdateReqMutex,pre:649-656/671-683)`mAllBlocksHash[reqID] = blocksHash`(vector 整体重赋值)、`mLastKVBlockAddress[reqID]` 插入(**unordered_map 插入可引发 rehash**,使另一线程的迭代器/指针全部失效)。新请求在预取任务运行中接入即触发 UB。

## I9【P1】SetKvTensor/SetKvTensorIndex/SetKvBuffer 无锁替换张量与指针表

**位置**:scatter:1114-1137(SetKvTensor)、1139-1156(SetKvTensorIndex)、1167-1207(SetKvBuffer)。

Python 在任务排队/运行期重新调用(vLLM KV cache 重分配)时:
1. `mKcacheTensor[layerID] = kCachesTensor` 使旧存储引用归零**立即释放**,而 worker(持 mMutexUseToken)在 `KVCacheInplaceScatter` 仍持有旧 aclTensor(scatter:63 `dstTensor = mKCacheAcl[layerID]`)、`mH2DPtrInfo`(scatter:475-481/515-533)仍持有旧 data_ptr → **设备侧 UAF**;
2. 旧 `mKCacheAcl[layerID]` 被直接覆盖**不 Release**(对比 PrepareScatterAclTensor:98-105 会先 Release)→ aclTensor 泄漏。

**修复方向**:替换入口持 mMutexUseToken(与 worker 串行)+ 覆盖前 `ReleaseAclTensor` 旧指针;或替换仅在确认无在飞任务时进行。

## I10【P1】预取结果 shm 偏移无 i < mMaxBs 校验 → OOB 写共享内存

shm 容量按 `mMaxBs * mTopkIndexerNum * perReqPerLayerTotalSize` 建(pre:531),而 `WritePrefetchResult` 偏移 `(i * mTopkIndexerNum + layerID) * layout.totalSize`(scatter:352)中 `i ∈ [0, reqIdList.size())` **全链路无上界校验**(SubmitTask/PreparePrefetchResults 均不检查 `oneRunBsLen <= mMaxBs`)。单步 batch 实际请求数超过配置的 mMaxBs(负载波动/配置漂移)→ memcpy 越出 mmap 区,踩踏相邻映射。**修复方向**:SubmitTask 入口断言 `reqIDList.size() <= mMaxBs`(fail-loud)。

## I11【P1】增块路径写 mPrefetchBlockTableIdx / mIndexerLoadResultT 无容量校验

**位置**:scatter:667(FinishOneLayerPrefetch)、671-672、1262-1264(SubmitTaskIndexer)。

`mPrefetchBlockTableIdx[reqID][layerID]` 容量固定为 SetIndexerMap 初始化时的 `oneDsaSparseBlock.size()`(pre:833),`mIndexerSparseLen` 同样固定;经 `update_indexer_map("add")` / AddIndexerMap 增长后**不扩容**:
- `mPrefetchBlockTableIdx[reqID][layerID][pairIdx] = ...`(scatter:667)越界写 vector<int>(堆损坏);
- `memcpy(mIndexerLoadResultT[reqID][layerID].data_ptr(), ..., successIndex*sizeof(int))`(scatter:671)越界写设备张量。

对比初始化路径 pre:841-849 有同类容量检查,增块路径缺失。**修复方向**:写前校验容量(对齐初始化路径),超限走降级日志。

## I12【P1】Get* 访问器无锁 + operator[] 插入 + 空指针解引用

**位置**:scatter:1498-1523(GetTokenBlockMap/GetBlockTokenMap/GetIndexerBlockMap/GetIndexerTokensMap 等)。

```cpp
std::vector<int> GetTokenBlockMap(int layerID, std::string reqID)
{ return mHBMMangers[reqID][layerID]->mGlobalToSlot; }   // 无锁 + operator[] 缺键插入空 + -> 解引用空 unique_ptr
```

任一 worker 任务在飞时 Python 调用:撕裂拷贝(与预取线程并发改同一 vector);请求已删后再调用:`operator[]` 插入空容器后 `[layerID]` 直接越界/解引用空 unique_ptr。**修复方向**:`find` 替代 `operator[]` + 持相应锁(或与 I1 统一)。

---

# 三、P2:资源泄漏 / 脆弱约定 / 降级

## I13【P2】get_model_input 静默 del 旧 stat:池 slot 永久泄漏 + C++ map 残留

**位置**:gsa.py:459-461。同一 req_id 在 `scheduled_new_reqs` 出现而 `gsa_stats` 仍留有旧条目时,`del self.gsa_stats[req_id]` 丢弃旧 GSAReqStat:① `slab_host_slot/indexer_slab_host_slot/req_dump_blocks` 未归还三个池 → 永久泄漏,长期运行后 `GSASharedCPUPool.allocate` 的 assert 崩;② C++ `mAllTopkIndicesTensor[reqID]` 仍按值持有 tensor、`mBlocksMap` 等映射残留;③ `_block_hash_injected_reqs` 残留旧 req_id。被注释掉的 `# assert req_id in self.gsa_stats`(452 行)说明作者实测碰到过该分支。**修复方向**:del 前走完整释放路径(等 drain → free 池 → del_finish_meta → discard hash 集合)。

## I14【P2】set_modelrunning_status(True) 无 try/finally,异常路径闸门永久卡死

**位置**:gsa.py:1834(置 True)/ 2334(置 False)。begin→finished 之间任何异常上抛(A1 防御 raise :1565、allocate assert :136、`gsa_stats[req_id]` KeyError、numpy TypeError 等)跳过 execute_finished → 预取线程被永久暂停;若上层(PD 引擎)捕获异常继续步进,下一步 `_update_prefetch_status_all`(:1382)与 preempt drain(:1781)将无界自旋挂死。B1 修复只保证正常路径配对。**修复方向**:`execute_begin` 包 try/except,异常时先恢复 `set_modelrunning_status(False)` 再 raise。

## I15【P2】MTP + rec_result=None 时陈旧 topk 静默提交

**位置**:gsa.py:2389-2412 vs 2423-2432。`select_gsa_prefetch_topk[:, index]` 的唯一写入点在 `if self.num_spec_tokens != 1 and rec_result is not None:` 分支(:2405);若 spec>1 但 rec_result 为 None,else 分支只填 req_seq_len,`_submit_prefetch_task`(:2428)仍提交整个 `select_gsa_prefetch_topk` → **上一步的陈旧 topk 被当作当前结果预取**,无任何报错。**修复方向**:else 分支显式清零/跳过提交,或提交前校验填充标志。

## I16【P2】_update_dsa_block_table 同一新增块两处索引口径不一致

**位置**:gsa.py:1178 vs 1182。同一 `if req_meta.add_blocks_len:` 块内,numpy 侧写**稀疏位置** `indexer_block_tables[:, indexer_sparse_len()-1]`,而传给 C++ `update_indexer_map` 的 idx 是**稠密块计数** `len(req_meta.slab_host_slot)-1`,两者相差 `prompt_block_len - prompt_sparse_len`(稀疏率 0.2 时相差数倍)。C++ `UpdateIndexerMap`(pre:862-864)将该值 emplace 进 `mUseBlockTableIdx`(既有内容是稀疏 idx),随后经 SubmitTask 拷贝为 `mRealTimeIdx` 被 `TokenIndexMapInter`(pre:906)用作 dense→sparse 映射 → 错位映射/越界(与 I5 的越界读叠加)。

## I17【P2】缓冲尺寸与守卫上界脱节;indexer_last_q_index 无越界守卫

**位置**:gsa.py:786/797/826/1624。
- ① `gsa_prefetch_index` 尺寸为 `max_graph_bs`(:786),而 A1 守卫上界用 `gsa_swap_topk_bs = max_graph_bs + 2`(:797/:826):当 GSA 总 token 数落在 `(max_graph_bs, max_graph_bs+2]` 且全属预取请求时,:1609 `self.gsa_prefetch_index.np[:len] = ...` numpy ValueError 崩溃——守卫与目标缓冲尺寸脱节;
- ② `indexer_last_q_index`(尺寸 max_graph_bs,:837)接收 `indexer_last_q_index_tmp` 展平(:1624),非 last-chunk 的 indexer req 在 `_collect_indexer_meta`(:1488-1493)贡献整段 prefill chunk 的 query 区间,总长可达 `max_num_scheduled_tokens`(如 8192)≫ max_graph_bs → :1624 直接 ValueError;即便不炸,`cal_indexer_topk_blocks` :2111 `indexer_last_query.npu[:, index_list]` 存在 NPU 侧越界风险。该路径无任何守卫。

## I18【P2】slot 不足仅打日志后 fail-continue,半初始化状态继续推理

**位置**:gsa.py:1269-1283。`len(slots) != ceil(seq_len/block_size)` 分支连打 4 条 error 后直接返回,`slot_mapping[index_in_batch]` 保持 `[]`、dump 列表为空 → 该步新 token 既不写 slot_mapping 也不 dump 进 indexer cache,请求继续用残缺状态推理(indexer topk 基于缺失数据)→ 静默质量退化,无异常、无标记。**修复方向**:fail-loud 抛异常(与 E3 同精神)或至少标记该请求跳过 indexer 路径。

## I19【P2】UCMLoadSSD 丢弃 handle,TaskWrapper::tasks_ 每次调用永久泄漏

**位置**:kvcache_prefetch.h:94-96。`mStoreV1_->Load(std::move(task))` 返回的 handle 被**丢弃**(无人 Wait)——`TaskWrapper::tasks_` 的条目 + Latch waiter **只在 `Wait()` 中 erase**(task_wrapper.h:93,已实测)→ 每次 `ucm_load_ssd` 调用泄漏一条任务记录(TaskPtr + WaiterPtr + map 节点,数百字节)。warmup/预取路径每次调用累积,长跑内存缓慢增长。**修复方向**:Load 后用 `Check` 轮询至完成再 `Wait` 收尾清理(可复用 H2 的 `WaitStoreBounded` 模式),或至少在返回前 `Wait` 一次(该调用本身是异步语义,轮询更合适)。

## I20【P2】ObtainStream malloc 后 throw 泄漏 + worker 线程 lambda 内异常 terminate

**位置**:trans_ascend.cpp:57-75。
- ① `malloc(context)` 后 `aclrtCreateContext` 失败即 throw → context 泄漏;`malloc(stream_)` 后 `aclrtCreateStream` 失败即 throw → 已 malloc 的 stream 数组与已创建的 stream 全部泄漏;
- ② 更严重:`ObtainStream` 在 worker 线程 lambda 中调用,**位于 `ExecuteTask` 的 try-catch 之外**(task_manger.cpp:61 vs 100-111)→ 线程内未捕获异常 → **`std::terminate` 整进程无声死亡**(启动期 ACL 环境异常时触发)。

**修复方向**:RAII 包裹(失败路径先释放已分配资源);worker lambda 全程 try-catch,失败时记日志并让该 worker 退出而非 terminate。

## I21【P2,待验证】GetAclDataType 无 float8_e4m3fn 映射

**位置**:trans_ascend.cpp:138-152。switch 无 `at::kFloat8_e4m3fn` 分支 → `default: ACL_DT_UNDEFINED`。c8 模式(A5,dtype=torch.float8_e4m3fn,prefetch_engine.py:84)下若 KV cache 张量经 `AtTensorToAclTensorShared` 转 aclTensor,将产生 dtype 未定义的 aclTensor → aclnn 算子行为未定义。**需构建后确认 c8 路径是否经过此函数**(若 c8 路径不走 scatter aclTensor 则无实害)。

## I22【P2】杂项

| 位置 | 问题 |
|---|---|
| pre:825/814/821 `SetIndexerMapForLayer` | `blockSlot.size()-1` 空 vector 下溢为 -1;`blockTableList.back()` 空容器 UB;`blockTableList[index+offset]` 无界索引(prefetchIdx+remainIdx 总长超 blockTableList 时越界读) |
| scatter:1442-1474 `DelReqIDRun` | erase 清单**不含** `mAllBlocksHash`/`mLastKVBlockAddress` → 每请求 16B×块数的 hash vector 与块地址 map 永不释放(累积泄漏,与 I13 叠加) |
| scatter:1218 vs 309 `mIsIndexerOffload` | `std::vector<bool>`(位压缩,共享字读-改-写)Python 无锁整体赋值 vs 预取线程读 → data race 可翻转邻位 offload 判定 |
| scatter:512-513 | `mLoadNum[layerID] * mKVSizeBytesToken`(int × uint32 → uint32)**32 位域截断**后再赋 size_t;累计 token 数 × 每 token 字节 > 4GiB 时 H2D 拷贝字节数错误(对比 :477 先 `static_cast<size_t>` 的正确写法,同函数两处口径不一致) |
| scatter:1366-1407 `IndexerInitLoad` | `int number = blockTable[0].size()` 后按行 0 长度索引所有层:`blockTable/slabIdx` 为空 UB、各层行长不齐越界读 |
| gsa_shared_memory.py | 隐式版本计数协议:读写双方每步必须严格 ±1 对齐,任何一次错位(异常/重试/跳步)后**每步固定 30s 超时自旋**,永不自愈(残余风险:TP lockstep 下配对成立时正常,fail-loud 可接受,列为脆弱约定) |

---

# 四、根因归纳(四个系统性模式)

1. **锁域矩阵有洞**(I1/I2/I6/I7/I8/I12,同根因):预取 worker(`mMutexUseToken`)与 Python/indexer 路径(`mUpdateReqMutex`)触碰的容器集合大量重叠,但锁不重叠;唯一兜底(运行标志)的生命周期又是错的(I2)。**建议一把统一修复**:DelReqIDRun 持双锁 / 删除收敛到 worker 侧 / 标志改由 TaskManager 置位,三项联动;
2. **"仅日志不拦截"防御模式**(I5/I18):五处打印错误后继续用坏数据走完全链,从 UB 读一路污染到设备索引;
3. **Python 侧顺序/配对脆弱**(I3/I13/I14/I15):preempt 与 finish 路径顺序不对称、begin/finished 无异常保护、静默丢弃状态、陈旧数据静默提交;
4. **跨边界缓冲共享无所有权**(I4/I9):numpy 视图/裸指针/aclTensor 跨线程共享,替换时无同步无释放。

# 五、修复顺序建议

1. **第一批(P0)**:I1 + I2 联动(锁域 + 标志生命周期,同一根因,建议一个设计文档 + 分步 commit);
2. **第二批(P1,静默数据损坏)**:I3(preempt 顺序,改动小收益大)、I5(三处 fail-safe,改动小)、I4;
3. **第三批(P1,并发 UB)**:I6/I7/I8/I12(随 I1 的锁域方案统一收编,若 I1 采用双锁方案则这几项自动消解)、I9/I10/I11;
4. **第四批(P2)**:I13/I14(改动小)、I15~I22 按场景。

# 六、验证手段

- **I1/I2(锁域)**:TSAN 全量 UT + 故障注入(高频抢占 + 任务积压压测,循环 DelBlocksMap/SubmitTask 打竞态窗口);
- **I3**:抢占-重分配场景 e2e,断言无跨请求 KV 污染(数值比对);
- **I5**:构造 mRealTimeIdx 过期/短表用例,断言 token 被跳过或抛异常而非继续;
- **I10**:batch = mMaxBs+1 的单测,断言 SubmitTask 拒绝;
- **I19**:循环调用 ucm_load_ssd 数千次,断言 store 侧 tasks_/内存不增长;
- **I21**:c8 模式构建后走一次 scatter 路径冒烟,确认 aclnn 接受或补充 fp8 映射;
- **回归**:无 MTP 常规 decode + PD-mix 全链路。

---

> **备注**:本清单为第三轮(结构性并发专项)审阅结论,与主清单《GSA_ISSUES_AND_FIX_PLAN.md》(A~G 类 + H 类)互补;H 类修复(`b85c3b1` 及此前 17 项)已生效的部分在本轮审阅中视为基线,未重复报告。I 类尚未修复,状态均为"待修复"。
