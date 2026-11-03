# GSA/UCM 缺陷分析与修复清单

- **分析对象**:`C:\Users\cgj\Desktop\CGJ\gsa_profiling\code\ucm`(即本仓库;内容等价于 cgj1994/feature_gsa_v2 分支提交 `0177103` 合并后的代码,本地为同代或略新版本)
- **分析日期**:2026-11-03
- **行号说明**:以下所有行号均为**本地仓库当前代码**实测行号,修复时可直接定位
- **严重程度定义**:
  - **P0**:确定性崩溃或确定性内存踩写,可复现触发条件明确
  - **P1**:条件性崩溃 / 静默数据损坏 / 竞态踩写,特定场景触发
  - **P2**:行为回归 / 脆弱隐式约定,当前不崩但易被后续改动引爆

---

## 总览表

| 编号 | 标题 | 严重度 | 触发条件 | 主要位置 | PD-mix 是否触发 | 状态 |
|---|---|---|---|---|---|---|
| A1 | CopyPreSwapTokens 行索引越界写 | **P0** | MTP + 总 token 数 > dim1 | kvcache_pre_scatter.cpp:1235-1238 | **是** | ✅ 已修复 |
| A2 | CopyPreSwapTokens 负偏移越界写 | **P0** | use_tokens 内容过期 | kvcache_pre_scatter.cpp:1239-1242 | **是** | ✅ 已修复 |
| B1 | 前向/预取互斥闸是死代码 → 设备侧并发踩写 | **P0** | 预取块与前向读写块重叠 | kvcache_pre_scatter.cpp:51-88 | **是** | ✅ 已修复 |
| E1 | P 节点 `init_dsa_data` AttributeError 启动崩溃 | **P0** | PD 分离 P 节点 | gsa.py:1892-1905 | 否(死代码) | ✅ 已修复 |
| F1 | `when_imported` 单次触发保护静默丢弃后续补丁钩子 | **P0** | 同一模块注册两个补丁 | patch/utils.py:96-100 | **是** | ✅ 已修复 |
| F2 | `indexer_select_post_process` 重复定义相互覆盖 | **P0** | v0.23.0_pmr 补丁生效时 | v0.23.0_pmr/.../device_op_patch.py:13,188 | **是** | ✅ 已修复 |
| A3 | 老引擎 freeBlockPtr 裸写无界 | P1 | 行宽与表大小错配 | sparse/gsa/.../kvcache_pre.cpp:432 | 是(老引擎) | ✅ 已修复 |
| B2 | 块 ID 陈旧 → 跨请求并发踩写 | P1 | 请求抢占/完成时 DMA 在途 | kvcache_pre_scatter.cpp:1305-1313 | **是** | ✅ 已修复(窗口收窄) |
| B3 | 裸指针 use-after-free 风险 | P1 | Python 张量先于 C++ 释放 | pybinds.cpp:143-148 + gsa.py:1544-1546 | **是** | ✅ 已修复 |
| C1 | C++ map 跨线程并发读写无锁 | P1 | Python 写 API 与预取线程同时活跃 | kvcache_pre_scatter.cpp:1167-1171 附近 | **是** | ✅ 已修复 |
| D1 | `-1` 槽位索引可达设备侧 scatter | P1 | aclgraph 捕获/回放 | sfa_kv_transfer_patch.py:387-392 | **是** | ✅ 已修复 |
| E2 | `_load_kv_dram` 未捕获 C++ throw | P1 | PD 分离 D 节点 + store 注入未就绪 | gsa.py:2576 | 否(死代码) | ✅ 已修复 |
| E3 | `_load_kv_dram` 超时静默 KV 空洞 + 阻塞 hang | P1 | PD 分离 D 节点 + lookup 超时 | gsa.py:2575-2587 | 否(死代码) | ✅ 已修复 |
| E4 | `_ensure_store_injected` 非异常静默失败 | P1 | connector/store 未就绪 | gsa.py:955-974 | 否(死代码) | ✅ 已修复 |
| F3 | `req_sparsed_slots=None` 无判空 | P1 | 非 GSA 调度路径构造的 SchedulerOutput | sched_output_patch.py:20-23 | **是** | ✅ 已修复 |
| F4 | `list(kv_cache)` 无判空 | P1 | warmup/profile 阶段 kv_cache=None | sfa_kv_transfer_patch.py:643,649 | **是** | ✅ 已修复 |
| E5 | `attention_finished` gate 反转(行为回归) | P2 | PD 分离 D 节点 | gsa.py:1660 | 否(死代码) | 待修复 |
| F5 | `init_kvcache_ptrs` 无判空直接索引 | P2 | kv_cache 为 None/空 | gsa.py:2771-2782 | **是** | 待修复 |
| G1 | 跨文件隐式常量约定 | P2 | 修改任一常量 | sparse/utils.py:64-65 | **是** | 待修复 |

> 注意:**A/B/C/D/F/G 类问题与是否开启 PD 分离无关,PD-mix 日常部署同样暴露在风险之下**;E 类仅在 PD 分离部署时进入执行路径(PD-mix 下 agent 不创建,是死代码)。

---

# 一、A 类:C++ 宿主内存越界写(确定性踩内存)

## A1【P0】CopyPreSwapTokens 行索引越界写 —— MTP 下必然越界

**位置**:`ucm/sandbox/sparse/gsav2/prefetch/src/kvcache_pre_scatter.cpp:1235-1238`

```cpp
for (int index = swapRange[0]; index < swapRange[1]; index++) {
    void* spaCpuPtr = reinterpret_cast<void*>(
        spaTokenBase + (size_t)(layerID * spaMaxBs + index) * sparseLen * sizeof(int32_t));
    std::memcpy(spaCpuPtr, usePtr, sparseLen * sizeof(int32_t));   // ← 无任何边界检查
}
```

**原因(口径错位)**:
- `index` 的语义是**全 batch 累计 token 位置**。Python 侧收集逻辑 `gsa.py:1537-1538`:
  ```python
  for i in range(query_local[index_in_batch], query_local[index_in_batch + 1]):
      self.swap_index.append(i)   # 每请求追加 num_spec_tokens 个全局 token 序号
  ```
- 目标缓冲 dim1 却是**请求数量级**。分配处 `gsa.py:785-793`:
  ```python
  self.gsa_swap_index = DSAInputStruct((max_graph_bs + 2), ...)          # :785-787
  self.gsa_swap_topk = TopkIndices(
      (self.topk_layer_num, (max_graph_bs + 2), 1, SPA_TOKENS_LEN), ...)  # :789-793
  ```
  C++ 拿到的 `spaMaxBs` 即 `gsa.py:1558` 传入的 `self.gsa_swap_topk_bs`。

**触发条件**:MTP/投机解码(`num_spec_tokens = 1 + k > 1`)且 `总请求数 × num_spec_tokens > max_graph_bs + 2`。每越界一行写 `SPA_TOKENS_LEN × 4 = 20KB` pinned 宿主内存,且 `#pragma omp parallel for num_threads(8)`(:1231)8 线程并发跨 layer 踩,堆损坏表现在别处、根因难查。

**为什么没被发现**:无 MTP 时 `index ≤ 请求数 ≤ dim1`,不越界——常规单步解码压测暴露不出来。

**放大因素**:该 C++ 写发生在 `_pre_swap_tokens_all`(gsa.py:1551-1560)中,**早于** `_copy_gsa_buffers_to_npu`(gsa.py:1565)里本会因超长报错的 numpy 赋值——先踩后报,numpy 检查救不了。

**修复建议**:
1. C++ 侧:`index` 加边界检查(可复用同仓库 `CheckInputIndex()`,见 `sparse/gsa/prefetch/src/kvcache_pre.cpp:188-196` 的实现),违例 abort 并带上下文;
2. Python 侧(根本修):`gsa_swap_topk`/`gsa_swap_index` 的 dim1 改为 token 口径 `max_graph_bs * num_spec_tokens + 2`,并在 `_pre_swap_tokens_all` 前断言 `len(swap_index) ≤ shape[1]`,把越界挡在 Python 层。

### ✅ 修复记录(A1)

- **修复内容**(双层防御,fail-loud):
  1. **C++ 侧** `CopyPreSwapTokens`(`prefetch/src/kvcache_pre_scatter.cpp:1226`)在 OpenMP 并行区**之前**(omp 区域内抛异常是 UB)新增两个校验:
     - `swapRange` 合法性:size==2、start ≥ 0、start ≤ end;
     - **关键校验** `swapRange[1] > spaMaxBs` → 抛 `std::out_of_range`,错误信息带区间端点、spaMaxBs、并提示 "accumulated token position vs request-count dim mismatch, MTP num_spec_tokens > 1?"。
     - 校验整个区间端点即可覆盖区间内所有 `index`(区间连续递增),无需逐 index 检查,零热路径开销。
  2. **Python 侧** `_pre_swap_tokens_all`(`gsa.py:1541`)调用 C++ 前新增拦截:`max_row_end = max(rng[1]) > self.gsa_swap_topk_bs` 时抛 `IndexError`,带 num_spec_tokens、请求数、dim1 等上下文,把堆踩写变成进入 C++ 之前即可定位的错误。
- **修复原因**:
  - 根因是 index 的"token 口径"与 buffer dim1 的"请求口径"错位;口径重排(dim1 改为 `max_graph_bs * num_spec_tokens + 2`)是根本修,但会改变 buffer 形状、影响所有 shape 消费方(NPU 侧 kernel、aclgraph 捕获),需要独立验证,本次先落地确定性防御,把静默堆损坏升级为带上下文的显式失败;根本修作为后续项;
  - 选择抛异常而非 clamp:文档原则"违例 abort 带上下文(而非默默 clamp)"——clamp 只会掩盖口径错位并继续产生错误结果。
- **影响面**:无 MTP 常规路径 `max_row_end ≤ dim1`,两个校验均零行为变化;开 MTP 且 batch 超限时从"静默踩写 20KB×8 线程"变为首次越界即报错。
- **验证**:Python 语法 `ast.parse` 通过;C++ 需随构建跑(改动为纯前置校验,不动拷贝逻辑);单测可仿 `test_kvcache_pre_scatter_indexer.cpp` 构造 swapRange 越界用例断言 throw。

---

## A2【P0】CopyPreSwapTokens 第二处 memcpy 偏移可为负 —— 向前越界写

**位置**:同一函数 `kvcache_pre_scatter.cpp:1239-1242`

```cpp
std::memcpy(reinterpret_cast<uint32_t*>(spaCpuPtr) + sparseLen - changeLen + mtpToken
                - 1 - index % mtpToken,
            needAppend.data(),
            (changeLen - mtpToken + 1 + index % mtpToken) * sizeof(int));
```

**原因**:`changeLen` 由调用方 `PreSwapTokens` 计算,`kvcache_pre_scatter.cpp:1271-1274`:

```cpp
int maxID = *std::max_element(data, data + sparseLen * mTopkIndexerNum);  // ← :1271
int start = maxID + 1;
int end = seqLens[bs] + mtpToken - 1;
int change_len = end - start;
```

`maxID` 取自 `use_tokens` 缓冲——该缓冲**跨步保留**且依赖 `torch.zeros` 初始化。当内容过期(请求被抢占后恢复、`use_tokens`/`prefetch_use_token` 指针互换后未刷新,gsa.py:1520-1521)时 maxID 偏小 → `change_len` 可达序列长度量级(> SPA_TOKENS_LEN=5120)→ 偏移项 `sparseLen - changeLen + ...` **变负,写到本行起始地址之前**,覆盖前一个请求/前一层的行。

**现有 guard 只挡下界**(`:1276-1281`):

```cpp
if (change_len <= mtpToken - 1) {     // ← 只处理"过小"
    mLogger.log(LogLevel::ERROR, "[ERROR] PreSwapTokens change len < 0 !!!!!! ...");
    change_len = mtpToken;
}
```

上界完全不设防;同目录现成的 `CheckInputIndex()` 没有被使用。

**修复建议**:`change_len` 加上界校验(≤ sparseLen),并校验最终偏移 ≥ 0、偏移+长度 ≤ sparseLen,违例 abort 带上下文(而非默默 clamp)。

### ✅ 修复记录(A2)

- **修复内容**(`kvcache_pre_scatter.cpp` `PreSwapTokens`,全部在 **omp 并行区外**——循环内抛异常是 UB):
  1. 入口新增 `mtpToken >= 1` 断言(防御 `CopyPreSwapTokens` 循环内 `index % mtpToken` 除零);
  2. 在既有下界 clamp(`change_len <= mtpToken - 1` 时置为 mtpToken,**保留原行为**)之后,新增上界校验:`change_len > sparseLen` 时抛 `std::out_of_range`,错误信息带 maxID/start/end 与提示 "stale use_tokens making maxID too small?";
  3. `CopyPreSwapTokens` 第二处 memcpy 旁新增不变量注释:`appendOffset + appendLen == sparseLen` 恒等,非负性由调用方校验保证。
- **修复原因(为何在调用方一次校验即可)**:
  - 偏移/长度都是 `changeLen, mtpToken, index%mtpToken, sparseLen` 的函数,数学上 `change_len ∈ [mtpToken, sparseLen]` 时:
    `appendOffset = sparseLen - change_len + mtpToken - 1 - index%mtpToken ≥ sparseLen - change_len ≥ 0`;
    `appendLen = change_len - mtpToken + 1 + index%mtpToken ≥ 1`;
    `appendOffset + appendLen == sparseLen`(尾部恰好贴齐,不越界);
  - 因此在 omp 区外一次校验 `change_len` 区间即可**证明**循环内全部写操作安全,无需在 8 线程循环体内逐次校验;
  - 上界违例(use_tokens 过期 → maxID 偏小 → change_len 达序列长度量级)时:旧行为是偏移为负、每行向前覆盖写;新行为是进入拷贝前即抛出带上下文的异常。
- **影响面**:`change_len` 正常时(≤ sparseLen)零行为变化;下界 clamp 路径保留原样。
- **验证**:需随 C++ 构建跑;单测可构造 maxID 偏小的 use_tokens 断言 throw(仿 `test_abort_paths.cpp` 风格);线上用例为抢占-恢复场景压测。

---

## A3【P1】老引擎 freeBlockPtr 裸写零检查

**位置**:`ucm/sparse/gsa/prefetch/src/kvcache_pre.cpp:432`(注意是老路径 `sparse/gsa`,非 `sandbox/gsav2`)

```cpp
} else {
    freeBlockPtr[oneFreeBlockIndex] = it->second;   // ← int* 裸写,零边界检查
    oneFreeBlockIndex += 1;
}
```

**原因**:写入长度上限完全依赖 Python 侧 freeBlock 张量行宽与 `mDocsTables[reqID][i]` 表大小的隐式一致,任何尺寸错配即静默越界。

**修复建议**:传入 freeBlock 行容量并在循环内断言;或改写为带长度参数的安全接口。

### ✅ 修复记录(A3)

- **修复内容**(`sparse/gsa/prefetch/src/kvcache_pre.cpp` `RunOneBsPrefetch`,:404 omp 循环内):
  1. 循环体开头取真实行容量:`const int64_t freeBlockCapacity = mFreeBlock[i][bsIndex].numel()`(不依赖任何外部常量,Python 侧行宽怎么构造都对);
  2. 写入 `freeBlockPtr[oneFreeBlockIndex]` 前校验 `oneFreeBlockIndex >= freeBlockCapacity` → `mLogger.log(ERROR, ...)`(带 layer/bsIndex/docs 表大小/容量上下文)+ `std::abort()`。
- **修复原因**:
  - 写入循环的长度上限完全由 `mDocsTables[reqID][i]` 遍历决定(排除 hit 块后的剩余块数),对 `mFreeBlock` 行宽零校验——行宽与表大小错配时每多写一个元素踩 4 字节宿主内存,且在 **16 线程 omp 并行区内**(同时越界写多层),堆损坏极难归因;
  - 读取方 `RunPrefetchH2D`(:374-382)以 `mFreeBlockLen` 为上界、语义自洽,错配只会发生在写入侧;
  - **风格选择**:omp 区内抛异常是 UB,老引擎既定风格是 `mLogger.log(ERROR) + std::abort()`(同文件 `CheckInputIndex` :188-196 一致);abort 带上下文优于 clamp——clamp 会掩盖错配并让后续层继续用错误数据。
- **影响面**:正常运行(docs 表未 hit 块数 ≤ 行宽)校验恒不触发,零行为变化;仅错配场景从"静默堆踩写"变为"带上下文立即中止"。
- **验证**:需随 C++ 构建跑(老引擎 `sparse/gsa` 独立编译);单测可构造 docs 表大于行宽的用例断言 abort(仿 `test_abort_paths` 风格)。

---

# 二、B 类:设备侧 KV cache 并发踩写

## B1【P0】前向/预取互斥闸是死代码 —— 设备侧 KV 并发写长期裸奔

**位置**:
- 闸门实现:`kvcache_pre_scatter.cpp:1361`(`SetModelRunningStatus` 设 `mStopPrefetch`);老引擎 `sparse/gsa/prefetch/src/kvcache_pre.cpp:581`
- 无保护的直写:`kvcache_pre_scatter.cpp:51-88`(`KVCacheInplaceScatter`)

**原因**:全仓库 grep 确认 `set_modelrunning_status` 在生产 Python 代码中**没有任何调用**(仅单测调用,见 `gsav2/prefetch/tests/ut/test_gsa_prefetch_engine.cpp:87-92`)。因此预取线程在**自己的 ACL 流**上执行:

```cpp
hbmBufferND = mKBufferDevice[streamIdx].slice(0, mLoadNum[layerID]);
dstTensor = mKCacheAcl[layerID];   // ← SetKvTensor 注册的 vLLM KV cache 裸 device 指针
```

把 host slab 数据 scatter 进 vLLM 的 KV cache,**与计算流上的 attention kernel 没有任何跨流事件依赖**(无 record/wait event)。安全性完全押在"预取的块 ≠ 当前前向正在读写的块"这一逻辑约定上,而该约定没有任何机制保证。

**修复建议**:在每步 forward 前后真正调用 `set_modelrunning_status(True/False)`(生产代码激活死代码闸门);并给 DMA 目标块加 epoch/generation 校验(见 B2)。

### ✅ 修复记录(B1)

- **修复内容**(三层):
  1. **C++ 消费点**(此前 `mStopPrefetch` 在 gsav2 中只有写入、**零读取**,连自旋等待都不存在):`CallPrefetchProcess` 与 `IndexerLoadProcess` 两个任务线程入口新增自旋等待 `while (mStopPrefetch) sleep 2us`(移植老引擎 `kvcache_pre.cpp:452` 的实现,在拿锁前执行);
  2. **C++ 原子性**:`mStopPrefetch` 从 `bool` 改为 `std::atomic<bool>`(跨线程标志,原裸 bool 本身就是数据竞争;老引擎的同类问题不在本次范围);
  3. **Python 配对调用**:`execute_begin`(步首,gate 之后)置 True 暂停预取;`execute_finished`(步末,由 `maybe_execute_ucm_sparse_finished` 在 forward 完成后调用)置 False 恢复。两处都在 `is_decode_node` gate 之后,保证仅 D 节点、且预取引擎一定存在。
- **修复原因**:
  - gsav2 预取线程在**自己的 ACL 流**上执行 `KVCacheInplaceScatter`(kvcache_pre_scatter.cpp:51-88)直写 vLLM KV cache 裸指针,与计算流上的 attention kernel **无任何跨流事件依赖**;互斥闸是唯一防护,却从未被生产代码调用(全仓 grep 仅单测调用);
  - 选用 `execute_finished` 而非逐层调用的 `attention_finished` 解除暂停——后者每层触发一次,放在函数头会在**第一层**结束就放行预取、后续层仍在跑,配对错误;`execute_finished` 是真正的"步末"钩子。
- **配对时序**:`execute_begin(pause) → forward → execute_finished(unpause + 提交预取任务) → 步间窗口(预取线程自由运行) → 下一 execute_begin(pause)`——预取从"与前向并发"变为"步间串行",消除设备侧并发写;代价是损失预取与前向的重叠(性能取舍,如需恢复重叠应实现 B2 的块所有权校验后再放开)。
- **崩溃恢复性**:若某步在 pause 后异常退出,下一次 `execute_finished` 仍会恢复标志(配对最终一致),预取最多暂停到该异常步,不会永久卡死。
- **验证**:Python 语法通过;C++ 需随构建跑;运行验证看日志中预取线程的步间执行窗口与并发压测下 KV 数值比对。

---

## B2【P1】块 ID 陈旧 → 跨请求并发踩写(逻辑别名)

**位置**:`kvcache_pre_scatter.cpp:1305-1313`(`DelBlocksMap`);对照老引擎 `sparse/gsa/prefetch/src/kvcache_pre.cpp:446`(`mDelSeqIds` 检查,但也是 check-then-act,有竞态窗口)

**原因**:预取目标 block id 来自**上一步**的 topk/自由块表。请求在 DMA 在途时被抢占或完成后,vLLM 分配器会把这些 device 块**重新分配给其他请求**(其 prefill 正在计算流上写这些块)→ DMA 与 kernel 并发写同一 KV 块 → 静默 KV 数据损坏。`DelBlocksMap` 只是延迟删 map,**撤不回已提交的 DMA**。

**修复建议**:预取任务携带块所有权 epoch,提交前/DMA 前校验所有权未变化;或预取目标块从专用保留池分配(不与 vLLM 分配器混用)。

### ✅ 修复记录(B2,窗口显著收窄;彻底关闭需保留池,列为后续项)

- **修复内容**(墓碑校验,epoch 的轻量等价物——`mDelSeqIds` 本身就是"请求所有权已变"的墓碑,缺的只是执行路径对它的检查):
  1. 新增 `bool GSAPrefetchEngineC::IsReqDeleted(const std::string& reqID)`(`kvcache_pre.h` 声明 + scatter 实现):持 `mUpdateReqMutex` 查 `mDelSeqIds`,临界区仅为一次 set 查找;
  2. **三个请求粒度拦截点**(覆盖全部设备侧 DMA 组装路径):
     - `ProcessPrefetchRequest` 开头:请求已删 → 跳过本层准备(rank0 计算路径,含 omp 区内——omp+mutex 合法,临界区极短);
     - `MergeAndCopyPrefetchLayer` 逐请求循环(**DMA 聚合的最后请求粒度拦截点**):已删请求的块表不进入 H2D 缓冲、不参与 `KVCacheInplaceScatter`;非 rank0 的 scatter 路径同样经此拦截;
     - `IndexerLoadProcess` 逐请求循环(该函数已持 `mUpdateReqMutex`,直接内联查 `mDelSeqIds`):indexer 加载路径同样跳过。
- **修复原因**:
  - `DelBlocksMap` 只置 `mDelSeqIds` + 延迟删 map(任务运行中不删,避免 mid-update erase)——但任务执行路径**从不检查墓碑**:`ProcessPrefetchRequest`/`UCMTensorMerge` 聚合照常消费已删请求的 `mDocsTables` 块表并提交 scatter;而 vLLM 在请求完成/抢占时**立刻**把 device 块归还分配器并可能重新分配给新请求 → DMA 与新所有者的 prefill kernel 并发写同一块(静默 KV 损坏);
  - 检查后窗口从"整个任务生命周期(数十层全程)"收窄到"单层内 check 与 scatter 提交之间"(毫秒级);配合 B1 的步间串行化(检查/提交都发生在步间窗口,forward 未运行),冲突方仅剩 vLLM 调度器线程的块重分配。
- **锁序分析**(新增 `mMutexUseToken → mUpdateReqMutex` 嵌套):`CallPrefetchProcess`(持 mMutexUseToken)内经 `IsReqDeleted` 取 mUpdateReqMutex;反向路径逐一核对——`SetTokensMap`(唯一持 mMutexUseToken 的 Python 接口)内部调用的 `ProcessTokensMapTopk`/`LastChunkMove` 均无锁;其余 mUpdateReqMutex 持有者不调用 mMutexUseToken 路径 → 单向顺序,无死锁。
- **遗留(后续项,彻底解)**:已提交给 ACL 流的单次 DMA 仍无法撤回——单层内 check 之后、scatter 完成之前删除的请求,该层 DMA 照旧执行。彻底关闭需要:(a) 预取目标块从专用保留池分配(不与 vLLM 分配器混用,数据流改动大),或 (b) 逐块 epoch(块粒度世代号,Python 侧每次块表更新时递增,scatter 前逐块校验)。两者都需要跨 Python/C++ 的接口扩展,超出本轮"最小安全修复"范围。
- **验证**:需随 C++ 构建跑;用例为"请求完成瞬间有在途预取任务"的压测(长 prompt + 高频完成),观察 DEBUG 日志出现 "skip deleted req_id" 且不再有完成请求块上的预取写。

---

## B3【P1】裸指针生命周期 —— use-after-free(代码注释自己承认)

**位置**:
- `ucm/sandbox/sparse/gsav2/prefetch/src/pybinds.cpp:143-148`(注释原文承认 `get_self_ptr`/`set_store_v1`/`set_kv_buffer`/`pre_swap_tokens` 传裸指针,Python 对象先于 C++ 使用被释放即 UAF)
- Python 侧缓存一次、长期使用:`gsa.py:1544-1546`
  ```python
  if self.gsa_swap_topk_cpu_ptr is None:
      self.gsa_swap_topk_cpu_ptr = self.gsa_swap_topk.cpu.data_ptr()   # ← 缓存裸指针
      self.gsa_swap_topk_npu_ptr = self.gsa_swap_topk.npu.data_ptr()
  ```
- 每步传入的 `req_meta.use_tokens.ptr`(gsa.py:1529)

**修复建议**:被 C++ 持有指针的张量显式登记到 GSA 对象生命周期清单(如 `self._keepalive = [...]`),确保与引擎同生共死;C++ 侧增加指针有效性的弱校验。

### ✅ 修复记录(B3)

- **修复内容**(`gsa.py` `_pre_swap_tokens_all`,:1564 一带):一次性缓存改为**每次调用校验一致性**——
  ```python
  cur_cpu_ptr = self.gsa_swap_topk.cpu.data_ptr()
  cur_npu_ptr = self.gsa_swap_topk.npu.data_ptr()
  if (self.gsa_swap_topk_cpu_ptr != cur_cpu_ptr
          or self.gsa_swap_topk_npu_ptr != cur_npu_ptr):
      self.log.info(...  # 记录新旧地址
      self.gsa_swap_topk_cpu_ptr = cur_cpu_ptr   # 刷新缓存
      self.gsa_swap_topk_npu_ptr = cur_npu_ptr
  ```
- **修复原因与安全性论证**:
  1. **张量本体不会先于 C++ 释放**:`gsa_swap_topk.cpu`/`.npu` 张量由 `self.gsa_swap_topk`(TopkIndices)持有,后者挂在 GSA 对象上,与 C++ 引擎同生命周期——"Python 张量先于 C++ 被释放"的唯一现实路径是 `.cpu`/`.npu` **张量对象被替换**(重新分配缓冲后 `self.gsa_swap_topk.cpu = new_tensor` 之类);
  2. 旧实现首次取地址后**永不校验**,一旦缓冲被替换,缓存地址即成悬垂指针——C++ 拿它做 host memcpy / H2D 拷贝就是 use-after-free(读已回收内存 → 静默脏数据,或崩溃);
  3. 新实现每次调用比对当前 `data_ptr()`:不一致即说明缓冲被替换,刷新缓存 + log(自愈 + 检测)。校验成本为每次调用两次 `data_ptr()`(Python C 层直接取 `THPVariable` 内部指针,纳秒级),相对 pre_swap_tokens 的批量 memcpy 可忽略;
  4. **use_tokens 指针无需同款处理**:每次调用都由调用方从 `req_meta.use_tokens` 现取 `data_ptr` 现传(:1529 一带),指向的存储由 use_tokens tensor 持有且**同步使用**(单次调用内完成拷贝),无跨调用缓存,天然安全;
  5. 首次调用时两缓存为 None(`:743-744`),必然走刷新分支并记录初始地址——与原"首次 log"行为等价。
- **备注**:文档原建议的"keepalive 清单"方案适用于"C++ 长期持有指针"场景;本处指针是**每次调用同步传入、单次使用**(`pre_swap_tokens` 调用期间完成全部拷贝),真正的风险仅在"缓存跨调用不校验",故逐次校验是更精准的最小修复。pybinds.cpp 注释里 `set_kv_buffer`/`set_store_v1` 等**长期持有**型裸指针(注释自己承认的 UAF)属 C++ 侧弱校验问题,超出 Python 侧修复范围,保留为后续项。
- **验证**:`ast.parse` 语法通过;常规步行为不变(地址稳定,校验恒 False);缓冲重建场景日志出现 "gsa_swap_topk buffer replaced" 记录且新地址生效。

---

# 三、C 类:C++ 容器跨线程竞争

## C1【P1】map 并发读写无锁 → 堆损坏(UB)

**位置**:`ucm/sandbox/sparse/gsav2/prefetch/src/kvcache_pre_scatter.cpp`
- **有锁的**:`IndexerLoadProcess`(:512 `std::lock_guard lock(mUpdateReqMutex)`)、`DelBlocksMap`(:1307-1312)
- **没锁的**:`SetBlocksMap`/`SetTokensMap`/`UpdateIndexerMap`/`SubmitTask`(写 `mRealTimeIdx`/`mUseBlockTableIdx`/`mSeqLen`,约 :1167-1171 一带)

**原因**:预取线程(`CallPrefetchProcessFun`/`IndexerLoadProcess`)遍历 `mDocsTables`/`mBlocksMap`/`mAllTopkIndicesTensor` 的同时,Python 线程可并发调用上述写接口。`std::map` 一边遍历一边插入/删除是 UB,典型表现即堆损坏。另外 `RunOneBsPrefetch` 的 `#pragma omp parallel for` 16 线程循环体内写 `allNeedLoadBlock[reqID][layerID]`,依赖 `CallPrefetchProcessFun:560-561` **先**预插入外层 key 的脆弱顺序才安全。

**修复建议**:统一给所有 map 写接口加 `mUpdateReqMutex`(与预取线程遍历互斥);`RunOneBsPrefetch` 改为循环外集中预分配。

### ✅ 修复记录(C1)

- **修复内容**(全部加既有的 `mUpdateReqMutex`,与头文件 `kvcache_pre.h:58-61` 注释声明的锁职责对齐,共 6 个接口):
  1. `SetBlocksMap`(kvcache_pre.cpp:621 → :631 持锁):写 `mAllBlocksHash/mPromptLen/mLastKVBlockAddress`——读方为 indexer 工作线程(`UCMIndexerLoadAsync` 在 `IndexerLoadProcess` 持 `mUpdateReqMutex` 下运行)与 `DelReqIDRun`(持同锁);
  2. `SetLastIndexerBlock`(:645 → :656 持锁):读写 `mLastKVBlockAddress`,同上;
  3. `UpdateIndexerMap`(:836 → :848 持锁):`mUseBlockTableIdx` 的 emplace_back/pop_back 与工作线程遍历并发是 UB;
  4. `SubmitTask`(scatter:1170 → 循环段 :1184 持锁):写 `mRealTimeIdx/mSeqLen`、读 `mUseBlockTableIdx`——临界区收窄为 map 拷贝循环(不含 `mTaskManger->SubmitTask` 提交);
  5. `SubmitTaskIndexer`(:1210 → :1220 循环段持锁):memcpy 写 `mUseBlockTableIdx`;
  6. `SetTopkIndices`(:1240 → :1251 持锁):`mAllTopkIndicesTensor` 的 map 插入与 `DelReqIDRun` 的 erase 并发是 UB。
- **修复原因**:`std::map`/`std::unordered_map` 一边遍历一边插入/删除是 UB,典型表现为堆损坏(与 A 类越界写叠加时极难归因)。原有锁覆盖不完整:`SetTokensMap`(mMutexUseToken)、`SetIndexerMap`/`DelBlocksMap`/`DelReqIDRun`/`IndexerLoadProcess`(mUpdateReqMutex)已持锁,但上述 6 个 Python 线程入口完全裸奔。
- **死锁分析**(已核对全部 14 处持锁点):无任何路径嵌套获取两把锁——`CallPrefetchProcess`(mMutexUseToken)调用的函数不含 mUpdateReqMutex;`IndexerLoadProcess`/`SetIndexerMap`(mUpdateReqMutex)调用的函数不含 mMutexUseToken;`MultiBSThreadFun` 在 `CallPrefetchProcess` 返回(锁已释放)后才取 mUpdateReqMutex;`SetIndexerMapForLayer` 仅被持锁的 `SetIndexerMap` 内部调用,无需加锁(避免了自死锁)。
- **遗留(后续项)**:`RunOneBsPrefetch` 的 16 线程 omp 循环依赖调用方预插入外层 key 的脆弱顺序,本次未动(需要数据结构重构)。
- **验证**:需随 C++ 构建跑;并发压测下 TSAN 或长期稳定性观察(`SubmitTask` 每步执行,锁竞争回归主要看每步耗时日志)。

---

# 四、D 类:设备侧 scatter 负索引

## D1【P1】`-1` 槽位索引可达 `npu_scatter_nd_update_`

**位置**:`ucm/integration/vllm/patch/gsa/v0_23_0/vllm_ascend/sfa_kv_transfer_patch.py:387-392`

```python
torch_npu.npu_scatter_nd_update_(
    self.dsa_k_cache.view(-1, k_li.shape[-1]),
    attn_metadata.dsa_slot_mapping[: attn_metadata.gsa_slot_len].view(-1, 1),  # ← 无 <0 过滤
    k_li.view(-1, k_li.shape[-1]),
)
```

**原因链**:
1. `dsa_slot_mapping` 尾部恒为 **-1**:缓冲初始化 `sparse/utils.py:82`(`self.cpu.fill_(-1)`),每步重置 `gsa.py:1147`,实际填充 `gsa.py:1310`;
2. 常规步 `gsa_slot_len` = agent 实际填充数,无 -1;但 **aclgraph 捕获路径** `model_runner_v1_patch.py:2006`:
   ```python
   attn_metadata_i.gsa_slot_len = common_attn_metadata.num_input_tokens   # 含 padding
   ```
   捕获/回放中索引可含 -1 或陈旧槽位,补丁无任何 `< 0` 过滤;
3. 后果二选一:CANN 按负索引回绕 → 写到 `dsa_k_cache` 最后一行(静默脏数据);或 C++ 不校验负值 → 设备侧 OOB 写。

**修复建议**:scatter 前过滤 `dsa_slot_mapping < 0`(或捕获路径强制填 0);顺带说明一个已自洽点:槽位 id 池 1..11999(`MAX_SLOTS_BLOCK_NUM=12000`)× block_size < `dsa_k_cache` dim0=17000(`MAX_INDEXER_BLOCK_NUM`)——但这是跨文件隐式常量约定,见 G1。

### ✅ 修复记录(D1)

- **修复内容**(`v0_23_0` 与 `v0.23.0_pmr` 两个 `sfa_kv_transfer_patch.py`,各两处 GSA 路径 scatter——k 缓存与 k_scale 缓存):
  ```python
  slot_idx = attn_metadata.dsa_slot_mapping[: attn_metadata.gsa_slot_len].view(-1)
  valid = slot_idx >= 0
  torch_npu.npu_scatter_nd_update_(
      self.dsa_k_cache.view(-1, k_li.shape[-1]),
      slot_idx[valid].view(-1, 1),                       # 索引过滤
      k_li.view(-1, k_li.shape[-1])[valid],              # 源数据同步过滤
  )
  ```
- **修复原因**:
  - `dsa_slot_mapping` 缓冲以 `fill_(-1)` 初始化(`utils.py:82`),常规步 `gsa_slot_len` = 实际填充数(-1 不进入 scatter);但 **aclgraph 捕获路径**(`model_runner_v1_patch.py:2006`)`gsa_slot_len = num_input_tokens`(含 padding)→ 捕获/回放中 -1 索引可达 scatter → CANN 负索引回绕写 `dsa_k_cache` 最后一行(静默脏数据)或设备侧越界;
  - 索引与源数据(k_li / k_li_scale)必须**同步**过滤(布尔掩码作用于两者),否则索引与更新值错位。
- **实现约束**:补丁函数经 `FunctionType(code, vars(mod))` 以 **sfa_v1 模块全局**运行——不能引入新的模块级名字,但**局部变量**不受限,故用局部 `slot_idx`/`valid` 实现且只调用张量方法(`>=`、`[mask]`),零新全局依赖。
- **影响面/性能**:常规步 `valid` 全真,增加一次比较 + 两次掩码 gather(每层);全 -1 时(warmup 捕获)scatter 空操作。捕获/回放正确性优先,该开销可接受;如压测敏感,可后续在 `attn_metadata` 上预计算一次掩码(需改 metadata 结构,列为后续项)。
- **验证**:两文件 `ast.parse` 语法通过;aclgraph 捕获/回放场景冒烟 + `dsa_k_cache` 最后一行 canary 数据比对。

---

# 五、E 类:PD 分离逻辑缺陷(合并 gsa_for_pd.py 引入)

> 本类问题的共同背景:统一 GSA 把 buffer 分配端加了 `is_decode_node` gate,消费端漏加对齐 gate。

## E1【P0】P 节点 `init_dsa_data` AttributeError —— 确定性启动崩溃

**位置与链条(四环)**:

1. **buffer 分配被 gate 住**:
   - `gsa.py:495-498`(`__init__` worker 初始化块):
     ```python
     if (role == UcmSparseRole.WORKER and PTOPK_PREFETCH_ENABLE and self.is_decode_node):
         self._init_worker_shared_tables()
         self._init_worker_logger_and_pools(vllm_config)   # → _init_worker_indexer_state → init_gsa_metadata
     ```
   - `gsa.py:753-754`(`init_gsa_metadata` 早退):
     ```python
     if not PTOPK_PREFETCH_ENABLE or not self.is_decode_node:
         return    # ← P 节点直接返回,gsa_swap_topk 等永不分配
     ```
   - `gsa.py:527`:`self.is_decode_node = vllm_config.kv_transfer_config.is_kv_consumer` —— P 节点(is_kv_producer)恒 False;且全文件无 `__init__` 级默认值兜底(`gsa_swap_topk` 等属性从未创建);
2. **消费端无 guard**:`gsa.py:1892-1905`(`init_dsa_data`):
   ```python
   def init_dsa_data(self, model_runner):
       model_runner.gsa_swap_topk = self.gsa_swap_topk   # ← P 节点 AttributeError
   ```
3. **触发入口无角色判断**:`integration/vllm/patch/gsa/v0_23_0/vllm_ascend/worker_patch.py:33-38`(仅判 `has_ucm_sparse() and PTOPK_PREFETCH_ENABLE` + `index_topk` 属性)→ `:140-146`(`maybe_init_ucm_sparse_dsa` → `init_dsa_data`);
4. **agent 创建无角色区分**:`sparse/state.py`(`ensure_ucm_sparse_initialized`:只要 `kv_transfer_config` 非 None 即创建 agent)→ P 节点上 `has_ucm_sparse()` 为 True。

**触发条件**:PD 分离部署 + GSA + `PTOPK_PREFETCH_ENABLE=True` → **P 节点 worker 100% 启动崩溃**。

**注意**:真正的判据是 `is_kv_consumer=False`(误配置两个角色 flag 都为 False 时,即使无意做 PD 分离同样触发)。

**修复建议**:`init_dsa_data` 开头补 `if not PTOPK_PREFETCH_ENABLE or not self.is_decode_node: return`(与 `build_sparse_meta` 同款 gate)——这是最小止血修复。

### ✅ 修复记录(E1)

- **修复内容**:`gsa.py` `init_dsa_data`(`ucm/sandbox/sparse/gsav2/gsa.py:1892`)开头新增早退 gate:
  ```python
  if not PTOPK_PREFETCH_ENABLE or not self.is_decode_node:
      logger.info("[GSA] skip init_dsa_data on non-decode node (is_decode_node=%s)", self.is_decode_node)
      return
  ```
- **修复原因**:
  1. buffer 分配端(`__init__` worker 块 `gsa.py:495-498`、`init_gsa_metadata` `gsa.py:753-754`)均带 `is_decode_node` gate,P 节点上 `gsa_swap_topk` 等属性从未创建;`init_dsa_data` 作为消费端补齐同款 gate,与 `build_sparse_meta`(`gsa.py:2221` 一带)的 gate 风格一致;
  2. **P 节点上 `self.log`(GSALogger,`gsa.py:636` 于 `_init_worker_logger_and_pools` 内创建)同样不存在**,因此 guard 内只能使用模块级 `logger`(`gsa.py:97` `init_logger(__name__)`),否则 guard 自身就会 AttributeError——这是修复过程中规避的一个二次陷阱;
  3. `self.is_decode_node` 由 `_init_runtime_state()`(`gsa.py:492` 无条件调用,`:527` 赋值)保证在任意节点上先于 `init_dsa_data` 存在。
- **影响面**:P 节点跳过 buffer 挂载;P 节点上所有 GSA 数据流(`build_sparse_meta`/`execute_begin` 等)本就带 `is_decode_node` gate 不消费这些属性,model_runner 上无其它无 gate 消费点。PD-mix(D 节点)行为不变。
- **验证**:`py_compile`/`ast.parse` 语法通过;PD 分离部署 P 节点启动冒烟(预期不再 AttributeError,日志出现 skip 记录)。

---

## E2【P1】`_load_kv_dram` 未捕获 C++ throw

**位置**:
- Python 调用点:`gsa.py:2576`(`build_sparse_meta` 内,对 resumed/new 请求逐个调用,`:2382/:2384`)
- C++ 抛出点:`sandbox/sparse/gsav2/prefetch/include/kvcache_prefetch.h:77`(`store_init` 内 null 检查)、`:81`(`UCMLoadSSD`)、`:99`(`UCMLookup`)——store 未注入时 `throw std::runtime_error("storeV1 is null")`(单测 `test_ucm_kv_prefetch.cpp:46-59` 佐证)

**原因**:`_load_kv_dram` 的 `while True` 重试循环(`gsa.py:2575-2587`)**没有 try/except**:

```python
while True:
    result = self.ucm_prefetch_engine.ucm_lookup(lookup_ids)   # ← :2576,C++ 可能 throw
```

`_ensure_store_injected`(见 E4)非异常失败路径只 log+return、`_store_injected` 保持 False,后续 `_load_kv_dram` 一旦遇到 `num_prompt_tokens > SEG_PREFILL_THRESHOLD`(`:2566`)的请求,C++ throw → 穿透 `build_sparse_meta` → `execute_model` → **D 节点在首个长 prompt 请求上崩溃**。

**修复建议**:`_load_kv_dram` 外层包 try/except;`_store_injected` 未成功时跳过本步(下一步重试),而非裸调 `ucm_lookup`。

### ✅ 修复记录(E2,与 E4/E3 联合修复)

- **修复内容**:
  1. `build_sparse_meta`(gsa.py:`_ensure_store_injected` 调用处):改为接收注入状态,`store_ready=False` 时**跳过两个 `_load_kv_dram` 循环**并 log info(下一步调度重试),不再在 store 未注入时裸调 C++;
  2. `_load_kv_dram` 内部:`ucm_lookup` 包 try/except——C++ 抛出(store 未初始化/内部错误)时 `log.error` + return(lookup 为只读探测、无已提交状态,跳过本步安全);`ucm_load_ssd` 包 try/except——lookup 已全部命中、提交加载任务失败意味着该请求将无 DRAM KV(静默 KV 空洞),**带上下文上抛**(`raise ... from e`)而非吞掉;
- **修复原因**:C++ `UCMLookup/UCMLoadSSD`(`prefetch/include/kvcache_prefetch.h:81/99`)在 `mStoreV1_ == nullptr` 时 `throw std::runtime_error`;注入重试窗口(≤5 步)内旧代码照常进入 `_load_kv_dram` → 未捕获 RuntimeError 打穿 `build_sparse_meta → execute_model` → D 节点在首个长 prompt 请求上崩溃。
- **错误处理策略(与 E3 一致的原则)**:lookup 前的失败可安全重试(跳过);lookup 之后的失败意味着 KV 空洞 → fail-fast 上抛。
- **验证**:`ast.parse` 语法通过;PD 分离 D 节点长 prompt 请求 + store 延迟就绪场景冒烟(预期:store 未就绪时日志出现 skip 记录、不再崩溃)。

---

## E3【P1】`_load_kv_dram` 超时静默 KV 空洞 + 阻塞 hang

**位置**:`gsa.py:2575-2587`

```python
retry += 1
if retry > UCM_LOOKUP_MAX_RETRY:
    self.log.error(f"[GSA] ERROR look up timeout for req_id {req_id}\n")
    return    # ← 仅 log 后返回:不加载、不 raise → KV 空洞 → 静默错误输出
...
time.sleep(UCM_LOOKUP_TIME_SPACE)   # ← 阻塞式重试,最坏 ≈10s/请求卡住 forward
```

**修复建议**:超时路径显式 fail-fast(抛出或降级标记,由上层决策),不能静默 return;重试改异步或大幅收紧预算。

### ✅ 修复记录(E3,与 E2 联合修复)

- **修复内容**:`_load_kv_dram` 超时路径(`retry > UCM_LOOKUP_MAX_RETRY`)从 `log.error + return` 改为 `raise RuntimeError`,错误信息说明"继续运行将留下静默 KV 空洞(若干 block 在 store 中永远不命中)";`ucm_load_ssd` 提交失败同样上抛(见 E2)。
- **修复原因**:超时后静默返回 = 该请求不提交 DRAM 加载任务、带着缺失的 KV 继续推理 → **输出静默错误**,比崩溃更难定位;fail-fast 把决策权交给上层(重调度/告警/人工介入)。
- **遗留(后续项)**:阻塞式 `while True × sleep` 重试预算(最坏 ≈10s/请求卡 forward)未改——改异步需要把 lookup/load 移出调度关键路径,属独立设计项;当前至少保证超时会显式报错而非假死。
- **验证**:store 中制造永久 miss 的 block 做超时用例,断言 RuntimeError 与错误信息。

---

## E4【P1】`_ensure_store_injected` 非异常静默失败

**位置**:`gsa.py:942-989`,四条只 log 不 raise 的 return 路径(`:955-959` connector 为 None、`:960-964` 无 store 属性、`:965-969` store 为 None、`:970-974` cc_store() 返回 0);异常路径(`:975-982`)重试 5 次后 raise。

**原因**:非异常失败与"注入成功"在调用方看来不可区分(只能靠 `_store_injected` 私有状态),而下游 `_load_kv_dram` 不检查该状态(见 E2),形成"静默失败 → 下游裸调 → 崩溃"链条。

**修复建议**:失败返回 bool,`build_sparse_meta` 中未注入成功时跳过 `_load_kv_dram`(与 E2 一并修)。

### ✅ 修复记录(E4,与 E2 联合修复)

- **修复内容**:`_ensure_store_injected` 签名改为 `-> bool`,四条非异常 return 路径与异常重试路径统一返回 `False`(注入成功返回 `True`);调用方 `build_sparse_meta` 依据返回值决定是否执行 `_load_kv_dram`(见 E2)。
- **修复原因**:原实现失败与成功在调用方不可区分(仅靠私有状态 `_store_injected`),下游不检查该状态直接裸调 `ucm_lookup`,形成"静默失败 → 下游裸调 → 崩溃"链条。注:本地版本 `_log_store_inject_retry`(gsa.py:921-940)本就带"重试超限后 raise"的 fail-fast(优于远端提交版),保留不动;本次仅补齐"重试窗口内返回状态"的缺口。
- **验证**:`ast.parse` 语法通过。

---

## E5【P2】`attention_finished` gate 反转(行为回归,非崩溃)

**位置**:`gsa.py:1648-1660`

```python
if not PTOPK_PREFETCH_ENABLE or not self.is_pd_mix:   # ← 新增 not self.is_pd_mix
    return
```

**原因**:PD 分离的 D 节点(`is_pd_mix=False`)不再执行 indexer dump 路径。若该路径承担 indexer 数据落盘/上报职责,属于静默功能缺失。

**修复建议**:确认 D 节点是否需要 dump;需要则去掉 `not self.is_pd_mix` 条件。

---

# 六、F 类:补丁框架层缺陷

## F1【P0】`when_imported` 单次触发保护静默丢弃后续补丁钩子

**位置**:`ucm/integration/vllm/patch/utils.py:96-100`

```python
if module_name in sys.modules:
    mod = sys.modules[module_name]
    if not getattr(mod, "_ucm_patched", False):
        setattr(mod, "_ucm_patched", True)
        func(mod)     # ← 只有第一个注册的钩子执行,后续同模块钩子被 _ucm_patched 挡掉
```

**原因**:模块已被 import 的情况下,**第一个**注册的 `when_imported` 钩子执行并把模块标记为 `_ucm_patched`;同一模块上后续注册的任何补丁钩子(如 sfa_v1_patch、sfa_kv_transfer_patch 先后 hook 同一模块)看到标志位已置,**静默跳过**——补丁丢失无任何告警,表现为功能静默缺失或间接崩溃(依赖被丢弃补丁初始化的属性不存在)。

**修复建议**:去掉模块级单次触发标志,改为按钩子粒度去重(如给每个 func 设 `_ucm_hook_done` 标志),或在 hook 被跳过时至少 log.warning。

### ✅ 修复记录(F1)

- **修复内容**(`integration/vllm/patch/utils.py` `when_imported`,两处):
  1. **"模块已导入"路径**(:96-100):去重标志从模块属性 `mod._ucm_patched` 改为**函数对象属性** `func._ucm_hook_done`——每个钩子执行一次,同一模块上后续注册的钩子不再被第一个钩子挤掉;
  2. **注册侧**(:92-99):`_POST_IMPORT_HOOKS` append 前按函数对象去重(`if func not in hooks`),使"模块尚未导入"路径(HookingFinder :44-47 遍历执行全部注册钩子)不会对重复注册的同一函数执行多次。
- **修复原因**:
  - 原机制下,第一个 `when_imported` 钩子在已导入模块上执行后置 `mod._ucm_patched=True`,同模块后续钩子(实际场景:sfa_v1_patch 与 sfa_kv_transfer_patch 等先后 hook 同一模块)**静默跳过**且无任何告警——补丁丢失后表现为功能缺失、或下游因补丁未初始化的属性不存在而间接崩溃,极难排查;
  - "尚未导入"路径本就执行全部钩子(HookingFinder 不检查 `_ucm_patched`),改为钩子级去重后两条路径语义一致;
  - `_ucm_patched` 全仓库无其他消费方(grep 确认),可安全移除该模块级标志;
  - 函数对象默认按标识比较,`func not in hooks` 幂等安全,不影响正常多次注册不同钩子。
- **验证**(临时脚本实跑通过):stub `wrapt`/`ucm.logger` 后执行 utils.py 真实源码——
  - 同一已导入模块先后注册 A/B/A2 三个钩子 + 同一函数对象重复注册:执行序列 `['A','B','A2','DUP']`(全部执行、重复仅一次);
  - 注册表条目数 = 4(注册侧去重生效);
  - `ast.parse` 语法通过。
- **影响面**:所有经由 `when_imported` 的补丁(整个 integration 目录)——修复后此前被丢弃的钩子会真正执行,属于"修复即生效"型;若某些补丁依赖"被丢弃"才不报错,则该修复会暴露其自身错误(这是期望行为,按报错逐个处理)。

---

## F2【P0】`indexer_select_post_process` 重复定义相互覆盖

**位置**:`ucm/integration/vllm/patch/gsa/v0.23.0_pmr/vllm_ascend/device_op_patch.py:13 与 :188` —— **同名函数定义两次,第二个静默覆盖第一个**。

**参照**:`v0_23_0` 目录已修复(第二个改名 `indexer_select_post_process_a5`,device_op_patch.py:200),`v0.23.0_pmr` 目录**未同步修复**。

**原因**:第一个实现(:13 起)的全部逻辑成为死代码,实际生效的是第二个;若二者行为有差异(不同模型拓扑的 select 后处理),走 pmr 路径时前一种模型的行为被静默替换。

**修复建议**:把 `v0_23_0` 的修复方案(`_a5` 改名)同步到 `v0.23.0_pmr/device_op_patch.py:188`。

### ✅ 修复记录(F2)

- **修复内容**(`v0.23.0_pmr/vllm_ascend/device_op_patch.py`):
  1. 第二个定义(`:188` 起,A5 路径)改名为 `indexer_select_post_process_a5`(现 `:201`),并在其前移植了 `v0_23_0` 版本的 NOTE 注释块,说明两个实现(A3/DSV3.2 走 `torch.ops._C_ascend.*` 避免图编译问题;A5 走 `torch_npu.*` 直连)为何**必须**不同名;
  2. `_patch` 中 `A5DeviceAdaptor` 补丁块的 6 处引用同步改为 `indexer_select_post_process_a5`。
- **修复原因**:
  - 两个同名 `def` 在模块导入时第二个遮蔽第一个,`_patch` 按模块级名字取函数,导致 `BaseDeviceAdaptor` 上绑定的实际是 **A5 变体**——A3 路径(A3/DSV3.2 模型)静默走错 indexer 实现(图编译问题回归),第一个实现整体成为死代码;
  - `v0_23_0` 目录此前已修复(第二定义改名 `_a5`),但 `v0.23.0_pmr` 目录未同步——本修复即把现成方案移植过来,保证两个版本目录行为一致。
- **修复后结构**(与 `v0_23_0` 对齐):
  - `:13 indexer_select_post_process`(A3/DSV3.2 路径)→ `_patch` 绑定到 `BaseDeviceAdaptor`;
  - `:201 indexer_select_post_process_a5`(A5 路径)→ `_patch` 绑定到 `A5DeviceAdaptor`。
- **验证**:`ast.parse` 语法通过;grep 确认文件中两个定义名唯一、`A5DeviceAdaptor` 块引用全部指向 `_a5`。

---

## F3【P1】`req_sparsed_slots=None` 无判空

**位置**:
- 默认值来源:`integration/vllm/patch/gsa/v0_23_0/vllm/sched_output_patch.py:20-23`
  ```python
  req_sparsed_slots = kwargs.pop("req_sparsed_slots", None)   # ← 默认 None
  self.req_sparsed_slots = req_sparsed_slots
  ```
- 无判空消费点:`model_runner_v1_patch.py:494`(`req_sparsed_slots = scheduler_output.req_sparsed_slots` 后直接 `req_sparsed_slots[req_id]`,:496)、`:693`;`vllm/gpu_model_runner_patch.py:670/:678`、`:1099-1101`、`:1129-1132` 同样模式。

**原因**:`SchedulerOutput` 若由未打 GSA 调度补丁的路径构造(或 GSA 调度器早退未填该字段),`req_sparsed_slots` 为 None → 消费点 `None[req_id]` → TypeError 运行时崩溃。

**修复建议**:消费点统一判 `if not req_sparsed_slots: ...`(空 dict/None 同样处理,默认非 sparse)。

### ✅ 修复记录(F3)

- **修复内容**(统一模式:`scheduler_output.req_sparsed_slots or {}` + `.get(req_id, INVALID_SLOT)`,共 3 个文件、5 处消费点):
  1. `v0_23_0/vllm_ascend/model_runner_v1_patch.py`:`_prepare_inputs` 内两处循环(`sparsed_positions` 填充、`optimistic_seq_lens_cpu` 填充);
  2. `v0_23_0/vllm/gpu_model_runner_patch.py`:三处循环(req_data 遍历、`sparse_query_pos` 填充、`seq_lens.np` 填充);
  3. `v0.23.0_pmr/vllm_ascend/model_runner_v1_patch.py` 与 `v0.23.0_pmr/vllm/gpu_model_runner_patch.py`:同样的三处模式(pmr 目录与 v0_23_0 是平行副本,一并修复)。
- **修复原因**:`sched_output_patch.py:20-23` 中 `kwargs.pop("req_sparsed_slots", None)` 默认 None——`SchedulerOutput` 若由未打 GSA 调度补丁的路径构造(或 GSA 调度器早退未填字段),消费点 `None[req_id]` 抛 TypeError → 运行时崩溃。除 None 外,dict **缺 key**(部分填充)同样会 KeyError,故用 `.get(req_id, INVALID_SLOT)` 一并兜底——缺失语义即为"非 sparse 请求",与 `INVALID_SLOT` 判定天然一致。
- **顺带核实**:`model_runner_v1_patch.py` 中 `:696/:707` 裸引用局部名 `req_sparsed_slots`(而非 `scheduler_output.` 前缀)——确认其定义 `:494` 在 `_prepare_inputs` 内无条件执行,同函数顺序流可达,**非 NameError**,本次不改写其结构。
- **验证**:两个文件 `ast.parse` 通过;grep 确认全 patch 目录不再有非注释的 `req_sparsed_slots[req_id]` 直接下标访问。

---

## F4【P1】`list(kv_cache)` 无判空

**位置**:`integration/vllm/patch/gsa/v0_23_0/vllm_ascend/sfa_kv_transfer_patch.py:643 与 :649`

```python
maybe_save_kv_layer_to_connector(layer_name, list(kv_cache))   # ← 无条件调用
```

**原因**:补丁后的 MLA/SFA forward 里,两处无条件 `list(kv_cache)`。vLLM 在 warmup / profile / dummy run 阶段传入 `kv_cache=None` → `list(None)` 抛 `TypeError: 'NoneType' object is not iterable` → 启动/预热期崩溃。函数内无 `kv_cache is None` 保护(已 grep 确认)。

**修复建议**:`if kv_cache is None: return`(或提前早退整段 GSA 逻辑)。

### ✅ 修复记录(F4)

- **修复内容**(守在调用点——`list(kv_cache)` 在函数调用前求值,函数内部判空救不了):
  1. `v0_23_0/vllm_ascend/sfa_kv_transfer_patch.py`:补丁后 forward 内两处调用点(`:643` o_proj 分支早退处、`:649` 函数尾部)各包一层 `if kv_cache is not None`;
  2. `v0.23.0_pmr/vllm_ascend/sfa_kv_transfer_patch.py`:同样两处;
  3. `v0.23.0_pmr/vllm_ascend/mla_v1_patch.py:164`:同款一处。
- **修复原因**:vLLM 在 warmup / profile / dummy run 阶段以 `kv_cache=None` 调用 attention 层(vLLM 原生 forward 对 None 的处理就是跳过 cache 写);补丁后的 forward 无条件 `list(kv_cache)` → `list(None)` 抛 `TypeError: 'NoneType' object is not iterable` → 启动/预热期崩溃。跳过保存与 vLLM 原生语义一致(无 cache 即无 KV 可存)。
- **验证**:三个文件 `ast.parse` 语法通过;warmup/profile 冒烟(此前必崩路径)。
- **备注**:未改动 `maybe_save_kv_layer_to_connector` 本体签名(其余调用方均传 list);如后续希望函数自防御,可加 `if not kv_cache_layer: return` 前置,但调用点的 `list(None)` 崩溃仍需调用点判空,故以调用点为准。

---

## F5【P2】`init_kvcache_ptrs` 无判空直接索引

**位置**:`sandbox/sparse/gsav2/gsa.py:2771-2782`

```python
self.prefetch_engine.prefetch_engine_c.set_kvTensor_Layer(
    kv_cache[0], kv_cache[1], ...)   # ← 未检查 kv_cache 非空/长度
```

**修复建议**:入口断言 `kv_cache is not None and len(kv_cache) >= 2`。

---

# 七、G 类:跨文件隐式约定(加固项)

## G1【P2】跨文件常量约定

**位置**:`sparse/utils.py:64-65`

```python
MAX_INDEXER_BLOCK_NUM = 17000
MAX_SLOTS_BLOCK_NUM = 12000
```

**原因**:安全性依赖一组散落各文件的隐式一致:槽位 id 池 1..11999(`MAX_SLOTS_BLOCK_NUM`)× block_size 必须 < `dsa_k_cache` dim0=17000(`MAX_INDEXER_BLOCK_NUM`);`SPA_TOKENS_LEN=5120`、`DEFAULT_TOPK_LEN=2048`、`topk_indices_len=2048`(gsa.py:661)同理。修改任一常量即静默破坏边界安全(直接引发新的越界)。

**修复建议**:常量间关系改为程序推导(如 `MAX_INDEXER_BLOCK_NUM = MAX_SLOTS_BLOCK_NUM + margin`),并在分配处 assert 关系成立。

---

# 八、修复顺序建议

1. **第一批(P0,确定性崩溃/踩写)**:
   - E1(P 节点启动崩溃,改动最小:一处加 gate)
   - A1 + A2(`CopyPreSwapTokens` 边界检查 + dim 口径修正)
   - F1(补丁框架钩子丢失——影响所有补丁可靠性)
   - F2(同步 `_a5` 改名,一处)
2. **第二批(P1,条件性崩溃/静默损坏)**:
   - E2 + E4(一并修:`_ensure_store_injected` 返回状态 + `_load_kv_dram` 包异常)
   - E3(超时 fail-fast)
   - B1(激活互斥闸,生产代码两处调用)
   - F3、F4(判空,改动小)
3. **第三批(P1,需要设计的)**:
   - B2(块所有权 epoch)
   - C1(统一加锁,注意预取线程性能)
   - D1(负索引过滤)
   - B3(keepalive 清单)
   - A3(老引擎接口安全化)
4. **第四批(P2)**:E5、F5、G1。

## 验证手段

- **A 类/C 类**(宿主侧):ASAN/Valgrind 包住 `pre_swap_tokens` 路径跑 MTP 压测;pinned buffer 前后加 canary 字段;抢占-恢复场景(use_tokens 过期)专项用例
- **B 类**(设备侧):ascend 越界检测工具跑图捕获/回放;预取与 forward 并发压测 + KV 数值比对
- **E 类**:PD 分离双节点启动冒烟(P 节点不崩)+ 首请求长 prompt(D 节点 store 未注入路径)
- **F 类**:补丁单测(两个 when_imported 钩子先后注册同模块,断言都执行);warmup/profile 阶段冒烟
- **回归**:无 MTP 常规 decode、PD-mix 全链路,确认修复不引入性能/行为回归
