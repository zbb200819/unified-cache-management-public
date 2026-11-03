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
| B1 | 前向/预取互斥闸是死代码 → 设备侧并发踩写 | **P0** | 预取块与前向读写块重叠 | kvcache_pre_scatter.cpp:51-88 | **是** | 待修复 |
| E1 | P 节点 `init_dsa_data` AttributeError 启动崩溃 | **P0** | PD 分离 P 节点 | gsa.py:1892-1905 | 否(死代码) | ✅ 已修复 |
| F1 | `when_imported` 单次触发保护静默丢弃后续补丁钩子 | **P0** | 同一模块注册两个补丁 | patch/utils.py:96-100 | **是** | 待修复 |
| F2 | `indexer_select_post_process` 重复定义相互覆盖 | **P0** | v0.23.0_pmr 补丁生效时 | v0.23.0_pmr/.../device_op_patch.py:13,188 | **是** | 待修复 |
| A3 | 老引擎 freeBlockPtr 裸写无界 | P1 | 行宽与表大小错配 | sparse/gsa/.../kvcache_pre.cpp:432 | 是(老引擎) | 待修复 |
| B2 | 块 ID 陈旧 → 跨请求并发踩写 | P1 | 请求抢占/完成时 DMA 在途 | kvcache_pre_scatter.cpp:1305-1313 | **是** | 待修复 |
| B3 | 裸指针 use-after-free 风险 | P1 | Python 张量先于 C++ 释放 | pybinds.cpp:143-148 + gsa.py:1544-1546 | **是** | 待修复 |
| C1 | C++ map 跨线程并发读写无锁 | P1 | Python 写 API 与预取线程同时活跃 | kvcache_pre_scatter.cpp:1167-1171 附近 | **是** | 待修复 |
| D1 | `-1` 槽位索引可达设备侧 scatter | P1 | aclgraph 捕获/回放 | sfa_kv_transfer_patch.py:387-392 | **是** | 待修复 |
| E2 | `_load_kv_dram` 未捕获 C++ throw | P1 | PD 分离 D 节点 + store 注入未就绪 | gsa.py:2576 | 否(死代码) | 待修复 |
| E3 | `_load_kv_dram` 超时静默 KV 空洞 + 阻塞 hang | P1 | PD 分离 D 节点 + lookup 超时 | gsa.py:2575-2587 | 否(死代码) | 待修复 |
| E4 | `_ensure_store_injected` 非异常静默失败 | P1 | connector/store 未就绪 | gsa.py:955-974 | 否(死代码) | 待修复 |
| F3 | `req_sparsed_slots=None` 无判空 | P1 | 非 GSA 调度路径构造的 SchedulerOutput | sched_output_patch.py:20-23 | **是** | 待修复 |
| F4 | `list(kv_cache)` 无判空 | P1 | warmup/profile 阶段 kv_cache=None | sfa_kv_transfer_patch.py:643,649 | **是** | 待修复 |
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

---

## B2【P1】块 ID 陈旧 → 跨请求并发踩写(逻辑别名)

**位置**:`kvcache_pre_scatter.cpp:1305-1313`(`DelBlocksMap`);对照老引擎 `sparse/gsa/prefetch/src/kvcache_pre.cpp:446`(`mDelSeqIds` 检查,但也是 check-then-act,有竞态窗口)

**原因**:预取目标 block id 来自**上一步**的 topk/自由块表。请求在 DMA 在途时被抢占或完成后,vLLM 分配器会把这些 device 块**重新分配给其他请求**(其 prefill 正在计算流上写这些块)→ DMA 与 kernel 并发写同一 KV 块 → 静默 KV 数据损坏。`DelBlocksMap` 只是延迟删 map,**撤不回已提交的 DMA**。

**修复建议**:预取任务携带块所有权 epoch,提交前/DMA 前校验所有权未变化;或预取目标块从专用保留池分配(不与 vLLM 分配器混用)。

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

---

# 三、C 类:C++ 容器跨线程竞争

## C1【P1】map 并发读写无锁 → 堆损坏(UB)

**位置**:`ucm/sandbox/sparse/gsav2/prefetch/src/kvcache_pre_scatter.cpp`
- **有锁的**:`IndexerLoadProcess`(:512 `std::lock_guard lock(mUpdateReqMutex)`)、`DelBlocksMap`(:1307-1312)
- **没锁的**:`SetBlocksMap`/`SetTokensMap`/`UpdateIndexerMap`/`SubmitTask`(写 `mRealTimeIdx`/`mUseBlockTableIdx`/`mSeqLen`,约 :1167-1171 一带)

**原因**:预取线程(`CallPrefetchProcessFun`/`IndexerLoadProcess`)遍历 `mDocsTables`/`mBlocksMap`/`mAllTopkIndicesTensor` 的同时,Python 线程可并发调用上述写接口。`std::map` 一边遍历一边插入/删除是 UB,典型表现即堆损坏。另外 `RunOneBsPrefetch` 的 `#pragma omp parallel for` 16 线程循环体内写 `allNeedLoadBlock[reqID][layerID]`,依赖 `CallPrefetchProcessFun:560-561` **先**预插入外层 key 的脆弱顺序才安全。

**修复建议**:统一给所有 map 写接口加 `mUpdateReqMutex`(与预取线程遍历互斥);`RunOneBsPrefetch` 改为循环外集中预分配。

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

---

## E4【P1】`_ensure_store_injected` 非异常静默失败

**位置**:`gsa.py:942-989`,四条只 log 不 raise 的 return 路径(`:955-959` connector 为 None、`:960-964` 无 store 属性、`:965-969` store 为 None、`:970-974` cc_store() 返回 0);异常路径(`:975-982`)重试 5 次后 raise。

**原因**:非异常失败与"注入成功"在调用方看来不可区分(只能靠 `_store_injected` 私有状态),而下游 `_load_kv_dram` 不检查该状态(见 E2),形成"静默失败 → 下游裸调 → 崩溃"链条。

**修复建议**:失败返回 bool,`build_sparse_meta` 中未注入成功时跳过 `_load_kv_dram`(与 E2 一并修)。

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

---

## F2【P0】`indexer_select_post_process` 重复定义相互覆盖

**位置**:`ucm/integration/vllm/patch/gsa/v0.23.0_pmr/vllm_ascend/device_op_patch.py:13 与 :188` —— **同名函数定义两次,第二个静默覆盖第一个**。

**参照**:`v0_23_0` 目录已修复(第二个改名 `indexer_select_post_process_a5`,device_op_patch.py:200),`v0.23.0_pmr` 目录**未同步修复**。

**原因**:第一个实现(:13 起)的全部逻辑成为死代码,实际生效的是第二个;若二者行为有差异(不同模型拓扑的 select 后处理),走 pmr 路径时前一种模型的行为被静默替换。

**修复建议**:把 `v0_23_0` 的修复方案(`_a5` 改名)同步到 `v0.23.0_pmr/device_op_patch.py:188`。

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

---

## F4【P1】`list(kv_cache)` 无判空

**位置**:`integration/vllm/patch/gsa/v0_23_0/vllm_ascend/sfa_kv_transfer_patch.py:643 与 :649`

```python
maybe_save_kv_layer_to_connector(layer_name, list(kv_cache))   # ← 无条件调用
```

**原因**:补丁后的 MLA/SFA forward 里,两处无条件 `list(kv_cache)`。vLLM 在 warmup / profile / dummy run 阶段传入 `kv_cache=None` → `list(None)` 抛 `TypeError: 'NoneType' object is not iterable` → 启动/预热期崩溃。函数内无 `kv_cache is None` 保护(已 grep 确认)。

**修复建议**:`if kv_cache is None: return`(或提前早退整段 GSA 逻辑)。

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
