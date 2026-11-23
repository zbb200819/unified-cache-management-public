"""GSA SFA patches for vLLM-Ascend 0.30.0.

Rebased from the 0.23.0 GSA patch layer onto the 0.30.0
``vllm_ascend.attention.sfa_v1``. Upstream 0.30.0 changes accounted for:

* ``AscendSFAImpl.__init__`` no longer derives ``layer_id`` from
  ``kv_sharing_target_layer_name`` (it only stores the name), does not set
  ``layer_num`` / ``is_pdmix`` / ``num_spec_tokens`` / the indexer aliases
  (``n_head`` / ``wq_b`` / ``wk_weights_proj`` / ``k_norm``) /
  ``enable_sparse_li_c8`` + c8 cache dtypes, and allocates no per-layer DSA
  caches. Instead of replacing the whole (large, fast-moving) upstream
  ``__init__`` like the 0.23.0 patch did, this patch wraps it and appends the
  GSA block afterwards -- the GSA additions are purely additive.
* ``AscendIndexer`` (vllm_ascend.attention.indexer) now owns the q/k
  lightning-indexer compute (``wq_b`` / ``wk_weights_proj`` / ``q_hadamard``
  buffer / ``enable_sparse_li_c8`` / c8 dtypes). The injected impl-level
  ``indexer_select_post_process`` keeps computing q_li itself (the GSA
  forward calls it with the SFA-stage hidden states, bypassing the indexer's
  own forward for sparse layers) and sources those fields from the wrapped
  aliases / ``self.indexer.q_hadamard``.
* ``DeviceOperator.indexer_select_post_process`` lost its ``sfa_impl`` first
  argument and takes explicit ``indexer_k_cache_idx`` /
  ``indexer_scale_cache_idx`` slots plus ``enable_sparse_li_c4``; the patched
  0.30.0 device op accepts a keyword-only ``sfa_impl`` that selects the GSA
  DSA caches (see ``v0300/vllm_ascend/device_op_patch.py``), so the impl-level
  method passes ``sfa_impl=self`` instead of the 0.23.0 positional style.
* ``record_attention_compute_start`` and the ``FLOAT8_E8M0FNU_DTYPE``
  constant were removed upstream; the latter is now
  ``torch_npu.float8_e8m0fnu``.
* ``_execute_sparse_flash_attention_process`` grew turboquant / NoPE
  (``sparse_mla``) branches and a ``block_table`` kwarg; the GSA patch adds
  back the ``swap_len`` argument (the GSA forward slices the block table for
  the swap sub-batch) and forwards it to the patched device-op dispatch.
* ``AscendSFAMetadataBuilder._build`` is NOT re-patched: the 0.23.0 GSA
  version carried no GSA hooks (GSA metadata is attached in the patched
  ``NPUModelRunner._build_attention_metadata``), and the 0.30.0 in-tree
  builder covers the upstream DSA-CP / c8-reshape plumbing natively.
"""

from __future__ import annotations

from types import FunctionType

from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


def _make_gsa_aware_init(orig_init):
    """Wrap the upstream ``AscendSFAImpl.__init__`` with the GSA block.

    The wrapper keeps the exact upstream signature so positional call sites
    work unchanged, runs the original init first, then appends the GSA
    fields. Everything the GSA forward / device-op routing reads
    (``layer_id`` / ``layer_num`` / ``is_pdmix`` / ``cur_topk_index`` /
    ``dsa_k_cache`` / ``dsa_k_scale`` / indexer aliases / c8 dtypes) is set
    here.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        **kwargs,
    ) -> None:
        from ucm.sparse.state import has_ucm_sparse
        from ucm.sparse.utils import MAX_INDEXER_BLOCK_NUM, init_indexer_layer_map

        orig_init(
            self,
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **kwargs,
        )

        self.head_size = head_size

        # 0.30.0 stores the name but no longer derives the layer index.
        # GSA models always construct SFA impls with the shared target-layer
        # prefix (restored by ``v0300/vllm/mla_attention_patch.py``). For a
        # plain (non-GSA) DSA run under the patch layer, a None name maps to
        # layer_num so every ``layer_id < layer_num`` GSA guard stays false.
        if kv_sharing_target_layer_name is not None:
            self.layer_id = int(kv_sharing_target_layer_name.split(".")[2])
        else:
            self.layer_id = 0

        self.num_spec_tokens = 1
        speculative_config = self.vllm_config.speculative_config
        if speculative_config:
            self.num_spec_tokens += speculative_config.num_speculative_tokens
        self.layer_num = self.vllm_config.model_config.get_num_layers(
            self.vllm_config.parallel_config
        )
        if kv_sharing_target_layer_name is None:
            self.layer_id = self.layer_num
        self.is_pdmix = self.vllm_config.kv_transfer_config is None or (
            self.vllm_config.kv_transfer_config.is_kv_producer
            and self.vllm_config.kv_transfer_config.is_kv_consumer
        )

        # Indexer aliases consumed by the injected impl-level
        # ``indexer_select_post_process`` (0.23.0 kept these on the impl;
        # 0.30.0 moved them onto AscendIndexer, so mirror them back).
        if self.has_indexer:
            self.n_head: int = self.indexer.n_head  # 64
            self.wq_b = self.indexer.wq_b
            self.wk_weights_proj = self.indexer.wk_weights_proj
            self.k_norm = self.indexer.k_norm
        else:
            self.n_head = 0
            self.wq_b = None
            self.wk_weights_proj = None
            self.k_norm = None
        self.is_rope_neox_style = True
        self.use_torch_npu_lightning_indexer = False
        if self.vllm_config.model_config.hf_config.model_type in ["glm_moe_dsa"]:
            self.is_rope_neox_style = False
            self.use_torch_npu_lightning_indexer = True

        # LI C8: mirror the 0.30.0 canonical derivation
        # (indexer.py / model_runner_v1.py): the layer-scoped switch comes
        # from AscendConfig, the cache dtype from attention_config.
        from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype

        self.enable_sparse_li_c8 = get_ascend_config().is_sparse_li_c8_layer(
            self.layer_name
        )
        if self.enable_sparse_li_c8:
            self.c8_k_cache_dtype = kv_cache_dtype_str_to_dtype(
                self.vllm_config.attention_config.indexer_kv_dtype,
                self.vllm_config.model_config,
            )
            if self.c8_k_cache_dtype == torch.float8_e4m3fn:
                self.c8_k_scale_cache_dtype = torch.float32
            elif self.c8_k_cache_dtype == torch.int8:
                self.c8_k_scale_cache_dtype = torch.float16

        # GSA: per-layer DSA (sparse) indexer cache. Only layers below
        # layer_num with an indexer on pdmix / decode nodes allocate it.
        if has_ucm_sparse():
            if self.has_indexer and (
                self.is_pdmix or not self.vllm_config.kv_transfer_config.is_kv_producer
            ):
                # gsa param
                num_slots = MAX_INDEXER_BLOCK_NUM
                if self.enable_sparse_li_c8:
                    indexer_type = self.c8_k_cache_dtype
                else:
                    indexer_type = self.vllm_config.model_config.dtype

                self.dsa_k_cache = torch.empty(
                    (num_slots, self.vllm_config.cache_config.block_size, 1, 128),
                    dtype=indexer_type,
                    device=self.vllm_config.device_config.device_type,
                    pin_memory=False,
                )
                if self.enable_sparse_li_c8:
                    self.dsa_k_scale = torch.empty(
                        (num_slots, self.vllm_config.cache_config.block_size, 1, 1),
                        dtype=self.c8_k_scale_cache_dtype,
                        device=self.vllm_config.device_config.device_type,
                        pin_memory=False,
                    )
            else:
                self.dsa_k_cache = None
        self.topk_layer_info, self.topk_layer_map, self.indexr_num = (
            init_indexer_layer_map(self.vllm_config)
        )
        if self.layer_id < self.layer_num:
            self.cur_topk_index = self.topk_layer_map[self.layer_id]
        else:
            self.cur_topk_index = self.layer_num - 1

    __init__.__annotations__ = getattr(orig_init, "__annotations__", {})
    return __init__


def indexer_select_post_process(
    self,
    x: torch.Tensor,
    q_c: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    kv_cache: tuple[torch.Tensor, ...],
    attn_metadata: M,
    cos: torch.Tensor,
    sin: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_key: torch.Tensor,
):
    from vllm.triton_utils import HAS_TRITON
    from vllm_ascend.ops.triton.rope import rope_forward_triton_siso

    from ucm.sparse.state import has_ucm_sparse
    from ucm.sparse.utils import IS_INDEXER_OFFLOAD

    if not self.has_indexer:
        raise RuntimeError(
            f"indexer_select_post_process should not be called when indexer is None. layer_name={self.layer_name}."
        )

    assert self.wk_weights_proj is not None
    assert self.wq_b is not None

    kw, _ = self.wk_weights_proj(x)
    weights = kw[:, self.head_dim :]
    if isinstance(q_c, tuple):
        q_c_tensor, q_c_scale = q_c
        q_c_tensor = q_c_tensor.view(-1, q_c_tensor.shape[-1])
        quant_matmul_kwargs = dict(
            bias=None,
            output_dtype=x.dtype,
        )
        if q_c_tensor.dtype == torch.float8_e4m3fn:
            if q_c_scale.dim() == 2:
                q_c_scale = q_c_scale.view(q_c_scale.shape[0], -1, 2)
            quant_matmul_kwargs.update(
                scale_dtype=torch_npu.float8_e8m0fnu,
                pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
                group_sizes=[
                    1,
                    1,
                    getattr(self.wq_b.quant_method.quant_method, "group_size", 32),
                ],
            )
        elif q_c_scale.dim() > 1 and q_c_scale.shape[-1] == 1:
            q_c_scale = q_c_scale.squeeze(dim=-1)
        q_li = torch_npu.npu_quant_matmul(
            q_c_tensor,
            self.wq_b.weight,
            self.wq_b.weight_scale,
            pertoken_scale=q_c_scale,
            **quant_matmul_kwargs,
        )
    else:
        q_li, _ = self.wq_b(q_c)
    q_li = q_li.view(-1, self.n_head, self.head_dim)
    if HAS_TRITON:
        q_li = rope_forward_triton_siso(
            q_li,
            cos,
            sin,
            rope_dim=self.qk_rope_head_dim,
            is_neox_style=self.is_rope_neox_style,
        )
    else:
        q_li_pe, q_li_nope = torch.split(
            q_li, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1
        )

        q_li_pe = q_li_pe.unsqueeze(2)
        q_li_pe = torch_npu.npu_rotary_mul(q_li_pe, cos, sin)
        q_li_pe = q_li_pe.squeeze(2)
        q_li = torch.cat([q_li_pe, q_li_nope], dim=-1)

    q_li_scale = None
    q_li_shape_ori = None
    if self.enable_sparse_li_c8:
        # Same math as AscendIndexer._quantize_li_tensor (the hadamard
        # buffer lives on the indexer in 0.30.0).
        assert self.indexer.q_hadamard is not None
        q_li_shape_ori = q_li.shape
        q_li = q_li @ self.indexer.q_hadamard
        q_li, q_li_scale = torch_npu.npu_dynamic_quant(
            q_li.view(-1, self.head_dim), dst_type=self.c8_k_cache_dtype
        )
        q_li_scale = q_li_scale.to(self.c8_k_scale_cache_dtype)  # [b*s,]

    if (
        has_ucm_sparse()
        and self.layer_id < self.layer_num
        and (self.is_pdmix or not self.vllm_config.kv_transfer_config.is_kv_producer)
    ):
        actual_seq_lengths_key = attn_metadata.dsa_seq_len
        if IS_INDEXER_OFFLOAD and self.has_indexer:
            torch.ops._C_gsa.indexer_gather_qw(
                q_li.view(-1, self.n_head * self.head_dim),
                weights.view(-1, self.n_head),
                attn_metadata.indexer_last_query[self.layer_id][
                    : attn_metadata.indexer_last_query_len
                ].view(-1, self.n_head * self.head_dim),
                attn_metadata.indexer_last_weight[self.layer_id][
                    : attn_metadata.indexer_last_query_len
                ].view(-1, self.n_head),
                attn_metadata.indexer_last_q_index[
                    : attn_metadata.indexer_last_query_len
                ]
                .view(-1, 1)
                .expand((-1, self.n_head * self.head_dim)),
                attn_metadata.indexer_last_q_index[
                    : attn_metadata.indexer_last_query_len
                ]
                .view(-1, 1)
                .expand((-1, self.n_head)),
            )

    # GSA routing reads the per-layer dsa_k_cache via ``sfa_impl``, so the
    # cache slots only matter for the upstream fallback path. The forward
    # hands in the composed SFA cache tuple (main + indexer caches), where
    # the indexer k/scale caches sit at (1, 2) with a packed main cache
    # (C8/TQ4) or (2, 3) with split k/v -- the same derivation the 0.23.0
    # device op performed internally on this tuple layout.
    if self.uses_packed_sfa_main_cache:
        indexer_k_cache_idx = 1
        indexer_scale_cache_idx = 2
    else:
        indexer_k_cache_idx = 2
        indexer_scale_cache_idx = 3

    # 0.30.0 device-op signature: explicit indexer cache slots + c4/c8
    # switches, with the GSA routing selected by the keyword-only
    # ``sfa_impl`` (dsa_k_cache / dsa_block_table). GSA never combines with
    # the C4 lightning indexer, so c4 is always False on this path.
    return DeviceOperator.indexer_select_post_process(
        q_li,
        q_li_scale,
        q_li_shape_ori,
        weights,
        kv_cache,
        indexer_k_cache_idx,
        indexer_scale_cache_idx,
        attn_metadata,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        self.enable_sparse_li_c8,
        False,
        self.use_torch_npu_lightning_indexer,
        sfa_impl=self,
    )


def _execute_sparse_flash_attention_process(
    self,
    ql_nope,
    q_pe,
    kv_cache,
    topk_indices,
    attn_metadata,
    actual_seq_lengths_query,
    actual_seq_lengths_key,
    # v0230 convention: the 8th positional is the GSA swap sub-batch's
    # block-table row selector (``select_block_index`` slice). 0.30.0 added
    # a trailing ``block_table`` kwarg; its only in-tree caller passes 7
    # positional args, so reordering keeps the GSA forward (which passes 8)
    # v0230-compatible without breaking upstream call sites.
    swap_len: torch.Tensor = None,
    block_table=None,
):
    if self.enable_sparse_sfa_turboquant:
        attn_out, _, _ = self._turboquant_sfa(
            self._tq_rotate_query(ql_nope, q_pe),
            kv_cache[0],
            topk_indices,
            attn_metadata.block_table if block_table is None else block_table,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
            sparse_mode=3,
        )
        assert self.turboquant is not None
        return self.turboquant.inverse(attn_out)
    if self.qk_rope_head_dim == 0 and not self.enable_sparse_sfa_c8:
        return sparse_mla(ql_nope, kv_cache[0], topk_indices, attn_metadata, self.scale)
    if self.qk_rope_head_dim == 0:
        # Present packed NoPE cache with the logical page size used by its metadata.
        cache = kv_cache[0]
        kv_cache = (cache.view(-1, attn_metadata.block_size, *cache.shape[2:]), *kv_cache[1:])
    # GSA: the device-op dispatch consumes swap_len to slice the block table
    # for the swap sub-batch (positionally, before the keyword-only args --
    # see v0300/vllm_ascend/device_op_patch.py).
    output = DeviceOperator.execute_sparse_flash_attention_process(
        self,
        ql_nope,
        q_pe,
        kv_cache,
        topk_indices,
        attn_metadata,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        swap_len,
        block_table=block_table,
    )
    if self.qk_rope_head_dim == 0:
        # As in floating NoPE, graph-capacity rows may be unwritten by the kernel.
        assert isinstance(output, torch.Tensor)
        valid = torch.arange(ql_nope.shape[0], device=ql_nope.device) < actual_seq_lengths_query[-1]
        output = output.masked_fill(~valid[:, None, None], 0)
    return output


def _make_patched(mod, fn):
    _patched = FunctionType(
        fn.__code__,
        vars(mod),
        name=fn.__name__,
        argdefs=fn.__defaults__,
        closure=fn.__closure__,
    )
    _patched.__annotations__ = fn.__annotations__
    _patched.__kwdefaults__ = fn.__kwdefaults__
    return _patched


@when_imported("vllm_ascend.attention.sfa_v1")
def _patch(mod):
    target = mod.AscendSFAImpl

    # Inject the GSA names into the sfa_v1 module namespace (mirroring the
    # 0.23.0 patch) so the lifted GSA forward in sfa_kv_transfer_patch can
    # resolve them through vars(mod). The forward also imports them
    # function-locally; this belt-and-braces injection keeps the module
    # self-contained for any other lifted code.
    try:
        from vllm.forward_context import ForwardContext, get_forward_context

        from ucm.sparse.state import (
            has_ucm_sparse,
            maybe_execute_sparse_attention_begin,
            maybe_execute_sparse_attention_finished,
        )
        from ucm.sparse.utils import (
            DEFAULT_TOPK_LEN,
            IS_INDEXER_OFFLOAD,
            IS_OPEN_GSA,
            MAX_INDEXER_BLOCK_NUM,
            SPA_TOKENS_LEN,
            init_indexer_layer_map,
        )

        for _name, _val in {
            "ForwardContext": ForwardContext,
            "get_forward_context": get_forward_context,
            "maybe_execute_sparse_attention_begin": maybe_execute_sparse_attention_begin,
            "maybe_execute_sparse_attention_finished": maybe_execute_sparse_attention_finished,
            "has_ucm_sparse": has_ucm_sparse,
            "SPA_TOKENS_LEN": SPA_TOKENS_LEN,
            "DEFAULT_TOPK_LEN": DEFAULT_TOPK_LEN,
            "MAX_INDEXER_BLOCK_NUM": MAX_INDEXER_BLOCK_NUM,
            "IS_INDEXER_OFFLOAD": IS_INDEXER_OFFLOAD,
            "IS_OPEN_GSA": IS_OPEN_GSA,
            "init_indexer_layer_map": init_indexer_layer_map,
        }.items():
            if not hasattr(mod, _name):
                setattr(mod, _name, _val)
    except Exception as e:
        logger.warning(f"GSA sfa_v1 module-name injection failed: {e}")

    # Add the GSA/DSA fields to AscendSFAMetadata (class-level defaults;
    # the patched NPUModelRunner._build_attention_metadata attaches the real
    # per-step buffers onto the built instances). has_prefill is attached by
    # the runner patch using the 0.23.0 builder formula; the class default
    # keeps getattr(attn, "has_prefill", False) readers (gsa.py) safe.
    if hasattr(mod, "AscendSFAMetadata"):
        _cls = mod.AscendSFAMetadata
        _cls.__annotations__.setdefault("has_prefill", bool)
        for _fname, _fdefault in {
            "has_prefill": False,
            "is_graph_dummy": False,
            "topk_indices_global": None,
            "dsa_block_table": None,
            "dsa_slot_mapping": None,
            "dsa_seq_len": None,
            "gsa_prefetch_index": None,
            "gsa_prefetch_topk": None,
            "gsa_swap_index": None,
            "gsa_swap_topk": None,
            "gsa_warmup_index": None,
            "gsa_warmup_topk": None,
            "gsa_prefetch_len": 0,
            "gsa_swap_len": 0,
            "gsa_warmup_len": 0,
            "gsa_slot_len": 0,
            "indexer_last_query": None,
            "indexer_last_q_index": None,
            "indexer_last_weight": None,
            "indexer_last_query_len": 0,
            "num_padding_reqs": 1,
            "gsa_prefill_query": None,
            "select_block_index": None,
            "gsa_topk_win_tmp": None,
        }.items():
            _cls.__annotations__.setdefault(_fname, object)
            if not hasattr(_cls, _fname):
                setattr(_cls, _fname, _fdefault)
        logger.info("UCM GSA AscendSFAMetadata gsa fields patch applied")

    # __init__: wrap (not replace) the upstream init with the GSA block.
    if hasattr(target, "__init__"):
        replacement = _make_gsa_aware_init(target.__init__)
        _patched = _make_patched(mod, replacement)
        # _make_patched lifts into vars(mod); the wrapped orig_init comes
        # from the closure, which is unaffected by the lift.
        patch_or_inject(target, "__init__", _patched)
        logger.info("UCM GSA AscendSFAImpl.__init__ wrapper patch applied")

    # Impl-level lightning-indexer q computation + GSA routing. 0.30.0 does
    # not define this method on the impl (AscendIndexer.forward owns the
    # compute and calls the device op without sfa_impl), so it is injected;
    # the GSA forward (sfa_kv_transfer_patch) calls it for sparse layers.
    if not hasattr(target, "indexer_select_post_process"):
        _patched = _make_patched(mod, indexer_select_post_process)
        patch_or_inject(target, "indexer_select_post_process", _patched)
        logger.info("UCM GSA AscendSFAImpl.indexer_select_post_process injected")

    if hasattr(target, "_execute_sparse_flash_attention_process"):
        _patched = _make_patched(mod, _execute_sparse_flash_attention_process)
        patch_or_inject(target, "_execute_sparse_flash_attention_process", _patched)
        logger.info(
            "UCM GSA AscendSFAImpl._execute_sparse_flash_attention_process patch applied"
        )
