"""GSA SFA patches for vLLM-Ascend 0.32.0.

Rebased from the 0.23.0 GSA patch layer onto the 0.32.0
``vllm_ascend.attention.sfa_v1``, then aligned with the ucm-v0300 in-tree
GSA adaptation (dsa caches on the indexer backend, unified indexer call).
Upstream 0.32.0 changes accounted for:

* ``AscendSFAImpl.__init__`` is byte-identical to the 0.30.0 version, so the
  GSA init wrapper ports verbatim: it keeps the exact upstream signature,
  runs the original init first, then appends the GSA fields (``layer_id`` /
  ``layer_num`` / ``is_pdmix`` / ``cur_topk_index`` / the
  ``dsa_k_cache`` / ``dsa_k_scale`` aliases to the indexer backend). The
  per-layer DSA caches now live on the indexer backend
  (``indexer_patch.py``); everything the wrapper reads from the wrapped init
  (``head_dim`` / ``has_indexer`` / ``indexer`` / ``vllm_config`` /
  ``layer_name`` / ``qk_rope_head_dim`` / ``uses_packed_sfa_main_cache``)
  is unchanged upstream.
* ``_execute_sparse_flash_attention_process`` dropped the trailing
  floating-NoPE ``masked_fill`` cleanup (upstream removed it in 0.32.0);
  the GSA patch still adds back the ``swap_len`` argument as the 8th
  positional (the GSA forward slices the block table for the swap sub-batch)
  and forwards it to the patched device-op dispatch
  (see ``v0320/vllm_ascend/device_op_patch.py``).
* ``AscendSFAMetadata`` still exists; the class-level GSA field defaults
  (consumed by ``gsa.py`` and the GSA SFA forward via
  ``getattr(attn, ...)``) are attached as before.
* ``AscendIndexer`` (vllm_ascend.attention.indexer) still owns the q/k
  lightning-indexer compute; the GSA forward therefore calls the indexer
  once per layer (the patched backend forward handles the DSA cache write
  and the selection -- see ``indexer_patch.py``) instead of the 0.23.0
  impl-level ``indexer_select_post_process``; the impl-level method is no
  longer injected. The 0.32.0 indexer forward has no
  ``attn_q_gather_handle`` parameter (upstream removed the gather handle),
  so the unified call passes none.
* ``_prepare_indexer_metadata`` (still a no-op hook upstream) is patched
  to copy the per-step GSA routing fields (dsa slot mapping / block table /
  seq lens / indexer_last buffers) from the SFA metadata onto the indexer
  metadata, so the backend's GSA forward finds them on its independently
  built metadata.
* ``DeviceOperator.indexer_select_post_process`` still takes explicit
  ``indexer_k_cache_idx`` / ``indexer_scale_cache_idx`` slots plus the c4
  switch; the patched 0.32.0 device op accepts a keyword-only ``sfa_impl``
  that selects the GSA DSA caches. With the unified indexer call the
  backend itself is the ``sfa_impl`` (it owns the dsa caches and the
  routing attrs).
* ``AscendSFAMetadataBuilder._build`` is NOT re-patched: GSA metadata is
  attached in the patched ``NPUModelRunner._build_attention_metadata``.
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
    the ``dsa_k_cache`` / ``dsa_k_scale`` aliases to the indexer backend)
    is set here.
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
        from ucm.sparse.utils import PTOPK_PREFETCH_ENABLE, init_indexer_layer_map

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

        # 0.30.0/0.32.0 stores the name but no longer derives the layer
        # index. GSA models always construct SFA impls with the shared
        # target-layer prefix (restored by ``v0320/vllm/mla_attention_patch.py``).
        # For a plain (non-GSA) DSA run under the patch layer, a None name
        # maps to layer_num so every ``layer_id < layer_num`` GSA guard
        # stays false.
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

        # LI C8: mirror the canonical derivation (indexer.py / model_runner):
        # the layer-scoped switch comes from AscendConfig, the cache dtype
        # from attention_config.
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

        # GSA: the per-layer DSA (sparse) indexer caches moved onto the
        # indexer backend (allocated in its wrapped __init__, see
        # ``indexer_patch.py``); the impl keeps aliases for the begin hook
        # and the GSA forward. Only layers below layer_num with an indexer
        # on pdmix / decode nodes have a cache.
        if has_ucm_sparse() and PTOPK_PREFETCH_ENABLE:
            if self.has_indexer and (
                self.is_pdmix or not self.vllm_config.kv_transfer_config.is_kv_producer
            ):
                self.dsa_k_cache = self.indexer.impl.dsa_k_cache
                if self.enable_sparse_li_c8:
                    self.dsa_k_scale = self.indexer.impl.dsa_k_scale
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


def _prepare_indexer_metadata(self, indexer_metadata, attn_metadata) -> None:
    """GSA (mirrors the ucm-v0300 in-tree sfa_v1.py): copy the per-step DSA
    routing fields from the SFA attention metadata onto the indexer
    metadata (built independently by the indexer's own builder), so the
    backend's GSA forward (``indexer_patch.py``) finds them.

    Upstream 0.32.0 keeps this as a no-op hook; the gate makes it a no-op
    again whenever the GSA sparse path is inactive.
    """
    from ucm.sparse.state import has_ucm_sparse
    from ucm.sparse.utils import PTOPK_PREFETCH_ENABLE

    if (
        has_ucm_sparse()
        and PTOPK_PREFETCH_ENABLE
        and self.layer_id < self.layer_num
        and (
            self.is_pdmix
            or not self.vllm_config.kv_transfer_config.is_kv_producer
        )
    ):
        indexer_metadata.dsa_slot_mapping = attn_metadata.dsa_slot_mapping
        indexer_metadata.gsa_slot_len = attn_metadata.gsa_slot_len
        indexer_metadata.dsa_block_table = attn_metadata.dsa_block_table
        indexer_metadata.dsa_seq_len = attn_metadata.dsa_seq_len
        indexer_metadata.indexer_last_query = attn_metadata.indexer_last_query
        indexer_metadata.indexer_last_weight = attn_metadata.indexer_last_weight
        indexer_metadata.indexer_last_q_index = attn_metadata.indexer_last_q_index
        indexer_metadata.indexer_last_query_len = attn_metadata.indexer_last_query_len


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
    # block-table row selector (``select_block_index`` slice). 0.30.0/0.32.0
    # has a trailing ``block_table`` kwarg; its only in-tree caller passes 7
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
    # see v0320/vllm_ascend/device_op_patch.py).
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

    # GSA metadata copy hook: upstream 0.32.0 defines this as a no-op on
    # the impl; the patched version copies the per-step DSA routing fields
    # onto the indexer metadata for the backend's GSA forward (the stock
    # branch keeps a no-op through the gate).
    if hasattr(target, "_prepare_indexer_metadata"):
        _patched = _make_patched(mod, _prepare_indexer_metadata)
        patch_or_inject(target, "_prepare_indexer_metadata", _patched)
        logger.info("UCM GSA AscendSFAImpl._prepare_indexer_metadata patch applied")

    if hasattr(target, "_execute_sparse_flash_attention_process"):
        _patched = _make_patched(mod, _execute_sparse_flash_attention_process)
        patch_or_inject(target, "_execute_sparse_flash_attention_process", _patched)
        logger.info(
            "UCM GSA AscendSFAImpl._execute_sparse_flash_attention_process patch applied"
        )
