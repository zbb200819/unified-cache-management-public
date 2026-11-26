"""GSA indexer backend patches for vLLM-Ascend 0.32.0.

Mirrors the ucm-v0300 in-tree adaptation of
``vllm_ascend.attention.indexer.AscendSFAIndexerBackend`` in the patch layer:

* the backend ``__init__`` gains the per-layer GSA block (routing attrs
  ``layer_id`` / ``layer_num`` / ``is_pdmix`` / ``cur_topk_index``, the
  per-layer DSA caches ``dsa_k_cache`` / ``dsa_k_scale``, and
  ``_gsa_layer_active``) -- the caches move off the SFA impl, which keeps
  aliases (``sfa_v1_patch._make_gsa_aware_init``);
* ``_write_dsa_cache`` scatters the k path output into the DSA caches with
  the raw dsa slot mapping (no PCP/DSA-CP gather transform);
* ``forward`` routes GSA layers through the DSA path (k path -> DSA write
  -> offload gather -> selection with ``sfa_impl=self``, dispatched by the
  patched ``DeviceOperator.indexer_select_post_process``), while non-GSA
  layers keep the upstream pipeline untouched (including the 0.32.0 fused
  cache-store path, which the GSA branch never takes because it bypasses
  ``write_cache`` entirely).

The wrapper accepts ``**kwargs`` so both call surfaces work unchanged: the
0.32.0 ``IndexerWrapper.forward`` passes five positional arguments (the
upstream 0.32.0 indexer forward has no ``attn_q_gather_handle`` parameter;
the wrapper only waits on it when a later version passes one).
"""

from __future__ import annotations

from types import FunctionType

from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


def _make_gsa_aware_backend_init(orig_init):
    """Wrap ``AscendSFAIndexerBackend.__init__`` with the GSA block.

    The wrapper keeps the exact upstream signature, runs the original init
    first, then appends the GSA fields. Everything the GSA forward and the
    device-op routing read (``layer_id`` / ``layer_num`` / ``is_pdmix`` /
    ``cur_topk_index`` / ``dsa_k_cache`` / ``dsa_k_scale`` /
    ``_gsa_layer_active``) is set here.
    """

    def __init__(
        self,
        vllm_indexer,
        qk_rope_head_dim,
    ) -> None:
        from ucm.sparse.state import has_ucm_sparse
        from ucm.sparse.utils import (
            MAX_INDEXER_BLOCK_NUM,
            PTOPK_PREFETCH_ENABLE,
            init_indexer_layer_map,
        )

        orig_init(self, vllm_indexer, qk_rope_head_dim)

        # GSA (mirrors the ucm-v0300 in-tree indexer.py): the backend owns
        # the per-layer DSA caches and the routing attrs consumed by
        # DeviceOperator.indexer_select_post_process via ``sfa_impl=self``.
        # The layer index comes from the k cache prefix, exactly as the SFA
        # impl derives it from the layer prefix.
        self.vllm_config = get_current_vllm_config()
        self.layer_id = int(self.k_cache.prefix.split(".")[2])
        self.layer_num = self.vllm_config.model_config.get_num_layers(
            self.vllm_config.parallel_config
        )
        self.is_pdmix = (
            self.vllm_config.kv_transfer_config is None
            or (
                self.vllm_config.kv_transfer_config.is_kv_producer
                and self.vllm_config.kv_transfer_config.is_kv_consumer
            )
        )
        self.topk_layer_info, self.topk_layer_map, self.indexr_num = (
            init_indexer_layer_map(self.vllm_config)
        )
        if self.layer_id < self.layer_num:
            self.cur_topk_index = self.topk_layer_map[self.layer_id]
        else:
            self.cur_topk_index = self.layer_num - 1
        self._gsa_layer_active = False
        if has_ucm_sparse() and PTOPK_PREFETCH_ENABLE:
            if self.is_pdmix or not self.vllm_config.kv_transfer_config.is_kv_producer:
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
            self._gsa_layer_active = (
                self.layer_id < self.layer_num
                and (
                    self.is_pdmix
                    or not self.vllm_config.kv_transfer_config.is_kv_producer
                )
            )

    __init__.__annotations__ = getattr(orig_init, "__annotations__", {})
    return __init__


def _write_dsa_cache(
    self,
    k_li,
    k_li_scale,
    indexer_metadata,
) -> None:
    """GSA (mirrors the ucm-v0300 in-tree indexer.py): scatter the k path
    output into the per-layer DSA caches instead of the vLLM indexer cache.

    The raw dsa slot mapping is used (no PCP/DSA-CP gather transform),
    matching the dev 0.23.0 inline GSA write.
    """
    import torch_npu

    slot_mapping = indexer_metadata.dsa_slot_mapping[: indexer_metadata.gsa_slot_len]
    torch_npu.npu_scatter_nd_update_(
        self.dsa_k_cache.view(-1, k_li.shape[-1]),
        slot_mapping.view(-1, 1),
        k_li.view(-1, k_li.shape[-1]),
    )  # b, s, n, d
    if self.enable_sparse_li_c8:
        assert k_li_scale is not None
        torch_npu.npu_scatter_nd_update_(
            self.dsa_k_scale.view(-1, k_li_scale.shape[-1]),
            slot_mapping.view(-1, 1),
            k_li_scale.view(-1, k_li_scale.shape[-1]),
        )


def _make_gsa_aware_backend_forward(orig_forward):
    """Wrap ``AscendSFAIndexerBackend.forward`` with the GSA routing.

    GSA layers (``_gsa_layer_active``) take the DSA path: k path -> DSA
    cache write -> offload gather -> selection whose device-op dispatch
    reads this layer's DSA caches / dsa_block_table via ``sfa_impl=self``.
    Other layers delegate to the upstream forward unchanged (the upstream
    pipeline keeps the vLLM-owned cache write -- including the 0.32.0
    fused cache-store branch -- and the stock selection).
    """

    def forward(
        self,
        hidden_states,
        q_c,
        k_hidden_states,
        indexer_metadata,
        compute_topk=True,
        **kwargs,
    ):
        if not getattr(self, "_gsa_layer_active", False):
            return orig_forward(
                self,
                hidden_states,
                q_c,
                k_hidden_states,
                indexer_metadata,
                compute_topk=compute_topk,
                **kwargs,
            )

        import torch_npu
        from vllm.triton_utils import HAS_TRITON
        from vllm_ascend.ops.triton.rope import rope_forward_triton_siso

        from ucm.sparse.utils import IS_INDEXER_OFFLOAD

        cos = indexer_metadata.cos
        sin = indexer_metadata.sin
        k_li, k_li_scale, indexer_weights = self.forward_k(k_hidden_states, cos, sin)
        handle = kwargs.get("attn_q_gather_handle")
        if handle is not None:
            # Only present when a later version passes one; keep the DSA
            # write ordered after Q (wait() adds a stream dependency).
            handle.wait()
        # GSA: persist into the per-layer DSA caches with the raw dsa slot
        # mapping instead of the vLLM indexer cache.
        self._write_dsa_cache(k_li, k_li_scale, indexer_metadata)
        if not compute_topk:
            return None

        assert self.wk_weights_proj is not None
        assert self.wq_b is not None
        assert indexer_metadata.actual_seq_lengths_query is not None
        assert indexer_metadata.actual_seq_lengths_key is not None

        if hidden_states is k_hidden_states:
            # The k path already ran wk_weights_proj on these hidden states
            # (SFA hands both stages the same tensor); reuse its weights
            # tail instead of duplicating the GEMM.
            weights = indexer_weights
        else:
            kw, _ = self.wk_weights_proj(hidden_states)
            weights = kw[:, self.head_dim :]

        if isinstance(q_c, tuple):
            q_c_tensor, q_c_scale = q_c
            q_c_tensor = q_c_tensor.view(-1, q_c_tensor.shape[-1])
            quant_matmul_kwargs = dict(
                bias=None,
                output_dtype=hidden_states.dtype,
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
                q_li,
                [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim],
                dim=-1,
            )

            q_li_pe = q_li_pe.unsqueeze(2)
            q_li_pe = torch_npu.npu_rotary_mul(q_li_pe, cos, sin)
            q_li_pe = q_li_pe.squeeze(2)
            q_li = torch.cat([q_li_pe, q_li_nope], dim=-1)

        q_li_scale = None
        q_li_shape_ori = None
        if self.enable_sparse_li_quant:
            q_li_shape_ori = q_li.shape
            q_li, q_li_scale = self._quantize_li_tensor(q_li)

        # GSA: key lengths come from the DSA metadata; the offload gather
        # publishes q/weights to the indexer_last buffers for later layers;
        # the selection op routes to this layer's DSA caches via
        # ``sfa_impl`` (device_op.py GSA routing). GSA never combines with
        # the C4 lightning indexer.
        actual_seq_lengths_key = indexer_metadata.dsa_seq_len
        if IS_INDEXER_OFFLOAD:
            torch.ops._C_gsa.indexer_gather_qw(
                q_li.view(-1, self.n_head * self.head_dim),
                weights.view(-1, self.n_head),
                indexer_metadata.indexer_last_query[self.layer_id][
                    : indexer_metadata.indexer_last_query_len
                ].view(-1, self.n_head * self.head_dim),
                indexer_metadata.indexer_last_weight[self.layer_id][
                    : indexer_metadata.indexer_last_query_len
                ].view(-1, self.n_head),
                indexer_metadata.indexer_last_q_index[
                    : indexer_metadata.indexer_last_query_len
                ]
                .view(-1, 1)
                .expand((-1, self.n_head * self.head_dim)),
                indexer_metadata.indexer_last_q_index[
                    : indexer_metadata.indexer_last_query_len
                ]
                .view(-1, 1)
                .expand((-1, self.n_head)),
            )
        return DeviceOperator.indexer_select_post_process(
            q_li,
            q_li_scale,
            q_li_shape_ori,
            weights,
            self.k_cache.kv_cache,
            INDEXER_K_CACHE_SLOT,
            INDEXER_SCALE_CACHE_SLOT,
            indexer_metadata,
            indexer_metadata.actual_seq_lengths_query,
            actual_seq_lengths_key,
            self.enable_sparse_li_c8,
            False,
            self.use_torch_npu_lightning_indexer,
            sfa_impl=self,
        )

    return forward


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


@when_imported("vllm_ascend.attention.indexer")
def _patch(mod):
    target = mod.AscendSFAIndexerBackend

    # Belt-and-braces: class-level defaults for the GSA fields the patched
    # AscendSFAImpl._prepare_indexer_metadata copies onto each step's
    # indexer metadata instances (dynamic attachment also works because the
    # upstream dataclass has no __slots__, but defaults keep getattr readers
    # safe on early steps).
    if hasattr(mod, "AscendSFAIndexerMetadata"):
        _cls = mod.AscendSFAIndexerMetadata
        for _fname, _fdefault in {
            "dsa_slot_mapping": None,
            "gsa_slot_len": 0,
            "dsa_block_table": None,
            "dsa_seq_len": None,
            "indexer_last_query": None,
            "indexer_last_weight": None,
            "indexer_last_q_index": None,
            "indexer_last_query_len": 0,
        }.items():
            if not hasattr(_cls, _fname):
                setattr(_cls, _fname, _fdefault)

    if hasattr(target, "__init__"):
        replacement = _make_gsa_aware_backend_init(target.__init__)
        _patched = _make_patched(mod, replacement)
        # _make_patched lifts into vars(mod); the wrapped orig_init comes
        # from the closure, which is unaffected by the lift.
        patch_or_inject(target, "__init__", _patched)
        logger.info("UCM GSA AscendSFAIndexerBackend.__init__ wrapper patch applied")

    if not hasattr(target, "_write_dsa_cache"):
        _patched = _make_patched(mod, _write_dsa_cache)
        patch_or_inject(target, "_write_dsa_cache", _patched)
        logger.info("UCM GSA AscendSFAIndexerBackend._write_dsa_cache injected")

    if hasattr(target, "forward"):
        replacement = _make_gsa_aware_backend_forward(target.forward)
        _patched = _make_patched(mod, replacement)
        patch_or_inject(target, "forward", _patched)
        logger.info("UCM GSA AscendSFAIndexerBackend.forward wrapper patch applied")
