"""GSA device_op patches for vLLM-Ascend 0.30.0.

Rebased from the 0.23.0 GSA patch layer onto the vLLM-Ascend 0.30.0
device_op. Upstream 0.30.0 changes accounted for here:

- ``indexer_select_post_process`` lost its ``sfa_impl`` first argument and
  now receives explicit ``indexer_k_cache_idx`` / ``indexer_scale_cache_idx``
  slot indices (the caller, ``vllm_ascend.attention.indexer``, passes the
  indexer's own cache tuple). The GSA routing needs the SFA impl state
  (``layer_id`` / ``layer_num`` / ``is_pdmix`` / ``cur_topk_index`` /
  ``dsa_k_cache`` / ``dsa_k_scale``), so the patched version keeps the 0.30.0
  signature and adds an optional keyword-only ``sfa_impl``: the upstream
  ``indexer.py`` call site keeps working (``sfa_impl=None`` -> upstream
  behavior), while the GSA SFA forward passes ``sfa_impl=self`` to select the
  per-layer DSA cache and block table.
- ``execute_kv_quant_sparse_flash_attention`` was renamed by upstream to
  ``_execute_kv_quant_sparse_flash_attention_vllm`` (staticmethod on both
  Base and A5); the GSA patch now targets the new name on both adaptors.
- ``execute_sparse_flash_attention_process`` no longer takes ``swap_len``;
  the GSA patch restores it (the GSA forward slices the block table for the
  swap sub-batch) and forwards ``actual_sparse_lengths`` (prepared by
  ``gsa.py _update_attn_gsa_lens``) into the kv-quant dispatch.

The C8 kv-quant sparse attention is routed through
``torch.ops._C_gsa.npu_kv_quant_sparse_flash_attention_gsa`` (standalone
gsa_ops wheel) on both A3 (BaseDeviceAdaptor) and A5 (A5DeviceAdaptor).
"""

from __future__ import annotations

from types import FunctionType

from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


def indexer_select_post_process(
    q_li: torch.Tensor,
    q_li_scale: torch.Tensor | None,
    q_li_shape_ori: tuple[Any, ...] | None,
    weights: torch.Tensor,
    kv_cache: tuple,
    indexer_k_cache_idx: int,
    indexer_scale_cache_idx: int,
    attn_metadata,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_key: torch.Tensor,
    enable_sparse_li_c8: bool,
    enable_sparse_li_c4: bool,
    use_torch_npu_lightning_indexer: bool,
    *,
    sfa_impl=None,
) -> torch.Tensor:
    from ucm.sparse.state import has_ucm_sparse

    indexer_cache_idx = indexer_k_cache_idx
    if enable_sparse_li_c4:
        raise RuntimeError("C4 lightning indexer is only supported on A5 devices.")

    # GSA routing: with sparse state active, attention layers read the
    # per-layer DSA cache (dsa_k_cache / dsa_k_scale) and the per-step
    # top-k block table instead of the shared indexer cache tuple.
    if (
        sfa_impl is not None
        and has_ucm_sparse()
        and sfa_impl.layer_id < sfa_impl.layer_num
        and (
            sfa_impl.is_pdmix
            or not sfa_impl.vllm_config.kv_transfer_config.is_kv_producer
        )
    ):
        block_table = attn_metadata.dsa_block_table[sfa_impl.cur_topk_index][
            : len(actual_seq_lengths_query)
        ]
        dsa_cache = sfa_impl.dsa_k_cache
        if enable_sparse_li_c8:
            dsa_cache_scale = sfa_impl.dsa_k_scale
    else:
        block_table = attn_metadata.block_table
        dsa_cache = kv_cache[indexer_cache_idx]
        if enable_sparse_li_c8:
            dsa_cache_scale = kv_cache[indexer_scale_cache_idx]

    if enable_sparse_li_c8:
        # ``kv_cache`` is the indexer's own cache tuple (k + scale).
        assert len(kv_cache) == 2
        assert q_li_scale is not None
        assert q_li_shape_ori is not None
        weights = weights.to(torch.float16)
        topk_indices = torch.ops._C_ascend.npu_lightning_indexer_quant(
            query=q_li.view(q_li_shape_ori),
            key=dsa_cache,
            weights=weights,
            query_dequant_scale=q_li_scale.view(q_li_shape_ori[:-1]),
            key_dequant_scale=dsa_cache_scale.squeeze(2),  # B S N D -> B S D
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
            block_table=block_table,
            query_quant_mode=0,
            key_quant_mode=0,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=2048,
            sparse_mode=3,
        )
    elif use_torch_npu_lightning_indexer:
        topk_indices, _ = torch_npu.npu_lightning_indexer(
            query=q_li,
            key=dsa_cache,
            weights=weights,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
            block_table=block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=2048,
            sparse_mode=3,
        )
    else:
        topk_indices, _ = torch.ops._C_ascend.npu_lightning_indexer(
            query=q_li,
            key=dsa_cache,
            weights=weights,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
            block_table=block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=2048,
            sparse_mode=3,
        )
    return topk_indices


def execute_sparse_flash_attention_process(
    cls,
    sfa_impl,
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv_cache: tuple,
    topk_indices: torch.Tensor,
    attn_metadata,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_key: torch.Tensor,
    swap_len: torch.Tensor = None,
    *,
    block_table: torch.Tensor | None = None,
    sparse_mode: int = 3,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if block_table is None:
        block_table = attn_metadata.block_table
    if swap_len is not None:
        block_table = attn_metadata.block_table[swap_len]
    kv = kv_cache[0]

    # The kv-quant sparse attention op only accepts packed quantized KV.
    # Do not route by the feature flag alone: tests and fallback paths may
    # pass a normal BF16/FP16 KV cache even when the mocked impl exposes a
    # truthy attribute.
    use_kv_quant_sparse_attention = kv.dtype in (
        torch.int8,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    )
    if use_kv_quant_sparse_attention:
        result = cls._execute_kv_quant_sparse_flash_attention_vllm(
            sfa_impl,
            ql_nope,
            q_pe,
            kv,
            block_table,
            topk_indices,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
            sparse_mode=sparse_mode,
            return_lse=return_lse,
            # actual_sparse_lengths: per-req 真实 topk 长度（kernel 按 bIdx=req
            # 索引读取，batchSize = len(actual_seq_lengths_query)），由 gsa.py
            # _update_attn_gsa_lens 准备（decode 分支下发 per-req 数组：swap req=
            # SPA_TOKENS_LEN，非 swap=DEFAULT_TOPK_LEN，padding req=0；prefill
            # 分支置 None——两次 SFA 的 topk 均全宽有效且 swap 子批是重排 batch，
            # 全量数组行号与其不对应），用于 kernel 跳过 topk_indices 尾部 -1
            # padding 的迭代与 padding req。
            actual_sparse_lengths=getattr(
                attn_metadata, "actual_sparse_lengths", None
            ),
        )
    else:
        key_rope = kv_cache[1]
        result = torch.ops._C_ascend.npu_sparse_flash_attention(
            query=ql_nope,
            key=kv,
            value=kv,
            sparse_indices=topk_indices,
            scale_value=sfa_impl.scale,
            sparse_block_size=1,
            block_table=block_table,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_kv=actual_seq_lengths_key,
            query_rope=q_pe,
            key_rope=key_rope,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=sparse_mode,
            attention_mode=2,
            return_softmax_lse=return_lse,
        )
    if not isinstance(result, tuple):
        if return_lse:
            raise RuntimeError(
                "Sparse flash attention did not return softmax max/sum for DCP LSE merge."
            )
        return result
    if return_lse:
        return result
    else:
        return result[0]


def _execute_kv_quant_sparse_flash_attention_vllm(
    sfa_impl,
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv: torch.Tensor,
    block_table: torch.Tensor,
    topk_indices: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_key: torch.Tensor,
    *,
    sparse_mode: int = 3,
    return_lse: bool = False,
    actual_sparse_lengths: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """C8 sparse attention via torch.ops._C_gsa (Base on A3, A5DeviceAdaptor on A5).

    Replaces ``BaseDeviceAdaptor._execute_kv_quant_sparse_flash_attention_vllm``
    and its A5 override (vllm-ascend device_op.py), which call
    ``torch.ops._C_ascend.npu_kv_quant_sparse_flash_attention_vllm`` /
    ``torch_npu.npu_kv_quant_sparse_flash_attention`` without
    ``actual_sparse_lengths``. The GSA op
    (``torch.ops._C_gsa.npu_kv_quant_sparse_flash_attention_gsa``, registered
    in ops/torch_binding.cpp) is the same aclnn-backed kernel but lives in the
    standalone gsa_ops wheel so it no longer depends on the build-time
    vllm-ascend csrc patch.

    Parameter mapping: the GSA op accepts the same keyword args as the ascend
    op plus 6 optional ones (key_dequant_scale / value_dequant_scale / sinks /
    actual_sparse_lengths / pre_tokens / next_tokens).
    ``actual_sparse_lengths`` is a per-req 真实 topk 长度数组（kernel 按
    bIdx=req 索引读取，batchSize = len(actual_seq_lengths_query) = req 数；
    gsa.py _update_attn_gsa_lens 准备：decode 分支下发，swap req=
    SPA_TOKENS_LEN / 非 swap=DEFAULT_TOPK_LEN / padding req=0 即跳过；prefill
    分支置 None——两次 SFA 的 topk 均全宽有效且 swap 子批是重排 batch，全量
    数组行号与其不对应），kernel 用它跳过 topk_indices 尾部 -1 padding 迭代
    与 padding req（5120-padding-1 性能关键）；其余 5 个 optional 默认 None，
    省略以匹配原调用面。``key_quant_mode`` / ``value_quant_mode`` 在 GSA
    schema 中位于 optional args 之前（见 torch_binding.cpp:223-235），用
    keyword 传递以避免位置依赖。
    """
    query = torch.cat([ql_nope, q_pe], dim=-1).contiguous()
    result = torch.ops._C_gsa.npu_kv_quant_sparse_flash_attention_gsa(
        query=query,
        key=kv,
        value=kv,
        sparse_indices=topk_indices,
        scale_value=sfa_impl.scale,
        key_quant_mode=2,
        value_quant_mode=2,
        block_table=block_table,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_kv=actual_seq_lengths_key,
        actual_sparse_lengths=actual_sparse_lengths,
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=sparse_mode,
        attention_mode=2,
        quant_scale_repo_mode=1,
        tile_size=getattr(sfa_impl, "sfa_qsfa_tile_size", 128),
        rope_head_dim=getattr(sfa_impl, "qk_rope_head_dim", q_pe.shape[-1]),
        return_softmax_lse=return_lse,
    )
    # 0.30.0's execute_sparse_flash_attention_process consumes the raw
    # result (tensor, or (out, softmax_max, softmax_sum) when return_lse),
    # so no extra formatting is needed here.
    return result


# NOTE: A3 (BaseDeviceAdaptor) and A5 (A5DeviceAdaptor) intentionally use
# different indexer implementations. The function above (the first
# ``indexer_select_post_process``) is the A3/DSV3.2 path: it routes the
# lightning indexer through ``torch.ops._C_ascend.*`` (npu_lightning_indexer /
# npu_lightning_indexer_quant, not yet migrated to gsa_ops) to avoid graph
# compilation issues observed when using ``torch_npu.npu_lightning_indexer``
# on A3. The C8 sparse attention path
# (``_execute_kv_quant_sparse_flash_attention_vllm`` above) is routed through
# ``torch.ops._C_gsa.npu_kv_quant_sparse_flash_attention_gsa`` (standalone
# gsa_ops wheel). The function below is the A5 path, which goes through
# ``torch_npu.*`` directly.
#
# They MUST keep distinct names. If both are named
# ``indexer_select_post_process`` the second ``def`` shadows the first at
# module import time, so ``_patch`` (which looks up the function by its
# module-level name) would always bind the A5 variant onto every adaptor,
# including BaseDeviceAdaptor on A3 -- which is the bug this rename fixes.
def indexer_select_post_process_a5(
    q_li: torch.Tensor,
    q_li_scale: torch.Tensor | None,
    q_li_shape_ori: tuple[Any, ...] | None,
    weights: torch.Tensor,
    kv_cache: tuple,
    indexer_k_cache_idx: int,
    indexer_scale_cache_idx: int,
    attn_metadata,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_key: torch.Tensor,
    enable_sparse_li_c8: bool,
    enable_sparse_li_c4: bool,
    use_torch_npu_lightning_indexer: bool,
    *,
    sfa_impl=None,
) -> torch.Tensor:
    from ucm.sparse.state import has_ucm_sparse

    indexer_cache_idx = indexer_k_cache_idx
    indexer_scale_cache_idx = indexer_scale_cache_idx

    # GSA routing (see the A3 variant above for the rationale).
    if (
        sfa_impl is not None
        and has_ucm_sparse()
        and sfa_impl.layer_id < sfa_impl.layer_num
        and (
            sfa_impl.is_pdmix
            or not sfa_impl.vllm_config.kv_transfer_config.is_kv_producer
        )
    ):
        block_table = attn_metadata.dsa_block_table[sfa_impl.cur_topk_index][
            : len(actual_seq_lengths_query)
        ]
        dsa_cache = sfa_impl.dsa_k_cache
        if enable_sparse_li_c8:
            dsa_k_scale = sfa_impl.dsa_k_scale.squeeze(2)
    else:
        block_table = attn_metadata.block_table
        dsa_cache = kv_cache[indexer_cache_idx]
        if enable_sparse_li_c8:
            dsa_k_scale = kv_cache[indexer_scale_cache_idx].squeeze(2)

    if enable_sparse_li_c4:
        assert len(kv_cache) == 2
        assert q_li_shape_ori is not None
        assert q_li_scale is not None

        # C4 uses cann_ops_transformer.quant_lightning_indexer (V2 API)
        # with quant_mode=5: q/k=float4_e2m1, scale=float8_e8m0, w=float32.
        # V2 API uses: topk (not sparse_count), mask_mode (not sparse_mode),
        # cu_seqlens_q/seqused_k (not actual_seq_lengths_*),
        # and requires metadata from quant_lightning_indexer_metadata.
        # k_descale (PA_BBND, quant_mode=5): (block_num, block_size, k_n, d/64, 2)
        key_dequant_scale = kv_cache[indexer_scale_cache_idx]
        # quant_mode=5 requires weights to be float32.
        weights_c4 = weights.to(torch.float32)

        # TND layout requires cu_seqlens_q (b+1,) cumulative.
        # actual_seq_lengths_query is cum_query_lens = query_start_loc[1:] (b,).
        # Reconstruct full cu_seqlens_q by prepending 0.
        cu_seqlens_q = F.pad(actual_seq_lengths_query, (1, 0), value=0)
        seqused_k = actual_seq_lengths_key

        num_heads_q = q_li_shape_ori[1]
        head_dim = q_li_shape_ori[-1]
        batch_size = seqused_k.shape[0]

        quant_li_metadata_op, quant_li_op = (
            A5DeviceAdaptor._load_cann_quant_lightning_indexer_ops()
        )

        metadata = quant_li_metadata_op(
            num_heads_q=num_heads_q,
            num_heads_k=1,
            head_dim=head_dim,
            topk=2048,
            quant_mode=5,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            batch_size=batch_size,
            max_seqlen_q=-1,
            max_seqlen_k=-1,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=3,
            cmp_ratio=1,
        )

        # FP4 (float4_e2m1fn_x2) packs 2 values per byte: npu_dynamic_mx_quant
        # returns half the elements.  View q_li to the packed shape (d/2)
        # rather than the original logical shape (d).
        q_li_packed_shape = (*q_li_shape_ori[:-1], q_li_shape_ori[-1] // 2)
        topk_indices, _ = quant_li_op(
            q_li.view(q_li_packed_shape),
            kv_cache[indexer_cache_idx],
            weights_c4,
            q_li_scale,
            key_dequant_scale,
            topk=2048,
            quant_mode=5,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            block_table=block_table,
            metadata=metadata,
            max_seqlen_q=-1,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=3,
            cmp_ratio=1,
            return_value=0,
        )
    elif enable_sparse_li_c8:
        # ``kv_cache`` is the indexer's own cache tuple (k + scale).
        assert len(kv_cache) == 2
        assert q_li_shape_ori is not None

        if q_li_scale is not None:
            q_li_scale = q_li_scale.view(q_li_shape_ori[:-1])

            topk_indices = torch_npu.npu_quant_lightning_indexer(
                query=q_li.view(q_li_shape_ori),
                key=dsa_cache,
                weights=weights,
                query_dequant_scale=q_li_scale,
                key_dequant_scale=dsa_k_scale,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
                block_table=block_table,
                query_quant_mode=0,
                key_quant_mode=0,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=2048,
                sparse_mode=3,
            )
        else:
            topk_indices, _ = torch_npu.npu_lightning_indexer(
                query=q_li.view(q_li_shape_ori),
                key=dsa_cache,
                weights=weights,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
                block_table=block_table,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=2048,
                sparse_mode=3,
            )
    else:
        topk_indices, _ = torch_npu.npu_lightning_indexer(
            query=q_li,
            key=dsa_cache,
            weights=weights,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
            block_table=block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=2048,
            sparse_mode=3,
        )
    return topk_indices


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


@when_imported("vllm_ascend.device.device_op")
def _patch(mod):
    if hasattr(mod.BaseDeviceAdaptor, "indexer_select_post_process"):
        _patched = _make_patched(mod, indexer_select_post_process)
        patch_or_inject(
            mod.BaseDeviceAdaptor,
            "indexer_select_post_process",
            staticmethod(_patched),
        )
        logger.info(
            "UCM GSA BaseDeviceAdaptor.indexer_select_post_process patch applied"
        )

    if hasattr(mod.BaseDeviceAdaptor, "execute_sparse_flash_attention_process"):
        _patched = _make_patched(mod, execute_sparse_flash_attention_process)
        patch_or_inject(
            mod.BaseDeviceAdaptor,
            "execute_sparse_flash_attention_process",
            classmethod(_patched),
        )
        logger.info(
            "UCM GSA BaseDeviceAdaptor.execute_sparse_flash_attention_process patch applied"
        )

    if hasattr(mod.BaseDeviceAdaptor, "_execute_kv_quant_sparse_flash_attention_vllm"):
        # Route C8 (kv-quant) sparse attention through torch.ops._C_gsa
        # (standalone gsa_ops wheel) instead of torch.ops._C_ascend /
        # torch_npu built-ins.
        _patched = _make_patched(mod, _execute_kv_quant_sparse_flash_attention_vllm)
        patch_or_inject(
            mod.BaseDeviceAdaptor,
            "_execute_kv_quant_sparse_flash_attention_vllm",
            staticmethod(_patched),
        )
        logger.info(
            "UCM GSA BaseDeviceAdaptor._execute_kv_quant_sparse_flash_attention_vllm patch applied"
        )

    if hasattr(mod.A5DeviceAdaptor, "indexer_select_post_process"):
        # A5 keeps the torch_npu-based implementation; do NOT use the
        # module-level ``indexer_select_post_process`` here, which after the
        # rename above resolves to the A3 (_C_ascend) variant.
        _patched = _make_patched(mod, indexer_select_post_process_a5)
        patch_or_inject(
            mod.A5DeviceAdaptor,
            "indexer_select_post_process",
            staticmethod(_patched),
        )
        logger.info(
            "UCM GSA A5DeviceAdaptor.indexer_select_post_process patch applied"
        )

    if hasattr(mod.A5DeviceAdaptor, "_execute_kv_quant_sparse_flash_attention_vllm"):
        # A5's upstream override has no ``actual_sparse_lengths`` kwarg
        # (TypeError once the patched ``execute_sparse_flash_attention_process``
        # passes it). Route it through the same torch.ops._C_gsa kernel as
        # Base: the call surface is identical (scales live in the packed kv
        # cache via quant_scale_repo_mode=1) and the fp8-e4m3 kv dtype is
        # handled inside the C++ dispatch. This also covers the upstream
        # RoPE0/LSE fallback that used to delegate to Base.
        _patched = _make_patched(mod, _execute_kv_quant_sparse_flash_attention_vllm)
        patch_or_inject(
            mod.A5DeviceAdaptor,
            "_execute_kv_quant_sparse_flash_attention_vllm",
            staticmethod(_patched),
        )
        logger.info(
            "UCM GSA A5DeviceAdaptor._execute_kv_quant_sparse_flash_attention_vllm patch applied"
        )
