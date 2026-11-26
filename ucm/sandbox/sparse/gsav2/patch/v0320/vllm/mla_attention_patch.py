"""Pass the MLA layer prefix as kv_sharing_target_layer_name in vLLM 0.32.0.

vLLM 0.32.0 (like 0.30.0) passes ``kv_sharing_target_layer_name=None`` when
constructing the MLA impl, but the GSA SFA impl derives ``layer_id`` from the
layer prefix (``int(prefix.split(".")[2])``) and looks up ``dsa_k_cache``/
``topk_layer_map`` per layer. This patch restores ``prefix`` as the value so
the GSA ``AscendSFAImpl.__init__`` wrapper keeps working.

Rebased from the 0.30.0 patch. Upstream 0.32.0 changes inside ``__init__``
accounted for: ``kv_b_proj`` now sets ``_cpu_skip_gemm_dispatch`` (was
``skip_weight_relayout``), ``pcp_shard_decode_requests`` is cached on the
layer, and the zero-arg ``super().__init__()`` is rewritten as an explicit
``super(MLAAttention, self)`` call (no ``__class__`` closure cell survives the
FunctionType re-homing).
"""

from __future__ import annotations

from types import FunctionType

from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


def __init__(
    self,
    num_heads: int,
    scale: float,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    q_lora_rank: int | None,
    kv_lora_rank: int,
    kv_b_proj: ColumnParallelLinear,
    dcp_q_replicate: bool = False,
    cache_config: CacheConfig | None = None,
    quant_config: QuantizationConfig | None = None,
    prefix: str = "",
    attn_backend: type[AttentionBackend] | None = None,
    use_sparse: bool = False,
    indexer: object | None = None,
    topk_indices_buffer: torch.Tensor | None = None,
    index_group_builder: "SparseMLAIndexGroupBuilder | None" = None,
    non_causal_multi_token_decode: bool = False,
    sliding_window: int | None = None,
    prefill_backend_cls: type[MLAPrefillBackend] | None = None,
    **extra_impl_args,
):
    super(MLAAttention, self).__init__()
    self.num_heads = num_heads
    self.scale = scale
    self.qk_nope_head_dim = qk_nope_head_dim
    self.qk_rope_head_dim = qk_rope_head_dim
    self.v_head_dim = v_head_dim
    self.q_lora_rank = q_lora_rank
    self.kv_lora_rank = kv_lora_rank
    self.kv_b_proj = kv_b_proj
    self.dcp_q_replicate = dcp_q_replicate
    self.W_UK_T_dcp_qrep: torch.Tensor | None = None
    self.head_size = kv_lora_rank + qk_rope_head_dim
    self.layer_name = prefix
    self.indexer = indexer
    self.non_causal_multi_token_decode = non_causal_multi_token_decode
    self.sliding_window = sliding_window
    self.num_kv_heads = 1
    self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

    if cache_config is not None:
        kv_cache_dtype: CacheDType = cache_config.cache_dtype
    else:
        kv_cache_dtype = "auto"
    self.quant_config = quant_config

    if cache_config is not None and cache_config.kv_cache_dtype_skip_layers:
        from vllm.model_executor.models.utils import extract_layer_index

        layer_idx = extract_layer_index(prefix)
        if str(layer_idx) in cache_config.kv_cache_dtype_skip_layers:
            kv_cache_dtype = "auto"
        logger.debug(
            "Layer %s: kv_cache_dtype=%s",
            prefix,
            kv_cache_dtype,
        )

    dtype = torch.get_default_dtype()
    if attn_backend is not None:
        assert attn_backend.is_mla(), (
            f"MLAAttention: attn_backend must be an MLA backend, "
            f"got {attn_backend.get_name()} instead"
        )
        self.attn_backend = attn_backend
    else:
        self.attn_backend = get_attn_backend(
            self.head_size,
            dtype,
            kv_cache_dtype,
            use_mla=True,
            use_sparse=use_sparse,
            num_heads=self.num_heads,
        )

    normalized_kv_cache_dtype = _canonicalize_sparse_mla_kv_cache_dtype(
        self.attn_backend, kv_cache_dtype
    )
    if normalized_kv_cache_dtype != kv_cache_dtype:
        if cache_config is not None:
            cache_config.cache_dtype = normalized_kv_cache_dtype
        kv_cache_dtype = normalized_kv_cache_dtype
        logger.info_once(
            "Using %s KV cache format for %s backend.",
            kv_cache_dtype,
            self.attn_backend.get_name(),
        )

    if (
        self.attn_backend.get_name() == "FLASHINFER_MLA_SPARSE"
        and kv_cache_dtype != "fp8_ds_mla"
        and is_quantized_kv_cache(kv_cache_dtype)
    ):
        logger.info_once(
            "Using standard fp8 KV cache format. To use DeepSeek's fp8_ds_mla "
            "KV cache format, please set `--attention-backend FLASHMLA_SPARSE`"
        )

    # Initialize KV cache quantization attributes
    self.kv_cache_dtype = kv_cache_dtype
    _init_kv_cache_quant(self, quant_config, prefix)

    if (
        cache_config is not None
        and cache_config.enable_prefix_caching
        and envs.VLLM_BATCH_INVARIANT
        and (
            self.attn_backend.get_name() == "TRITON_MLA"
            or self.attn_backend.get_name() == "FLASHINFER"
        )
    ):
        logger.warning_once(
            "Disabling prefix caching for TRITON_MLA / FLASHINFER "
            "with batch invariance, as it is not yet supported.",
        )
        cache_config.enable_prefix_caching = False

    # Sparse MLA reads top-k indices from a shared buffer. Pass it
    # explicitly so backbone "skip" layers (indexer=None) still find it.
    if use_sparse:
        extra_impl_args["topk_indices_buffer"] = topk_indices_buffer
        if index_group_builder is not None:
            extra_impl_args["index_group_builder"] = index_group_builder

    impl_cls = cast(type[MLAAttentionImpl], self.attn_backend.get_impl_cls())
    impl = impl_cls(
        num_heads=self.num_heads,
        head_size=self.head_size,
        scale=self.scale,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=sliding_window,
        kv_cache_dtype=self.kv_cache_dtype,
        logits_soft_cap=None,
        attn_type=AttentionType.DECODER,
        # GSA: the SFA impl derives layer_id / per-layer GSA state from the
        # layer prefix, so pass it instead of the 0.32.0 default None.
        kv_sharing_target_layer_name=prefix,
        # MLA Args
        q_lora_rank=self.q_lora_rank,
        kv_lora_rank=self.kv_lora_rank,
        qk_nope_head_dim=self.qk_nope_head_dim,
        qk_rope_head_dim=self.qk_rope_head_dim,
        qk_head_dim=self.qk_nope_head_dim + self.qk_rope_head_dim,
        v_head_dim=self.v_head_dim,
        kv_b_proj=kv_b_proj,
        indexer=indexer,
        **extra_impl_args,
    )
    self.impl = impl  # type: ignore[assignment]
    index_group = getattr(impl, "index_group", None)
    self.hisparse_cache = (
        index_group.cache(cast(Any, impl).index_group_index)
        if isinstance(index_group, HiSparseMLAIndexGroup)
        else None
    )
    self.q_pad_num_heads = getattr(self.impl, "q_pad_num_heads", None)
    self.is_amx_bmm_enabled = getattr(self.impl, "uses_amx_bmm", False)
    # AMX reads kv_b_proj's weight directly and never calls it live; the
    # reference CPU MLA backend calls it but isn't perf-critical. Skip
    # the packed-kernel dispatch either way.
    kv_b_proj._cpu_skip_gemm_dispatch = True
    self.use_direct_call = not current_platform.opaque_attention_op()

    vllm_config = get_current_vllm_config()
    parallel_config = vllm_config.parallel_config
    self.use_pcp = parallel_config.prefill_context_parallel_size > 1
    self.pcp_shard_decode_requests = parallel_config.pcp_shard_decode_requests
    compilation_config = vllm_config.compilation_config
    if prefix in compilation_config.static_forward_context:
        raise ValueError(f"Duplicate layer name: {prefix}")
    compilation_config.static_forward_context[prefix] = self

    self.prefill_backend: MLAPrefillBackend | None
    if self.impl.is_sparse and not (
        self.impl.supports_dense_mha_prefill and self.supports_dense_mha_prefill
    ):
        logger.warning_once(
            "Sparse MLA layer has no dense-MHA prefill path; using the top-k "
            "MQA path only."
        )
        self.prefill_backend = None
    else:
        try:
            prefill_backend_cls = prefill_backend_cls or get_mla_prefill_backend(
                vllm_config
            )
        except ValueError:
            if (
                not self.impl.is_sparse
                or vllm_config.attention_config.mla_prefill_backend is not None
            ):
                raise
            logger.warning_once(
                "No MLA prefill backend supports this model; sparse MLA will "
                "use the top-k MQA path only (no dense-MHA prefill)."
            )
            self.prefill_backend = None
        else:
            self.prefill_backend = prefill_backend_cls(
                num_heads=self.num_heads,
                scale=self.scale,
                kv_lora_rank=self.kv_lora_rank,
                qk_nope_head_dim=self.qk_nope_head_dim,
                qk_rope_head_dim=self.qk_rope_head_dim,
                v_head_dim=self.v_head_dim,
                vllm_config=vllm_config,
            )

    self.kv_cache = torch.tensor([])

    self.use_sparse = use_sparse

    if vllm_config.kernel_config.enable_jit_warmup:
        backend_name = self.attn_backend.get_name()
        if backend_name in (
            "FLASHMLA_SPARSE",
            "FLASHINFER_MLA_SPARSE",
            "FLASHINFER_MLA_SPARSE_SM120",
            "DEEPSEEK_V32_INDEXER",
        ):
            from vllm.v1.attention.backends.mla.compressor_utils import (
                _COMPRESSED_SLOT_MAPPING_KERNEL,
            )
            from vllm.v1.attention.backends.mla.indexer import (
                _BUILD_PREFILL_CHUNK_METADATA_KERNEL,
                _PREPARE_UNIFORM_DECODE_KERNEL,
            )

            _COMPRESSED_SLOT_MAPPING_KERNEL.register_warmup()
            _PREPARE_UNIFORM_DECODE_KERNEL.register_warmup()
            _BUILD_PREFILL_CHUNK_METADATA_KERNEL.register_warmup()

            if backend_name != "DEEPSEEK_V32_INDEXER":
                from vllm.v1.attention.backends.mla.sparse_swa import (
                    _COMPUTE_PREFILL_METADATA_KERNEL,
                )

                _COMPUTE_PREFILL_METADATA_KERNEL.register_warmup()

    if self.use_pcp and self.impl.dcp_world_size > 1 and not self.supports_pcp_dcp:
        raise NotImplementedError(
            f"{type(self).__name__} does not support PCP+DCP."
        )
    self.dcp_manager: MLADCPManager | None = None
    if self.impl.dcp_world_size > 1:
        query_dtype = (
            current_platform.fp8_dtype()
            if is_quantized_kv_cache(self.kv_cache_dtype)
            and self.kv_cache_dtype != "fp8_ds_mla"
            and self.impl.supports_quant_query_input
            else dtype
        )
        self.dcp_manager = MLADCPManager(
            vllm_config=vllm_config,
            device=next(kv_b_proj.parameters()).device,
            num_heads=self.num_heads,
            query_head_dim=self.kv_lora_rank + self.qk_rope_head_dim,
            output_head_dim=self.kv_lora_rank,
            query_dtype=query_dtype,
            output_dtype=dtype,
            padded_num_heads=self.q_pad_num_heads,
            is_lse_base_on_e=self.impl.lse_base_on_e,
            use_pcp=self.use_pcp,
        )

    self.is_aiter_triton_fp8_bmm_enabled = rocm_aiter_ops.is_fp8bmm_enabled()

    # If kv_b_proj_weight is unquantized, quantize it to mxfp4 if supported
    self.is_aiter_triton_fp4_bmm_enabled = (
        rocm_aiter_ops.is_fp4bmm_enabled()
        and hasattr(self.kv_b_proj, "weight")
        and self.kv_b_proj.weight.dtype == torch.bfloat16
    )

    # Attributes for forward_impl method
    self._vllm_config = get_current_vllm_config()
    self._chunked_prefill_workspace_size: int | None = None
    self._decode_concat_quant_fp8_op = _DecodeConcatQuantFP8(
        static=True,
        group_shape=GroupShape.PER_TENSOR,
        compile_native=True,
    )
    self._quant_fp8_op = QuantFP8(
        static=True,
        group_shape=GroupShape.PER_TENSOR,
        compile_native=True,
    )


@when_imported("vllm.model_executor.layers.attention.mla_attention")
def _patch(mod):
    if hasattr(mod.MLAAttention, "__init__"):
        _patched = FunctionType(
            __init__.__code__,
            vars(mod),
            name=__init__.__name__,
            argdefs=__init__.__defaults__,
            closure=__init__.__closure__,
        )
        _patched.__annotations__ = __init__.__annotations__
        _patched.__kwdefaults__ = __init__.__kwdefaults__
        patch_or_inject(mod.MLAAttention, "__init__", _patched)
        logger.info("UCM GSA MLAAttention.__init__ patch applied")
