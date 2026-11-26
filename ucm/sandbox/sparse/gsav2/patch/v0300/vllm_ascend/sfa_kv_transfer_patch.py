"""Replace the vLLM-Ascend 0.30.0 SFA forward with the GSA-aware version.

Rebased from the 0.23.0 GSA patch (``v0230/vllm_ascend/sfa_kv_transfer_patch.py``)
onto the 0.30.0 ``AscendSFAImpl.forward``, then aligned with the ucm-v0300
in-tree GSA adaptation (dsa caches on the indexer backend, unified indexer
call). The 0.30.0 forward was restructured (prolog-v3/mlapo preprocess
helpers, ``SFAForwardContext`` parallel plumbing, unified
``AscendIndexer.forward`` k path + top-k call, ``g_proj`` gating,
``attention_transfer_window``, sequence-parallel o_proj finalize); this
patch keeps the upstream body and injects the GSA semantics from the 0.23.0
version:

* ``maybe_execute_sparse_attention_begin/finished`` hooks around the layer
  (pdmix / decode workers, ``layer_id < layer_num``, prefill steps only --
  gated by ``PTOPK_PREFETCH_ENABLE`` and the runner-attached
  ``has_prefill``).
* One unified indexer call for sparse layers: the patched backend forward
  (``indexer_patch.py``) computes the k path, persists it into the
  per-layer ``dsa_k_cache`` (raw dsa slot mapping) and runs the selection
  whose device-op dispatch routes to the GSA DSA caches via
  ``sfa_impl=self``. No inline k-path recompute and no impl-level
  ``indexer_select_post_process`` -- both were 0.23.0-era workarounds for
  the indexer not owning the caches.
* The prefill/decode ``gsa_prefetch/warmup/swap`` top-k bookkeeping and the
  swap sub-batch attention call (``swap_len`` = ``select_block_index`` slice,
  v0230 8th-positional convention).

Names resolved through ``vars(mod)``: everything the 0.30.0 module already
imports. UCM names are imported function-locally (and belt-and-braces injected
into the module namespace by ``sfa_v1_patch._patch``).
"""

from __future__ import annotations

from types import FunctionType

from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


def forward(
    self,
    layer_name,
    hidden_states: torch.Tensor,  # query in unified attn
    kv_cache: tuple[torch.Tensor, ...],
    attn_metadata: M,
    output: torch.Tensor | None = None,
    input_is_sequence_parallel: bool = False,
    output_scale: float = 1.0,
) -> torch.Tensor:
    from ucm.sparse.state import (
        has_ucm_sparse,
        maybe_execute_sparse_attention_begin,
        maybe_execute_sparse_attention_finished,
    )
    from ucm.sparse.utils import DEFAULT_TOPK_LEN, PTOPK_PREFETCH_ENABLE, SPA_TOKENS_LEN

    assert output is not None, "Output tensor must be provided."
    # GSA (v0230): pdmix runs the sparse path on producer+consumer workers;
    # decode-only workers run it whenever they are not the PD producer.
    is_pdmix = self.vllm_config.kv_transfer_config is None or (
        self.vllm_config.kv_transfer_config.is_kv_producer
        and self.vllm_config.kv_transfer_config.is_kv_consumer
    )
    gsa_active = (
        (is_pdmix or not self.vllm_config.kv_transfer_config.is_kv_producer)
        and has_ucm_sparse()
        and PTOPK_PREFETCH_ENABLE
        and self.layer_id < self.layer_num
    )
    if self.enable_sparse_sfa_turboquant:
        assert self.turboquant is not None
        # Build the transform tensors outside graph capture.
        self.turboquant._initialize(hidden_states.device)
    if attn_metadata is None:
        return self._forward_dummy(hidden_states, output)

    if self.qk_rope_head_dim == 0 and not input_is_sequence_parallel:
        num_tokens = min(hidden_states.shape[0], attn_metadata.slot_mapping.shape[0])
        if get_forward_context().cudagraph_runtime_mode != CUDAGraphMode.FULL:
            num_tokens = min(num_tokens, attn_metadata.num_actual_tokens)
        if num_tokens == 0:
            return self._forward_dummy(hidden_states, output)
        hidden_states = hidden_states[:num_tokens]
    gate_hidden_states = hidden_states if self.g_proj is not None else None

    composed_kv_cache = self._compose_sfa_kv_cache(kv_cache)
    assert composed_kv_cache is not None
    kv_cache = composed_kv_cache

    cos = attn_metadata.cos
    sin = attn_metadata.sin
    slot_mapping_sfa = self._get_sfa_kv_slot_mapping(attn_metadata)
    indexer_attn_metadata = self._get_indexer_attn_metadata()

    # Inputs and outputs may be padded for CUDA graphs
    num_input_tokens = (
        hidden_states.shape[0]
        if self.qk_rope_head_dim == 0 and not input_is_sequence_parallel
        else attn_metadata.num_input_tokens
    )
    parallel_context = self._get_parallel_forward_context(
        attn_metadata,
        num_input_tokens,
        hidden_states,
    )
    actual_seq_lengths_query = parallel_context.actual_seq_lengths_query
    actual_seq_lengths_key = parallel_context.actual_seq_lengths_key

    fused_type: PreprocessType = self.preprocess_type
    # PROLOG_V3 serves every attention state (decode, spec decoding and
    # prefill); only MLAPO carries a per-call token-count limit.
    if self.preprocess_type == PreprocessType.MLAPO and num_input_tokens > MLAPO_MAX_SUPPORTED_TOKENS:
        fused_type = PreprocessType.NATIVE

    # IMPORTANT!
    # NOTE: With PIECEWISE ACL graphs, this method is executed in
    # eager-mode PyTorch. Even view and slice operations can add CPU
    # overhead without launching NPU kernels. Minimize PyTorch operations
    # in this method and benchmark changes to avoid regressions.
    # Inputs and outputs may contain DP or graph padding. Keep the
    # preallocated output for the caller while running attention only on
    # the token rows described by metadata, matching upstream backends.
    hidden_states = hidden_states[:num_input_tokens]
    if gate_hidden_states is not None:
        gate_hidden_states = gate_hidden_states[:num_input_tokens]

    if fused_type == PreprocessType.NATIVE:
        # Trim in the input layout before CP adds padding and shards it.
        if input_is_sequence_parallel:
            hidden_states = self._prepare_sequence_parallel_hidden_states(hidden_states, attn_metadata)
        else:
            hidden_states = self._prepare_native_hidden_states(hidden_states, attn_metadata)
        if gate_hidden_states is not None:
            gate_hidden_states = hidden_states

    if fused_type != PreprocessType.NATIVE:
        # Keep the raw hidden states for the indexer's k path: the fused
        # preprocess below returns new tensors and does not modify this
        # one in place.
        k_hidden_states = hidden_states if self.runtime_has_indexer else None
        wait_for_kv_layer_from_connector(layer_name)
        if self.layerwise_kv_cache_hook is not None:
            # The fused preprocess is the first operation that may read or
            # update the paged SFA and LI caches.
            self.layerwise_kv_cache_hook.wait_for_layer(layer_name)

        if fused_type == PreprocessType.PROLOG_V3:
            hidden_states, ql_nope, q_pe, q_c = self._sfa_preprocess_prolog_v3(
                hidden_states=hidden_states,
                kv_cache=kv_cache,
                cos=cos,
                sin=sin,
                # npu_mla_prolog_v3 requires int64 cache indices; reuse
                # the per-step conversion so all layers of a step share
                # one Cast kernel (the .to() inside is a no-op on int64).
                slot_mapping=_int64_kv_slots(slot_mapping_sfa, attn_metadata),
                attn_metadata=attn_metadata,
            )
        else:
            hidden_states, ql_nope, q_pe, q_c = self._sfa_preprocess_mlapo(
                hidden_states=hidden_states,
                kv_cache=kv_cache,
                cos=cos,
                sin=sin,
                slot_mapping=slot_mapping_sfa,
                num_input_tokens=num_input_tokens,
                attn_metadata=attn_metadata,
            )
    # native
    else:
        assert self.fused_qkv_a_proj is not None, "q lora is required for DSA."
        qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
        q_c, kv_no_split = qkv_lora.split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            dim=-1,
        )
        assert self.q_a_layernorm is not None, "q_a_layernorm must be initialized"
        q_c = self.q_a_layernorm(q_c)

        # The prepared hidden states feed the indexer's k path (same stage
        # as the weights path input).
        k_hidden_states = hidden_states if self.runtime_has_indexer else None

        wait_for_kv_layer_from_connector(layer_name)
        if self.layerwise_kv_cache_hook is not None:
            # Q/KV projections above overlap the full-layer broadcast.
            # Wait before the first main or indexer cache access.
            self.layerwise_kv_cache_hook.wait_for_layer(layer_name)

        kv_outputs = self.exec_kv(
            kv_no_split,
            cos,
            sin,
            kv_cache,
            parallel_context.kv_slot_mapping,
            attn_metadata,
        )
        k_pe, k_nope = kv_outputs[:2]
        knope_scale = kv_outputs[2] if len(kv_outputs) == 3 else None
        # k_li no longer exists at this point: it is computed by
        # indexer.forward_k below and gathered at cache-write time, so
        # the fused gather below only carries the main KV. (Non-GSA; the
        # GSA branch below recomputes the k path for the DSA cache.)
        fused_kv_no_split, kv_ag_handles = self._prepare_kv_for_parallel(
            k_pe,
            k_nope,
            knope_scale,
            parallel_context.gather_full_o_proj,
        )

        ql_nope, q_pe = self._q_proj_and_k_up_proj(q_c)
        if self.qk_rope_head_dim:
            q_pe = self.rope_single(q_pe, cos, sin)
        self._record_query_gather_context(
            ql_nope,
            q_pe,
            attn_metadata,
        )

        (
            k_pe,
            k_nope,
        ) = self._store_parallel_kv(
            k_pe,
            k_nope,
            knope_scale,
            fused_kv_no_split,
            kv_ag_handles,
            kv_cache,
            self._get_sfa_kv_slot_mapping(attn_metadata),
            attn_metadata,
            parallel_context.gather_full_o_proj,
        )

    # A sharded-decode PCP rank can own no request but must still write
    # and gather KV with the other ranks.
    has_local_query = attn_metadata.num_actual_tokens > 0
    # GSA: has_prefill is attached by the patched
    # NPUModelRunner._build_attention_metadata using the 0.23.0 builder
    # formula (any query longer than decode_threshold).
    has_prefill = attn_metadata.has_prefill
    if gsa_active and self.runtime_has_indexer:
        # ============ GSA top-k section (v0230 semantics) ============
        # (1) Begin hook: launch the layer's GSA sparse-attention
        # orchestration before this layer's k write (prefill steps only,
        # matching dev). The composed cache tuple places the indexer
        # k/scale caches at (1, 2) with a packed main cache (C8/TQ4) or
        # (2, 3) with split k/v -- the same derivation the 0.23.0 forward
        # used. ``self.dsa_k_cache`` aliases the indexer backend's
        # per-layer cache (sfa_v1_patch._make_gsa_aware_init).
        if has_prefill:
            if self.uses_packed_sfa_main_cache:
                dsa_k_cache_idx = 1
                dsa_k_scale_cache_idx = 2
            else:
                dsa_k_cache_idx = 2
                dsa_k_scale_cache_idx = 3
            k_cache = kv_cache[0]
            v_cache = kv_cache[0] if self.uses_packed_sfa_main_cache else kv_cache[1]
            vllm_indexer_cache = kv_cache[dsa_k_cache_idx] if self.has_indexer else None
            dsa_k_scale_cache = self.dsa_k_scale if self.enable_sparse_li_c8 else None
            vllm_indexer_scale_cache = (
                kv_cache[dsa_k_scale_cache_idx] if self.enable_sparse_li_c8 else None
            )
            _, _, _, _ = maybe_execute_sparse_attention_begin(
                None,
                k_cache,
                v_cache,
                layer_name,
                get_forward_context(),
                indexer_cache=self.dsa_k_cache,
                indexer_scale=dsa_k_scale_cache,
                vllm_indexer_cache=vllm_indexer_cache,
                vllm_indexer_scale=vllm_indexer_scale_cache,
                has_indexer=self.has_indexer,
                use_li_c8=self.enable_sparse_li_c8,
            )

        # (2) One unified indexer call: k path + DSA cache write + top-k.
        # The patched backend forward (indexer_patch.py) computes the k
        # path, persists it into the per-layer dsa_k_cache (raw dsa slot
        # mapping) and runs the selection whose device-op dispatch routes
        # to the DSA caches via ``sfa_impl`` (device_op_patch.py), so no
        # inline k-path recompute is needed here. The GSA routing fields
        # are copied onto the indexer metadata first
        # (sfa_v1_patch._prepare_indexer_metadata).
        assert k_hidden_states is not None
        assert indexer_attn_metadata is not None
        self._prepare_indexer_metadata(indexer_attn_metadata, attn_metadata)
        topk_indices = self.indexer(
            hidden_states,
            q_c,
            k_hidden_states,
            indexer_attn_metadata,
            compute_topk=not self.skip_topk and has_local_query,
            attn_q_gather_handle=self._get_indexer_attn_q_gather_handle(attn_metadata),
        )

        # (3) skip_topk layers reuse the shared index cache (their
        # dsa_k_cache stays fresh through the k path + write above).
        if self.skip_topk:
            topk_indices = self._get_indexcache_topk_indices(parallel_context.topk_num_tokens)
        elif self.use_index_cache and has_local_query:
            self._update_indexcache_topk_indices(topk_indices)
    elif self.runtime_has_indexer:
        # One unified indexer call: k path -> cache write -> top-k
        # selection (the selection kernel reads the freshly written
        # cache). MTP skip_topk layers still run the k path and the write
        # (compute_topk=False) so their cache stays up to date, then
        # reuse the shared top-k indices. Cache layout, RoPE, parallel
        # sequence lengths, and decode count all come from the indexer's
        # independently built metadata.
        assert k_hidden_states is not None
        assert indexer_attn_metadata is not None
        self._prepare_indexer_metadata(indexer_attn_metadata, attn_metadata)
        topk_indices = self.indexer(
            hidden_states,
            q_c,
            k_hidden_states,
            indexer_attn_metadata,
            compute_topk=not self.skip_topk and has_local_query,
            attn_q_gather_handle=self._get_indexer_attn_q_gather_handle(attn_metadata),
        )
        if self.skip_topk:
            topk_indices = self._get_indexcache_topk_indices(parallel_context.topk_num_tokens)
        elif self.use_index_cache and has_local_query:
            self._update_indexcache_topk_indices(topk_indices)
    elif self.skip_topk:
        # Static shared-index layers keep no runtime indexer cache and
        # only reuse the shared top-k indices.
        topk_indices = self._get_indexcache_topk_indices(parallel_context.topk_num_tokens)
    else:
        raise RuntimeError(f"skip_topk is False but indexer is None. layer_name={self.layer_name}.")

    # Notify for every layer that wrote the cache, not just indexer layers:
    # by this point all of the layer's KV (main + indexer) has been
    # scattered - indexer layers persisted it inside indexer.forward
    # above - so the connector can dispatch the PD pull.
    notify_kv_cache_written(self.layer_name or "")

    # Open the prefetch gate for every SFA layer. Some GLM-5.2 layers
    # reuse cached top-k indices and have no indexer, so recording this
    # inside the indexer's forward would leave their gate closed.
    with attention_transfer_window():
        if has_local_query:
            if gsa_active:
                # ============ GSA attention section (v0230 semantics) ===
                # has_prefill was read above the top-k section.
                if has_prefill:
                    # Prefill: copy the fresh top-k rows into the
                    # prefetch / warmup slots, patch the swapped rows back
                    # in, run the main batch, then recompute the swap
                    # sub-batch with its corrected lengths and overwrite.
                    if attn_metadata.gsa_prefetch_len > 0 and self.has_indexer:
                        attn_metadata.gsa_prefetch_topk[self.cur_topk_index][
                            : attn_metadata.gsa_prefetch_len
                        ].copy_(
                            topk_indices[
                                attn_metadata.gsa_prefetch_index[
                                    : attn_metadata.gsa_prefetch_len
                                ]
                            ]
                        )
                    if attn_metadata.gsa_warmup_len > 0 and self.has_indexer:
                        attn_metadata.gsa_warmup_topk[self.cur_topk_index][
                            : attn_metadata.gsa_warmup_len
                        ].copy_(
                            topk_indices[
                                attn_metadata.gsa_warmup_index[
                                    : attn_metadata.gsa_warmup_len
                                ]
                            ]
                        )
                    if attn_metadata.gsa_swap_len != 0:
                        topk_indices.index_copy_(
                            0,
                            attn_metadata.gsa_swap_index[: attn_metadata.gsa_swap_len],
                            attn_metadata.gsa_swap_topk[self.cur_topk_index][
                                : attn_metadata.gsa_swap_len, :, :DEFAULT_TOPK_LEN
                            ],
                        )

                    attn_output = self._execute_sparse_flash_attention_process(
                        ql_nope,
                        q_pe,
                        kv_cache,
                        topk_indices,
                        attn_metadata,
                        actual_seq_lengths_query,
                        actual_seq_lengths_key,
                    )

                    if attn_metadata.gsa_swap_len != 0:
                        # swap sub-batch: v0230 8th-positional convention,
                        # swap_len = the swap requests' select_block_index
                        # slice (block-table row selector), with the swap
                        # requests' prefill query rows and key lengths.
                        attn_output_gsa = self._execute_sparse_flash_attention_process(
                            ql_nope[
                                attn_metadata.gsa_swap_index[: attn_metadata.gsa_swap_len]
                            ],
                            q_pe[
                                attn_metadata.gsa_swap_index[: attn_metadata.gsa_swap_len]
                            ],
                            kv_cache,
                            attn_metadata.gsa_swap_topk[self.cur_topk_index][
                                : attn_metadata.gsa_swap_len
                            ],
                            attn_metadata,
                            attn_metadata.gsa_prefill_query[
                                : attn_metadata.gsa_swap_len // self.num_spec_tokens
                            ],
                            actual_seq_lengths_key[
                                attn_metadata.select_block_index[
                                    : attn_metadata.gsa_swap_len // self.num_spec_tokens
                                ]
                            ],
                            attn_metadata.select_block_index[
                                : attn_metadata.gsa_swap_len // self.num_spec_tokens
                            ],
                        )
                        attn_output.index_copy_(
                            0,
                            attn_metadata.gsa_swap_index[: attn_metadata.gsa_swap_len],
                            attn_output_gsa,
                        )
                else:
                    # Decode: gather-copy the fresh top-k rows into the
                    # warmup / prefetch slots, then expand the swap rows to
                    # spec-token width and run the whole batch.
                    if attn_metadata.gsa_warmup_len > 0 and self.has_indexer:
                        torch.ops._C_gsa.gsa_gather_copy(
                            attn_metadata.gsa_warmup_topk[self.cur_topk_index][
                                : attn_metadata.gsa_warmup_len
                            ].view(-1, DEFAULT_TOPK_LEN),
                            topk_indices.view(-1, DEFAULT_TOPK_LEN),
                            attn_metadata.gsa_warmup_index[
                                : attn_metadata.gsa_warmup_len
                            ]
                            .view(-1, 1)
                            .expand((-1, DEFAULT_TOPK_LEN)),
                        )
                    if attn_metadata.gsa_prefetch_len != 0 and self.has_indexer:
                        torch.ops._C_gsa.gsa_gather_copy(
                            attn_metadata.gsa_prefetch_topk[self.cur_topk_index][
                                : attn_metadata.gsa_prefetch_len
                            ].view(-1, DEFAULT_TOPK_LEN),
                            topk_indices.view(-1, DEFAULT_TOPK_LEN),
                            attn_metadata.gsa_prefetch_index[
                                : attn_metadata.gsa_prefetch_len
                            ]
                            .view(-1, 1)
                            .expand((-1, DEFAULT_TOPK_LEN)),
                        )
                    topk_indices_out = torch.ops._C_gsa.gsa_topk_expand_tmp(
                        topk_indices.view(-1, DEFAULT_TOPK_LEN),
                        attn_metadata.gsa_swap_topk[self.cur_topk_index][
                            : attn_metadata.gsa_swap_len
                        ].view(-1, SPA_TOKENS_LEN),
                        attn_metadata.gsa_swap_index[: attn_metadata.gsa_swap_len]
                        .view(-1, 1)
                        .expand((-1, SPA_TOKENS_LEN)),
                        attn_metadata.gsa_topk_win_tmp,
                        [0, SPA_TOKENS_LEN - DEFAULT_TOPK_LEN, 0, 0],
                        -1,
                        SPA_TOKENS_LEN,
                    )
                    attn_output = self._execute_sparse_flash_attention_process(
                        ql_nope,
                        q_pe,
                        kv_cache,
                        topk_indices_out,
                        attn_metadata,
                        actual_seq_lengths_query,
                        actual_seq_lengths_key,
                    )
            else:
                attn_output = self._execute_sparse_flash_attention_process(
                    ql_nope,
                    q_pe,
                    kv_cache,
                    topk_indices,
                    attn_metadata,
                    actual_seq_lengths_query,
                    actual_seq_lengths_key,
                )

    if gsa_active and has_prefill and has_local_query:
        # ============ GSA finished hook (v0230 semantics) ============
        # Prefill steps that produced attention output only, matching the
        # in-tree ucm-v0300 gate.
        k_cache = kv_cache[0]
        v_cache = kv_cache[0] if self.uses_packed_sfa_main_cache else kv_cache[1]
        maybe_execute_sparse_attention_finished(
            ql_nope,
            k_cache,
            v_cache,
            attn_output,
            layer_name,
            get_forward_context(),
            has_indexer=self.has_indexer,
            use_li_c8=self.enable_sparse_li_c8,
        )

    if has_local_query:
        attn_output = self._v_up_proj(attn_output)
    else:
        # The O-proj below still joins its PCP/TP collectives.
        attn_output = hidden_states.new_zeros((ql_nope.shape[0], self.local_num_heads * self.v_head_dim))
    if gate_hidden_states is not None:
        assert self.g_proj is not None
        attn_output.mul_(torch.sigmoid(self.g_proj(gate_hidden_states.contiguous())[0]))
    if input_is_sequence_parallel:
        self._finalize_sequence_parallel_o_proj(
            attn_output,
            output,
            parallel_context.gather_full_o_proj,
            attn_metadata,
            output_scale,
        )
    else:
        self._finalize_o_proj(
            attn_output,
            output[:num_input_tokens],
            parallel_context.gather_full_o_proj,
        )

    maybe_save_kv_layer_to_connector(layer_name, list(kv_cache))

    return output


@when_imported("vllm_ascend.attention.sfa_v1")
def patch_sfa_kv_transfer(mod):
    # Run the merged GSA forward with the original sfa_v1 module globals.
    patched_forward = FunctionType(
        forward.__code__,
        vars(mod),
        name=forward.__name__,
        argdefs=forward.__defaults__,
        closure=forward.__closure__,
    )
    patched_forward.__annotations__ = forward.__annotations__
    patched_forward.__kwdefaults__ = forward.__kwdefaults__

    patch_or_inject(mod.AscendSFAImpl, "forward", patched_forward)
    logger.info("UCM Ascend SFA GSA KV-transfer forward patch applied")
