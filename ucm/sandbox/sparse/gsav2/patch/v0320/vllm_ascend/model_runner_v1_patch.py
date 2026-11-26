"""GSA NPUModelRunner patches for vLLM-Ascend 0.32.0.

Rebased from the 0.23.0 GSA patch layer (via the 0.30.0 port) onto the
vLLM-Ascend 0.32.0 ``vllm_ascend.worker.v1.model_runner.NPUModelRunner``
(the module moved from ``vllm_ascend.worker.model_runner_v1`` in 0.32.0;
the old file survives only as a byte-identical, unreferenced duplicate).
Upstream 0.32.0 changes accounted for here:

* ``__init__`` / ``_set_up_drafter`` / ``_build_attn_state`` are NOT
  re-patched: their GSA buffers (``sparse_positions`` / ``sparse_query_pos`` /
  the ``[GSA]`` metadata block) live in the patched ``GPUModelRunner.__init__``,
  which ``NPUModelRunner.__init__`` invokes via ``super().__init__``.
* ``_prepare_inputs`` grew the DSV41 prefix-replay plumbing
  (``_fill_replay_start`` / ``block_table.pad_replayed_slots``); the GSA
  sparsed-position rewrite keeps its slot-mapping domain (``sparse_positions``
  feeds ``compute_slot_mapping``; the upstream replay padding still consumes
  the absolute ``positions`` -- replay batches are dense, sparse requests
  never replay).
* ``execute_model`` dropped ``self._restore_c8_mxfp_loaded_v_scales``; the
  three GSA hooks (begin / warmup / finished) keep their positions around
  ``_model_forward``.
* ``sample_tokens`` no longer builds ``output_spec_token_ids`` /
  ``spec_token_ids`` for ``ModelRunnerOutput``; the GSA spec
  ``execute_finished(rec_result=...)`` hook stays right after the
  draft-token/finalize block.
* ``_build_attention_metadata`` dropped the ``AscendDSACPMetadataBuilder``
  special-casing, grew ``replay_start`` on the common metadata and
  ``_decoder_replay_common_metadata``; the GSA metadata attach block, the
  cudagraph-capture GSA fields and the ``is_pdmix`` routing stay in
  ``_build_attn_group_metadata``.

The four ``maybe_execute_ucm_sparse_*`` helpers are injected verbatim from the
0.23.0 patch layer (they only talk to ``ucm.sparse.state``).
"""

from __future__ import annotations

from types import FunctionType

from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


def _prepare_inputs(
    self,
    scheduler_output: "SchedulerOutput",
    num_scheduled_tokens: np.ndarray,
) -> tuple[
    torch.Tensor,
    SpecDecodeMetadata | None,
    int,
]:
    """
    :return: tuple[
        logits_indices,
        spec_decode_metadata,
        total_num_scheduled_tokens,
    ]
    """
    from ucm.sparse.base import INVALID_SLOT
    from ucm.sparse.state import has_ucm_sparse

    total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
    assert total_num_scheduled_tokens > 0
    num_reqs = self.input_batch.num_reqs
    assert num_reqs > 0

    # OPTIMIZATION: Start copying the block table first.
    # This way, we can overlap the copy with the following CPU operations.
    self.input_batch.block_table.commit_block_table(num_reqs)

    req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)

    # Get the attention state.
    if not scheduler_output.scheduled_spec_decode_tokens:
        num_valid_tokens = num_scheduled_tokens
    else:
        num_valid_tokens = np.array(
            [
                scheduler_output.num_scheduled_tokens[i]
                - len(scheduler_output.scheduled_spec_decode_tokens.get(i, []))
                for i in self.input_batch.req_ids
            ],
            dtype=np.int32,
        )
    attn_state = self._build_attn_state(num_reqs, num_scheduled_tokens, num_valid_tokens)

    # Determine if it's a splitfuse batch
    with_prefill = attn_state not in [AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding]
    self.with_prefill = with_prefill

    # Get positions.
    cu_num_tokens = self._get_cumsum_and_arange(
        num_scheduled_tokens, self.query_pos.np
    )
    positions_np = self._positions_np_buf[:total_num_scheduled_tokens]
    np.add(
        self.input_batch.num_computed_tokens_cpu[req_indices],
        self.query_pos.np[: cu_num_tokens[-1]],
        out=positions_np,
    )

    # GSA: per-request sparsed positions. For sparse (top-k) requests the
    # position domain is the sparse slot window (query-relative) instead of
    # the absolute token position; the slot-mapping kernel consumes it.
    sparsed_positions = positions_np.copy()
    req_sparsed_slots = scheduler_output.req_sparsed_slots
    for req_id in self.input_batch.req_id_to_index:
        is_sparsed_request = req_sparsed_slots[req_id] != INVALID_SLOT
        req_index = self.input_batch.req_id_to_index[req_id]
        offset = (
            0 if req_index == 0 else cu_num_tokens[req_index - 1]
        )  # TODO: support MTP
        end = cu_num_tokens[req_index]
        if is_sparsed_request:
            # sparsed_positions[offset] = req_sparsed_slots[req_id] - 1
            sparsed_positions[offset:end] = list(
                range(
                    req_sparsed_slots[req_id] - end + offset, req_sparsed_slots[req_id]
                )
            )

    if self.use_dcp:
        self.dcp_manager.init_batch_info(
            num_scheduled_tokens,
            self.input_batch.num_reqs,
            self.input_batch.num_computed_tokens_cpu,
            self.input_batch.num_prompt_tokens,
        )

    # Build previous positions before DCP prepares speculative inputs.
    prev_req_id_to_index = self.input_batch.prev_req_id_to_index
    self._compute_prev_positions(num_reqs)
    prev_positions_gpu = None
    if (
        self.use_async_scheduling
        and self.input_batch.prev_sampled_token_ids is not None
        and prev_req_id_to_index
    ):
        self.prev_positions.copy_to_gpu(num_reqs)
        prev_positions_gpu = self.prev_positions.gpu[:num_reqs]

    # for dcp, prefill mtp should use origin scheduleroutput ,
    # GSA: the DCP MTP input generation consumes the sparsed positions.
    if self.speculative_config and self.use_dcp:
        self.dcp_manager.generate_dcp_mtp_input(
            total_num_scheduled_tokens,
            scheduler_output.num_scheduled_tokens,
            with_prefill,
            self.input_batch,
            self.arange_np,
            req_indices,
            sparsed_positions,
            cu_num_tokens,
            self._draft_token_ids,  # type: ignore[has-type]
            scheduler_output,
            self.num_spec_tokens,
            prev_positions=prev_positions_gpu,
        )

    self.query_lens = torch.from_numpy(num_scheduled_tokens)

    # Get token indices.
    # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
    # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
    # where M is the max_model_len.
    token_indices = positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
    token_indices_tensor = torch.from_numpy(token_indices)
    # Prepare input_ids.
    # NOTE(woosuk): We use torch.index_select instead of np.take here
    # because torch.index_select is much faster than np.take for large
    # tensors.
    torch.index_select(
        self.input_batch.token_ids_cpu_tensor.flatten(),
        0,
        token_indices_tensor,
        out=self.input_ids.cpu[:total_num_scheduled_tokens],
    )
    if self.enable_prompt_embeds:
        is_token_ids = self.input_batch.is_token_ids_tensor.flatten()
        torch.index_select(
            is_token_ids, 0, token_indices_tensor, out=self.is_token_ids.cpu[:total_num_scheduled_tokens]
        )

    # Because we did not pre-allocate a massive prompt_embeds CPU tensor on
    # the InputBatch, we need to fill in the prompt embeds into the expected
    # spots in the GpuModelRunner's pre-allocated prompt_embeds tensor.
    if self.input_batch.req_prompt_embeds and (self.is_multimodal_model or self.enable_prompt_embeds):
        output_idx = 0
        for req_idx in range(num_reqs):
            num_sched = num_scheduled_tokens[req_idx]

            # Skip if this request doesn't have embeddings
            if req_idx not in self.input_batch.req_prompt_embeds:
                output_idx += num_sched
                continue

            # Skip if no tokens scheduled
            if num_sched <= 0:
                output_idx += num_sched
                continue

            req_embeds = self.input_batch.req_prompt_embeds[req_idx]
            start_pos = self.input_batch.num_computed_tokens_cpu[req_idx]

            # Skip if trying to read beyond available embeddings
            if start_pos >= req_embeds.shape[0]:
                output_idx += num_sched
                continue

            # Copy available embeddings
            end_pos = start_pos + num_sched
            actual_end = min(end_pos, req_embeds.shape[0])
            actual_num_sched = actual_end - start_pos

            if actual_num_sched > 0:
                self.inputs_embeds.cpu[output_idx : output_idx + actual_num_sched].copy_(
                    req_embeds[start_pos:actual_end]
                )

            output_idx += num_sched

    self.query_start_loc.np[0] = 0
    self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
    self.query_start_loc.copy_to_gpu()

    # Now, query_start_loc is padded.
    # But gdn needs an unpadded one.
    # gdn_query_start_loc is an unpadded version of query_start_loc.
    # TODO delete it if fia's check is removed.
    if self._has_gdn:
        self.gdn_query_start_loc.np[0] = 0
        self.gdn_query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
        self.gdn_query_start_loc.np[num_reqs + 1 :].fill(cu_num_tokens[-1])
        self.gdn_query_start_loc.copy_to_gpu()


    # Compute optimistic seq_lens (assumes all draft tokens from previous
    # iteration accepted). Store in optimistic_seq_lens_cpu for use by
    # _build_attention_metadata (max_seq_len) and discard_request_mask.
    # seq_lens (GPU) will be computed later using the same optimistic values.
    torch.add(
        self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs],
        torch.from_numpy(num_scheduled_tokens),
        out=self.optimistic_seq_lens_cpu[:num_reqs],
    )
    self.optimistic_seq_lens_cpu[num_reqs:].fill_(0)

    # GSA: sparse (top-k) requests only attend to their sparsed window, so
    # their optimistic seq_len is the sparse slot count instead of
    # computed + scheduled.
    is_sparsed_np = np.zeros((num_reqs,), dtype=np.bool_)
    for req_id in self.input_batch.req_id_to_index:
        is_sparsed_request = scheduler_output.req_sparsed_slots[req_id] != INVALID_SLOT
        req_index = self.input_batch.req_id_to_index[req_id]
        if is_sparsed_request:
            self.optimistic_seq_lens_cpu[req_index] = req_sparsed_slots[req_id]
            is_sparsed_np[req_index] = True

    # Fill unused with -1. Needed for reshape_and_cache in attention_cp
    self.query_start_loc.gpu[num_reqs + 1 :].fill_(-1)

    # Copy the tensors to the NPU.
    self._prepare_input_ids(scheduler_output, num_reqs, total_num_scheduled_tokens, cu_num_tokens)
    # Calculate M-RoPE positions.
    # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
    if self.uses_mrope:
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        self._calc_mrope_positions(scheduler_output)
        self.mrope_positions.gpu.copy_(
            self.mrope_positions.cpu,
            non_blocking=True,
        )

    # Record the index of requests that should not be sampled,
    # so that we could clear the sampled tokens before returning
    num_tokens = [self.requests[r].num_tokens for r in self.input_batch.req_ids]
    num_tokens_np = np.array(num_tokens, dtype=np.int32)
    base_num_reqs = self.input_batch.num_reqs
    num_reqs = base_num_reqs
    # GSA: sparse requests never hit their full num_tokens boundary in the
    # optimistic domain; exclude them from the discard mask.
    discard_requests_mask = (
        self.optimistic_seq_lens_cpu[:num_reqs].numpy() < num_tokens_np
    ) & (~is_sparsed_np)

    discard_request_indices = np.nonzero(discard_requests_mask)[0]
    self.num_discarded_requests = len(discard_request_indices)
    self.discard_request_indices.np[: self.num_discarded_requests] = discard_request_indices
    self.discard_request_indices.copy_to_gpu(self.num_discarded_requests)

    self.discard_request_mask.np[:num_reqs] = discard_requests_mask
    self.discard_request_mask.copy_to_gpu(num_reqs)

    # Sync num_accepted_tokens from CPU (set by
    # _update_states_after_model_execute for hybrid models).
    if self.num_accepted_tokens_event is not None:
        self.num_accepted_tokens_event.synchronize()
        self._sync_num_accepted_tokens(
            num_reqs, has_prev_mapping=bool(prev_req_id_to_index)
        )
        self.num_accepted_tokens.np[num_reqs:].fill(1)
        self.num_accepted_tokens.copy_to_gpu()
    else:
        self.num_accepted_tokens.np.fill(1)
        self.num_accepted_tokens.gpu.fill_(1)

    # Update num_computed_tokens on GPU. In async spec decode,
    # CPU values are optimistic (all drafts accepted). The kernel
    # corrects on GPU using the previous step's
    # valid_sampled_token_count_gpu. Otherwise, just copy from CPU.
    valid_sampled_token_count_gpu = self.valid_sampled_token_count_gpu
    if self.use_async_spec_decode:
        computed_token_tensor_cpu = self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs].to(
            device=self.device, non_blocking=True
        )
    if (
        self.use_async_spec_decode
        and valid_sampled_token_count_gpu is not None
        and prev_req_id_to_index
    ):
        if prev_positions_gpu is None:
            self.prev_positions.copy_to_gpu(num_reqs)
        self.prev_num_draft_tokens.copy_to_gpu()
        update_num_computed_tokens_for_batch_change(
            self.num_computed_tokens,
            self.num_accepted_tokens.gpu[:num_reqs],
            self.prev_positions.gpu[:num_reqs],
            valid_sampled_token_count_gpu,
            self.prev_num_draft_tokens.gpu,
            computed_token_tensor_cpu,
        )
    else:
        self.num_computed_tokens[:num_reqs].copy_(
            self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs],
            non_blocking=True,
        )

    self.req_indices.np[:total_num_scheduled_tokens] = req_indices
    self.req_indices.copy_to_gpu(total_num_scheduled_tokens)
    req_indices_gpu = self.req_indices.gpu[:total_num_scheduled_tokens]

    self.query_pos.copy_to_gpu(total_num_scheduled_tokens)
    self.num_scheduled_tokens.np[:num_reqs] = num_scheduled_tokens
    self.num_scheduled_tokens.copy_to_gpu(num_reqs)
    num_scheduled_tokens_gpu = self.num_scheduled_tokens.gpu[:num_reqs]

    dcp_manager = getattr(self, "dcp_manager", None)
    if dcp_manager is not None:
        cp_async_rebuild = dcp_manager.rebuild_async_spec_decode_inputs(
            use_async_spec_decode=self.use_async_spec_decode,
            valid_sampled_token_count_gpu=valid_sampled_token_count_gpu,
            prev_req_id_to_index=prev_req_id_to_index,
            prev_positions_gpu=prev_positions_gpu,
            with_prefill=with_prefill,
            enable_prompt_embeds=self.enable_prompt_embeds,
            has_req_prompt_embeds=bool(self.input_batch.req_prompt_embeds),
            supports_mm_inputs=self.supports_mm_inputs,
            num_reqs=num_reqs,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            req_indices=req_indices,
            req_indices_gpu=req_indices_gpu,
            query_pos_gpu=self.query_pos.gpu,
            query_pos_np=self.query_pos.np,
            positions=self.positions,
            positions_np=positions_np,
            num_computed_tokens=self.num_computed_tokens,
            num_computed_tokens_cpu=self.input_batch.num_computed_tokens_cpu,
            prev_positions_np=self.prev_positions.np,
            prev_num_draft_tokens_np=self.prev_num_draft_tokens.np,
            valid_sampled_token_count_event=self.valid_sampled_token_count_event,
            valid_sampled_token_count_cpu=self.valid_sampled_token_count_cpu,
            input_batch=self.input_batch,
            input_ids=self.input_ids,
            scheduler_output=scheduler_output,
            arange_np=self.arange_np,
            cu_num_tokens=cu_num_tokens,
            draft_token_ids=self._draft_token_ids,  # type: ignore[has-type]
            num_spec_tokens=self.num_spec_tokens,
            prepare_input_ids=self._prepare_input_ids,
        )
    else:
        cp_async_rebuild = DCPAsyncSpecDecodeRebuildResult(
            rebuilt=False,
            positions_ready_on_device=False,
        )

    if cp_async_rebuild.positions_ready_on_device:
        pass
    elif cp_async_rebuild.rebuilt:
        # The async rebuild computed corrected positions on CPU.
        # Copy positions_np to GPU so input_ids and positions stay aligned.

        self.positions[:total_num_scheduled_tokens].copy_(
            torch.from_numpy(
                positions_np[:total_num_scheduled_tokens]
            ).to(self.device),
            non_blocking=True,
        )
    else:
        self.positions[:total_num_scheduled_tokens] = (
            self.num_computed_tokens[req_indices_gpu].to(torch.int64)
            + self.query_pos.gpu[:total_num_scheduled_tokens]
        )

    # GSA: sparse runs read seq_lens from the optimistic (sparse-window)
    # domain and upload the sparsed positions into the dedicated
    # sparse_positions buffer consumed by the slot-mapping kernel below.
    # Non-GSA runs alias sparse_positions to positions (upstream behavior).
    if has_ucm_sparse():
        self.sparse_positions[:total_num_scheduled_tokens].copy_(
            torch.from_numpy(sparsed_positions[:total_num_scheduled_tokens]),
            non_blocking=True,
        )
        self.seq_lens[:num_reqs].copy_(
            self.optimistic_seq_lens_cpu[:num_reqs], non_blocking=True
        )
    else:
        self.sparse_positions = self.positions
        self.seq_lens[:num_reqs] = (
            self.num_computed_tokens[:num_reqs] + num_scheduled_tokens_gpu
        )
    self.seq_lens[num_reqs:].fill_(0)

    # In async spec decode mode, optimistic_seq_lens_cpu assumes all
    # tokens from the previous speculative step were accepted. Correct it
    # on CPU using the valid-sampled-token counts that are already copied
    # asynchronously for scheduler bookkeeping. This avoids an extra
    # NPU->CPU seq_lens copy and the synchronize() in attention metadata.
    # Mirrors update_num_computed_tokens_for_batch_change on the GPU side.
    async_spec_decode_active = (
        self.use_async_spec_decode
        and valid_sampled_token_count_gpu is not None
        and prev_req_id_to_index
    )
    if self._needs_seq_lens_cpu_sync and async_spec_decode_active:
        self._correct_optimistic_seq_lens_cpu(num_reqs)

    replay_start = self._fill_replay_start(num_reqs)

    # For non-DCP, compute slot_mapping on GPU.
    # GSA: the third arg is the sparsed-position buffer (query-relative
    # slot positions for sparse requests, absolute for the rest).
    self.input_batch.block_table.compute_slot_mapping(
        num_reqs,
        self.query_start_loc.gpu[: num_reqs + 1],
        self.sparse_positions[:total_num_scheduled_tokens],
    )

    if replay_start is not None:
        # After every group's slot mapping is final, so this sees the same
        # slots the attention kernels are about to be handed.
        self.input_batch.block_table.pad_replayed_slots(
            num_reqs,
            self.query_start_loc.gpu[: num_reqs + 1],
            self.positions[:total_num_scheduled_tokens],
            replay_start,
            self.prefix_replay_tokens,
        )

    if self.use_async_spec_decode and self.uses_mrope:
        drift = self.num_computed_tokens[req_indices_gpu].to(
            torch.int64
        ) - computed_token_tensor_cpu[req_indices_gpu]
        target = self.mrope_positions if self.uses_mrope else self.xdrope_positions
        target.gpu[:, :total_num_scheduled_tokens] += drift

    use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
    if not use_spec_decode:
        # NOTE(woosuk): Due to chunked prefills, the batch may contain
        # partial requests. While we should not sample any token
        # from these partial requests, we do so for simplicity.
        # We will ignore the sampled tokens from the partial requests.
        # TODO: Support prompt logprobs.
        spec_decode_metadata = None
        num_draft_tokens = None
        num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)
        logits_indices = self.query_start_loc.gpu[1 : num_reqs + 1] - 1
    else:
        # Get the number of draft tokens for each request.
        # Iterate over the dictionary rather than all requests since not all
        # requests have draft tokens.
        num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
        # For chunked prefills, use -1 as mask rather than 0, as guided
        # decoding may rollback speculative tokens.
        new_schedule_reqs = [x.req_id for x in scheduler_output.scheduled_new_reqs]
        num_decode_draft_tokens = np.full(num_reqs, -1, dtype=np.int32)
        for (
            req_id,
            draft_token_ids,
        ) in scheduler_output.scheduled_spec_decode_tokens.items():
            req_idx = self.input_batch.req_id_to_index[req_id]
            draft_len = len(draft_token_ids)
            num_draft_tokens[req_idx] = draft_len
            if (self.is_kv_consumer and req_id in new_schedule_reqs) or \
               (self.input_batch.num_computed_tokens_cpu[req_idx] >= \
                self.input_batch.num_prompt_tokens[req_idx]):
                num_decode_draft_tokens[req_idx] = draft_len
            else:
                num_decode_draft_tokens[req_idx] = -1

        spec_decode_metadata = self._calc_spec_decode_metadata(
            num_draft_tokens,
            cu_num_tokens,
        )
        logits_indices = spec_decode_metadata.logits_indices
        num_sampled_tokens = num_draft_tokens + 1

        # For DECODE only cuda graph of some attention backends (e.g., GDN).
        self.num_decode_draft_tokens.np[:num_reqs] = num_decode_draft_tokens
        self.num_decode_draft_tokens.np[num_reqs:].fill(-1)
        self.num_decode_draft_tokens.copy_to_gpu()
    self.logits_indices = logits_indices

    # Hot-Swap lora model
    if self.lora_config:
        assert np.sum(num_sampled_tokens) <= self.vllm_config.scheduler_config.max_num_batched_tokens
        self.set_active_loras(self.input_batch, num_scheduled_tokens, num_sampled_tokens)
    if lmhead_tp_enable():
        max_num_reqs_across_dp = self.max_num_reqs * self.uniform_decode_query_len
        logits_indices = nn.functional.pad(logits_indices, (0, max_num_reqs_across_dp - logits_indices.shape[0]))

    return (
        logits_indices,
        spec_decode_metadata,
        total_num_scheduled_tokens,
    )


def execute_model(
    self,
    scheduler_output: "SchedulerOutput",
    intermediate_tensors: IntermediateTensors | None = None,
) -> ModelRunnerOutput | IntermediateTensors | None:
    self._cpp_execution_time_ms = None
    profiling_chunk_config = self.ascend_config.scheduler_config.profiling_chunk_config
    execution_start_time = _start_profiling_chunk_timing(
        profiling_chunk_config,
        scheduler_output,
    )
    if self.execute_model_state is not None:
        raise RuntimeError("State error: sample_tokens() must be called after execute_model() returns None.")

    # If ngram_gpu is used, we need to copy the scheduler_output to avoid
    # the modification has influence on the scheduler_output in engine core process.
    # The replace is much faster than deepcopy.
    if (
        self.speculative_config is not None
        and self.speculative_config.use_ngram_gpu()
    ):
        num_scheduled_tokens_copy = scheduler_output.num_scheduled_tokens.copy()
        spec_decode_tokens_copy = (
            scheduler_output.scheduled_spec_decode_tokens.copy()
        )
        scheduler_output = replace(
            scheduler_output,
            num_scheduled_tokens=num_scheduled_tokens_copy,
            scheduled_spec_decode_tokens=spec_decode_tokens_copy,
        )

    # NOTE: The async-scheduling deepcopy was removed in
    # vllm-project/vllm#29821. Draft token ids are now propagated via
    # _copy_draft_token_ids_to_cpu()/take_draft_token_ids(), and the
    # scheduler output is updated by EngineCore in its own process
    # (Scheduler.update_draft_token_ids_in_output). Nothing below mutates
    # scheduler_output in the worker, so no copy is needed.
    pp_group = get_pp_group()
    if pp_group.world_size > 1 and not pp_group.is_last_rank:
        cached_reqs = scheduler_output.scheduled_cached_reqs
        new_token_ids = cached_reqs.new_token_ids
        if new_token_ids and all(not token_ids for token_ids in new_token_ids):
            scheduler_output = replace(
                scheduler_output,
                scheduled_cached_reqs=replace(
                    cached_reqs,
                    new_token_ids=[],
                ),
            )

    if has_kv_transfer_group():
        kv_connector_metadata = scheduler_output.kv_connector_metadata
        assert kv_connector_metadata is not None
        # Preemption stores must run before _update_states() zeroes newly
        # allocated blocks that may reuse the same physical KV cache IDs.
        get_kv_transfer_group().handle_preemptions(kv_connector_metadata)

    num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
    with record_function_or_nullcontext("prepare input"):
        with self.synchronize_input_prep():
            # Fix up prev_req_id_to_index for requests that were discarded
            # in the previous sample_tokens step. If a request has
            # prev_num_draft_len > 0 but is missing from
            # prev_req_id_to_index, the parent _update_states would
            # hit a KeyError. Reset prev_num_draft_len to 0 for such
            # requests so they fall through safely.
            if (
                self.use_async_scheduling
                and self.num_spec_tokens
                and self.input_batch.prev_req_id_to_index is not None
            ):
                for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                    if (
                        req_id not in self.input_batch.prev_req_id_to_index
                        and (req_state := self.requests.get(req_id)) is not None
                        and req_state.prev_num_draft_len
                    ):
                        req_state.prev_num_draft_len = 0

            # Update persistent batch states.
            deferred_state_corrections_fn = self._update_states(
                scheduler_output
            )

            if has_ec_transfer() and not get_ec_transfer().is_consumer:
                self._start_dump_data(scheduled_tokens = scheduler_output.num_scheduled_tokens)
                with self.maybe_get_ec_connector_output(
                    scheduler_output,
                    encoder_cache=self.encoder_cache,
                ) as ec_connector_output:
                    self._execute_mm_encoder(scheduler_output)
                    self._finalize_dump_data()
                    return make_empty_encoder_model_runner_output(scheduler_output)

            if not num_scheduled_tokens:
                if (
                    self.parallel_config.distributed_executor_backend == "external_launcher"
                    and self.parallel_config.data_parallel_size > 1
                ):
                    # this is a corner case when both external launcher
                    # and DP are enabled, num_scheduled_tokens could be
                    # 0, and has_unfinished_requests in the outer loop
                    # returns True. before returning early here we call
                    # dummy run to ensure coordinate_batch_across_dp
                    # is called into to avoid out of sync issues.
                    self._dummy_run(1, skip_gdn_state_update=True)
                if not has_kv_transfer_group():
                    # Return empty ModelRunnerOutput if no work to do.
                    return EMPTY_MODEL_RUNNER_OUTPUT
                return self.kv_connector_no_forward(scheduler_output, self.vllm_config)
            if self.cache_config.kv_sharing_fast_prefill:
                assert not self.num_prompt_logprobs, (
                    "--kv-sharing-fast-prefill produces incorrect "
                    "logprobs for prompt tokens, tokens, please disable "
                    "it when the requests need prompt logprobs"
                )

            num_reqs = self.input_batch.num_reqs
            req_ids = self.input_batch.req_ids
            tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
            if (scheduler_output.total_num_scheduled_tokens <= 0
                    or not tokens or sum(tokens) == 0):
                if not has_kv_transfer_group():
                    return EMPTY_MODEL_RUNNER_OUTPUT
                return self.kv_connector_no_forward(scheduler_output, self.vllm_config)
            self._start_dump_data(scheduled_tokens = scheduler_output.num_scheduled_tokens)
            num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
            max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())
            (
                logits_indices,
                spec_decode_metadata,
                total_num_scheduled_tokens,
            ) = self._prepare_inputs(
                scheduler_output,
                num_scheduled_tokens_np,
            )

            num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens
            cascade_attn_prefix_lens = None
            # Disable cascade attention when using microbatching (DBO)
            if self.cascade_attn_enabled and not self.parallel_config.enable_dbo:
                # Pre-compute cascade attention prefix lengths
                cascade_attn_prefix_lens = self._compute_cascade_attn_prefix_lens(
                    num_scheduled_tokens_np,
                    self.input_batch.num_computed_tokens_cpu[:num_reqs],
                    scheduler_output.num_common_prefix_blocks,
                )

            (
                cudagraph_mode,
                batch_desc,
                should_ubatch,
                num_tokens_across_dp,
                cudagraph_stats,
            ) = self._determine_batch_execution_and_padding(
                num_tokens=num_tokens_unpadded,
                num_reqs=num_reqs,
                num_scheduled_tokens_np=num_scheduled_tokens_np,
                max_num_scheduled_tokens=max_num_scheduled_tokens,
                use_cascade_attn=cascade_attn_prefix_lens is not None,
                force_eager=self.model_config.enforce_eager,
                num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),
            )

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Running batch with cudagraph_mode: %s, batch_descriptor: %s, "
                    "should_ubatch: %s, num_tokens_across_dp: %s",
                    cudagraph_mode,
                    batch_desc,
                    should_ubatch,
                    num_tokens_across_dp,
                )

            num_tokens_padded = batch_desc.num_tokens
            num_reqs_padded = batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
            ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
                should_ubatch,
                num_scheduled_tokens_np,
                num_tokens_padded,
                num_reqs_padded,
                self.parallel_config.num_ubatches,
            )

            if self.dynamic_eplb:
                self.update_eplb_heat_collection_status(num_tokens_padded, num_tokens_across_dp)

            pad_attn = cudagraph_mode == CUDAGraphMode.FULL

            # postprocess_mamba runs later in _update_states_after_model_execute.
            if self.cache_config.mamba_cache_mode == "align":
                # preprocess_mamba reads req_state.num_computed_tokens (CPU)
                # to decide copy operations, so we must apply deferred
                # corrections before it runs.
                if deferred_state_corrections_fn:
                    deferred_state_corrections_fn()
                    deferred_state_corrections_fn = None
                mamba_bufs = self._get_mamba_bufs()
                preprocess_bufs = mamba_bufs.preprocess
                mamba_utils.preprocess_mamba(
                    scheduler_output,
                    self.kv_cache_config,
                    self.cache_config,
                    self.mamba_state_idx,
                    self.input_batch,
                    self.requests,
                    self.compilation_config.static_forward_context,
                    self._get_ascend_mamba_state_copy_funcs(),
                    preprocess_bufs,
                )
                # preprocess_mamba resets num_accepted_tokens_cpu to 1
                # for requests whose state was copied to a new block.
                # Re-sync to GPU so the mamba kernel reads from the
                # correct initial state slot (init_token_idx = 0).
                self.num_accepted_tokens.np[:num_reqs] = (
                    self.input_batch.num_accepted_tokens_cpu[:num_reqs]
                )
                self.num_accepted_tokens.copy_to_gpu(num_reqs)

                if mamba_bufs.postprocess_align is not None:
                    mamba_utils.stage_postprocess_inputs_to_gpu(
                        mamba_bufs.postprocess_align,
                        scheduler_output,
                        self.input_batch.req_ids,
                        num_reqs,
                        self.requests,
                        self.mamba_state_idx,
                    )

            if self.use_compress:
                if deferred_state_corrections_fn:
                    deferred_state_corrections_fn()
                    deferred_state_corrections_fn = None
                num_reqs = self.input_batch.num_reqs
                req_indices = np.repeat(
                    self.arange_np[:num_reqs], num_scheduled_tokens_np
                )
                dsa_positions_np = self._dsa_positions_np_buf[
                    :total_num_scheduled_tokens
                ]
                np.add(
                    self.input_batch.num_computed_tokens_cpu[req_indices],
                    self.query_pos.np[:total_num_scheduled_tokens],
                    out=dsa_positions_np,
                )

            use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
            ubatch_slices_attn = ubatch_slices_padded if pad_attn else ubatch_slices

            if (
                cudagraph_mode == CUDAGraphMode.FULL
                or (enable_sp() and not self.model_config.use_mla)
                and self.dcp_size == 1
            ):
                # Currently, Graph Mode and SP will both pad num_tokens,
                # Another possible condition is num_tokens_padded != num_tokens_unpadded
                # but this scope is way too big and the consequences are unpredictable
                num_reqs_padded = self._pad_query_start_loc_for_fia(
                    self.query_start_loc,
                    num_tokens_padded,
                    num_reqs_padded,
                    num_reqs,
                    cudagraph_mode,
                    batch_desc.num_reqs,
                )

            (attn_metadata, spec_decode_common_attn_metadata) = self._build_attention_metadata(
                num_tokens=num_tokens_unpadded,
                num_tokens_padded=num_tokens_padded,
                num_reqs=num_reqs,
                num_reqs_padded=num_reqs_padded,
                max_query_len=max_num_scheduled_tokens,
                ubatch_slices=ubatch_slices_attn,
                logits_indices=logits_indices,
                use_spec_decode=use_spec_decode,
                num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
                num_scheduled_tokens_np=num_scheduled_tokens_np,
                cascade_attn_prefix_lens=cascade_attn_prefix_lens,
                cudagraph_runtime_mode=cudagraph_mode,
                batch_descriptor=batch_desc,
            )

            self._sanitize_placeholder_input_ids_for_forward(
                scheduler_output,
                num_tokens_padded,
            )

        (
            input_ids,
            inputs_embeds,
            positions,
            intermediate_tensors,
            model_kwargs,
            ec_connector_output,
        ) = self._preprocess(
            scheduler_output,
            num_tokens_padded,
            intermediate_tensors,
        )

        # update global cos, sin
        update_cos_sin(positions)

    if self.kvpp.scheduler is not None:
        self.kvpp.prepare_forward(bool(np.any(self.input_batch.num_computed_tokens_cpu[:num_reqs] > 0)))

    if self.dynamic_eplb:
        self.eplb_updator.forward_before()

    # Encoder-decoder models and raw-token multimodal models can only
    # compile pure decode steps where no encoder inputs are present. The
    # DeepSeek-V4 vision router needs raw sentinel ids during image
    # prefill, so keep that pass eager.
    num_encoder_reqs = len(scheduler_output.scheduled_encoder_inputs)
    has_encoder_input = num_encoder_reqs > 0 and (
        self.model_config.is_encoder_decoder
        or self.model_config.requires_raw_input_tokens
    )
    # Run forward pass
    defer_kv_connector_finalize = self.speculative_config is not None and (
        get_pp_group().is_last_rank or self.broadcast_pp_output
    )
    active_device_metadata_executor = self._prepare_device_metadata_for_forward(cudagraph_mode)
    with (
        record_function_or_nullcontext("forward"),
        set_ascend_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens_padded,
            num_tokens_across_dp=num_tokens_across_dp,
            aclgraph_runtime_mode=cudagraph_mode,
            batch_descriptor=batch_desc,
            num_actual_tokens=scheduler_output.total_num_scheduled_tokens,
            model_instance=self.model,
            device_metadata_executor=active_device_metadata_executor,
            skip_compiled=has_encoder_input,
            has_sinks=self._has_sinks,
            eplb_heat_collection_status=self.eplb_heat_collection_status if self.dynamic_eplb else False,
        ),
        self.maybe_get_kv_connector_output(
            scheduler_output,
            **(
                {"defer_finalize": defer_kv_connector_finalize}
            ),
        ) as kv_connector_output,
    ):
        # Mamba state copy must run AFTER the KV transfer load finishes,
        # otherwise the copy would race with in-flight layerwise loads and
        # read half-loaded state. With a layerwise-capable connector we
        # defer the copy: prepare_mamba_state_copy stages per-layer copy
        # metadata here, and each layer's copy is executed right after its
        # own KV load completes (wait_for_layer_load), overlapping the copy
        # with the remaining layers' loads. Non-layerwise connectors keep
        # the batched copy after all loads finish (do_mamba_copy_block).
        mamba_copy_connector = None
        if self.cache_config.mamba_cache_mode == "align":
            if has_kv_transfer_group():
                connector = get_kv_transfer_group()
                prepare_mamba_state_copy = getattr(
                    connector,
                    "prepare_mamba_state_copy",
                    None,
                )
                if callable(prepare_mamba_state_copy) and prepare_mamba_state_copy(preprocess_bufs):
                    mamba_copy_connector = connector
            if mamba_copy_connector is None:
                mamba_utils.do_mamba_copy_block(preprocess_bufs)

        # GSA: run the sparse step-begin hook (build_sparse_meta +
        # execute_begin) right before the model forward.
        self.maybe_execute_ucm_sparse_begin(scheduler_output, attn_metadata)

        hidden_states = self._model_forward(
            num_tokens_padded, input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs
        )

        # GSA: post-forward warmup hook (no-op unless the sparse state
        # requests a warmup step).
        self.maybe_execute_ucm_sparse_warmup()
        if self.vllm_config.speculative_config is None:
            # GSAV2 may rewrite logits_indices; keep the returned one for sampling.
            logits_indices = self.maybe_execute_ucm_sparse_finished(logits_indices)

        self._cpp_execution_time_ms = _finish_profiling_chunk_timing(
            profiling_chunk_config,
            execution_start_time,
        )
        # Verify every scheduled layer executed its deferred copy.
        if self.cache_config.mamba_cache_mode == "align" and mamba_copy_connector is not None:
            mamba_copy_connector.finish_mamba_state_copy()
    if active_device_metadata_executor is not None and active_device_metadata_executor.submission_in_flight:
        active_device_metadata_executor.release()
    self.kvpp.complete_forward()

    with record_function_or_nullcontext("post process"):
        aux_hidden_states = None
        if self.use_aux_hidden_state_outputs:
            hidden_states, aux_hidden_states = hidden_states
        if not self.broadcast_pp_output:
            # Common case.
            if not get_pp_group().is_last_rank:
                # Return the intermediate tensors.
                assert isinstance(hidden_states, IntermediateTensors)
                hidden_states.kv_connector_output = kv_connector_output
                self.kv_connector_output = kv_connector_output
                self._finalize_dump_data()
                if self.dynamic_eplb:
                    self.eplb_updator.forward_end(self.eplb_heat_collection_status)
                return hidden_states
            if self.is_pooling_model:
                # Return the pooling output.
                output = self._pool(
                    hidden_states, num_scheduled_tokens, num_scheduled_tokens_np, kv_connector_output
                )
                output.kv_connector_output = kv_connector_output
                self._finalize_dump_data()
                return output

            sample_hidden_states = hidden_states[logits_indices]
            logits = self.model.compute_logits(sample_hidden_states)
        else:
            # Rare case.
            assert not self.is_pooling_model

            if not get_pp_group().is_last_rank:
                sample_hidden_states = hidden_states[logits_indices]
                get_pp_group().send_tensor_dict(hidden_states.tensors, all_gather_group=get_tp_group())
                logits = None
            else:
                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states)

            model_output_broadcast_data: dict[str, Any] = {}
            if logits is not None:
                model_output_broadcast_data["logits"] = logits.contiguous()
            broadcasted = get_pp_group().broadcast_tensor_dict(
                model_output_broadcast_data, src=len(get_pp_group().ranks) - 1
            )
            assert broadcasted is not None
            logits = broadcasted["logits"]

        # Apply structured output bitmasks if present
        self.execute_model_state = ExecuteModelState(
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            attn_metadata,
            positions,
            ec_connector_output,
            cudagraph_stats,
            batch_desc,
        )
        self.kv_connector_output = kv_connector_output

    # Now the batch has been launched we can wait for corrections from the
    # previous model forward without breaking async scheduling.
    if deferred_state_corrections_fn:
        deferred_state_corrections_fn()
    return None


def sample_tokens(
    self, grammar_output: "GrammarOutput | None"
) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
    from ucm.sparse.state import has_ucm_sparse

    kv_connector_output = self.kv_connector_output
    self.kv_connector_output = None
    pp = get_pp_group()
    use_pp_spec_decode = self.speculative_config is not None and pp.world_size > 1

    if self.execute_model_state is None:
        # Nothing to do (PP non-final rank case), output isn't used.
        if not kv_connector_output:
            return None  # noqa
        # In case of PP with kv transfer, we need to pass through the
        # kv_connector_output
        if kv_connector_output.is_empty():
            return EMPTY_MODEL_RUNNER_OUTPUT

        output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
        output.kv_connector_output = kv_connector_output
        return output

    # Unpack ephemeral state.
    (
        scheduler_output,
        logits,
        spec_decode_metadata,
        spec_decode_common_attn_metadata,
        hidden_states,
        sample_hidden_states,
        aux_hidden_states,
        attn_metadata,
        positions,
        ec_connector_output,
        cudagraph_stats,
        batch_desc,
    ) = self.execute_model_state
    # Clear ephemeral state.
    self.execute_model_state = None

    # Apply structured output bitmasks if present.
    if grammar_output is not None:
        # here we are different from gpu_model_runner,
        # the apply_grammar_bitmask uses torch.compile to optimize this,ascend does not support it now
        logits_dtype = logits.dtype
        logits = logits.to("cpu").float()
        apply_grammar_bitmask(scheduler_output, grammar_output, self.input_batch, logits)
        logits = logits.to(self.device).to(logits_dtype)

    with record_function_or_nullcontext("sample_token"):
        sampler_output = self._sample(logits, spec_decode_metadata)

    if self.need_accepted_tokens:
        if self.sampling_done_event is None:
            self.sampling_done_event = torch.npu.Event()

        assert self.sampling_done_event is not None
        self.sampling_done_event.record()

    self.valid_sampled_token_count_gpu = None

    input_fits_in_drafter = self.speculative_config is None or self._input_fits_in_drafter(
        spec_decode_common_attn_metadata
    )

    def propose_draft_token_ids(sampled_token_ids):
        if not input_fits_in_drafter:
            self._skip_drafting(
                scheduler_output,
                sampled_token_ids if use_padded_batch else None,
            )
            return
        assert spec_decode_common_attn_metadata is not None
        self._draft_token_ids = self.propose_draft_token_ids(
            sampled_token_ids,
            self.input_batch.sampling_metadata,
            scheduler_output,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            positions,
            scheduler_output.total_num_scheduled_tokens,
            hidden_states,
            aux_hidden_states,
            sample_hidden_states,
            batch_desc,
        )
        self._copy_draft_token_ids_to_cpu(scheduler_output)

    use_padded_batch = False
    early_pp_padded_drafter = False
    if self.speculative_config:
        use_padded_batch = (
            self.speculative_config.use_eagle()
            or self.speculative_config.uses_draft_model()
            or self.speculative_config.uses_extract_hidden_states()
            or self.speculative_config.use_ngram_gpu()
        ) and not self.speculative_config.disable_padded_drafter_batch
        early_pp_padded_drafter = (
            use_pp_spec_decode
            and not self.use_async_scheduling
            and use_padded_batch
        )
        if early_pp_padded_drafter:
            self._draft_token_ids = None
            self._draft_token_req_ids = None
            with record_function_or_nullcontext("draft_token"):
                propose_draft_token_ids(sampler_output.sampled_token_ids)

    (
        num_nans_in_logits,
        num_nans_device,
        logprobs_lists,
        valid_sampled_token_ids,
        prompt_logprobs_dict,
        req_ids_output_copy,
        req_id_to_index_output_copy,
        invalid_req_indices,
    ) = self._bookkeeping_sync(
        scheduler_output,
        sampler_output,
        logits,
        hidden_states,
        scheduler_output.total_num_scheduled_tokens,
        spec_decode_metadata,
    )

    with record_function_or_nullcontext("draft_token"):
        if self.speculative_config:
            if not early_pp_padded_drafter:
                self._draft_token_ids = None
                self._draft_token_req_ids = None
            if use_padded_batch and not early_pp_padded_drafter:
                # EAGLE speculative decoding can use the GPU sampled tokens
                # as inputs, and does not need to wait for bookkeeping to finish.
                propose_draft_token_ids(sampler_output.sampled_token_ids)
            if self.speculative_config and not use_padded_batch:
                # ngram and other speculative decoding methods use the sampled
                # tokens on the CPU, so they are run after bookkeeping.
                propose_draft_token_ids(valid_sampled_token_ids)

        # vLLM v0.18 defers KV connector finalization during target-model
        # forward when speculative decoding is enabled. Finalize here after
        # draft model runs so KV pool save/put can complete.
        if self.speculative_config is not None:
            self.finalize_kv_connector()

    # GSA: spec-decode runs call the sparse finished hook here (the non-spec
    # path already ran it in execute_model) with the accepted-token counts so
    # the sparse runtime can settle the step.
    if has_ucm_sparse() and self.vllm_config.speculative_config:
        if spec_decode_metadata is None:
            self.maybe_execute_ucm_sparse_finished()
        else:
            valid_counts = (sampler_output.sampled_token_ids != -1).sum(dim=1)
            self.maybe_execute_ucm_sparse_finished(
                rec_result=torch.where(
                    valid_counts > 0, valid_counts - 1, torch.tensor(-1)
                ).tolist()
            )

    model_runner_output = ModelRunnerOutput(
        req_ids=req_ids_output_copy,
        req_id_to_index=req_id_to_index_output_copy,
        sampled_token_ids=valid_sampled_token_ids,
        logprobs=logprobs_lists,
        prompt_logprobs_dict=prompt_logprobs_dict,
        kv_connector_output=kv_connector_output,
        pooler_output=[],
        ec_connector_output=ec_connector_output if self.supports_mm_inputs else None,
        num_nans_in_logits=num_nans_in_logits,
        cudagraph_stats=cudagraph_stats,
    )
    if self.dynamic_eplb:
        self.eplb_updator.forward_end(self.eplb_heat_collection_status)

    self._finalize_dump_data()

    if self.need_accepted_tokens:
        assert self.sampling_done_event is not None
        with (
            record_function_or_nullcontext("async_state_update"),
            torch.npu.stream(global_stream()),
        ):
            global_stream().wait_event(self.sampling_done_event)
            self._update_states_after_model_execute(sampler_output.sampled_token_ids, scheduler_output)

    if not self.use_async_scheduling:
        return model_runner_output

    # R3 is served by the V2 runner through vLLM's AuxOutput connector.
    async_output = AsyncGPUModelRunnerOutput(
        model_runner_output=model_runner_output,
        sampled_token_ids=sampler_output.sampled_token_ids,
        logprobs_tensors=sampler_output.logprobs_tensors,
        invalid_req_indices=invalid_req_indices,
        async_output_copy_stream=self.async_output_copy_stream,
        vocab_size=self.input_batch.vocab_size,
        num_nans=num_nans_device,
    )
    self.input_batch.set_async_sampled_token_ids(
        async_output.sampled_token_ids_cpu,
        async_output.async_copy_ready_event,
    )
    return async_output


def _build_attention_metadata(
    self,
    num_tokens: int,
    num_reqs: int,
    max_query_len: int,
    num_tokens_padded: int | None = None,
    num_reqs_padded: int | None = None,
    ubatch_slices: UBatchSlices | None = None,
    logits_indices: torch.Tensor | None = None,
    use_spec_decode: bool = False,
    for_cudagraph_capture: bool = False,
    num_scheduled_tokens: dict[str, int] | None = None,
    num_scheduled_tokens_np: np.ndarray | None = None,
    dcp_dummy_metadata: DCPDummyRunMetadata | None = None,
    cascade_attn_prefix_lens: list[list[int]] | None = None,
    skip_gdn_state_update: bool = False,
    cudagraph_runtime_mode: CUDAGraphMode | None = None,
    batch_descriptor: BatchDescriptor | None = None,
    offload_dummy: bool = False,
) -> tuple[PerLayerAttnMetadata, CommonAttentionMetadata | None]:
    """
    :return: tuple[attn_metadata, spec_decode_common_attn_metadata]
    """
    from ucm.sparse.state import has_ucm_sparse
    from ucm.sparse.utils import IS_INDEXER_OFFLOAD

    # Attention metadata is not needed for attention free models
    if len(self.kv_cache_config.kv_cache_groups) == 0:
        return {}, None
    num_tokens_padded = num_tokens_padded or num_tokens
    num_reqs_padded = num_reqs_padded or num_reqs
    if self.sparse_kv_offload_enabled:
        update_sparse_kv_offload_metadata(
            num_tokens,
            num_reqs,
            num_tokens_padded,
            num_reqs_padded,
            self.input_batch.req_ids[:num_reqs],
            self.query_start_loc,
            self._offload_req_ids_tensor,
            self._offload_token_to_req,
        )
    if self.sparse_kv_offload_config.use_fused_copy_sfa and self.sparse_kv_offload_enabled:
        assert self._offload_pool_slots is not None
        assert self._offload_pool_active is not None
        assert self._offload_request_states is not None
        groups = [group for cache_groups in self.attn_groups for group in cache_groups]
        groups.extend(getattr(self.drafter, "draft_attn_groups", ()))
        lim_cache_histories = [
            history
            for group in groups
            for builder in group.metadata_builders
            if (history := getattr(builder, "lim_last_cache", None)) is not None
        ]
        restore_tails, dense_fills, _ = self._offload_request_states.prepare(
            req_ids=self.input_batch.req_ids[:num_reqs],
            live_req_ids=self.input_batch.req_id_to_index,
            slots=self._offload_pool_slots.np,
            active=self._offload_pool_active.np,
            prebound_slots=get_prebound_copy_sfa_slots() if not offload_dummy else {},
            computed_tokens=getattr(self.input_batch, "num_computed_tokens_cpu", None),
            padded_reqs=num_reqs_padded,
            block_size=self.cache_config.block_size,
            hot_tokens=self.sparse_kv_offload_config.topk_buffer_size,
            dummy=offload_dummy,
            lim_cache_histories=lim_cache_histories,
        )
        self._copy_sfa_need_eager_tail_restore = restore_tails
        if dense_fills:
            assert self.sparse_kv_offload_manager is not None
            self.sparse_kv_offload_manager.dense_fill_copy_sfa_rows(
                dense_fills,
                block_size=self.cache_config.block_size,
                block_table=self.input_batch.block_table[0].get_numpy_array(),
            )
    attn_metadata: PerLayerAttnMetadata = {}
    device_metadata_tasks: list[DeviceMetadataTask] | None = (
        [] if self.device_metadata_executor is not None else None
    )
    if ubatch_slices is not None:
        attn_metadata = [dict() for _ in range(len(ubatch_slices))]

    if for_cudagraph_capture:
        # For some attention backends (e.g., FA) with sliding window models we need
        # to make sure the backend see a max_seq_len that is larger to the
        # sliding window size when capturing to make sure the correct kernel is selected.
        max_seq_len = self.max_model_len
    else:
        max_seq_len = self.optimistic_seq_lens_cpu.numpy()[:num_reqs].max().item()


    kv_cache_groups = self.kv_cache_config.kv_cache_groups

    def _get_dcp_metadata(block_table_tensor):
        if not self.use_dcp:
            return None, block_table_tensor

        fixed_decode_seq_lens_cpu = None
        if dcp_dummy_metadata is not None:
            fixed_decode_seq_lens_cpu = dcp_dummy_metadata.seq_lens_cpu
        elif self.use_async_spec_decode:
            fixed_decode_seq_lens_cpu = self.optimistic_seq_lens_cpu[:num_reqs].numpy()

        assert num_reqs_padded is not None
        return self.dcp_manager.generate_dcp_metadata(
            num_tokens,
            self.query_lens,
            self.input_batch,
            num_scheduled_tokens_np,
            block_table_tensor,
            num_reqs_padded,
            num_reqs,
            fixed_decode_seq_lens_cpu,
        )

    def _get_block_table_and_slot_mapping(
        kv_cache_gid: int,
    ):
        assert num_reqs_padded is not None and num_tokens_padded is not None
        kv_cache_spec = kv_cache_groups[kv_cache_gid].kv_cache_spec
        if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
            blk_table_tensor = torch.zeros(
                (num_reqs_padded, 1),
                dtype=torch.int32,
                device=self.device,
            )
            slot_mapping = torch.zeros(
                (num_tokens_padded,),
                dtype=torch.int64,
                device=self.device,
            )
        else:
            blk_table = self.input_batch.block_table[kv_cache_gid]
            slot_mapping = blk_table.slot_mapping.gpu[:num_tokens_padded]
            blk_table_tensor = blk_table.get_device_tensor()[:num_reqs_padded]
            # Fill unused with -1. Needed for reshape_and_cache in full cuda
            # graph mode. `blk_table_tensor` -1 to match mamba PAD_SLOT_ID
            slot_mapping[num_tokens:num_tokens_padded].fill_(-1)
            blk_table_tensor[num_reqs:num_reqs_padded].fill_(0)
        return blk_table_tensor, slot_mapping

    block_table_gid_0, slot_mapping_gid_0 = _get_block_table_and_slot_mapping(0)
    self.long_seq_metadata, block_table_gid_0 = _get_dcp_metadata(block_table_gid_0)
    if dcp_dummy_metadata is not None:
        # A DCP dummy decode must not inherit request state from the
        # previous real batch. Keep this local to metadata construction so
        # the persistent input batch is not modified.
        num_computed_tokens_cpu = self.input_batch.num_computed_tokens_cpu_tensor[
            :num_reqs_padded
        ].clone()
        num_computed_tokens_cpu.zero_()
        num_computed_tokens_cpu[:num_reqs].copy_(
            torch.from_numpy(dcp_dummy_metadata.num_computed_tokens_cpu)
        )
        is_prefilling = torch.zeros_like(num_computed_tokens_cpu, dtype=torch.bool)
    else:
        num_computed_tokens_cpu = self.input_batch.num_computed_tokens_cpu_tensor[
            :num_reqs_padded
        ]
        num_prompt_tokens_cpu = self.input_batch.num_prompt_tokens_cpu_tensor[
            :num_reqs_padded
        ]
        is_prefilling = num_computed_tokens_cpu < num_prompt_tokens_cpu
        is_prefilling[num_reqs:] = False
    seq_lens_cpu = self.optimistic_seq_lens_cpu[:num_reqs_padded]
    if self.use_async_spec_decode:
        # GPU tensors are authoritative in async mode.
        seq_lens_cpu = None
        num_computed_tokens_cpu = None

    req_doc_ranges = None
    if self.is_mm_prefix_lm:
        req_doc_ranges = {}
        hf_text_config = self.model_config.hf_text_config
        span_pad_modulus = getattr(
            hf_text_config,
            "mm_prefix_span_leading_pad_modulus",
            4 if getattr(hf_text_config, "vision_n_layers", 0) > 0 else 0,
        )
        for req_id in self.input_batch.req_ids:
            image_doc_ranges = []
            req_state = self.requests[req_id]
            for mm_feature in req_state.mm_features:
                if mm_feature.modality == "audio":
                    continue
                pos_info = mm_feature.mm_position
                if span_pad_modulus:
                    leading_pad = (
                        span_pad_modulus
                        - 1
                        - pos_info.offset % span_pad_modulus
                    )
                    image_doc_ranges.append(
                        (
                            pos_info.offset + leading_pad,
                            pos_info.offset + pos_info.length - 1,
                        )
                    )
                else:
                    image_doc_ranges.extend(
                        pos_info.extract_embeds_range()
                    )
            # Only track requests that actually carry image spans. Empty
            # lists would make mm_req_doc_ranges truthy and needlessly
            # trigger the vision SWA index build on text-only batches.
            if image_doc_ranges:
                req_idx = self.input_batch.req_id_to_index[req_id]
                req_doc_ranges[req_idx] = image_doc_ranges

    cm_base = AscendCommonAttentionMetadata(
        query_start_loc=self.query_start_loc.gpu[: num_reqs_padded + 1],
        query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs_padded + 1],
        seq_lens=self.seq_lens[:num_reqs_padded],
        # Always pass optimistic_seq_lens_cpu via _seq_lens_cpu so NPU
        # attention backends can get CPU seq_lens without GPU->CPU sync.
        # This is separate from seq_lens_cpu (None in async) which eagle
        # proposer checks to distinguish async/non-async behavior.
        _seq_lens_cpu=self.optimistic_seq_lens_cpu[:num_reqs_padded],
        seq_lens_cpu_upper_bound=self.optimistic_seq_lens_cpu[:num_reqs_padded],
        # TODO
        seq_lens_cpu=seq_lens_cpu,
        # TODO
        # num_computed_tokens_cpu=self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs_padded],
        num_computed_tokens_cpu=num_computed_tokens_cpu,
        num_reqs=num_reqs_padded,
        num_actual_tokens=num_tokens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        block_table_tensor=block_table_gid_0,
        slot_mapping=slot_mapping_gid_0,
        causal=True,
        is_prefilling=is_prefilling,
        num_input_tokens=num_tokens_padded,
        actual_seq_lengths_q=self.actual_seq_lengths_q,
        positions=self.positions,
        positions_cpu=self._dsa_positions_cpu_buf if self.use_compress else None,
        attn_state=self.attn_state,
        decode_token_per_req=self.decode_token_per_req,
        context_parallel_metadata=self.long_seq_metadata,
        # 0 for every request that is not replaying, and for every padded
        # row; a backend clamps the sliding-window lower bound of a
        # replaying request to it.
        replay_start=self._attn_replay_start(num_reqs_padded),
        req_ids_tensor=(
            self._offload_req_ids_tensor.gpu[:num_reqs_padded]
            if self._offload_req_ids_tensor is not None
            else None
        ),
        token_to_req=(
            self._offload_token_to_req.gpu[:num_tokens_padded]
            if self._offload_token_to_req is not None
            else None
        ),
        req_topk_buffer_slots=(self._offload_pool_slots.cpu[:num_reqs_padded]
                               if self._offload_pool_slots is not None else None),
        req_topk_buffer_active=(self._offload_pool_active.cpu[:num_reqs_padded]
                                     if self._offload_pool_active is not None else None),
        offload_dummy=offload_dummy,
        copy_sfa_restore_tails=self._copy_sfa_need_eager_tail_restore,
        mm_req_doc_ranges=req_doc_ranges,
    )

    if self.use_dcp:
        self.dcp_manager.prepare_common_attn_metadata(cm_base)

    if logits_indices is not None and self.cache_config.kv_sharing_fast_prefill:
        cm_base.num_logits_indices = logits_indices.size(0)
        cm_base.logits_indices_padded = self._prepare_kv_sharing_fast_prefill(logits_indices)

    def _build_attn_group_metadata(
        kv_cache_gid: int,
        attn_gid: int,
        common_attn_metadata: CommonAttentionMetadata,
        common_ratio_to_sas_metadata: dict,
        common_v41_metadata: dict,
        common_v41_batch_metadata: dict,
        ubid: int | None = None,
    ) -> None:
        attn_group = self.attn_groups[kv_cache_gid][attn_gid]
        builder = attn_group.get_metadata_builder(ubid or 0)
        is_gdn_noop = skip_gdn_state_update and isinstance(
            builder,
            GDNAttentionMetadataBuilder,
        )
        if is_gdn_noop:
            # Dummy-run callers coalesce this; fall back to the unpadded
            # batch size instead of asserting in the execute path.
            gdn_num_reqs = (
                num_reqs if num_reqs_padded is None else num_reqs_padded
            )
            # Idle DP dummy: keep captured GDN tensor ranks for graph
            # replay/collectives. num_actual_tokens=0 is the kernel no-op
            # (ops slice mixed_qkv[:0]); query_start_loc is a zero-filled
            # prefix sized to the graph, not a real token schedule.
            common_attn_metadata = replace(
                common_attn_metadata,
                query_start_loc=self.gdn_query_start_loc.gpu[
                    : gdn_num_reqs + 1
                ],
                query_start_loc_cpu=self.gdn_query_start_loc.cpu[
                    : gdn_num_reqs + 1
                ],
                num_actual_tokens=0,
                max_query_len=0,
                is_prefilling=(
                    torch.zeros_like(common_attn_metadata.is_prefilling)
                    if common_attn_metadata.is_prefilling is not None
                    else None
                ),
            )
        device_metadata_provider = (
            self.device_metadata_providers.get(id(builder))
            if self.device_metadata_providers is not None
            else None
        )
        cascade_attn_prefix_len = (
            cascade_attn_prefix_lens[kv_cache_gid][attn_gid] if cascade_attn_prefix_lens else 0
        )

        extra_attn_metadata_args: dict[str, Any] = {}
        if isinstance(builder, GDNAttentionMetadataBuilder) and not is_gdn_noop:
            assert ubid is None, "UBatching not supported with GDN yet"
            extra_attn_metadata_args["num_actual_reqs"] = num_reqs
            if use_spec_decode:
                extra_attn_metadata_args.update(
                    num_accepted_tokens=self.num_accepted_tokens.gpu[:num_reqs_padded],
                    num_decode_draft_tokens_cpu=self.num_decode_draft_tokens.cpu[:num_reqs_padded],
                )

        if isinstance(builder, AscendDSAMetadataBuilder):
            if for_cudagraph_capture:
                common_ratio_to_sas_metadata = {}
            extra_attn_metadata_args = dict(
                num_actual_reqs=num_reqs,
                common_ratio_to_sas_metadata=common_ratio_to_sas_metadata,
                full_graph_mode=cudagraph_runtime_mode == CUDAGraphMode.FULL,
            )
        elif isinstance(builder, v41_metadata_builder_type):
            extra_attn_metadata_args = dict(
                num_actual_reqs=num_reqs,
                skip_ring_state_update=skip_gdn_state_update,
                common_v41_metadata=common_v41_metadata,
                common_v41_batch_metadata=common_v41_batch_metadata,
                full_graph_mode=cudagraph_runtime_mode == CUDAGraphMode.FULL,
            )

        # GSA: pdmix (co-located P+D) and decode-side workers attach the
        # per-step GSA metadata; prefill-only producers stay dense.
        is_pdmix = self.vllm_config.kv_transfer_config is None or (
            self.vllm_config.kv_transfer_config.is_kv_producer
            and self.vllm_config.kv_transfer_config.is_kv_consumer
        )

        if (for_cudagraph_capture
                and not isinstance(builder, (
                    AscendDSAMetadataBuilder,
                    AscendSFADCPMetadataBuilder,
                    v41_metadata_builder_type,
                ))):
            attn_metadata_i = builder.build_for_cudagraph_capture(common_attn_metadata)
            if (
                hasattr(self.vllm_config.model_config, "hf_text_config")
                and hasattr(self.vllm_config.model_config.hf_text_config, "index_topk")
                and has_ucm_sparse()
                and (is_pdmix or not self.vllm_config.kv_transfer_config.is_kv_producer)
            ):
                attn_metadata_i.gsa_slot_len = common_attn_metadata.num_input_tokens
                attn_metadata_i.gsa_prefetch_len = common_attn_metadata.num_input_tokens
                attn_metadata_i.gsa_warmup_len = 0
                attn_metadata_i.gsa_swap_len = common_attn_metadata.num_input_tokens + 1
                if IS_INDEXER_OFFLOAD:
                    attn_metadata_i.indexer_last_query_len = (
                        common_attn_metadata.num_input_tokens
                    )
        else:
            attn_metadata_i = builder.build(
                common_prefix_len=cascade_attn_prefix_len,
                common_attn_metadata=common_attn_metadata,
                **extra_attn_metadata_args,
            )
        if device_metadata_provider is not None:
            assert device_metadata_tasks is not None
            device_metadata_tasks.extend(device_metadata_provider.take_device_metadata_tasks())
        if isinstance(builder, AscendDSAMetadataBuilder):
            common_ratio_to_sas_metadata = builder.common_ratio_to_sas_metadata  # type: ignore[assignment]

        # GSA: attach the per-step GSA/DSA metadata (allocated by the UCM
        # sparse runtime onto this runner) to every attention metadata
        # object built for a GSA-enabled layer set.
        if (
            hasattr(self.vllm_config.model_config, "hf_text_config")
            and hasattr(self.vllm_config.model_config.hf_text_config, "index_topk")
            and has_ucm_sparse()
            and (is_pdmix or not self.vllm_config.kv_transfer_config.is_kv_producer)
        ):
            attn_metadata_i.num_padding_reqs = common_attn_metadata.num_input_tokens
            # v0230 sfa_v1 _build: has_prefill = any(query_len >
            # decode_threshold). 0.30.0's builder dropped the field, but
            # gsa.py's _update_attn_gsa_lens (actual_sparse_lengths=None on
            # prefill steps) and the GSA SFA forward read it off this
            # metadata, so attach it here (query_start_loc_cpu covers the
            # padded batch; zero-length padding entries never exceed the
            # threshold).
            attn_metadata_i.has_prefill = bool(
                torch.any(
                    (
                        common_attn_metadata.query_start_loc_cpu[1:]
                        - common_attn_metadata.query_start_loc_cpu[:-1]
                    )
                    > getattr(builder, "decode_threshold", 1)
                )
            )
            attn_metadata_i.gsa_prefill_query = self.gsa_prefill_query
            attn_metadata_i.gsa_topk_win_tmp = self.gsa_topk_win_tmp
            # TODO
            attn_metadata_i.dsa_seq_len = self.dsa_seq_len.npu[
                : common_attn_metadata.num_reqs
            ]
            attn_metadata_i.dsa_block_table = self.dsa_block_table.npu[
                :, : common_attn_metadata.num_reqs
            ]
            attn_metadata_i.dsa_slot_mapping = self.dsa_slot_mapping.npu[
                : common_attn_metadata.num_input_tokens
            ]
            attn_metadata_i.gsa_prefetch_index = self.gsa_prefetch_index.npu[
                : common_attn_metadata.num_input_tokens
            ]
            attn_metadata_i.gsa_swap_index = self.gsa_swap_index.npu[
                : common_attn_metadata.num_input_tokens
            ]
            # actual_sparse_lengths: per-req 真实 topk 长度，传给
            # npu_kv_quant_sparse_flash_attention_gsa。kernel 按 bIdx=req 索引读取
            # （batchSize = len(cum_query_lens) = req 数），故按 num_reqs 切片
            # （req 域），勿用 num_input_tokens（token 域）。最终下发值由
            # GSA._update_attn_gsa_lens 按 has_prefill 决定（prefill 步置 None）。
            attn_metadata_i.actual_sparse_lengths = self.actual_sparse_lengths.npu[
                : common_attn_metadata.num_reqs
            ]
            attn_metadata_i.gsa_warmup_index = self.gsa_warmup_index.npu
            attn_metadata_i.gsa_prefetch_topk = self.gsa_prefetch_topk.npu[
                :, : common_attn_metadata.num_input_tokens
            ]
            attn_metadata_i.gsa_swap_topk = self.gsa_swap_topk.npu[
                :, : common_attn_metadata.num_input_tokens + 1
            ]
            attn_metadata_i.gsa_warmup_topk = self.gsa_warmup_topk.npu
            #### indexer offload
            if IS_INDEXER_OFFLOAD:
                attn_metadata_i.indexer_last_query = self.indexer_last_query.npu
                attn_metadata_i.indexer_last_q_index = self.indexer_last_q_index.npu
                attn_metadata_i.indexer_last_weight = self.indexer_last_weight.npu

        if ubid is None:
            assert isinstance(attn_metadata, dict)
            attn_metadata_dict = attn_metadata
        else:
            assert isinstance(attn_metadata, list)
            attn_metadata_dict = attn_metadata[ubid]

        for layer_name in attn_group.layer_names:
            attn_metadata_dict[layer_name] = attn_metadata_i

    # Prepare the attention metadata for each KV cache group and make layers
    # in the same group share the same metadata.
    common_ratio_to_sas_metadata: dict[Any, Any] = {}
    common_v41_batch_metadata: dict[str, Any] = {}
    spec_decode_common_attn_metadata = None
    for kv_cache_gid, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups):
        # V4.1 cache coordinates are shared only inside one framework KV
        # cache group. This lets a source's LongKV and Indexer reuse the
        # same [T, 2] mapping without aliasing any SWA group's mapping.
        common_v41_metadata: dict[str, Any] = {}
        cm = copy(cm_base)  # shallow copy
        # Basically only the encoder seq_lens, block_table and slot_mapping change
        # for each kv_cache_group.
        cm.encoder_seq_lens, cm.encoder_seq_lens_cpu = self._get_encoder_seq_lens(
            num_scheduled_tokens or {},
            kv_cache_group.kv_cache_spec,
            num_reqs_padded,
        )

        # Now, query_start_loc is padded.
        # But gdn needs an unpadded one.
        # gdn_query_start_loc is an unpadded version of query_start_loc.
        # TODO delete it if fia's check is removed.
        if self._has_gdn:
            attn_group = self.attn_groups[kv_cache_gid][0]
            builder = attn_group.get_metadata_builder(0)
            if isinstance(builder, GDNAttentionMetadataBuilder):
                cm.query_start_loc_cpu = self.gdn_query_start_loc.cpu[: num_reqs_padded + 1]
                cm.query_start_loc = self.gdn_query_start_loc.gpu[: num_reqs_padded + 1]

        if kv_cache_gid > 0:
            cm.block_table_tensor, cm.slot_mapping = _get_block_table_and_slot_mapping(
                kv_cache_gid
            )
        if self.speculative_config and isinstance(self.drafter, (AscendStep3p5MTPProposer, AscendDSparkProposer)):
            # step3p5 MTP draft layers span multiple KV cache groups; capture
            # each group's block table / slot mapping so the proposer can
            # build per-step attention metadata for the active MTP layer.
            self.drafter.set_per_group_attn_metadata(
                kv_cache_gid, cm.block_table_tensor, cm.slot_mapping)
        elif self.speculative_config and isinstance(self.drafter, AscendGemma4Proposer):
            self.drafter.set_per_group_block_table(kv_cache_gid, cm.block_table_tensor)
        if self.speculative_config and spec_decode_common_attn_metadata is None:
            if isinstance(
                self.drafter,
                AscendEagleProposer | AscendGemma4Proposer
                | AscendDraftModelProposer | AscendDflashProposer | AscendDSparkProposer,
            ):
                if self.drafter.attn_layer_names[0] in kv_cache_group.layer_names:
                    spec_decode_common_attn_metadata = cm
            elif isinstance(self.drafter, AscendExtractHiddenStatesProposer):
                if self.drafter.kv_cache_gid == kv_cache_gid:
                    spec_decode_common_attn_metadata = cm
            else:
                spec_decode_common_attn_metadata = cm
        for attn_gid in range(len(self.attn_groups[kv_cache_gid])):
            _build_attn_group_metadata(
                kv_cache_gid,
                attn_gid,
                cm,
                common_ratio_to_sas_metadata,
                common_v41_metadata,
                common_v41_batch_metadata,
            )
    if req_doc_ranges is not None:
        if isinstance(attn_metadata, list):
            for ub_metadata in attn_metadata:
                for _metadata in ub_metadata.values():
                    _metadata.mm_prefix_range = req_doc_ranges  # type: ignore[attr-defined]
        else:
            for _metadata in attn_metadata.values():
                _metadata.mm_prefix_range = req_doc_ranges  # type: ignore[attr-defined]

    if spec_decode_common_attn_metadata is not None and (
        num_reqs != num_reqs_padded or num_tokens != num_tokens_padded
    ):
        # Currently the drafter still only uses piecewise cudagraphs (and modifies
        # the attention metadata in directly), and therefore does not want to use
        # padded attention metadata.
        spec_decode_common_attn_metadata = spec_decode_common_attn_metadata.unpadded(num_tokens, num_reqs)
    if device_metadata_tasks:
        assert self.device_metadata_executor is not None
        self.device_metadata_executor.submit(
            device_metadata_tasks,
            batch_descriptor if cudagraph_runtime_mode == CUDAGraphMode.FULL else None,
        )
    self._maybe_eager_restore_copy_sfa_tails(attn_metadata)
    self._decoder_replay_common_metadata = cm_base
    return attn_metadata, spec_decode_common_attn_metadata


def maybe_execute_ucm_sparse_begin(
    self, scheduler_output: "SchedulerOutput", attn_metadata: AscendCommonAttentionMetadata
):
    from ucm.sparse.state import get_ucm_sparse, has_ucm_sparse

    if not has_ucm_sparse():
        return
    ucm_sparse = get_ucm_sparse()
    ucm_sparse.build_sparse_meta(
        scheduler_output, self.requests, self.input_batch, attn_metadata
    )
    ucm_sparse.execute_begin(scheduler_output, attn_metadata)


def maybe_execute_ucm_sparse_warmup(self):
    from ucm.sparse.state import get_ucm_sparse, has_ucm_sparse

    if not has_ucm_sparse():
        return
    ucm_sparse = get_ucm_sparse()
    ucm_sparse.warmup()


def maybe_execute_ucm_sparse_finished(self, logits_indices=None, rec_result=None):
    from ucm.sparse.state import get_ucm_sparse, has_ucm_sparse

    if not has_ucm_sparse():
        if logits_indices is not None:
            return logits_indices
        return None
    ucm_sparse = get_ucm_sparse()
    return ucm_sparse.execute_finished(logits_indices, rec_result)


def ucm_sparse_request_finished_in_worker(self, request_id: str | int):
    from ucm.sparse.state import get_ucm_sparse, has_ucm_sparse

    if not has_ucm_sparse():
        return
    ucm_sparse = get_ucm_sparse()
    ucm_sparse.request_finished_in_worker(request_id)


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


@when_imported("vllm_ascend.worker.v1.model_runner")
def _patch(mod):
    target = mod.NPUModelRunner
    for name, fn in (
        ("_prepare_inputs", _prepare_inputs),
        ("execute_model", execute_model),
        ("sample_tokens", sample_tokens),
        ("_build_attention_metadata", _build_attention_metadata),
    ):
        if hasattr(target, name):
            _patched = _make_patched(mod, fn)
            patch_or_inject(target, name, _patched)
            logger.info("UCM GSA NPUModelRunner.%s patch applied", name)
        else:
            logger.warning("UCM GSA NPUModelRunner.%s not found; skipped", name)

    for name, fn in (
        ("maybe_execute_ucm_sparse_begin", maybe_execute_ucm_sparse_begin),
        ("maybe_execute_ucm_sparse_warmup", maybe_execute_ucm_sparse_warmup),
        ("maybe_execute_ucm_sparse_finished", maybe_execute_ucm_sparse_finished),
        ("ucm_sparse_request_finished_in_worker", ucm_sparse_request_finished_in_worker),
    ):
        if not hasattr(target, name):
            _patched = _make_patched(mod, fn)
            patch_or_inject(target, name, _patched)
            logger.info("UCM GSA NPUModelRunner.%s injected", name)
