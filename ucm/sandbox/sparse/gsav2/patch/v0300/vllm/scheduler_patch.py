"""GSA scheduler patches for vLLM 0.30.0.

Rebased from the 0.23.0 GSA patch layer onto the vLLM 0.30.0 Scheduler.
GSA semantics (final, per v0230):
- ``__init__``: load a tokenizer and initialize the SCHEDULER-role ucm sparse
  state (``ensure_ucm_sparse_initialized``).
- ``schedule``: reset the scheduler free-slot counter each step
  (``update_scheduler_free_slot(0)``), estimate ``num_slots_sparsed`` per
  request (running + waiting), pass it to
  ``kv_cache_manager.allocate_slots`` and forward ``req_sparsed_slots`` via
  ``SchedulerOutput``.
- ``_preempt_request``: release the preempted request's sparse slots
  (``update_scheduler_free_slot(-blocks)``).
- ``add_request``: notify sparse state on request admission
  (``request_begin``).
- ``_free_request``: notify sparse state on request completion
  (``request_finished_in_scheduler``) plus debug prints of the generated text.

``update_from_output`` is intentionally NOT patched: the 0.23.0 GSA delta for
it was pure re-indentation with no semantic change.
"""

from __future__ import annotations

from types import FunctionType

from vllm.multimodal import MULTIMODAL_REGISTRY

from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


def __init__(
    self,
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    structured_output_manager: StructuredOutputManager,
    block_size: int,
    hash_block_size: int | None = None,
    mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    include_finished_set: bool = False,
    log_stats: bool = False,
) -> None:
    from ucm.sparse.base import UcmSparseRole
    from ucm.sparse.state import (
        ensure_ucm_sparse_initialized,
        get_ucm_sparse,
        has_ucm_sparse,
    )
    from ucm.sparse.utils import IS_OPEN_GSA

    self.vllm_config = vllm_config
    self.scheduler_config = vllm_config.scheduler_config
    self.cache_config = vllm_config.cache_config
    self.lora_config = vllm_config.lora_config
    self.model_uses_mrope = vllm_config.model_config.uses_mrope
    self.kv_cache_config = kv_cache_config
    self.kv_events_config = vllm_config.kv_events_config
    self.parallel_config = vllm_config.parallel_config
    self.log_stats = log_stats
    self.observability_config = vllm_config.observability_config
    self.spec_decode_metrics_level = (
        self.observability_config.per_request_spec_decode_metrics
    )
    self.kv_metrics_collector: KVCacheMetricsCollector | None = None
    if self.observability_config.kv_cache_metrics:
        self.kv_metrics_collector = KVCacheMetricsCollector(
            self.observability_config.kv_cache_metrics_sample,
        )
    self.structured_output_manager = structured_output_manager
    self.is_encoder_decoder = vllm_config.model_config.is_encoder_decoder
    self.is_mm_encoder_only = vllm_config.is_mm_encoder_only

    # include_finished_set controls whether a separate set of finished
    # request ids should be included in the EngineCoreOutputs returned
    # by update_from_outputs(). This is currently used in the multi-engine
    # case to track request lifetimes efficiently.
    self.finished_req_ids_dict: dict[int, set[str]] | None = (
        defaultdict(set) if include_finished_set else None
    )
    # Track requests scheduled in prior step (MRV1-only).
    self.prev_step_scheduled_req_ids: set[str] = set()

    # Scheduling constraints.
    self.max_num_running_reqs = self.scheduler_config.max_num_seqs
    self.max_num_scheduled_tokens = (
        self.scheduler_config.max_num_scheduled_tokens
        if self.scheduler_config.max_num_scheduled_tokens is not None
        else self.scheduler_config.max_num_batched_tokens
    )
    self.max_model_len = vllm_config.model_config.max_model_len
    self.enable_kv_cache_events = (
        self.kv_events_config is not None
        and self.kv_events_config.enable_kv_cache_events
    )
    # Diffusion models may not sample any tokens for a denoising step.
    self.num_sampled_tokens_per_step = (
        1 if not vllm_config.model_config.is_diffusion else 0
    )

    # Create KVConnector for the Scheduler. Note that each Worker
    # will have a corresponding KVConnector with Role=WORKER.
    # KV Connector pushes/pull of remote KVs for P/D and offloading.
    self.connector = None

    self.ucm_sparse = None
    print("tokenizer path:", self.vllm_config.model_config.model, flush=True)
    self.tokenizer = AutoTokenizer.from_pretrained(
        self.vllm_config.model_config.model,
        use_chat_template=True,
    )

    if IS_OPEN_GSA:
        ensure_ucm_sparse_initialized(vllm_config, role=UcmSparseRole.SCHEDULER)
        if has_ucm_sparse():
            self.ucm_sparse = get_ucm_sparse()
            logger.info(
                "UCM Sparse initialized successfully: {}".format(self.ucm_sparse)
            )

    self.connector_prefix_cache_stats: PrefixCacheStats | None = None
    self.recompute_kv_load_failures = True
    self.defer_block_free = False
    # Whether a preempted request's in-flight output must be dropped; see
    # KVConnectorBase_V1.requires_kv_delivery.
    self.requires_kv_delivery = False
    kv_transfer_config = self.vllm_config.kv_transfer_config
    if kv_transfer_config is not None:
        assert not self.is_encoder_decoder, (
            "Encoder-decoder models are not currently supported with KV connectors"
        )
        self.connector = KVConnectorFactory.create_connector(
            config=self.vllm_config,
            role=KVConnectorRole.SCHEDULER,
            kv_cache_config=self.kv_cache_config,
        )
        if self.log_stats:
            self.connector_prefix_cache_stats = PrefixCacheStats()
        kv_load_failure_policy = kv_transfer_config.kv_load_failure_policy
        self.recompute_kv_load_failures = kv_load_failure_policy == "recompute"

        # With overlapping batches (async scheduling or PP), a step may
        # still be writing a freed request's KV blocks. A consumer KV
        # Connector can reallocate and fill those blocks via a load that
        # isn't ordered against that write, so defer freeing them.
        multiple_inflight_batches = self.vllm_config.max_concurrent_batches > 1
        if multiple_inflight_batches and kv_transfer_config.is_kv_consumer:
            self.defer_block_free = True

        self.requires_kv_delivery = self.connector.requires_kv_delivery

    self.kv_event_publisher = EventPublisherFactory.create(
        self.kv_events_config,
        self.parallel_config.data_parallel_index,
    )
    self.ec_connector = None
    if self.vllm_config.ec_transfer_config is not None:
        self.ec_connector = ECConnectorFactory.create_connector(
            config=self.vllm_config, role=ECConnectorRole.SCHEDULER
        )

    num_gpu_blocks = self.cache_config.num_gpu_blocks
    assert num_gpu_blocks is not None and num_gpu_blocks > 0

    self.block_size = block_size
    self.dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
    self.pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size

    # req_id -> Request
    self.requests: dict[str, Request] = {}
    # Scheduling policy
    try:
        self.policy = SchedulingPolicy(self.scheduler_config.policy)
    except ValueError as e:
        raise ValueError(
            f"Unknown scheduling policy: {self.scheduler_config.policy}"
        ) from e
    # Priority queues for requests.
    self.waiting = create_request_queue(self.policy)
    # requests skipped in waiting flow due async deps or constraints.
    self.skipped_waiting = create_request_queue(self.policy)
    self.running: list[Request] = []

    # The request IDs that are finished in between the previous and the
    # current steps. This is used to notify the workers about the finished
    # requests so that they can free the cached states for those requests.
    # This is flushed at the end of each scheduling step.
    self.finished_req_ids: set[str] = set()

    # IDs of requests preempted since the last call to schedule().
    self.reset_preempted_req_ids: set[str] = set()

    # Counter for requests waiting for streaming input. Used to calculate
    # number of unfinished requests
    self.num_waiting_for_streaming_input: int = 0

    # KV Connector: requests in process of async KV loading or recving
    self.finished_recving_kv_req_ids: set[str] = set()
    self.failed_recving_kv_req_ids: set[str] = set()

    # Grammar compilation failures to finish as per-request errors in
    # update_from_output.
    self.grammar_compile_error_reqs: set[str] = set()

    # Encoder-related.
    # Calculate encoder cache size if applicable
    supports_mm_inputs = mm_registry.supports_multimodal_inputs(
        vllm_config.model_config
    )
    mm_budget = (
        MultiModalBudget(vllm_config, mm_registry) if supports_mm_inputs else None
    )

    # NOTE: Text-only encoder-decoder models are implemented as
    # multi-modal models for convenience
    # Example: https://github.com/vllm-project/bart-plugin
    if self.is_encoder_decoder:
        assert mm_budget and len(mm_budget.mm_max_toks_per_item) <= 1, (
            "Encoder-decoder models are expected to implement the "
            "multimodal interface with at most one modality."
        )

    self.max_num_encoder_input_tokens = (
        mm_budget.encoder_compute_budget if mm_budget else 0
    )
    encoder_cache_size = mm_budget.encoder_cache_size if mm_budget else 0
    manager_cls_obj = vllm_config.ec_manager_config.get_encoder_cache_manager_obj()
    if manager_cls_obj is None:
        manager_cls_obj = (
            EncoderDecoderCacheManager
            if self.is_encoder_decoder
            else EncoderCacheManager
        )
    self.encoder_cache_manager = manager_cls_obj.create_manager(
        cache_size=encoder_cache_size, vllm_config=vllm_config
    )
    speculative_config = vllm_config.speculative_config
    self.use_eagle = False
    self.use_eagle_block_drop = False
    self.num_spec_tokens = vllm_config.num_speculative_tokens
    self.num_lookahead_tokens = vllm_config.num_lookahead_tokens
    # Positions past the computed tokens that the drafter reads mid-prefill.
    # Eagle-family drafters read 1 ahead, but multi-module MTP reads
    # num_spec_tokens ahead at chunked-prefill boundaries. Determines the
    # encoder scheduling shift, the deferred encoder free, the KV cache
    # manager's re-prefillable window (this minus 1), and how many tokens to
    # reserve between a chunk boundary and the prefill end.
    self.num_prefill_lookahead = 0
    self.dynamic_sd_lookup: list[int] | None = None
    if speculative_config is not None:
        if speculative_config.num_speculative_tokens_per_batch_size:
            self.dynamic_sd_lookup = build_dynamic_sd_schedule_lookup(
                speculative_config.num_speculative_tokens_per_batch_size,
                vllm_max_batch_size=self.scheduler_config.max_num_seqs,
                vllm_num_speculative_tokens=self.num_spec_tokens,
            )
        self.use_eagle = speculative_config.use_eagle()
        if self.use_eagle:
            self.num_prefill_lookahead = (
                self.num_spec_tokens
                if speculative_config.use_multi_module_mtp()
                else 1
            )
        self.use_eagle_block_drop = speculative_config.use_eagle_block_drop()
        if self.use_eagle and not self.use_eagle_block_drop:
            logger.warning(
                "EAGLE trailing prefix-cache block dropping is disabled. "
                "This is experimental and may affect speculative-token "
                "acceptance rates."
            )

    # Create the KV cache manager.
    if hash_block_size is None:
        hash_block_size = block_size
    self.hash_block_size = hash_block_size
    self.kv_cache_manager = KVCacheManager(
        kv_cache_config=kv_cache_config,
        max_model_len=self.max_model_len,
        max_in_flight_tokens=vllm_config.max_in_flight_tokens,
        enable_caching=self.cache_config.enable_prefix_caching,
        use_eagle=self.use_eagle_block_drop,
        num_prefill_lookahead=self.num_prefill_lookahead,
        log_stats=self.log_stats,
        enable_kv_cache_events=self.enable_kv_cache_events,
        dcp_world_size=self.dcp_world_size,
        pcp_world_size=1,
        scheduler_block_size=self.block_size,
        hash_block_size=hash_block_size,
        metrics_collector=self.kv_metrics_collector,
        watermark=self.scheduler_config.watermark,
        enable_mamba_fine_grained_prefix_cache=(
            self.cache_config.enable_mamba_fine_grained_prefix_cache
        ),
    )
    # Bind after construction so connectors can access the cache manager.
    if self.connector is not None:
        self.connector.bind_kv_cache_manager(self.kv_cache_manager)

    self.use_pp = self.parallel_config.pipeline_parallel_size > 1
    self.use_v2_model_runner = vllm_config.use_v2_model_runner
    # Scheduler iteration counter. Drives the V2+PP+async decode-throttle
    # cadence (`next_decode_eligible_step`).
    self.current_step = 0
    # DP prefill balancing: Flag to track whether the last cadence-aligned
    # prefill batch fully drained the waiting queue. Prefill throttling
    # is disabled in this case.
    self.prefill_capacity_bound = False
    self.scheduler_reserve_full_isl = (
        self.scheduler_config.scheduler_reserve_full_isl
    )

    self.has_mamba_layers = kv_cache_config.has_mamba_layers
    self.needs_kv_cache_zeroing = kv_cache_config.needs_kv_cache_zeroing
    # Blocks that async KV loads will overwrite this step, skipped from
    # zeroing since the zeroing could race the out-of-band write.
    self._skip_zero_block_ids: set[int] = set()
    self.need_mamba_block_aligned_split = (
        self.has_mamba_layers and self.cache_config.mamba_cache_mode == "align"
    )
    # TODO: Support models with multiple Mamba specs that require different
    # prefill checkpoint alignments instead of selecting the first one.
    self.mamba_prefill_checkpoint_alignment = next(
        (
            group.kv_cache_spec.prefill_checkpoint_alignment
            for group in kv_cache_config.kv_cache_groups
            if isinstance(group.kv_cache_spec, MambaSpec)
        ),
        None,
    )
    self.mamba_has_prefill_checkpoint_blocks = self.has_mamba_layers and all(
        not isinstance(group.kv_cache_spec, MambaSpec)
        or group.kv_cache_spec.num_prefill_checkpoint_blocks > 0
        for group in kv_cache_config.kv_cache_groups
    )
    # A finer prefix_match_unit is configured: a mamba partial tail entry
    # can only be registered by a step ending exactly at the prompt's last
    # hash boundary, so the split adds that stop.
    self.mamba_partial_cache_hit = (
        self.need_mamba_block_aligned_split
        and self.hash_block_size < self.block_size
        and self.kv_cache_manager.coordinator.enable_partial_hash_hits
    )
    # Opt-in: also stop at the junction, where an eagle sibling resumes. The
    # manager decides whether it can check-point there (per-group eagle bit,
    # no MTP re-prefill tail); splitting for a stop it would refuse costs a
    # forward pass and displaces the block-boundary stop.
    self.mamba_fine_grained_prefix_cache = (
        self.mamba_partial_cache_hit
        and self.kv_cache_manager.mamba_fine_grained_prefix_cache
    )

    # Counts of non-empty steps scheduled / processed. update_from_output
    # is called once per scheduled step in FIFO order, so these stay in sync.
    self.sched_step_seq = 0
    self.processed_step_seq = 0
    # FIFO of (fence_seq, blocks): blocks become safe to free once
    # processed_step_seq >= fence_seq.
    self.deferred_frees: deque[tuple[int, list[KVCacheBlock]]] = deque()

    self.perf_metrics: ModelMetrics | None = None
    if self.log_stats and vllm_config.observability_config.enable_mfu_metrics:
        self.perf_metrics = ModelMetrics(vllm_config)

    self.enable_return_routed_experts = (
        vllm_config.model_config.enable_return_routed_experts
    )
    self.return_sampling_mask = vllm_config.model_config.return_sampling_mask

    if self.enable_return_routed_experts:
        assert self.dcp_world_size == 1 and self.pcp_world_size == 1, (
            "enable_return_routed_experts does not support context parallelism "
            "(dcp_world_size > 1 or pcp_world_size > 1)"
        )

        self.routed_experts_mgr = RoutedExpertsManager(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
        )
        # Block-ID snapshot taken at schedule time (before forward),
        # so update_from_output can read slot data even if a later
        # schedule() frees the blocks (async scheduling race).
        self._re_block_ids: dict[str, list[int]] = {}

    self._pause_state: PauseState = PauseState.UNPAUSED

    # In-flight requests still prefilling (prefill chunks + in-progress
    # async KV loads). Their remaining-block reservation gates async loads.
    self._inflight_prefills: set[Request] = set()


def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
    from ucm.sparse.base import INVALID_SLOT
    from ucm.sparse.state import get_ucm_sparse, has_ucm_sparse

    self.current_step += 1
    # NOTE(woosuk) on the scheduling algorithm:
    # There's no "decoding phase" nor "prefill phase" in the scheduler.
    # Each request just has the num_computed_tokens and
    # num_tokens_with_spec. num_tokens_with_spec =
    # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
    # At each step, the scheduler tries to assign tokens to the requests
    # so that each request's num_computed_tokens can catch up its
    # num_tokens_with_spec. This is general enough to cover
    # chunked prefills, prefix caching, speculative decoding,
    # and the "jump decoding" optimization in the future.

    scheduled_new_reqs: list[Request] = []
    scheduled_resumed_reqs: list[Request] = []
    scheduled_running_reqs: list[Request] = []
    preempted_reqs: list[Request] = []

    req_to_new_blocks: dict[str, KVCacheBlocks] = {}
    num_scheduled_tokens: dict[str, int] = {}
    token_budget = self.max_num_scheduled_tokens
    spec = self.vllm_config.speculative_config
    draft_slots = spec.max_num_new_slots_for_drafting if spec is not None else 0
    input_budget = self.scheduler_config.max_num_batched_tokens
    if self._pause_state == PauseState.PAUSED_ALL:
        # Do not schedule any requests when paused.
        token_budget = 0

    if has_ucm_sparse():
        get_ucm_sparse().update_scheduler_free_slot(0)

    # Encoder-related.
    scheduled_encoder_inputs: dict[str, list[int]] = {}
    encoder_compute_budget = self.max_num_encoder_input_tokens
    # Spec decode-related.
    scheduled_spec_decode_tokens: dict[str, list[int]] = {}
    # Whether the running batch contains any prefill requests.
    prefill_scheduled = False
    # Whether any scheduled request has a synchronous connector KV load.
    has_sync_kv_loads = False

    # For logging.
    scheduled_timestamp = time.monotonic()

    self.kv_cache_manager.new_step_starts()

    # DP prefill balancing: on a throttled (non-cadence-aligned) step, defer
    # all prefill compute unless saturated.
    defer_prefills = (
        throttle_prefills and not self.prefill_capacity_bound
    ) and any(not r.is_prefill_chunk for r in self.running)

    # First, schedule the RUNNING requests.
    req_index = 0
    req_sparsed_slots: dict[str, int] = {}
    while req_index < len(self.running) and token_budget > 0:
        request = self.running[req_index]
        if input_budget <= draft_slots:
            break

        num_slots_sparsed = INVALID_SLOT
        if self.ucm_sparse:
            num_slots_sparsed = self.ucm_sparse.estimate_num_slots_sparsed(
                request, self.num_lookahead_tokens
            )
        req_sparsed_slots.update({request.request_id: num_slots_sparsed})

        if (
            request.num_output_placeholders > 0
            # This is (num_computed_tokens + 1) - (num_output_placeholders - 1).
            # Since output placeholders are also included in the computed tokens
            # count, we subtract (num_output_placeholders - 1) to remove any draft
            # tokens, so that we can be sure no further steps are needed even if
            # they are all rejected.
            and request.num_computed_tokens + 2 - request.num_output_placeholders
            >= request.num_prompt_tokens + request.max_tokens
        ):
            # Async scheduling: Avoid scheduling an extra step when we are sure that
            # the previous step has reached request.max_tokens. We don't schedule
            # partial draft tokens since this prevents uniform decode optimizations.
            req_index += 1
            continue

        if self.current_step < request.next_decode_eligible_step:
            # V2+PP+async: enforce `pp_size` steps between same-req decodes
            # to match worker-side sampled-tokens broadcast slot ring cadence.
            req_index += 1
            continue

        if defer_prefills and request.is_prefill_chunk:
            # DP prefill balancing: defer this in-progress prefill chunk to a
            # cadence-aligned step; decodes still run to fill this step.
            req_index += 1
            continue

        if (
            self.ec_connector is not None
            and request.mm_features
            and not self.ec_connector.ensure_cache_available(
                request,
                request.num_computed_tokens - request.num_output_placeholders,
            )
        ):
            req_index += 1
            continue

        num_new_tokens = (
            request.num_tokens_with_spec
            + request.num_output_placeholders
            - request.num_computed_tokens
        )
        if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
            num_new_tokens = self.scheduler_config.long_prefill_token_threshold
        num_new_tokens = min(
            num_new_tokens, token_budget, input_budget - draft_slots
        )

        # Make sure the input position does not exceed the max model len.
        # This is necessary when using spec decoding.
        num_new_tokens = min(
            num_new_tokens,
            self.max_model_len
            - request.num_computed_tokens
            - self.num_sampled_tokens_per_step,
        )

        # Apply Mamba alignment before encoder caps.
        if self.need_mamba_block_aligned_split:
            num_new_tokens = self._mamba_block_aligned_split(
                request, num_new_tokens
            )

        # Schedule encoder inputs.
        encoder_inputs_to_schedule = None
        external_load_encoder_input: list[int] = []
        new_encoder_compute_budget = encoder_compute_budget
        if request.has_encoder_inputs:
            (
                encoder_inputs_to_schedule,
                num_new_tokens,
                new_encoder_compute_budget,
                external_load_encoder_input,
            ) = self._try_schedule_encoder_inputs(
                request,
                request.num_computed_tokens,
                num_new_tokens,
                encoder_compute_budget,
                shift_computed_tokens=self.num_prefill_lookahead,
            )

        # Multi-module MTP: avoid ending a prefill chunk within
        # num_prefill_lookahead of the prefill end.
        num_new_tokens = self._reserve_prefill_lookahead(
            request, request.num_computed_tokens, num_new_tokens
        )

        if num_new_tokens == 0:
            # The request cannot be scheduled because one of the following
            # reasons:
            # 1. No new tokens to schedule. This may happen when
            #    (1) PP>1 and we have already scheduled all prompt tokens
            #    but they are not finished yet.
            #    (2) Async scheduling and the request has reached to either
            #    its max_total_tokens or max_model_len.
            # 2. The encoder budget is exhausted.
            # 3. The encoder cache is exhausted.
            # 4. Insufficient budget for a block-aligned chunk in hybrid
            #    models with mamba cache mode \"align\".
            # 5. Insufficient budget to keep a multi-module MTP prefill
            #    chunk out of the prefill-lookahead window.
            # NOTE(woosuk): Here, by doing `continue` instead of `break`,
            # we do not strictly follow the FCFS scheduling policy and
            # allow the lower-priority requests to be scheduled.
            req_index += 1
            continue

        # Schedule newly needed KV blocks for the request.
        with record_function_or_nullcontext("schedule: allocate_slots"):
            while True:
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_lookahead_tokens=self.num_lookahead_tokens,
                    num_slots_sparsed=num_slots_sparsed,
                )

                if new_blocks is not None:
                    # The request can be scheduled.
                    break

                if (
                    self.connector is not None
                    and self.connector.has_pending_block_frees()
                ):
                    break

                # The request cannot be scheduled.
                # Preempt the lowest-priority request.
                if self.policy == SchedulingPolicy.PRIORITY:
                    preempted_req = max(
                        self.running,
                        key=lambda r: (r.priority, r.arrival_time),
                    )
                else:
                    preempted_req = self.running[-1]

                # A deferred free will not help with immediate allocation.
                if not self._request_blocks_can_be_freed(preempted_req):
                    break

                if self.policy == SchedulingPolicy.PRIORITY:
                    victim_index = self.running.index(preempted_req)
                    del self.running[victim_index]
                    if victim_index < req_index:
                        req_index -= 1

                    if preempted_req in scheduled_running_reqs:
                        preempted_req_id = preempted_req.request_id
                        scheduled_running_reqs.remove(preempted_req)
                        restored = num_scheduled_tokens.pop(preempted_req_id)
                        token_budget += restored
                        input_budget += restored + draft_slots
                        req_to_new_blocks.pop(preempted_req_id)
                        scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                        preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                            preempted_req_id, None
                        )
                        if preempted_encoder_inputs:
                            # Restore encoder compute budget if the preempted
                            # request had encoder inputs scheduled in this step.
                            num_embeds_to_restore = sum(
                                preempted_req.get_num_encoder_embeds(i)
                                for i in preempted_encoder_inputs
                            )
                            encoder_compute_budget += num_embeds_to_restore
                else:
                    preempted_req = self.running.pop()

                self._preempt_request(
                    preempted_req,
                    scheduled_timestamp,
                    drop_stale_output=self.requires_kv_delivery,
                )
                preempted_reqs.append(preempted_req)
                if preempted_req == request:
                    # No more request to preempt. Cannot schedule this request.
                    break

        if new_blocks is None:
            # Cannot schedule this request.
            break

        # Schedule the request.
        scheduled_running_reqs.append(request)
        prefill_scheduled |= request.is_prefill_chunk
        request_id = request.request_id
        req_to_new_blocks[request_id] = new_blocks
        num_scheduled_tokens[request_id] = num_new_tokens
        token_budget -= num_new_tokens
        input_budget -= num_new_tokens + draft_slots
        req_index += 1

        # Speculative decode related.
        if request.spec_token_ids:
            num_scheduled_spec_tokens = (
                num_new_tokens
                + request.num_computed_tokens
                - request.num_tokens
                - request.num_output_placeholders
            )
            if num_scheduled_spec_tokens > 0:
                spec_token_ids = request.spec_token_ids
                if len(spec_token_ids) > num_scheduled_spec_tokens:
                    spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
                scheduled_spec_decode_tokens[request.request_id] = spec_token_ids

            # New spec tokens will be set in `update_draft_token_ids` before the
            # next step when applicable.
            request.spec_token_ids = []

        # Encoder-related.
        if encoder_inputs_to_schedule:
            scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
            # Allocate the encoder cache.
            for i in encoder_inputs_to_schedule:
                self.encoder_cache_manager.allocate(request, i)
                if self.ec_connector is not None:
                    self.ec_connector.update_state_after_alloc(request, i)
            encoder_compute_budget = new_encoder_compute_budget
        if external_load_encoder_input:
            for i in external_load_encoder_input:
                self.encoder_cache_manager.allocate(request, i)
                if self.ec_connector is not None:
                    self.ec_connector.update_state_after_alloc(request, i)

    # Record the LoRAs in scheduled_running_reqs
    scheduled_loras: set[int] = set()
    if self.lora_config:
        scheduled_loras = set(
            req.lora_request.lora_int_id
            for req in scheduled_running_reqs
            if req.lora_request and req.lora_request.lora_int_id > 0
        )
        assert len(scheduled_loras) <= self.lora_config.max_loras

    # Next, schedule the WAITING requests.
    if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
        step_skipped_waiting = create_request_queue(self.policy)

        while (self.waiting or self.skipped_waiting) and token_budget > 0:
            if input_budget <= draft_slots:
                break
            # Paused streaming sessions (WAITING_FOR_STREAMING_REQ) are not
            # in `running` but still hold a model-runner request slot.
            num_running = len(self.running) + self.num_waiting_for_streaming_input
            if num_running >= self.max_num_running_reqs:
                break

            request_queue = self._select_waiting_queue_for_scheduling()
            assert request_queue is not None

            request = request_queue.peek_request()

            num_slots_sparsed = INVALID_SLOT  ####  suit for skipped
            if self.ucm_sparse:
                num_slots_sparsed = self.ucm_sparse.estimate_num_slots_sparsed(
                    request, self.num_lookahead_tokens
                )
            req_sparsed_slots.update({request.request_id: num_slots_sparsed})

            request_id = request.request_id

            # try to promote blocked statuses while traversing skipped queue.
            if self._is_blocked_waiting_status(
                request.status
            ) and not self._try_promote_blocked_waiting_request(request):
                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                    logger.debug(
                        "%s is still in WAITING_FOR_REMOTE_KVS state.",
                        request_id,
                    )
                request_queue.pop_request()
                step_skipped_waiting.prepend_request(request)
                continue

            if (
                request.num_stale_output_tokens > 0
                and not request.drop_stale_output
            ):
                # Deliverable stale output still in flight: resuming now
                # could resample a position that output later delivers.
                # It drains within the pipeline depth.
                request_queue.pop_request()
                step_skipped_waiting.prepend_request(request)
                continue

            # Check that adding the request still respects the max_loras
            # constraint.
            if (
                self.lora_config
                and request.lora_request
                and (
                    len(scheduled_loras) == self.lora_config.max_loras
                    and request.lora_request.lora_int_id not in scheduled_loras
                )
            ):
                # Scheduling would exceed max_loras, skip.
                request_queue.pop_request()
                step_skipped_waiting.prepend_request(request)
                continue

            num_external_computed_tokens = 0
            load_kv_async = False
            connector_prefix_cache_queries, connector_prefix_cache_hits = 0, 0
            did_prefix_cache_lookup = False

            # Get already-cached tokens.
            if request.num_computed_tokens == 0:
                did_prefix_cache_lookup = True
                (
                    new_computed_blocks,
                    num_new_local_computed_tokens,
                    request.shared_prefix_boundary,
                    hit_diverged,
                ) = self._get_local_prefix_cache_hit(request)

                # Get externally-cached tokens if using a KVConnector.
                if self.connector is not None:
                    # Present a block-aligned local hit to the connector so
                    # a strictly longer remote hit can supersede a local
                    # sub-block tail without racing its copy-on-write.
                    partial_tail = num_new_local_computed_tokens % self.block_size
                    block_aligned_local = (
                        num_new_local_computed_tokens - partial_tail
                    )
                    ext_tokens, load_kv_async = (
                        self.connector.get_num_new_matched_tokens(
                            request, block_aligned_local
                        )
                    )

                    if ext_tokens is None:
                        # The request cannot be scheduled because
                        # the KVConnector couldn't determine
                        # the number of matched tokens.
                        request_queue.pop_request()
                        step_skipped_waiting.prepend_request(request)
                        continue

                    if partial_tail and ext_tokens > partial_tail:
                        # Remote strictly exceeds the full local hit: drop the
                        # sub-block tail so no CoW is needed, and let the load
                        # cover it. Trim the partial block out of the local
                        # computed blocks so it is not adopted from the cache.
                        new_computed_blocks = (
                            self.kv_cache_manager.truncate_computed_blocks(
                                new_computed_blocks, block_aligned_local
                            )
                        )
                        num_new_local_computed_tokens = block_aligned_local
                        num_external_computed_tokens = ext_tokens
                    elif partial_tail:
                        # Remote does not exceed the full local hit: keep the
                        # local sub-block tail and load nothing external.
                        num_external_computed_tokens = 0
                        # Nothing to load remotely -> not an async-load step;
                        # clearing avoids the `load_kv_async` assert below.
                        load_kv_async = False
                    else:
                        num_external_computed_tokens = ext_tokens

                    if hit_diverged and num_external_computed_tokens == 0:
                        # No external tokens back the deeper local hit, so its
                        # resume boundary would have no valid Mamba state.
                        # Reconcile to the boundary every group agrees on.
                        (
                            new_computed_blocks,
                            num_new_local_computed_tokens,
                            request.shared_prefix_boundary,
                        ) = self.kv_cache_manager.get_computed_blocks(request)

                    connector_prefix_cache_queries = (
                        request.num_tokens - num_new_local_computed_tokens
                    )
                    connector_prefix_cache_hits = num_external_computed_tokens

                # Total computed tokens (local + external).
                num_computed_tokens = (
                    num_new_local_computed_tokens + num_external_computed_tokens
                )
                assert num_computed_tokens <= request.num_tokens

                # Skip request with pending mm encoding prefetches
                if self._ec_transfer_pending(request, num_computed_tokens):
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                # Track first scheduled prefill, not post-preemption repeat prefills
                if request.prefill_stats and request.num_preemptions <= 0:
                    assert num_computed_tokens <= request.num_prompt_tokens
                    request.prefill_stats.set(
                        num_prompt_tokens=request.num_prompt_tokens,
                        num_local_cached_tokens=num_new_local_computed_tokens,
                        num_external_cached_tokens=num_external_computed_tokens,
                    )
            else:
                # KVTransfer: WAITING reqs have num_computed_tokens > 0
                # after async KV recvs are completed. A streaming-input
                # session resumes here too, carrying whatever media its
                # latest chunk added, so this branch needs the same gate.
                new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                num_new_local_computed_tokens = 0
                num_computed_tokens = request.num_computed_tokens

                if self._ec_transfer_pending(request, num_computed_tokens):
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

            encoder_inputs_to_schedule = None
            external_load_encoder_input = []
            new_encoder_compute_budget = encoder_compute_budget
            pad_spec_decode = False

            if load_kv_async:
                # KVTransfer: loading remote KV, do not allocate for new work.
                assert num_external_computed_tokens > 0
                num_new_tokens = 0
            elif defer_prefills and num_computed_tokens < request.num_tokens - 1:
                # DP prefill balancing: defer this step's local prefill
                # compute to a cadence-aligned step.
                break
            else:
                request_token_budget = min(token_budget, input_budget - draft_slots)
                # Number of tokens to be scheduled.
                # We use `request.num_tokens` instead of
                # `request.num_prompt_tokens` to consider the resumed
                # requests, which have output tokens.
                num_new_tokens = request.num_tokens - num_computed_tokens

                # Pad new decode requests to uniform spec decoding size to
                # preserve full cudagraph for this step.
                # Not for diffusion where draft tokens can't be padded.
                if (
                    (self.num_spec_tokens > 0 and self.dynamic_sd_lookup is None)
                    and self.num_sampled_tokens_per_step > 0
                    and num_new_tokens == 1
                    and not prefill_scheduled
                    and (scheduled_running_reqs or num_computed_tokens > 0)
                ):
                    padded_num_tokens = 1 + self.num_spec_tokens
                    # Pad only when there is room for the sampled token(s).
                    if (
                        num_computed_tokens
                        + padded_num_tokens
                        + self.num_sampled_tokens_per_step
                        <= self.max_model_len
                    ):
                        if padded_num_tokens > request_token_budget:
                            # Prefer to not schedule than schedule un-padded.
                            break
                        num_new_tokens = padded_num_tokens
                        pad_spec_decode = True

                threshold = self.scheduler_config.long_prefill_token_threshold
                if 0 < threshold < num_new_tokens:
                    num_new_tokens = threshold

                # chunked prefill has to be enabled explicitly to allow
                # pooling requests to be chunked
                if (
                    not self.scheduler_config.enable_chunked_prefill
                    and num_new_tokens > request_token_budget
                ):
                    # If chunked_prefill is disabled,
                    # we can stop the scheduling here.
                    break

                num_new_tokens = min(num_new_tokens, request_token_budget)
                assert num_new_tokens > 0

                # Apply Mamba alignment before encoder caps.
                if self.need_mamba_block_aligned_split:
                    num_new_tokens = self._mamba_block_aligned_split(
                        request,
                        num_new_tokens,
                        num_new_local_computed_tokens,
                        num_external_computed_tokens,
                    )
                    if num_new_tokens == 0:
                        break
                    if (
                        pad_spec_decode
                        and num_new_tokens != 1 + self.num_spec_tokens
                    ):
                        # Alignment clipped the placeholder rows. The split
                        # aligns prefill chunks, but the padded tail rows are
                        # speculative positions, not prefill tokens. A padded
                        # request must keep all 1 + num_spec rows or the
                        # sampler's row count stops matching its query rows,
                        # so drop the padding instead of shortening it.
                        num_new_tokens = 1
                        pad_spec_decode = False

                # Schedule encoder inputs.
                if request.has_encoder_inputs:
                    (
                        encoder_inputs_to_schedule,
                        num_new_tokens,
                        new_encoder_compute_budget,
                        external_load_encoder_input,
                    ) = self._try_schedule_encoder_inputs(
                        request,
                        num_computed_tokens,
                        num_new_tokens,
                        encoder_compute_budget,
                        shift_computed_tokens=self.num_prefill_lookahead,
                    )

                # Multi-module MTP: avoid ending a prefill chunk within
                # num_prefill_lookahead of the prefill end.
                num_new_tokens = self._reserve_prefill_lookahead(
                    request, num_computed_tokens, num_new_tokens
                )

                if num_new_tokens == 0:
                    # The request cannot be scheduled.
                    break

            # During async KV load, no forward pass is run yet.
            # Allocate speculative lookahead slots later to avoid
            # mismatching local and remote block counts.
            limit_lookahead_tokens = load_kv_async and self.num_lookahead_tokens > 0
            effective_lookahead_tokens = (
                0 if limit_lookahead_tokens else self.num_lookahead_tokens
            )

            # Determine if we need to allocate cross-attention blocks.
            num_encoder_tokens = 0
            if (
                self.is_encoder_decoder
                and request.has_encoder_inputs
                and encoder_inputs_to_schedule
            ):
                num_encoder_tokens = sum(
                    request.get_num_encoder_embeds(i)
                    for i in encoder_inputs_to_schedule
                )

            reserved_blocks = 0
            if load_kv_async:
                # An async load holds its blocks for the whole transfer with
                # no forward progress and isn't preemptible here. Admit it
                # only if it fits in (free - other in-flight reservations), to
                # avoid deadlock and predictable preemptions.
                reserved_blocks = self._inflight_prefill_reserved_blocks()

            new_blocks = self.kv_cache_manager.allocate_slots(
                request,
                num_new_tokens,
                num_new_computed_tokens=num_new_local_computed_tokens,
                new_computed_blocks=new_computed_blocks,
                num_lookahead_tokens=effective_lookahead_tokens,
                num_external_computed_tokens=num_external_computed_tokens,
                delay_cache_blocks=load_kv_async,
                num_encoder_tokens=num_encoder_tokens,
                full_sequence_must_fit=self.scheduler_reserve_full_isl,
                num_slots_sparsed=num_slots_sparsed,
                reserved_blocks=reserved_blocks,
                has_scheduled_reqs=bool(self.running),
            )

            if new_blocks is None:
                # The request cannot be scheduled.

                # NOTE: we need to untouch the request from the encode cache
                # manager
                if request.has_encoder_inputs:
                    self.encoder_cache_manager.free(request)
                break

            # KVTransfer: the connector uses this info to determine
            # if a load is needed. Note that
            # This information is used to determine if a load is
            # needed for this request.
            if self.connector is not None:
                self.connector.update_state_after_alloc(
                    request,
                    self.kv_cache_manager.get_blocks(request_id),
                    num_external_computed_tokens,
                )
                if (
                    self.connector_prefix_cache_stats is not None
                    and connector_prefix_cache_queries != 0
                ):
                    self.connector_prefix_cache_stats.record(
                        num_tokens=connector_prefix_cache_queries,
                        num_hits=connector_prefix_cache_hits,
                        preempted=request.num_preemptions > 0,
                    )

            # Record at admission so unscheduled lookups are not counted.
            if did_prefix_cache_lookup:
                self.kv_cache_manager.record_prefix_cache_stats(
                    request, num_new_local_computed_tokens
                )

            request = request_queue.pop_request()
            if load_kv_async:
                # If loading async, allocate memory and put request
                # into the WAITING_FOR_REMOTE_KV state.
                request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                step_skipped_waiting.prepend_request(request)
                # Set num_computed_tokens even though KVs are not yet loaded.
                # request.num_computed_tokens will not be used anywhere until
                # the request finished the KV transfer.
                #
                # If a transfer error is reported by the connector,
                # request.num_computed_tokens will be re-set accordingly in
                # _update_requests_with_invalid_blocks.
                #
                # When the transfer is finished, either successfully or not,
                # request.num_computed_tokens will correctly reflect the number
                # of computed tokens.
                # _update_waiting_for_remote_kv will then cache
                # only the successfully loaded tokens.
                request.num_computed_tokens = num_computed_tokens
                self._inflight_prefills.add(request)
                if self.needs_kv_cache_zeroing:
                    # Skip zeroing of the blocks the async load will
                    # overwrite; the zeroing could race the write.
                    self._skip_zero_block_ids.update(
                        self.kv_cache_manager.get_zeroing_block_ids_in_range(
                            request.request_id,
                            num_new_local_computed_tokens,
                            num_computed_tokens,
                        )
                    )
                continue

            self.running.append(request)
            if num_external_computed_tokens > 0:
                # load_kv_async is False here
                has_sync_kv_loads = True
            if self.log_stats:
                request.record_event(
                    EngineCoreEventType.SCHEDULED, scheduled_timestamp
                )
            if request.status == RequestStatus.WAITING:
                scheduled_new_reqs.append(request)
            elif request.status == RequestStatus.PREEMPTED:
                scheduled_resumed_reqs.append(request)
            else:
                raise RuntimeError(f"Invalid request status: {request.status}")

            if self.lora_config and request.lora_request:
                scheduled_loras.add(request.lora_request.lora_int_id)
            req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(
                request_id
            )
            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            input_budget -= num_new_tokens + draft_slots
            request.status = RequestStatus.RUNNING
            request.num_computed_tokens = num_computed_tokens
            if pad_spec_decode:
                assert num_new_tokens == 1 + self.num_spec_tokens
                scheduled_spec_decode_tokens[request_id] = [
                    -1
                ] * self.num_spec_tokens
            # Only track requests that will still be prefilling after this chunk.
            if num_computed_tokens + num_new_tokens < request.num_tokens:
                self._inflight_prefills.add(request)
            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)
                encoder_compute_budget = new_encoder_compute_budget
            # Allocate for external load encoder cache
            if external_load_encoder_input:
                for i in external_load_encoder_input:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)

        # re-queue requests skipped in this pass ahead of older skipped items.
        if step_skipped_waiting:
            self.skipped_waiting.prepend_requests(step_skipped_waiting)

        # DP prefill balancing: on a step that admitted prefills (release),
        # record whether it was capacity-bound.
        if not defer_prefills:
            self.prefill_capacity_bound = bool(self.waiting)

    # Check if the scheduling constraints are satisfied.
    total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
    assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens

    assert token_budget >= 0
    assert input_budget >= 0
    assert len(self.running) <= self.max_num_running_reqs
    # Since some requests in the RUNNING queue may not be scheduled in
    # this step, the total number of scheduled requests can be smaller than
    # len(self.running).
    assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(
        scheduled_running_reqs
    ) <= len(self.running)

    # Get the longest common prefix among all requests in the running queue.
    # This can be potentially used for cascade attention.
    num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
    with record_function_or_nullcontext("schedule: get_num_common_prefix_blocks"):
        if self.running:
            any_request_id = self.running[0].request_id
            num_common_prefix_blocks = (
                self.kv_cache_manager.get_num_common_prefix_blocks(any_request_id)
            )

    # Construct the scheduler output.
    if self.use_v2_model_runner:
        scheduled_new_reqs.extend(scheduled_resumed_reqs)
        scheduled_resumed_reqs.clear()
        new_reqs_data = [
            NewRequestData.from_request(
                req,
                req_to_new_blocks[req.request_id].get_block_ids(),
                req._all_token_ids,
                uses_mrope=self.model_uses_mrope,
            )
            for req in scheduled_new_reqs
        ]
    else:
        new_reqs_data = [
            NewRequestData.from_request(
                req,
                req_to_new_blocks[req.request_id].get_block_ids(),
                uses_mrope=self.model_uses_mrope,
            )
            for req in scheduled_new_reqs
        ]

    with record_function_or_nullcontext("schedule: make_cached_request_data"):
        cached_reqs_data = self._make_cached_request_data(
            scheduled_running_reqs,
            scheduled_resumed_reqs,
            num_scheduled_tokens,
            scheduled_spec_decode_tokens,
            req_to_new_blocks,
        )

    # Record the request ids that were scheduled in this step (MRV1-only).
    if not self.use_v2_model_runner:
        self.prev_step_scheduled_req_ids.clear()
        self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())

    # Mamba "align" boundary states must be handed off with exact block ids;
    # they cannot be reconstructed from a connector's append-only block
    # table. Drained every step so stale offers cannot accumulate.
    boundary_state_offloads = self.kv_cache_manager.take_boundary_state_offloads()

    kv_connector_block_state = None
    if self.connector is not None:
        # Any request scheduled this step can become a connector job now,
        # not only the ones that were allocated blocks: a store save lands
        # on the step that fills a block, which allocated none.
        block_state_req_ids = set(num_scheduled_tokens)
        block_state_req_ids.update(
            req_id for req_id in boundary_state_offloads if req_id in self.requests
        )
        kv_connector_block_state = KVConnectorBlockState(
            req_ids=block_state_req_ids,
            resolve_block_ids=self.kv_cache_manager.get_block_ids,
            boundary_state_offloads=boundary_state_offloads,
        )

    kv_cache_block_copies, cow_retained_blocks = (
        self.kv_cache_manager.take_kv_cache_block_copies()
    )
    if kv_cache_block_copies:
        # The copies run with this step's execution; the first non-empty
        # step at or after it gets seq `sched_step_seq + 1` (0-token steps
        # do not advance the seq), and its completion implies the copies
        # have run.
        self._free_cow_retained_blocks(cow_retained_blocks, self.sched_step_seq + 1)
    pending_kv_cache_block_copies = kv_cache_block_copies or None

    # Dynamic speculative decoding: compute optimal K
    num_spec_tokens_to_schedule = self.num_spec_tokens
    if self.dynamic_sd_lookup is not None and len(num_scheduled_tokens) > 0:
        num_spec_tokens_to_schedule = self.dynamic_sd_lookup[
            len(num_scheduled_tokens)
        ]

    scheduled_encoder_input_stats = None
    if (
        self.log_stats
        and self.observability_config.enable_logging_iteration_details
    ):
        scheduled_encoder_input_stats = self._make_scheduled_encoder_input_stats(
            scheduled_encoder_inputs
        )

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=new_reqs_data,
        scheduled_cached_reqs=cached_reqs_data,
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=total_num_scheduled_tokens,
        scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
        scheduled_encoder_inputs=scheduled_encoder_inputs,
        scheduled_encoder_input_stats=scheduled_encoder_input_stats,
        num_common_prefix_blocks=num_common_prefix_blocks,
        req_sparsed_slots=req_sparsed_slots,
        preempted_req_ids=self.reset_preempted_req_ids,
        # finished_req_ids is an existing state in the scheduler,
        # instead of being newly scheduled in this step.
        # It contains the request IDs that are finished in between
        # the previous and the current steps.
        finished_req_ids=self.finished_req_ids,
        free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
        new_block_ids_to_zero=self._get_new_block_ids_to_zero(),
        has_sync_kv_loads=has_sync_kv_loads,
        kv_cache_block_copies=pending_kv_cache_block_copies,
        kv_connector_block_state=kv_connector_block_state,
        num_spec_tokens_to_schedule=num_spec_tokens_to_schedule,
        ec_manager_metadata=self.encoder_cache_manager.get_manager_metadata(),
    )

    # NOTE(Kuntai): this function is designed for multiple purposes:
    # 1. Plan the KV cache store
    # 2. Wrap up all the KV cache load / save ops into an opaque object
    # 3. Clear the internal states of the connector
    if self.connector is not None:
        meta = self._build_kv_connector_meta(self.connector, scheduler_output)
        scheduler_output.kv_connector_metadata = meta

    # Build the connector meta for ECConnector
    if self.ec_connector is not None:
        ec_meta: ECConnectorMetadata = self.ec_connector.build_connector_meta(
            scheduler_output
        )
        scheduler_output.ec_connector_metadata = ec_meta

    # Connector-only block state must not be dispatched to workers.
    scheduler_output.kv_connector_block_state = None

    # Advance the fence only for non-empty steps (those that actually
    # write KV and have their output processed later in update_from_output).
    if self.defer_block_free and total_num_scheduled_tokens > 0:
        self.sched_step_seq += 1

    with record_function_or_nullcontext("schedule: update_after_schedule"):
        self._update_after_schedule(scheduler_output)
    return scheduler_output


def _preempt_request(
    self, request: Request, timestamp: float, drop_stale_output: bool = False
) -> None:
    """Preempt a request and put it back to the waiting queue.

    NOTE: The request should be popped from the running queue outside of this
    method.

    drop_stale_output: drop (rather than deliver) any in-flight output; used
    by reset_prefix_cache, whose same-step resume would otherwise deliver
    tokens out of order, and for connectors with a pending KV hand-off,
    which the preemption's block free would leave without valid KV.
    """
    from ucm.sparse.state import get_ucm_sparse, has_ucm_sparse

    assert request.status == RequestStatus.RUNNING, (
        "Only running requests can be preempted"
    )
    if has_ucm_sparse():
        get_ucm_sparse().update_scheduler_free_slot(
            -1 * len(request._all_token_ids) // self.cache_config.block_size
        )
    self._free_request_blocks(request)
    self.encoder_cache_manager.free(request)
    self._inflight_prefills.discard(request)
    request.status = RequestStatus.PREEMPTED
    request.num_computed_tokens = 0
    if request.spec_token_ids:
        request.spec_token_ids = []
    # Async scheduling: mark all in-flight output as stale. Its tokens are
    # still delivered on return (dropping them would perturb spec-decode
    # acceptance) but must not mutate the reset counters; each step drains
    # its share in update_from_output. num_in_flight_tokens already
    # includes any undrained stale share, so assign rather than accumulate.
    # An undrained drop-mode share stays dropped: its positions have
    # already been resampled.
    request.drop_stale_output = drop_stale_output or (
        request.drop_stale_output and request.num_stale_output_tokens > 0
    )
    request.num_stale_output_tokens = request.num_in_flight_tokens
    request.num_output_placeholders = 0
    request.num_preemptions += 1
    if self.log_stats:
        request.record_event(EngineCoreEventType.PREEMPTED, timestamp)

    # Put the request back to the waiting queue.
    self.waiting.prepend_request(request)
    self.reset_preempted_req_ids.add(request.request_id)


def add_request(self, request: Request) -> None:
    existing = self.requests.get(request.request_id)

    if self.ucm_sparse:
        self.ucm_sparse.request_begin(request.request_id, request.prompt_token_ids)

    if existing is not None:
        update = StreamingUpdate.from_request(request)
        if existing.status != RequestStatus.WAITING_FOR_STREAMING_REQ:
            assert existing.streaming_queue is not None, "duplicate request id"
            # Queue next input chunk (or finished sentinel).
            existing.streaming_queue.append(update)
        elif update is not None:
            # Commence next input chunk.
            self._update_request_as_session(existing, update)
        else:
            # Streaming-input session finished.
            self.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    else:
        if request.resumable:
            request.streaming_queue = deque()
        self._enqueue_waiting_request(request)
        self.requests[request.request_id] = request
        if self.spec_decode_metrics_level != "none":
            request.spec_decode_metrics = RequestSpecDecodeMetrics.new(
                self.num_spec_tokens
            )
        if self.connector is not None:
            self.connector.on_new_request(request)
        if self.log_stats:
            request.record_event(EngineCoreEventType.QUEUED)


def _free_request(
    self, request: Request, delay_free_blocks: bool = False
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    assert request.is_finished()

    print(
        "------------------------------------------------------------------------------------------------",
        flush=True,
    )
    print(
        f"[GSA] out tokens for req_id {request.request_id} and prompt len {len(request.prompt_token_ids)} out_put len {len(request._output_token_ids)}",
        flush=True,
    )
    print(
        f"{self.tokenizer.decode(request._output_token_ids, skip_special_tokens=False)}",
        flush=True,
    )
    print(
        "------------------------------------------------------------------------------------------------",
        flush=True,
    )
    if self.ucm_sparse:
        self.ucm_sparse.request_finished_in_scheduler(request.request_id)

    self._inflight_prefills.discard(request)
    connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)

    # EC Connector: mirror the KV hook. The contract requires firing
    # before the encoder cache is freed so the connector can inspect
    # per-request state (e.g. which mm_hashes it recorded during
    # save_caches()) and emit ec_transfer_params for the response body.
    ec_xfer_params: dict[str, Any] | None = None
    if self.ec_connector is not None:
        ec_delay_free, ec_xfer_params = self.ec_connector.request_finished(request)
        connector_delay_free_blocks |= ec_delay_free

    self.encoder_cache_manager.free(request)
    request_id = request.request_id
    self.finished_req_ids.add(request_id)
    if self.finished_req_ids_dict is not None:
        self.finished_req_ids_dict[request.client_index].add(request_id)

    delay_free_blocks |= connector_delay_free_blocks
    if not delay_free_blocks:
        self._free_blocks(request)

    return kv_xfer_params, ec_xfer_params


def _patch_method(mod, cls, name, fn) -> None:
    _patched = FunctionType(
        fn.__code__,
        vars(mod),
        name=fn.__name__,
        argdefs=fn.__defaults__,
        closure=fn.__closure__,
    )
    _patched.__annotations__ = fn.__annotations__
    _patched.__kwdefaults__ = fn.__kwdefaults__
    patch_or_inject(cls, name, _patched)


@when_imported("vllm.v1.core.sched.scheduler")
def _patch(mod):
    try:
        from transformers import AutoTokenizer

        if not hasattr(mod, "AutoTokenizer"):
            setattr(mod, "AutoTokenizer", AutoTokenizer)
    except Exception as e:
        logger.warning(f"GSA scheduler AutoTokenizer injection failed {e}")

    if hasattr(mod.Scheduler, "__init__"):
        _patch_method(mod, mod.Scheduler, "__init__", __init__)
        logger.info("UCM GSA Scheduler.__init__ patch applied")

    if hasattr(mod.Scheduler, "schedule"):
        _patch_method(mod, mod.Scheduler, "schedule", schedule)
        logger.info("UCM GSA Scheduler.schedule patch applied")

    if hasattr(mod.Scheduler, "_preempt_request"):
        _patch_method(mod, mod.Scheduler, "_preempt_request", _preempt_request)
        logger.info("UCM GSA Scheduler._preempt_request patch applied")

    if hasattr(mod.Scheduler, "add_request"):
        _patch_method(mod, mod.Scheduler, "add_request", add_request)
        logger.info("UCM GSA Scheduler.add_request patch applied")

    if hasattr(mod.Scheduler, "_free_request"):
        _patch_method(mod, mod.Scheduler, "_free_request", _free_request)
        logger.info("UCM GSA Scheduler._free_request patch applied")
