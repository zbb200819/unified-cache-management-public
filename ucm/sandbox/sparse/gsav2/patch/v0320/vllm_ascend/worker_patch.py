"""GSA NPUWorker patches for vLLM-Ascend 0.32.0.

Rebased from the 0.30.0 GSA patch layer onto the vLLM-Ascend 0.32.0 NPUWorker.
GSA semantics (final, per v0230):
- ``init_device``: initialize the worker-side sparse DSA state
  (``maybe_init_ucm_sparse_dsa``) after the model runner is created, gated on
  the model having an indexer (``index_topk``) and sparse state being enabled.
- ``_init_worker_distributed_environment``: initialize the sparse state after
  the distributed environment is up (``ensure_ucm_sparse_initialized``).
- Injected helper: ``maybe_init_ucm_sparse_dsa``.

``execute_model`` is intentionally NOT patched: the 0.23.0 GSA delta for it
was pure re-indentation with no semantic change (it only dropped a fork-side
PerfCounters timing line), so the upstream body is used as-is.

Upstream 0.32.0 changes accounted for: the v2 model-runner branch of
``init_device`` now routes ``is_mm_encoder_only`` configs through
``MMEncoderModelRunner`` under ``torch_cuda_wrapper``.
"""

from __future__ import annotations

from types import FunctionType

from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


def init_device(self):
    from ucm.sparse.state import has_ucm_sparse

    # NOTE: KEEP device the member of `NPUWorker`, as it will be checked
    # in ray scenario. see https://github.com/vllm-project/vllm/pull/26845
    # for more details
    self.device = self._init_device()
    # FLA import-time capability queries must run on the selected worker
    # device, and registration must finish before any model is constructed.
    # TODO: Drop this separate call when supported vLLM FLA or Triton Ascend
    # versions fix import-time NPU init and general registration is restored.
    register_ascend_fla_customop()
    # Initialize workspace manager
    num_ubatches = 1
    init_workspace_manager(self.device, num_ubatches)
    # Init ModelRunner here, so that we have access to self.device.
    if self.use_v2_model_runner:
        logger.warning("npu model runner v2 is in developing, some features doesn't work for now.")
        if self.vllm_config.is_mm_encoder_only:
            from vllm.v1.worker.mm_encoder_model_runner import MMEncoderModelRunner

            from vllm_ascend.worker.v2.utils import torch_cuda_wrapper

            with torch_cuda_wrapper():
                self.model_runner = MMEncoderModelRunner(self.vllm_config, self.device)
        else:
            from vllm_ascend.worker.v2.model_runner import NPUModelRunner as NPUModelRunnerV2

            self.model_runner = NPUModelRunnerV2(self.vllm_config, self.device)
    else:
        self.model_runner = NPUModelRunner(self.vllm_config, self.device)

    if (
        hasattr(self.vllm_config.model_config, "hf_text_config")
        and hasattr(self.vllm_config.model_config.hf_text_config, "index_topk")
        and has_ucm_sparse()
    ):
        self.maybe_init_ucm_sparse_dsa(self.model_runner)

    if self.rank == 0:
        # If usage stat is enabled, collect relevant info.
        report_usage_stats(self.vllm_config)


def _init_worker_distributed_environment(self) -> None:
    """Initialize the distributed environment."""
    from ucm.sparse.state import ensure_ucm_sparse_initialized

    init_batch_invariance()
    init_distributed_environment(
        self.parallel_config.world_size, self.rank, self.distributed_init_method, self.local_rank, "hccl"
    )
    group_config = self.vllm_config
    if group_config.engram_config is not None and self.parallel_config.prefill_context_parallel_size > 1:
        # Skip upstream's DP-only Engram group so Ascend can create
        # _ENGRAM_DP once with DP x PCP ranks. Copy the config to preserve
        # the original Engram settings for Ascend and model initialization.
        group_config = copy.copy(group_config)
        group_config.engram_config = None
    with set_current_vllm_config(group_config):
        ensure_model_parallel_initialized(
            self.parallel_config.tensor_parallel_size,
            self.parallel_config.pipeline_parallel_size,
            self.parallel_config.prefill_context_parallel_size,
            self.parallel_config.decode_context_parallel_size,
        )
    init_ascend_model_parallel(self.parallel_config, self.vllm_config.engram_config)
    ensure_ec_transfer_initialized(self.vllm_config)
    ensure_ucm_sparse_initialized(self.vllm_config)


def maybe_init_ucm_sparse_dsa(self, model_runner):
    from ucm.sparse.state import get_ucm_sparse, has_ucm_sparse

    if not has_ucm_sparse():
        return
    ucm_sparse = get_ucm_sparse()
    ucm_sparse.init_dsa_data(model_runner)


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


@when_imported("vllm_ascend.worker.worker")
def _patch(mod):
    if hasattr(mod.NPUWorker, "init_device"):
        _patch_method(mod, mod.NPUWorker, "init_device", init_device)
        logger.info("UCM GSA NPUWorker.init_device patch applied")

    if hasattr(mod.NPUWorker, "_init_worker_distributed_environment"):
        _patch_method(
            mod,
            mod.NPUWorker,
            "_init_worker_distributed_environment",
            _init_worker_distributed_environment,
        )
        logger.info(
            "UCM GSA NPUWorker._init_worker_distributed_environment patch applied"
        )

    if not hasattr(mod.NPUWorker, "maybe_init_ucm_sparse_dsa"):
        _patch_method(
            mod,
            mod.NPUWorker,
            "maybe_init_ucm_sparse_dsa",
            maybe_init_ucm_sparse_dsa,
        )
        logger.info("UCM GSA NPUWorker.maybe_init_ucm_sparse_dsa injected")
