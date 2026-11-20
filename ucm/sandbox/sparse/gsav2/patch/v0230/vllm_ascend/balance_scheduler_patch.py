"""GSA scheduler patches for vLLM-Ascend balance scheduler (0.23.0).

This mirrors ``v0230/vllm/scheduler_patch.py`` but targets
``vllm_ascend.patch.platform.patch_balance_schedule.BalanceScheduler``
instead of the upstream vLLM ``Scheduler``.

What is already migrated in-source
-----------------------------------
Unlike ``RecomputeScheduler``, the in-tree ``BalanceScheduler.schedule`` has
*already* been wired with the bulk of the GSA sparse-attention integration:

* ``req_sparsed_slots: dict[str, int]`` is populated for both the RUNNING and
  WAITING loops.
* ``num_slots_sparsed`` is estimated via ``self.ucm_sparse.estimate_num_slots_sparsed``
  and threaded into ``allocate_slots(num_slots_sparsed=...)`` in both loops.
* ``req_sparsed_slots`` is passed to the emitted ``SchedulerOutput``.

So this patch does **not** re-implement those pieces.

What this patch adds
--------------------
The one GSA hook that ``BalanceScheduler.schedule`` is missing is the per-step
free-slot allocator reset:

    if has_ucm_sparse() and PTOPK_PREFETCH_ENABLE:
        get_ucm_sparse().update_scheduler_free_slot(0)

In the base vLLM ``Scheduler.schedule`` (already patched by
``v0230/vllm/scheduler_patch.py``) this reset runs at the top of every step so
``alloc_block_sch`` (the scheduler-side reservation counter consumed by
``estimate_free_slots_sparse``) does not drift across steps.

``BalanceScheduler.schedule`` only calls ``super().schedule()`` on the
*balance-disabled / no-preemption* short-circuit path.  On the
*balance-enabled* custom path (the whole point of this scheduler) it runs its
own body and never invokes ``super().schedule()``, so the reset is skipped and
``alloc_block_sch`` accumulates indefinitely -> GSA over-rejects prefills on
the decode node (false ``gsa_preempted_flag``).

We add the reset as a thin wrapper around the original ``schedule`` so it runs
before both paths.  Calling ``update_scheduler_free_slot(0)`` twice on the
balance-disabled path (once here, once inside ``super().schedule()``) is
harmless: the implementation is ``if num == 0: self.alloc_block_sch = 0``,
i.e. idempotent.

Remaining (intentionally un-migrated) deltas
--------------------------------------------
These gaps from the reference ``scheduler_patch.schedule`` are left as-is on
the balance scheduler because they are inert under its typical DP usage and
migrating them would re-type the entire 500-line ``schedule`` body for no
behavioral gain:

* ``reserved_blocks = self._inflight_prefill_reserved_blocks()`` + the
  ``self._inflight_prefills`` in-flight-prefill tracking set.  These only
  matter for async KV-connector loads (``load_kv_async=True``).  The balance
  scheduler does not perform async loads, so ``reserved_blocks`` would always
  be 0 and ``_inflight_prefills`` always empty.
* ``new_block_ids_to_zero`` plumbing into ``SchedulerOutput``.  Only active
  when ``needs_kv_cache_zeroing`` is true; not on by default.
* ``effective_lookahead_tokens`` uses the balance scheduler's existing
  ``0 if request.num_computed_tokens == 0 else self.num_lookahead_tokens``
  rather than the upstream ``0 if (load_kv_async and self.use_eagle) else
  self.num_lookahead_tokens``.  Since ``load_kv_async`` is never true here the
  two are equivalent for this scheduler.

See the accompanying review write-up for the full gap analysis.
"""

from __future__ import annotations

from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


def _make_gsa_aware_schedule(orig_schedule):
    """Wrap ``BalanceScheduler.schedule`` with the GSA free-slot reset.

    Returns a replacement ``schedule`` that:
      1. Resets the GSA scheduler-side free-slot allocator
         (``update_scheduler_free_slot(0)``) at the top of every step.
      2. Delegates to the original balance ``schedule`` unchanged.

    The reset is guarded by ``has_ucm_sparse() and PTOPK_PREFETCH_ENABLE`` to
    match the base vLLM ``Scheduler.schedule`` and the worker
    ``NPUModelRunner`` GSA execute path (both use the same guard).
    """

    def schedule(self, throttle_prefills: bool = False):
        from ucm.sparse.state import get_ucm_sparse, has_ucm_sparse
        from ucm.sparse.utils import PTOPK_PREFETCH_ENABLE

        if has_ucm_sparse() and PTOPK_PREFETCH_ENABLE:
            get_ucm_sparse().update_scheduler_free_slot(0)
        return orig_schedule(self, throttle_prefills)

    schedule.__annotations__ = orig_schedule.__annotations__
    schedule.__kwdefaults__ = getattr(orig_schedule, "__kwdefaults__", None)
    return schedule


@when_imported("vllm_ascend.patch.platform.patch_balance_schedule")
def _patch(mod):
    """Add the missing GSA free-slot reset to ``BalanceScheduler.schedule``.

    ``BalanceScheduler`` already inherits GSA ``__init__`` / ``_preempt_request``
    / ``_free_request`` / ``update_from_output`` / ``add_request`` hooks from
    the base ``Scheduler`` (which is itself patched by
    ``v0230/vllm/scheduler_patch.py``).  Only its overridden ``schedule``
    bypasses the per-step reset, so that is all we touch here.

    ``BalanceDPEngineCoreProc`` and the module-level ``run_engine_core`` /
    ``EngineCoreProc.run_engine_core`` / ``vllm...Scheduler`` rebinds at the
    bottom of the patched module are left untouched.
    """
    target = mod.BalanceScheduler
    if hasattr(target, "schedule"):
        replacement = _make_gsa_aware_schedule(target.schedule)
        patch_or_inject(target, "schedule", replacement)
        logger.info(
            "UCM GSA BalanceScheduler.schedule free-slot-reset patch applied"
        )
