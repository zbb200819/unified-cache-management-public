"""Inject ``reset_row`` into vLLM 0.32.0 ``BlockTable`` / ``MultiGroupBlockTable``.

vLLM 0.32.0 (like 0.30.0) has no ``reset_row`` in favor of
``add_row``/``clear_row``. The GSA sparse request path
(``GPUModelRunner._update_states``) resets a request's row and re-appends its
(re-mapped) block ids, so the method is re-injected with the v0.23.0
semantics: zero the row across the GPU/CPU/NP views and reset the per-row
block counter. The 0.32.0 ``BlockTable`` keeps the same
``block_table.gpu/cpu/np`` views and ``num_blocks_per_row`` counter, so the
v0.30.0 injection applies verbatim.
"""

from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


def _block_table_reset_row(self, row_idx: int) -> None:
    self.num_blocks_per_row[row_idx] = 0
    self.block_table.gpu[row_idx].fill_(0)
    self.block_table.cpu[row_idx].fill_(0)
    self.block_table.np[row_idx].fill(0)


def _multi_group_block_table_reset_row(self, row_idx: int) -> None:
    for _i, block_table in enumerate(self.block_tables):
        block_table.reset_row(row_idx)


@when_imported("vllm.v1.worker.block_table")
def _patch_vllm_block_table(mod):
    if not hasattr(mod, "BlockTable"):
        logger.warning("Skip vLLM block_table reset_row patch: BlockTable missing")
        return
    patch_or_inject(mod.BlockTable, "reset_row", _block_table_reset_row)
    if hasattr(mod, "MultiGroupBlockTable"):
        patch_or_inject(
            mod.MultiGroupBlockTable,
            "reset_row",
            _multi_group_block_table_reset_row,
        )
    logger.info("UCM GSA vLLM BlockTable.reset_row patch applied")
