"""FA4 CuTe block-sparse attention on SM90 (H100 / H200 / GH200).

Upstream FA4 implements block sparsity on Hopper natively, so this backend
runs our fork's Hopper kernels (`veda_comfy/kernels/fa4`), which are
upstream 4.0.0b32 untouched: everything we changed is on the SM8x path.

The kernels take full / partial index lists instead of a dense mask: key
tiles without padding are "full" (no per-token mask_mod), tiles with
padding slots are "partial" (mask_mod reads the slot validity). Masking
every block would cost ~25-30% for nothing.

Self-contained on purpose (see backends/base.py).
"""

from __future__ import annotations

import functools
import threading

import torch

from . import base
from ..core import tiling

_MAJOR = 9
_CALL_LOCK = threading.Lock()


@functools.cache
def _modules():
    import cutlass  # pylint: disable=import-outside-toplevel
    import cutlass.cute as cute  # pylint: disable=import-outside-toplevel
    from ..kernels.fa4 import block_sparsity  # pylint: disable=import-outside-toplevel
    from ..kernels.fa4 import interface  # pylint: disable=import-outside-toplevel
    from ..kernels.fa4 import utils  # pylint: disable=import-outside-toplevel
    return cutlass, cute, block_sparsity, interface, utils


@functools.cache
def _valid_key_mask_mod():
    """Singleton mask_mod (FA4 keys its compile cache on the callable)."""
    cutlass, cute, _, _, utils = _modules()

    @cute.jit
    def valid_key(batch, head, m_idx, n_idx, seqlen_info, aux_tensors):
        del batch, head, m_idx, seqlen_info
        slot_valid = aux_tensors[0]
        valid = utils.scalar_to_ssa(slot_valid[n_idx[0]], cutlass.Int32)
        zero = utils.scalar_to_ssa(0, cutlass.Int32)
        return valid != zero

    return valid_key


def _pack(member: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-packed column indices of member [H', R, C] -> (cnt, idx)."""
    order = torch.argsort((~member).to(torch.int8), dim=-1, stable=True)
    return (member.sum(-1)[None].to(torch.int32),
            order[None].to(torch.int32).contiguous())


def index_lists(block_mask, layout):
    """(partial_cnt, partial_idx, full_cnt, full_idx) in
    BlockSparseTensorsTorch order; cnt [1, H', R], idx [1, H', R, C]."""
    allowed = block_mask & layout.kv_ok
    full_cnt, full_idx = _pack(allowed & layout.full_tile)
    part_cnt, part_idx = _pack(allowed & ~layout.full_tile)
    return part_cnt, part_idx, full_cnt, full_idx


class Fa4Sm90Backend(base.Backend):
    """Upstream FA4 Hopper kernels with full / partial block lists."""

    name = 'fa4-sm90'
    display = 'FA4 (SM90)'

    def attend(self, q, k, v, block_mask, layout):
        _, _, block_sparsity, interface, _ = _modules()
        part_cnt, part_idx, full_cnt, full_idx = index_lists(block_mask,
                                                             layout)
        # block_size by keyword: the 5th positional field is the varlen
        # cu_total_m_blocks, not the block size.
        tensors = block_sparsity.BlockSparseTensorsTorch(
            mask_block_cnt=part_cnt, mask_block_idx=part_idx,
            full_block_cnt=full_cnt, full_block_idx=full_idx,
            block_size=(tiling.TILE_SIZE, tiling.TILE_SIZE))
        with _CALL_LOCK, torch.no_grad():
            out = interface.flash_attn_func(
                q[None], k[None], v[None], softmax_scale=q.shape[-1] ** -0.5,
                mask_mod=_valid_key_mask_mod(),
                aux_tensors=[layout.slot_valid],
                block_sparse_tensors=tensors)
        if isinstance(out, tuple):
            out = out[0]
        return out[0]

    def warmup_note(self) -> str:
        return 'compiling kernels for this GPU (first run only, ~10 s)'


def create(info) -> base.Backend:
    if info.kind != 'cuda' or info.cc is None or info.cc[0] != _MAJOR:
        raise base.BackendUnavailable('fa4-sm90 is for Hopper GPUs')
    try:
        _modules()
    except ImportError as error:
        raise base.BackendUnavailable(
            f'FA4 kernels are not installed ({error}); run install_fa4 to '
            'enable them') from error
    return Fa4Sm90Backend()
