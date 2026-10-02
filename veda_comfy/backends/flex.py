"""torch FlexAttention backend: a real block-sparse kernel without FA4.

The Veda mask is already a 128x128 block mask on the tile-ordered sequence,
which is exactly FlexAttention's BlockMask granularity: full key tiles skip
the mask_mod, partial ones (padding slots) apply it. Needs CUDA and Triton
(bundled with Linux torch; the `triton-windows` build matching torch on
Windows). The first call per shape compiles, which takes a while once.

Uncompiled FlexAttention materializes the full score matrix, which would
run out of memory on a real sequence, so the compiled function gets a
generous recompile budget (every head-group tiling of every geometry is a
new shape).
"""

from __future__ import annotations

import contextlib
import functools
import importlib.util
import weakref

import torch

from . import base
from ..core import tiling

_RECOMPILE_LIMIT = 256


def _pack(member: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-packed column indices of member [H', R, C] -> (cnt, idx)."""
    order = torch.argsort((~member).to(torch.int8), dim=-1, stable=True)
    return (member.sum(-1)[None].to(torch.int32),
            order[None].to(torch.int32).contiguous())


@functools.cache
def _flex():
    from torch.nn.attention import flex_attention as fa  # pylint: disable=import-outside-toplevel
    return fa, torch.compile(fa.flex_attention, dynamic=None)


def _recompile_budget():
    """Raises dynamo's per-function recompile limit for our calls only."""
    config = torch._dynamo.config  # pylint: disable=protected-access
    for name in ('recompile_limit', 'cache_size_limit'):
        if hasattr(config, name) and getattr(config, name) < _RECOMPILE_LIMIT:
            return config.patch(**{name: _RECOMPILE_LIMIT})
    return contextlib.nullcontext()


class FlexBackend(base.Backend):
    """FlexAttention with a BlockMask built from the Veda block mask."""

    name = 'flex'
    display = 'FlexAttention'

    def __init__(self):
        # One mask_mod per tile layout: a fresh closure per call would be a
        # new function object for dynamo to guard on.
        self._mask_mods = weakref.WeakKeyDictionary()

    def _mask_mod(self, layout: tiling.TileLayout):
        mod = self._mask_mods.get(layout)
        if mod is None:
            slot_valid = layout.slot_valid

            def valid_key(b, h, q_idx, kv_idx):
                del b, h, q_idx
                return slot_valid[kv_idx] > 0

            mod = self._mask_mods[layout] = valid_key
        return mod

    def attend(self, q, k, v, block_mask, layout):
        module, compiled = _flex()
        allowed = block_mask & layout.kv_ok
        full_cnt, full_idx = _pack(allowed & layout.full_tile)
        part_cnt, part_idx = _pack(allowed & ~layout.full_tile)
        total = q.shape[0]
        mask = module.BlockMask.from_kv_blocks(
            part_cnt, part_idx, full_cnt, full_idx,
            BLOCK_SIZE=tiling.TILE_SIZE, mask_mod=self._mask_mod(layout),
            seq_lengths=(total, total))
        with _recompile_budget():
            out = compiled(q.permute(1, 0, 2)[None],
                           k.permute(1, 0, 2)[None],
                           v.permute(1, 0, 2)[None], block_mask=mask)
        return out[0].permute(1, 0, 2)

    def warmup_note(self) -> str:
        return 'compiling FlexAttention kernels (first run only, ~1 min)'


def create(info) -> base.Backend:
    if info.kind != 'cuda':
        raise base.BackendUnavailable('FlexAttention needs a CUDA GPU')
    if importlib.util.find_spec('triton') is None:
        hint = (' (install the "triton-windows" build that matches your '
                'torch with ComfyUI\'s python)' if info.os == 'windows'
                else '')
        raise base.BackendUnavailable(f'FlexAttention needs Triton{hint}')
    try:
        _flex()
    except (ImportError, AttributeError) as error:
        raise base.BackendUnavailable(
            f'this torch has no FlexAttention ({error}); torch >= 2.5 '
            'is required') from error
    return FlexBackend()
