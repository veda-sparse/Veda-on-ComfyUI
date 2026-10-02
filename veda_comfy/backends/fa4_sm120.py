"""FA4 CuTe block-sparse attention on SM120 / SM121.

Subtypes (see hardware.py): compute capability 12.0 is GeForce RTX 50 and
the RTX PRO Blackwell cards (RTX PRO 6000 Workstation / Server / Max-Q);
12.1 is GB10 (DGX Spark: aarch64 Linux, unified memory, usually CUDA 13).
FA4 sizes every arch-12 part for 99 KB of shared memory per block (its
`FlashAttentionForwardSm120.can_implement`), the same as sm86 / sm89.

Upstream FA4 rejects block sparsity on arch 12. Its SM120 kernels are thin
subclasses of the SM80 ones (only the SMEM bound differs), so the vendored
copy `veda_comfy/_vendor/fa4_sm8x` (FA4 4.0.0b32 + Miowtion's patch series,
whose last two patches lift the arch-12 gates) gives SM120 the patched SM80
block-sparse main loops. Private copy: no global `flash_attn` import.

Self-contained on purpose (see backends/base.py): shares nothing with the
other FA4 backends except the vendored package, which is generated, never
edited by hand.
"""

from __future__ import annotations

import functools
import threading

import torch

from . import base
from ..core import tiling

_MAJOR = 12
# The first call of a signature JIT-compiles through CuTe DSL / MLIR, which
# is not documented as thread-safe; only the host-side launch is held.
_CALL_LOCK = threading.Lock()


@functools.cache
def _modules():
    import cutlass  # pylint: disable=import-outside-toplevel
    import cutlass.cute as cute  # pylint: disable=import-outside-toplevel
    from .._vendor.fa4_sm8x import block_sparsity  # pylint: disable=import-outside-toplevel
    from .._vendor.fa4_sm8x import interface  # pylint: disable=import-outside-toplevel
    from .._vendor.fa4_sm8x import utils  # pylint: disable=import-outside-toplevel
    return cutlass, cute, block_sparsity, interface, utils


@functools.cache
def _valid_key_mask_mod():
    """The singleton mask_mod: key slot n is real iff aux_tensors[0][n] != 0.

    A module-level singleton because FA4 keys its compile cache on the
    callable; a fresh closure per call would recompile every call.
    """
    cutlass, cute, _, _, utils = _modules()

    @cute.jit
    def valid_key(batch, head, m_idx, n_idx, seqlen_info, aux_tensors):
        del batch, head, m_idx, seqlen_info
        slot_valid = aux_tensors[0]
        valid = utils.scalar_to_ssa(slot_valid[n_idx[0]], cutlass.Int32)
        zero = utils.scalar_to_ssa(0, cutlass.Int32)
        return valid != zero

    return valid_key


class Fa4Sm120Backend(base.Backend):
    """Patched FA4 SM80-derived kernels with a dense block mask."""

    def __init__(self, family: str, subtype: str):
        self.name = f'fa4-{family}'
        self.display = f'FA4 ({family.upper()})'
        self.subtype = subtype

    def attend(self, q, k, v, block_mask, layout):
        _, _, block_sparsity, interface, _ = _modules()
        tensors = block_sparsity.DenseBlockMaskTorch(
            block_mask=(block_mask & layout.kv_ok)[None],
            partial_kv_blocks=~layout.full_tile,
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
        return 'compiling FA4 kernels for this GPU (first run only)'


def create(info) -> base.Backend:
    if info.kind != 'cuda' or info.cc is None or info.cc[0] != _MAJOR:
        raise base.BackendUnavailable('fa4-sm120 is for SM120 / SM121 GPUs')
    try:
        _modules()
    except ImportError as error:
        hint = ''
        if info.cuda_major and info.cuda_major >= 13:
            hint = ' (CUDA 13 torch: install_fa4 picks the cu13 CuTe DSL)'
        raise base.BackendUnavailable(
            f'FA4 kernels are not installed ({error}); run install_fa4 to '
            f'enable them{hint}') from error
    return Fa4Sm120Backend(info.family, info.subtype)
