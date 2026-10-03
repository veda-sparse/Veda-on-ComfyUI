"""FA4 CuTe block-sparse attention on SM120 / SM121.

Subtypes (see hardware.py): compute capability 12.0 is GeForce RTX 50 and
the RTX PRO Blackwell cards (RTX PRO 6000 Workstation / Server / Max-Q);
12.1 is GB10 (DGX Spark: aarch64 Linux, unified memory, usually CUDA 13).
FA4 sizes every arch-12 part for 99 KB of shared memory per block (its
`FlashAttentionForwardSm120.can_implement`), the same as sm86 / sm89.

Upstream FA4 rejects block sparsity on arch 12. Its SM120 kernels are thin
subclasses of the SM80 ones (only the SMEM bound differs), so the vendored
copy `veda_comfy/kernels/fa4` (FA4 4.0.0b32 + Miowtion's patch series,
whose last two patches lift the arch-12 gates) gives SM120 the patched SM80
block-sparse main loops. A private package: no global `flash_attn` import.

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
    from ..kernels.fa4 import block_sparsity  # pylint: disable=import-outside-toplevel
    from ..kernels.fa4 import interface  # pylint: disable=import-outside-toplevel
    from ..kernels.fa4 import utils  # pylint: disable=import-outside-toplevel
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


_E4M3_MAX = 448.0


def _to_fp8(x: torch.Tensor) -> tuple[torch.Tensor, float]:
    """(e4m3 view of x, the scale that undoes it).

    One amax over the whole tile-ordered tensor: `x / scale` lands in e4m3's
    range, and `scale` is what the caller multiplies back in.
    """
    amax = x.abs().amax().float()
    scale = torch.clamp(amax / _E4M3_MAX, min=torch.finfo(torch.float32).tiny)
    return (x.float() / scale).to(torch.float8_e4m3fn), scale.item()


class Fa4Sm120Backend(base.Backend):
    """Patched FA4 SM80-derived kernels with a dense block mask."""

    def __init__(self, family: str, subtype: str, fp8: bool):
        suffix = '-fp8' if fp8 else ''
        self.name = f'fa4-{family}{suffix}'
        self.display = (f'FA4 ({family.upper()})'
                        + (' FP8' if fp8 else ''))
        self.subtype = subtype
        self.fp8 = fp8

    def attend(self, q, k, v, block_mask, layout):
        _, _, block_sparsity, interface, _ = _modules()
        tensors = block_sparsity.DenseBlockMaskTorch(
            block_mask=(block_mask & layout.kv_ok)[None],
            partial_kv_blocks=~layout.full_tile,
            block_size=(tiling.TILE_SIZE, tiling.TILE_SIZE))
        scale = q.shape[-1] ** -0.5
        v_scale = None
        if self.fp8:
            # e4m3 is a float format: one scale per tensor already puts the
            # values in range, and the relative error does not depend on the
            # magnitude, so per-tile scales would buy almost nothing. The q
            # and k scales fold into softmax_scale and the v scale into the
            # output, which keeps every descale out of the kernel.
            q, q_s = _to_fp8(q)
            k, k_s = _to_fp8(k)
            v, v_scale = _to_fp8(v)
            # The FP8 PV gemm reads V with the contraction dim contiguous,
            # so hand the kernel (head_dim_v, slots) per head.
            v = v.permute(2, 1, 0).contiguous()[None]
            scale *= q_s * k_s
        with _CALL_LOCK, torch.no_grad():
            out = interface.flash_attn_func(
                q[None], k[None], v if self.fp8 else v[None],
                softmax_scale=scale,
                mask_mod=_valid_key_mask_mod(),
                aux_tensors=[layout.slot_valid],
                block_sparse_tensors=tensors)
        if isinstance(out, tuple):
            out = out[0]
        out = out[0]
        return out if v_scale is None else out.mul_(v_scale)

    def warmup_note(self) -> str:
        return 'compiling kernels for this GPU (first run only, ~10 s)'


def create(info, fp8: bool = False) -> base.Backend:
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
    return Fa4Sm120Backend(info.family, info.subtype, fp8)
