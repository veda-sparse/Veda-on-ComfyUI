"""Calibrates Veda's FP8 kernel against the INT8 attention ComfyUI ships.

    python tools/compare_int8.py

ComfyUI's low-precision attention is SageAttention (`--use-sage-attention`,
`attention_sage` in `comfy/ldm/modules/attention.py`), which quantizes Q and
K to INT8 per block. That is the bar a precision variant has to clear: if
INT8 is good enough to ship as ComfyUI's fast path, FP8 is good enough when
it is at least as close to an fp32 reference.

The problem is deliberately **dense** - every block kept, every slot real -
so our kernel and the baselines compute the same attention and the only
difference left is arithmetic.

Baselines, in order of authority:
  * fp32 reference (`core.reference`) - ground truth.
  * bf16 (our FA4 kernel, and torch SDPA) - what ComfyUI runs by default.
  * SageAttention, called exactly as ComfyUI calls it, when installed.
  * a SageAttention-v1-style INT8 model in plain torch - per-block
    quantize/dequantize of Q and K, fp32 scores, fp16 PV. Mathematically
    the same arithmetic as the INT8 kernel (one scale pair per block pair
    descales an INT32 accumulator), so it stands in for the real thing on
    a machine that cannot build it.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))))

import torch  # noqa: E402

from veda_comfy import backends  # noqa: E402
from veda_comfy import hardware  # noqa: E402
from veda_comfy.core import reference  # noqa: E402
from veda_comfy.core import tiling  # noqa: E402

# SageAttention v1's block sizes (BLKQ / BLKK in its kernels).
_Q_BLOCK = 128
_K_BLOCK = 64


def _dense_problem(device, tiles=32, heads=8, head_dim=128, seed=0):
    """A problem with no padding: every one of the 128*tiles slots is real.

    The grid divides the tile shape exactly and the global region is a
    whole number of tiles, so `slot_valid` is all true and a full block
    mask really is dense attention.
    """
    shape = tiling.TileShape(1, 8, 16)
    span = tiling.TiledSpan(256, (tiles - 2, 8, 16), shape)
    seq_len = 256 + (tiles - 2) * tiling.TILE_SIZE
    layout = tiling.build_tile_layout([span], seq_len, device)
    assert bool(layout.slot_valid.all()), 'problem is not padding-free'
    generator = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(3):
        x = torch.randn(seq_len, heads, head_dim, generator=generator)
        out.append(x.to(device=device, dtype=torch.bfloat16))
    mask = torch.ones(heads, layout.n_tiles, layout.n_tiles,
                      dtype=torch.bool, device=device)
    return (*out, mask, layout)


def _quant_dequant(x: torch.Tensor, block: int) -> torch.Tensor:
    """x through INT8 and back, one scale per (block of rows, head)."""
    rows, heads, dim = x.shape
    blocks = x.float().view(rows // block, block, heads, dim)
    scale = blocks.abs().amax(dim=(1, 3), keepdim=True) / 127.0
    scale = scale.clamp(min=torch.finfo(torch.float32).tiny)
    return ((blocks / scale).round().clamp(-127, 127) * scale).view(x.shape)


def _sage_v1_model(q, k, v):
    """SageAttention v1's arithmetic, in plain torch.

    smooth_k is off because that is how ComfyUI calls it.
    """
    qd = _quant_dequant(q, _Q_BLOCK)
    kd = _quant_dequant(k, _K_BLOCK)
    scale = q.shape[-1] ** -0.5
    scores = torch.einsum('qhd,khd->hqk', qd, kd) * scale
    probs = torch.softmax(scores, dim=-1).half()
    return torch.einsum('hqk,khd->qhd', probs, v.half()).to(q.dtype)


def _sage_real(q, k, v):
    """The installed SageAttention, with ComfyUI's own kwargs."""
    from sageattention import sageattn  # noqa: PLC0415
    args = dict(is_causal=False, tensor_layout='NHD', sm_scale=None,
                smooth_k=False)
    return sageattn(q[None], k[None], v[None], **args)[0]


def _sdpa(q, k, v):
    out = torch.nn.functional.scaled_dot_product_attention(
        *(t.transpose(0, 1)[None] for t in (q, k, v)))
    return out[0].transpose(0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--tiles', type=int, default=32)
    parser.add_argument('--heads', type=int, default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('needs a CUDA GPU')
    device = torch.device('cuda')
    info = hardware.describe(device)
    names = backends.candidates(info, 'fa4-fp8')
    if not names[0].endswith('-fp8'):
        raise SystemExit(f'no FP8 kernel for {info.label}')
    fp8 = backends._load(names[0], info)
    bf16 = backends._load(names[1], info)

    q, k, v, mask, layout = _dense_problem(device, args.tiles, args.heads)
    tokens = q.shape[0]
    print(f'{info.label}: dense attention, {tokens} tokens x {args.heads} '
          f'heads x {q.shape[2]}')
    want = reference.block_sparse_attention(q, k, v, mask, layout).float()
    denom = want.norm().item()

    rows = [
        (f'{bf16.name} (bf16)', lambda: bf16.attend(q, k, v, mask, layout)),
        ('torch SDPA (bf16)', lambda: _sdpa(q, k, v)),
        ('ComfyUI INT8 model (torch)', lambda: _sage_v1_model(q, k, v)),
    ]
    try:
        import sageattention  # noqa: F401,PLC0415
        rows.append(('SageAttention (installed)', lambda: _sage_real(q, k, v)))
    except ImportError as error:
        print(f'  (SageAttention itself is not installed: {error.name}; '
              'the torch model below stands in for it)')
    rows.append((f'{fp8.name} (FP8)', lambda: fp8.attend(q, k, v, mask,
                                                         layout)))

    results = {}
    print(f'\n{"":28s}  rel L2 vs fp32   max err / absmax')
    for label, run in rows:
        got = run().float()
        results[label] = got
        rel = (got - want).norm().item() / denom
        peak = ((got - want).abs().max().item()
                / want.abs().max().item())
        print(f'  {label:26s}  {rel:12.3%}   {peak:14.3%}')

    int8 = results['ComfyUI INT8 model (torch)']
    ours = results[f'{fp8.name} (FP8)']
    rel = (ours - int8).norm().item() / int8.norm().item()
    print(f'\nFP8 against the INT8 path directly: rel L2 {rel:.3%}')
    int8_err = (int8 - want).norm().item() / denom
    fp8_err = (ours - want).norm().item() / denom
    verdict = ('FP8 is closer to the reference than the INT8 ComfyUI ships'
               if fp8_err <= int8_err else
               'FP8 is further from the reference than ComfyUI\'s INT8')
    print(f'{verdict} ({fp8_err:.3%} against {int8_err:.3%}).')


if __name__ == '__main__':
    main()
