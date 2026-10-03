"""Localises an FP8 kernel mismatch to one stage.

    python tools/diagnose_fp8.py

Runs the FP8 and bf16 block-sparse kernels on the same problem through
three lenses:

1. *structural* - both kernels see values that are exactly representable
   in e4m3, so quantisation cannot explain any difference. What is left is
   a layout or ordering fault.
2. *P only* - V is an identity-like matrix, so the output is the attention
   probabilities themselves. This separates the QK gemm and the softmax
   from the PV gemm.
3. *V only* - Q and K are shaped so the softmax is flat, so the output is
   the mean of V. This isolates the PV gemm.

Each lens prints the relative error against the bf16 kernel; the first one
that blows up is where the fault is.
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
from veda_comfy.core import selection  # noqa: E402
from veda_comfy.core import tiling  # noqa: E402

_E4M3_MAX = 448.0


def _round_trip(x: torch.Tensor) -> torch.Tensor:
    """x as the FP8 backend will see it, back in x's dtype."""
    amax = x.abs().amax().float()
    scale = torch.clamp(amax / _E4M3_MAX, min=torch.finfo(torch.float32).tiny)
    return ((x.float() / scale).to(torch.float8_e4m3fn).float()
            * scale).to(x.dtype)


def _problem(device, seq_len=4096, heads=4, seed=0):
    spans = [tiling.TiledSpan(256, (1, 8, 16), tiling.TileShape(1, 8, 16)),
             tiling.TiledSpan(512, (7, 8, 8), tiling.TileShape(4, 4, 8))]
    layout = tiling.build_tile_layout(spans, seq_len, device)
    generator = torch.Generator().manual_seed(seed)
    heads_idx = torch.arange(heads, device=device)
    tensors = []
    for _ in range(3):
        x = torch.randn(seq_len, heads, 128, generator=generator)
        tensors.append(tiling.gather_tiles(
            x.to(device=device, dtype=torch.bfloat16), layout, heads_idx))
    scores = torch.randn(heads, layout.n_video_tiles, layout.n_video_tiles,
                         generator=generator).to(device)
    blocks = selection.column_blocks(layout, selection.Budget(ratio=0.2),
                                     selection.Budget(ratio=0.2))
    mask = selection.block_mask(*selection.select(scores, layout, blocks),
                                layout)
    return (*tensors, mask, layout)


def _rel(got: torch.Tensor, want: torch.Tensor, layout) -> str:
    real = layout.slot_valid.bool()
    a, b = got[real].float(), want[real].float()
    denom = b.norm().item() or 1.0
    return (f'rel L2 {(a - b).norm().item() / denom:7.3%}   '
            f'absmax got {a.abs().max().item():8.4f} want '
            f'{b.abs().max().item():8.4f}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--seq-len', type=int, default=4096)
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
    print(f'{info.label}: {fp8.name} against {bf16.name}')

    q, k, v, mask, layout = _problem(device, args.seq_len)

    print('\n1. structural (inputs exactly representable in e4m3)')
    qr, kr, vr = (_round_trip(t) for t in (q, k, v))
    got = fp8.attend(q, k, v, mask, layout)
    want = bf16.attend(qr, kr, vr, mask, layout)
    print('   ', _rel(got, want, layout))

    print('\n2. P only (V is identity-like, so the output is the softmax)')
    eye = torch.zeros_like(v)
    idx = torch.arange(v.shape[0], device=device) % v.shape[2]
    eye[torch.arange(v.shape[0], device=device)[:, None],
        torch.arange(v.shape[1], device=device)[None, :], idx[:, None]] = 1.0
    print('   ', _rel(fp8.attend(q, k, eye, mask, layout),
                      bf16.attend(qr, kr, eye, mask, layout), layout))

    print('\n3. V only (flat softmax, so the output is a mean of V)')
    flat_q = torch.zeros_like(q)
    flat_k = torch.zeros_like(k)
    print('   ', _rel(fp8.attend(flat_q, flat_k, v, mask, layout),
                      bf16.attend(flat_q, flat_k, vr, mask, layout), layout))

    print('\nreference check (bf16 kernel against fp32 reference)')
    exact = reference.block_sparse_attention(q, k, v, mask, layout)
    print('   ', _rel(bf16.attend(q, k, v, mask, layout), exact, layout))


if __name__ == '__main__':
    main()
