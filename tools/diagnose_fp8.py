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
from veda_comfy.kernels.fa4.flash_fwd import FP8_V_PERMUTATION  # noqa: E402

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


def _attend_with_scale(backend, q, k, v, mask, layout, softmax_scale):
    """attend() with an explicit softmax scale, to test the folding."""
    from veda_comfy.kernels.fa4 import block_sparsity, interface
    tensors = block_sparsity.DenseBlockMaskTorch(
        block_mask=(mask & layout.kv_ok)[None],
        partial_kv_blocks=~layout.full_tile,
        block_size=(tiling.TILE_SIZE, tiling.TILE_SIZE))
    with torch.no_grad():
        out = interface.flash_attn_func(
            q[None], k[None], v[None], softmax_scale=softmax_scale,
            mask_mod=backend_mask_mod(), aux_tensors=[layout.slot_valid],
            block_sparse_tensors=tensors)
    return (out[0] if isinstance(out, tuple) else out)[0]


def backend_mask_mod():
    from veda_comfy.backends import fa4_sm120
    return fa4_sm120._valid_key_mask_mod()


def _lse(backend, q, k, v, mask, layout, fp8=False):
    """Log-sum-exp of the scores, straight from the kernel."""
    from veda_comfy.kernels.fa4 import block_sparsity, interface
    scale = 128 ** -0.5
    if fp8:
        from veda_comfy.backends.fa4_sm120 import _to_fp8, _v_permutation
        from veda_comfy.kernels.fa4.flash_fwd import FP8_P_SCALE  # noqa: F401
        q, q_s = _to_fp8(q)
        k, k_s = _to_fp8(k)
        v, _ = _to_fp8(v)
        v = v[_v_permutation(v.shape[0], v.device)]
        v = v.permute(2, 1, 0).contiguous()[None]
        scale *= q_s * k_s
    else:
        v = v[None]
    tensors = block_sparsity.DenseBlockMaskTorch(
        block_mask=(mask & layout.kv_ok)[None],
        partial_kv_blocks=~layout.full_tile,
        block_size=(tiling.TILE_SIZE, tiling.TILE_SIZE))
    with torch.no_grad():
        out = interface.flash_attn_func(
            q[None], k[None], v, softmax_scale=scale,
            mask_mod=backend_mask_mod(), aux_tensors=[layout.slot_valid],
            block_sparse_tensors=tensors, return_lse=True)
    lse = out[1]
    return lse[0].float()[:, layout.slot_valid.bool()]


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

    print('\nA. LSE (depends on QK and the softmax only, never on PV)')
    lse_fp8 = _lse(fp8, q, k, v, mask, layout, fp8=True)
    lse_bf16 = _lse(bf16, *(_round_trip(t) for t in (q, k, v)), mask, layout)
    d = (lse_fp8 - lse_bf16).abs()
    print(f'     max |diff| {d.max().item():.4f}   mean |diff| '
          f'{d.mean().item():.4f}   bf16 range '
          f'[{lse_bf16.min().item():.3f}, {lse_bf16.max().item():.3f}]')

    print('\n0. scale folding (bf16 kernel, operands scaled like FP8 does)')
    # The FP8 backend divides q and k by their amax scales and folds those
    # into softmax_scale. Running the known-good kernel the same way says
    # whether that arithmetic is sound before blaming the FP8 kernel.
    exact = reference.block_sparse_attention(q, k, v, mask, layout)
    qs = (q.abs().amax().float() / _E4M3_MAX).item()
    ks = (k.abs().amax().float() / _E4M3_MAX).item()
    scaled_q = (q.float() / qs).to(q.dtype)
    scaled_k = (k.float() / ks).to(k.dtype)
    folded = _attend_with_scale(bf16, scaled_q, scaled_k, v, mask, layout,
                                128 ** -0.5 * qs * ks)
    print('   ', _rel(folded, exact, layout))

    print('\n1. structural (inputs exactly representable in e4m3)')
    qr, kr, vr = (_round_trip(t) for t in (q, k, v))
    got = fp8.attend(q, k, v, mask, layout)
    want = bf16.attend(qr, kr, vr, mask, layout)
    print('   ', _rel(got, want, layout))

    print('\n2. P only (V is identity-like, so the output is the softmax)')
    # Only the first head_dim slots carry a one, so each output column is
    # one probability rather than a sum over every slot that folds onto it.
    eye = torch.zeros_like(v)
    width = min(v.shape[2], v.shape[0])
    rows = torch.arange(width, device=device)
    eye[rows[:, None], torch.arange(v.shape[1], device=device)[None, :],
        rows[:, None]] = 1.0
    print('   ', _rel(fp8.attend(q, k, eye, mask, layout),
                      bf16.attend(qr, kr, eye, mask, layout), layout))

    print('\n3. V only (flat softmax, so the output is a mean of V)')
    # Tiny but nonzero: all-zero operands make the quantisation scale
    # degenerate and the result NaN, which says nothing about the kernel.
    flat_q = torch.full_like(q, 1e-2)
    flat_k = torch.full_like(k, 1e-2)
    print('   ', _rel(fp8.attend(flat_q, flat_k, v, mask, layout),
                      bf16.attend(flat_q, flat_k, vr, mask, layout), layout))

    print('\n4. dense mask (every block kept: isolates the sparse walk)')
    full = torch.ones_like(mask) & layout.kv_ok
    print('   ', _rel(fp8.attend(q, k, v, full, layout),
                      bf16.attend(qr, kr, vr, full, layout), layout))

    print('\n5. who is peaked (V identity: count of outputs above 0.5)')
    p_fp8 = fp8.attend(q, k, eye, mask, layout)[layout.slot_valid.bool()]
    p_bf16 = bf16.attend(qr, kr, eye, mask, layout)[layout.slot_valid.bool()]
    for name, t in (('fp8', p_fp8), ('bf16', p_bf16)):
        row_sum = t.float().sum(-1)
        print(f'    {name:4s} >0.5: {(t.float() > 0.5).sum().item():7d}  '
              f'row sums mean {row_sum.mean().item():.4f} '
              f'min {row_sum.min().item():.4f} '
              f'max {row_sum.max().item():.4f}')

    print('\n6. the first 16 probabilities of one row, side by side')
    a = p_fp8[0, 0, :16].float().tolist()
    b = p_bf16[0, 0, :16].float().tolist()
    print('    bf16 ' + ' '.join(f'{x:7.4f}' for x in b))
    print('    fp8  ' + ' '.join(f'{x:7.4f}' for x in a))
    # A pure reordering keeps the multiset; duplication does not.
    import collections
    rounded = collections.Counter(round(x, 4) for x in a)
    print(f'    distinct fp8 values {len(rounded)}/16, '
          f'most common {rounded.most_common(2)}')

    print('\n7. inferred slot mapping (fp8 position -> bf16 position)')
    # Matching by value is ambiguous when probabilities are close, so the
    # same inference runs on several rows and only agreement counts.
    votes = []
    for row in range(6):
        a = p_fp8[row, 0, :16].float()
        b = p_bf16[row, 0, :16].float()
        order = []
        for i in range(16):
            d = (b - a[i]).abs()
            j = int(d.argmin())
            second = float(d.sort().values[1])
            order.append(j if second > 1.5 * float(d[j]) else -1)
        votes.append(order)
    agreed = []
    for i in range(16):
        col = [v[i] for v in votes if v[i] >= 0]
        agreed.append(max(set(col), key=col.count) if col else -1)
    print(f'    inferred  {agreed}')
    print(f'    in use    {list(FP8_V_PERMUTATION)}')
    if agreed == list(FP8_V_PERMUTATION):
        print('    the permutation in use already matches')
    else:
        need = [FP8_V_PERMUTATION[i] if agreed[i] < 0 else
                FP8_V_PERMUTATION[agreed[i]] for i in range(16)]
        print(f'    composed  {need}')

    print('\nreference check (bf16 kernel against fp32 reference)')
    print('   ', _rel(bf16.attend(q, k, v, mask, layout), exact, layout))


if __name__ == '__main__':
    main()
