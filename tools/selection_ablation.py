"""Compares block-selection rules on activations captured from a render.

    python tools/selection_ablation.py captures/*.pt --density 10 20 44

Why this exists: two conclusions in this repo were drawn from synthetic
q/k and both were wrong. i.i.d. Gaussian keys leave attention almost
unconcentrated - over 72 tiles the top 8 hold 12.7% of the mass against
11.1% for uniform - which is the one regime where block selection cannot
work and anything that captures the bulk instead wins. A sparse method
can only be sized on activations that have the concentration real
attention has.

Each capture is one (layer, step, head group) of a real forward: the
tile-ordered q / k / v, the layout, and the predictor's scores. The
reference output is dense attention on those same tensors, so a rule is
judged on the error it actually causes.

Two rules of comparison, both learned the hard way:

  * equal cost, not equal budget. A rule that keeps more tiles must be
    compared against a plain top-k given the same number, or it is being
    paid for in silence.
  * paired per head. The error distributions have different shapes, so a
    ratio of medians flatters whichever has the longer tail - it read as
    15% where pairing said 1%.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from veda_comfy.core import selection, tiling  # noqa: E402


def _layout(capture, device):
    """Rebuilds the tile layout a capture describes.

    Captures store spans rather than the TileLayout itself: that class
    is imported under ComfyUI's custom-node module path, so a pickle of
    it cannot be opened anywhere else.
    """
    spans = [tiling.TiledSpan(start, tuple(grid), tiling.TileShape(*shape))
             for start, grid, shape in capture['spans']]
    return tiling.build_tile_layout(spans, capture['seq_len'], device)


def _dense(q, k, v, layout, chunk: int = 2048):
    """Dense attention over the real geometries, in query chunks.

    The whole score matrix is N^2 per head - 33 GB at 32k slots - so it
    is never materialised; each chunk of query rows is softmaxed against
    all keys and discarded.
    """
    scale = q.shape[-1] ** -0.5
    out = torch.empty_like(q, dtype=torch.float32)
    kf, vf = k.float(), v.float()
    for start in range(0, q.shape[0], chunk):
        rows = q[start:start + chunk].float()
        scores = torch.einsum('qhd,khd->hqk', rows, kf) * scale
        probability = torch.softmax(scores, dim=-1)
        out[start:start + chunk] = torch.einsum('hqk,khd->qhd', probability,
                                                vf)
        del scores, probability
    return out * layout.slot_valid.to(out.dtype)[:, None, None]


def _block_sparse(q, k, v, block_mask, layout, tiles_at_once: int = 16):
    """reference.block_sparse_attention, a few query tiles at a time.

    The reference materialises the whole N^2 score matrix, which it says
    plainly is for self-tests and never a real sequence - three copies
    of 33 GB does not fit even on a 96 GB card. Chunking by whole tiles
    keeps the mask slicing trivial.
    """
    tile = tiling.TILE_SIZE
    scale = q.shape[-1] ** -0.5
    valid = layout.slot_valid.bool()
    out = torch.empty_like(q, dtype=torch.float32)
    kf, vf = k.float(), v.float()
    n_tiles = q.shape[0] // tile
    for first in range(0, n_tiles, tiles_at_once):
        last = min(first + tiles_at_once, n_tiles)
        start, stop = first * tile, last * tile
        keep = block_mask[:, first:last].repeat_interleave(tile, 1)
        keep = keep.repeat_interleave(tile, 2) & valid[None, None, :]
        scores = torch.einsum('qhd,khd->hqk', q[start:stop].float(), kf)
        scores = (scores * scale).masked_fill(~keep, float('-inf'))
        probability = torch.softmax(scores, dim=-1).nan_to_num(0.0)
        out[start:stop] = torch.einsum('hqk,khd->qhd', probability, vf)
        del scores, probability, keep
    return out * layout.slot_valid.to(out.dtype)[:, None, None]


def _error(out, want, layout):
    """Relative L2 over the real rows, per head: [H]."""
    real = layout.slot_valid.bool()
    diff = (out[real].float() - want[real].float()).pow(2).sum(dim=(0, 2))
    return (diff / want[real].float().pow(2).sum(dim=(0, 2))).sqrt()


def rule_topk(scores, layout, budget):
    """The shipped rule: a fixed number of key tiles per query tile."""
    blocks = selection.column_blocks(layout, budget, budget)
    index, keep = selection.select(scores, layout, blocks)
    return selection.block_mask(index, keep, layout)


def rule_topk_log_rows(scores, layout, budget):
    """Top-k after adding log(real rows) to every tile's score.

    A tile with half its rows padded can hold at most half the mass of a
    full one, and the predictor's score does not know that. Sol-Attn uses
    the row count when it estimates a dropped block's mass but not when
    it ranks; closing that gap is free, the count is already in the
    layout. The benefit should follow the padded fraction, which is why
    it is worth measuring on the two-stage geometries rather than only
    the searched ones.
    """
    rows = layout.valid_count.clamp(min=1).float().log()
    return rule_topk(scores + rows[None, None, :scores.shape[-1]], layout,
                     budget)


def rule_adaptive(scores, layout, budget, tau):
    blocks = selection.column_blocks(layout, budget, budget)
    index, keep = selection.select_adaptive(scores, layout, blocks, tau)
    return selection.block_mask(index, keep, layout)


def kept_per_row(mask, layout) -> float:
    n = layout.n_video_tiles
    return mask[:, :n, :n].sum(-1).float().mean().item()


def evaluate(capture, densities, taus):
    """One capture -> {rule: (mean kept per row, [H] error)}."""
    q, k, v = capture['q'], capture['k'], capture['v']
    layout = _layout(capture, q.device)
    scores = capture['scores']
    want = _dense(q, k, v, layout)
    n = layout.n_video_tiles
    out = {}
    for percent in densities:
        tiles = max(1, round(n * percent / 100.0))
        budget = selection.Budget(tiles=tiles)
        for name, mask in (
                (f'top-k {percent:g}%', rule_topk(scores, layout, budget)),
                (f'top-k+logB {percent:g}%',
                 rule_topk_log_rows(scores, layout, budget))):
            got = _block_sparse(q, k, v, mask, layout)
            out[name] = (kept_per_row(mask, layout),
                         _error(got, want, layout))
    for tau in taus:
        budget = selection.Budget(tiles=max(1, n // 10))
        mask = rule_adaptive(scores, layout, budget, tau)
        got = _block_sparse(q, k, v, mask, layout)
        out[f'adaptive tau {tau:g}'] = (kept_per_row(mask, layout),
                                        _error(got, want, layout))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('captures', nargs='+')
    parser.add_argument('--density', type=float, nargs='+',
                        default=[10.0, 20.0, 44.0],
                        help='percent of key tiles kept per query tile')
    parser.add_argument('--tau', type=float, nargs='+', default=[1.3])
    parser.add_argument('--device', default=None,
                        help='cuda makes the dense reference tractable')
    parser.add_argument('--baseline', default=None,
                        help='rule to pair against, e.g. "top-k 10%%"')
    args = parser.parse_args()

    paths = [p for pattern in args.captures for p in sorted(glob.glob(pattern))]
    if not paths:
        raise SystemExit('no captures matched')
    rows: dict[str, list] = {}
    kept: dict[str, list] = {}
    for path in paths:
        where = args.device or ('cuda' if torch.cuda.is_available()
                                else 'cpu')
        capture = torch.load(path, map_location=where, weights_only=False)
        for name, (count, error) in evaluate(capture, args.density,
                                             args.tau).items():
            rows.setdefault(name, []).append(error)
            kept.setdefault(name, []).append(count)

    print(f'{len(paths)} captures, '
          f'{sum(e.numel() for e in next(iter(rows.values())))} heads\n')
    print(f"{'rule':24} {'kept/row':>9} {'mean err':>9} {'median':>9}")
    errors = {n: torch.cat(e) for n, e in rows.items()}
    for name in rows:
        e = errors[name]
        print(f'{name:24} {sum(kept[name]) / len(kept[name]):9.1f} '
              f'{e.mean():9.4f} {e.median():9.4f}')

    base = args.baseline
    if base and base in errors:
        print(f'\npaired per head against {base!r}:')
        reference_error = errors[base]
        for name, e in errors.items():
            if name == base:
                continue
            ratio = (e / reference_error.clamp(min=1e-12))
            wins = (e < reference_error).float().mean()
            print(f'  {name:24} median ratio {ratio.median():6.4f}  '
                  f'better on {100 * wins:5.1f}% of heads  '
                  f'worst {ratio.max():7.2f}x')


if __name__ == '__main__':
    main()
