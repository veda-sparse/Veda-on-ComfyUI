"""Sage INT8 exact attention with Sol skipped-block correction.

The implementation is adapted from the project's validated Miowtion Sol
kernel. Veda supplies the learned block mask; Sage quantisation is used for
selected blocks and pooled BF16 K/V summaries approximate skipped blocks.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634
TILE = 128


def _summaries(k, v, counts):
    tiles, block, heads, dim = k.shape[0] // TILE, TILE, k.shape[1], k.shape[2]
    valid = torch.arange(block, device=k.device)[None, :] < counts[:, None]
    divisor = counts.clamp(min=1)[:, None, None]

    def total(x):
        shaped = x.view(tiles, block, heads, dim)
        return shaped.masked_fill(~valid[:, :, None, None], 0).sum(
            1, dtype=torch.float32)

    return (total(k) / divisor).bfloat16().contiguous(), total(v).bfloat16()


@triton.jit
def _forward(Q, K, V, Qi, Ki, Qs, Ks, KC, VC, Counts, Kept, Out,
             ExactTokens, N: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
             B: tl.constexpr, C: tl.constexpr, SCALE: tl.constexpr):
    row, head = tl.program_id(0), tl.program_id(1)
    r, d, j = tl.arange(0, B), tl.arange(0, D), tl.arange(0, C)
    sub = tl.arange(0, 64)
    qcount = tl.load(Counts + row)
    qoff = ((row * B + r[:, None]) * H + head) * D + d[None, :]
    qraw = tl.load(Q + qoff, mask=r[:, None] < qcount, other=0)
    qi = tl.load(Qi + qoff, mask=r[:, None] < qcount, other=0)
    qscale = tl.load(Qs + head * N + row)
    acc = tl.zeros((B, D), tl.float32)
    denom = tl.zeros((B,), tl.float32)
    maximum = tl.full((B,), -float('inf'), tl.float32)
    exact_tokens = tl.zeros((), tl.int32)
    for first in range(0, N, C):
        cols = first + j
        valid_cols = cols < N
        lengths = tl.load(Counts + cols, mask=valid_cols, other=0)
        summary_off = (cols[:, None] * H + head) * D + d[None, :]
        kc = tl.load(KC + summary_off, mask=valid_cols[:, None], other=0)
        vc = tl.load(VC + summary_off, mask=valid_cols[:, None], other=0)
        proxy = tl.dot(qraw, tl.trans(kc)) * (SCALE * LOG2E)
        selected = tl.load(Kept + (head * N + row) * N + cols,
                           mask=valid_cols, other=False)
        selected = selected & valid_cols & (lengths > 0) & (qcount > 0)
        approximate = valid_cols & (lengths > 0) & ~selected & (qcount > 0)
        scores = tl.where(approximate[None, :], proxy, -float('inf'))
        next_max = tl.maximum(maximum, tl.max(scores, 1))
        alpha = tl.exp2(tl.where(maximum == next_max, 0., maximum - next_max))
        probability = tl.where(approximate[None, :],
                               tl.exp2(scores - next_max[:, None]), 0.)
        acc = acc * alpha[:, None] + tl.dot(probability.to(tl.bfloat16), vc)
        denom = denom * alpha + tl.sum(probability * lengths[None, :], 1)
        maximum = next_max
        exact_tokens += tl.sum(tl.where(selected, lengths, 0))
        candidates = tl.where(selected, j, C)
        for _ in range(tl.sum(selected.to(tl.int32))):
            offset = tl.min(candidates)
            key_tile = first + offset
            candidates = tl.where(j == offset, C, candidates)
            key_count = tl.load(Counts + key_tile)
            for half in range(B // 64):
                slots = key_tile * B + half * 64 + sub
                koff = (slots[:, None] * H + head) * D + d[None, :]
                valid = half * 64 + sub < key_count
                ki = tl.load(Ki + koff, mask=valid[:, None], other=0)
                kscale = tl.load(Ks + head * (N * (B // 64))
                                 + key_tile * (B // 64) + half)
                score = tl.dot(qi, tl.trans(ki)).to(tl.float32)
                score *= qscale * kscale
                score = tl.where(valid[None, :], score, -float('inf'))
                next_max = tl.maximum(maximum, tl.max(score, 1))
                alpha = tl.exp2(tl.where(maximum == next_max, 0.,
                                         maximum - next_max))
                probability = tl.where(valid[None, :],
                                       tl.exp2(score - next_max[:, None]), 0.)
                value = tl.load(V + koff, mask=valid[:, None], other=0)
                acc = acc * alpha[:, None] + tl.dot(
                    probability.to(tl.float16), value.to(tl.float16),
                    out_dtype=tl.float16)
                denom = denom * alpha + tl.sum(probability, 1)
                maximum = next_max
    answer = acc / tl.where(denom > 0., denom, 1.)[:, None]
    answer = tl.where(r[:, None] < qcount, answer, 0.)
    tl.store(Out + qoff, answer.to(Out.dtype.element_ty))
    tl.store(ExactTokens + head * N + row, exact_tokens)


@torch.no_grad()
def attend(q, k, v, counts, selected, sage):
    """Run hybrid attention on Veda's 128-token tile mask."""
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 3:
        raise ValueError('Q/K/V must have the same [N,H,D] shape')
    if any(not x.is_cuda or x.dtype != torch.bfloat16 or not x.is_contiguous()
           for x in (q, k, v)):
        raise ValueError('Sol + Veda requires contiguous BF16 CUDA Q/K/V')
    slots, heads, dim = q.shape
    if dim != 128 or slots % TILE:
        raise ValueError('Sol + Veda requires D=128 and 128-token tiles')
    tiles = slots // TILE
    if counts.shape != (tiles,) or counts.dtype != torch.int32:
        raise ValueError('counts must be int32 with one value per tile')
    if selected.shape != (heads, tiles, tiles) or selected.dtype != torch.bool:
        raise ValueError('selected must be [H,tiles,tiles] bool')
    qi, qs = sage.quantize(q, TILE, pre_scale=dim ** -0.5 * LOG2E)
    ki, ks = sage.quantize(k, 64)
    kc, vc = _summaries(k, v, counts)
    kc = kc.view(tiles, heads, dim).contiguous()
    vc = vc.view(tiles, heads, dim).contiguous()
    out = torch.empty_like(q)
    exact = torch.empty(heads, tiles, device=q.device, dtype=torch.int32)
    _forward[(tiles, heads)](
        q, k, v, qi, ki, qs, ks, kc, vc, counts, selected.contiguous(), out,
        exact, N=tiles, H=heads, D=dim, B=TILE, C=32, SCALE=dim ** -.5,
        num_warps=4, num_stages=1)
    return out, exact
