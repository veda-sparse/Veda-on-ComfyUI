"""Exact block-sparse attention in fp32: the ground truth for every backend.

Token level and O(N^2) memory, so only for tests and the small backend
self-tests; never on a real sequence.
"""

from __future__ import annotations

import torch

from . import tiling


def block_sparse_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           block_mask: torch.Tensor,
                           layout: tiling.TileLayout) -> torch.Tensor:
    """Masked softmax attention on tile-ordered tensors.

    Args:
        q, k, v: [N, H', D] tile order, N = n_tiles * 128.
        block_mask: [H', n_tiles, n_tiles] bool, rows are query tiles.
        layout: Tile layout (key slot validity).

    Returns:
        [N, H', D] in q's dtype; rows of padding slots are 0.
    """
    tile = tiling.TILE_SIZE
    scale = q.shape[-1] ** -0.5
    allowed = block_mask.repeat_interleave(tile, 1).repeat_interleave(tile, 2)
    allowed = allowed & layout.slot_valid.bool()[None, None, :]
    scores = torch.einsum('qhd,khd->hqk', q.float(), k.float()) * scale
    scores = scores.masked_fill(~allowed, float('-inf'))
    probs = torch.softmax(scores, dim=-1).nan_to_num(0.0)
    out = torch.einsum('hqk,khd->qhd', probs, v.float())
    out = out * layout.slot_valid.to(out.dtype)[:, None, None]
    return out.to(q.dtype)


def pooled_correction_attention(q: torch.Tensor, k: torch.Tensor,
                                v: torch.Tensor, block_mask: torch.Tensor,
                                layout: tiling.TileLayout) -> torch.Tensor:
    """Block-sparse attention with Sol-Attn's pooled term for the tiles
    that were skipped.

    Every skipped tile still contributes one term, built from the mean of
    its keys and the sum of its values, so the softmax denominator sees
    the whole sequence instead of only the selected tiles. Sol-Attn
    (arXiv 2607.24027, and comfy_kitchen's sol_attn) uses this to keep a
    very sparse routing honest; Veda skips outright, which is why a query
    tile whose scores are flat has nothing left to attend.

    Reference only: O(N^2) and fp32, for tests and for deciding whether
    the kernel is worth the work.

    Args:
        q, k, v: [N, H', D] tile order, N = n_tiles * 128.
        block_mask: [H', n_tiles, n_tiles] bool, rows are query tiles.
        layout: Tile layout (key slot validity, real rows per tile).

    Returns:
        [N, H', D] in q's dtype; rows of padding slots are 0.
    """
    tile = tiling.TILE_SIZE
    n_tiles = layout.n_tiles
    scale = q.shape[-1] ** -0.5
    valid = layout.slot_valid.bool()
    exact = block_mask.repeat_interleave(tile, 1).repeat_interleave(tile, 2)
    exact = exact & valid[None, None, :]
    scores = torch.einsum('qhd,khd->hqk', q.float(), k.float()) * scale
    scores = scores.masked_fill(~exact, float('-inf'))

    # One pooled key and value per tile, over its real rows only.
    counts = layout.valid_count.float().clamp(min=1)
    live = valid.view(n_tiles, tile, 1, 1).float()
    k_mean = (k.float().view(n_tiles, tile, *k.shape[1:]) * live).sum(1)
    k_mean = k_mean / counts[:, None, None]
    v_sum = (v.float().view(n_tiles, tile, *v.shape[1:]) * live).sum(1)

    pooled = torch.einsum('qhd,thd->hqt', q.float(), k_mean) * scale
    skipped = (~block_mask) & (layout.valid_count > 0)[None, None, :]
    skipped = skipped.repeat_interleave(tile, 1)   # one row per query slot
    pooled = pooled.masked_fill(~skipped, float('-inf'))

    # One softmax over both branches, as Sol-Attn does it: unnormalised
    # exponentials, and a pooled term weighs its tile's live row count in
    # the denominator because v_sum already carries that many values.
    both = torch.cat([scores, pooled], dim=-1)
    weights = torch.exp(both - both.amax(-1, keepdim=True))
    weights = torch.where(torch.isfinite(both), weights,
                          torch.zeros_like(weights))
    cut = scores.shape[-1]
    fine, coarse = weights[..., :cut], weights[..., cut:]
    out = (torch.einsum('hqk,khd->qhd', fine, v.float())
           + torch.einsum('hqt,thd->qhd', coarse, v_sum))
    total = fine.sum(-1) + (coarse * counts[None, None, :]).sum(-1)
    out = out / total.clamp(min=1e-20).permute(1, 0)[..., None]
    out = out * layout.slot_valid.to(out.dtype)[:, None, None]
    return out.to(q.dtype)


def dense_attention(q: torch.Tensor, k: torch.Tensor,
                    v: torch.Tensor) -> torch.Tensor:
    """Plain softmax attention, [S, H, D] -> [S, H, D] (fp32 math)."""
    scale = q.shape[-1] ** -0.5
    scores = torch.einsum('qhd,khd->hqk', q.float(), k.float()) * scale
    out = torch.einsum('hqk,khd->qhd', torch.softmax(scores, -1), v.float())
    return out.to(q.dtype)
