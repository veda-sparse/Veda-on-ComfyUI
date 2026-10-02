"""Portable backend: gather the selected key tiles, then plain SDPA.

Runs wherever torch runs (CUDA without FA4 / Triton, Apple MPS, CPU). Veda
keeps an almost fixed number of key tiles per query tile, so each query tile
becomes one short SDPA problem over its gathered keys; padding slots are
masked. The gather costs memory traffic proportional to the kept blocks,
which is what makes this slower than a real block-sparse kernel, but it is
exact and has no compiler or kernel dependency.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from . import base
from ..core import tiling

# Default bound on the bytes one gathered chunk (keys, values and, on
# backends that materialize it, the score matrix) may take.
DEFAULT_CHUNK_BYTES = 512 * 2**20


class TorchGatherBackend(base.Backend):
    """Gather + SDPA."""

    name = 'torch'
    display = 'PyTorch SDPA'
    dtypes = (torch.bfloat16, torch.float16, torch.float32)

    def __init__(self, chunk_bytes: int = DEFAULT_CHUNK_BYTES):
        self.chunk_bytes = chunk_bytes

    def attend(self, q, k, v, block_mask, layout):
        tile = tiling.TILE_SIZE
        n, n_video = layout.n_tiles, layout.n_video_tiles
        heads, dim = q.shape[1], q.shape[2]
        out = torch.empty_like(q)
        key_ok = layout.slot_valid.view(n, tile).bool()
        elem = q.element_size()
        if n_video < n:
            self._global_rows(q, k, v, layout, out)
        q_heads = q.view(n, tile, heads, dim).permute(2, 0, 1, 3)
        k_heads = k.view(n, tile, heads, dim).permute(2, 0, 1, 3)
        v_heads = v.view(n, tile, heads, dim).permute(2, 0, 1, 3)
        selected = block_mask[:, :n_video]
        counts = selected.sum(-1)
        kmax = int(counts.max())
        order = torch.argsort((~selected).to(torch.int8), dim=-1,
                              stable=True)[..., :kmax]
        slot_ok = (torch.arange(kmax, device=q.device)[None, None, :]
                   < counts[..., None])
        head_idx = torch.arange(heads, device=q.device)[:, None, None]
        # Keys + values, plus the fp32 score matrix SDPA may materialize.
        per_row = heads * kmax * tile * (2 * dim * elem + tile * 4)
        rows = max(1, self.chunk_bytes // per_row)
        for r0 in range(0, n_video, rows):
            r1 = min(n_video, r0 + rows)
            count = r1 - r0
            idx = order[:, r0:r1]
            keys = k_heads[head_idx, idx].reshape(heads * count, 1,
                                                  kmax * tile, dim)
            values = v_heads[head_idx, idx].reshape(heads * count, 1,
                                                    kmax * tile, dim)
            allowed = (key_ok[idx] & slot_ok[:, r0:r1, :, None]).reshape(
                heads * count, 1, 1, kmax * tile)
            query = q_heads[:, r0:r1].reshape(heads * count, 1, tile, dim)
            o = F.scaled_dot_product_attention(query, keys, values,
                                               attn_mask=allowed)
            out[r0 * tile:r1 * tile] = o.view(heads, count, tile, dim).permute(
                1, 2, 0, 3).reshape(count * tile, heads, dim)
        return out

    def _global_rows(self, q, k, v, layout, out):
        """Global query tiles attend every real key (dense rows)."""
        tile = tiling.TILE_SIZE
        start = layout.n_video_tiles * tile
        total = q.shape[0]
        allowed = layout.slot_valid.bool()[None, None, None, :]
        rows = max(1, self.chunk_bytes // (tile * total * 4))
        for head in range(q.shape[1]):
            keys = k[:, head][None, None]
            values = v[:, head][None, None]
            for r0 in range(start, total, rows * tile):
                r1 = min(total, r0 + rows * tile)
                query = q[r0:r1, head][None, None]
                o = F.scaled_dot_product_attention(query, keys, values,
                                                   attn_mask=allowed)
                out[r0:r1, head] = o[0, 0]


def create(info, chunk_bytes: int | None = None) -> base.Backend:
    del info  # runs anywhere torch runs
    return TorchGatherBackend(chunk_bytes or DEFAULT_CHUNK_BYTES)
