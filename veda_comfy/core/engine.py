"""One Veda attention call, end to end, on any device.

For each head group of the layer's tile plan (heads sharing a tile shape),
in chunks of heads bounded by memory:

    gather q/k/v into tile order -> pool video tiles -> predictor logits ->
    top-k selection -> block mask -> backend kernel -> scatter back

Everything except the kernel is device-agnostic torch, identical for every
backend. Caches (layout specs, tile layouts, plan choices) live on the
engine, i.e. per patched model and device, and are dropped with it.
"""

from __future__ import annotations

import weakref

import torch

from . import bundle as veda_bundle
from . import h3_layout
from . import plans as veda_plans
from . import predictor
from . import selection
from . import tiling

# Bounds of one tile-ordered q / k / v / out copy of a head chunk. Four of
# them are alive at once, plus the backend's own temporaries.
_MIN_CHUNK_BYTES = 64 * 2**20
_MAX_CHUNK_BYTES = 1024 * 2**20
_DEFAULT_CHUNK_BYTES = 256 * 2**20


class Stats:
    """Kept-tile accounting, accumulated on the device and read once."""

    def __init__(self):
        self.kept = None
        self.total = 0

    def add(self, keep: torch.Tensor) -> None:
        count = keep.sum()
        self.kept = count if self.kept is None else self.kept + count
        self.total += keep.shape[0] * keep.shape[1] * keep.shape[1]

    def kept_fraction(self) -> float | None:
        if self.kept is None or not self.total:
            return None
        return float(self.kept) / self.total


class VedaEngine:
    """Sparse attention for one predictor bundle and settings on a device.

    Args:
        bundle: The predictor bundle.
        current: Budget of the target-video key tiles.
        history: Budget of the condition key tiles; keeps_all leaves the
            conditions global (dense both ways), which is also what happens
            when the layout has none.
        backend: A `backends.base.Backend`.
        device: The device of q / k / v.
    """

    def __init__(self, bundle: veda_bundle.PredictorBundle,
                 current: selection.Budget, history: selection.Budget,
                 backend, device: torch.device):
        self.bundle = bundle
        self.current = current
        self.history = history
        self.backend = backend
        self.device = device
        self.stats = Stats()
        self._specs = weakref.WeakKeyDictionary()
        self._tile_layouts: dict[tuple, tiling.TileLayout] = {}
        self._pinned = device.type == 'cuda'

    # -- caches ---------------------------------------------------------

    def layout_spec(self, layout) -> h3_layout.LayoutSpec:
        """LayoutSpec of a ComfyUI PackedLayout (cached per layout object).

        Raises:
            h3_layout.LayoutError: If the layout cannot be mapped.
        """
        try:
            spec = self._specs.get(layout)
        except TypeError:  # not weak-referenceable: describe every time
            return h3_layout.describe(layout)
        if spec is None:
            spec = h3_layout.describe(layout)
            self._specs[layout] = spec
        return spec

    def plan_for(self, spec: h3_layout.LayoutSpec) -> veda_plans.PlanChoice:
        return self.bundle.plans.select(spec.target.grid)

    def _tile_layout(self, spec: h3_layout.LayoutSpec,
                     shape: tiling.TileShape) -> tiling.TileLayout:
        tile_history = not self.history.keeps_all
        key = (spec, shape, tile_history)
        layout = self._tile_layouts.get(key)
        if layout is None:
            spans = []
            if tile_history:
                spans = [tiling.TiledSpan(s.start, s.grid,
                                          tiling.least_padding_shape(s.grid))
                         for s in spec.history]
            spans.append(tiling.TiledSpan(spec.target.start, spec.target.grid,
                                          shape))
            layout = tiling.build_tile_layout(spans, spec.seq_len,
                                              self.device)
            self._tile_layouts[key] = layout
        return layout

    def _weights(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Layer projections on the device ([H, 3D, D] bf16 each).

        Host copies stay authoritative so the predictor follows ComfyUI's
        offloading; on CUDA they are pinned once so the copy is async.
        """
        pq, pk = self.bundle.proj_q[layer], self.bundle.proj_k[layer]
        if self._pinned and not pq.is_pinned():
            try:
                pq = self.bundle.proj_q[layer] = pq.pin_memory()
                pk = self.bundle.proj_k[layer] = pk.pin_memory()
            except RuntimeError:
                self._pinned = False
        return (pq.to(self.device, non_blocking=True),
                pk.to(self.device, non_blocking=True))

    def _chunk_bytes(self) -> int:
        if self.device.type == 'cuda':
            free, _ = torch.cuda.mem_get_info(self.device)
            return int(min(max(free // 16, _MIN_CHUNK_BYTES),
                           _MAX_CHUNK_BYTES))
        return _DEFAULT_CHUNK_BYTES

    def reset(self) -> None:
        """Drops per-geometry caches (between sampling runs)."""
        self._tile_layouts.clear()
        self._specs = weakref.WeakKeyDictionary()
        self.stats = Stats()

    # -- the call -------------------------------------------------------

    @torch.no_grad()
    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  layer: int, spec: h3_layout.LayoutSpec,
                  plan: veda_plans.TilePlan) -> torch.Tensor:
        """Veda block-sparse attention of one layer.

        Args:
            q, k, v: [S, H, D] (any strides; post QK-norm and RoPE).
            layer: DiT block index.
            spec: Layout spec of this sequence.
            plan: Tile plan to use (from plan_for()).

        Returns:
            [S, H, D] contiguous, q's dtype.
        """
        seq_len, heads, dim = q.shape
        dtype = q.dtype
        if dtype not in self.backend.dtypes:
            q, k, v = (t.to(self.backend.dtypes[0]) for t in (q, k, v))
        out = q.new_empty(seq_len + 1, heads, dim)
        proj_q, proj_k = self._weights(layer)
        chunk_bytes = self._chunk_bytes()
        for group in plan.head_groups(layer, self.device):
            layout = self._tile_layout(spec, group.shape)
            blocks = selection.column_blocks(layout, self.current,
                                             self.history)
            per_head = layout.num_slots * dim * q.element_size()
            step = max(1, chunk_bytes // per_head)
            for heads_chunk in group.heads.split(step):
                q_t = tiling.gather_tiles(q, layout, heads_chunk)
                k_t = tiling.gather_tiles(k, layout, heads_chunk)
                scores = predictor.tile_logits(
                    predictor.pool_video_tiles(q_t, layout),
                    predictor.pool_video_tiles(k_t, layout),
                    proj_q.index_select(0, heads_chunk),
                    proj_k.index_select(0, heads_chunk))
                index, keep = selection.select(scores, layout, blocks)
                del scores
                self.stats.add(keep)
                mask = selection.block_mask(index, keep, layout)
                v_t = tiling.gather_tiles(v, layout, heads_chunk)
                o_t = self.backend.attend(q_t, k_t, v_t, mask, layout)
                del q_t, k_t, v_t, mask
                tiling.scatter_tiles_(out, o_t, layout, heads_chunk)
        out = out[:seq_len]
        return out if out.dtype == dtype else out.to(dtype)
