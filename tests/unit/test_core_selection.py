import math

import pytest
import torch

from veda_comfy.core import selection
from veda_comfy.core import tiling


def _layout():
    # reference: one 1x8x16 span; target 5x8x8 tiled 2x8x8 -> 3 tiles
    spans = [tiling.TiledSpan(20, (1, 8, 16), tiling.TileShape(1, 8, 16)),
             tiling.TiledSpan(200, (5, 8, 8), tiling.TileShape(2, 8, 8))]
    return tiling.build_tile_layout(spans, 200 + 320 + 10)


def test_budget_sparsity():
    assert selection.Budget.sparsity(90.0).ratio == pytest.approx(0.1)
    assert selection.Budget.sparsity(0.0).keeps_all
    with pytest.raises(ValueError):
        selection.Budget.sparsity(100.0)


def test_equal_cost_budget():
    budget = selection.Budget(ratio=0.1)
    # 320 real tokens -> n_ideal 3; padded to 4 columns -> 0.1 * 9 / 4
    assert budget.per_row(320, 4) == pytest.approx(0.225)
    assert selection.Budget(tiles=5).per_row(320, 4) == 5


def test_split_budget_and_bresenham_mean():
    assert selection.split_budget(2.3, 10) == (2, 3, 0.3)
    assert selection.split_budget(12, 10) == (10, 10, 0.0)
    assert selection.split_budget(0.2, 10)[0] == 1
    extra = selection.bresenham_extra(1000, 0.3, 'cpu')
    assert int(extra.sum()) == 300


def test_select_forces_diagonal_and_respects_blocks():
    layout = _layout()
    n_video = layout.n_video_tiles
    assert layout.n_ref_tiles == 1 and n_video == 4
    scores = torch.randn(2, n_video, n_video)
    scores[:, :, 0] = -1e9  # reference scores worst; its own row keeps it
    blocks = selection.column_blocks(layout, selection.Budget(tiles=1),
                                     selection.Budget(tiles=1))
    index, keep = selection.select(scores, layout, blocks)
    mask = selection.block_mask(index, keep, layout)
    video = mask[:, :n_video, :n_video]
    assert video.diagonal(dim1=1, dim2=2).all()
    # one reference column and one target column per row (budget 1 each)
    assert (video[:, :, :1].sum(-1) == 1).all()
    assert (video[:, :, 1:].sum(-1) == 1).all()
    # global rows / columns always on
    assert mask[:, n_video:, :].all()
    assert mask[:, :, n_video:].all()


def test_keep_all_block_keeps_every_nonempty_tile():
    layout = _layout()
    scores = torch.randn(1, 4, 4)
    blocks = selection.column_blocks(layout, selection.Budget(ratio=1.0),
                                     selection.Budget(ratio=1.0))
    index, keep = selection.select(scores, layout, blocks)
    mask = selection.block_mask(index, keep, layout)
    assert torch.equal(mask[0], layout.kv_ok.expand(layout.n_tiles, -1))


def test_kept_count_matches_fractional_budget():
    spans = [tiling.TiledSpan(0, (16, 16, 16), tiling.TileShape(4, 4, 8))]
    layout = tiling.build_tile_layout(spans, 4096)
    n = layout.n_video_tiles
    budget = selection.Budget(ratio=0.1)
    blocks = selection.column_blocks(layout, budget, budget)
    index, keep = selection.select(torch.randn(3, n, n), layout, blocks)
    expected = budget.per_row(4096, n)
    assert keep.sum(-1).float().mean().item() == pytest.approx(
        expected, abs=1.0 / n)
    assert math.floor(expected) <= keep.sum(-1).min()


def test_adaptive_keeps_rule_2_and_only_real_tiles():
    layout = _layout()
    n_video = layout.n_video_tiles
    torch.manual_seed(0)
    scores = torch.randn(3, n_video, n_video)
    budget = selection.Budget(tiles=1)
    blocks = selection.column_blocks(layout, budget, budget)
    index, keep = selection.select_adaptive(scores, layout, blocks, 1.3)
    mask = selection.block_mask(index, keep, layout)
    video = mask[:, :n_video, :n_video]
    assert video.diagonal(dim1=1, dim2=2).all()         # rule 2
    assert not (video & ~layout.kv_ok[:n_video]).any()  # never an empty tile
    assert mask[:, n_video:, :].all()                   # global rows
    assert mask[:, :, n_video:].all()                   # global columns


def test_adaptive_matches_sol_attns_published_thresholds():
    """tau is Sol-Attn's parameter, so on a Gaussian score row it has to
    behave like the Gaussian tail its tooltip quotes: 1.0 keeps ~16% of
    key tiles, 1.5 ~7%, 2.0 ~2.7%."""
    spans = [tiling.TiledSpan(0, (72, 24, 42), tiling.TileShape(8, 4, 4))]
    layout = tiling.build_tile_layout(spans, 72 * 24 * 42 + 10)
    n_video = layout.n_video_tiles
    budget = selection.Budget(tiles=32)
    blocks = selection.column_blocks(layout, budget, budget)
    torch.manual_seed(0)
    scores = torch.randn(1, n_video, n_video)
    for tau, want in ((1.0, 0.16), (1.5, 0.07), (2.0, 0.027)):
        index, keep = selection.select_adaptive(scores, layout, blocks, tau)
        mask = selection.block_mask(index, keep, layout)
        share = mask[:, :n_video, :n_video].sum(-1).float().mean() / n_video
        assert share == pytest.approx(want, abs=0.01)


def test_adaptive_never_falls_back_to_the_diagonal_alone():
    """Sol-Attn covers what it does not route with a pooled term; Veda
    skips it. A row with no outlier must still keep a floor."""
    spans = [tiling.TiledSpan(0, (16, 16, 16), tiling.TileShape(2, 8, 8))]
    layout = tiling.build_tile_layout(spans, 16 ** 3 + 10)
    n_video = layout.n_video_tiles
    budget = selection.Budget(tiles=4)
    blocks = selection.column_blocks(layout, budget, budget)
    flat = torch.zeros(1, n_video, n_video)
    index, keep = selection.select_adaptive(flat, layout, blocks, 1.3)
    kept = selection.block_mask(index, keep, layout)[:, :n_video, :n_video]
    assert (kept.sum(-1) == selection._ADAPTIVE_FLOOR).all()


def test_adaptive_has_no_row_to_row_alternation():
    """Identical rows keep identical counts: an adaptive threshold has no
    remainder to spread between neighbouring rows."""
    layout = _layout()
    n_video = layout.n_video_tiles
    scores = torch.randn(1, 1, n_video).expand(2, n_video, n_video)
    blocks = selection.column_blocks(layout, selection.Budget(tiles=1),
                                     selection.Budget(tiles=2.5))
    index, keep = selection.select_adaptive(scores, layout, blocks, 1.3)
    counts = selection.block_mask(index, keep, layout)[
        :, :n_video, :n_video].sum(-1)
    # They differ only by the forced diagonal, never by a spread remainder.
    assert counts.max() - counts.min() <= 1


def test_a_fractional_fixed_budget_alternates_between_neighbouring_rows():
    """The flicker mechanism, at the two scales that matter. Tile rows are
    ordered with the time block varying fastest (tiling.span_tiles), so
    neighbouring rows are neighbours in time: on the small grid a
    two-stage first pass runs at, every other temporal block gets 20%
    more context than the one beside it."""
    for per_row, n_tiles, swing in [(54.1, 594, 0.02),    # trained 1.00x
                                    (5.5, 72, 0.20)]:     # two-stage 0.33x
        lo, hi, frac = selection.split_budget(per_row, n_tiles)
        extra = selection.bresenham_extra(n_tiles, frac, 'cpu')
        counts = (lo + extra.to(torch.long))[:12]
        assert hi - lo == 1
        assert (hi - lo) / lo == pytest.approx(swing, abs=0.005)
        # The small grid alternates every row; the trained one rarely.
        flips = (counts[1:] != counts[:-1]).float().mean()
        assert flips > 0.9 if per_row < 10 else flips < 0.3


def test_a_pooled_term_rescues_a_budget_too_small_to_attend():
    """Sol-Attn covers what it does not route with one term per block, so
    the softmax still sees the whole sequence. Veda skips outright, which
    is what makes a tight budget - the two-stage first pass, where 32
    tiles is most of the grid - fall apart. Random scores exaggerate the
    gap (a trained predictor picks better tiles), but the direction is
    the point."""
    from veda_comfy.core import reference
    torch.manual_seed(0)
    spans = [tiling.TiledSpan(0, (16, 16, 16), tiling.TileShape(2, 8, 8))]
    layout = tiling.build_tile_layout(spans, 16 ** 3)
    n_video, slots = layout.n_video_tiles, layout.num_slots
    q, k, v = (torch.randn(slots, 2, 128) * 0.5 for _ in range(3))
    full = reference.dense_attention(q, k, v)
    real = layout.slot_valid.bool()

    def error(out):
        return ((out[real] - full[real]).pow(2).sum()
                / full[real].pow(2).sum()).sqrt().item()

    previous = None
    for tiles in (1, 4, 16):
        budget = selection.Budget(tiles=tiles)
        blocks = selection.column_blocks(layout, budget, budget)
        index, keep = selection.select(
            torch.randn(2, n_video, n_video), layout, blocks)
        mask = selection.block_mask(index, keep, layout)
        strict = error(reference.block_sparse_attention(q, k, v, mask, layout))
        pooled = error(
            reference.pooled_correction_attention(q, k, v, mask, layout))
        assert pooled < strict / 4, (tiles, strict, pooled)
        # The pooled result also degrades gracefully as the budget shrinks.
        if previous is not None:
            assert pooled <= previous + 0.05
        previous = pooled


def test_the_pooled_term_is_exact_when_nothing_is_skipped():
    from veda_comfy.core import reference
    torch.manual_seed(1)
    spans = [tiling.TiledSpan(0, (4, 8, 8), tiling.TileShape(2, 8, 8))]
    layout = tiling.build_tile_layout(spans, 4 * 8 * 8)
    slots = layout.num_slots
    q, k, v = (torch.randn(slots, 2, 128) * 0.5 for _ in range(3))
    everything = torch.ones(2, layout.n_tiles, layout.n_tiles,
                            dtype=torch.bool)
    strict = reference.block_sparse_attention(q, k, v, everything, layout)
    pooled = reference.pooled_correction_attention(q, k, v, everything,
                                                   layout)
    torch.testing.assert_close(pooled, strict, rtol=1e-4, atol=1e-4)


def test_the_adaptive_floor_is_the_pooled_term_standing_in():
    """With Sol-Attn's correction the skipped tiles still reach the
    softmax, so a flat row no longer needs topping up - which is the
    only reason the floor exists."""
    spans = [tiling.TiledSpan(0, (16, 16, 16), tiling.TileShape(2, 8, 8))]
    layout = tiling.build_tile_layout(spans, 16 ** 3 + 10)
    n_video = layout.n_video_tiles
    budget = selection.Budget(tiles=4)
    blocks = selection.column_blocks(layout, budget, budget)
    flat = torch.zeros(1, n_video, n_video)
    for floor, want in ((True, selection._ADAPTIVE_FLOOR), (False, 1)):
        index, keep = selection.select_adaptive(flat, layout, blocks, 1.3,
                                                floor=floor)
        kept = selection.block_mask(index, keep,
                                    layout)[:, :n_video, :n_video]
        assert (kept.sum(-1) == want).all(), (floor, kept.sum(-1))
