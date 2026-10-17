"""User settings of the Veda node and their parsing.

Every parse error is a ValueError whose message is shown on the node as is,
so it says what was wrong and how to write it instead.
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Sequence

from . import status as veda_status
from .core import selection

# Separators users type: ASCII and full-width commas / semicolons, spaces.
_SPLIT = re.compile(r'[\s,;，；、]+')
_RANGE = re.compile(r'^(\d+)\s*[-~–—]\s*(\d+)$')
_PERCENT = re.compile(r'^(\d+(?:\.\d+)?)\s*[%％]$')
_TILES = re.compile(r'^(\d+)$')


def parse_index_list(text: str, what: str) -> frozenset[int]:
    """'0, 1, 47-49' -> {0, 1, 47, 48, 49} (0-based, inclusive ranges).

    Raises:
        ValueError: On anything that is not an index or a range.
    """
    indices: set[int] = set()
    text = (text or '').strip()
    # Normalize spaces around range dashes so '47 - 49' is one token.
    text = re.sub(r'\s*([-~–—])\s*', r'\1', text)
    for token in _SPLIT.split(text):
        if not token:
            continue
        if token.isdigit():
            indices.add(int(token))
            continue
        match = _RANGE.match(token)
        if match is None:
            raise ValueError(veda_status.bilingual(
                f'{what}: cannot read {token!r}. Use 0-based indices and '
                'ranges, e.g. "0, 1, 47-49".',
                f'{what}：看不懂 {token!r}。请写从 0 开始的下标和范围，'
                '例如 "0, 1, 47-49"。'))
        lo, hi = sorted((int(match.group(1)), int(match.group(2))))
        indices.update(range(lo, hi + 1))
    return frozenset(indices)


TRAINED = 'trained'
# The default budget, in key tiles per query tile. An equal-cost ratio
# scales the kept tiles with the grid area, so on the small grids the
# two-stage workflows start from (a third of the trained grid) it
# collapses to ~5 tiles a row and the Bresenham remainder then alternates
# 5/6 between neighbouring tiles - which are neighbours in time, so the
# context swings 20% from one temporal block to the next and the video
# flickers. An absolute count does not move with the grid.
DEFAULT_BUDGET = '32'


def parse_sparsity(text: str, what: str,
                   trained: selection.Budget | None = None
                   ) -> selection.Budget:
    """'90%' -> skip 90% of the key tiles; '24' -> keep exactly 24 tiles.

    Args:
        text: What the user typed. 'trained' (the default) takes the
            budget the chosen predictor declares, so nobody has to know
            that T2VA was trained at 90% and R2VA at 32 tiles.
        what: Input name, for the error message.
        trained: The predictor's own budget, required for 'trained'.

    Raises:
        ValueError: On anything else, or a percentage outside [0, 100).
    """
    value = (text or '').strip()
    if value.lower() == TRAINED and trained is not None:
        return trained
    match = _PERCENT.match(value)
    if match is not None:
        percent = float(match.group(1))
        if percent >= 100.0:
            raise ValueError(veda_status.bilingual(
                f'{what}: {value} would skip every tile; use a percentage '
                'below 100%, e.g. "90%".',
                f'{what}：{value} 会跳过所有 tile；请写小于 100% 的比例，'
                '例如 "90%"。'))
        return selection.Budget.sparsity(percent)
    match = _TILES.match(value)
    if match is not None and int(match.group(1)) > 0:
        return selection.Budget(tiles=float(match.group(1)))
    raise ValueError(veda_status.bilingual(
        f'{what}: cannot read {value!r}. Write "trained" for the budget '
        'this predictor was trained at, a sparsity percentage such as '
        '"90%" (skip 90% of the tiles), or a whole number such as "24" '
        '(keep 24 tiles of 128 tokens per query tile).',
        f'{what}：看不懂 {value!r}。可以写 "trained"（用该打分器训练时的'
        '预算）、稀疏比例如 "90%"（跳过 90% 的 tile），'
        '或整数如 "24"（每个 query tile 保留 24 个 128-token 的 tile）。'))


def parse_resolution_budgets(
        text: str) -> tuple[tuple[float, selection.Budget], ...]:
    """Parse '1:32, 2:64, 64' into MP upper bounds and tile budgets.

    The final entry is the budget above the last threshold.
    """
    parts = [part.strip() for part in (text or '').split(',')]
    if len(parts) < 2 or any(not part for part in parts):
        raise ValueError(
            'resolution_budgets: use ascending MP limits followed by a '
            'final budget, e.g. "1:32, 2:64, 64".')
    rules = []
    previous = 0.0
    for part in parts[:-1]:
        if part.count(':') != 1:
            raise ValueError(
                'resolution_budgets: each MP band must be "limit:tiles".')
        bound_text, budget_text = part.split(':')
        try:
            bound = float(bound_text.strip())
        except ValueError as error:
            raise ValueError(
                f'resolution_budgets: invalid MP limit {bound_text!r}.'
            ) from error
        if not (0 < bound < float('inf') and bound > previous):
            raise ValueError(
                'resolution_budgets: MP limits must be finite, positive '
                'and strictly increasing.')
        budget = parse_sparsity(
            budget_text.strip(), 'resolution_budgets')
        if budget.tiles is None:
            raise ValueError(
                'resolution_budgets: use positive whole-number tile '
                'budgets, not percentages.')
        rules.append((bound, budget))
        previous = bound
    final = parse_sparsity(parts[-1], 'resolution_budgets')
    if final.tiles is None:
        raise ValueError(
            'resolution_budgets: final budget must be a tile count.')
    rules.append((float('inf'), final))
    return tuple(rules)


def budget_for_grid(
    grid: tuple[int, int, int],
    fallback: selection.Budget,
    rules: tuple[tuple[float, selection.Budget], ...],
) -> selection.Budget:
    """Choose a generated attention budget from the H3 spatial grid."""
    if not rules:
        return fallback
    # H3 video tokens represent 32x32 output pixels per spatial cell.
    megapixels = (grid[1] * 32) * (grid[2] * 32) / 1_000_000
    for limit, budget in rules:
        if megapixels <= limit:
            return budget
    return rules[-1][1]


@dataclasses.dataclass(frozen=True)
class TemporalPolicy:
    """Calibrated, opt-in generated tile scaling above a spatial threshold."""

    min_mp: float = 1.0
    baseline: int = 72
    target: int = 87
    target_tiles: int = 97
    max_tiles: int = 168
    strength: float = 1.0

    def __post_init__(self):
        if not math.isfinite(self.strength) or self.strength < 0:
            raise ValueError('video_length_scaling must be finite and >= 0')
        if not math.isfinite(self.min_mp) or self.min_mp <= 0:
            raise ValueError(
                'temporal_min_megapixels must be positive and finite')
        if self.baseline < 1 or self.target <= self.baseline:
            raise ValueError('temporal_target_positions must exceed a positive '
                             'temporal_baseline_positions')
        if self.target_tiles < 1 or self.max_tiles < 1:
            raise ValueError('temporal tile budgets must be positive')
        if self.max_tiles < self.target_tiles:
            raise ValueError('temporal_max_generated_budget must be at least '
                             'temporal_target_generated_budget')


def effective_generated_budget(
    grid: tuple[int, int, int],
    fallback: selection.Budget,
    rules: tuple[tuple[float, selection.Budget], ...],
    policy: TemporalPolicy | None = None,
) -> selection.Budget:
    """Single source of truth for attention and status displays."""
    budget = budget_for_grid(grid, fallback, rules)
    if policy is None or not rules or budget.tiles is None:
        return budget
    mp = grid[1] * grid[2] * 1024 / 1_000_000
    if mp <= policy.min_mp or grid[0] <= policy.baseline:
        return budget
    base = budget.tiles
    # Temporal growth is proportional to the current spatial tile budget.
    # A relative multiplier preserves user intent when spatial budgets change.
    # With strength=1, base=64 and T=87 the result is exactly 97.
    relative_increase = ((policy.target_tiles / 64.0) - 1.0)
    increase = (grid[0] - policy.baseline) / (policy.target - policy.baseline)
    scaled = math.ceil(base * (1.0 + policy.strength *
                               relative_increase * increase))
    # A temporal cap cannot reduce a budget explicitly selected by the
    # spatial rules. Users may configure larger spatial budgets.
    return selection.Budget(tiles=float(max(
        math.ceil(base), min(policy.max_tiles, scaled))))


def format_index_list(indices: Sequence[int] | frozenset[int]) -> str:
    """{0, 1, 47, 48, 49} -> '0-1, 47-49'."""
    values = sorted(indices)
    parts, start = [], None
    for i, value in enumerate(values):
        if start is None:
            start = value
        if i + 1 == len(values) or values[i + 1] != value + 1:
            parts.append(str(start) if start == value else f'{start}-{value}')
            start = None
    return ', '.join(parts)


def format_budget(budget: selection.Budget) -> str:
    """'90%' or '24 tiles' (the way the user wrote it)."""
    if budget.tiles is not None:
        return f'{budget.tiles:g} tiles'
    return f'{100.0 * (1.0 - min(budget.ratio, 1.0)):g}%'


@dataclasses.dataclass(frozen=True)
class VedaSettings:
    """Everything the node configures.

    Attributes:
        generated: Budget of the generated (target) video's key tiles.
        reference: Budget of the reference / condition key tiles (first /
            last frames, guide frames, reference images and videos);
            keeps_all means they stay in full attention (untiled).
        dense_layers: 0-based DiT blocks that run full attention.
        dense_steps: 0-based sampling steps that run full attention.
        verbose: Show performance diagnostics and log every decision.
    """

    generated: selection.Budget
    reference: selection.Budget
    dense_layers: frozenset[int] = frozenset()
    dense_steps: frozenset[int] = frozenset()
    verbose: bool = False
    resolution_budgets: tuple[tuple[float, selection.Budget], ...] = ()
    temporal_scaling: bool = False
    temporal_policy: TemporalPolicy = dataclasses.field(
        default_factory=TemporalPolicy)

    def describe(self) -> str:
        return (f'generated {format_budget(self.generated)} · reference '
                f'{format_budget(self.reference)}')

    def describe_full_attention(self) -> str | None:
        parts = []
        if self.dense_layers:
            parts.append(f'layers {format_index_list(self.dense_layers)}')
        if self.dense_steps:
            parts.append(f'steps {format_index_list(self.dense_steps)}')
        return ' · '.join(parts) or None


def step_index(sigma: float, sample_sigmas: Sequence[float]) -> int | None:
    """0-based sampling step whose interval contains sigma.

    Step i integrates from sample_sigmas[i] down to sample_sigmas[i + 1];
    multi-stage samplers evaluate the model inside that interval, which
    still counts as step i.
    """
    sigmas = [float(s) for s in sample_sigmas]
    if len(sigmas) < 2:
        return None
    tolerance = 1e-5 * max(1.0, abs(sigma))
    for i in range(len(sigmas) - 1):
        high, low = sigmas[i], sigmas[i + 1]
        if high + tolerance >= sigma > low + tolerance:
            return i
    if sigma >= sigmas[0]:
        return 0
    return len(sigmas) - 2 if sigma >= sigmas[-1] - tolerance else None
