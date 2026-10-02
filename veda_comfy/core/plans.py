"""Tile plans: which tile shape every (layer, head) uses on one geometry.

A plan is searched offline for one target token grid (T, H, W) and ships
inside the predictor bundle. Plans are only exact for the grid they were
searched on; any other grid can still use a plan (tile shapes pad any grid),
it is just outside what the predictor was trained on. `PlanTable.select`
therefore says how it chose, so the UI can tell the user.
"""

from __future__ import annotations

import dataclasses
import math

import torch

from . import tiling


@dataclasses.dataclass(frozen=True)
class HeadGroup:
    """Heads of one layer that share a tile shape."""

    shape: tiling.TileShape
    heads: torch.Tensor  # [H'] int64


@dataclasses.dataclass
class TilePlan:
    """Per-(layer, head) tile shapes for one geometry.

    Attributes:
        name: Geometry name, e.g. '16x9_t37' (aspect and latent frames).
        grid: Target token grid (T, H, W) the plan was searched on.
        shapes: Distinct shapes referenced by `head_shape`.
        head_shape: [num_layers][num_heads] index into `shapes`.
    """

    name: str
    grid: tuple[int, int, int]
    shapes: list[tiling.TileShape]
    head_shape: list[list[int]]

    def __post_init__(self):
        self._groups: dict[tuple[int, str], list[HeadGroup]] = {}

    @property
    def num_layers(self) -> int:
        return len(self.head_shape)

    @property
    def num_heads(self) -> int:
        return len(self.head_shape[0]) if self.head_shape else 0

    def head_groups(self, layer: int,
                    device: torch.device | str) -> list[HeadGroup]:
        """Groups of heads sharing a shape (cached per layer and device)."""
        key = (layer, str(device))
        if key not in self._groups:
            row = torch.tensor(self.head_shape[layer])
            self._groups[key] = [
                HeadGroup(self.shapes[i],
                          torch.nonzero(row == i).view(-1).to(device))
                for i in sorted(set(self.head_shape[layer]))
            ]
        return self._groups[key]

    def transposed(self) -> TilePlan:
        """The H<->W mirrored plan (for portrait / landscape fallback)."""
        t, h, w = self.grid
        return TilePlan(self.name + '_T', (t, w, h),
                        [s.transposed() for s in self.shapes],
                        [list(r) for r in self.head_shape])

    @classmethod
    def from_json(cls, data: dict) -> TilePlan:
        return cls(name=data['geometry'], grid=tuple(data['grid']),
                   shapes=[tiling.TileShape.parse(s) for s in data['shapes']],
                   head_shape=[list(r) for r in data['head_shape']])


@dataclasses.dataclass(frozen=True)
class PlanChoice:
    """The plan picked for a grid and how it was picked.

    Attributes:
        plan: The plan to use.
        exact: The plan was searched on exactly this grid.
        how: Human-readable reason, shown to the user.
    """

    plan: TilePlan
    exact: bool
    how: str


def _aspect(grid) -> float:
    return grid[2] / grid[1]


class PlanTable:
    """All plans of a bundle; picks the plan for a target grid."""

    def __init__(self, plans: list[TilePlan]):
        if not plans:
            raise ValueError('a predictor bundle needs at least one plan')
        self.plans = {p.name: p for p in plans}
        self._choices: dict[tuple[int, int, int], PlanChoice] = {}

    def select(self, grid: tuple[int, int, int]) -> PlanChoice:
        """Plan for a target token grid (T, H, W).

        Rule: the plan searched on this exact grid; else a plan with the
        same frame grid (H, W) and the nearest T; else the plan (or its H<->W
        transpose) with the nearest aspect ratio, then the nearest T, then
        the least padding. Never fails: an approximate plan still gives a
        valid tiling, it is just outside the trained distribution.
        """
        grid = tuple(int(g) for g in grid)
        if grid in self._choices:
            return self._choices[grid]
        plans = list(self.plans.values())
        exact = [p for p in plans if p.grid == grid]
        if exact:
            choice = PlanChoice(exact[0], True, f'trained plan {exact[0].name}')
        else:
            same_frame = [p for p in plans if p.grid[1:] == grid[1:]]
            if same_frame:
                plan = min(same_frame,
                           key=lambda p: (abs(p.grid[0] - grid[0]), p.grid[0]))
                choice = PlanChoice(
                    plan, False,
                    f'nearest length: plan {plan.name} (trained for '
                    f'{plan.grid[0]} latent frames, this video has {grid[0]})')
            else:
                candidates = plans + [p.transposed() for p in plans]
                target = math.log(_aspect(grid))

                def cost(p):
                    padding = sum(s.num_tiles(grid) for s in p.shapes)
                    return (round(abs(math.log(_aspect(p.grid)) - target), 3),
                            abs(p.grid[0] - grid[0]), padding)

                plan = min(candidates, key=cost)
                choice = PlanChoice(
                    plan, False,
                    f'nearest aspect: plan {plan.name} (trained for a '
                    f'{plan.grid[2] * 32}x{plan.grid[1] * 32} canvas, this '
                    f'video is {grid[2] * 32}x{grid[1] * 32})')
        self._choices[grid] = choice
        return choice

    def summary(self) -> str:
        """e.g. '16:9, 9:16, 1:1, 4:3 x latent frames 37/72/102'."""
        aspects, lengths = [], set()
        for name in sorted(self.plans):
            aspect, _, t = name.partition('_t')
            label = aspect.replace('x', ':')
            if label not in aspects:
                aspects.append(label)
            if t.isdigit():
                lengths.add(int(t))
        frames = '/'.join(str(t) for t in sorted(lengths))
        return f'{", ".join(aspects)} x latent frames {frames}'
