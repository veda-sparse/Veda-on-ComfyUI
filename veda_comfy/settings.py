"""User settings of the Veda node and their parsing.

Every parse error is a ValueError whose message is shown on the node as is,
so it says what was wrong and how to write it instead.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Sequence

from .core import selection

# Separators users type: ASCII and full-width commas / semicolons, spaces.
_SPLIT = re.compile(r'[\s,;，；、]+')
_RANGE = re.compile(r'^(\d+)\s*[-~–—]\s*(\d+)$')


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
            raise ValueError(f'{what}: cannot read {token!r}. Use 0-based '
                             'indices and ranges, e.g. "0, 1, 47-49".')
        lo, hi = sorted((int(match.group(1)), int(match.group(2))))
        indices.update(range(lo, hi + 1))
    return frozenset(indices)


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


@dataclasses.dataclass(frozen=True)
class VedaSettings:
    """Everything the node configures.

    Attributes:
        current: Budget of the target-video key tiles.
        history: Budget of the condition key tiles (keyframes, references);
            keeps_all means conditions stay dense (untiled).
        dense_layers: 0-based DiT blocks that run full attention.
        dense_steps: 0-based sampling steps that run full attention.
        backend: One of backends.CHOICES.
        untrained_geometry: 'sparse' (use the nearest plan) or 'dense' for
            sizes no bundled plan was trained on.
        verbose: Log every decision.
    """

    current: selection.Budget
    history: selection.Budget
    dense_layers: frozenset[int] = frozenset()
    dense_steps: frozenset[int] = frozenset()
    backend: str = 'auto'
    untrained_geometry: str = 'sparse'
    verbose: bool = False

    def describe(self) -> str:
        parts = [f'current {self.current.describe()}',
                 f'history {self.history.describe()}']
        if self.dense_layers:
            parts.append(f'full-attention layers '
                         f'{format_index_list(self.dense_layers)}')
        if self.dense_steps:
            parts.append(f'full-attention steps '
                         f'{format_index_list(self.dense_steps)}')
        return ' · '.join(parts)


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
