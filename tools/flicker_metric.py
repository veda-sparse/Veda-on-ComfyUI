"""Measures periodic temporal flicker in a generated video.

    python tools/flicker_metric.py out/a.mp4 out/b.mp4 --latent-t 37

Veda tiles the time axis, so a selection that is unstable from one tile
to the next modulates the attended context at the tile cadence and shows
up as flicker with a period, not as noise. This reports that directly:
the per-frame difference signal, its spectrum, and how much energy sits
at the period a tile of `--tile-depth` latent frames would produce.

A dense render is the control. Veda should not raise the peak at the
tile period above it; a render that does is flickering at the cadence
the tiling predicts, which is the signature the two-stage reports
describe (see docs/features/core_selection.md).
"""

from __future__ import annotations

import argparse
import math
import os

import av
import numpy as np

# H3 packs 17 output frames per 5 latent frames after the first two.
FRAMES_PER_LATENT = 17.0 / 5.0


def frame_differences(path: str, max_side: int = 160) -> np.ndarray:
    """Mean absolute luma difference between consecutive frames.

    Downscaled first: flicker is a whole-frame brightness and texture
    swing, and the small size keeps a long video cheap to read.
    """
    previous, diffs = None, []
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        for frame in container.decode(stream):
            image = frame.to_ndarray(format='gray')
            step = max(1, max(image.shape) // max_side)
            small = image[::step, ::step].astype(np.float32)
            if previous is not None:
                diffs.append(float(np.abs(small - previous).mean()))
            previous = small
    return np.asarray(diffs, dtype=np.float64)


def blocked_differences(path: str, blocks: int = 8) -> np.ndarray:
    """[frames - 1, blocks, blocks] mean absolute luma difference per
    spatial block.

    The whole-frame mean hides a local jump: a 128-token tile covers a
    small patch of a 1344x768 frame, so an artifact confined to it moves
    the frame average by almost nothing. Splitting the frame first keeps
    a local event visible.
    """
    previous, rows = None, []
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        for frame in container.decode(stream):
            image = frame.to_ndarray(format='gray').astype(np.float32)
            h, w = image.shape
            tile = image[:h // blocks * blocks, :w // blocks * blocks]
            tile = tile.reshape(blocks, h // blocks, blocks, w // blocks)
            tile = tile.mean(axis=(1, 3))
            if previous is not None:
                rows.append(np.abs(tile - previous))
            previous = tile
    return np.asarray(rows, dtype=np.float64)


def spectrum(diffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(periods in frames, amplitude) of the detrended difference signal.

    The mean and a linear trend carry no flicker information - every
    video has some motion - so they come out before the transform.
    """
    n = diffs.size
    if n < 8:
        return np.empty(0), np.empty(0)
    x = np.arange(n, dtype=np.float64)
    fit = np.polyfit(x, diffs, 1)
    residual = diffs - np.polyval(fit, x)
    residual = residual * np.hanning(n)
    amplitude = np.abs(np.fft.rfft(residual))[1:] * 2.0 / n
    periods = n / np.arange(1, amplitude.size + 1, dtype=np.float64)
    return periods, amplitude


def energy_near(periods, amplitude, target: float,
                tolerance: float = 0.25) -> float:
    """Largest amplitude within `tolerance` (relative) of `target`."""
    if periods.size == 0 or not target:
        return 0.0
    near = np.abs(periods - target) <= tolerance * target
    return float(amplitude[near].max()) if near.any() else 0.0


def tile_period(tile_depth: int) -> float:
    """Output frames spanned by one tile of `tile_depth` latent frames."""
    return tile_depth * FRAMES_PER_LATENT


def report(path: str, depth: int) -> dict:
    diffs = frame_differences(path)
    periods, amplitude = spectrum(diffs)
    target = tile_period(depth)
    peak = float(amplitude.max()) if amplitude.size else 0.0
    at_peak = float(periods[int(amplitude.argmax())]) if amplitude.size else 0
    return {
        'frames': diffs.size + 1,
        'mean_difference': float(diffs.mean()) if diffs.size else 0.0,
        'peak': peak,
        'peak_period': at_peak,
        'tile_period': target,
        'at_tile_period': energy_near(periods, amplitude, target),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('videos', nargs='+')
    parser.add_argument('--tile-depth', type=int, default=8,
                        help='latent frames per tile (8 for 8x4x4)')
    parser.add_argument('--latent-t', type=int, default=None,
                        help='only for the printed header')
    args = parser.parse_args()
    target = tile_period(args.tile_depth)
    print(f'tile of {args.tile_depth} latent frames = {target:.1f} output '
          f'frames; looking for periodic energy there')
    print(f"{'video':38} {'frames':>7} {'mean d':>8} {'peak':>8} "
          f"{'at':>7} {'@tile':>8}  flag")
    rows = []
    for path in args.videos:
        row = report(path, args.tile_depth)
        rows.append((path, row))
    baseline = min((r['at_tile_period'] for _, r in rows), default=0.0)
    for path, r in rows:
        ratio = r['at_tile_period'] / baseline if baseline else math.inf
        flag = 'FLICKER' if baseline and ratio > 1.5 else ''
        print(f"{os.path.basename(path)[:38]:38} {r['frames']:7} "
              f"{r['mean_difference']:8.3f} {r['peak']:8.4f} "
              f"{r['peak_period']:7.1f} {r['at_tile_period']:8.4f}  {flag}")


if __name__ == '__main__':
    main()
