# Resolution-adaptive generated attention budgets

## Goal

A single Veda MODEL patch can accelerate multiple H3 sampling stages at
different spatial resolutions. A fixed generated budget can be fast at low
resolution but exclude too much context at higher resolutions. Adaptive
budgets make the trade-off independently for each video size.

## Design and invariants

- Disabled by default: existing workflows keep their original generated
  and reference budgets.
- Enabled: `resolution_budgets` contains ascending MP upper limits followed
  by a final generated tile budget, e.g. `1:32, 2:64, 64`.
- Resolution comes from `spec.target.grid`: the spatial grid maps to output
  pixels at 32 pixels per spatial cell. Video duration does not affect the spatial MP threshold; optional
  temporal scaling adjusts the selected generated budget afterwards.
- Budget selection happens per attention call, not when the node is created;
  different passes through the same model can use different budgets.
- Only the generated key-tile budget changes. Reference attention budgets,
  predictors, tiling geometry and kernels remain unchanged.
- The running node text reports the budget for the current grid.
- All budgets represent counts of 128-token key tiles per query tile.
  Actual counts may be capped when fewer candidate tiles exist.

## Code paths

- `veda_comfy/nodes.py`: optional advanced controls
- `veda_comfy/settings.py`: parser and MP lookup
- `veda_comfy/comfy_patch.py`: propagates rules and reports active choice
- `veda_comfy/core/engine.py`: selects a generated budget per grid
- `tests/unit/test_adaptive_budgets.py`: threshold and validation tests

## Pitfalls

MP is an accessible user-facing measure, not a guarantee of video quality.
The needed attention budget also depends on duration, motion, references,
the predictor and tile plan. In a MiniMax-H3 two-stage workflow, one
reproducible temporal flash at 960x1664 was absent in a controlled run
when the generated tile budget increased from 32 to 64; this is not a
universal quality guarantee.

## Validation

Local unit tests and lint have passed on the development branch. Two
12.2-second, 87-position continuation chunks ran without observed flashes
at 97 generated tiles; the controlled first chunk flashed three times at
64 and once at 77. Generalisation to other workflows is untested.
Do not advertise any preset as artifact-free.

## Follow-up

Compare low-resolution and high-resolution speed, video quality and
GPU memory usage with identical prompts/seeds, including multi-stage runs.

## Calibrated temporal scaling (optional)

Enable `adaptive_budgets` and `experimental_temporal_scaling` to
increase generated attention for high-resolution clips longer than 72
temporal grid positions. The only temporal tuning widget is
`video_length_scaling` (default `1.0`, range 0 to 20).

- `1.0`: tested calibration, 64 tiles at 72 positions and 97 at 87.
- Higher: greater temporal attention and possible reduction in artifacts.
- Lower: less additional attention, potentially faster but riskier.
- `0.0`: no temporal increase.

The calculation scales relative to the **current spatial budget**, so
higher custom `resolution_budgets` remain meaningful. The default cap
of 168 tiles prevents unlimited growth on longer videos; a custom spatial
budget exceeding the cap is never reduced. Scaling applies only above
1 MP and does not affect reference attention.

In a 960x1664 MiniMax H3 continuation (87 positions), 64 generated tiles
showed three flashes, 77 showed one weak local flash and 97 showed none.
Another 87-position continuation using the calibrated node also showed
no visible flashes. These results do not establish universal quality
guarantees. Both adaptive modes are disabled by default.
