"""Veda learned block-sparse attention for MiniMax-H3 in ComfyUI.

Layout:
  core/      device-agnostic torch: tiling, plans, predictor, selection,
             the per-call engine (no ComfyUI imports)
  backends/  one module per kernel family, isolated from each other
  kernels/   our FlashAttention-4 CuTe fork (fa4/ + the quack/ helpers
             it needs); edited directly, see docs/features/fa4_fork.md
  nodes.py, comfy_patch.py, status.py  the ComfyUI side
"""

__version__ = '0.1.0'
