"""Veda learned block-sparse attention for MiniMax-H3 in ComfyUI.

Layout:
  core/      device-agnostic torch: tiling, plans, predictor, selection,
             the per-call engine (no ComfyUI imports)
  backends/  one module per kernel family, isolated from each other
  _vendor/   generated private copies of FlashAttention-4 (do not edit)
  nodes.py, comfy_patch.py, status.py  the ComfyUI side
"""

__version__ = '0.1.0'
