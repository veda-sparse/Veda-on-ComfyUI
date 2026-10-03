# Part of Veda's FlashAttention-4 CuTe fork (BSD-3-Clause, see
# LICENSE). Forked from flash-attn-4 4.0.0b32; see docs/features/
# fa4_fork.md. Edit directly.
"""Flash Attention CUTE (CUDA Template Engine) implementation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fa4")
except PackageNotFoundError:
    __version__ = "0.0.0"

from .interface import (
    flash_attn_func,
    flash_attn_varlen_func,
)

__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
]
