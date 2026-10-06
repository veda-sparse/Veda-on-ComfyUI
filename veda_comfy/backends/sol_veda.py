"""Sol + Veda backend using Sage quantization for selected blocks."""

from __future__ import annotations

import functools
import sys

import torch

from . import base


@functools.cache
def _kernel():
    from ..kernels.sol_veda import attention
    return attention


class SolVedaBackend(base.Backend):
    name = 'sol-veda'
    display = 'Sol + Veda'
    dtypes = (torch.bfloat16,)
    tolerance = 0.05

    def attend(self, q, k, v, block_mask, layout):
        return _kernel().attend(q, k, v, layout.valid_count,
                                block_mask & layout.kv_ok[None, None, :],
                                _sage()) [0]

    def warmup_note(self):
        return 'compiling Sol + Veda Triton kernel for this GPU (first run only)'


@functools.cache
def _sage():
    from ..kernels.sage import sparse_int8
    return sparse_int8


def create(info):
    if info.kind != 'cuda' or info.cc is None or info.cc < (8, 0):
        raise base.BackendUnavailable(
            'Sol + Veda needs a CUDA GPU of SM80 or newer')
    try:
        _kernel()
        _sage()
    except ImportError as error:
        package = 'triton-windows' if sys.platform == 'win32' else 'triton'
        raise base.BackendUnavailable(
            f'Triton is not installed ({error}); pip install {package}') from error
    return SolVedaBackend()
