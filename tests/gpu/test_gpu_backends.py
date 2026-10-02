"""GPU checks: every backend that loads on this GPU must be correct, sparse,
and agree with the others on a realistic problem.

    pytest tests/gpu -q            # on a CUDA or Apple silicon machine

Skipped without an accelerator. FA4 tests need the kernels installed
(install_fa4); they are reported as skipped otherwise, never silently
passed.
"""

import importlib

import pytest
import torch

from veda_comfy import backends
from veda_comfy import hardware
from veda_comfy.backends import base
from veda_comfy.core import reference
from veda_comfy.core import selection
from veda_comfy.core import tiling


def _devices():
    if torch.cuda.is_available():
        return [torch.device('cuda', i)
                for i in range(torch.cuda.device_count())]
    if torch.backends.mps.is_available():
        return [torch.device('mps')]
    return []


pytestmark = pytest.mark.skipif(not _devices(), reason='needs a GPU')


def _load(name, device):
    info = hardware.describe(device)
    module = importlib.import_module(
        f'veda_comfy.backends.{backends._MODULES[name]}')
    try:
        return module.create(info)
    except base.BackendUnavailable as error:
        pytest.skip(f'{name} unavailable on {info.label}: {error}')


@pytest.mark.parametrize('device', _devices(), ids=str)
def test_auto_resolution_picks_a_sparse_kernel(device):
    backends.reset()
    resolution = backends.resolve(device)
    assert resolution.backend is not None, resolution.report()
    info = resolution.device
    if info.kind == 'cuda':
        print(f'{info.label}: {resolution.report()}')


def _problem(device, seq_len=6000, heads=4, seed=0):
    """Mid-size problem: target 7x16x24 (2688 rows), history 1x16x24,
    text / audio as global rows."""
    spans = [tiling.TiledSpan(300, (1, 16, 24), tiling.TileShape(1, 8, 16)),
             tiling.TiledSpan(1000, (7, 16, 24), tiling.TileShape(4, 4, 8))]
    layout = tiling.build_tile_layout(spans, seq_len, device)
    g = torch.Generator().manual_seed(seed)
    x = [torch.randn(seq_len, heads, 128, generator=g).to(device,
                                                          torch.bfloat16)
         for _ in range(3)]
    q, k, v = (tiling.gather_tiles(t, layout, torch.arange(heads,
                                                           device=device))
               for t in x)
    n = layout.n_video_tiles
    scores = torch.randn(heads, n, n, generator=g).to(device)
    blocks = selection.column_blocks(layout, selection.Budget(ratio=0.1),
                                     selection.Budget(ratio=0.2))
    mask = selection.block_mask(*selection.select(scores, layout, blocks),
                                layout)
    return q, k, v, mask, layout


@pytest.mark.parametrize('name', ['fa4', 'flex', 'torch', 'mlx'])
@pytest.mark.parametrize('device', _devices(), ids=str)
def test_backend_matches_reference(device, name):
    info = hardware.describe(device)
    if name == 'fa4':
        name = backends.candidates(info, 'fa4')[0]
        if not name.startswith('fa4'):
            pytest.skip(f'no FA4 backend for {info.label}')
    if name not in backends.candidates(info, name)[:1]:
        pytest.skip(f'{name} is not a candidate on {info.label}')
    backend = _load(name, device)
    base.self_test(backend, device)
    q, k, v, mask, layout = _problem(device)
    out = backend.attend(q, k, v, mask, layout)
    want = reference.block_sparse_attention(q, k, v, mask, layout)
    real = layout.slot_valid.bool()
    err = (out[real].float() - want[real].float()).abs().max().item()
    assert err < 2e-2, f'{backend.name}: max error {err}'
