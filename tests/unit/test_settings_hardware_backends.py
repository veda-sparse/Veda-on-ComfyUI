import dataclasses
import importlib.util

import pytest
import torch

from veda_comfy import backends
from veda_comfy import hardware
from veda_comfy import settings
from veda_comfy.backends import base
from veda_comfy.backends import torch_gather
from veda_comfy.core import reference


@pytest.mark.parametrize('text,want', [
    ('', set()), ('0, 1, 47-49', {0, 1, 47, 48, 49}),
    ('3；5，7 9', {3, 5, 7, 9}), ('10 - 12', {10, 11, 12}),
    ('5-3', {3, 4, 5}), ('0~2', {0, 1, 2})])
def test_parse_index_list(text, want):
    assert settings.parse_index_list(text, 'x') == want


def test_parse_index_list_explains_errors():
    with pytest.raises(ValueError, match='0-based'):
        settings.parse_index_list('first', 'full_attention_steps')


def test_format_index_list():
    assert settings.format_index_list({0, 1, 47, 48, 49, 7}) == '0-1, 7, 47-49'


def test_step_index():
    sigmas = [1.0, 0.75, 0.5, 0.25, 0.0]
    assert settings.step_index(1.0, sigmas) == 0
    assert settings.step_index(0.75, sigmas) == 1
    assert settings.step_index(0.6, sigmas) == 1  # 2nd-order midpoint
    assert settings.step_index(0.25, sigmas) == 3
    assert settings.step_index(1.2, sigmas) == 0


def _info(cc, name='NVIDIA GeForce RTX 4090', kind='cuda'):
    base_info = hardware.DeviceInfo(
        kind, 0, name, cc, hardware._FAMILIES.get(cc, 'x'),
        hardware._cuda_subtype(cc, name) if cc else kind, 'linux', 'x86_64',
        None, False, '12.8' if cc else None)
    return base_info


@pytest.mark.parametrize('cc,name,family,subtype,fa4', [
    ((8, 6), 'NVIDIA GeForce RTX 3090', 'sm86', 'rtx30', 'fa4-sm80'),
    ((8, 9), 'NVIDIA GeForce RTX 4090', 'sm89', 'rtx40', 'fa4-sm80'),
    ((8, 0), 'NVIDIA A100-SXM4-80GB', 'sm80', 'a100', 'fa4-sm80'),
    ((9, 0), 'NVIDIA H100 80GB HBM3', 'sm90', 'hopper', 'fa4-sm90'),
    ((10, 0), 'NVIDIA B200', 'sm100', 'b200', 'fa4-sm100'),
    ((12, 0), 'NVIDIA GeForce RTX 5090', 'sm120', 'rtx50', 'fa4-sm120'),
    ((12, 0), 'NVIDIA RTX PRO 6000 Blackwell Workstation Edition', 'sm120',
     'rtx-pro-blackwell', 'fa4-sm120'),
    ((12, 1), 'NVIDIA GB10', 'sm121', 'dgx-spark', 'fa4-sm120'),
])
def test_sm_families_and_candidates(cc, name, family, subtype, fa4):
    info = _info(cc, name)
    assert (info.family, info.subtype) == (family, subtype)
    assert backends.candidates(info) == [fa4, 'flex', 'torch']
    assert backends.candidates(info, 'flex') == ['flex', fa4, 'torch']
    assert backends.candidates(info, 'fa4')[0] == fa4


def test_non_cuda_candidates():
    mps = dataclasses.replace(_info(None, 'Apple M3', 'mps'), family='mps')
    assert backends.candidates(mps) == ['mlx', 'torch']
    cpu = dataclasses.replace(_info(None, 'CPU', 'cpu'), family='cpu')
    assert backends.candidates(cpu) == ['torch']
    assert backends.candidates(cpu, 'fa4') == ['torch']


def test_torch_backend_passes_self_test_on_cpu():
    base.self_test(torch_gather.TorchGatherBackend(), torch.device('cpu'),
                   torch.float32)


def test_self_test_catches_a_kernel_that_ignores_the_mask():
    class Dense(base.Backend):
        name = 'dense'
        dtypes = (torch.float32,)

        def attend(self, q, k, v, block_mask, layout):
            mask = torch.ones_like(block_mask) & layout.kv_ok
            return reference.block_sparse_attention(q, k, v, mask, layout)

    with pytest.raises(base.BackendUnavailable, match='ignores the block'):
        base.self_test(Dense(), torch.device('cpu'), torch.float32)


def test_self_test_catches_wrong_results():
    class Wrong(torch_gather.TorchGatherBackend):
        name = 'wrong'

        def attend(self, *args):
            return super().attend(*args) * 1.5

    with pytest.raises(base.BackendUnavailable, match='wrong results'):
        base.self_test(Wrong(), torch.device('cpu'), torch.float32)


def test_resolve_reports_every_attempt():
    backends.reset()
    resolution = backends.resolve(torch.device('cpu'), 'fa4')
    assert resolution.backend.name == 'torch'
    assert resolution.attempts[-1] == ('torch', 'ok')


@pytest.mark.skipif(importlib.util.find_spec('mlx') is None
                    or not torch.backends.mps.is_available(),
                    reason='needs Apple silicon with mlx')
def test_mlx_and_torch_backends_on_mps():
    from veda_comfy.backends import mlx_gather
    info = hardware.describe('mps')
    for backend in (mlx_gather.create(info), torch_gather.create(info)):
        base.self_test(backend, torch.device('mps'))
