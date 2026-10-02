import pytest
import torch
from safetensors.torch import save_file

import veda_testing
from veda_comfy.core import bundle as veda_bundle
from veda_comfy.core import plans
from veda_comfy.core import tiling


@pytest.mark.parametrize('dtype', ['bfloat16', 'float8_e4m3fn', 'float32'])
def test_load_bundle_dequantizes(tmp_path, dtype):
    path = str(tmp_path / 'p.safetensors')
    weights = veda_testing.write_bundle(path, 3, 4,
                                        veda_testing.default_plans(3, 4),
                                        dtype=dtype)
    b = veda_bundle.load_bundle(path)
    assert (b.num_layers, b.num_heads, b.head_dim) == (3, 4, 128)
    assert b.keep_ratio == pytest.approx(0.1)
    tolerance = {'bfloat16': 1e-2, 'float32': 1e-2, 'float8_e4m3fn': 7e-2}
    for layer in range(3):
        got = b.proj_q[layer].float()
        want = weights[f'layers.{layer}.proj_q']
        assert got.dtype == torch.float32 and b.proj_q[layer].dtype == (
            torch.bfloat16)
        rel = (got - want).abs().max() / want.abs().max()
        assert rel < tolerance[dtype]
    assert len(b.plans.plans) == 2


def test_rejects_foreign_safetensors(tmp_path):
    path = str(tmp_path / 'lora.safetensors')
    save_file({'x': torch.zeros(2)}, path, metadata={'format': 'pt'})
    with pytest.raises(veda_bundle.BundleError, match='not a Veda predictor'):
        veda_bundle.load_bundle(path)


def test_rejects_shape_mismatch(tmp_path):
    path = str(tmp_path / 'p.safetensors')
    veda_testing.write_bundle(path, 2, 4, veda_testing.default_plans(3, 4))
    with pytest.raises(veda_bundle.BundleError, match='plan'):
        veda_bundle.load_bundle(path)


def _table():
    shapes = [tiling.TileShape(4, 4, 8), tiling.TileShape(2, 8, 8)]
    head_shape = [[0, 1]]
    return plans.PlanTable([
        plans.TilePlan('16x9_t37', (37, 24, 42), shapes, head_shape),
        plans.TilePlan('16x9_t72', (72, 24, 42), shapes, head_shape),
        plans.TilePlan('1x1_t37', (37, 24, 24), shapes, head_shape),
    ])


def test_plan_selection_rules():
    table = _table()
    exact = table.select((37, 24, 42))
    assert exact.exact and exact.plan.name == '16x9_t37'
    length = table.select((52, 24, 42))
    assert not length.exact and length.plan.name == '16x9_t37'
    assert 'nearest length' in length.how
    aspect = table.select((37, 23, 40))  # 1280x736, close to 16:9
    assert not aspect.exact and aspect.plan.name == '16x9_t37'
    portrait = table.select((37, 42, 24))  # only the transpose fits
    assert portrait.plan.name == '16x9_t37_T'
    assert portrait.plan.shapes[0] == tiling.TileShape(4, 8, 4)


def test_head_groups_partition_heads():
    plan = _table().plans['16x9_t37']
    groups = plan.head_groups(0, 'cpu')
    assert [g.shape for g in groups] == [tiling.TileShape(4, 4, 8),
                                         tiling.TileShape(2, 8, 8)]
    assert sorted(torch.cat([g.heads for g in groups]).tolist()) == [0, 1]


def test_summary():
    assert _table().summary() == '16:9, 1:1 x latent frames 37/72'
