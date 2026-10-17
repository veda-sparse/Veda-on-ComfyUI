"""Resolution-dependent generated attention budgets."""

import pytest

from veda_comfy import settings
from veda_comfy.core import selection


def test_resolution_budgets_switch_at_thresholds():
    rules = settings.parse_resolution_budgets('1:32, 2:64, 96')
    fallback = selection.Budget(tiles=16)
    # Grid dimensions are (temporal, height/32, width/32).
    assert settings.budget_for_grid(
        (72, 30, 30), fallback, rules).tiles == 32
    assert settings.budget_for_grid(
        (72, 52, 30), fallback, rules).tiles == 64
    assert settings.budget_for_grid(
        (72, 60, 60), fallback, rules).tiles == 96
    # 25x39 spatial cells are just under 1 MP; 25x40 are above.
    assert settings.budget_for_grid(
        (1, 25, 39), fallback, rules).tiles == 32
    assert settings.budget_for_grid(
        (1, 25, 40), fallback, rules).tiles == 64


def test_adaptive_disabled_preserves_fallback():
    fallback = selection.Budget.sparsity(90)
    assert settings.budget_for_grid((72, 52, 30), fallback, ()) is fallback
    assert not settings.VedaSettings(
        generated=fallback, reference=fallback).resolution_budgets


@pytest.mark.parametrize('bad', [
    '', '32', '1:32', '1:32,', '2:32, 1:64, 96',
    '0:32, 64', '-1:32, 64', 'nan:32, 64',
    'inf:32, 64', '1:0, 64', '1:32, 0',
    '1:90%, 64', '1:32, 90%', '1:32, 2:64, 3:96',
])
def test_invalid_resolution_budgets_explain_format(bad):
    with pytest.raises(ValueError, match='resolution_budgets'):
        settings.parse_resolution_budgets(bad)


def test_temporal_calibration_known_cases():
    rules = settings.parse_resolution_budgets('1:32, 2:64, 64')
    base = selection.Budget(tiles=32)
    policy = settings.TemporalPolicy()
    for grid, expected in [((87, 30, 17), 32),
                           ((72, 52, 30), 64),
                           ((87, 52, 30), 97),
                           ((1000, 52, 30), 168)]:
        actual = settings.effective_generated_budget(
            grid, base, rules, policy)
        assert actual.tiles == expected


def test_existing_modes_unchanged():
    rules = settings.parse_resolution_budgets('1:32, 2:64, 64')
    fallback = selection.Budget(tiles=16)
    assert settings.effective_generated_budget(
        (87, 52, 30), fallback, rules).tiles == 64
    assert settings.effective_generated_budget(
        (87, 52, 30), fallback, ()).tiles == 16
    assert not settings.VedaSettings(
        generated=fallback, reference=fallback).temporal_scaling


@pytest.mark.parametrize('strength,expected', [
    (0.0, 64), (0.5, 81), (1.0, 97), (1.5, 114), (2.0, 130),
])
def test_video_length_scaling_strength(strength, expected):
    rules = settings.parse_resolution_budgets('1:32, 2:64, 64')
    value = settings.effective_generated_budget(
        (87, 52, 30), selection.Budget(tiles=16), rules,
        settings.TemporalPolicy(strength=strength))
    assert value.tiles == expected


def test_strength_scales_relative_to_spatial_budget():
    rules = settings.parse_resolution_budgets('1:32, 2:80, 80')
    value = settings.effective_generated_budget(
        (87, 52, 30), selection.Budget(tiles=16), rules,
        settings.TemporalPolicy(strength=1.0))
    assert value.tiles == 122


def test_temporal_cap_never_reduces_spatial_budget():
    rules = settings.parse_resolution_budgets('1:32, 2:200, 200')
    actual = settings.effective_generated_budget(
        (87, 52, 30), selection.Budget(tiles=32), rules,
        settings.TemporalPolicy())
    assert actual.tiles == 200


def test_temporal_mp_boundary_is_exclusive():
    rules = settings.parse_resolution_budgets('1:32, 2:64, 64')
    policy = settings.TemporalPolicy(min_mp=1.0)
    assert settings.effective_generated_budget(
        (87, 25, 39), selection.Budget(tiles=16), rules,
        policy).tiles == 32


def test_policy_scales_higher_custom_spatial_budget():
    rules = settings.parse_resolution_budgets('1:32, 2:110, 110')
    actual = settings.effective_generated_budget(
        (87, 52, 30), selection.Budget(tiles=32), rules,
        settings.TemporalPolicy())
    # Relative scaling: ceil(110 * (97 / 64)) = 167.
    assert actual.tiles == 167


@pytest.mark.parametrize('kwargs', [
    {'min_mp': 0}, {'min_mp': float('nan')},
    {'baseline': 0}, {'baseline': 88, 'target': 87},
    {'target_tiles': 0}, {'max_tiles': 0},
    {'target_tiles': 200, 'max_tiles': 128},
    {'strength': -1}, {'strength': float('nan')},
])
def test_invalid_temporal_settings(kwargs):
    with pytest.raises(ValueError):
        settings.TemporalPolicy(**kwargs)
