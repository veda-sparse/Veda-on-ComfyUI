"""The ComfyUI node: schema, validation messages, patching."""

import pytest
import torch

from conftest import COMFYUI_ROOT
import veda_testing

pytestmark = pytest.mark.skipif(COMFYUI_ROOT is None,
                                reason='needs a ComfyUI checkout')


@pytest.fixture
def node(tmp_path, monkeypatch):
    import folder_paths
    from veda_comfy import nodes
    folder = tmp_path / 'veda'
    folder.mkdir()
    veda_testing.write_bundle(str(folder / 'tiny.safetensors'), 2, 2,
                              veda_testing.default_plans(2, 2))
    (folder / 'notes.txt').write_text('not a model')
    monkeypatch.setitem(folder_paths.folder_names_and_paths, nodes.FOLDER,
                        ([str(folder)], {'.safetensors'}))
    nodes._BUNDLES.clear()
    return nodes


def _patcher(layers=2, heads=2):
    import comfy.model_patcher
    import comfy.ops
    from comfy.ldm.minimax import model as h3

    class Wrapper(torch.nn.Module):
        def __init__(self, diffusion_model):
            super().__init__()
            self.diffusion_model = diffusion_model

    model = h3.MiniMaxH3Model(
        hidden_size=heads * 128, num_layers=layers,
        token_refiner_num_layers=1, num_attention_heads=heads,
        attention_head_dim=128, ffn_hidden_size=256,
        text_dim=heads * 128, timestep_input_dim=256,
        time_embed_hidden_size=heads * 128, time_embed_dim=128,
        dtype=torch.float32, device='cpu',
        operations=comfy.ops.disable_weight_init)
    return comfy.model_patcher.ModelPatcher(Wrapper(model), 'cpu', 'cpu')


def test_schema_hides_every_tuning_knob(node):
    schema = node.VedaSparseAttention.define_schema()
    visible = [i.id for i in schema.inputs if not i.advanced]
    assert visible == ['model', 'predictor']
    predictor = schema.inputs[1]
    assert 'tiny.safetensors' in predictor.options
    assert 'notes.txt' not in predictor.options
    assert predictor.default in predictor.options


def test_execute_installs_the_override(node):
    model = _patcher()
    out = node.VedaSparseAttention.execute(
        model, 'tiny.safetensors', full_attention_layers='1',
        full_attention_steps='0').result[0]
    options = out.model_options['transformer_options']
    assert callable(options['optimized_attention_override'])
    assert 'optimized_attention_override' not in model.model_options.get(
        'transformer_options', {})


@pytest.mark.parametrize('kwargs,match', [
    ({'full_attention_layers': '0, 9'}, 'blocks 0-1'),
    ({'full_attention_steps': 'last'}, '0-based'),
    ({'generated_sparsity': '100%'}, 'generated_sparsity'),
    ({'reference_sparsity': 'most'}, 'reference_sparsity'),
])
def test_execute_explains_bad_settings(node, kwargs, match):
    with pytest.raises(ValueError, match=match):
        node.VedaSparseAttention.execute(_patcher(), 'tiny.safetensors',
                                         **kwargs)


def test_execute_rejects_mismatched_models(node):
    with pytest.raises(ValueError, match='trained for'):
        node.VedaSparseAttention.execute(_patcher(heads=4), 'tiny.safetensors')
    import comfy.model_patcher

    class Other(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.diffusion_model = torch.nn.Linear(2, 2)

    other = comfy.model_patcher.ModelPatcher(Other(), 'cpu', 'cpu')
    with pytest.raises(ValueError, match='MiniMax-H3 only'):
        node.VedaSparseAttention.execute(other, 'tiny.safetensors')


def test_unknown_missing_predictor(node):
    with pytest.raises(ValueError, match='Pick another file'):
        node.VedaSparseAttention.execute(_patcher(), 'gone.safetensors')


def test_known_missing_predictor_says_how_to_get_it(node):
    """The node no longer downloads, so the error has to do the work."""
    from veda_comfy import predictors
    name = predictors.DEFAULT_PREDICTOR
    with pytest.raises(ValueError) as caught:
        node.VedaSparseAttention.execute(_patcher(), name)
    message = str(caught.value)
    assert predictors.KNOWN_PREDICTORS[name].url in message
    assert 'Browse Templates' in message


def test_example_workflows_match_the_schema(node):
    """The shipped templates carry one widget value per schema widget.

    widgets_values is positional, so a stale extra entry does not raise:
    it shifts every later widget. A leftover 'auto' from the removed
    `backend` input silently turned `verbose` on in both templates.
    """
    import glob
    import json
    import os

    from conftest import ROOT

    schema = node.VedaSparseAttention.define_schema()
    widgets = [i.id for i in schema.inputs if i.id != 'model']
    paths = glob.glob(os.path.join(ROOT, 'example_workflows', '*.json'))
    assert paths, 'no example workflows to check'
    found = 0
    for path in paths:
        with open(path, encoding='utf-8') as f:
            workflow = json.load(f)
        nodes_ = list(workflow['nodes'])
        for graph in workflow.get('definitions', {}).get('subgraphs', []):
            nodes_ += graph.get('nodes', [])  # templates may be collapsed
        for item in nodes_:
            if item.get('type') != 'VedaSparseAttention':
                continue
            found += 1
            values = item['widgets_values']
            assert len(values) == len(widgets), (
                f'{os.path.basename(path)}: {len(values)} values for '
                f'{len(widgets)} widgets {widgets}')
            assert values[widgets.index('verbose')] is False
            models = item['properties']['models']
            assert models[0]['directory'] == node.FOLDER
            assert models[0]['name'] == values[widgets.index('predictor')]
    assert found == len(paths), 'every template needs the Veda node'
