"""The ComfyUI node: "Veda Sparse Attention (MiniMax H3)".

One node, MODEL in -> MODEL out: an attention override, placed on the MODEL
wire after the model loader and any LoRA loaders and last before the sampler
or guider. Visible inputs are just the model and the predictor file; every
tuning knob is an advanced input (hidden until "show advanced" is on) with
the trained defaults. Problems are reported as text on the node, never as a
silent fallback.
"""

from __future__ import annotations

import os
import threading

import comfy.model_management
import comfy.patcher_extension
import folder_paths
from comfy_api.latest import ComfyExtension, io

from . import backends
from . import comfy_patch
from . import hardware
from . import predictors
from . import settings as veda_settings
from . import status as veda_status
from .core import bundle as veda_bundle

FOLDER = 'veda'
_BUNDLES: dict[str, tuple[tuple[float, int], veda_bundle.PredictorBundle]] = {}
_BUNDLE_LOCK = threading.Lock()


def register_model_folder() -> str:
    """Registers models/veda (only .safetensors, so stray files and
    stray files never show up in the list)."""
    path = os.path.join(folder_paths.models_dir, FOLDER)
    entry = folder_paths.folder_names_and_paths.get(FOLDER)
    if entry is None:
        folder_paths.folder_names_and_paths[FOLDER] = ([path], {'.safetensors'})
    else:
        if path not in entry[0]:
            entry[0].append(path)
        if entry[1]:
            entry[1].add('.safetensors')
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:  # read-only installs: the user can still add paths
        pass
    return path


def _predictor_options() -> list[str]:
    local = folder_paths.get_filename_list(FOLDER)
    return list(local) + [n for n in predictors.KNOWN_PREDICTORS
                          if n not in local]


def _predictor_path(name: str) -> str:
    """Local path of a predictor.

    Raises:
        ValueError: If the file is not there, with how to get it. The node
            does not fetch it: ComfyUI's missing-model dialog does that,
            driven by `properties.models` in the example workflows.
    """
    path = folder_paths.get_full_path(FOLDER, name)
    if path is not None:
        return path
    folder = folder_paths.get_folder_paths(FOLDER)[0]
    known = predictors.KNOWN_PREDICTORS.get(name)
    if known is None:
        raise ValueError(veda_status.bilingual(
            f'Predictor {name!r} is not in {folder}. Pick another file or '
            'put it there.',
            f'打分器 {name!r} 不在 {folder} 里。请换一个文件，或把它放进去。'))
    raise ValueError(predictors.how_to_get(known, folder))


def _bundle(path: str) -> veda_bundle.PredictorBundle:
    stat = os.stat(path)
    stamp = (stat.st_mtime, stat.st_size)
    with _BUNDLE_LOCK:
        cached = _BUNDLES.get(path)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        bundle = veda_bundle.load_bundle(path)
        _BUNDLES.clear()  # keep one bundle (~0.5 GB of host memory)
        _BUNDLES[path] = (stamp, bundle)
        return bundle


def _check_model(model, bundle) -> tuple[int, int, int]:
    diffusion = model.get_model_object('diffusion_model')
    if type(diffusion).__name__ != 'MiniMaxH3Model':
        raise ValueError(veda_status.bilingual(
            'Veda accelerates MiniMax-H3 only. Connect the MODEL of a '
            f'MiniMax-H3 checkpoint (got {type(diffusion).__name__}).',
            'Veda 只加速 MiniMax-H3。请连接 MiniMax-H3 模型的 MODEL'
            f'（现在是 {type(diffusion).__name__}）。'))
    attn = diffusion.blocks[0].attn
    shape = (len(diffusion.blocks), attn.heads, attn.head_dim)
    trained = (bundle.num_layers, bundle.num_heads, bundle.head_dim)
    if shape != trained:
        model_shape = ' x '.join(map(str, shape))
        trained_shape = ' x '.join(map(str, trained))
        raise ValueError(veda_status.bilingual(
            f'This model has {model_shape} (blocks x heads x head dim), but '
            f'the predictor was trained for {trained_shape}. Pick the '
            'predictor made for this model.',
            f'这个模型是 {model_shape}（block x 头数 x 头维度），打分器训练于 '
            f'{trained_shape}。请选择为这个模型训练的打分器。'))
    return shape


def _other_sparse_node(model) -> bool:
    callbacks = getattr(model, 'callbacks', {}) or {}
    prepare = callbacks.get(
        comfy.patcher_extension.CallbacksMP.ON_PREPARE_STATE, {})
    return 'block_sparse_attention' in prepare


def _takes_forward(model) -> bool:
    options = model.model_options.get('transformer_options', {})
    return options.get(comfy_patch.TAKE_FORWARD) is not None


class VedaSparseAttention(io.ComfyNode):
    """Veda learned block-sparse attention for MiniMax-H3."""

    @classmethod
    def define_schema(cls):
        default = predictors.DEFAULT_PREDICTOR
        return io.Schema(
            node_id='VedaSparseAttention',
            display_name='Veda Sparse Attention (MiniMax H3)',
            category='model/patch/minimax',
            search_aliases=['veda', 'sparse attention', 'minimax h3 speed',
                            'accelerate', 'faster video'],
            description=(
                'Speeds up MiniMax-H3 (T2VA / FL2VA / R2VA) by computing '
                'only the attention tiles a learned predictor marks as '
                'important. This is an attention override, so put it on '
                'the MODEL wire after the model and any LoRA loaders, last '
                'before the sampler or guider.'),
            inputs=[
                io.Model.Input('model', tooltip='The MiniMax-H3 model to '
                               'patch (after any LoRA loaders).'),
                io.Combo.Input(
                    'predictor', options=_predictor_options(),
                    default=default,
                    tooltip=f'Veda predictor in models/{FOLDER}. Open a '
                            'Veda template and ComfyUI\'s missing-model '
                            'dialog fetches it, or put the file there by '
                            'hand.'),
                io.String.Input(
                    'generated_sparsity',
                    default=veda_settings.DEFAULT_BUDGET, advanced=True,
                    tooltip='Manual generated attention budget (used when '
                            'adaptive_budgets is off). Each tile contains '
                            '128 tokens. '
                            'A whole number such as "32" (the default) '
                            'keeps that many whatever the video size, which '
                            'is what two-stage workflows need: a percentage '
                            'scales with the grid area and collapses on '
                            'their small first pass. "90%" still works and '
                            'skips 90% of the tiles; "trained" takes '
                            'whatever the predictor declares.'),
                io.String.Input(
                    'reference_sparsity',
                    default=veda_settings.DEFAULT_BUDGET, advanced=True,
                    tooltip='The same for the references: first / last '
                            'frames, guide frames, reference images and '
                            'videos. "0%" keeps full attention to and from '
                            'them.'),
                io.String.Input(
                    'full_attention_layers', default='', advanced=True,
                    tooltip='0-based DiT blocks that keep full attention, '
                            'e.g. "0, 1, 47-49". Empty = all sparse.'),
                io.String.Input(
                    'full_attention_steps', default='', advanced=True,
                    tooltip='0-based sampling steps that keep full '
                            'attention, e.g. "0" for the first step. Empty '
                            '= all sparse.'),
                io.Boolean.Input(
                    'verbose', default=False, advanced=True,
                    tooltip='Also show timing and diagnostics on the node '
                            'after each run, and log every decision.'),
                io.Boolean.Input(
                    'adaptive_budgets', default=False, advanced=True,
                    tooltip='Choose generated attention tile budgets '
                            'automatically for each resolution stage. '
                            'Disabled preserves the original setting.'),
                io.String.Input(
                    'resolution_budgets', default='1:32, 2:64, 64',
                    advanced=True,
                    tooltip='MP upper limit:generated tiles, in ascending '
                            'order, then final tile count for larger '
                            'resolutions. Example: 1:32, 2:64, 64. '
                            'Reference attention stays unchanged.'),
                io.Boolean.Input(
                    'experimental_temporal_scaling', default=False,
                    advanced=True,
                    tooltip=(
                        'Requires adaptive_budgets. Adds attention '
                        'for longer videos. Use video_length_scaling to '
                        'adjust quality versus speed. Default 1.0 '
                        'keeps the tested 64-to-97 calibration.'
                    )),
                io.Float.Input(
                    'video_length_scaling', default=1.0,
                    advanced=True, min=0.0, max=20.0, step=0.1,
                    tooltip='1.0 uses the tested temporal calibration. '
                            'Increase if long videos show flashing; '
                            'decrease for speed. 0 disables added '
                            'temporal attention. References are unaffected.'),
            ],
            outputs=[io.Model.Output(
                display_name='model',
                tooltip='The model with Veda sparse attention applied.')],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, model, predictor,
                generated_sparsity=veda_settings.DEFAULT_BUDGET,
                reference_sparsity=veda_settings.DEFAULT_BUDGET,
                full_attention_layers='',
                full_attention_steps='', verbose=False,
                adaptive_budgets=False,
                resolution_budgets='1:32, 2:64, 64',
                experimental_temporal_scaling=False,
                video_length_scaling=1.0,
                temporal_min_megapixels=None,
                temporal_baseline_positions=None,
                temporal_target_positions=None,
                temporal_target_generated_budget=None,
                temporal_max_generated_budget=None) -> io.NodeOutput:
        hidden = getattr(cls, 'hidden', None)
        node_id = getattr(hidden, 'unique_id', None)
        status = veda_status.NodeStatus(node_id)
        try:
            bundle = _bundle(_predictor_path(predictor))
        except veda_bundle.BundleError as error:
            raise ValueError(veda_status.bilingual(
                str(error),
                f'{predictor} 不是可用的 Veda 打分器（原因见上）。'
                f'models/{FOLDER} 里只放 Veda 打分器文件。')) from error
        num_layers, _, _ = _check_model(model, bundle)
        settings = veda_settings.VedaSettings(
            generated=veda_settings.parse_sparsity(
                generated_sparsity, 'generated_sparsity', bundle.generated),
            reference=veda_settings.parse_sparsity(
                reference_sparsity, 'reference_sparsity', bundle.reference),
            dense_layers=veda_settings.parse_index_list(
                full_attention_layers, 'full_attention_layers'),
            dense_steps=veda_settings.parse_index_list(
                full_attention_steps, 'full_attention_steps'),
            verbose=verbose,
            temporal_scaling=(experimental_temporal_scaling
                              and adaptive_budgets),
            temporal_policy=(veda_settings.TemporalPolicy(
                strength=video_length_scaling)
                if experimental_temporal_scaling and adaptive_budgets
                else veda_settings.TemporalPolicy()),
            resolution_budgets=(
                veda_settings.parse_resolution_budgets(resolution_budgets)
                if adaptive_budgets else ()))
        missing = sorted(i for i in settings.dense_layers if i >= num_layers)
        if missing:
            missing = veda_settings.format_index_list(missing)
            raise ValueError(veda_status.bilingual(
                f'full_attention_layers: this model has blocks '
                f'0-{num_layers - 1}; {missing} do not exist.',
                f'full_attention_layers：这个模型的 block 是 '
                f'0-{num_layers - 1}，{missing} 不存在。'))
        if _other_sparse_node(model):
            raise ValueError(veda_status.bilingual(
                'ComfyUI\'s "Model Sparse Attention" node is also applied; '
                'on H3 it replaces the attention blocks, so Veda cannot '
                'run. Remove one of the two.',
                '同时接了 ComfyUI 的 "Model Sparse Attention" 节点：它在 H3 '
                '上直接替换注意力 block，Veda 无法运行。请二选一。'))
        # Resolved here rather than at the first sampling step, so that a
        # kernel which cannot run stops the prompt on this node, with the
        # reason, before the sampler starts. The result is cached.
        device = comfy.model_management.get_torch_device()
        info = hardware.describe(device)
        resolution = backends.resolve(device, notify=status.show)
        if resolution.backend is None:
            raise comfy_patch.kernel_error(resolution)
        patched, patch = comfy_patch.apply(model, bundle, settings, node_id)
        lines = [f'Veda ready · {resolution.backend.display}'
                 f' · {info.short_name}',
                 ('Adaptive generated budget: based on video resolution'
                  + (' and duration' if settings.temporal_scaling else '')
                  + '; '
                  f'reference {veda_settings.format_budget(settings.reference)}'
                  if settings.resolution_budgets else
                  f'Sparsity: {settings.describe()}')]
        full = settings.describe_full_attention()
        if full:
            lines.append(f'Full attention: {full}')
        notes = []
        if patch.taken_over:
            names = ', '.join(patch.taken_over)
            notes.append((f'Took over the attention from {names}: Veda runs '
                          'the sparse layers, it keeps the full-attention '
                          'ones.',
                          f'已接管 {names} 的注意力：稀疏层由 Veda 计算，'
                          '全注意力层仍由它计算。'))
        lines += [veda_status.bilingual(en, zh) for en, zh in notes]
        if verbose:
            lines.append(f'Predictor: {bundle.describe()}')
            lines.append(f'  {resolution.report()}')
            if patch.taken_over:
                lines.append(
                    '  sparse layers use: '
                    + ('the Low VRAM attention forward (KJNodes)'
                       if _takes_forward(patched)
                       else 'ComfyUI\'s MiniMax-H3 attention forward'))
        if patch.taken_over:
            status.warn('\n'.join(lines))
        else:
            status.show('\n'.join(lines))
        return io.NodeOutput(patched)


class VedaExtension(ComfyExtension):
    async def get_node_list(self):
        return [VedaSparseAttention]


async def comfy_entrypoint() -> VedaExtension:
    register_model_folder()
    return VedaExtension()
