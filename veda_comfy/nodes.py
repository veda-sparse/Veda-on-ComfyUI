"""The ComfyUI node: "Veda Sparse Attention (MiniMax H3)".

One node, MODEL in -> MODEL out, placed after the model loader and any LoRA
loaders. Visible inputs are just the model and the predictor file; every
tuning knob is an advanced input (hidden until "show advanced" is on) with
the trained defaults. Problems are reported as text on the node, never as a
silent fallback.
"""

from __future__ import annotations

import logging
import os
import threading

import comfy.model_management
import comfy.patcher_extension
import comfy.utils
import folder_paths
from comfy_api.latest import ComfyExtension, io

from . import backends
from . import comfy_patch
from . import downloads
from . import hardware
from . import settings as veda_settings
from . import status as veda_status
from .core import bundle as veda_bundle
from .core import selection

FOLDER = 'veda'
_UNTRAINED = ('sparse (nearest plan)', 'full attention')
_BUNDLES: dict[str, tuple[tuple[float, int], veda_bundle.PredictorBundle]] = {}
_BUNDLE_LOCK = threading.Lock()


def register_model_folder() -> str:
    """Registers models/veda (only .safetensors, so partial downloads and
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
    return list(local) + [n for n in downloads.KNOWN_PREDICTORS
                          if n not in local]


def _predictor_path(name: str, node_id: str | None) -> str:
    """Local path of a predictor, downloading a known release if missing."""
    path = folder_paths.get_full_path(FOLDER, name)
    if path is not None:
        return path
    known = downloads.KNOWN_PREDICTORS.get(name)
    if known is None:
        raise ValueError(f'Predictor {name!r} is not in models/{FOLDER}. '
                         'Pick another file or put it there.')
    status = veda_status.NodeStatus(node_id)
    status.show(f'⬇ downloading {name} ({known.size / 2**20:.0f} MB, once)')
    bar = comfy.utils.ProgressBar(known.size, node_id=node_id)
    folder = folder_paths.get_folder_paths(FOLDER)[0]
    path = downloads.fetch(known, folder,
                           lambda done, total: bar.update_absolute(done,
                                                                   total))
    return path


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
        raise ValueError(
            'Veda accelerates MiniMax-H3 only. Connect the MODEL of a '
            f'MiniMax-H3 checkpoint (got {type(diffusion).__name__}).')
    attn = diffusion.blocks[0].attn
    shape = (len(diffusion.blocks), attn.heads, attn.head_dim)
    if shape != (bundle.num_layers, bundle.num_heads, bundle.head_dim):
        raise ValueError(
            f'This model has {shape[0]} blocks x {shape[1]} heads x '
            f'{shape[2]}, but the predictor was trained for '
            f'{bundle.num_layers} x {bundle.num_heads} x {bundle.head_dim}.')
    return shape


def _other_sparse_node(model) -> bool:
    callbacks = getattr(model, 'callbacks', {}) or {}
    prepare = callbacks.get(
        comfy.patcher_extension.CallbacksMP.ON_PREPARE_STATE, {})
    return 'block_sparse_attention' in prepare


class VedaSparseAttention(io.ComfyNode):
    """Veda learned block-sparse attention for MiniMax-H3."""

    @classmethod
    def define_schema(cls):
        default = downloads.DEFAULT_PREDICTOR
        return io.Schema(
            node_id='VedaSparseAttention',
            display_name='Veda Sparse Attention (MiniMax H3)',
            category='model/patch',
            search_aliases=['veda', 'sparse attention', 'minimax h3 speed',
                            'accelerate', 'faster video'],
            description=(
                'Speeds up MiniMax-H3 (T2VA / FL2VA / R2VA) by computing '
                'only the attention tiles a learned predictor marks as '
                'important (90% sparse by default). Weights are untouched: '
                'LoRAs and fine-tuned H3 checkpoints work as usual. Put it '
                'after the model and LoRA loaders. Bypass it to compare '
                'with full attention.'),
            inputs=[
                io.Model.Input('model', tooltip='A MiniMax-H3 model (after '
                               'any LoRA loaders).'),
                io.Combo.Input(
                    'predictor', options=_predictor_options(),
                    default=default,
                    tooltip=f'Veda predictor in models/{FOLDER}. The '
                            'official release is downloaded automatically '
                            'on first use (set HF_ENDPOINT for a mirror).'),
                io.Float.Input(
                    'current_sparsity', default=90.0, min=0.0, max=99.5,
                    step=0.5, advanced=True,
                    tooltip='% of the generated video\'s key tiles each '
                            'query tile skips. 90 = keep 10% (trained '
                            'value). Lower is closer to full attention and '
                            'slower.'),
                io.Int.Input(
                    'current_tiles', default=0, min=0, max=8192,
                    advanced=True,
                    tooltip='> 0: keep exactly this many video key tiles '
                            '(128 tokens each) per query tile instead of '
                            'current_sparsity.'),
                io.Float.Input(
                    'history_sparsity', default=90.0, min=0.0, max=99.5,
                    step=0.5, advanced=True,
                    tooltip='% of the condition key tiles (first/last '
                            'frames, guide frames, reference images and '
                            'videos) each query tile skips. 0 = conditions '
                            'use full attention.'),
                io.Int.Input(
                    'history_tiles', default=0, min=0, max=8192,
                    advanced=True,
                    tooltip='> 0: keep exactly this many condition key '
                            'tiles per query tile instead of '
                            'history_sparsity.'),
                io.String.Input(
                    'full_attention_layers', default='', advanced=True,
                    tooltip='0-based DiT blocks that keep full attention, '
                            'e.g. "0, 1, 47-49". Empty = all sparse.'),
                io.String.Input(
                    'full_attention_steps', default='', advanced=True,
                    tooltip='0-based sampling steps that keep full '
                            'attention, e.g. "0" for the first step. Empty '
                            '= all sparse.'),
                io.Combo.Input(
                    'backend', options=list(backends.CHOICES),
                    default='auto', advanced=True,
                    tooltip='Attention kernel. auto picks the fastest that '
                            'passes a self-test on this GPU: FA4 -> '
                            'FlexAttention -> torch (NVIDIA), MLX -> torch '
                            '(Apple).'),
                io.Combo.Input(
                    'untrained_size', options=list(_UNTRAINED),
                    default=_UNTRAINED[0], advanced=True,
                    tooltip='For sizes the predictor was not trained on: '
                            'stay sparse with the nearest tile plan, or '
                            'fall back to full attention.'),
                io.Boolean.Input(
                    'verbose', default=False, advanced=True,
                    tooltip='Log how every attention call was handled.'),
            ],
            outputs=[io.Model.Output(tooltip='The model with Veda sparse '
                                     'attention.')],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, model, predictor, current_sparsity=90.0,
                current_tiles=0, history_sparsity=90.0, history_tiles=0,
                full_attention_layers='', full_attention_steps='',
                backend='auto', untrained_size=_UNTRAINED[0],
                verbose=False) -> io.NodeOutput:
        hidden = getattr(cls, 'hidden', None)
        node_id = getattr(hidden, 'unique_id', None)
        status = veda_status.NodeStatus(node_id)
        try:
            bundle = _bundle(_predictor_path(predictor, node_id))
        except (veda_bundle.BundleError, downloads.DownloadError) as error:
            raise ValueError(str(error)) from error
        num_layers, _, _ = _check_model(model, bundle)
        settings = veda_settings.VedaSettings(
            current=selection.Budget.from_user(current_sparsity,
                                               current_tiles),
            history=selection.Budget.from_user(history_sparsity,
                                               history_tiles),
            dense_layers=veda_settings.parse_index_list(
                full_attention_layers, 'full_attention_layers'),
            dense_steps=veda_settings.parse_index_list(
                full_attention_steps, 'full_attention_steps'),
            backend=backend,
            untrained_geometry=('dense' if untrained_size == _UNTRAINED[1]
                                else 'sparse'),
            verbose=verbose)
        missing = sorted(i for i in settings.dense_layers if i >= num_layers)
        if missing:
            raise ValueError(f'full_attention_layers: this model has blocks '
                             f'0-{num_layers - 1}; '
                             f'{veda_settings.format_index_list(missing)} '
                             'do not exist.')
        patched, _ = comfy_patch.apply(model, bundle, settings, node_id)
        device = comfy.model_management.get_torch_device()
        info = hardware.describe(device)
        lines = backends.probe(device, backend)
        ready = [line.split(':')[0] for line in lines
                 if line.endswith(': available')]
        kernel = ready[0] if ready else 'none (full attention)'
        text = (f'Ready · {info.label} · kernel {kernel} · '
                f'{settings.describe()}')
        fa4 = [line for line in lines if line.startswith('fa4')
               and not line.endswith(': available')]
        if fa4 and info.kind == 'cuda':
            text += ' · install the FA4 kernels for full speed (install_fa4)'
        if _other_sparse_node(model):
            status.warn('⚠ ComfyUI\'s "Model Sparse Attention" node is also '
                        'applied; on H3 it replaces the attention blocks, so '
                        'Veda would not run. Remove one of the two.')
        else:
            status.show(text)
        if verbose:
            logging.info('Veda: %s; backends: %s', bundle.describe(),
                         '; '.join(lines))
        return io.NodeOutput(patched)


class VedaExtension(ComfyExtension):
    async def get_node_list(self):
        return [VedaSparseAttention]


async def comfy_entrypoint() -> VedaExtension:
    register_model_folder()
    return VedaExtension()
