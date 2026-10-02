"""Hooks Veda into ComfyUI's MiniMax-H3 attention.

H3's `Attention.forward` (comfy/ldm/minimax/model.py) calls
`optimized_attention` with q / k / v already through the (LoRA-patched)
qkv projection, QK-norm and RoPE, and with `transformer_options` carrying
`block_index`, `minimax_h3_layout` and the sampling sigmas. Veda replaces
exactly that call through `transformer_options["optimized_attention_override"]`:

  * weights are untouched, so any LoRA, fine-tune or quantized H3 checkpoint
    works, and T2VA / FL2VA / R2VA conditioning nodes are unchanged;
  * no `patches_replace` slot is taken, so block patches such as H3's Fun
    ControlNet keep working;
  * calls Veda declines run the override that was installed before it (or
    ComfyUI's attention), so other attention nodes still apply there.

The override is re-installed on every step (ON_PREPARE_STATE), like
ComfyUI's own sparse attention node, so a node applied later cannot
silently replace it.
"""

from __future__ import annotations

import collections
import contextlib
import logging
import weakref

import torch

import comfy.patcher_extension

try:  # ComfyUI with the comfy-aimdo allocation graph (0.38+)
    from comfy.model_prefetch import pause_malloc_graph as _pause_malloc_graph
except ImportError:  # older ComfyUI: nothing to pause
    _pause_malloc_graph = contextlib.nullcontext

from . import backends
from . import settings as veda_settings
from . import status as veda_status
from .core import engine as veda_engine
from .core import h3_layout

_KEY = 'veda_sparse_attention'


def _is_interrupt(error: BaseException) -> bool:
    # comfy.model_management.InterruptProcessingException, without importing
    # it at module scope.
    return type(error).__name__ == 'InterruptProcessingException'


class VedaPatch:
    """Options and runtime state of one patched model."""

    def __init__(self, bundle, settings: veda_settings.VedaSettings,
                 node_id: str | None):
        self.bundle = bundle
        self.settings = settings
        self.status = veda_status.NodeStatus(node_id)
        self.installed: set = set()
        self._engines: dict[str, veda_engine.VedaEngine | None] = {}
        self._resolutions: dict[str, backends.Resolution] = {}
        self._steps = weakref.WeakKeyDictionary()
        self._announced: set = set()
        self._failed = False
        self.calls = collections.Counter()

    # -- per device ------------------------------------------------------

    def _engine(self, device: torch.device) -> veda_engine.VedaEngine | None:
        key = str(device)
        if key not in self._engines:
            resolution = backends.resolve(
                device, self.settings.backend,
                notify=lambda text: self.status.show(f'⏳ {text}'))
            self._resolutions[key] = resolution
            if resolution.backend is None:
                self.status.warn(
                    f'⚠ no sparse kernel works on {resolution.device.label}, '
                    f'running full attention ({resolution.report()})')
                self._engines[key] = None
            else:
                self._engines[key] = veda_engine.VedaEngine(
                    self.bundle, self.settings.current, self.settings.history,
                    resolution.backend, device)
        return self._engines[key]

    def _step(self, transformer_options) -> int | None:
        sigmas = transformer_options.get('sigmas')
        schedule = transformer_options.get('sample_sigmas')
        if sigmas is None or schedule is None:
            return None
        try:
            return self._steps[sigmas]
        except (KeyError, TypeError):
            pass
        step = veda_settings.step_index(float(sigmas.flatten()[0]),
                                        schedule.flatten().tolist())
        try:
            self._steps[sigmas] = step
        except TypeError:
            pass
        return step

    # -- the override ------------------------------------------------------

    def make_override(self, previous):
        """The attention override; declined calls go to `previous`."""
        patch = self

        def override(func, q, k, v, heads, mask=None, attn_precision=None,
                     skip_reshape=False, skip_output_reshape=False,
                     **kwargs):
            def dense(reason: str):
                patch.calls[reason] += 1
                kw = dict(mask=mask, attn_precision=attn_precision,
                          skip_reshape=skip_reshape,
                          skip_output_reshape=skip_output_reshape, **kwargs)
                if previous is None:
                    return func(q, k, v, heads, **kw)
                return previous(func, q, k, v, heads, **kw)

            options = kwargs.get('transformer_options') or {}
            layout = options.get('minimax_h3_layout')
            if (mask is not None or not skip_reshape or q.dim() != 4
                    or q.shape[0] != 1 or q.shape != k.shape
                    or q.shape != v.shape or layout is None
                    or getattr(layout, 'seq_len', None) != q.shape[2]):
                return dense('other attention')
            reason = patch._dense_reason(q, options)
            if reason is not None:
                return dense(reason)
            try:
                return patch._sparse(q, k, v, layout, options,
                                     skip_output_reshape, dense)
            except Exception as error:  # never lose the user's render
                if _is_interrupt(error):
                    raise
                patch._failed = True
                logging.error('Veda: sparse attention failed', exc_info=True)
                patch.status.warn(f'⚠ sparse attention failed '
                                  f'({type(error).__name__}: {error}); '
                                  'finishing this run with full attention')
                return dense('error')

        return override

    def _dense_reason(self, q, options) -> str | None:
        """Why this H3 call runs full attention, or None to go sparse."""
        s, bundle = self.settings, self.bundle
        if self._failed:
            return 'error'
        layer = options.get('block_index')
        if not isinstance(layer, int) or layer >= bundle.num_layers:
            return 'layer outside the predictor'
        if q.shape[1] != bundle.num_heads or q.shape[3] != bundle.head_dim:
            self._announce(('shape', tuple(q.shape)),
                           f'⚠ model has {q.shape[1]} heads of dim '
                           f'{q.shape[3]}, the predictor expects '
                           f'{bundle.num_heads} x {bundle.head_dim}: running '
                           'full attention', warn=True)
            return 'head mismatch'
        if layer in s.dense_layers:
            return 'full-attention layer'
        if s.dense_steps and self._step(options) in s.dense_steps:
            return 'full-attention step'
        if s.current.keeps_all and s.history.keeps_all:
            return 'sparsity 0'
        return None

    def _sparse(self, q, k, v, layout, options, skip_output_reshape, dense):
        # ComfyUI records each block's allocations into a malloc graph
        # (comfy-aimdo) and expects everything allocated inside a block to
        # be gone by its end. Veda keeps device state across calls (tile
        # layouts, plan head groups, statistics) and its kernels allocate
        # their own workspaces, so the whole sparse path runs with the
        # graph paused, like ComfyUI's own sparse attention node does for
        # its persistent state. Otherwise the process aborts natively.
        with _pause_malloc_graph():
            return self._sparse_unpaused(q, k, v, layout, options,
                                         skip_output_reshape, dense)

    def _sparse_unpaused(self, q, k, v, layout, options,
                         skip_output_reshape, dense):
        engine = self._engine(q.device)
        if engine is None:
            return dense('no sparse kernel')
        try:
            spec = engine.layout_spec(layout)
        except h3_layout.LayoutError as error:
            self._announce(('layout', str(error)),
                           f'⚠ cannot read this H3 layout ({error}): '
                           'running full attention', warn=True)
            return dense('layout')
        choice = engine.plan_for(spec)
        if not choice.exact and self.settings.untrained_geometry == 'dense':
            self._announce(('plan', spec.target.grid),
                           f'⚠ untrained size ({choice.how}): running full '
                           'attention as configured', warn=True)
            return dense('untrained size')
        self._announce(('plan', spec.target.grid), self._active_text(
            engine, spec, choice), warn=not choice.exact)
        batch, heads, seq_len, dim = q.shape
        out = engine.attention(q[0].transpose(0, 1), k[0].transpose(0, 1),
                               v[0].transpose(0, 1), options['block_index'],
                               spec, choice.plan)
        self.calls['sparse'] += 1
        if skip_output_reshape:
            return out.transpose(0, 1).unsqueeze(0)
        return out.reshape(batch, seq_len, heads * dim)

    def _active_text(self, engine, spec, choice) -> str:
        grid = spec.target.grid
        size = f'{grid[2] * 32}x{grid[1] * 32}, {grid[0]} latent frames'
        history = ''
        if spec.history:
            history = (' · history tiled' if not self.settings.history.keeps_all
                       else ' · history full attention')
        mark = '⚡' if choice.exact else '⚠'
        return (f'{mark} Veda on {engine.backend.name} · {size} · '
                f'{choice.how}{history} · {self.settings.describe()}')

    def _announce(self, key, text: str, warn: bool = False) -> None:
        if key in self._announced:
            return
        self._announced.add(key)
        (self.status.warn if warn else self.status.show)(text)

    # -- lifecycle ---------------------------------------------------------

    def install(self, transformer_options: dict) -> None:
        """Puts the override on top of whatever override is on the hook;
        idempotent once it is on top."""
        current = transformer_options.get('optimized_attention_override')
        if current in self.installed:
            return
        override = self.make_override(current)
        self.installed.add(override)
        transformer_options['optimized_attention_override'] = override

    def on_cleanup(self) -> None:
        """End of a sampling run: report, then reset per-run state."""
        sparse = self.calls.get('sparse', 0)
        if sparse or self.calls:
            engines = [e for e in self._engines.values() if e is not None]
            kept = [e.stats.kept_fraction() for e in engines]
            kept = [x for x in kept if x is not None]
            dense_total = sum(n for r, n in self.calls.items()
                              if r not in ('sparse', 'other attention'))
            parts = [f'{sparse} sparse / {dense_total} full-attention calls']
            if kept:
                parts.append(f'kept {100.0 * sum(kept) / len(kept):.1f}% of '
                             'video key tiles')
            if engines:
                parts.append(engines[0].backend.name)
            if self._failed:
                parts.append('fell back to full attention after an error')
            self.status.show(('✅ ' if sparse and not self._failed else '⚠ ')
                             + 'Veda last run: ' + ' · '.join(parts))
            if self.settings.verbose:
                logging.info('Veda: call breakdown %s', dict(self.calls))
            for engine in engines:
                engine.stats = veda_engine.Stats()
        self.calls.clear()
        self._announced.clear()
        self._failed = False


def apply(model, bundle, settings: veda_settings.VedaSettings,
          node_id: str | None):
    """Returns a clone of `model` with Veda attention installed."""
    patch = VedaPatch(bundle, settings, node_id)
    patched = model.clone()
    patch.install(patched.model_options.setdefault('transformer_options', {}))
    patched.add_callback_with_key(
        comfy.patcher_extension.CallbacksMP.ON_PREPARE_STATE, _KEY,
        lambda model_patcher, timestep, model_options: patch.install(
            model_options['transformer_options']))
    patched.add_callback_with_key(
        comfy.patcher_extension.CallbacksMP.ON_CLEANUP, _KEY,
        lambda model_patcher: patch.on_cleanup())
    return patched, patch
