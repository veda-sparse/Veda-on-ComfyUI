"""Backend registry: which kernels may run on which device, in what order.

Candidates per device (first that loads and passes its self-test wins):

    CUDA sm80/86/87/89  fa4-sm80 (patched FA4) -> flex -> torch
    CUDA sm90           fa4-sm90 (upstream FA4) -> flex -> torch
    CUDA sm100/103/110  fa4-sm100 (upstream FA4) -> flex -> torch
    CUDA sm120/121      fa4-sm120 (patched FA4) -> flex -> torch
    ROCm                flex -> torch
    Apple MPS           mlx -> torch
    CPU                 torch

Backend modules are imported lazily and only here, so a broken or missing
kernel package costs one entry in the report, never the node.
"""

from __future__ import annotations

import dataclasses
import importlib
import logging
import threading
from collections.abc import Callable

import torch

from . import base
from .. import hardware

# User-facing choices (the node's `backend` input). 'fa4-fp8' runs the
# same kernels with e4m3 operands on the architectures whose FA4 build has
# FP8 tensor cores (SM8x / SM120).
CHOICES = ('auto', 'fa4', 'fa4-fp8', 'flex', 'torch', 'mlx')

_MODULES = {
    'fa4-sm80': 'fa4_sm80', 'fa4-sm90': 'fa4_sm90',
    'fa4-sm100': 'fa4_sm100', 'fa4-sm120': 'fa4_sm120',
    'flex': 'flex', 'torch': 'torch_gather', 'mlx': 'mlx_gather',
}
_FA4_BY_MAJOR = {8: 'fa4-sm80', 9: 'fa4-sm90', 10: 'fa4-sm100',
                 11: 'fa4-sm100', 12: 'fa4-sm120'}
# Only the SM80-family kernel carries our FP8 patch.
_FP8_SUFFIX = '-fp8'
_FP8_CAPABLE = ('fa4-sm80', 'fa4-sm120')


def _load(name: str, info: hardware.DeviceInfo) -> base.Backend:
    """Instantiates one candidate by name (an '-fp8' suffix selects the
    e4m3 variant of that kernel)."""
    fp8 = name.endswith(_FP8_SUFFIX)
    base_name = name[:-len(_FP8_SUFFIX)] if fp8 else name
    module = importlib.import_module(f'.{_MODULES[base_name]}', __name__)
    return module.create(info, fp8=fp8) if fp8 else module.create(info)


@dataclasses.dataclass
class Resolution:
    """Outcome of picking a backend for one device.

    Attributes:
        device: The device info.
        backend: The backend to use, None if nothing works (run dense).
        attempts: (candidate, 'ok' or why it was skipped), in order.
    """

    device: hardware.DeviceInfo
    backend: base.Backend | None
    attempts: list[tuple[str, str]]

    def report(self) -> str:
        return '; '.join(f'{name}: {status}' for name, status in self.attempts)


def candidates(info: hardware.DeviceInfo, requested: str = 'auto'
               ) -> list[str]:
    """Backend names to try on a device, best first."""
    if info.kind == 'cuda' and info.family != 'rocm':
        fa4 = _FA4_BY_MAJOR.get(info.cc[0]) if info.cc else None
        auto = [fa4, 'flex', 'torch']
    elif info.kind == 'cuda':
        fa4, auto = None, ['flex', 'torch']
    elif info.kind == 'mps':
        fa4, auto = None, ['mlx', 'torch']
    else:
        fa4, auto = None, ['torch']
    auto = [name for name in auto if name]
    if requested in ('auto', '', None):
        return auto
    if requested in ('fa4', 'fa4-fp8'):
        if fa4 is None:
            return auto
        first = fa4
        if requested == 'fa4-fp8':
            if fa4 not in _FP8_CAPABLE:
                return auto
            first = fa4 + _FP8_SUFFIX
    else:
        first = requested
    return [first] + [name for name in auto if name != first]


_LOCK = threading.RLock()
_RESOLVED: dict[tuple[str, str], Resolution] = {}


def resolve(device: torch.device, requested: str = 'auto',
            notify: Callable[[str], None] | None = None) -> Resolution:
    """Loads and self-tests backends on `device` until one works (cached).

    Args:
        device: The device attention runs on.
        requested: One of CHOICES.
        notify: Called with a short status line before slow steps (kernel
            compilation on the first call).
    """
    info = hardware.describe(device)
    key = (str(device), requested)
    with _LOCK:
        if key in _RESOLVED:
            return _RESOLVED[key]
        attempts: list[tuple[str, str]] = []
        chosen = None
        for name in candidates(info, requested):
            try:
                backend = _load(name, info)
                note = backend.warmup_note()
                if notify is not None and note:
                    notify(f'{backend.display}: {note}')
                base.self_test(backend, device)
            except base.BackendUnavailable as error:
                attempts.append((name, str(error)))
                continue
            except Exception as error:  # a broken backend must not crash
                logging.warning('Veda: backend %s failed to load', name,
                                exc_info=True)
                attempts.append((name, f'{type(error).__name__}: {error}'))
                continue
            attempts.append((backend.name, 'ok'))
            chosen = backend
            break
        resolution = Resolution(info, chosen, attempts)
        _RESOLVED[key] = resolution
        return resolution


def probe(device: torch.device, requested: str = 'auto'
          ) -> list[tuple[str, str, str | None]]:
    """(name, display name, error or None) of each candidate on `device`,
    without compiling or self-testing (cheap; for the status shown before
    sampling)."""
    info = hardware.describe(device)
    out = []
    for name in candidates(info, requested):
        try:
            backend = _load(name, info)
            out.append((backend.name, backend.display, None))
        except Exception as error:  # report every failure the same way
            out.append((name, name, str(error)))
    return out


def reset() -> None:
    """Forgets resolutions (tests, or after installing kernels)."""
    with _LOCK:
        _RESOLVED.clear()
