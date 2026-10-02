"""Test setup: import `veda_comfy` from the repo, find ComfyUI if present.

ComfyUI-backed tests need a ComfyUI checkout: set COMFYUI_ROOT, or run the
tests from `ComfyUI/custom_nodes/<this repo>`; they are skipped otherwise.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)


def _comfyui_root():
    candidates = [os.environ.get('COMFYUI_ROOT'),
                  os.path.dirname(os.path.dirname(ROOT))]
    for path in candidates:
        if path and os.path.isfile(os.path.join(path, 'comfy', 'ops.py')):
            return path
    return None


COMFYUI_ROOT = _comfyui_root()
if COMFYUI_ROOT is not None and COMFYUI_ROOT not in sys.path:
    sys.path.append(COMFYUI_ROOT)
