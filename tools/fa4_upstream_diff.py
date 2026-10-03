"""Provenance for Veda's FlashAttention-4 CuTe fork.

    python tools/fa4_upstream_diff.py --write   # refresh UPSTREAM.diff
    python tools/fa4_upstream_diff.py           # check it is current

`veda_comfy/kernels/fa4` and `veda_comfy/kernels/quack` are a fork we
edit directly (see docs/features/fa4_fork.md). This tool keeps the record
of what we changed: it downloads the pinned upstream wheels, reproduces
the mechanical transformations that are not ours to claim (keeping only
the modules our entry points import, and rewriting the upstream package
imports to relative ones), and diffs that baseline against the fork.

The result, `veda_comfy/kernels/UPSTREAM.diff`, is what the BSD-3 and
Apache-2.0 notices point at, and what to read when deciding whether an
upstream change is worth merging. CI checks it is up to date, which also
catches an edit that was never committed.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import io
import os
import re
import sys
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNELS = os.path.join(ROOT, 'veda_comfy', 'kernels')
DIFF_PATH = os.path.join(KERNELS, 'UPSTREAM.diff')

FA4_VERSION = '4.0.0b32'
FA4_WHEEL = ('https://files.pythonhosted.org/packages/40/d0/'
             'b1e9ab8eefc9e67179a372c3e1a8e9f83d963c3a8e5bd4c738e55dd134c0/'
             'flash_attn_4-4.0.0b32-py3-none-any.whl')
FA4_WHEEL_SHA256 = ('9299da3524d579d84eb92954d3243aa095d55ebdafcf80d6a9766fd9'
                    '6fe8a562')
QUACK_VERSION = '0.6.5'
QUACK_WHEEL = ('https://files.pythonhosted.org/packages/73/d8/'
               'f5e56c38b286308d115a171e3c5c44f24e9d03980c3cfd22588df40261fa/'
               'quack_kernels-0.6.5-py3-none-any.whl')
QUACK_WHEEL_SHA256 = ('df1ddd31366c82eb9692fe087c671209189dcbde53c909c81a865d'
                      '4ee66b158d')

# Imports of these upstream packages become relative ones inside the fork.
_RULES = [
    (re.compile(r'^(\s*)from flash_attn\.cute import '), r'\1from . import '),
    (re.compile(r'^(\s*)from flash_attn\.cute\.([\w.]+) import '),
     r'\1from .\2 import '),
    (re.compile(r'^(\s*)import flash_attn\.cute\.(\w+) as (\w+)(\s*(?:#.*)?)$'),
     r'\1from . import \2 as \3\4'),
    (re.compile(r'^(\s*)from quack import '), r'\1from ..quack import '),
    (re.compile(r'^(\s*)from quack\.([\w.]+) import '),
     r'\1from ..quack.\2 import '),
    (re.compile(r'^(\s*)import quack\.(\w+) as (\w+)(\s*(?:#.*)?)$'),
     r'\1from ..quack import \2 as \3\4'),
]
_QUACK_RULES = [
    (re.compile(r'^(\s*)from quack import '), r'\1from . import '),
    (re.compile(r'^(\s*)from quack\.([\w.]+) import '), r'\1from .\2 import '),
    (re.compile(r'^(\s*)import quack\.(\w+) as (\w+)(\s*(?:#.*)?)$'),
     r'\1from . import \2 as \3\4'),
]


def _download(url: str, digest: str, path: str | None, what: str) -> bytes:
    if path:
        with open(path, 'rb') as f:
            data = f.read()
    else:
        print(f'downloading {what} ...', flush=True)
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != digest:
        sys.exit(f'{what} wheel sha256 mismatch')
    return data


def _members(wheel: bytes, prefix: str) -> dict[str, str]:
    out = {}
    with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
        for name in archive.namelist():
            if name.startswith(prefix) and name.endswith('.py'):
                out[name[len(prefix):]] = archive.read(name).decode('utf-8')
    return out


def _rewrite(text: str, rules) -> str:
    lines = []
    for line in text.splitlines(keepends=True):
        for pattern, replacement in rules:
            line = pattern.sub(replacement, line)
        lines.append(line)
    return ''.join(lines)


def _fork_files(package: str) -> dict[str, str]:
    root = os.path.join(KERNELS, package)
    out = {}
    for base, _, names in os.walk(root):
        for name in names:
            if name.endswith('.py'):
                path = os.path.join(base, name)
                with open(path, encoding='utf-8') as f:
                    out[os.path.relpath(path, root).replace(os.sep, '/')] = (
                        f.read())
    return out


def _strip_header(text: str) -> str:
    """Drops the fork banner so the diff shows real changes only."""
    lines = text.splitlines(keepends=True)
    kept = [x for x in lines[:3] if not x.startswith('# ')]
    return ''.join(kept + lines[3:])


def build_diff(fa4_wheel: str | None, quack_wheel: str | None) -> str:
    sources = {
        'fa4': (_members(_download(FA4_WHEEL, FA4_WHEEL_SHA256, fa4_wheel,
                                   'FA4'), 'flash_attn/cute/'), _RULES,
                f'flash-attn-4 {FA4_VERSION}'),
        'quack': (_members(_download(QUACK_WHEEL, QUACK_WHEEL_SHA256,
                                     quack_wheel, 'QuACK'), 'quack/'),
                  _QUACK_RULES, f'quack-kernels {QUACK_VERSION}'),
    }
    chunks = [
        "Veda's FlashAttention-4 CuTe fork, against upstream.\n",
        '\nGenerated by tools/fa4_upstream_diff.py. The baseline is the\n'
        'pinned wheel with two mechanical changes excluded: only the\n'
        'modules our entry points import are kept, and upstream package\n'
        'imports are rewritten to relative ones.\n',
    ]
    for package, (upstream, rules, label) in sources.items():
        fork = _fork_files(package)
        chunks.append(f'\n{"=" * 70}\n{package}: forked from {label}\n'
                      f'{"=" * 70}\n')
        missing = sorted(set(fork) - set(upstream))
        if missing:
            chunks.append(f'files added by us: {", ".join(missing)}\n')
        for rel in sorted(fork):
            if rel not in upstream:
                continue
            base = _rewrite(upstream[rel], rules)
            ours = _strip_header(fork[rel])
            if base == ours:
                continue
            chunks.append(''.join(difflib.unified_diff(
                base.splitlines(keepends=True), ours.splitlines(keepends=True),
                fromfile=f'upstream/{rel}', tofile=f'fork/{rel}', n=3)))
        dropped = len(set(upstream) - set(fork))
        chunks.append(f'\n({dropped} upstream modules are not part of the '
                      'fork: nothing our entry points import.)\n')
    return ''.join(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--write', action='store_true',
                        help='refresh UPSTREAM.diff instead of checking it')
    parser.add_argument('--wheel', help='local FA4 wheel (else download)')
    parser.add_argument('--quack-wheel', help='local QuACK wheel')
    args = parser.parse_args()
    diff = build_diff(args.wheel, args.quack_wheel)
    if args.write:
        with open(DIFF_PATH, 'w', encoding='utf-8') as f:
            f.write(diff)
        changed = diff.count('--- upstream/')
        print(f'wrote {os.path.relpath(DIFF_PATH, ROOT)} '
              f'({changed} modified modules)')
        return
    if not os.path.exists(DIFF_PATH):
        sys.exit('UPSTREAM.diff is missing; run with --write')
    with open(DIFF_PATH, encoding='utf-8') as f:
        if f.read() != diff:
            sys.exit('UPSTREAM.diff is stale; run '
                     'tools/fa4_upstream_diff.py --write')
    print('UPSTREAM.diff is current')


if __name__ == '__main__':
    main()
