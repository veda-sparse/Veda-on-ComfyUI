"""Regenerates the private FlashAttention-4 (and QuACK) copies in
veda_comfy/_vendor.

    python tools/vendor_fa4.py            # rewrite veda_comfy/_vendor
    python tools/vendor_fa4.py --check    # verify the committed copies
    python tools/vendor_fa4.py --work-dir DIR   # patched tree to edit

Three packages are generated from pinned wheels, never edited by hand:

  fa4_upstream  unmodified FA4 (SM90 / SM100 backends)
  fa4_sm8x      FA4 + Miowtion's SM8x / SM120 block-sparse patch series
                (tools/fa4_patches/sm8x, applied with `git apply`;
                SM80-family and SM120 backends)
  quack         QuACK (`quack-kernels`), FA4's runtime helper library,
                shared by both FA4 copies

All of them get the same mechanical changes:
  1. Imports of `flash_attn.cute` / `quack` are rewritten to relative ones,
     so each copy is a private package: no global `flash_attn` / `quack`
     import, no clash with another custom node, no import-order dependency.
  2. Small compatibility edits (COMPAT_EDITS), each of which must match
     exactly once or the tool fails. Today: FA4 and QuACK import `fcntl` at
     module scope (Linux-only) for their file-locked JIT caches; on Windows
     the import becomes optional and those caches run without file locks.

Every input is pinned by sha256 (wheels, the upstream files the patches
touch, the patched results), so a regeneration is reproducible and an
upstream change cannot slip in silently. Needs `git` on PATH.

To write a new patch, `--work-dir DIR` leaves the patched FA4 tree in a
git repo (upstream as the first commit, the patch series as the second).
Edit it there, then `git -C DIR format-patch HEAD~1` and drop the result
into the matching `tools/fa4_patches/<series>/`; the hashes in
PATCHED_SHA256 then have to be refreshed from the new output.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(ROOT, 'veda_comfy', '_vendor')
PATCHES = os.path.join(ROOT, 'tools', 'fa4_patches', 'sm8x')

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

# Upstream FA4 files the patch series modifies, and the patched results, as
# pinned by Miowtion (miowtion/kernels/fa4_sm8x/__init__.py and its
# vendored files). sha256 before import rewriting / compat edits.
UPSTREAM_SHA256 = {
    'block_sparsity.py':
        '9331cb2abd4be4da87a74c0839b665d5f914ac6f51cb97491e97b26199668ac9',
    'block_sparse_utils.py':
        '608f6c5a1c68daadfca402ec1259917019a195dc748ce9f24b92f0551d2ff6ba',
    'flash_fwd.py':
        'be6e88c7d9f122aa5ab29ff5946f6270af0b580e4403351b869f1330b71ca53c',
    'flash_bwd.py':
        '80cb6bfb436160d73b7214529f94d28b7333c98dff75a53d20552eb33645fea2',
    'interface.py':
        '144a3dd6f72f955e43834500808c7d47b3b4a76fdcd0b7188f9b459d85007cab',
    # SM120 block sparsity relies on these being thin subclasses of the
    # patched SM80 kernels; the FP8 patch also widens sm120's dtype gate.
    'flash_fwd_sm120.py':
        'abd017add69914e46f0fcfc017ad51c3d07d79fd87d15f969036ddd82bcc7da2',
    'flash_bwd_sm120.py':
        'ad2d802c97dbf654fd03724bcfa91702b18acec8518e5dd4a33f137ba238bf7d',
}
PATCHED_SHA256 = {
    'block_sparse_utils.py':
        '69b7a955e7a7cb756b0b475241e771feba7497670b570dd5ddb279e2b51ac7ad',
    'block_sparsity.py':
        '9b539633f15a8475c873dfff30ee7090144b466cc6bed4665c6149c0c7d66cdf',
    'flash_bwd.py':
        '0d7fffb59e3013d0e24172f7c897f9a0fec71549c62ed1b1b29e19358b33038c',
    'flash_fwd.py':
        'fef42bc617e167f4b8c160e5560a8900129f9591179f0989b2ac7f7fac1f43f5',
    'flash_fwd_sm120.py':
        'a3f0b2dcbe727b256555be25c04aada7e87a0ecf996bd6e4e8e949bfb5939719',
    'interface.py':
        '38d9a337f7ab266203b1cd7db26dc3603b63c6235c09e0a9d58727d2bc63294d',
}

_OPTIONAL_FCNTL = ('try:\n'
                   '    import fcntl\n'
                   'except ImportError:  # veda: no fcntl on Windows; the\n'
                   '    fcntl = None  # JIT cache runs without file locks\n')
_NO_LOCK = ('    def __enter__(self) -> "FileLock":\n'
            '        if fcntl is None:  # veda: Windows, no advisory locks\n'
            '            return self\n')

# (package, path inside it, old text, new text); old must match once.
COMPAT_EDITS = [
    ('fa4', 'cache_utils.py', 'import fcntl\n', _OPTIONAL_FCNTL),
    ('fa4', 'cache_utils.py', '    def __enter__(self) -> "FileLock":\n',
     _NO_LOCK),
    ('fa4', 'cache_utils.py',
     '        if self._fd is not None:\n'
     '            fcntl.flock(self._fd, fcntl.LOCK_UN)\n',
     '        if fcntl is not None and self._fd is not None:\n'
     '            fcntl.flock(self._fd, fcntl.LOCK_UN)\n'),
    ('quack', 'cache/jit.py', 'import fcntl\n', _OPTIONAL_FCNTL),
    ('quack', 'cache/jit.py', '    def __enter__(self) -> "FileLock":\n',
     _NO_LOCK),
    ('quack', 'cache/async_compile.py', 'import fcntl\n', _OPTIONAL_FCNTL),
    ('quack', 'cache/async_compile.py',
     '    try:\n        fd = os.open(lock_path, os.O_RDONLY | os.O_CREAT)\n',
     '    if fcntl is None:  # veda: Windows, no advisory locks\n'
     '        return False\n'
     '    try:\n        fd = os.open(lock_path, os.O_RDONLY | os.O_CREAT)\n'),
    # rmsnorm / softmax / cross_entropy pull QuACK's autotuner, which
    # subclasses a Triton class at import time; Windows torch has no
    # Triton and FA4 uses none of them, so they load on first access.
    ('quack', '__init__.py',
     'from quack.rmsnorm import rmsnorm  # noqa: E402\n'
     'from quack.softmax import softmax  # noqa: E402\n'
     'from quack.cross_entropy import cross_entropy  # noqa: E402\n',
     '\n\ndef __getattr__(name):  # veda: lazy, keeps Triton optional\n'
     '    if name in ("rmsnorm", "softmax", "cross_entropy"):\n'
     '        import importlib\n'
     '        module = importlib.import_module("." + name, __name__)\n'
     '        return getattr(module, name)\n'
     '    raise AttributeError(name)\n\n\n'),
    # The forkserver preload names the module by its upstream path.
    ('quack', 'cache/async_compile.py', '["quack.cache._pool_preload"]',
     '[__name__.rsplit(".", 1)[0] + "._pool_preload"]'),
]

_SKIP_DIRS = ('quack/testing/',)
_LEFTOVER = re.compile(r'^\s*(from|import)\s+(flash_attn|quack)\b')


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _download(url: str, digest: str, path: str | None, what: str) -> bytes:
    if path:
        with open(path, 'rb') as f:
            data = f.read()
    else:
        print(f'downloading {what} ...', flush=True)
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
    if _sha256(data) != digest:
        sys.exit(f'{what} wheel sha256 mismatch: refusing to vendor')
    return data


def _unzip(wheel: bytes, prefix: str) -> tuple[dict[str, bytes],
                                              dict[str, bytes]]:
    """(*.py under prefix keyed by relative path, license files)."""
    files, licenses = {}, {}
    with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
        for name in archive.namelist():
            if (name.startswith(prefix) and name.endswith('.py')
                    and not name.startswith(_SKIP_DIRS)):
                files[name[len(prefix):]] = archive.read(name)
            elif '/licenses/' in name:
                licenses[os.path.basename(name)] = archive.read(name)
    return files, licenses


def _write_tree(root: str, files: dict[str, bytes]) -> str:
    cute = os.path.join(root, 'flash_attn', 'cute')
    os.makedirs(cute, exist_ok=True)
    for rel, data in files.items():
        with open(os.path.join(cute, rel), 'wb') as f:
            f.write(data)
    return cute


def _git(work_dir: str, *args: str) -> None:
    subprocess.run(['git', *args], cwd=work_dir, check=True,
                   stdout=subprocess.DEVNULL)


def _patch_files() -> list[str]:
    return sorted(os.path.join(PATCHES, p) for p in os.listdir(PATCHES)
                  if p.endswith('.patch'))


def _apply_patches(files: dict[str, bytes],
                   verify: bool = True) -> dict[str, bytes]:
    with tempfile.TemporaryDirectory() as tmp:
        cute = _write_tree(tmp, files)
        subprocess.run(['git', 'apply', '--exclude=tests/*',
                        *_patch_files()], cwd=tmp, check=True)
        patched = {}
        for rel in files:
            with open(os.path.join(cute, rel), 'rb') as f:
                patched[rel] = f.read()
    if verify:
        for rel, digest in PATCHED_SHA256.items():
            if _sha256(patched[rel]) != digest:
                sys.exit(f'patched {rel} differs from the pinned result')
    return patched


def update_hashes(fa4_wheel: str | None) -> None:
    """Rewrites PATCHED_SHA256 in this file from the current patches.

    Editing those digests by hand is how a stale vendored copy sneaks in:
    generate() then exits, and anything that hid its stderr kept building
    against the previous kernel.
    """
    files, _ = _unzip(_download(FA4_WHEEL, FA4_WHEEL_SHA256, fa4_wheel,
                                'FA4'), 'flash_attn/cute/')
    patched = _apply_patches(files, verify=False)
    changed = {rel: _sha256(data) for rel, data in patched.items()
               if data != files[rel]}
    path = os.path.abspath(__file__)
    with open(path, encoding='utf-8') as f:
        source = f.read()
    start = source.index('PATCHED_SHA256 = {')
    end = source.index('}', start) + 1
    body = ''.join(f"    {rel!r}:\n        {digest!r},\n"
                   for rel, digest in sorted(changed.items()))
    source = source[:start] + 'PATCHED_SHA256 = {\n' + body + '}' + source[end:]
    with open(path, 'w', encoding='utf-8') as f:
        f.write(source)
    print(f'PATCHED_SHA256 now pins {len(changed)} files: '
          f'{", ".join(sorted(changed))}')


def work_dir(path: str, fa4_wheel: str | None) -> None:
    """Lays out the patched FA4 tree in a git repo, ready to edit."""
    if os.path.exists(path) and os.listdir(path):
        sys.exit(f'{path} exists and is not empty')
    files, _ = _unzip(_download(FA4_WHEEL, FA4_WHEEL_SHA256, fa4_wheel,
                                'FA4'), 'flash_attn/cute/')
    os.makedirs(path, exist_ok=True)
    _write_tree(path, files)
    _git(path, 'init', '-q')
    _git(path, 'add', '-A')
    _git(path, '-c', 'user.name=vendor', '-c', 'user.email=vendor@local',
         'commit', '-q', '-m', f'FlashAttention-4 {FA4_VERSION} (upstream)')
    subprocess.run(['git', 'apply', '--exclude=tests/*', *_patch_files()],
                   cwd=path, check=True)
    _git(path, 'add', '-A')
    _git(path, '-c', 'user.name=vendor', '-c', 'user.email=vendor@local',
         'commit', '-q', '-m', 'sm8x block-sparse patch series')
    print(f'patched FA4 tree in {path} (2 commits). Edit, then:\n'
          f'  git -C {path} commit -am "<title>" && '
          f'git -C {path} format-patch HEAD~1')


def _import_rules(own: str, depth: int) -> list[tuple[re.Pattern, str]]:
    """Rewrites for a file `depth` directories below its package root.

    `own` is the upstream dotted name of the file's own package
    ('flash_attn.cute' or 'quack'); `quack` is a sibling package under
    veda_comfy/_vendor for every vendored file.
    """
    here = '.' * (depth + 1)        # the file's own package root
    vendor = '.' * (depth + 2)      # veda_comfy._vendor
    rules = []
    for name, base in ((own, here),) + (
            (('quack', vendor + 'quack'),) if own != 'quack' else ()):
        n = re.escape(name)
        # from X import a / from X.m.n import a
        rules.append((re.compile(rf'^(\s*)from {n} import '),
                      rf'\g<1>from {base} import '))
        rules.append((re.compile(rf'^(\s*)from {n}\.([\w.]+) import '),
                      rf'\g<1>from {base}{"" if base.endswith(".") else "."}'
                      r'\g<2> import '))
        # import X.m as y  /  import X.a.b as y
        rules.append((re.compile(
            rf'^(\s*)import {n}\.(\w+) as (\w+)(\s*(?:#.*)?)$'),
                      rf'\g<1>from {base} import \g<2> as \g<3>\g<4>'))
        rules.append((re.compile(
            rf'^(\s*)import {n}\.([\w.]+)\.(\w+) as (\w+)(\s*(?:#.*)?)$'),
            rf'\g<1>from {base}{"" if base.endswith(".") else "."}\g<2> '
            r'import \g<3> as \g<4>\g<5>'))
        # import X.m / import X.a.b (side-effect imports, name unused)
        rules.append((re.compile(rf'^(\s*)import {n}\.(\w+)(\s*(?:#.*)?)$'),
                      rf'\g<1>from {base} import \g<2>\g<3>'))
        rules.append((re.compile(
            rf'^(\s*)import {n}\.([\w.]+)\.(\w+)(\s*(?:#.*)?)$'),
            rf'\g<1>from {base}{"" if base.endswith(".") else "."}\g<2> '
            r'import \g<3>\g<4>'))
    # import quack as y (the package object itself)
    rules.append((re.compile(r'^(\s*)import quack as (\w+)(\s*(?:#.*)?)$'),
                  rf'\g<1>from {vendor} import quack as \g<2>\g<3>'))
    return rules


def _rewrite(package: str, rel: str, data: bytes, header: str) -> bytes:
    text = data.decode('utf-8')
    for target, path, old, new in COMPAT_EDITS:
        if target == package and path == rel:
            if text.count(old) != 1:
                sys.exit(f'compat edit does not match {package}/{rel} '
                         f'exactly once: {old!r}')
            text = text.replace(old, new)
    own = 'quack' if package == 'quack' else 'flash_attn.cute'
    rules = _import_rules(own, rel.count('/'))
    lines = []
    for line in text.splitlines(keepends=True):
        for pattern, replacement in rules:
            line = pattern.sub(replacement, line)
        if _LEFTOVER.match(line):
            sys.exit(f'{package}/{rel}: unhandled import: {line.strip()}')
        lines.append(line)
    return (header + ''.join(lines)).encode('utf-8')


def _write_package(target: str, package: str, sources: dict[str, bytes],
                   licenses: dict[str, bytes], header: str,
                   provenance: dict) -> None:
    for rel, data in sorted(sources.items()):
        path = os.path.join(target, *rel.split('/'))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(_rewrite(package, rel, data, header))
    for rel, data in licenses.items():
        with open(os.path.join(target, rel), 'wb') as f:
            f.write(data)
    with open(os.path.join(target, 'VENDORED.json'), 'w') as f:
        json.dump(provenance, f, indent=1, sort_keys=True)
        f.write('\n')


def generate(out_dir: str, fa4_wheel: str | None,
             quack_wheel: str | None) -> None:
    fa4_files, fa4_licenses = _unzip(
        _download(FA4_WHEEL, FA4_WHEEL_SHA256, fa4_wheel, 'FA4'),
        'flash_attn/cute/')
    if any('/' in rel for rel in fa4_files):
        sys.exit('unexpected subpackage in FA4')
    for rel, digest in UPSTREAM_SHA256.items():
        if _sha256(fa4_files[rel]) != digest:
            sys.exit(f'upstream FA4 {rel} differs from the pinned base')
    fa4_header = ('# Generated by tools/vendor_fa4.py from FlashAttention-4 '
                  f'{FA4_VERSION}\n# (BSD-3-Clause, see LICENSE). '
                  'Do not edit.\n')
    edits = sorted({f'{p}/{r}' for p, r, _, _ in COMPAT_EDITS})
    for name, sources in (('fa4_upstream', fa4_files),
                          ('fa4_sm8x', _apply_patches(fa4_files))):
        patches = (sorted(p for p in os.listdir(PATCHES)
                          if p.endswith('.patch'))
                   if name == 'fa4_sm8x' else [])
        _write_package(os.path.join(out_dir, name), 'fa4', sources,
                       fa4_licenses, fa4_header,
                       {'package': 'flash-attn-4', 'version': FA4_VERSION,
                        'wheel': FA4_WHEEL, 'wheel_sha256': FA4_WHEEL_SHA256,
                        'patches': patches, 'compat_edits': edits,
                        'generator': 'tools/vendor_fa4.py'})
    quack_files, quack_licenses = _unzip(
        _download(QUACK_WHEEL, QUACK_WHEEL_SHA256, quack_wheel, 'QuACK'),
        'quack/')
    quack_header = ('# Generated by tools/vendor_fa4.py from quack-kernels '
                    f'{QUACK_VERSION}\n# (Apache-2.0, see LICENSE). '
                    'Do not edit.\n')
    _write_package(os.path.join(out_dir, 'quack'), 'quack', quack_files,
                   quack_licenses, quack_header,
                   {'package': 'quack-kernels', 'version': QUACK_VERSION,
                    'wheel': QUACK_WHEEL, 'wheel_sha256': QUACK_WHEEL_SHA256,
                    'patches': [], 'compat_edits': edits,
                    'generator': 'tools/vendor_fa4.py'})


NAMES = ('fa4_upstream', 'fa4_sm8x', 'quack')


def _tree(root: str) -> dict[str, str]:
    out = {}
    for base, _, names in os.walk(root):
        if '__pycache__' in base:
            continue
        for name in names:
            path = os.path.join(base, name)
            with open(path, 'rb') as f:
                out[os.path.relpath(path, root)] = _sha256(f.read())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--check', action='store_true',
                        help='verify veda_comfy/_vendor instead of writing')
    parser.add_argument('--wheel', help='local FA4 wheel (else download)')
    parser.add_argument('--work-dir',
                        help='lay out the patched FA4 tree here for editing')
    parser.add_argument('--update-hashes', action='store_true',
                        help='rewrite PATCHED_SHA256 from the patch series')
    parser.add_argument('--quack-wheel', help='local QuACK wheel')
    args = parser.parse_args()
    if args.update_hashes:
        update_hashes(args.wheel)
        return
    if args.work_dir:
        work_dir(args.work_dir, args.wheel)
        return
    with tempfile.TemporaryDirectory() as tmp:
        generate(tmp, args.wheel, args.quack_wheel)
        if args.check:
            for name in NAMES:
                if _tree(os.path.join(tmp, name)) != _tree(
                        os.path.join(VENDOR, name)):
                    sys.exit(f'veda_comfy/_vendor/{name} is not what '
                             'tools/vendor_fa4.py generates; regenerate it')
            print('vendored copies are up to date')
            return
        for name in NAMES:
            target = os.path.join(VENDOR, name)
            shutil.rmtree(target, ignore_errors=True)
            shutil.copytree(os.path.join(tmp, name), target)
        print(f'wrote {", ".join(NAMES)} into veda_comfy/_vendor')


if __name__ == '__main__':
    main()
