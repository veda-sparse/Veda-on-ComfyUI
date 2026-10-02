"""Installs Veda's optional fast kernels into the Python that runs ComfyUI.

    <ComfyUI python> custom_nodes/Veda-on-ComfyUI/install_fa4.py [--check]

(install_fa4.bat / install_fa4.sh find that Python for you.)

NVIDIA (Linux / Windows): installs the FlashAttention-4 runtime
(requirements-fa4.txt: CuTe DSL etc.; FA4 itself ships in this repo).
torch is pinned to the installed version, so this never swaps or upgrades
torch. CUDA 13 builds of torch (e.g. DGX Spark) get the cu13 CuTe DSL.

Apple silicon: installs MLX instead.

Afterwards it runs Veda's backend self-tests on every GPU and prints which
kernel each one will use. --check only runs the self-tests.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def _pip(args: list[str]) -> None:
    commands = [[sys.executable, '-m', 'pip', 'install', *args]]
    if shutil.which('uv'):
        commands.append(['uv', 'pip', 'install', '--python', sys.executable,
                         *args])
    for command in commands:
        print('$', ' '.join(command), flush=True)
        if subprocess.call(command) == 0:
            return
    sys.exit('installation failed; see the messages above')


def _requirements(cuda_major: int | None) -> str:
    with open(os.path.join(HERE, 'requirements-fa4.txt'),
              encoding='utf-8') as f:
        lines = f.read().splitlines()
    if cuda_major is not None and cuda_major >= 13:
        lines = [line.replace('nvidia-cutlass-dsl==',
                              'nvidia-cutlass-dsl[cu13]==') for line in lines]
    return '\n'.join(lines) + '\n'


def install() -> None:
    import torch  # pylint: disable=import-outside-toplevel
    print(f'Python {sys.version.split()[0]} at {sys.executable}')
    print(f'torch {torch.__version__}')
    if sys.platform == 'darwin':
        if platform.machine() != 'arm64':
            sys.exit('Veda kernels need Apple silicon or an NVIDIA GPU.')
        _pip(['mlx'])
        return
    if not torch.cuda.is_available() or torch.version.hip:
        sys.exit('FA4 needs an NVIDIA GPU with a CUDA build of torch; '
                 'Veda still runs with its portable kernels.')
    cuda_major = int(torch.version.cuda.split('.')[0])
    with tempfile.TemporaryDirectory() as tmp:
        requirements = os.path.join(tmp, 'requirements.txt')
        constraints = os.path.join(tmp, 'constraints.txt')
        with open(requirements, 'w', encoding='utf-8') as f:
            f.write(_requirements(cuda_major))
        with open(constraints, 'w', encoding='utf-8') as f:
            f.write(f'torch=={torch.__version__}\n')
        _pip(['-r', requirements, '-c', constraints])


def check() -> int:
    """Runs the backend self-tests on every accelerator; 0 if one works."""
    sys.path.insert(0, HERE)
    import torch  # pylint: disable=import-outside-toplevel
    from veda_comfy import backends  # pylint: disable=import-outside-toplevel
    devices = [torch.device('cuda', i)
               for i in range(torch.cuda.device_count())]
    if not devices and torch.backends.mps.is_available():
        devices = [torch.device('mps')]
    if not devices:
        devices = [torch.device('cpu')]
    ok = False
    for device in devices:
        resolution = backends.resolve(
            device, notify=lambda text: print(f'  ... {text}', flush=True))
        name = resolution.backend.name if resolution.backend else 'none'
        print(f'{resolution.device.label}: Veda uses {name}')
        for candidate, status in resolution.attempts:
            print(f'  {candidate}: {status}')
        ok = ok or resolution.backend is not None
    return 0 if ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--check', action='store_true',
                        help='only run the backend self-tests')
    args = parser.parse_args()
    if args.check:
        print('\nBackend self-tests:', flush=True)
        sys.exit(check())
    install()
    # A fresh interpreter: packages installed a moment ago are not visible
    # to this one's import system (cached directory listings).
    sys.exit(subprocess.call([sys.executable, os.path.abspath(__file__),
                              '--check']))


if __name__ == '__main__':
    main()
