"""Reports what this GPU and the installed CuTe DSL can do, for kernel work.

    python tools/probe_gpu_kernels.py

Prints the device, the tensor-core throughput of the dtypes we care about,
and which CuTe warp-level MMA / LdMatrix atoms exist and actually compile
here. Veda's FP8 kernel work depends on facts this tool checks rather than
on what upstream's Python-level asserts allow, so re-run it on every new
architecture before assuming a path is unavailable.
"""

from __future__ import annotations

import argparse
import time

import torch

try:  # only on machines with the FA4 kernel runtime
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import warp
    from cutlass.torch import from_dlpack
except ImportError:  # reported by cute_atoms()
    cutlass = None


def _mma_build(make_op):
    """A jit body that builds one tiled MMA (values captured by closure).

    Closure cells, not default arguments: the DSL traces parameters as
    runtime values, so a bool default arrives as a DSL value and the op
    rejects it.
    """
    def build(dummy):
        cute.make_tiled_mma(cute.make_mma_atom(make_op()))

    return build


def _ldmatrix_build(op, transpose, dtype):
    def build(dummy):
        cute.make_copy_atom(op(transpose=transpose, num_matrices=4), dtype)

    return build


def _compiles(build) -> str | None:
    """None if `build` compiles for the resident GPU, else the reason.

    The jit-traced function must find `cute` in module globals, so every
    import above stays at module scope.
    """
    dummy = from_dlpack(torch.zeros(8, device='cuda'))
    try:
        cute.compile(cute.jit(build), dummy)
        return None
    except Exception as error:
        first = str(error).splitlines()
        detail = next((x for x in first if x.strip()), '')
        return f'{type(error).__name__}: {detail[:90]}'


def _bench(fn, repeat: int = 30) -> float:
    fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeat


def throughput(size: int = 8192) -> None:
    """Dense GEMM throughput per dtype, as reachable through torch."""
    print('\n== tensor-core throughput (torch paths) ==')
    flops = 2 * size**3
    a = torch.randn(size, size, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(size, size, device='cuda', dtype=torch.bfloat16)
    base = _bench(lambda: a @ b)
    print(f'bf16  {flops / base / 1e12:6.0f} TFLOPS  (1.00x)')
    try:
        af = a.to(torch.float8_e4m3fn)
        bf = b.t().contiguous().t().to(torch.float8_e4m3fn)
        one = torch.tensor(1.0, device='cuda')
        seconds = _bench(lambda: torch._scaled_mm(af, bf, scale_a=one,
                                                  scale_b=one,
                                                  out_dtype=torch.bfloat16))
        print(f'fp8   {flops / seconds / 1e12:6.0f} TFLOPS  '
              f'({base / seconds:.2f}x)')
    except Exception as error:
        print(f'fp8   unavailable: {type(error).__name__}: {error}')
    try:
        ai = torch.randint(-127, 127, (size, size), device='cuda',
                           dtype=torch.int8)
        bi = torch.randint(-127, 127, (size, size), device='cuda',
                           dtype=torch.int8)
        seconds = _bench(lambda: torch._int_mm(ai, bi))
        print(f'int8  {flops / seconds / 1e12:6.0f} TOPS    '
              f'({base / seconds:.2f}x)')
    except Exception as error:
        print(f'int8  unavailable: {type(error).__name__}: {error}')


def cute_atoms() -> None:
    """Which warp-level atoms the installed CuTe DSL exposes and compiles."""
    if cutlass is None:
        print('\n== CuTe DSL ==\nnot installed')
        return
    print(f'\n== CuTe DSL {cutlass.__version__} ==')
    print('warp atoms:', ', '.join(sorted(
        n for n in dir(warp) if n.endswith('Op'))))

    print('\nMMA atoms (what the QK / PV gemms can use):')
    mma_cases = [
        ('MmaF16BF16Op bf16 16x8x16',
         lambda: warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32,
                                   (16, 8, 16))),
        ('MmaFP8Op e4m3 16x8x32',
         lambda: warp.MmaFP8Op(cutlass.Float8E4M3FN, cutlass.Float32,
                               (16, 8, 32))),
        ('MmaFP8Op e4m3 16x8x16',
         lambda: warp.MmaFP8Op(cutlass.Float8E4M3FN, cutlass.Float32,
                               (16, 8, 16))),
        ('MmaFP8Op e5m2 16x8x32',
         lambda: warp.MmaFP8Op(cutlass.Float8E5M2, cutlass.Float32,
                               (16, 8, 32))),
    ]
    for name, make_op in mma_cases:
        reason = _compiles(_mma_build(make_op))
        print(f'  {"ok   " if reason is None else "FAILS"} {name}'
              + ('' if reason is None else f': {reason}'))

    # The PV gemm needs V with the contraction dim contiguous. For 16-bit
    # operands the kernel gets that from a transposing ldmatrix; whether a
    # 1-byte transposing load exists here decides if Veda must transpose V
    # before the kernel sees it.
    print('\nldmatrix atoms on 1-byte operands (V for the PV gemm):')
    for name in sorted(n for n in dir(warp)
                       if n.startswith('LdMatrix') and n.endswith('Op')):
        op = getattr(warp, name)
        for transpose in (False, True):
            reason = _compiles(_ldmatrix_build(op, transpose,
                                               cutlass.Float8E4M3FN))
            print(f'  {"ok   " if reason is None else "FAILS"} '
                  f'{name}(transpose={transpose})'
                  + ('' if reason is None else f': {reason}'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--size', type=int, default=8192,
                        help='square GEMM size for the throughput test')
    parser.add_argument('--skip-throughput', action='store_true')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('this tool needs a CUDA GPU')
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    print(f'{name} (sm{major}{minor}) · torch {torch.__version__} · '
          f'CUDA {torch.version.cuda}')
    if not args.skip_throughput:
        throughput(args.size)
    cute_atoms()


if __name__ == '__main__':
    main()
