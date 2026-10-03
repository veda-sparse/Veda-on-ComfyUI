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
    try:
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.nvgpu import warp
        from cutlass.torch import from_dlpack
    except ImportError as error:
        print(f'\n== CuTe DSL ==\nnot installed ({error})')
        return
    print(f'\n== CuTe DSL {cutlass.__version__} ==')
    print('warp atoms:', ', '.join(sorted(
        n for n in dir(warp) if n.endswith('Op'))))

    # Each case builds a tiled MMA inside a jit function: the only honest
    # test is whether it compiles for the resident GPU.
    cases = [
        ('MmaF16BF16Op bf16 16x8x16',
         lambda: warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32,
                                   (16, 8, 16))),
        ('MmaFP8Op e4m3 16x8x32',
         lambda: warp.MmaFP8Op(cutlass.Float8E4M3FN, cutlass.Float32,
                               (16, 8, 32))),
        ('MmaFP8Op e5m2 16x8x32',
         lambda: warp.MmaFP8Op(cutlass.Float8E5M2, cutlass.Float32,
                               (16, 8, 32))),
    ]
    for name, make_op in cases:
        @cute.jit
        def build(dummy: cute.Tensor, make_op=make_op):
            cute.make_tiled_mma(cute.make_mma_atom(make_op()))

        dummy = from_dlpack(torch.zeros(8, device='cuda'))
        try:
            cute.compile(build, dummy)
            print(f'  compiles: {name}')
        except Exception as error:
            print(f'  FAILS:    {name}: {type(error).__name__}: '
                  f'{str(error).splitlines()[0][:90]}')

    # ldmatrix: FP8 operands need a non-transposing load, because the
    # transposing ldmatrix is 16-bit only on pre-SM100 parts. Veda's PV
    # gemm therefore needs V transposed before the kernel sees it.
    ld_ops = sorted(n for n in dir(warp) if 'LdMatrix' in n)
    print('ldmatrix atoms:', ', '.join(ld_ops) or 'none')
    for name in ld_ops:
        op = getattr(warp, name)
        for transpose in (False, True):
            @cute.jit
            def build(dummy: cute.Tensor, op=op, transpose=transpose):
                cute.make_copy_atom(op(transpose=transpose, num_matrices=4),
                                    cutlass.Float8E4M3FN)

            dummy = from_dlpack(torch.zeros(8, device='cuda'))
            try:
                cute.compile(build, dummy)
                print(f'  compiles: {name}(transpose={transpose}) on fp8')
            except Exception as error:
                print(f'  FAILS:    {name}(transpose={transpose}) on fp8: '
                      f'{str(error).splitlines()[0][:70]}')


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
