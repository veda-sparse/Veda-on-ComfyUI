# 硬件支持矩阵与验证记录

状态：✅ 在该硬件上跑过 `tests/gpu` 和测速；🧪 代码路径就绪、自检把关、尚未在该硬件上验证；
— 不适用。

| 硬件 | SM | 首选后端 | Windows | Linux | 备注 |
|---|---|---|---|---|---|
| RTX 30（3090 等）、RTX A 系列 | sm86 | fa4-sm80 | 🧪 | 🧪 | Miowtion 的补丁声明覆盖 sm80/86/89，只在 4090 上验证过 |
| A100 | sm80 | fa4-sm80 | — | 🧪 | |
| RTX 40（4090 等）、L4 / L40、RTX 6000 Ada | sm89 | fa4-sm80 | 🧪 | 🧪 | Miowtion 在 4090 / Linux 上验证过同一份补丁 |
| H100 / H200 | sm90 | fa4-sm90 | — | 🧪 | 上游 FA4 原生块稀疏 |
| B200 / GB200、B300 | sm100 / sm103 | fa4-sm100 | — | 🧪 | q_stage 强制为 1，待验证 |
| RTX 50（5090 等） | sm120 | fa4-sm120 | 🧪 | 🧪 | |
| RTX PRO 6000 Blackwell | sm120 | fa4-sm120 | 🧪 | 🧪 | Miowtion 在 Linux 上验证过 SM120 前向 |
| DGX Spark（GB10） | sm121 | fa4-sm120 | — | 🧪 | aarch64 + CUDA 13：install_fa4 装 cu13 的 CuTe DSL |
| 其他 NVIDIA（无 FA4 时） | — | flex → torch | 🧪 | 🧪 | flex 需要 Triton（Windows：triton-windows） |
| Apple silicon（M 系列） | — | mlx → torch | — | — | macOS ✅（M3 Pro） |

## 验证记录

- 2026-10-03，Apple M3 Pro（macOS 15），torch 2.14.1 + MLX 0.32.3：mlx / torch 后端在 CPU 与 MPS
  自检通过，`tests/gpu` 3 passed / 2 skipped（无 CUDA）。`tools/bench_attention.py`（16:9，
  latent_t 37，38 228 token，90% 稀疏，随机打分器，单层）：MPS SDPA 全注意力 9.29 s；mlx 后端
  3.52 s（2.64x）；torch 后端 5.68 s（1.57x，与另一次 8.89 s 的全注意力相比）。kernel 级优化尚未
  开始。
