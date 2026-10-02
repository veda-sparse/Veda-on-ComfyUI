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
| RTX 50（5090 等） | sm120 | fa4-sm120 | ✅（RTX 5070） | 🧪 | Windows 端到端验证见下 |
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
- 2026-10-03，**RTX 5070 12 GB（SM120），Windows 11，torch 2.14.1+cu130，ComfyUI 0.38.0**，
  CuTe DSL 4.8.0（Windows wheel），vendored FA4 4.0.0b32 + QuACK 0.6.5：
  - `tests/unit` 79 passed / 1 skipped（MLX）；`install_fa4.bat` 安装后自检选中 fa4-sm120；
    `tests/gpu` 3 passed / 2 skipped（MLX；flex 缺 Triton）。FA4 首次编译约 6 s。
  - `tools/bench_attention.py`（单层，随机打分器，90%）：16:9 5.2 s（38k token）SDPA 1577 ms、
    Veda fa4-sm120 137 ms（11.5x）；16:9 10.1 s（74k）5930 → 412 ms（14.4x）；14.4 s（104k）
    11838 → 878 ms（13.5x）。torch 后端 5.2 s：716 ms（2.2x）。注意 Windows 版 torch 的 SDPA
    走不了 flash，这个基线偏慢。分阶段（5.2 s）：kernel 77%、gather 10%、打分 8%、scatter 3%、
    选块 1%。
  - `tools/e2e_minimax_h3.py`（真实 ComfyUI、真实权重：FL2VA int8 + 8 步 Turbo LoRA，
    1344x768，124 帧，seed 12/13）：全注意力（ComfyUI 默认，该权重为 int8 attention）39.8 s/步、
    整个 prompt 335 s；Veda 16.1 s/步（2.47x）、prompt 152–160 s（2.1–2.2x，含模型加载）。Veda
    注意力 6.7 s/步（134 ms/层，其中 kernel 93 ms），"Attention computed: 12.8% of full
    attention"；剩下约 9.4 s/步是 MLP 与 12 GB 显存下的权重搬运。打分器由节点自动下载。
  - R2VA（Ref2VA int8 + 4 步 Turbo LoRA，1 张参考图 512x512，1344x768，124 帧，seed 21）：
    全注意力 39.7 s/步、prompt 185 s；Veda 18.0 s/步（2.2x）、prompt 105 s（含模型加载）；
    reference 段按 tile 稀疏（"Attention computed: 16.4%"，video tile 保留 8.0%）。打分器只在
    FL2VA 上训练过，R2VA 的画质需要人工对比确认。
  - 过程中修掉的 Windows / ComfyUI 问题见 pitfalls.md（QuACK 的 fcntl 与 Triton、malloc graph
    abort、安装后同进程 import 缓存、cp1252 控制台）。生成的视频待人工对比确认。

