# 硬件支持矩阵与验证记录

状态：✅ 在该硬件上跑过 `tests/gpu` 和测速；🧪 代码路径就绪、自检把关、尚未在该硬件上验证；
— 不适用。

| 硬件 | SM | kernel | Windows | Linux | 备注 |
|---|---|---|---|---|---|
| RTX 30（3090 等）、RTX A 系列 | sm86 | triton-int8 | 🧪 | 🧪 | 静态审查过，见 int8_kernel.md；3090 上有用户走到了自检，卡在 Triton 自己的编译环境上（见下） |
| A100 | sm80 | triton-int8 | — | 🧪 | |
| RTX 40（4090 等）、L4 / L40、RTX 6000 Ada | sm89 | triton-int8 | 🧪 | 🧪 | 静态审查过，见 int8_kernel.md |
| H100 / H200 | sm90 | triton-int8 | — | 🧪 | TMA 可用（有 cluster），尚未利用 |
| B200 / GB200、B300 | sm100 / sm103 | triton-int8 | — | 🧪 | 同上 |
| RTX 50（5090 等） | sm120 | triton-int8 | ✅（RTX 5070） | 🧪 | TMA 只能 `.shared::cta`，`num_ctas` 必须为 1 |
| RTX PRO 6000 Blackwell | sm120 | triton-int8 | 🧪 | 🧪 | |
| DGX Spark（GB10） | sm121 | triton-int8 | — | 🧪 | aarch64 + CUDA 13 |
| SM75 及以下 / ROCm / CPU | — | 无 | — | — | 节点让模型跑自己的注意力 |
| Apple silicon（M 系列） | — | mlx | — | — | macOS ✅（M3 Pro） |

一个 kernel 覆盖 SM80 起的所有 CUDA 卡，所以这张表现在是"验证到哪了"，不是"用哪个实现"。

## 验证记录

- 2026-10-03，**sm86（3090）/ sm89（4090）：只做了静态审查，没有上机**，所以上表仍是 🧪。
  这条通路没有按架构分支（门槛只有 `cc >= (8, 0)`），TMA 默认关闭所以 SM90+ 的代码到不了，
  用到的 MMA 都在 SM80 的指令集里，共享内存峰值约 88 KiB 而这两代的每 SM 预算是 100 KiB——
  和已经验证过的 sm120 相同。详细推理与"还剩什么只能靠硬件回答"见
  [features/int8_kernel.md](features/int8_kernel.md)。另外注意 Veda 论文和打分器 model card
  里的 4090 数字来自 Miowtion 训练栈的 kernel，不是这里的 Triton INT8 kernel，不能当作本仓库
  在 sm89 上的实测。
- 2026-10-03，**RTX 5070 12 GB（SM120），Windows 11，torch 2.14.1+cu130，triton-windows
  3.8.0**，`triton-int8`：
  - `tests/gpu` 通过。`tools/compare_int8.py`（4096 token × 8 头 × 128，稠密）：对 fp32 参考
    **1.344%**，与装机版 SageAttention 的 1.344% 相同，ComfyUI 的 INT8 模型 1.342%，
    torch SDPA bf16 0.246%。带 padding 的自检用例上，与自身算术差 0.133%。
  - `tools/bench_attention.py`（16:9，latent_t 102，104 484 token，90% 稀疏，随机打分器，单层）：
    SDPA 全注意力 11 834 ms，Veda **528.2 ms/层（22.40x）**。同一问题上被它取代的 FA4 CuTe bf16
    是 793.8 ms（14.91x），即 INT8 kernel 快 1.50 倍。
  - 稠密吞吐参照：SageAttention INT8 125 TFLOPS，torch SDPA bf16 26 TFLOPS（Windows 版 torch
    的 SDPA 走不了 flash，这个基线偏慢）。
  - launch 配置由 `tools/tune_int8.py` 在 119k slot 的真实问题上扫出来：4 warps / 3 stages
    157.4 ms，上游给 head_dim 128 选的 8 warps 是 178.3 ms。**TMA 更慢**：即使把 K 量化成
    `[H, D, slots]` 让循环里没有转置，也要 186.2 ms。合理——TMA 在 Hopper / 数据中心 Blackwell
    上的主要收益是把一个 key tile 多播给 cluster 里的多个 CTA，而消费级 Blackwell 有 TMA 没有
    cluster，8 KB 的块摊不掉描述符开销。TMA 代码保留在 `USE_TMA` 后面，等有 SM90/SM100 机器
    再扫。
  - **端到端**（`tools/e2e_minimax_h3.py`，真实 ComfyUI + 真实权重：T2VA FL2VA int8 +
    8 步 Turbo LoRA，1344x768，124 帧 = 5.2 秒，8 步，seed 42）：

    | 配置 | 每步 | 总计 | 其中注意力 |
    |---|---|---|---|
    | 稠密，ComfyUI 默认注意力 | 40.7 s | 342.1 s | ~31.1 s |
    | 稠密，ComfyUI `--use-sage-attention` | 24.8 s | 230.9 s | ~15.2 s |
    | Veda sparse INT8 | **14.0 s** | **130.1 s** | **4.41 s** |

    即：SageAttention 的量化本身 1.64x，我们的稀疏在其之上再 1.77x（对默认路线合计 2.91x）；
    只看注意力是 31.1 → 15.2 → 4.41。Veda 的 4.41 s/步里 kernel 2.95、gather 0.66、打分 0.46、
    scatter 0.23、选块 0.10。这一步剩下的 9.6 s 是 MLP 与 12 GB 显存下的权重搬运，占 69%——
    注意力侧的全部开销（1.46 s）清零也只有 1.11x，所以优化重心不在这里。
- 2026-10-06，RTX 3090（SM86），Windows portable ComfyUI，Python 3.13，CUDA 13.0 —
  **用户报告，不是我们跑的测试**：节点显示 "Veda off: no sparse kernel works on RTX 3090
  (SM86)"，下面跟着 `triton-int8 failed its self-test: CalledProcessError: ... tcc.exe ...
  cuda_utils.c ... exit status 1`。**和 SM86 无关**：候选表对 SM80 起的卡只有
  `triton-int8` 一个，3090 已经通过了 `_MIN_CC` 并走到自检，失败的是 Triton 第一次调用时用
  C 编译器构建 `cuda_utils.c` 这个辅助模块——portable 版的 `python_embeded` 不带头文件和
  导入库，tcc 找不到 `Python.h` / `python313.lib` 就以 1 退出，而它自己的输出被
  `CalledProcessError` 吞掉了（triton-windows 的 issue 83 / 156 / 186 是同一条）。对策：把
  对应版本 python.org 构建里的 `Include\` 和 `libs\` 复制进 `python_embeded\`，删掉
  `%USERPROFILE%\.triton\cache` 再重启。节点现在会这么说，而不是去怪显卡。
- 2026-10-06，**RTX 5070 12 GB（SM120），Windows 11，torch 2.14.1+cu130，triton-windows
  3.8.0，SageAttention 2.2.0+cu130，ComfyUI 0.38.0 + KJNodes 1.5.2**：与其他注意力节点的
  交叉验证，以及显存开销的定位（`tools/e2e_minimax_h3.py`，T2VA FL2VA int8 + 8 步 Turbo
  LoRA，1344x768，124 帧 = 5.2 秒，2 步，seed 42；**每个配置前重启 ComfyUI**，否则峰值会被
  上一次运行的残留影响——同一组参数顺序测会得出 1.61 GB，单独测是 3.24 GB）。

  | 配置 | 每步 | 峰值显存 | 每层分块 |
  |---|---|---|---|
  | 全注意力 | 39.3 s | 0.64 GB | - |
  | Low VRAM 节点（head_chunks=4） | 39.6 s | 0.64 GB | - |
  | Veda（本次修复前） | 13.6 s | 3.24 GB | 6 x 12 head |
  | Veda | 13.6 s | **1.31 GB** | 8 x 8 head |
  | Low VRAM + Veda | 13.6 s | 1.31 GB | 8 x 8 head |
  | Low VRAM（head_chunks=14）+ Veda | 14.3 s | 1.03 GB | 15 x 4 head |
  | Mem Eff Sage 节点 | 24.2 s | - | - |
  | Mem Eff Sage + Veda（被 Veda 接管） | 13.5 s | - | - |

  - 三个上报的问题都复现并修掉了：Low VRAM 节点下 Veda 不再因为只看到 14 / 9 个 head 而关闭
    （`Attention calls: 100 sparse`）；Mem Eff Sage 在 Veda 之前时被接管（24.2 → 13.5 s/步），
    在 Veda 之后时节点给出提示而不是沉默。
  - **Low VRAM 节点单独用在 5.2 s 上看不出省显存**（0.64 GB，和全注意力一样）：它缩小的是
    kernel 的内部临时量，而这个尺寸下的峰值由 ComfyUI 的权重搬运决定。
  - **14.4 s（104k token）在 12 GB 卡上是跑不动的**：Veda 与 Low VRAM + Veda 都是 2.77 GB
    峰值、约 1400 s/步、`free 0.00 GB`，即全程在换权重。Veda 这时的 1906 MB 工作集里有
    1.49 GB 是该层的输出缓冲（104k x 56 x 128 x 2），全注意力同样要付，没有可压缩的空间。
- 2026-10-04，Apple M3 Pro（macOS），torch 2.14.1 + MLX 0.32.3：`tests/gpu` 2 passed /
  1 skipped（INT8 对照需要 CUDA），`tests/unit` 96 passed。覆盖的是把 `mx.eval` 收进
  `_flusher` 并删掉 `_to_torch` 前那次多余 flush 的改动——Metal 自检与 fp32 参考比对都过。
- 2026-10-03，Apple M3 Pro（macOS 15），torch 2.14.1 + MLX 0.32.3：mlx 后端在 MPS 自检通过。
  `tools/bench_attention.py`（16:9，latent_t 37，38 228 token，90% 稀疏，随机打分器，单层）：
  MPS SDPA 全注意力 9.29 s；mlx 后端 3.52 s（2.64x）。
- 2026-10-03，RTX 5070（SM120），Windows 11，ComfyUI 0.38.0 — **下面这组是被 `triton-int8`
  取代之前的 FA4 CuTe bf16 数据**，保留作为对照：
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

