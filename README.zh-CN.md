# Veda 稀疏注意力 ComfyUI 节点（MiniMax-H3）

[English](README.md)

**让 MiniMax-H3 在 ComfyUI 里生成音视频更快，工作流、LoRA、权重都照常使用。**

[Veda](https://arxiv.org/abs/2605.30325)（ICML 2026）用一个小的打分器找出真正重要的约 10% 注意力，
其余跳过。本节点把它接进 ComfyUI 原生的 MiniMax-H3（T2VA / FL2VA / R2VA），只是一个节点：
**MODEL 进，MODEL 出**。

* **长视频收益最大**：长片子的大部分时间花在注意力上，Veda 在 10–14 秒的片子上端到端约 2–3 倍
  （越长越快，短片子收益小）。
* **其他都不变**：不改权重，Turbo / 风格 LoRA、微调或量化的 H3 权重、首尾帧和参考条件、其他注意力或
  block 补丁都照常工作。
* **默认安全**：每个 kernel 在你的显卡上先自检再使用；Veda 处理不了的情况自动走普通注意力，并在节点上
  说明原因，不会出坏图。

## 快速开始

1. **安装**：ComfyUI Manager 搜索 "Veda"，或 `comfy node install veda-sparse-attention`，或把本仓库
   clone 到 `ComfyUI/custom_nodes/`，重启 ComfyUI。需要 ComfyUI >= 0.38.0。
2. **打开模板**：*工作流 -> 浏览模板 -> Veda-on-ComfyUI*，选 "Veda MiniMax H3 T2VA" 或
   "Veda MiniMax H3 R2VA"，弹出的缺失模型对话框里可以一键下载。
3. **写提示词，运行**。第一次会下载打分器（约 275 MB，存到 `models/veda`）并编译一次 kernel。

在自己的 H3 工作流里使用：把 **Veda Sparse Attention (MiniMax H3)** 接在模型加载和 LoRA 之后、
guider / 采样器之前。对比效果：选中节点按 **Ctrl+B**（旁路），同一个 seed 就是全注意力。

### 最快的 kernel（可选，只需一次）

| 机器 | 在 `custom_nodes/Veda-on-ComfyUI` 里运行 | Kernel |
|---|---|---|
| NVIDIA + Windows（便携版 / 桌面版） | 双击 `install_fa4.bat` | FlashAttention-4 |
| NVIDIA + Linux（含 DGX Spark） | `./install_fa4.sh` | FlashAttention-4 |
| Apple silicon | `./install_fa4.sh`（安装 MLX） | MLX |

不装也能用（FlexAttention 或 torch 的通用 kernel）。脚本不会改动你的 torch，最后会自检并打印每张卡
将使用的 kernel。

## 节点上显示什么

```
✅ Veda done · FA4 (SM120)
Video: 1344x768 · 5.2 s
Attention computed: 10.9% of full attention (89.1% skipped)
```

采样前显示将用的 kernel 和稀疏度；采样中显示视频尺寸和匹配到的训练方案；运行结束显示实际算了多少比例的
全注意力。⚠ 开头的行表示回退了或者超出了训练范围（少见的尺寸、缺 kernel 等），后面写着原因。打开
`verbose` 会额外显示各阶段耗时、注意力调用次数和打分器信息。

## 设置

默认只显示 `model` 和 `predictor`，其余都是高级输入（点 "显示高级输入"），默认值就是训练值：

| 输入 | 默认 | 含义 |
|---|---|---|
| `generated_sparsity` | `90%` | 生成视频（generated）注意力的稀疏度：`90%` 表示跳过 90% 的 key tile（训练值）；填整数如 `24` 表示每个 query tile 固定保留 24 个 128-token 的 key tile。 |
| `reference_sparsity` | `90%` | 参考（reference）同上：首尾帧、引导帧、参考图和参考视频。`0%` 表示参考走全注意力。 |
| `full_attention_layers` | 空 | 保持全注意力的 DiT 层，0 起，例如 `0, 1, 47-49`。 |
| `full_attention_steps` | 空 | 保持全注意力的采样步，0 起，例如 `0`。 |
| `backend` | auto | `fa4` / `flex` / `torch` / `mlx`；auto 选自检通过的最快那个。 |
| `verbose` | 关 | 每次运行后在节点上额外显示各阶段耗时、调用次数和打分器信息。 |

发布的打分器训练于 **1344x768、768x1344、768x768、1024x768，5 / 10 / 14 秒**，配 8 步 Turbo LoRA。
其他尺寸按纵横比、再按时长匹配最接近的训练方案；其他尺寸、步数以及 R2VA / FL2VA 的参考都能用，但不在训练
分布内，建议和全注意力（旁路）对比确认。

## 常见问题

* **会改变 LoRA 风格吗？** 不会，Veda 只决定算哪些注意力块，权重（和 LoRA）不动。
* **能和其他注意力节点一起用吗？** Veda 不处理的调用会交给之前设置的注意力实现。不要在 H3 上和 ComfyUI
  自带的 "Model Sparse Attention" 同时使用（节点会提示）。
* **国内网络？** 启动 ComfyUI 前设置 `HF_ENDPOINT=https://hf-mirror.com`，或手动下载打分器放到
  `models/veda`。

许可：代码 MIT；附带的 FlashAttention-4 为 BSD-3-Clause；打分器沿用 MiniMax H3 Community License。
见 [NOTICE.md](NOTICE.md)。
