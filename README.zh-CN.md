<div align="center">

<img src="https://raw.githubusercontent.com/veda-sparse/Veda-on-ComfyUI/main/assets/icon.svg" width="120" alt="Veda logo">

# Veda 稀疏注意力 ComfyUI 节点（MiniMax-H3）

</div>

[论文](https://arxiv.org/abs/2605.30325) · [项目主页](https://veda-sparse.github.io/) · [打分器](https://huggingface.co/Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview) · [训练代码](https://github.com/veda-sparse/Miowtion) · [English](README.md)

## 简介

**Veda** 是面向视频扩散模型的学习型稀疏注意力方法。长视频的开销主要在注意力上，而一个蒸馏出来的
小打分器可以提前判断注意力图里哪些 tile 决定了结果。Veda 只算其中约前 10%，其余跳过。

本仓库把 Veda 封装成 ComfyUI 节点，作用于 ComfyUI 原生的 MiniMax-H3（T2VA、FL2VA、R2VA）：
一个节点，MODEL 进，MODEL 出。

### 特点

- **更快**：相对 ComfyUI 默认注意力，端到端 2.9 倍，只看注意力 7.1 倍（RTX 5070 实测）。
  视频越长收益越大。
- **即插即用**：它是 attention override，不改权重，所以 Turbo / 风格 LoRA、微调或量化的 H3
  权重、首尾帧与参考条件、其他 block 补丁都照常工作。
- **默认安全**：每个 kernel 在使用前先在你的显卡上自检。Veda 处理不了的情况交还给模型自己的
  注意力，并在节点上说明原因，而不是给你一张坏图。
- **不用额外安装**：稀疏 kernel 跟节点一起发布，从 Comfy Registry 安装时会自动带上
  Triton（NVIDIA，SM80 及以上）或 MLX（Apple silicon）。

## 安装

需要 ComfyUI >= 0.38.0。在 ComfyUI Manager 里搜索 "Veda"，或者：

```bash
comfy node install veda-sparse-attention
```

也可以把本仓库 clone 到 `ComfyUI/custom_nodes/` 后重启 ComfyUI。

## 使用

打开 *工作流 -> 浏览模板 -> Veda-on-ComfyUI*，选 "Veda MiniMax H3 T2VA" 或
"Veda MiniMax H3 R2VA"，缺失模型对话框里可以直接下载打分器。写提示词，运行。

在已有的 H3 工作流里加 Veda：把 **Veda Sparse Attention (MiniMax H3)** 接在 MODEL 线上，
位置在模型与 LoRA 加载之后、guider / 采样器之前。选中节点按 **Ctrl+B** 旁路，就能用同一个
seed 跑全注意力做对比。

不要在 H3 上和 ComfyUI 自带的 "Model Sparse Attention" 同时使用：那个节点直接替换注意力
block，Veda 根本不会被调用。节点检测到两者同时存在时会提示。

### 打分器

节点自己不下载任何东西。打分器（275 MB）进入 `models/veda` 有两种方式：

* **用模板**：打开任一 Veda 模板，ComfyUI 会在缺失模型对话框里给出打分器，点一下就下好。
* **手动放**：从[打分器仓库](https://huggingface.co/Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview)
  下载 `minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors`，放进
  `ComfyUI/models/veda/`，重启 ComfyUI。

该目录下任何 `.safetensors` 都会出现在 `predictor` 列表里，所以自己训练的打分器也是一样选。
选中的文件不在时，节点会说明并打印下载地址，而不会自己去联网。

### 节点上的文字

```
Veda done · Triton INT8 (SM120)
Video: 1344x768 · 5.2 s
Attention computed: 10.9% of full attention (89.1% skipped)
```

采样前显示将要使用的 kernel 和稀疏度；采样中显示视频尺寸和匹配到的训练方案；结束时显示实际
算了全注意力的百分之多少。`Veda off` 开头的行写着 Veda 没有生效的原因；「Tile plan」一行出现
`nearest trained size: ...` 表示这次的尺寸不在训练集里。打开 `verbose` 会额外显示各阶段耗时、
调用次数和打分器信息。

### 选项

默认只显示 `model` 和 `predictor`，其余都是高级输入（点"显示高级输入"），默认值即训练值。

| 输入 | 默认 | 含义 |
|---|---|---|
| `generated_sparsity` | `90%` | 生成视频注意力的稀疏度。`90%` 表示跳过每个 query tile 可见的 90% key tile，这是训练值；调低更接近全注意力，也更慢。填整数如 `24` 则固定保留那么多个 128-token 的 key tile。 |
| `reference_sparsity` | `90%` | 参考部分同上：首尾帧、引导帧、参考图和参考视频。`0%` 表示参考走全注意力。 |
| `full_attention_layers` | 空 | 保持全注意力的 DiT block，0 起，例如 `0, 1, 47-49`。 |
| `full_attention_steps` | 空 | 保持全注意力的采样步，0 起，例如 `0`。 |
| `verbose` | 关 | 每次运行后在节点上额外显示各阶段注意力耗时、调用次数和打分器信息。 |

发布的打分器训练于 1344x768、768x1344、768x768、1024x768，时长 5 / 10 / 14 秒，配 8 步
Turbo LoRA。其他尺寸退回到纵横比、其次时长最接近的方案。其他尺寸、其他步数以及
R2VA / FL2VA 的参考都能用，但不在训练分布内，建议和全注意力对比确认。

## 性能

T2VA，1344x768，124 帧（5.2 秒），8 步 Turbo LoRA，90% 稀疏，RTX 5070 12 GB + Windows 11。

| 注意力 | 每步 | 8 步总计 | 其中注意力 |
|---|---|---|---|
| ComfyUI 默认 | 40.7 s | 342 s | 31.1 s |
| ComfyUI `--use-sage-attention` | 24.8 s | 231 s | 15.2 s |
| Veda sparse INT8 | **14.0 s** | **130 s** | **4.41 s** |

单层注意力在 104k token、90% 稀疏下耗时 528 ms，全注意力是 11.8 s。Veda 的 INT8 算术就是
ComfyUI 在 `--use-sage-attention` 后面跑的那套代码，所以画质和那条路线一致，只是多了稀疏。
每步剩下的时间是 MLP 和权重搬运，不在注意力上。

## 硬件

一个 Triton kernel 覆盖 SM80 起的所有 NVIDIA 显卡，Windows 和 Linux 都一样。更老的卡、ROCm
和 CPU 没有 kernel：节点会说明，模型跑自己的注意力。

| 硬件 | 状态 |
|---|---|
| RTX 30 / A100 / RTX 40 / L40（sm80–89） | 代码路径就绪，尚未在该硬件上验证 |
| H100 / H200（sm90） | 代码路径就绪，尚未在该硬件上验证 |
| B200 / B300（sm100 / sm103） | 代码路径就绪，尚未在该硬件上验证 |
| RTX 50、RTX PRO 6000 Blackwell（sm120） | **已验证：RTX 5070 + Windows 11** |
| DGX Spark / GB10（sm121） | 代码路径就绪，尚未在该硬件上验证 |
| Apple silicon（M 系列） | 已验证：M3 Pro + macOS 15 |

分架构的说明和完整实测记录见
[docs/hardware.md](https://github.com/veda-sparse/Veda-on-ComfyUI/blob/main/docs/hardware.md)。

## 许可

代码 MIT。INT8 kernel 的算术取自 SageAttention v1（BSD-3-Clause）。打分器沿用其基础模型的
MiniMax H3 Community License。详见
[NOTICE.md](https://github.com/veda-sparse/Veda-on-ComfyUI/blob/main/NOTICE.md)。

## 引用

```bibtex
@inproceedings{han2026veda,
  title={Veda: Scalable Video Diffusion via Distilled Sparse Attention},
  author={Han, Shihao and Yang, Hao and Hu, Xinting and Mei, Xiaofeng
          and Jiang, Yi and Qi, Xiaojuan},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```
