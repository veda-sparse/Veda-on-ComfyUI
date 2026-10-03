<div align="center">

<img src="assets/icon.svg" width="120" alt="Veda logo">

# Veda Sparse Attention for ComfyUI (MiniMax-H3)

**Faster MiniMax-H3 video + audio generation in ComfyUI, with your usual
workflow, LoRAs and checkpoints.** [中文说明](README.zh-CN.md)

</div>

[Veda](https://arxiv.org/abs/2605.30325) (ICML 2026) trains a small
predictor that finds the ~10% of attention that matters and skips the rest.
This custom node plugs it into ComfyUI's native MiniMax-H3 (T2VA, FL2VA and
R2VA) as one node: **MODEL in, MODEL out**.

* **Faster where it hurts:** long clips spend most of their time in
  attention; Veda reports 2-3x end to end on 10-14 s clips (more on longer,
  less on short ones).
* **Nothing else changes:** weights are untouched, so Turbo / style LoRAs,
  fine-tuned or quantized H3 checkpoints, first/last-frame and reference
  conditioning, and other attention or block patches keep working.
* **Safe by default:** every kernel passes a self-test on your GPU before
  it is used; anything Veda cannot handle runs normal attention and says so
  on the node, never a broken render.

## Quick start

1. **Install** with ComfyUI Manager (search "Veda") or
   `comfy node install veda-sparse-attention`, or clone this repo into
   `ComfyUI/custom_nodes/`. Restart ComfyUI. Needs ComfyUI >= 0.38.0.
2. **Open a template:** *Workflow -> Browse Templates -> Veda-on-ComfyUI*:
   "Veda MiniMax H3 T2VA" or "Veda MiniMax H3 R2VA". Missing models can be
   downloaded from the dialog that pops up.
3. **Write your prompt and run.** The first run downloads the predictor
   (~275 MB, into `models/veda`) and compiles the kernels once.

To add Veda to your own H3 workflow: put **Veda Sparse Attention (MiniMax
H3)** after the model loader and LoRA loaders, right before the guider /
sampler. To compare, select it and press **Ctrl+B** (bypass): same seed,
full attention.

### Kernels

Nothing to install. The sparse kernel ships with the node: installing from
the Comfy Registry pulls in Triton (NVIDIA, SM80 and newer) or MLX (Apple
silicon) automatically.

It is the same INT8 arithmetic ComfyUI itself uses for low-precision
attention (`--use-sage-attention`), so the quality is what you would get
there, with Veda's sparsity on top. On an RTX 5070 at 104k tokens and 90%
sparsity, one attention layer takes 528 ms against 11.8 s for full
attention.

## What the node shows

The text on the node tells you what is happening, e.g.

```
Veda done · Triton INT8 (SM120)
Video: 1344x768 · 5.2 s
Attention computed: 10.9% of full attention (89.1% skipped)
```

Before sampling it shows the kernel it will use and the sparsity; while
sampling, the video size and which trained tile plan it matched. A line
starting `Veda off` means there is no kernel on this GPU or the layout could
not be read, with the reason; the tile plan line reads `nearest trained
size: ...` when the video is outside what the predictor was trained on. Turn
on `verbose` for timing per phase, attention calls and predictor details.

## Settings

Only `model` and `predictor` are visible; everything else is an advanced
input (click "show advanced inputs") with the trained defaults:

| Input | Default | Meaning |
|---|---|---|
| `generated_sparsity` | `90%` | Sparsity of the generated video's attention: `90%` skips 90% of the key tiles (the trained value). A whole number such as `24` keeps exactly that many 128-token key tiles instead. |
| `reference_sparsity` | `90%` | The same for references: first/last frames, guide frames, reference images and videos. `0%` = references use full attention. |
| `full_attention_layers` | empty | 0-based DiT blocks that keep full attention, e.g. `0, 1, 47-49`. |
| `full_attention_steps` | empty | 0-based sampling steps that keep full attention, e.g. `0`. |
| `verbose` | off | After each run, also show attention time per phase, call counts and predictor details on the node. |

The released predictor was trained for **1344x768, 768x1344, 768x768 and
1024x768 at 5 / 10 / 14 s** with the 8-step Turbo LoRA. Other sizes use the
tile plan of the nearest aspect ratio and duration; other sizes, step
counts and R2VA / FL2VA references work but are outside its training data,
so check the result against full attention (bypass).

## Hardware

| Hardware | Status |
|---|---|
| RTX 30 / A100 / RTX 40 / L40 (sm80-89) | see [docs/hardware.md](docs/hardware.md) |
| H100 / H200 (sm90) | see [docs/hardware.md](docs/hardware.md) |
| B200 / B300 (sm100 / sm103) | see [docs/hardware.md](docs/hardware.md) |
| RTX 50, RTX PRO 6000 Blackwell (sm120) | **tested: RTX 5070, Windows 11** (2.5x per sampling step, 5 s 16:9 T2VA) |
| DGX Spark / GB10 (sm121) | see [docs/hardware.md](docs/hardware.md) |
| Apple silicon (M series) | tested (M3 Pro) |

One Triton kernel covers every NVIDIA GPU from SM80 on, Windows and Linux
alike. Older cards, ROCm and CPU have no kernel: the node says so and the
model runs its own attention.

## FAQ

* **Does it change my LoRA's style?** No. Veda only decides which attention
  blocks to compute; the model weights (and your LoRAs) are untouched.
* **Can I use it with other attention nodes?** Calls Veda declines go to
  whatever attention override was set before it. Do not combine it with
  ComfyUI's own "Model Sparse Attention" node on H3 (the node warns).
* **Behind a firewall / in mainland China?** Set `HF_ENDPOINT` (for example
  `https://hf-mirror.com`) before starting ComfyUI, or download the
  predictor by hand into `models/veda`.
* **Batch size?** MiniMax-H3 itself runs batch size 1.

## Links and license

* Paper: [Veda: Scalable Video Diffusion via Distilled Sparse Attention](https://arxiv.org/abs/2605.30325) · project page: <https://veda-sparse.github.io/>
* Predictor: [Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview](https://huggingface.co/Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview) (MiniMax H3 Community License)
* Training code: [veda-sparse/Miowtion](https://github.com/veda-sparse/Miowtion)

Code: MIT. Bundled FlashAttention-4: BSD-3-Clause. See [NOTICE.md](NOTICE.md).
