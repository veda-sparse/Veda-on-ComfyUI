# 注意力后端

## 目标

同一套选择结果，在每种硬件上用一个说得清的 kernel 计算；加一种硬件不能动到别的硬件。

## 设计与不变量

- **唯一接口**：`base.Backend.attend(q, k, v, block_mask, layout) -> out`，输入是 tile 顺序的
  `[N, H', 128]`（padding 槽位为 0）和 `[H', n_tiles, n_tiles]` 的 bool 块掩码。
- **使用前自检**（`base.self_test`）：在真实设备上跑一个包含 reference / target / global / 部分
  tile 的小问题，和 fp32 参考比误差，并检查结果**不是**全注意力——曾经有一版 FlashAttention-4
  收到块稀疏参数会静默算 dense，这类问题必须在自检里暴露，而不是在用户的视频里。
- **容差属于后端**（`Backend.tolerance`，默认 2%）：量化 kernel 的误差下限由格式决定，用一个
  固定常数会把正确的实现判死。INT8 声明 5%，理由见
  [int8_kernel.md](int8_kernel.md)。
- **一个设备一个 kernel，没有回退链，用户也不选**：

  | 设备 | kernel |
  |---|---|
  | CUDA SM80 及以上 | `triton-int8` |
  | ROCm gfx11 / gfx12（RDNA3 / RDNA4） | `triton-int8` |
  | Apple MPS | `mlx` |
  | 其他（CPU、其他 ROCm 架构、SM75 及以下） | 无（节点让模型跑自己的注意力） |

  曾经是五个后端加两层回退。删掉的理由在 int8_kernel.md 里：回退链会静默把用户换到更慢或更不准
  的 kernel 上，而他们以为自己在跑原来那个。
- **隔离**（AGENTS.md 1.6）：后端模块之间不互相 import，各自只依赖 `base`、torch 和自己的 kernel 包。

## 各后端

- `triton-int8`：Triton 写的 INT8 块稀疏注意力，算术取自 SageAttention v1（即 ComfyUI
  `--use-sage-attention` 背后那一份）。覆盖 SM80 起的所有 CUDA 卡和 ROCm 上的 RDNA3 /
  RDNA4。详见
  [int8_kernel.md](int8_kernel.md)。
- `mlx`：按 query tile gather 选中的 key tile 后做注意力，跑在 MLX 上（Apple silicon），
  q/k/v 在统一内存里拷贝交接。

## 代码位置与接口

- `veda_comfy/backends/base.py`：契约、`self_test`、自检用例。
- `veda_comfy/backends/__init__.py`：注册表、`candidates`、`resolve`、`probe`。
- `veda_comfy/backends/triton_int8.py`、`mlx_gather.py`：两个实现。

## 测试

- `tests/unit/test_settings_hardware_backends.py`：各 SM / 子型号的候选、自检能识别"忽略掩码"
  和"结果错误"、CPU 上用测试专用的精确后端跑通自检、MPS 上 mlx 自检。
- `tests/unit/reference_backend.py`：那个测试专用后端。它刻意**不在** `veda_comfy.backends` 里
  ——O(N²)，不该给任何人用——但没有它，CPU CI 对 `Backend` 契约的覆盖就是零。
- `tests/gpu/test_gpu_backends.py`：本机 kernel 在 6000 token 问题上对齐参考实现，并对齐
  ComfyUI 的 INT8 算术。
- `tools/bench_attention.py`：H3 真实形状的单层注意力测速（全注意力 vs Veda）。

## 踩坑记录

- **回退链会骗人**。调 FP8 的那几天里，`bench_attention --backend int8` 一直打印的是 FA4 的数字，
  因为 INT8 自检没过就静默回落了，而摘要行在很靠上的位置。现在没有回退，装不上就明说。

## 验证记录

见 [../hardware.md](../hardware.md)。
