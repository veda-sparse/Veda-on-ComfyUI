# 注意力后端

## 目标

同一套选择结果，在每种硬件上用能跑的最快 kernel 计算；任何一个后端的问题都不能影响其他后端。

## 设计与不变量

- **唯一接口**：`base.Backend.attend(q, k, v, block_mask, layout) -> out`，输入是 tile 顺序的
  `[N, H', 128]`（padding 槽位为 0）和 `[H', n_tiles, n_tiles]` 的 bool 块掩码。
- **使用前自检**（`base.self_test`）：在真实设备上跑一个包含 reference / target / global / 部分
  tile 的小问题，和 fp32 参考比误差，并检查结果**不是**全注意力——上游 FA4 的 SM80 kernel 收到
  块稀疏参数会静默算 dense，这类问题必须在自检里暴露，而不是在用户的视频里。
- **候选顺序**（`backends/__init__.py`，第一个加载成功且自检通过的胜出，结果按设备缓存）：

  | 设备 | 顺序 |
  |---|---|
  | CUDA sm80/86/87/89 | fa4-sm80 → flex → torch |
  | CUDA sm90 | fa4-sm90 → flex → torch |
  | CUDA sm100/103/110 | fa4-sm100 → flex → torch |
  | CUDA sm120/121 | fa4-sm120 → flex → torch |
  | ROCm | flex → torch |
  | Apple MPS | mlx → torch |
  | CPU | torch |

  用户指定 backend 时，它排第一，其余按上表作为后备。
- **隔离**（AGENTS.md 1.6）：后端模块之间不互相 import；几个 FA4 后端的相似代码刻意重复。

## 各后端

- `fa4-sm80` / `fa4-sm120`：`_vendor/fa4_sm8x`（打过补丁的 FA4），`DenseBlockMaskTorch` 直接
  传块掩码；padding 槽位通过单例 `mask_mod` + `aux_tensors=[slot_valid]` 屏蔽，只有 partial
  tile 才走 mask_mod。sm120 = RTX 50 / RTX PRO Blackwell，sm121 = GB10（DGX Spark），都按
  99 KB SMEM 的 SM80 派生 kernel。
- `fa4-sm90` / `fa4-sm100`：`_vendor/fa4_upstream`（未改动的 FA4），full / partial 索引列表；
  sm100 上 FA4 在 seqlen > 128 时选 q_stage=2（稀疏 Q 块变 256），对我们的调用强制 q_stage=1
  （私有副本上的线程局部开关，沿用 Miowtion，待 B200 验证）。
- `flex`：torch FlexAttention，BlockMask 直接由块掩码构造（full / partial 两组列表），需要
  CUDA + Triton（Windows 需 `triton-windows`）；编译后的函数放宽 dynamo 重编译上限，否则超过
  上限会退回不编译的实现并物化整个分数矩阵。
- `torch`：按 query tile gather 选中的 key tile 后调 SDPA（带 padding mask），任何设备可用。
- `mlx`：同样的 gather 算法跑在 MLX 上（Apple silicon），q/k/v 在统一内存里拷贝交接。

## 测试

- `tests/unit/test_settings_hardware_backends.py`：各 SM / 子型号的候选顺序、自检能识别"忽略
  掩码"和"结果错误"、torch 后端 CPU 自检、MPS 上 mlx / torch 自检。
- `tests/unit/test_vendor_fa4_backends.py`：FA4 后端在没有 kernel 时给出安装提示、拒绝其他架构。
- `tests/gpu/test_gpu_backends.py`：每个在本机可加载的后端在 6000 token 问题上对齐参考实现。
- `tools/bench_attention.py`：H3 真实形状的单层注意力测速（全注意力 vs Veda）。

## 踩坑记录

（暂无）

## 验证记录

见 [../hardware.md](../hardware.md)。
