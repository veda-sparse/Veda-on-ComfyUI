# INT8 块稀疏 kernel（Triton）

## 目标

一个 kernel 覆盖所有 CUDA 卡（SM80 起），精度等于 ComfyUI 自己的低精度注意力，速度比任何 16 位
稀疏实现都快，且用户什么都不用装、什么都不用选。

## 为什么是 INT8 而不是 FP8

这是本项目最反直觉的一条结论，当初的问题是"ComfyUI 为什么用 INT8？这很奇怪"。实测（`tools/compare_int8.py`，
RTX 5070，4096 token × 8 头 × 128）：

| 方案 | 相对 fp32 的 L2 |
| --- | --- |
| torch SDPA bf16 | 0.246% |
| **INT8 QK per-block + fp16 PV（= ComfyUI / SageAttention）** | **1.342%** |
| FA4 CuTe + FP8 e4m3 全量化 | 5.336% |

不是实现问题，是格式决定的。e4m3 只有 3 位尾数，相对误差 ~3.6%/值；INT8 配 per-block `amax/127`
是 127 个均匀档，对高斯分布的 Q/K 约 0.9%。**浮点把动态范围花在了根本不出现的量级上**。拆开看，
FP8 的 QK 一项就贡献 3.89%，per-block scale、V smoothing、从量化后的 P 取行和，全都救不回来。
所以 SageAttention 选 INT8 是对的，ComfyUI 跟着它也是对的。

既然如此为什么不在 FA4 的 CuTe fork 里做 INT8？因为做不到：CuTe DSL 4.8.0 的 warp 层只有
`MmaF16BF16Op`（只收 Float16/BFloat16）、`MmaFP8Op`、`MmaTF32Op`、`MmaUniversalOp`（只收
Float16/32/64）和 `MmaSM120BlockScaledOp`。`MmaI8Op` **只存在于 `tcgen05`**，那是 SM100 数据中心
的第五代张量核指令，消费卡用不了；`cute.arch` 也没有 inline PTX 的逃生口。MLIR dialect 里确实有
`MMAIntOverflow` 和 `MmaAtomSM80Type`（C++ CUTLASS 的 s8 MMA 在那层），但 Python 没有暴露入口。
Triton 的 `tl.dot` 直接支持 int8，所以 kernel 写在 Triton 里。

## 为什么没有 fallback

一度有五个后端：四个按 SM 家族分的 FA4 CuTe 后端、一个 FP8 变体，后面跟着 FlexAttention 和纯 torch
gather。全部删掉了，连同 61 个 vendored 模块、补丁工具和 `install_fa4`（约 6.3 万行）。

- **性能上没有保留的理由**：同一个稀疏问题（104k token，90% 稀疏，RTX 5070）Triton INT8
  **528.2 ms/层**，FA4 CuTe bf16 **793.8 ms/层**——快 1.50 倍，对稠密 SDPA 是 22.4x 对 14.9x。
- **精度上也没有**：INT8 1.34%，FP8 5.34%。
- **回退链本身是个坑**：自检失败就静默换一个更慢或更不准的 kernel，用户以为自己在跑 A，实际在跑 B。
  调 FP8 那几天里 `bench_attention --backend int8` 报的一直是 FA4 的数字，就是这么回事。现在
  一个设备一个 kernel，装不上就明说。

## 设计与不变量

- **算术与上游逐字相同**。来源是 `sageattention` 1.0.6（BSD-3，Thu-ML）的 `quant_per_block.py`
  与 `attn_qk_int8_per_block.py`，也就是 ComfyUI `--use-sage-attention` 背后那一份：Q 按 128 行、
  K 按 64 行各一个 INT8 scale；`softmax_scale · log2e` 折进 Q 的量化，于是 kernel 里用 `exp2`；
  P 和 V 走 fp16。实测我们的 kernel 与装在机器上的 SageAttention 同为 **1.344%**。
- **只走保留的 tile**。主循环遍历 per（头，query tile）的保留 key 块列表，而不是整条序列。
  列表在主机侧由 `selection.tile_index_list` 生成（对掩码做 stable 降序 argsort，保留项自然按
  tile 升序排在前面），并在 `kernels/sage.key_blocks` 里展开到 64 行粒度，这样 kernel 的循环是扁平的
  ——Triton 的流水线对扁平循环处理得最好，也为后面的 TMA 变体留好形状。
- **padding 用 `valid_count` 掩掉**。利用 tiling 的不变量"一个 tile 的真实行总是前缀"，一次比较就够。
  **上游没有等价物**：它把越界的 key 当 0 读进来，于是这些列以 `exp2(0 - m)` 的权重混进 softmax。
  对它自己只有最后一块越界的情况影响很小，对我们半块都是 padding 的 tile 就是错的。
- **`m_i` 初值是有限的 -1e30，不是 -inf**。整块都是 padding 时 `m_ij` 保持 -inf，
  `(-inf) - (-inf)` 会产出 NaN。
- **量化在 kernel 包内部做**，不改 `Backend.attend` 接口。
- **自检容差是后端的属性**。`Backend.tolerance` 默认 2%（16 位 kernel 的量级），INT8 声明 5%：
  它和自己的算术只差 0.13%，但那套算术在 padding 很多的自检用例上本身就离 fp32 有 2.7%。
  用一个固定常数会把一个正确的量化 kernel 判死。

## 现状

RTX 5070 / SM120 / Windows 11 / torch 2.14.1+cu130 / triton-windows 3.8.0：

| 指标 | 值 |
| --- | --- |
| 稀疏（104k token，90%） | 528.2 ms/层，22.40x over SDPA |
| 稠密吞吐（SageAttention 参照） | 125 TFLOPS（bf16 SDPA 26） |
| 精度 vs fp32 | 1.344%（= 装机版 SageAttention） |
| 精度 vs 自己的算术（含 padding） | 0.133% |

## TMA：实现了，但在消费卡上更慢

稀疏走法每次循环才知道下一个 key 块的地址，所以每一轮都要花发射槽和寄存器去算一个硬件本可以
按块坐标自己完成的拷贝——TMA 正是这个形状。实现在 `_attention_tma_kernel`，K 由
`quantize_transposed` 直接量化成 `[H, D, slots]`，循环里连转置都没有。

**实测在 RTX 5070 上慢 15%**（186.2 vs 157.4 ms，119k slot）。这不矛盾：TMA 在 Hopper 和数据中心
Blackwell 上的主要收益来自把一个 key tile **多播给 cluster 里的多个 CTA**，而消费级 Blackwell
有 TMA 却没有 thread block cluster——`cp.async.bulk.tensor` 只能落 `.shared::cta`，`num_ctas`
必须是 1，不能 multicast。于是它退化成"另一种发起拷贝的方式"，而 8 KB 的块太小，摊不掉描述符
的开销；`cp.async` 配 3 级流水已经把延迟藏住了。

所以 `USE_TMA = False`。代码留着：SM90 / SM100 有 cluster，很可能是赢的，但**没人在那上面跑过
`tools/tune_int8.py`**，而发布一个没测过的默认值正是在自己看不见的硬件上变慢的办法。

## 4090 / 3090（SM89 / SM86）：静态审查，未上机

2026-10-03 按代码读了一遍这条通路在 sm89 / sm86 上会不会出问题。结论：**没有发现会挡住它的
东西，但没有上机就不算验证**，`hardware.md` 里这两行仍然是 🧪。

- **没有按架构分支**。`backends.candidates()` 的门槛只有 `info.cc >= (8, 0)`，不看 SM 家族；
  kernel 里也没有 `cc` 分支。`hardware._FAMILIES` 有 `(8, 6)`/`(8, 9)`，所以节点上显示的是
  `Triton INT8 (SM86)` / `(SM89)`，`_cuda_subtype` 给出 `rtx30` / `rtx40`。
- **TMA 那条路到不了**。`_tma_available()` 第一个条件就是 `USE_TMA`，而它是 `False`，所以
  只有 SM90 起才有的 `_attention_tma_kernel` 和 `tl.make_tensor_descriptor` 都不会被碰到。
- **指令都在 SM80 的集合里**。`tl.dot` 的 int8 × int8 → int32 是 `mma.s8s8s32`（SM80 起），
  `tl.dot(p.to(fp16), v, out_dtype=fp16)` 是 `mma.f16.f16.f16`（同样 SM80 起）。
- **共享内存的预算在 SM120 上已经验证过**。循环里每一级是 K `[D=128, BLK=64]` int8 = 8 KiB
  加 V `[64, 128]` fp16 = 16 KiB，3 级最多 72 KiB，再加循环外的 Q `[128, 128]` int8 = 16 KiB，
  峰值约 88 KiB。关键在于 **sm86 / sm89 的每 SM 共享内存是 100 KiB，和 sm120 一样**
  （sm80 是 164 KiB，更宽松），而这组 4 warps / 3 stages 正是在 sm120 上实测跑通并调出来的，
  所以这不是一个未知数，是一个已经在同样预算下成立过的配置。

剩下真正只能靠硬件回答的是两件事：Triton 在 sm86 / sm89 上实际分配的共享内存会不会越过
每 block 99 KiB 的上限（越了的话是编译期 `OutOfResources`），以及 4 warps / 3 stages 在这两
代上是不是还是最优（要用 `tools/tune_int8.py` 扫）。

第一件事即使发生也不会坏图：`backends.resolve()` 把 `_load` 和 `self_test` 包在
`except BackendUnavailable` / `except Exception` 里，失败会记进 `attempts`、写日志，并在节点上
显示 `Veda off: no sparse kernel works on ...` 加原因，然后让模型跑自己的注意力。代价是没有
加速，不是坏结果。

## 待做

- 在 SM80 / SM86 / SM89 / SM90 / SM100 上回归（4090 / 3090 目前只有上面那份静态审查），
  并在有 cluster 的卡上扫一次 TMA，结果写进 hardware.md。
- kernel 对"完美线性缩放"的理想值是 66% 效率（端到端 2.95 s/步 对 1.95 s）。整条注意力路径
  只占真实采样步的 31%，所以这件事排在换更大显存之后。

## 代码位置与接口

- `veda_comfy/kernels/sage/sparse_int8.py`：量化 kernel、主 kernel、`key_blocks`、`attend`。
- `veda_comfy/backends/triton_int8.py`：`Backend` 实现与容差声明。
- `veda_comfy/core/selection.py`：`tile_index_list`。
- `veda_comfy/backends/__init__.py`：注册表（一个设备一个 kernel）。
- `tools/compare_int8.py`：对 fp32、SDPA、ComfyUI 的 INT8（真包 + torch 模型）四方比对，
  第二段用自检用例把"padding 掩码错了"和"这就是量化误差"区分开。
- `tools/probe_gpu_kernels.py`：硬件能力与 Triton/TMA 可用性。

## 测试

- `tests/gpu/test_gpu_backends.py::test_backend_matches_reference`：对 fp32 参考，容差取后端声明值。
- `::test_int8_matches_comfyui_s_own_quantisation`：对 SageAttention 算术的 torch 模型，<1%。
  这是精度契约——用户打开 Veda 得到的质量就是 ComfyUI 会给的质量。
- `tests/unit/reference_backend.py`：只在测试里用的精确后端，让 CPU CI 仍能覆盖 `Backend` 契约。

## 踩坑记录

- **"ComfyUI 用 INT8 很奇怪"其实是 ComfyUI 对了**。先量再下结论：`tools/compare_int8.py` 的四方
  比对一跑就清楚了，比读三天 kernel 有用。
- **只看 Python 层的 assert 会得出错误结论**，但反过来也成立：确认一个能力不存在，要去构造它、
  让它报错，而不是没看见就当有。FP8 的 `assert arch // 10 == 10` 是"只有 SM100 写了实现"；
  而 INT8 是真的没有入口——两次都是构造出来才知道的。
- **V 的 stride 传错只在稀疏/带 padding 时才看得出来**吗？不，它一开始就 140% 全错。真正的教训是
  launch 里那一长串位置参数没有名字：`q.stride(0), q.stride(1), dim, 1` 看起来很合理，
  实际 V 的 slot stride 是 `H*D`。
- **自检容差是 kernel 的属性，不是常数**。写死 2% 会在换精度时把正确的实现判死，而且报错信息
  （"max error 0.109"）不说允许多少，查起来很费劲。
- **`tl.load` 的 `mask` 默认 `other=0`**，这不等于掩掉：0 会进 softmax。要掩就在分数上掩成 -inf。

## Sol 的池化误差修正（`_attention_pooled_kernel`）

被跳过的 tile 各贡献一项，用该 tile 的 int8 量化均值 key 打分、携带它的 fp16 求和 value，
融进同一个 online softmax；分母按该 tile 的真实行数计（value 已经是和）。走哪些 tile 由
选择阶段已经算出的 block mask 决定，不从 index 列表反推。**默认路径不变**：只有传了
`pooled_mask` 才走这个 kernel，TMA 路径不支持。

数值上与 fp32 参考 `reference.pooled_correction_attention` 对齐到 1e-4（RTX PRO 6000，
32 tile × 4 头，预算 2/4/8/16 四档）。

**固定预算下它不便宜**：实测一层 56 头、预算 32，训练尺寸 +82%、双采第二段 +68%、
第一段 +73%。这与按 FLOP 估的 +14.5% 差很远——池化分支不是 FLOP 受限的，每 `BLK_T=64`
个 tile 的一块就要做一次和 exact key block 同形状的 QK 与 PV。把摘要预先转置成
`[H, D, tiles]` 去掉循环内的 int8 转置**没有改善**（82% vs 79%），瓶颈在别处。

**但等时间下它是净赢的**，因为池化项的代价随 tile 总数走、exact 路径随预算走，所以可以
拿预算换修正。双采第一段（72 tile、8 头、一层）：

| 配置 | 耗时 | 相对 L2 |
|---|---|---|
| 直接跳过，预算 32 | 0.63 ms | 1.2792 |
| 直接跳过，预算 64 | 1.18 ms | 0.5352 |
| **加池化，预算 4** | **0.42 ms** | **0.4306** |
| **加池化，预算 16** | **0.63 ms** | **0.4201** |

同样 0.63 ms，误差从 1.28 降到 0.42；预算 4 加修正比预算 64 直接跳过又快 2.8 倍又更准。
另外注意**加了修正之后误差几乎与预算无关**（预算 4→32 只从 0.43 到 0.41），这正是
`_ADAPTIVE_FLOOR` 存在的理由消失的地方。

只在 SM120 上验过（RTX PRO 6000 Blackwell）；sm86/89/90 未测。
