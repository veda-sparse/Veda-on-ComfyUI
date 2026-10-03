# FP8 注意力（SM8x / SM120）

## 目标

在消费级 Ada / Blackwell 上用 FP8 tensor core 跑 Veda 的块稀疏注意力。FP8 的稠密 GEMM 在
RTX 5070 上实测是 bf16 的 **2.00 倍**（139 vs 69 TFLOPS，`tools/probe_gpu_kernels.py`），而 attention
kernel 在 90% 稀疏下约占 Veda 单层时间的 77%，且本身已接近 MMA 受限（38k token 时 4.2 TFLOP/层
对应 93 ms，约为 bf16 峰值的 65%），所以这是剩下最大的一块。

## 设计与不变量

- **上游的 FP8 门禁是实现门，不是硬件门**。上游 `interface.py` 里 `assert arch // 10 == 10` 与实现
  一致：只有 `flash_fwd_sm100.py` 有 FP8 路径，`flash_fwd.py`（SM80 系，SM120 继承）和
  `flash_fwd_sm90.py` 一行 FP8 代码都没有。但 SM89 / SM120 有 FP8 tensor core，CuTe DSL 也暴露了
  `warp.MmaFP8Op`（m16n8k32，实测在 sm120 上编译通过）。我们在**自己维护的 fork**
  （`veda_comfy/kernels/fa4`，见 [fa4_fork.md](fa4_fork.md)）里给 SM80 系 kernel 加 FP8。
- **scale 一律不进 kernel**。e4m3 是浮点格式，在其**正规范围内**相对误差与数量级无关，所以每个
  张量一个 scale 就够，不需要 per-tile scale（这点和 INT8 相反——定点量化必须 per-block，这也是
  ComfyUI 选 INT8 时要配 per-block scale 的原因）。于是：
  - q、k 的 scale 折进 `softmax_scale`（`softmax_scale' = softmax_scale · q_scale · k_scale`，
    这样 LSE 也是对的）；
  - v 的 scale 折进输出（softmax 行和为 1，输出整体线性于 v 的 scale）。
- **P 必须另外放大**。softmax 之后 p 典型在 1e-4 量级，低于 e4m3 的最小正规数 2⁻⁶ ≈ 0.0156，落进
  次正规区会被截成零。`FP8_P_SCALE = 256.0` 把 p 抬进正规范围，再从输出里除掉——和
  SageAttention2++ / FA3 的做法一致。这是"e4m3 不需要 scale"这句话唯一的例外，也是最早栽进去的坑。
- **V 在主机侧物理转置**。PV gemm 要的是 Vᵀ，而 1 字节 ldmatrix 的转置版要求 128 位对齐，1 字节
  tile 的转置视图给不出来。gather 阶段本来就在重排 V，顺手转置几乎免费，kernel 于是只用非转置
  的 8 位 ldmatrix。
- **量化在后端内部做**，不改 `Backend.attend` 接口，也不碰其他后端（AGENTS.md 1.6）。
- **输出 dtype 与操作数 dtype 分离**：FP8 操作数累加进 16 位输出，所以 Q/K/V 与 O 不再同型。
  这正是上游自己留的 TODO（"need a different layout for O if O dtype is not the same as V dtype"）。
  fork 给 O 单独的 smem 布局、gmem 拷贝、epilogue 转换，并把 epilogue 复用的 Q 缓冲按 O 的字节数放大。

## P 操作数的片段布局（最难的一段）

QK 的累加器要直接当 PV gemm 的 A 操作数，不经过 shared memory。16 位路径上 quack 的
`reshape_acc_to_frgA` 只做一个 **strided view** 就够了，因为 m16n8k16 的 A 寄存器顺序恰好等于
累加器自己的 (column, row, tile) 顺序。

m16n8k32 不一样：每个线程的 A 要 **4 个连续的 k**，而累加器一行只给 2 个（C 片段里每线程拿的是
相邻两列，stride 2）。所以 A 的第 i 个槽位要从累加器的第 `8·(i//2) + 2t + (i%2)` 列取——这是一次
真正的重排，不是换个看法。

**关键坑**：用 `cute.make_layout` 造一个重新嵌套的 strided view **不管用**。CuTe 会按 stride 把布局
规范化，写进去的顺序不会被保留，于是行又混回 k 里。两种不同嵌套跑出逐位相同的结果就是这个原因，
而且因为结果一样，很容易误判成"改动没生效"。正解是**显式逐元素搬运**
（`_acc_to_frgA_fp8`，`flash_fwd.py`）：按 PTX 填寄存器的顺序（4 个 k → row → 再往后 16 的 k 块）
把 16 个值取出来，任何规范化都撤不掉。

剩下的只是 k 的一个固定置换，由主机侧 `FP8_V_PERMUTATION` 作用在 V 上抵消：

```python
FP8_V_PERMUTATION = (0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15)
```

该函数**没有** DSL 装饰器，所以不走预处理器，`cutlass.range_constexpr` 在里面会直接报
`range_constexpr should be preprocessed by preprocessor`。用普通 Python `range` 即可——
这个辅助函数本来就在 trace 时执行，循环自然全展开。

## 现状

**跑通并验证**（2026-10-03，RTX 5070 / SM120 / Windows 11 / torch 2.14.1+cu130 / CuTe DSL 4.8.0）：

| 对比 | 相对 L2 |
| --- | --- |
| FP8 kernel vs bf16 kernel | 2.54% |
| bf16 kernel vs fp32 参考 | 0.25% |
| 仅 P（V 取单位阵） | 2.58% |
| 仅 V（softmax 拉平） | 0.35% |
| LSE（只依赖 QK + softmax） | max \|diff\| 0.0015，范围 [8.539, 9.104] |

2.5% 就是 e4m3 三位尾数的量化地板（P 的舍入是主要项，见"仅 P"与"仅 V"的差距），不是布局故障。
bf16 路径无回归。

## 代码位置与接口

- `veda_comfy/kernels/fa4/flash_fwd.py`：`FP8_P_SCALE`、`FP8_V_PERMUTATION`、`_acc_to_frgA_fp8`、
  MMA 原子选择、`v_transposed`、输出 dtype 全链路。
- `veda_comfy/kernels/fa4/interface.py`：FP8 架构门禁、Vᵀ 形状断言、launch cache 的 uint8 视图。
- `veda_comfy/backends/fa4_sm80.py`、`fa4_sm120.py`：`_to_fp8()`、`_v_permutation()`、scale 折叠。
- `veda_comfy/backends/__init__.py`：`fa4-fp8` 选项、`_load()`、`_FP8_CAPABLE`。
- `tools/probe_gpu_kernels.py`：硬件与 CuTe 能力报告。
- `tools/diagnose_fp8.py`：分级诊断（LSE / scale 折叠 / 仅 P / 仅 V / 稠密掩码 / 槽位映射推断）。
  FP8 再出问题先跑它——它能把故障压到某一级。

## 测试

- `tests/gpu/test_gpu_backends.py::test_fp8_matches_reference_and_bf16`：对 fp32 参考与 bf16 kernel
  双向比对，容差 L2 < 5%、逐点 < 10%。
- `tests/unit/test_kernels_fork.py`：fork 的私有性与可解析性。

## 踩坑记录

- **只看 Python 层的 assert 会得出错误结论**。`assert arch // 10 == 10` 看起来是"硬件不支持"，
  实际是"只有 SM100 写了 kernel"。对策：`tools/probe_gpu_kernels.py` 直接问硬件和 DSL，
  任何新架构下结论前先跑它。
- **"e4m3 相对误差与数量级无关"只在正规范围内成立**。softmax 的 p ≈ 3e-4 低于最小正规数，
  不放大就被吃掉。见 `FP8_P_SCALE`。
- **strided view 会被 CuTe 按 stride 规范化**。改嵌套顺序却得到逐位相同的输出，不代表改动没生效，
  而代表这条路走不通。要重排就显式搬运。
- **逐位相同要先证明代码进了执行路径**。一次用 env 开关在布局运算处打印（它在 trace 时以 Python
  执行），立刻排除了"改动没生效"这个假设，省掉了后面所有瞎猜。
- **`range_constexpr` 只在被预处理的入口里有效**，普通辅助函数里用会直接编译失败。
- **e4m3 没有 m16n8k16 指令**，要了也会发 k=32。这解释了若干次"换了 MMA 形状结果不变"。
- **jit 追踪的函数在模块全局里找符号**：探测工具里把 `import cutlass` 放在函数内，
  导致所有编译检查报 `NameError: name 'cute' is not defined`，看起来像"全都不支持"。
- **默认参数会被 DSL 当成运行时值**：`def build(..., transpose=transpose)` 让 op 收到 DSL value
  而不是 Python bool，报 `expects the 'transpose' Op parameter to be a bool instance`。改用闭包。
- **SM8x 的 launch cache 会按原 dtype 回放**：FP8 张量必须和首次编译时一样以 uint8 视图传入，
  否则重放时类型对不上（`_FP8_DTYPES`）。
