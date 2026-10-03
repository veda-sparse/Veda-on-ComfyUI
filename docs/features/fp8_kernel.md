# FP8 注意力（SM8x / SM120）

## 目标

在消费级 Blackwell / Ada 上用 FP8 tensor core 跑 Veda 的块稀疏注意力。FP8 的稠密 GEMM 在
RTX 5070 上实测是 bf16 的 **2.00 倍**（139 vs 69 TFLOPS，`tools/probe_gpu_kernels.py`），而 attention
kernel 在 90% 稀疏下约占 Veda 单层时间的 77%，且本身已接近 MMA 受限（38k token 时 4.2 TFLOP/层
对应 93 ms，约为 bf16 峰值的 65%），所以这是剩下最大的一块。

## 设计与不变量

- **上游的 FP8 门禁是实现门，不是硬件门**。`interface.py` 里 `assert arch // 10 == 10` 与实现一致：
  只有 `flash_fwd_sm100.py` 有 FP8 路径，`flash_fwd.py`（SM80 系，SM120 继承）和 `flash_fwd_sm90.py`
  一行 FP8 代码都没有。但 SM89 / SM120 有 FP8 tensor core，CuTe DSL 也暴露了
  `warp.MmaFP8Op`（m16n8k32，实测在 sm120 上编译通过）。因此我们在**自己的补丁系列**
  （`tools/fa4_patches/sm8x/0008-*.patch`）里给 SM80 系 kernel 加 FP8，而不是另起炉灶。
- **scale 一律不进 kernel**。e4m3 是浮点格式，相对误差与数量级无关，所以每个张量一个 scale 就够，
  不需要 per-tile scale（这点和 INT8 相反——定点量化必须 per-block）。于是：
  - q、k 的 scale 折进 `softmax_scale`（`softmax_scale' = softmax_scale · q_scale · k_scale`，
    这样 LSE 也是对的）；
  - v 的 scale 折进输出（softmax 行和为 1，输出整体线性于 v 的 scale）。

  kernel 里因此没有任何 descale 代码，补丁面积小很多。
- **量化在后端内部做**，不改 `Backend.attend` 接口，也不碰其他后端（AGENTS.md 1.6）。
- **输出 dtype 与操作数 dtype 分离**：FP8 操作数累加进 16 位输出，所以 Q/K/V 与 O 不再同型。
  这正是上游自己留的 TODO（"need a different layout for O if O dtype is not the same as V dtype"）。
  补丁给 O 单独的 smem 布局、gmem 拷贝、epilogue 转换，并把 epilogue 复用的 Q 缓冲按 O 的字节数放大。

## 现状

**已完成并验证**

- 可行性：`MmaFP8Op`（e4m3/e5m2，16x8x32 与 16x8x16）在 sm120 上编译通过；FP8 GEMM 2.00x bf16。
- 补丁基础设施：`tools/vendor_fa4.py --work-dir` 拉出可编辑的补丁树；补丁 0008 入系列，
  上游与打补丁后的文件各自 sha256 锁定，CI 的 `--check` 保证提交的副本就是生成结果。
- kernel 侧：MMA 原子、dtype 白名单、smem 字节计算、输出 dtype 全链路（布局、拷贝、epilogue、
  缓冲大小）、`_check_type` 接受 FP8 进 16 位出。
- 外围：两个打补丁后端的 `-fp8` 变体（后端内量化 + scale 折叠）、注册表接线（只在 SM8x / SM120
  提供，sm90 请求时正确回落）、GPU 测试。
- **bf16 路径无回归**：补丁合入后 RTX 5070 上 `tests/gpu` 3 passed、测速仍是 136.9 ms/层（11.53x）。

**未完成：1 字节操作数的 smem→寄存器加载**

FP8 卡在操作数加载，不在 MMA。按顺序撞到三堵墙：

1. V 用转置版 8 位 ldmatrix：`src ptr alignment (64 bits) does not meet requirement (128 bits)`。
   1 字节 tile 的转置视图给不出 16 字节对齐。
2. V 改用通用拷贝：`expects layouts of src and dst to be LayoutType`——通用拷贝不接受 swizzle 过的
   复合布局。
3. 给 V 去掉 swizzle 后，同样的报错移到 Q/K 的非转置 8 位 ldmatrix 上（报错里的 `S<4,4,4>` 是 Q/K 的
   atom）。

根因是 `ampere_helpers.get_smem_layout_atom` 的 swizzle 推导是按 16 位操作数写的，1 字节操作数需要
自己的 K 分块与 swizzle（CUTLASS 对 FP8 操作数有专门的布局）。这是 FP8 attention 公认最麻烦的一段，
SageAttention / FA3 也都为此写了专用布局。

**下一步（按代价排序）**

1. 给 FP8 操作数写专用的 smem 布局原子，使其满足 8 位 ldmatrix 的对齐与布局要求。改动集中在
   `get_smem_layout_atom`，是正解。
2. 引擎侧产出 V^T（gather 阶段本来就在重排 V，几乎免费），kernel 用非转置 8 位 ldmatrix 读 V。
   绕开转置这一堵墙，但 V 的 gmem→smem 路径要改。
3. 混合精度（QK 走 FP8、PV 保持 bf16）：避开 V 的全部问题，但 q/k/v 在 gmem 里 dtype 不同，
   interface 的 dtype 推导、uint8 视图等都要改，收益也只有约 2/3。

## 代码位置与接口

- `tools/fa4_patches/sm8x/0008-*.patch`：kernel 补丁（见上）。
- `tools/vendor_fa4.py`：`--work-dir` 补丁开发模式；`UPSTREAM_SHA256` / `PATCHED_SHA256` 锁定。
- `veda_comfy/backends/fa4_sm80.py`、`fa4_sm120.py`：`_to_fp8()` 与 `attend()` 里的 scale 折叠。
- `veda_comfy/backends/__init__.py`：`fa4-fp8` 选项、`_load()`、`_FP8_CAPABLE`。
- `tools/probe_gpu_kernels.py`：硬件与 CuTe 能力报告。

## 测试

- `tests/gpu/test_gpu_backends.py::test_fp8_matches_reference_and_bf16`：对 fp32 参考与 bf16 kernel
  双向比对。目前 **xfail**（见上）；布局问题解决后去掉 xfail 并定下容差。
- `tests/unit/test_vendor_fa4_backends.py`：补丁数量、生成副本的私有性与可解析性。

## 踩坑记录

- **只看 Python 层的 assert 会得出错误结论**。`assert arch // 10 == 10` 看起来是"硬件不支持"，
  实际是"只有 SM100 写了 kernel"。对策：`tools/probe_gpu_kernels.py` 直接问硬件和 DSL，
  任何新架构下结论前先跑它。
- **jit 追踪的函数在模块全局里找符号**：探测工具里把 `import cutlass` 放在函数内，
  导致所有编译检查报 `NameError: name 'cute' is not defined`，看起来像"全都不支持"。
- **默认参数会被 DSL 当成运行时值**：`def build(..., transpose=transpose)` 让 op 收到 DSL value
  而不是 Python bool，报 `expects the 'transpose' Op parameter to be a bool instance`。改用闭包。

## 验证记录

- 2026-10-03，RTX 5070（SM120）、Windows 11、torch 2.14.1+cu130、CuTe DSL 4.8.0：
  能力探测与 bf16 无回归结果如上；FP8 端到端尚未跑通（操作数布局）。
