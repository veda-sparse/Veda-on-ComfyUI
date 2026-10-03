# FA4 CuTe fork

## 目标

依赖一份我们自己说了算的 FlashAttention-4 CuTe 实现：能加 SM8x / SM120 的块稀疏与 FP8，能在
Windows 上 import，且不影响同一个 ComfyUI 进程里其他用 FA4 的节点。

## 设计与不变量

- **为什么不原地 patch**：Miowtion 在进程里替换 `sys.modules["flash_attn.cute.*"]`，要求在任何人
  import FA4 之前执行。ComfyUI 一个进程加载几十个节点，加载顺序不可控，原地替换还会影响所有用
  FA4 的节点；而且锁定 `flash-attn-4==4.0.0b32` 会和别的节点的版本要求互相覆盖。
- **为什么不停留在补丁系列**：早期用 `tools/vendor_fa4.py` + `tools/fa4_patches/` 从 wheel 生成副本。
  FP8 要动的是 MMA 原子、smem 布局、输出 dtype、片段重排——横跨同一批文件的几十处，补丁之间
  互相依赖，每次试一个想法都要重新生成、重新对 sha256。更糟的是**改动是否真的进了执行路径**
  变得不显然（有一次正则静默失配，三轮 GPU 测的都是旧 kernel）。所以改成 fork。
- **做法**：`veda_comfy/kernels/fa4` 和 `veda_comfy/kernels/quack` 是**直接编辑的源码**，和仓库里
  其他代码一样读、改、提交。只保留入口点真正 import 到的模块。所有 CuTe 路径（含 sm90 / sm100）
  都走这一份，不再有"原版副本 + 打补丁副本"的分叉——一份代码，一套行为。
- **出处可查**：`tools/fa4_upstream_diff.py` 下载锁定 sha256 的上游 wheel，复现两处不属于我们的
  机械变换（只留用到的模块、把上游包 import 改成相对 import），再与 fork 对 diff，写进
  `veda_comfy/kernels/UPSTREAM.diff`。这就是 BSD-3 / Apache-2.0 声明指向的东西，也是判断"上游某个
  改动值不值得合"的依据。CI 跑 `--check` 保证它是最新的——顺带也能抓到忘记提交的编辑。
- **Windows 兼容**：FA4 在 `cache_utils.py` 顶层 `import fcntl`，QuACK 在 `cache/jit.py`、
  `cache/async_compile.py` 顶层 `import fcntl`，FA4 的 `interface.py` 会把它们都 import 进来，
  所以原版在 Windows 上根本 import 不了——尽管 CuTe DSL 从 4.8.0 起已经有 Windows wheel。
  fork 里 `fcntl` 改为可选，没有时文件锁直接放行（只影响 JIT 缓存的跨进程加锁）。另外 QuACK 的
  forkserver 预加载写死了模块名 `quack.cache._pool_preload`，改成按自己的 `__name__` 推出路径。
  不用"往 `sys.modules` 塞一个假 `fcntl`"的办法：那会让别的包误以为自己在 Unix 上。
- **lint 不管 fork**：`veda_comfy/kernels/fa4` 和 `quack` 在 ruff 的排除列表里。上游风格不是我们的
  风格，让它们一致只会把 `UPSTREAM.diff` 淹掉。

## 我们改了什么

`UPSTREAM.diff` 是权威清单。大的几块：

- **SM80 系块稀疏**（`flash_fwd.py`、`block_sparsity.py`、`interface.py`）：前向主循环按块掩码走，
  `DenseBlockMaskTorch`，低开销 launch，解除 arch-12 限制。
- **FP8**（同上 + `ampere_helpers.py`）：MMA 原子、smem 布局原子、`v_transposed`、输出 dtype
  全链路、`_acc_to_frgA_fp8`。细节见 [fp8_kernel.md](fp8_kernel.md)。
- **Windows**：`cache_utils.py`、quack 的 `cache/*`（见上）。

## 升级流程

1. 改 `tools/fa4_upstream_diff.py` 里的 `FA4_VERSION` / `FA4_WHEEL` / `FA4_WHEEL_SHA256`
   （QuACK 同理；FA4 要求的 QuACK 版本见其 wheel 的 `Requires-Dist`）。
2. `python tools/fa4_upstream_diff.py --write`，读新的 diff：上游动过的地方和我们动过的地方
   重叠多少，决定是手工合还是重新 fork。
3. 同步 `requirements-fa4.txt` 里的运行时依赖版本。
4. 在 SM8x、SM90、SM100、SM12x（以及 Windows）上跑 `tests/gpu`，结果写进 hardware.md。
   单独一个 `kernels:` 提交。

## 测试

- `tests/unit/test_fa4_fork.py`：相对 import 真的能解析、fork 不泄漏到全局 `flash_attn` /
  `quack`、FA4 后端在没装 kernel 时给安装提示而不是栈回溯。
- CI 的 `tools/fa4_upstream_diff.py`（无 `--write`）：`UPSTREAM.diff` 与工作区一致。

## 踩坑记录

- **import 改写吞掉换行**：`import flash_attn.cute.x as y` 的规则用 `\s*$` 匹配到了行尾换行，
  改写后两行粘在一起。规则里把换行显式带回去。
- **自己的生成脚本静默失配是最贵的 bug**：一次 sha256 更新的正则没匹配上，而我把 stderr 重定向
  掉了，于是三轮 GPU 跑的都是旧 kernel。fork 之后不再有生成步骤，这类问题从根上没了；
  仍然适用的教训是**改完要 grep 文件确认改动在里面**，断言失败不等于脚本会停。
