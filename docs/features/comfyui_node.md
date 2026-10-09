# ComfyUI 节点与接入

## 目标

用户只多放一个节点（MODEL 进 MODEL 出），H3 的工作流、LoRA、各种 H3 权重、T2VA / FL2VA / R2VA
条件节点全部照常使用；新手只看到两个输入，问题都在节点上说清楚。

## 设计与不变量

- **节点的自我介绍用生态里的既有说法**：ComfyUI 自己在 `comfy_extras/nodes_sparse_attention.py`
  的散文里把这个机制叫 "attention override"（"attention overrider" 在上游和第三方里都不存在），
  所以节点 description 和文档统一用 attention override，并照 `Model Sparse Attention` 的句式写：
  先说做什么，再说放哪里，最后说什么时候不生效。放置位置这一句上游核心节点都不写，但 H3 相关的
  官方教程、KJNodes 的 `Patch * Attention KJ`、`ComfyUI-H3-SLA-Attention` 都写，用户确实需要：
  **MODEL 线上，模型与 LoRA 加载之后、sampler / guider 之前**。顺序对走 override 的注意力节点
  不敏感（override 是叠上去的，见下）；LoRA 和整个替换注意力 forward 的节点必须在前（见下）。
- **category 是 `model/patch/minimax`**，和 ComfyUI 自带的 `ModelSamplingMiniMaxH3`、
  `Apply MiniMax H3 Fun ControlNet` 同一层，用户找 H3 的补丁节点时在一起。输入名 `model`
  （小写，`MODEL` 是 `io.Model` 渲染出来的接口类型，不是输入名）与输出 `display_name='model'`
  都照 `Model Sparse Attention` 对齐；**输入名是工作流 JSON 的一部分，改名会破坏已保存的工作流**。
- **接入点是 `optimized_attention_override`，不是 block patch。** ComfyUI H3 的
  `Attention.forward`（`comfy/ldm/minimax/model.py`）在 qkv 投影（已含 LoRA）、QK-norm、RoPE
  之后调用 `optimized_attention`，`transformer_options` 里带着 `block_index`、
  `minimax_h3_layout`（打包布局）和 `sigmas` / `sample_sigmas`。替换这一个调用：
  - 权重不动，所以 LoRA / 微调 / 量化权重自动兼容；
  - 不占 `patches_replace["dit"]`，与 H3 Fun ControlNet 等 block patch 共存；
  - 不处理的调用（文本 refiner、带 mask 的调用、`full_attention_*`、出错）交给安装 Veda 之前的
    override（或 ComfyUI 的默认注意力），所以其他注意力节点照样生效。
- 每一步（`ON_PREPARE_STATE`）重新把 override 放到最上层，和 ComfyUI 自带的稀疏注意力节点
  一样，避免后应用的节点悄悄覆盖；`ON_CLEANUP` 时在节点上显示本次运行的统计并清空运行状态。
- **KJNodes 的按头分组（`minimax_head_chunks`）由 Veda 接管**："MiniMax H3 Low VRAM Attention"
  让 H3 的注意力 forward 把 56 个头切成 N 组、每组调一次 `optimized_attention`。Veda 拿到的是
  头的一个切片，既对不上打分器，也无从知道是第几组（从调用顺序去猜，一次异常就会错位，而且错位
  后画质悄悄变差而不是报错）。所以 `install` 每一步把请求挪到 `veda_held_head_chunks`、把
  `minimax_head_chunks` 置 1：forward 一次交出全部头，稀疏路径自己按显存分块；Veda 不处理的调用
  在 override 里照 KJ 的公式分组调用，保住它省显存的效果。分组只是按头切开，结果逐位相同。
  这个分组数同时作为 Veda 自己分块的下限（`engine.attention(head_chunks=...)`）：否则节点接上
  去对稀疏层毫无作用——Veda 的默认上限已经比 `head_chunks=4` 更紧（实测两者都是 1.31 GB），
  要求更多时才会继续降（`head_chunks=14`：每块 4 个 head，峰值 1.03 GB）。
- **整个替换注意力 forward 的节点：在 Veda 之前就接管，在之后就报出来。** KJNodes 的
  "MiniMax H3 Mem Eff Sage Attention Patch" 用 `add_object_patch` 把
  `blocks.N.attn.forward` 换成直接调 sage 的 forward，根本不经过 `optimized_attention`，
  Veda 只在执行节点时显示一次 ready，之后一个调用也收不到（用户看到的是"没有下文、还更慢"）。
  - 在 Veda 之前：节点执行时找出这类 object patch（KJ 的约定：仍调 `optimized_attention` 的
    forward 带 `_uses_optimized_attention`，带了就不动），按 block 换成路由 forward。Veda 可能
    稀疏的 block（`_block_reason` 只看采样前就知道的条件：层、步、出错、无 kernel）走调用
    override 的 forward——优先用 KJ Low VRAM 节点留在 `sol_take_forward` 的那个，否则用
    ComfyUI 的原生 forward；其余 block 仍交给被替换的 forward（并把按头分组的请求还给它）。
    节点上用 warning 说明接管了谁，verbose 再写稀疏层走的是哪个 forward。
  - 在 Veda 之后：object patch 是在 Veda 之后才加的，Veda 已无法改。每一步
    （`ON_PREPARE_STATE`）检查采样用的 patcher，发现就在节点上警告"把 Veda 移到它后面"。
  - 兜底：一次采样结束、Veda **一个调用都没收到**（`run.calls` 为空）时，节点警告
    "Veda did not run"。注意条件是"一个都没收到"，不是"一个都没加速"：两段式 / 分块采样器的
    第二段会把上百个调用送进来又全被拒掉，那不是被别的节点替换了（见下）。
- **被拒的调用要分清是不是 H3 自注意力**：override 装在 `transformer_options` 上，模型里所有
  `optimized_attention` 都会经过它，包括 token refiner。`minimax_h3_layout` 在 H3 的 `forward`
  里（model.py:624）就写进去了，而 refiner 在 719 行才跑，所以**refiner 也会看到一个 seq_len
  对不上的 layout**。区分两者的是 `block_index`：只有 DiT 的 block 循环（755 行）会设它。
  layout 对不上但带 `block_index` 的，记成 `_WRONG_LAYOUT`，计入摘要（这些调用确实跑了，而且
  是全价）；其余记成 `other attention`，不计入——否则 refiner 会把"算了全注意力的百分之多少"
  这个数字搅乱。
- ComfyUI 自带的 "Model Sparse Attention" 在 H3 上用 block patch 直接替换注意力，Veda 的
  override 就不会被调用；节点检测到它时给出警告，而不是静默无效。
- **不坏图**：稀疏路径里的任何异常（OOM、kernel 失败）都被捕获，提示后本次运行剩余部分走全注意力；
  ComfyUI 的中断异常照常抛出。
- **可见的回退**：没有可用 kernel、布局读不懂、未训练的尺寸、头数不匹配……每一种都在节点上显示
  一次（`status.NodeStatus` → `send_progress_text`），不需要前端扩展。
- **"硬件不支持"和"装坏了"要分开说**：`Resolution.has_candidate` 为假才是这张卡不在范围内；
  为真而 `backend` 是 None，说明覆盖它的 kernel 没能在这台机器上启动起来，那是环境问题。
  两者的提示完全不同，混在一起会让用户去查一个并不存在的硬件问题（3090 的例子见
  hardware.md）。具体怎么修只有后端自己知道，所以由 `Backend.explain_failure(error)` 返回，
  registry 收集到 `Resolution.hints`——这样 Triton 的知识留在 Triton 后端里（规则 1.6）。
- **警告和错误中英双语，界面文字跟随语言**：`send_progress_text` 和 `ValueError` 都是后端发出的
  字符串，而后端不知道每个浏览器的界面语言（多个浏览器可以同时连同一个 ComfyUI），所以无法
  "自适应"。因此运行时的警告与错误一律英文在前、中文在后，每条先说 Veda 是否在跑，再说怎么办；
  日志只写英文。节点名、描述、tooltip 是前端渲染的静态文字，ComfyUI 会读自定义节点的
  `locales/<lang>/nodeDefs.json`，这部分真正跟随界面语言（`locales/zh`）。普通状态文字只用英文。
- **整个稀疏路径在 `pause_malloc_graph()` 里跑**：ComfyUI 0.38 起，H3 前向的每个 block 都在
  comfy-aimdo 的显存分配录制（malloc graph）里执行，block 内的分配必须在 block 结束前释放。
  Veda 跨调用保存设备状态（tile 布局、head 组、统计量），kernel 也有自己的 workspace，在录制区
  里分配会让进程直接 abort（Python 的 try/except 接不住）。暂停录制后这些分配走普通分配器。
- **Veda 的显存开销由问题决定，不由空闲显存决定**：一次注意力按 head 分块，早期只用
  `free // 8`（上限 1 GB）定块大小，于是显卡越空块越大——实测 1344x768 x 5.2 s 下 Veda 峰值
  3.24 GB，而全注意力只要 0.64 GB，多出来的几 GB 正是 ComfyUI 拿来搬权重的那部分（issue #1 的
  OOM 就发生在权重搬运的 MLP 上，而不是 Veda 里）。现在再加一条与机器无关的上限：一个块的
  tile 缓冲不超过该层输出缓冲的 1 / `_WORKING_SET_SHARE`，即每块约 H/6 个 head。实测峰值
  3.24 → 1.31 GB，每步耗时不变（13.6 s）。空闲显存那条上限保留，小显存上仍然继续收缩。
- 打分器权重放在主机内存（bf16），每次调用把这一层的投影（约 11 MB）拷到设备；CUDA 上 pin 住
  做异步拷贝。这样它跟随 ComfyUI 的 offload，而不是常驻 0.5 GB 显存。

## 设置语义

| 输入 | 语义 |
|---|---|
| `generated_sparsity` | generated = 正在生成的目标视频 token。`trained` 取打分器自己声明的预算（见下），节点上会写明取到的是什么。一个输入框两种写法：`90%` 是稀疏比例（保留比例 = 1 - 90%，按等 kernel 代价计算，见 core_selection.md 规则 3）；整数如 `24` 是每个 query tile 固定保留的 key tile 数。 |
| `reference_sparsity` | reference = 条件视觉 token（FL2VA 关键帧 / AddGuide 引导帧 = `cond` 段，R2VA 参考图与参考视频 = `ref_img` 段），写法同上。`0%` 时参考段不 tile，作为 global 行双向全注意力。 |
| `full_attention_layers` | 0 起的 DiT block 下标，这些层不做稀疏，跑完整注意力。 |
| `full_attention_steps` | 0 起的采样步下标。第 i 步覆盖 `sample_sigmas[i] >= sigma > sample_sigmas[i+1]`，所以多阶段采样器的中间求值也算在第 i 步。 |
| `selection` | `fixed` 固定保留上面那个数量；`adaptive` 用 Sol-Attn 的规则，按该行打分分布的 `tau` 个标准差定阈值，保留数随内容和尺寸变化。 |
| `tau` | `adaptive` 的阈值，单位是标准差，默认 1.3（与 ComfyUI 的 Sol-Attn 一致）。 |
| `verbose` | 运行结束后在节点和终端显示诊断：本次的几何（请求尺寸 / latent / token 网格）、选到的方案与补齐比例、序列长度与各 span、选块策略、后端与设备、保留的 video tile 比例、**每个 query tile 保留的 key tile 数的 min / mean / max**（摆动大且数量小正是闪烁的来源）、分块与工作集、各阶段耗时、调用次数与全注意力原因、打分器信息。 |

尺寸没有完全匹配的训练方案时，固定使用纵横比最接近、其次时长最接近的方案（含 H/W 转置），节点上的
「Tile plan」一行会写成 `nearest trained size: ...`，用措辞而不是图标说明这次不在训练分布内。

**节点文字**（`send_progress_text`，多行）：就绪时显示 kernel（大写的正式名，如 `Triton INT8 (SM120)`）和稀疏度；
第一次稀疏调用时显示视频尺寸 / 时长和匹配的方案；运行结束显示"实际计算了全注意力的百分之多少"——按 128x128
块计：保留的 video 块 + 永远全算的 global 行列，除以全注意力的 (S/128)² 块。不显示调用次数（那是 verbose
的内容）。

- **打分器自带训练预算，节点默认跟着它走**：T2VA 的 bundle 里是 `keep_ratio=0.1`（= 90% 稀疏），
  R2VA 换成了一组新字段 `target_budget_kind/value` 与 `ref_budget_kind/value`（都是 `tiles` 32）。
  `bundle._budget` 两种都认：没有新字段时退回 `keep_ratio`，两者都没有才报 incomplete metadata。
  **旧的加载器硬要 `keep_ratio`，所以 R2VA 的 bundle 根本加载不了**（`BundleError: incomplete
  metadata ('keep_ratio')`）。稀疏度输入的默认值因此改成 `trained`：用户不需要记住哪个打分器
  训练在什么预算上，想覆盖照样可以写 `90%` 或 `24`。

- **补齐太多时报出更好的输入尺寸**：tile 是整块算的，补齐出来的行照样进 kernel，是白花的时间。
  但**门槛不能设成 0**：方案自己搜出来的网格也有补齐（16:9 t72 是 5.6%，因为它混用的几种形状
  并不都整除 24x42），对着训练尺寸喊"换一个"是坏建议。所以比较的是**相对该方案自己网格的
  增量**（`_PADDING_HINT`，3 个百分点），超了才提示，并给出按 `tiling_period`（各形状每条轴的
  最小公倍数）向上对齐后的请求尺寸——像素 + 帧数 + 秒数，以及对应的 latent / token 网格。

## 代码位置与接口

- `veda_comfy/nodes.py`：schema、校验（模型类型、层数 / 头数与打分器一致、下标范围）、
  打分器解析与下载、就绪状态。
- `veda_comfy/comfy_patch.py`：`VedaPatch`（override、拒绝原因、统计、生命周期）与 `apply()`。
- `veda_comfy/status.py`、`veda_comfy/predictors.py`（纯元数据，不联网）、
  `veda_comfy/settings.py`。

## 测试

- `tests/unit/test_comfy_integration.py`：用 ComfyUI 真实的 `MiniMaxH3Model.forward`（小随机
  模型）跑 T2VA / FL2VA / R2VA：全保留预算必须复现全注意力（1e-4）；90% 稀疏时每个 block 都走
  稀疏路径；reference 段数正确；`full_attention_*` 生效；拒绝的调用到达之前的 override；
  照 KJNodes Low VRAM 节点按头分组的 forward 下仍然每层稀疏、结果与不分组相同，拒绝的调用
  仍按头分组；绕开 override 的 forward 会让运行结束时警告，路由后的 forward 让稀疏层走 Veda、
  全注意力层回到被替换的 forward（含 `sol_take_forward` 的情形）。
- `tests/unit/test_nodes.py` 另测：错误信息英文一行、中文一行；`locales/zh/nodeDefs.json` 的输入
  与 schema 一致；节点执行时接管在它之前的 forward 替换，并在节点上说明；
  在它之后的替换由 `ON_PREPARE_STATE` 报出来。
- `tests/unit/test_nodes.py`：schema（只有 model / predictor 可见）、各种错误信息、patch 安装。
- `tests/unit/test_predictors.py`：发布元数据被钉死（完整 commit、sha256）、URL 指向该
  revision、缺文件时的提示包含地址，以及**模块里不出现网络/环境变量字样**。

## 踩坑记录

- **pytest 收集仓库根的 `__init__.py`**：仓库根目录本身是 ComfyUI 的包目录，pytest 会把它当包
  导入，相对 import 失败。对策：根 `__init__.py` 只在 `__package__` 非空时 import。
- **comfy_kitchen 的原地 RoPE 不支持 autograd**：测试里的小模型参数默认 `requires_grad=True`，
  `rms_rope_split_half_` 直接报错。对策：测试模型 `requires_grad_(False)`（ComfyUI 加载的模型
  本来就是这样）。
- **测试用 HTTP server 启动要 35 秒**：`HTTPServer.server_bind` 会做反向 DNS
  （`getfqdn`）。对策：测试里覆写 `server_bind`。

- **CPU 版 torch 上 ComfyUI 的 `get_torch_device()` 直接 assert**：ComfyUI 的 `cpu_state`
  默认是 `CPUState.GPU`，只有传了 `--cpu` 或检测到 MPS 时才改
  （`comfy/model_management.py`）。CPU 版 torch 两个条件都不满足，于是
  `get_torch_device()` 落到 `torch.cuda.current_device()`，抛
  "Torch not compiled with CUDA enabled"——CI 的 Linux 和 Windows 上 19 个测试直接 error，
  macOS 因为有 MPS 看不到。对策：`tests/conftest.py` 在任何人 import `model_management`
  之前设 `comfy.cli_args.args.cpu = True`。这同时消掉一个平台差异：在此之前 macOS 拿 MPS
  跑这些 CPU 测试、还会去探 mlx 后端，Linux 不会。
- **删后端时集成测试没跟着改，而且被红色的 CI 盖住了**：706d768 删掉 fallback 后端后，
  `test_comfy_integration.py` 还在给 `VedaSettings` 传 `backend='torch'`、还在断言节点显示
  "PyTorch SDPA"，11 个测试从那时起一直 fail。因为上面那条让 CI 本来就是红的，这 11 个
  混在 error 里没人注意到。对策：集成测试通过 monkeypatch `backends.resolve` 用
  `tests/unit/reference_backend.py`，不再依赖哪个后端恰好存在；**CI 红的时候先修 CI，
  不要让它一直红着**。

- **malloc graph 导致原生 abort**：Windows + RTX 5070 的第一次端到端运行在第一次稀疏调用
  就 "Fatal Python error: Aborted"。CPU 集成测试发现不了（malloc graph 只在 CUDA 上启用）。
  对策见上；`tests/gpu` 之外，任何 GPU 改动都要用 `tools/e2e_minimax_h3.py` 在真实 ComfyUI 里
  跑一遍。

- **KJNodes "MiniMax H3 Low VRAM Attention" 让 Veda 关掉**：节点显示 "Veda off: the model has
  14 heads of dim 128, the predictor expects 56 x 128"（`head_chunks=4`；6 组时是 10/9，最后显示
  9）。原因：它按头分组调用注意力，Veda 每次只看到一组头。对策见上「按头分组由 Veda 接管」。

- **KJNodes "MiniMax H3 Mem Eff Sage Attention Patch" 让 Veda 静默失效**：节点只显示 ready，
  之后没有任何文字，渲染还变慢（全程是 sage 的全注意力）。原因：它整个替换了 `attn.forward`，
  不经过 override；而 Veda 只在收到调用时才说话。对策见上「整个替换注意力 forward 的节点」。

## 验证记录

- 2026-10-03，Apple M3 Pro，CPU 集成测试（ComfyUI 0.38.0）80 个全过；真实 ComfyUI 0.38.0 server
  （`--cpu`）加载节点 0.0 s，`object_info` schema 与 `models/veda` 目录注册正确，示例工作流出现在
  模板列表。
