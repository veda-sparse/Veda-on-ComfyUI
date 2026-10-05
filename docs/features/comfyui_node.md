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
  **MODEL 线上，模型与 LoRA 加载之后、sampler / guider 之前**。顺序对其他注意力节点不敏感
  （override 是叠上去的，见下），只有 LoRA 必须在前。
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
- ComfyUI 自带的 "Model Sparse Attention" 在 H3 上用 block patch 直接替换注意力，Veda 的
  override 就不会被调用；节点检测到它时给出警告，而不是静默无效。
- **不坏图**：稀疏路径里的任何异常（OOM、kernel 失败）都被捕获，提示后本次运行剩余部分走全注意力；
  ComfyUI 的中断异常照常抛出。
- **可见的回退**：没有可用 kernel、布局读不懂、未训练的尺寸、头数不匹配……每一种都在节点上显示
  一次（`status.NodeStatus` → `send_progress_text`），不需要前端扩展。
- **整个稀疏路径在 `pause_malloc_graph()` 里跑**：ComfyUI 0.38 起，H3 前向的每个 block 都在
  comfy-aimdo 的显存分配录制（malloc graph）里执行，block 内的分配必须在 block 结束前释放。
  Veda 跨调用保存设备状态（tile 布局、head 组、统计量），kernel 也有自己的 workspace，在录制区
  里分配会让进程直接 abort（Python 的 try/except 接不住）。暂停录制后这些分配走普通分配器。
- 打分器权重放在主机内存（bf16），每次调用把这一层的投影（约 11 MB）拷到设备；CUDA 上 pin 住
  做异步拷贝。这样它跟随 ComfyUI 的 offload，而不是常驻 0.5 GB 显存。

## 设置语义

| 输入 | 语义 |
|---|---|
| `generated_sparsity` | generated = 正在生成的目标视频 token。一个输入框两种写法：`90%` 是稀疏比例（保留比例 = 1 - 90%，按等 kernel 代价计算，见 core_selection.md 规则 3）；整数如 `24` 是每个 query tile 固定保留的 key tile 数。 |
| `reference_sparsity` | reference = 条件视觉 token（FL2VA 关键帧 / AddGuide 引导帧 = `cond` 段，R2VA 参考图与参考视频 = `ref_img` 段），写法同上。`0%` 时参考段不 tile，作为 global 行双向全注意力。 |
| `full_attention_layers` | 0 起的 DiT block 下标，这些层不做稀疏，跑完整注意力。 |
| `full_attention_steps` | 0 起的采样步下标。第 i 步覆盖 `sample_sigmas[i] >= sigma > sample_sigmas[i+1]`，所以多阶段采样器的中间求值也算在第 i 步。 |
| `verbose` | 运行结束后节点上额外显示诊断信息：Veda 注意力总耗时、每次模型调用的耗时、各阶段（gather / score / select / kernel / scatter）耗时（CUDA event 计时）、保留的 video tile 比例、调用次数与全注意力原因、打分器信息。 |

尺寸没有完全匹配的训练方案时，固定使用纵横比最接近、其次时长最接近的方案（含 H/W 转置），节点上的
「Tile plan」一行会写成 `nearest trained size: ...`，用措辞而不是图标说明这次不在训练分布内。

**节点文字**（`send_progress_text`，多行）：就绪时显示 kernel（大写的正式名，如 `Triton INT8 (SM120)`）和稀疏度；
第一次稀疏调用时显示视频尺寸 / 时长和匹配的方案；运行结束显示"实际计算了全注意力的百分之多少"——按 128x128
块计：保留的 video 块 + 永远全算的 global 行列，除以全注意力的 (S/128)² 块。不显示调用次数（那是 verbose
的内容）。

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
  仍按头分组。
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

## 验证记录

- 2026-10-03，Apple M3 Pro，CPU 集成测试（ComfyUI 0.38.0）80 个全过；真实 ComfyUI 0.38.0 server
  （`--cpu`）加载节点 0.0 s，`object_info` schema 与 `models/veda` 目录注册正确，示例工作流出现在
  模板列表。
