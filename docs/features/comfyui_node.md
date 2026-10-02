# ComfyUI 节点与接入

## 目标

用户只多放一个节点（MODEL 进 MODEL 出），H3 的工作流、LoRA、各种 H3 权重、T2VA / FL2VA / R2VA
条件节点全部照常使用；新手只看到两个输入，问题都在节点上说清楚。

## 设计与不变量

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
- ComfyUI 自带的 "Model Sparse Attention" 在 H3 上用 block patch 直接替换注意力，Veda 的
  override 就不会被调用；节点检测到它时给出警告，而不是静默无效。
- **不坏图**：稀疏路径里的任何异常（OOM、kernel 失败）都被捕获，提示后本次运行剩余部分走全注意力；
  ComfyUI 的中断异常照常抛出。
- **可见的回退**：没有可用 kernel、布局读不懂、未训练的尺寸、头数不匹配……每一种都在节点上显示
  一次（`status.NodeStatus` → `send_progress_text`），不需要前端扩展。
- 打分器权重放在主机内存（bf16），每次调用把这一层的投影（约 11 MB）拷到设备；CUDA 上 pin 住
  做异步拷贝。这样它跟随 ComfyUI 的 offload，而不是常驻 0.5 GB 显存。

## 设置语义

| 输入 | 语义 |
|---|---|
| `current_sparsity` / `current_tiles` | current = 正在生成的目标视频 token。比例按等 kernel 代价计算（见 core_selection.md 规则 3）；`tiles > 0` 时每个 query tile 固定保留这么多个目标 key tile。 |
| `history_sparsity` / `history_tiles` | history = 条件视觉 token（FL2VA 关键帧 / AddGuide 引导帧 = `cond` 段，R2VA 参考图与参考视频 = `ref_img` 段）。保留比例 >= 1（即 0% 稀疏）时条件段不 tile，作为 global 行双向全注意力。 |
| `full_attention_layers` | 0 起的 DiT block 下标，这些层不做稀疏，跑完整注意力。 |
| `full_attention_steps` | 0 起的采样步下标。第 i 步覆盖 `sample_sigmas[i] >= sigma > sample_sigmas[i+1]`，所以多阶段采样器的中间求值也算在第 i 步。 |
| `untrained_size` | 目标网格没有完全匹配的训练方案时，用最近的方案继续稀疏，或改走全注意力。 |

## 代码位置与接口

- `veda_comfy/nodes.py`：schema、校验（模型类型、层数 / 头数与打分器一致、下标范围）、
  打分器解析与下载、就绪状态。
- `veda_comfy/comfy_patch.py`：`VedaPatch`（override、拒绝原因、统计、生命周期）与 `apply()`。
- `veda_comfy/status.py`、`veda_comfy/downloads.py`、`veda_comfy/settings.py`。

## 测试

- `tests/unit/test_comfy_integration.py`：用 ComfyUI 真实的 `MiniMaxH3Model.forward`（小随机
  模型）跑 T2VA / FL2VA / R2VA：全保留预算必须复现全注意力（1e-4）；90% 稀疏时每个 block 都走
  稀疏路径；history 段数正确；`full_attention_*` 生效；拒绝的调用到达之前的 override。
- `tests/unit/test_nodes.py`：schema（只有 model / predictor 可见）、各种错误信息、patch 安装。
- `tests/unit/test_downloads.py`：断点续传、sha256 校验、错误信息。

## 踩坑记录

- **pytest 收集仓库根的 `__init__.py`**：仓库根目录本身是 ComfyUI 的包目录，pytest 会把它当包
  导入，相对 import 失败。对策：根 `__init__.py` 只在 `__package__` 非空时 import。
- **comfy_kitchen 的原地 RoPE 不支持 autograd**：测试里的小模型参数默认 `requires_grad=True`，
  `rms_rope_split_half_` 直接报错。对策：测试模型 `requires_grad_(False)`（ComfyUI 加载的模型
  本来就是这样）。
- **测试用 HTTP server 启动要 35 秒**：`HTTPServer.server_bind` 会做反向 DNS
  （`getfqdn`）。对策：测试里覆写 `server_bind`。

## 验证记录

- 2026-10-03，Apple M3 Pro，CPU 集成测试（ComfyUI 0.38.0）80 个全过；真实 ComfyUI 0.38.0 server
  （`--cpu`）加载节点 0.0 s，`object_info` schema 与 `models/veda` 目录注册正确，示例工作流出现在
  模板列表。
