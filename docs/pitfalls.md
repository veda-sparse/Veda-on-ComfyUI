# 踩坑总表

| 现象 | 原因 | 对策 | 详情 |
|---|---|---|---|
| ComfyUI 的 INT8 不是"奇怪"，是更准 | e4m3 只有 3 位尾数（~3.6%/值），INT8 per-block 是 127 个均匀档（~0.9%） | 量化注意力一律走 INT8；结论靠 `tools/compare_int8.py` 实测，不靠推理 | [int8_kernel](features/int8_kernel.md) |
| CuTe 做不了 INT8 注意力 | DSL 4.8.0 的 warp 层没有整数 MMA，`MmaI8Op` 只在 SM100 的 `tcgen05` 下 | kernel 写在 Triton（`tl.dot` 原生支持 int8） | [int8_kernel](features/int8_kernel.md) |
| 回退链让测速一直在量错的 kernel | 自检失败就静默换后端，摘要行在很靠上的位置 | 一个设备一个 kernel，不回退，装不上就报错 | [backends](features/backends.md) |
| 自检把正确的量化 kernel 判死 | 2% 的逐点容差是 16 位时代的常数 | 容差改成 `Backend.tolerance`，报错同时打印允许值 | [backends](features/backends.md) |
| Triton `tl.load(mask=...)` 没有掩掉越界 key | `other` 默认 0，0 会以 `exp2(0-m)` 的权重进 softmax | 在分数上掩成 -inf，`m_i` 用有限的 -1e30 初值避免 NaN | [int8_kernel](features/int8_kernel.md) |
| pytest 报 "attempted relative import with no known parent package" | pytest 把仓库根的 `__init__.py` 当包导入 | 根 `__init__.py` 只在 `__package__` 非空时 import | [comfyui_node](features/comfyui_node.md) |
| 测试里的 H3 前向报 "in-place RoPE ... do not support autograd" | 测试模型参数 `requires_grad=True` | `requires_grad_(False)` | [comfyui_node](features/comfyui_node.md) |
| 本地 HTTP 测试服务器启动 35 秒 | `server_bind` 的反向 DNS | 覆写 `server_bind` | [comfyui_node](features/comfyui_node.md) |
| CI 的 Linux / Windows 上 19 个 ComfyUI 测试 error：`AssertionError: Torch not compiled with CUDA enabled` | ComfyUI 的 `cpu_state` 默认是 `CPUState.GPU`，只有 `--cpu` 或检测到 MPS 时才改；CPU 版 torch 两个都不满足，`get_torch_device()` 落到 `torch.cuda.current_device()` | `tests/conftest.py` 在任何人 import `model_management` 之前设 `comfy.cli_args.args.cpu = True`（顺带让三个平台跑同一个设备） | [comfyui_node](features/comfyui_node.md) |
| 删掉后端之后 11 个集成测试一直 fail，但没人发现 | 上面那条让 CI 早就是红的，红色里多 11 个 fail 看不出来；集成测试还在传已经删掉的 `backend='torch'` | 集成测试改用 `tests/unit/reference_backend.py`，不再依赖某个后端存在；**CI 红的时候先修 CI** | [comfyui_node](features/comfyui_node.md) |
| ComfyUI 进程在第一次稀疏调用时 "Fatal Python error: Aborted"（Windows RTX 5070 实测） | ComfyUI 0.38 的 comfy-aimdo malloc graph 要求 block 内分配的显存在 block 结束前释放；Veda 的缓存（tile 布局、head 组、统计）和 kernel workspace 是在 block 内分配的 | 稀疏路径整体包在 `comfy.model_prefetch.pause_malloc_graph()` 里 | [comfyui_node](features/comfyui_node.md) |
| CI 里 `actions/checkout` 报 `remote: Repository not found`，而仓库名是对的 | workflow 声明了 `permissions:` 块，没列出的 scope 全变 `none`，默认 token 没了 `contents` 权限 | 声明权限时把 `contents: read` 一起写上 | [packaging_release](features/packaging_release.md) |
| 改一行 description 就触发了一次 registry 发布 | 发布 workflow 的触发条件是 `paths: pyproject.toml`，不是版本号变化 | job 里和 `HEAD~1` 比一次 `version`，不同才发布 | [packaging_release](features/packaging_release.md) |
| 节点里有 `__import__(...)` 行内调用和 `subprocess.run` | 本意只是懒得写顶部 import、想要个好看的 CPU 名字，但这两种形状正是 registry 安全审核要找的东西 | 顶部 `import sys`；subprocess 整个删掉，用 `platform.processor()` | [packaging_release](features/packaging_release.md) |
| 用户的 `HF_TOKEN` 会被发给 `HF_ENDPOINT` 指定的镜像站 | 下载时无条件附带 Authorization 头，而 README 主动建议设 `HF_ENDPOINT=hf-mirror.com` | 只在 endpoint 的 host 是 `huggingface.co`（或其子域）时才带 token | [packaging_release](features/packaging_release.md) |
| registry 把版本标成 `Flagged`，五条 finding 全是 `info` | 扫描器是 YARA 字符串匹配：`mx.eval(` 被当成 Python 的动态执行，`os.environ[` 被当成凭据读取 | MLX 的 flush 走具名 helper `_flusher` 并删掉多余的那次，环境变量收到 `_settings()` 一处读；网络和环境那两条是功能固有，留着向审核解释 | [packaging_release](features/packaging_release.md) |
| 接了 KJNodes "MiniMax H3 Low VRAM Attention" 后 Veda 显示 "the model has 14 heads ..., the predictor expects 56" 并关掉 | 它把 H3 注意力按 `minimax_head_chunks` 分组调用，Veda 每次只拿到一组头 | Veda 每步把分组请求接管过来：forward 一次交出全部头，Veda 不处理的调用由 Veda 自己按同样方式分组 | [comfyui_node](features/comfyui_node.md) |
| 接了 KJNodes "MiniMax H3 Mem Eff Sage Attention Patch" 后 Veda 只显示 ready，之后没有下文，渲染变慢 | 它用 object patch 整个替换 `attn.forward`，不经过 `optimized_attention`，override 收不到调用；Veda 没收到调用时什么也不说 | 在 Veda 之前：按 block 路由（稀疏层走 Veda，其余仍走它）；在之后：每步检查并警告；一次运行没收到调用就警告 | [comfyui_node](features/comfyui_node.md) |
| 开了 Veda 之后 ComfyUI 在 MLP 里 OOM，显卡越大反而占得越多 | 块大小取 `free // 8`，空闲显存越多块越大，Veda 把 ComfyUI 用来搬权重的显存占掉了（实测峰值 3.24 GB vs 全注意力 0.64 GB） | 再加一条与机器无关的上限：每块不超过该层输出缓冲的 1/6（约 H/6 个 head），峰值降到 1.31 GB 且不影响速度 | [comfyui_node](features/comfyui_node.md) |
| 节点说 "no sparse kernel works on RTX 3090 (SM86)"，但 SM86 明明在支持范围内 | 真正失败的是 Triton 第一次用 C 编译器构建 `cuda_utils.c`（portable 版 `python_embeded` 缺头文件和导入库），自检把它包成 BackendUnavailable，消息却写成"这张卡没有 kernel" | 区分"没有候选 kernel"（硬件不在范围内）和"有候选但没启动起来"（环境问题），后者由后端自己给出修复办法（`Backend.explain_failure`） | [comfyui_node](features/comfyui_node.md) |
| 两段式采样器（selflift-Avatar 等）第一段正常，第二段报 "Veda did not run ... 请把 Veda 放到它之后" | 第二段在另一个分辨率 / 分块上跑，layout 的 seq_len 对不上，100 个调用全被拒；而被拒的调用都记成 `other attention`，被摘要排除，于是一次有调用的运行看起来像"一个调用都没收到" | 带 `block_index` 的拒绝单独记成 `_WRONG_LAYOUT` 并计入摘要；"没收到调用"的警告改成只在 `run.calls` 为空时触发 | [comfyui_node](features/comfyui_node.md) |
| R2VA 打分器放进 `models/veda` 后节点报 `incomplete metadata ('keep_ratio')`，完全用不了 | 加载器硬要 `keep_ratio`，而 R2VA 的 bundle 改用 `target_budget_kind/value` + `ref_budget_kind/value`（32 tiles） | `bundle._budget` 同时认两种 schema；稀疏度默认值改成 `trained`，由 bundle 自己说训练在什么预算上 | [comfyui_node](features/comfyui_node.md) |
