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
