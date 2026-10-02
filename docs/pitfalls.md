# 踩坑总表

| 现象 | 原因 | 对策 | 详情 |
|---|---|---|---|
| FA4 在 Windows 上 import 失败（`No module named 'fcntl'`） | FA4 的 `cache_utils.py` 与 QuACK 的 `cache/*.py` 顶层 `import fcntl` | FA4 与 QuACK 都 vendor，`fcntl` 改为可选 | [fa4_vendoring](features/fa4_vendoring.md) |
| install_fa4 装完后自检说 `No module named 'cutlass'` | 同一个进程里刚装的包被 import 缓存挡住 | 安装后在新的子进程里跑自检 | [packaging_release](features/packaging_release.md) |
| 上游 FA4 在 SM8x 上"块稀疏"其实算的是 dense | SM80 kernel 收到块稀疏参数但不使用 | 用打过补丁的副本；自检检查"不是 dense" | [backends](features/backends.md) |
| 生成的 vendored 文件语法错误（两行粘连） | import 改写正则吞掉了行尾换行 | 保留结尾空白；对生成文件做 `ast.parse` 测试 | [fa4_vendoring](features/fa4_vendoring.md) |
| pytest 报 "attempted relative import with no known parent package" | pytest 把仓库根的 `__init__.py` 当包导入 | 根 `__init__.py` 只在 `__package__` 非空时 import | [comfyui_node](features/comfyui_node.md) |
| 测试里的 H3 前向报 "in-place RoPE ... do not support autograd" | 测试模型参数 `requires_grad=True` | `requires_grad_(False)` | [comfyui_node](features/comfyui_node.md) |
| 本地 HTTP 测试服务器启动 35 秒 | `server_bind` 的反向 DNS | 覆写 `server_bind` | [comfyui_node](features/comfyui_node.md) |
| ComfyUI 进程在第一次稀疏调用时 "Fatal Python error: Aborted"（Windows RTX 5070 实测） | ComfyUI 0.38 的 comfy-aimdo malloc graph 要求 block 内分配的显存在 block 结束前释放；Veda 的缓存（tile 布局、head 组、统计）和 kernel workspace 是在 block 内分配的 | 稀疏路径整体包在 `comfy.model_prefetch.pause_malloc_graph()` 里 | [comfyui_node](features/comfyui_node.md) |
