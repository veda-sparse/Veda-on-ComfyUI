# 外部依赖

| 依赖 | 来源 | 版本 | 用途 | 必需？ |
|---|---|---|---|---|
| ComfyUI | 用户安装 | >= 0.38.0 | H3 模型、`minimax_h3_layout`、V3 节点 API（`advanced` 输入）、`send_progress_text` | 是 |
| torch / safetensors / numpy | ComfyUI 自带 | 随 ComfyUI | 全部 | 是（不由本仓库声明） |
| triton（Linux）/ triton-windows（Windows） | 随节点自动安装（`pyproject.toml` 的 environment marker） | `>=3.0`，已验证 triton-windows 3.8.0 | `triton-int8` 后端的 kernel | CUDA 上是 |
| mlx | 随节点自动安装（仅 Darwin + arm64） | 已验证 0.32.3 | `mlx` 后端（Apple silicon） | Apple 上是 |

依赖写在 `pyproject.toml` 的 `dependencies` 里并带 environment marker，所以从 Comfy Registry
装节点时 kernel 一起装好，用户不需要跑任何脚本。Linux 上 torch 本来就会带同一个 `triton`，
所以那一条多数情况下是空操作。

SageAttention **不是**运行时依赖：我们复刻的是它的算术，不是它的包。`tools/compare_int8.py`
在它装着的时候会额外和真包对一次，没装就只和 torch 模型比。

ComfyUI 内部接口：`transformer_options` 的 `optimized_attention_override`、`block_index`、
`minimax_h3_layout`（`segments`、`signature`、`seq_len`、`position_ids`）、`sigmas`、
`sample_sigmas`；`ModelPatcher.add_callback_with_key`（`ON_PREPARE_STATE`、`ON_CLEANUP`）；
`folder_paths.folder_names_and_paths`；`comfy.utils.ProgressBar`；
`PromptServer.send_progress_text`。
