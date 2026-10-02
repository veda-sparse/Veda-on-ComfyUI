# 外部依赖

| 依赖 | 来源 | 版本 | 用途 | 必需？ |
|---|---|---|---|---|
| ComfyUI | 用户安装 | >= 0.38.0 | H3 模型、`minimax_h3_layout`、V3 节点 API（`advanced` 输入）、`send_progress_text` | 是 |
| torch / safetensors / numpy | ComfyUI 自带 | 随 ComfyUI | 全部 | 是（不由本仓库声明） |
| FlashAttention-4（CuTe） | vendored（`veda_comfy/_vendor`） | 4.0.0b32，wheel sha256 锁定 | fa4-* 后端 | 随仓库分发 |
| nvidia-cutlass-dsl | pip（`requirements-fa4.txt`） | ==4.8.0（CUDA 13 用 `[cu13]`） | FA4 的 JIT 编译与运行时；4.8.0 是第一个有 Windows wheel 的版本 | 可选 |
| QuACK（quack-kernels） | vendored（`veda_comfy/_vendor/quack`） | 0.6.5，wheel sha256 锁定 | FA4 依赖（Windows 兼容补丁） | 随仓库分发 |
| apache-tvm-ffi / torch-c-dlpack-ext | pip | `>=0.1.12,<0.2` / 最新 | FA4 依赖 | 可选 |
| triton / triton-windows | 随 Linux torch / pip | 与 torch 匹配 | flex 后端 | 可选 |
| mlx | pip | 已验证 0.32.3 | mlx 后端（Apple silicon） | 可选 |

我们依赖的 FA4 内部接口：`interface.flash_attn_func`（`mask_mod`、`aux_tensors`、
`block_sparse_tensors`）、`block_sparsity.BlockSparseTensorsTorch` / `DenseBlockMaskTorch`
（补丁新增）、`utils.scalar_to_ssa`、`interface._get_fwd_config` 与 `FwdConfig.q_stage`
（仅 fa4-sm100 的 q_stage 开关；签名不符时该后端报不可用）。

ComfyUI 内部接口：`transformer_options` 的 `optimized_attention_override`、`block_index`、
`minimax_h3_layout`（`segments`、`signature`、`seq_len`、`position_ids`）、`sigmas`、
`sample_sigmas`；`ModelPatcher.add_callback_with_key`（`ON_PREPARE_STATE`、`ON_CLEANUP`）；
`folder_paths.folder_names_and_paths`；`comfy.utils.ProgressBar`；
`PromptServer.send_progress_text`。
