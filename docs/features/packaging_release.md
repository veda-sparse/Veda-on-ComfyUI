# 安装、打包与发布

## 目标

用户通过 ComfyUI Manager / Comfy Registry 一键安装，不需要额外依赖就能用；想要最快的 kernel
时运行一个脚本即可；维护者改一次版本号就能发布。

## 设计与不变量

- **运行依赖为零**：`requirements.txt` 不列任何包（Manager 安装 requirements 时只要有一个包装
  不上就整体失败）。FA4 运行时（CuTe DSL 等）在 `requirements-fa4.txt`，由 `install_fa4.py`
  按需安装。
- **install_fa4.py**（用跑 ComfyUI 的那个 Python 执行）：
  - NVIDIA：安装 `requirements-fa4.txt`，用 constraint 把 torch 锁在当前版本（绝不替换用户的
    CUDA 版 torch）；torch 是 CUDA 13 构建时（如 DGX Spark）换成 `nvidia-cutlass-dsl[cu13]`；
  - Apple silicon：安装 `mlx`；
  - 最后对每块 GPU 跑后端自检并打印选中的 kernel（`--check` 只自检）。
  - pip 不可用时退回 `uv pip install --python <当前 python>`。
- **install_fa4.bat** 按顺序找：便携版 `..\..\..\python_embeded\python.exe`、桌面版 / venv 的
  `..\..\.venv\Scripts\python.exe`、`..\..\venv\Scripts\python.exe`，最后 `python`；也可以把
  python.exe 路径作为第一个参数。**install_fa4.sh** 找 `../../.venv`、`../../venv`、
  `$VIRTUAL_ENV`，最后 `python3`。
- **Comfy Registry**：`pyproject.toml` 的 `[project]`（name `veda-sparse-attention`、version、
  license、classifiers）和 `[tool.comfy]`（PublisherId、DisplayName、Icon、
  `requires-comfyui >= 0.38.0`）。`.comfyignore` 排除 tests / tools / docs 等。
- **示例工作流**：`tools/make_example_workflows.py` 从 ComfyUI 官方 H3 模板派生（只插入 Veda
  节点、改分辨率到训练尺寸、T2VA 打开 8 步 Turbo LoRA、加说明），放在 `example_workflows/`，
  ComfyUI 会把它们列在模板浏览器的本节点分类下；节点的 `properties.models` 带打分器下载地址，
  缺失模型对话框可以一键下载。
- **CI**（`.github/workflows/tests.yml`）：Linux / Windows / macOS 上安装 ComfyUI v0.38.2 +
  CPU torch，跑 `ruff` 与 `tests/unit`（含 ComfyUI 集成测试）；另一个 job 跑
  `tools/vendor_fa4.py --check`。
- **发布**（`.github/workflows/publish_action.yml`）：main 上 `pyproject.toml` 变化时用 secret
  `REGISTRY_ACCESS_TOKEN` 发布（流程见 AGENTS.md 第 4 节）。

## 踩坑记录

- **Registry 的 Icon 最大 400x400**：图标 SVG 的 width/height 设为 400（viewBox 不变）。

## 待办

- 在 registry.comfy.org 创建 publisher（与 `PublisherId` 一致）并在仓库配置
  `REGISTRY_ACCESS_TOKEN`。
