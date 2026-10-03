# 安装、打包与发布

## 目标

用户通过 ComfyUI Manager / Comfy Registry 一键安装，装完就能用最快的 kernel，不需要跑任何脚本；
维护者改一次版本号就能发布。

## 设计与不变量

- **依赖随节点装好**：`pyproject.toml` 的 `dependencies` 用 environment marker 声明
  `triton`（Linux）、`triton-windows`（Windows）、`mlx`（Darwin + arm64），所以装节点就把 kernel
  装好了。`requirements.txt` 列同样的内容，给从 clone 安装的人。
  - 这在以前做不到：FA4 的运行时（CuTe DSL 等）在不同 CUDA 版本下包名都不一样，必须用一个
    `install_fa4.py` 脚本按需安装，还要用 constraint 把 torch 锁住以免替换用户的 CUDA 版 torch。
    换成 Triton 之后依赖是单个包、有 Windows wheel、Linux 上 torch 本来就带，marker 就够了，
    那三个安装脚本和 `requirements-fa4.txt` 一起删掉了。
  - 注意 Manager 装 requirements 时只要有一个包装不上就整体失败，所以每一条都必须带 marker，
    不能让 macOS 去装 triton。

- **Comfy Registry**：`pyproject.toml` 的 `[project]`（name `veda-sparse-attention`、version、
  license、classifiers）和 `[tool.comfy]`（PublisherId、DisplayName、Icon、
  `requires-comfyui >= 0.38.0`）。`.comfyignore` 排除 tests / tools / docs 等。
- **示例工作流**：`tools/make_example_workflows.py` 从 ComfyUI 官方 H3 模板派生（只插入 Veda
  节点、改分辨率到训练尺寸、T2VA 打开 8 步 Turbo LoRA、加说明），放在 `example_workflows/`，
  ComfyUI 会把它们列在模板浏览器的本节点分类下；节点的 `properties.models` 带打分器下载地址，
  缺失模型对话框可以一键下载。
- **CI**（`.github/workflows/tests.yml`）：Linux / Windows / macOS 上安装 ComfyUI v0.38.2 +
  CPU torch，跑 `ruff` 与 `tests/unit`（含 ComfyUI 集成测试）。
- **发布**（`.github/workflows/publish_action.yml`）：main 上 `pyproject.toml` 变化时用 secret
  `REGISTRY_ACCESS_TOKEN` 发布（流程见 AGENTS.md 第 4 节）。

## 踩坑记录

- **Registry 的 Icon 最大 400x400**：图标 SVG 的 width/height 设为 400（viewBox 不变）。

## 待办

- 在 registry.comfy.org 创建 publisher（与 `PublisherId` 一致）并在仓库配置
  `REGISTRY_ACCESS_TOKEN`。
