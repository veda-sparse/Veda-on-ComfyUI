# docs 目录

本目录是项目知识库：本文件是目录，每个子功能一个文档。改代码时同步改对应文档；踩过的坑写进
feature 文档的「踩坑记录」，并在 `pitfalls.md` 加一行索引。

| 文档 | 内容 | 状态 |
|---|---|---|
| [features/comfyui_node.md](features/comfyui_node.md) | 节点、attention override 接入、用户交互（状态文字、回退、下载）、设置语义 | CPU 集成测试通过（ComfyUI 0.38）；真实 ComfyUI server 加载验证 |
| [features/core_selection.md](features/core_selection.md) | H3 布局映射、tile 排列、方案表选择、打分、generated / reference 预算与选择规则、引擎 | CPU 测试通过 |
| [features/backends.md](features/backends.md) | 后端契约、自检、容差、一个设备一个 kernel | mlx 在 MPS 通过；CUDA 见 hardware.md |
| [features/int8_kernel.md](features/int8_kernel.md) | Triton INT8 块稀疏 kernel：为什么是 INT8 不是 FP8、为什么没有 fallback、设计与性能 | RTX 5070 验证通过；TMA 待做 |
| [features/packaging_release.md](features/packaging_release.md) | 依赖自动安装、Comfy Registry 打包与发布、示例工作流、CI | 当前 |
| [hardware.md](hardware.md) | 硬件支持矩阵与验证记录 | 持续更新 |
| [dependencies.md](dependencies.md) | 外部依赖、锁定版本 | 当前 |
| [pitfalls.md](pitfalls.md) | 踩坑总表 | 持续更新 |

## feature 文档模板

```markdown
# <功能名>

## 目标
## 设计与不变量        （为什么这样做；哪些性质不能被破坏）
## 代码位置与接口
## 测试               （unit / gpu，各测什么）
## 踩坑记录           （现象 / 原因 / 对策）
## 验证记录           （日期、硬件、commit、结果）
## 待办
```
