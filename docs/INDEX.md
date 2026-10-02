# docs 目录

本目录是项目知识库：本文件是目录，每个子功能一个文档。改代码时同步改对应文档；踩过的坑写进
feature 文档的「踩坑记录」，并在 `pitfalls.md` 加一行索引。

| 文档 | 内容 | 状态 |
|---|---|---|
| [features/comfyui_node.md](features/comfyui_node.md) | 节点、attention override 接入、用户交互（状态文字、回退、下载）、设置语义 | CPU 集成测试通过（ComfyUI 0.38）；真实 ComfyUI server 加载验证 |
| [features/core_selection.md](features/core_selection.md) | H3 布局映射、tile 排列、方案表选择、打分、generated / reference 预算与选择规则、引擎 | CPU 测试通过 |
| [features/backends.md](features/backends.md) | 后端契约、自检、候选顺序、各后端实现与隔离规则 | torch / mlx 在 CPU + MPS 通过；CUDA 后端见 hardware.md |
| [features/fa4_vendoring.md](features/fa4_vendoring.md) | FA4 私有副本的生成、补丁、Windows 兼容、升级流程 | 生成与静态检查通过 |
| [features/packaging_release.md](features/packaging_release.md) | 安装脚本、Comfy Registry 打包与发布、示例工作流、CI | 当前 |
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
