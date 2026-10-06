# Sol + Veda

## 目标

记录将实验中的 Sol + Veda 路径整理进 ComfyUI 节点时必须保持的算法契约。
这份文档不把 Miowtion 的独立评测 runner 或训练代码当作 ComfyUI 运行时依赖。

## 设计与不变量

Sol + Veda 是混合路径：Veda predictor 产生 reference/current visual block mask，
现有 Sage INT8 kernel 计算 selected blocks，Sol 对 skipped visual blocks 使用
pooled BF16 `mean(K)` / `sum(V)` 做零阶补偿。exact 和 approximate 分支必须在同一个
online softmax 的分子、分母中合并。

global rows/columns、padding、mandatory blocks 和音频条件保持 exact；Sol 不能把
音频 reference 放进 visual Top-K，也不能把 selected block 再次计入 approximate 分支。
Veda 的 128-token logical tile 与 Sage 的 64-token physical key block 必须明确转换，
并且每个 physical block 使用正确的 valid-token count。

## 代码位置与接口

ComfyUI 的实现位于独立 backend/kernel 模块：

- `veda_comfy/backends/sol_veda.py`：接收 Veda 的 block mask；
- `veda_comfy/kernels/sol_veda/`：复用 Sage INT8 quantization 与 exact block loop，
  增加 Sol pooled correction；
- `veda_comfy/core/engine.py`：保持 predictor、tile plan 和 mask 生成不变；
- `veda_comfy/nodes.py`：以显式 kernel 选项暴露 `Veda INT8` 和 `Sol + Veda`。

不得把 Miowtion 的训练脚本、独立生成 runner、模型权重或评测媒体复制到节点包中。
默认 `Veda INT8` 行为必须保持不变。

## 当前整理状态

该分支已将服务器评测中使用的 Sage INT8 + Sol 混合路径整理为 ComfyUI
backend，并在节点中增加 `Sol + Veda` 选项。它沿用 Veda 的 predictor、tile
plan 和 block mask；默认 `Veda INT8` 路径保持不变。

本次按用户要求未运行完整测试；GPU 数值验证和性能调优仍应在合并前完成。
