# 核心：布局、tile、方案、打分与选择

## 目标

在 ComfyUI 的 H3 打包序列上，复现 Veda 训练时的块选择，结果与设备、后端无关。

## 设计与不变量

- **规则是契约**：tile 排列（`core/tiling.py`）、方案表（`core/plans.py`）、打分
  （`core/predictor.py`）和选择（`core/selection.py`）与 Miowtion 训练 / 搜索时完全一致，改动
  它们等于让打分器面对没见过的输入。改之前先和 Miowtion 对齐。
- **布局映射**（`core/h3_layout.py`）：ComfyUI 的打包顺序与 Miowtion 相同：
  `text | cond（关键帧/引导帧） | 参考 | 目标音频 | 目标视频`，没有 padding 行。
  - target = `video` 段，网格 `(latent_t, latent_h/2, latent_w/2)` 来自 `layout.signature`；
  - reference = `cond` 与 `ref_img` 段，网格从 `position_ids` 恢复，并逐行校验是 T,H,W 行优先；
    校验不过的段保持 global（全注意力，永远正确，只是慢一点），并记在 `LayoutSpec.skipped`；
  - 文本和所有音频段是 global 行。
- **tile**：每个 tile 128 行，tile 形状 (t,h,w) 的乘积为 128；真实行压在 tile 前部
  （`valid_count` 前缀），padding 槽位为 -1 / 0。reference 段用最少 padding 的形状
  （Miowtion 对 tiled conditions 的规则），target 用方案表里每层每头的形状（每层最多两种）。
- **方案选择**（按目标 token 网格）：纵横比最近（含 H/W 转置）→ 时长（latent_t）最近 → padding
  最少；完全匹配时自然选中。永远能选出方案；非精确匹配时在节点上说明。
- **选择规则**（见 `selection.py` 模块注释 1–7）：global 双向全保留；只有 video→video 象限
  稀疏；比例预算按等 kernel 代价 `ratio * n_ideal^2 / n_cols`；小数部分按 Bresenham 分摊；
  对角 tile 强制保留并占预算；空 tile 不选；reference 与 generated 两个列块各自 top-k、各自预算。
- **引擎**（`core/engine.py`）：每个头组 gather → pool → logits → select → mask → 后端 →
  scatter；按头分块，一份 tile 顺序副本不超过 `clamp(free/16, 64 MB, 1 GB)`（CUDA）或 256 MB。
  统计（保留比例）累加在设备上，运行结束才读一次，不在热路径上同步。

## 代码位置与接口

`veda_comfy/core/*`，不 import ComfyUI；`engine.VedaEngine.attention(q, k, v, layer, spec, plan)`，
q/k/v 为 `[S, H, D]` 视图。

## 测试

`tests/unit/test_core_*.py`：tile 前缀性质与顺序、gather/scatter 逐位往返、global 行、预算公式、
Bresenham 均值、对角强制、两个列块预算、方案选择三条规则、bundle 三种存储 dtype 与错误、
池化忽略 padding、未训练打分器等价于均值池化 QK、引擎全保留 = 全注意力、按头分块 = 逐头参考。

## 踩坑记录

（暂无）

## 待办

- R2VA / FL2VA 的 reference tiling 与 Ref2VA 权重下的打分器 recall 需要用真实权重测
  （打分器只在 FL2VA + Turbo 8 步 T2VA 上训练过）。
