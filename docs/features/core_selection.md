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

## 选块策略：fixed 与 adaptive

- **fixed（默认）**：每个 query tile 保留固定数量的 key tile，默认 **32 个**。以前默认是
  等代价比例（90% 稀疏），但保留数会随网格面积一起缩放：双采工作流的第一段跑在训练网格的
  0.33 倍上，90% 稀疏只剩 5.5 个 tile。5.5 不是整数，余数由 Bresenham 在行之间摊，而 tile
  顺序里 t 块变化最快（`tiling.span_tiles` 的 `permute(2,4,0,1,3,5)`），**相邻行就是时间上
  相邻的块**——于是每隔一个时间块上下文就多 20%，约 1.1 秒一个周期，看起来就是闪烁。
  绝对数量不随网格变化，没有余数。
- **adaptive**：抄 ComfyUI Sol-Attn 节点的规则和参数（`tau`，默认 1.3，范围 0–4，步进
  0.05）。阈值取该行打分分布的 `tau` 个标准差，保留高过阈值的 tile，数量随内容和尺寸变化。
  在高斯打分上实测与上游 tooltip 一致：tau 1.0 保留 16.0%、1.5 保留 6.8%、2.0 保留 2.4%
  （上游写的是 ~16% / ~7% / ~2.7%）。
  - 与 Sol-Attn 的两点不同，都是有意的：**不加它的三宽带**（`|i-j| <= 1`），只保留 Veda 自己
    的规则 2 对角线，免得偏离打分器训练时的选择规则；**加一个下限 `_ADAPTIVE_FLOOR`**，因为
    Sol-Attn 对没选中的块还有一个 pooled 项兜底，而 Veda 是直接跳过——打分平坦的行在纯阈值下
    会只剩对角线。

## Sol 的池化误差修正（参考实现）

`reference.pooled_correction_attention`：每个被跳过的 tile 仍然贡献一项，用它的 `mean(K)` 和
`sum(V)` 构成，于是 softmax 的分母仍然看得到整条序列。算法取自 Sol-Attn
（arXiv 2607.24027，以及 comfy_kitchen 的 `sol_attn`）：**用未归一化的 exp，池化项在分母里按
它代表的真实行数加权**（`v_sum` 已经是该 tile 的和，所以分子不用再乘）。

目前只有 fp32 参考实现，用来回答"值不值得写进 kernel"。随机打分下的相对 L2（32 个 tile）：

| 每行预算 | 直接跳过 | 加池化项 | 降低 |
|---|---|---|---|
| 1 | 5.647 | 0.246 | 95.6% |
| 4 | 2.689 | 0.234 | 91.3% |
| 16 | 1.000 | 0.178 | 82.2% |

**预算越紧收益越大**，而双采第一段正是预算最紧的地方。注意随机打分会夸大差距（训练好的打分器
选得准得多），但方向明确。

代价（每个 query tile 一行，预算 32）：池化项随 tile 总数走、exact 路径随预算走，所以
训练尺寸（594 tile）+14.5%，而双采第一段（72 tile）只有 **+1.8%**。K 的 tile 均值打分器本来
就要算（`predictor.pool_video_tiles` 的 mean），新增的只有每个 tile 的 `sum(V)`。

有了它之后 `_ADAPTIVE_FLOOR` 就不需要了：打分平坦的行由池化项兜底，和 Sol-Attn 一样，
adaptive 退化成纯 tau 阈值、与 budget 完全无关。**kernel 实现与这一步都还没做。**

## 测试

`tests/unit/test_core_*.py`：tile 前缀性质与顺序、gather/scatter 逐位往返、global 行、预算公式、
Bresenham 均值、对角强制、两个列块预算、方案选择三条规则、bundle 三种存储 dtype 与错误、
池化忽略 padding、未训练打分器等价于均值池化 QK、引擎全保留 = 全注意力、按头分块 = 逐头参考。

## 踩坑记录

（暂无）

## 待办

- R2VA / FL2VA 的 reference tiling 与 Ref2VA 权重下的打分器 recall 需要用真实权重测
  （打分器只在 FL2VA + Turbo 8 步 T2VA 上训练过）。
