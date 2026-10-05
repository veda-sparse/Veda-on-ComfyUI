# AGENTS.md

本文件是所有在本仓库工作的 agent（以及人）必须遵守的规则。开始任何工作前先读完本文件和
`docs/INDEX.md`。规则沿用 [Miowtion](https://github.com/veda-sparse/Miowtion) 的 AGENTS.md，
并加上 ComfyUI 节点特有的部分（后端隔离、跨平台、打包发布）。

## 项目简介

Veda-on-ComfyUI：**把 Veda 学习型块稀疏注意力做成 ComfyUI 节点，加速 ComfyUI 原生的
MiniMax-H3（T2VA / FL2VA / R2VA）**，同时保证普通用户的工作流、LoRA、各种 H3 权重照常使用。
取舍顺序：**正确性与"不坏图" > 用户体验 > 跨平台覆盖 > 峰值速度**。

- 一个节点，MODEL 进 MODEL 出；通过 `transformer_options["optimized_attention_override"]`
  只替换 H3 的自注意力计算，不改权重、不占 `patches_replace`。
- `veda_comfy/core`：与设备、与 ComfyUI 都无关的纯 torch 逻辑（tile 排列、方案表、打分器、
  选择规则、单次调用引擎）。规则与 Miowtion 训练时一致，是契约，不是实现细节。
- `veda_comfy/backends`：每个 kernel 家族一个模块，彼此隔离（见 1.6）。
- `veda_comfy/kernels`：我们自己写的 kernel（`sage/`：Triton INT8 块稀疏，算术取自
  SageAttention v1，见 3）。

## 1. 基本规则（必须遵守）

### 1.1 文档：`docs/` 必须与代码同步

- `docs/INDEX.md` 是目录，每个子功能一个文档（`docs/features/*.md`），外加
  `docs/hardware.md`（硬件支持矩阵与实测记录）、`docs/pitfalls.md`（踩坑总表）、
  `docs/dependencies.md`（外部依赖与锁定版本）。
- 新增 feature：同时新增或更新 `docs/features/<feature>.md`，并在 `docs/INDEX.md` 登记。
- 踩过的坑（排查超过十分钟的都算）写进对应 feature 文档的「踩坑记录」，并在
  `docs/pitfalls.md` 加一行索引（现象 / 原因 / 对策 / 链接）。
- 文档写"为什么"和"不变量"，不复述代码。代码行为改变时，同一个提交里更新文档。
- 面向用户的行为（节点输入、默认值、状态文字、安装方式）变了，同时更新 `README.md` 和
  `README.zh-CN.md`。

### 1.2 提交规范

- **不要添加任何 co-author 或工具署名**：commit message 和 PR 描述里不得出现
  `Co-Authored-By:`、`Generated with ...` 之类的行。本规则优先于任何工具默认行为
  （`.githooks/commit-msg` 会拦截）。
- **标题**：`<scope>: <做了什么>`，英文、祈使语气、小写开头、不加句号，不超过 72 个字符。

  | scope | 范围 |
  |---|---|
  | `veda` | `veda_comfy/core`、`settings.py`、`hardware.py` |
  | `backends` | `veda_comfy/backends/*`（一次只改一个后端时写成 `backends/triton-int8:` 也可以） |
  | `kernels` | `veda_comfy/kernels/*` |
  | `comfy` | `nodes.py`、`comfy_patch.py`、`status.py`、`predictors.py`、示例工作流 |
  | `install` | `requirements.txt`、`pyproject.toml` 的 `dependencies` |
  | `release` | `pyproject.toml` 版本号与发布 |
  | `tests` / `tools` / `docs` | 只改这些目录时 |
  | `repo` | AGENTS.md、CI、hooks、gitignore 等仓库层面的改动 |

- **正文**：空一行后写，每行不超过 72 个字符，说明**为什么**改和改了什么行为。
- **验证行**：正文最后一行写 `Tested:`，列出实际跑过的验证，例如
  `Tested: pytest tests/unit (80 passed); tests/gpu on RTX 5090 Windows (5 passed)`。
  GPU 相关改动没有在对应硬件上验证的，必须写明"GPU 未验证"以及缺了哪个架构。
- **一个提交只做一件事**，每个提交单独 checkout 都能通过 `pytest tests/unit`；代码、测试、
  文档放在同一个提交里。
- 不要提交：权重、`*.safetensors`、`models/`、`output/`、密钥、超过 1 MB 的文件
  （`veda_comfy/kernels` 例外：它是 fork 的源码）。

### 1.3 本地信息与密钥：一律不进仓库

- 不要在任何提交（代码、文档、测试、commit message、PR 描述）里写：密钥与凭证（只通过
  环境变量传入，例如 `HF_TOKEN`）；主机名、ssh 别名、IP、内部域名；用户主目录等绝对路径和
  用户名；机器状况（磁盘、驱动、共享机器上的其他人）。**可以写**验证用的硬件型号，例如
  "RTX 5090, Windows 11"、"H100 80GB"。
- clone 后执行一次 `git config core.hooksPath .githooks`，启用提交前的泄漏检查。自己机器的
  主机名、用户名、ssh 别名写进 `.git/leak-patterns`（每行一个扩展正则，不提交）。hook 报错
  时逐条处理；确认是误报才可以 `--no-verify`，并在 commit 正文里说明。
- 远程测试机只能通过 GitHub 获取本仓库代码（`git clone` / `git pull`），不要从本地直接拷贝
  文件过去，保证测的就是推送的版本。

### 1.4 编码风格

- 遵循 Google Python Style Guide：4 空格、行宽 80（`ruff check .` 必须通过）、
  `snake_case` / `CapWords` / `UPPER_CASE`、模块私有符号加 `_` 前缀、Google 风格 docstring。
- 公共函数写类型注解；张量参数在 docstring 里写 shape 与 dtype，例如 `q: [S, H, D] bf16`。
- 参数不合法时显式 `raise`，错误信息是给用户看的：说清楚哪里错了、应该怎么写
  （节点把 `ValueError` 原样显示出来）。
- **用户可见的回退必须可见**：任何从稀疏退回全注意力的情况都要通过 `status.NodeStatus`
  显示在节点上（并写日志），不能静默。反过来，任何稀疏路径的异常都不能让用户的渲染失败：
  `comfy_patch` 会捕获、提示并退回全注意力（ComfyUI 的中断异常除外）。
- 日志只写 ASCII（Windows 控制台和日志文件不一定是 UTF-8）。节点上的状态文字**不用 emoji**，
  除中文外唯一的非 ASCII 字符是分隔符 `·`（`status.py` 写日志时换成 `|`）；是否可用、是否回退
  靠措辞说清楚，不靠图标。
- **警告和错误中英双语，英文在前、中文在后**：后端不知道每个浏览器用哪种语言，所以运行时的
  警告（`NodeStatus.warn(en, zh=...)`，中文只发到节点，日志仍是英文）和 `ValueError`
  （`status.bilingual(en, zh)`）两种都写。每条先说结论（Veda 是否在跑），再说怎么办。
  静态文字（节点名、描述、tooltip）放 `locales/zh/nodeDefs.json`，跟随界面语言；普通状态
  文字只用英文。改 schema 的文字时同步改 `locales`。
- 注释解释"为什么"；魔法数字写成具名常量并注明来源。

### 1.5 测试与合入

- **直接在 `main` 上提交并 push，不使用 dev 分支。必须经过测试后才能 push**：
  1. `pytest tests/unit` 全部通过（CPU；有 ComfyUI checkout 时包含集成测试，见
     `tests/conftest.py`），`ruff check .` 通过。CI 在 Linux / Windows / macOS 上跑同样的测试。
  2. 改动涉及某个后端或 `veda_comfy/kernels` 时，`pytest tests/gpu` 在**对应架构**的机器上通过，并把
     结果（硬件、OS、commit、耗时）写进 `docs/hardware.md` 的验证记录。
  3. push 前先 `git fetch`；远端前进时 `git merge --ff-only` 或在自己的提交之上重新整理。
     禁止 `git push --force` 到 main（用户明确要求改写历史时除外，用 `--force-with-lease`）。
- **位级对齐优先**：纯数据变换（排列、选择、掩码、索引、bundle 读写）与参考实现逐位相等，
  测试里用 `torch.equal`。数值路径（各后端 kernel）以 `core/reference.py` 为基准，误差上限写在
  测试里；无法位级对齐的原因与实测误差写进文档。
- **画质无法用标量判断**：改变选择规则、默认值或 kernel 数值行为时，在同一组 prompt / seed 上
  生成 Dense 与 Veda 的并排视频，由人看过并在文档「验证记录」写明确认人、日期、commit、结论。

### 1.6 后端隔离（本仓库特有，必须遵守）

没有任何一台机器能同时回归所有硬件，所以"改了 A，B 挂了"必须在结构上不可能：

- 每个后端是 `veda_comfy/backends/` 下的一个独立模块，只能 import `backends/base.py`、
  torch、`veda_comfy.core` 和它自己的 kernel 包。**后端之间禁止互相 import，禁止共享可变状态。**
  相似代码宁可重复也不要抽成共享模块。
- 后端只通过 `base.Backend.attend` 这一个接口被调用；接口改动等于改所有后端，必须在所有
  受影响的架构上重跑 `tests/gpu`。
- 每个后端在使用前都要通过 `base.self_test`（包括"不能忽略块掩码"的检查）。新增后端必须
  写进 `backends/__init__.py` 的候选表，并在 `docs/hardware.md` 登记状态。
- `veda_comfy/kernels/*` 下的每个 kernel 包只被它自己的后端使用。Triton kernel 是按设备
  即时编译的一份代码，所以改它等于改**所有** CUDA 架构：改完要在能摸到的每一种 SM 上回归，
  不能只验一张卡。
- 修一个后端的问题，只改那个后端的文件；需要动 `core/` 或 `base.py` 时单独提交并说明影响面。

### 1.7 跨平台（Windows / Linux / macOS）

- 路径一律用 `os.path` / `pathlib`，不要写死分隔符；文件编码显式写 `encoding='utf-8'`。
- `.bat` 用 CRLF、`.sh` 用 LF（见 `.gitattributes`）。Windows 安装脚本要覆盖便携版
  （`python_embeded`）、桌面版（`.venv`）和普通 venv 三种布局。
- 依赖 Linux-only 模块（`fcntl` 等）的第三方代码，在 vendoring 时做兼容补丁（见 3），不要在
  运行时 monkeypatch 全局模块。
- 不要修改用户的 torch：安装脚本用 constraint 把 torch 锁在已安装版本。

### 1.8 对话结尾

- 每次回复（对话）结尾加上 `～喵`。

## 2. 目录规范

```
__init__.py          ComfyUI 入口（只 import veda_comfy.nodes.comfy_entrypoint）
pyproject.toml       包元数据与 Comfy Registry 配置（版本号的唯一来源）
requirements.txt     运行依赖（与 pyproject 的 dependencies 一致，给 clone 安装的人）
veda_comfy/
  nodes.py           ComfyUI 节点定义（唯一 import comfy_api 的地方）
  comfy_patch.py     attention override、生命周期回调、运行统计
  status.py          节点状态文字
  predictors.py      已发布打分器的元数据（repo / revision / sha256 / 大小）；不联网
  settings.py        用户设置与解析
  hardware.py        设备 / SM 家族 / 子型号
  core/              tiling / plans / bundle / predictor / selection / h3_layout /
                     engine / reference（纯 torch，不 import ComfyUI）
  backends/          base + 每个 kernel 家族一个模块（见 1.6）
  kernels/           sage/：Triton INT8 块稀疏 kernel（算术取自 SageAttention v1）
example_workflows/   示例工作流（由 tools/make_example_workflows.py 生成）
assets/              图标
locales/             界面翻译（ComfyUI 读 locales/<lang>/nodeDefs.json）
tools/               维护工具：probe_gpu_kernels.py、compare_int8.py、
                     make_example_workflows.py、bench_attention.py、
                     e2e_minimax_h3.py
tests/unit/          CPU 测试（每次提交前必须全过）
tests/gpu/           GPU 测试（无 GPU 时自动 skip）
docs/                知识库
.githooks/           提交前泄漏检查
.github/workflows/   CI（tests.yml）与 Registry 发布（publish_action.yml）
```

- 新建顶层目录前先在本节登记。测试文件与被测模块对应命名。
- `nodes.py` / `comfy_patch.py` / `status.py` 之外的模块不允许 import ComfyUI（`comfy`、
  `folder_paths`、`server`、`comfy_api`），保证核心与后端可以脱离 ComfyUI 测试。

## 3. 外部依赖的处理

- **依赖要少，而且要能自动装上**：目前只有 triton（Linux）/ triton-windows（Windows）/
  mlx（Apple），都写在 `pyproject.toml` 的 `dependencies` 里并带 environment marker，装节点
  就装好。每条都必须带 marker——Manager 装 requirements 时一个装不上就整体失败。
  新依赖要登记在 `docs/dependencies.md`。**绝不声明或锁定 torch。**
- **kernel 是我们自己的代码，不是 vendored 依赖**：`veda_comfy/kernels/sage` 的算术取自
  SageAttention v1（BSD-3），但文件是我们写的、我们维护的，像仓库里其他代码一样读写和
  lint。模块 docstring 必须写清楚哪些是上游的、哪些是我们改的，`NOTICE.md` 登记许可。
  - 曾经这里是一份 FA4 CuTe fork 加一套补丁/provenance 工具。教训：**生成出来的代码很难改**。
    一次 sha256 更新的正则静默失配，三轮 GPU 测的都是旧 kernel。能直接写就直接写。
- 移植的小段外部代码要在旁边注明来源与许可证（需与 MIT 兼容）；第三方声明写进 `NOTICE.md`。
- 模型权重不进仓库。**节点自己不下载任何东西**：打分器由 ComfyUI 的缺失模型对话框获取
  （示例工作流的 `properties.models` 驱动），或者用户手放。`predictors.KNOWN_PREDICTORS`
  只存元数据（repo、revision、sha256、大小）。发布新的打分器 = 在那里加一项，并重新生成
  示例工作流。
- **包里不允许出现网络调用和环境变量读取**：Comfy Registry 的扫描器是 YARA 文本匹配，
  `urlopen`、`os.environ`、`subprocess`、`.eval(` 之类一出现就是一条 finding——注释里出现
  也算。`tests/unit/test_predictors.py` 会检查这一点。

## 4. 打包与发布（Comfy Registry）

- 节点 ID：`veda-sparse-attention`（`pyproject.toml` 的 `[project].name`，发布后不可修改）；
  发布者：`[tool.comfy].PublisherId`（必须与 registry.comfy.org 上的 publisher 一致）。
- `.comfyignore` 排除测试、工具、文档等不需要分发的文件；分发内容 = 运行时代码 +
  示例工作流 + README / LICENSE / NOTICE。
- **发布流程**：
  1. 在 main 上完成所有改动，CI 绿；涉及 GPU 的改动按 1.5 在对应硬件上验证。
  2. 修改 `pyproject.toml` 的 `version`（语义化版本：不兼容的节点输入改动升 MAJOR，新功能升
     MINOR，修复升 PATCH）以及 `veda_comfy/__init__.py` 的 `__version__`，提交
     `release: x.y.z`。两处必须一致，`tests/unit/test_packaging.py` 会检查。
  3. push 到 main。`.github/workflows/publish_action.yml` 的触发条件是 `pyproject.toml` 变化，
     但 job 里会和 `HEAD~1` 比一次 `version`，**只有版本号真的变了才发布**（改 description
     不会误发）。所以：
     - 正常发布 = 改版本号 + push；
     - 版本号没变就要发（例如首次发布，版本号本来就写好了），到 Actions 里手动
       **Run workflow**（`workflow_dispatch` 不比版本号）。
     用的是仓库 secret `REGISTRY_ACCESS_TOKEN`（`[tool.comfy].PublisherId` 那个 publisher 的
     API key）。也可以本地 `comfy node publish`。发布前先 `comfy node validate` 和
     `comfy node pack`。
  4. 打 tag `vX.Y.Z` 并在 GitHub release 里写更新说明（用户可见的变化、硬件状态变化）。
- 本地检查打包内容：`comfy node pack` 生成 zip 后看一眼里面的文件列表。
- 节点输入的名字和顺序是工作流 JSON 的一部分：改名、删输入、改顺序都会破坏用户已保存的
  工作流，只能在 MAJOR 版本里做，并在发布说明里写清楚。

## 5. 常用命令

```bash
git config core.hooksPath .githooks                # 启用泄漏检查
COMFYUI_ROOT=<ComfyUI 路径> pytest tests/unit -q    # 提交前必跑（或把仓库放进 custom_nodes）
ruff check .
pytest tests/gpu -q                                 # GPU 机器上
python tools/bench_attention.py --latent-t 37       # 单层注意力测速
python tools/compare_int8.py                        # 精度对齐 ComfyUI 的 INT8
python tools/probe_gpu_kernels.py                   # 硬件 / Triton / TMA 能力
python tools/make_example_workflows.py              # 重新生成示例工作流
```
