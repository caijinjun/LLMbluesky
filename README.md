# ATC HMI BlueSky Visual Collaboration Package

面向项目组协作的 BlueSky 间隔管理 / 冲突解脱可视化系统整理版。

本仓库包含：

- 可运行的 BlueSky GUI/HMI 演示工程
- 动态扇区间隔管理场景
- 安全约束冲突解脱模块
- LLM 解释层 / OpenAI-compatible mock server
- Headless smoke validation 入口与结果目录
- 项目组二次开发所需脚本与文档

> 说明：本仓库是协作精简版，已排除运行缓存、大批量 headless 日志、`__pycache__` 和部分大尺寸可再生成资源，便于 GitHub 维护。

仓库保留首次启动所需的 `bluesky_project/data/navdata/apt.zip`。首次运行会自动生成本地导航缓存，耗时通常高于后续启动；生成的缓存不会提交到 Git。

## 当前版本（2026-07-22）

本次更新同步了当前验收演示版本，主要包括：

- `Three-route demo` 与 `Chengdu-Chongqing real` 两套可切换地图；
- 成都-重庆真实扇区的 10 条有向航路、38 个航路点和初始交通场景；
- 最多 14 架航空器的动态放行、全航空器对冲突监测和持续状态更新；
- `altitude_first` / `speed_first` 两种决策偏好；
- 可修改的最小水平间隔参数，默认 5 NM；
- FL270-FL390 范围内、以 1000 ft 为单位的高度候选动作，以及 250-330 kt 速度候选动作；
- 20 分钟前向预测、全航空器对安全验证和结构化指令记录；
- 本地模板解释器、OpenAI-compatible wrapper 和可选 Qwen 动作选择服务；
- 无界面快速回归入口及真实航路 100 轮验证说明。

详细验证结论见 [`docs/REAL_CHENGDU_100RUN_VALIDATION_20260722.md`](docs/REAL_CHENGDU_100RUN_VALIDATION_20260722.md)。

## 快速开始

### 1. GUI 演示

在 PowerShell 中运行：

```powershell
.\RUN_BLUESKY_GUI.ps1
```

进入 BlueSky 后：

1. 打开底部 `AI` tab。
2. 在 `Map` 中选择 `Three-route demo` 或 `Chengdu-Chongqing real`。
3. 点击 `Load map`，等待航路和初始场景加载完成。
4. 选择 `Preference`: `speed_first` 或 `altitude_first`。
5. 按需修改 `Min sep NM`，默认值为 5.0 NM。
6. `LLM wrapper` 保持 `template_explainer`。
7. 点击 `Start auto traffic`。

系统会生成动态交通、检测预测冲突、调用安全验证后的解脱动作，并在 HMI 中展示标准指令和解释。

### 2. GUI + Mock LLM API

PowerShell 窗口 1：

```powershell
.\RUN_MOCK_LLM.ps1
```

PowerShell 窗口 2：

```powershell
.\RUN_BLUESKY_GUI_WITH_MOCK_LLM.ps1
```

在 `AI` tab 中将 `LLM wrapper` 设置为 `openai_compatible_api`。

### 3. Headless smoke test

```powershell
.\RUN_HEADLESS_SMOKE.ps1
```

期望结果：`success: true`，`num_loss_events: 0`，`fallback_calls: 0`。

## 目录结构

```text
bluesky_project/       # BlueSky GUI/HMI 工程与插件
docs/                  # 设计文档
headless_validation/   # Headless 验证入口与结果目录
llm/                   # 本地 OpenAI-compatible mock server
RUN_*.ps1              # Windows PowerShell 启动脚本
README_DEMO.md         # 原始演示说明
```

## 最新系统流程文档

更详细的系统流程、模块边界、安全验证逻辑、LLM wrapper 约束和二次开发入口见：

```text
docs/SYSTEM_FLOW_DETAILED.md
docs/SYSTEM_PIPELINE_DEEP_DIVE.md
```

其中 `SYSTEM_FLOW_DETAILED.md` 偏系统总览，`SYSTEM_PIPELINE_DEEP_DIVE.md` 偏工程交接，逐阶段说明输入、处理、输出、关键函数、关键参数和调试路径。

当前版本的核心流程是：

```text
BlueSky 动态交通
	↓
ACDATA / synthetic fallback 获取飞机状态
	↓
CPA 预测冲突检测
	↓
构建冲突图
	↓
生成 hold / altitude / speed 候选动作
	↓
20 min 前向安全验证
	↓
离散约束搜索选择 verified actions
	↓
必要时 altitude reversal fallback / recovery altitude plan
	↓
LLM wrapper 只生成解释和标准话术
	↓
下发 verified BlueSky ALT / SPD 命令
	↓
HMI 表格、颜色、执行状态和 JSONL 日志更新
```

## 设计边界

LLM 层只负责解释和标准话术包装，不绕过安全验证。系统流程为：

1. 本地安全求解器搜索速度/高度/保持等候选动作；
2. 安全验证器过滤不可行动作；
3. GUI/HMI 展示已验证指令；
4. LLM wrapper 只把已验证动作转成结构化说明、标准话术和简短理由。

最终安全证据应优先参考 `headless_validation/results/` 下的 headless 验证结果。GUI fallback 只用于演示连续性。

## 当前验证状态

2026-07-22 在成都-重庆真实航路场景上完成 100 轮无界面回归：

- `altitude_first` 50 轮、`speed_first` 50 轮；
- 每轮 1200 秒、最多 14 架航空器；
- 100/100 轮完成，记录到的失去间隔事件为 0；
- fallback 调用为 0；
- 共生成 208 条解脱指令，高度指令 152 条、速度指令 56 条。

该结果是工程回归证据，不构成形式化安全证明。当前已知限制是 `speed_first` 在个别高负载种子中可能连续调整同一航空器的目标速度；后续 UI/策略改进应保留种子 `2026082329` 作为固定回归用例。

## 协作建议

- 新增 HMI 功能优先放在 `bluesky_project/plugins/` 或相关 UI 模块中。
- 新增验证脚本放在 `headless_validation/`。
- 新增 LLM wrapper 或 mock API 放在 `llm/`。
- 不要提交运行日志、缓存、模型权重或大规模生成数据。
- 大数据集建议放 release artifact、网盘或 Git LFS。

## 依赖

BlueSky 原工程依赖见：

```text
bluesky_project/requirements.txt
```

建议使用 Python 虚拟环境安装依赖后运行。
