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

## 快速开始

### 1. GUI 演示

在 PowerShell 中运行：

```powershell
.\RUN_BLUESKY_GUI.ps1
```

进入 BlueSky 后：

1. 打开底部 `AI` tab。
2. 点击 `Reset sector`。
3. 选择 `Preference`: `speed_first` 或 `altitude_first`。
4. `LLM wrapper` 保持 `template_explainer`。
5. 点击 `Start auto traffic`。

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

## 设计边界

LLM 层只负责解释和标准话术包装，不绕过安全验证。系统流程为：

1. 本地安全求解器搜索速度/高度/保持等候选动作；
2. 安全验证器过滤不可行动作；
3. GUI/HMI 展示已验证指令；
4. LLM wrapper 只把已验证动作转成结构化说明、标准话术和简短理由。

最终安全证据应优先参考 `headless_validation/results/` 下的 headless 验证结果。GUI fallback 只用于演示连续性。

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
