# BlueSky 间隔管理可视化系统详细流程说明

本文档说明当前仓库中 BlueSky 间隔管理 / 冲突解脱可视化系统的最新流程、模块边界、运行方式和二次开发入口。

## 1. 系统定位

本系统是一个面向空中交通管制（ATC）人机协同决策的 BlueSky 可视化原型。它在 BlueSky QtGL GUI 中增加一个 `AI Decision Assist` 面板，用于：

1. 动态生成多航路、多机扇区交通；
2. 实时读取 BlueSky 飞机状态；
3. 预测未来 CPA 和间隔风险；
4. 基于本地安全验证器搜索速度 / 高度解脱动作；
5. 将 verified actions 转换为 BlueSky 指令；
6. 使用 LLM wrapper 生成标准话术和解释；
7. 在 HMI 中显示冲突、命令、执行状态和安全状态；
8. 输出 JSONL 日志供复盘和后续模型训练。

核心安全边界：**LLM 不直接控制飞机，也不直接生成未经验证的 BlueSky 命令。所有下发到 BlueSky 的动作都来自本地 verifier 通过的候选动作。**

## 2. 关键文件

```text
bluesky_project/bluesky/ui/qtgl/aiassist.py       # HMI 面板、动态交通、冲突检测、求解器、LLM wrapper
bluesky_project/bluesky/ui/qtgl/mainwindow.py     # QtGL 主窗口，挂载 AI tab
bluesky_project/bluesky/ui/qtgl/radarwidget.py    # ACDATA stream 与雷达显示数据
RUN_BLUESKY_GUI.ps1                               # 启动 GUI demo
RUN_BLUESKY_GUI_WITH_MOCK_LLM.ps1                 # 启动 GUI demo 并配置 mock LLM endpoint
RUN_MOCK_LLM.ps1                                  # 启动 OpenAI-compatible mock server
RUN_HEADLESS_SMOKE.ps1                            # Headless smoke test
docs/LLM_VISUAL_DEMO_DESIGN.md                    # 原设计说明
docs/SYSTEM_FLOW_DETAILED.md                      # 本详细说明
headless_validation/                              # Headless 验证入口和结果
llm/                                              # Mock LLM API
```

## 3. 快速运行

### 3.1 GUI 演示

```powershell
.\RUN_BLUESKY_GUI.ps1
```

启动后，在 BlueSky 底部打开 `AI` tab：

1. 点击 `Reset sector`；
2. 选择 `Preference`：`altitude_first` 或 `speed_first`；
3. 可调整 `Min sep NM`；
4. `LLM wrapper` 默认选择 `template_explainer`；
5. 点击 `Start auto traffic`。

### 3.2 GUI + Mock LLM

PowerShell 窗口 1：

```powershell
.\RUN_MOCK_LLM.ps1
```

PowerShell 窗口 2：

```powershell
.\RUN_BLUESKY_GUI_WITH_MOCK_LLM.ps1
```

在 GUI 中将 `LLM wrapper` 切换为 `openai_compatible_api`。

### 3.3 Headless smoke test

```powershell
.\RUN_HEADLESS_SMOKE.ps1
```

期望结果：

```text
success: true
num_loss_events: 0
fallback_calls: 0
```

## 4. 最新版本识别标志

当前最新版本具备以下特征：

- GUI 标题：`AI Decision Assist - dynamic sector conflict detection`
- 主类：`AiAssistPanel`
- 动态扇区名称：`ATC_HMI_DYNAMIC_14AC_SECTOR`
- 最大飞机数：`MAX_AIRCRAFT = 14`
- 检测周期：`DETECT_INTERVAL_MS = 4000`
- 生成周期：`SPAWN_INTERVAL_MS = 30000`
- lookahead：`LOOKAHEAD_MIN = 20.0`
- 支持 `Min sep NM` GUI 动态调节
- 支持 LLM wrapper：`template_explainer`、`openai_compatible_api`、`off`
- 支持 `altitude_first` / `speed_first` 偏好
- 支持 altitude reversal fallback 和 recovery altitude plan
- 日志输出到：`bluesky_project/output/hmi_dynamic_logs/dynamic_sector_*.jsonl`

## 5. 总体流程

```text
启动 BlueSky GUI
    ↓
加载 AiAssistPanel 到底部 AI tab
    ↓
Reset sector 初始化动态扇区
    ↓
Start auto traffic 生成初始交通并启动定时器
    ↓
读取 BlueSky ACDATA / 必要时 synthetic fallback
    ↓
每 4 秒进行 CPA 冲突预测
    ↓
构建冲突图 conflict graph
    ↓
为相关飞机生成 hold / altitude / speed 候选动作
    ↓
对候选动作做 20 min 前向安全验证
    ↓
离散约束搜索选择 verified actions
    ↓
若失败，尝试 altitude reversal fallback
    ↓
若仍失败，尝试 altitude recovery graph coloring
    ↓
LLM wrapper 生成标准话术和解释
    ↓
下发 verified BlueSky ALT / SPD 命令
    ↓
HMI 更新冲突表、命令表、颜色、执行状态
    ↓
写入 JSONL 日志
```

## 6. 动态扇区与交通生成

系统定义了一个成都附近的轻量动态扇区：

```python
CENTER_LAT = 30.7000
CENTER_LON = 104.1000
ZOOM = 0.22
MAX_AIRCRAFT = 14
```

航路包括：

| Route | Direction | Entry | Exit | Heading | FL candidates |
|---|---|---|---|---|---|
| R1-EW | West → East | W_IN | E_IN | 90 | 320/340/360 |
| R1-WE | East → West | E_IN | W_IN | 270 | 320/340/360 |
| R2-NS | North → South | N_IN | S_IN | 180 | 330/350/370 |
| R2-SN | South → North | S_IN | N_IN | 0 | 330/350/370 |
| R3-SWNE | Southwest → Northeast | SW_IN | NE_IN | 45 | 310/330/350 |
| R3-NESW | Northeast → Southwest | NE_IN | SW_IN | 225 | 310/330/350 |
| R3-NWSE | Northwest → Southeast | NW_IN | SE_IN | 135 | 300/340/380 |
| R3-SENW | Southeast → Northwest | SE_IN | NW_IN | 315 | 300/340/380 |

### 6.1 Reset sector

`Reset sector` 会执行：

- `RESET`
- `HOLD`
- `PAN <center>`
- `ZOOM <zoom>`
- `BOX HMI_SECTOR,...`
- 清空内部状态和表格
- 新建 JSONL 日志

### 6.2 Start auto traffic

`Start auto traffic` 会：

1. 下发 `OP`；
2. 生成初始 4 架飞机；
3. 开启随机 spawn timer；
4. 开启 detect timer；
5. 延迟触发一次检测。

### 6.3 Entry gate

生成新飞机前，系统会调用 `_spawn_candidate_is_safe` 检查入口安全。

参数：

```python
ENTRY_LOOKAHEAD_MIN = 2.5
ENTRY_VERIFY_DT_SEC = 10
SPAWN_RETRY_LIMIT = 16
```

如果候选飞机在入口 2.5 分钟内会违反最小间隔，系统会延迟生成，而不是强行注入。

## 7. 状态来源

### 7.1 优先来源：BlueSky ACDATA

系统监听 BlueSky 的 `ACDATA` stream，读取：

- `id`
- `lat`
- `lon`
- `alt`
- `trk`
- `gs`
- `cas`

相关函数：

```python
on_simstream_received
_refresh_from_radarwidget
_set_latest_acdata
```

### 7.2 演示 fallback：synthetic meta

在短时 GUI 演示中，QtGL 的 ACDATA 可能延迟或稀疏。为了保持 HMI 工作流可见，系统提供：

```python
_synthetic_states_from_meta
```

它根据 route、heading、speed、flight level 和 spawn time 估算飞机状态。

注意：**synthetic fallback 仅用于 GUI 演示连续性。定量安全证据应以 headless validation 为准。**

## 8. 冲突检测

检测函数：

```python
detect_and_resolve
_detect_and_resolve_impl
```

每轮检测做以下操作：

1. 刷新 ACDATA；
2. 筛选 `DYN*` 飞机；
3. 两两计算 CPA；
4. 若 CPA 进入预测 gate 且当前目标计划不安全，则加入 detections。

关键参数：

```python
LOOKAHEAD_MIN = 20.0
HSEP_NM = 5.0
VSEP_FT = 1000.0
PREDICT_GATE_NM = 20.0
VERIFY_VSEP_FT = 1000.0
VERIFY_DT_SEC = 5
```

GUI 中 `Min sep NM` 会动态更新水平间隔要求。

CPA 计算：

```python
_cpa(a, b)
```

当前目标是否安全：

```python
_current_targets_are_safe(a, b)
```

## 9. 候选动作生成

函数：

```python
_candidate_actions(state, allow_alt_reversal=False)
```

候选动作包括：

### 9.1 Hold

保持当前目标高度和速度。

### 9.2 Altitude actions

高度调整范围：

```python
ALT_DELTAS_FL = [10, 20, 30]
SAFE_LEVELS = FL270 ... FL390
VS_FPM = 2000
```

生成 BlueSky 命令：

```text
ALT <acid>,FL<target>,<vertical_speed>
```

### 9.3 Speed actions

速度调整范围：

```python
SPEED_DELTAS_KT = [-20, 20, -30, 30]
MIN_SPEED_KT = 250
MAX_SPEED_KT = 330
SPEED_ACCEL_KT_PER_SEC = 1.0
```

生成 BlueSky 命令：

```text
SPD <acid>,<target_speed>
```

### 9.4 Preference

GUI 中的 `Preference` 决定候选动作排序：

- `speed_first`：hold → speed → altitude
- `altitude_first`：hold → altitude → speed

偏好只改变搜索顺序，不绕过安全验证。

## 10. 前向安全验证

核心函数：

```python
_action_pair_is_safe(a, action_a, b, action_b)
```

验证方式：

- horizon：`LOOKAHEAD_MIN = 20 min`
- step：`VERIFY_DT_SEC = 5 sec`
- 根据速度加速度和垂直率预测未来位置 / 高度
- 每个采样点检查：
  - 水平间隔是否低于 `Min sep NM`
  - 垂直间隔是否低于 `VERIFY_VSEP_FT`

只要任一时刻同时违反水平和垂直间隔，该动作对就判定不安全。

## 11. 离散约束搜索 solver

函数：

```python
_build_resolution_plan(state_by_id, detections)
```

流程：

1. 根据 detections 构建 conflict graph；
2. 为每个冲突飞机生成候选动作；
3. 按节点度数和紧迫度排序；
4. 使用回溯搜索选择兼容动作；
5. 每个边上的动作组合都必须通过 `_action_pair_is_safe`；
6. 返回 verified actions。

限制：

```python
MAX_SOLVER_NODES = 6000
SOLVER_TIME_BUDGET_SEC = 0.55
```

solver 输出字段包括：

```json
{
  "method": "discrete_constraint_search",
  "preference": "speed_first",
  "num_conflict_aircraft": 0,
  "num_conflict_pairs": 0,
  "search_nodes": 0,
  "pair_checks": 0,
  "used_alt_reversal_fallback": false,
  "timed_out": false,
  "selected_actions": {},
  "success": true
}
```

## 12. Fallback 与 recovery

### 12.1 Altitude reversal fallback

默认情况下，如果飞机正在爬升 / 下降，系统不会立刻给反向高度命令，以避免震荡。

如果第一轮搜索失败，会打开：

```python
allow_alt_reversal=True
```

再搜索一次。

### 12.2 Recovery altitude graph coloring

如果离散搜索仍失败，系统调用：

```python
_build_recovery_altitude_plan
```

该方法基于冲突图进行高度层分配，尽可能把相邻冲突飞机分到不同高度层。

对应 solver 方法：

```text
altitude_recovery_graph_coloring
```

如果 recovery 也无法生成 verified actions，系统会：

- GUI 显示 blocked；
- 记录 `conflict_detected_no_verified_action`；
- 不下发任何未经验证命令。

## 13. LLM wrapper

函数：

```python
_llm_wrap_decision(detections, actions, solver)
```

LLM wrapper 在 solver 之后运行。它只处理 verified actions。

支持三种模式：

### 13.1 template_explainer

本地确定性模板，不依赖网络。输出：

```json
{
  "provider": "template_explainer",
  "prompt_contract": "conflict_state + controller_preference -> structured_actions + standard_phrase + rationale",
  "preference": "speed_first",
  "conflicts": [],
  "structured_actions": [],
  "standard_instructions": [],
  "explanation": ""
}
```

### 13.2 openai_compatible_api

调用环境变量指定的 chat completions API：

```text
ATC_LLM_API_URL=http://127.0.0.1:8000/v1/chat/completions
ATC_LLM_MODEL=qwen3-4b
ATC_LLM_API_KEY=optional_key
```

API 只被要求解释 verified plan，不允许修改 commands。

如果 API 失败，系统记录 `api_error`，但保留本地 verified decision。

### 13.3 off

关闭 LLM wrapper，仅显示本地 verified actions。

## 14. BlueSky 命令下发

下发函数：

```python
_stack(command)
```

当前只下发 verified action 对应的：

```text
ALT <acid>,FL<target>,<vs>
SPD <acid>,<speed>
```

同时系统会记录：

- `issued_commands`
- `last_targets`
- `last_speeds`
- `command_records`
- `last_command_by_aircraft`

## 15. HMI 显示

### 15.1 Current conflicts

列：

- Pair
- CPA time
- CPA sep
- Now sep
- State
- Command

状态包括：

- Monitoring
- Issued
- Executing
- Loss
- Blocked

### 15.2 Issued commands

列：

- Time
- Aircraft
- Type
- BlueSky cmd
- Instruction/Reason
- Execution

### 15.3 状态栏

显示：

- Aircraft
- Cycles
- Active conflicts
- Commands
- Execution
- Loss
- LLM
- Log

### 15.4 飞机颜色

函数：

```python
_update_aircraft_alert_colors
```

颜色含义：

- green：正常
- yellow：预测冲突
- red：当前 loss

## 16. 执行状态监控

函数：

```python
_update_command_execution_statuses
```

系统根据 ACDATA 持续判断命令是否完成：

- 高度命令：当前高度距离目标 FL 小于约 150 ft 视为 reached；
- 速度命令：当前速度距离目标速度小于约 5 kt 视为 reached。

状态栏中显示：

```text
Execution: reached/tracked
```

## 17. 日志格式

GUI 日志目录：

```text
bluesky_project/output/hmi_dynamic_logs/
```

每次 reset 新建：

```text
dynamic_sector_YYYYmmdd_HHMMSS.jsonl
```

常见事件：

- `log_started`
- `sector_reset`
- `auto_traffic_started`
- `auto_traffic_stopped`
- `aircraft_spawned`
- `spawn_delayed_entry_gate`
- `min_separation_changed`
- `detect_cycle_clear`
- `conflicts_detected_and_resolved`
- `conflict_monitoring_no_new_command`
- `preventive_solver_failed_recovery_issued`
- `conflict_detected_no_verified_action`

典型 resolved 事件包含：

```json
{
  "event": "conflicts_detected_and_resolved",
  "detections": [
    {
      "pair": ["DYN001", "DYN002"],
      "tcpa_min": 3.2,
      "hsep_nm": 4.8,
      "vsep_ft": 0
    }
  ],
  "solver": {
    "method": "discrete_constraint_search",
    "preference": "speed_first",
    "selected_actions": {
      "DYN001": "speed:290kt"
    },
    "success": true
  },
  "commands": ["SPD DYN001,290"],
  "llm_output": {
    "provider": "template_explainer",
    "standard_instructions": []
  }
}
```

## 18. Headless validation

GUI 主要用于 HMI 展示和交互演示。定量安全验证应使用：

```text
headless_validation/
```

运行入口：

```powershell
.\RUN_HEADLESS_SMOKE.ps1
```

GUI fallback 不能替代 headless verifier。

## 19. 项目组二次开发入口

### 19.1 改 HMI 布局

主要修改：

```text
bluesky_project/bluesky/ui/qtgl/aiassist.py
```

重点函数：

```python
_build_ui
_refresh_conflict_table
_add_command_row
_update_status_labels
_log_text
```

### 19.2 改冲突检测规则

重点函数：

```python
_cpa
_current_targets_are_safe
_detect_and_resolve_impl
```

相关参数：

```python
LOOKAHEAD_MIN
HSEP_NM
VERIFY_VSEP_FT
PREDICT_GATE_NM
VERIFY_DT_SEC
```

### 19.3 改候选动作空间

重点函数：

```python
_candidate_actions
_predicted_state
_action_pair_is_safe
```

相关参数：

```python
ALT_DELTAS_FL
SPEED_DELTAS_KT
SAFE_LEVELS
MIN_SPEED_KT
MAX_SPEED_KT
VS_FPM
SPEED_ACCEL_KT_PER_SEC
```

### 19.4 接入真实 LLM

重点函数：

```python
_llm_wrap_decision
_call_llm_api
```

建议保持接口原则：

- 输入：verified decision payload；
- 输出：解释、标准话术、结构化摘要；
- 禁止 LLM 生成新 action；
- 禁止 LLM 绕过 verifier。

### 19.5 增加验证脚本

放在：

```text
headless_validation/
```

不要把大规模日志直接提交到 GitHub。

## 20. 当前安全边界总结

必须保持：

1. LLM 不直接下发 BlueSky 命令；
2. BlueSky 命令只能来自本地 verified actions；
3. verifier 失败时不得输出可执行指令；
4. GUI fallback 仅用于演示，不作为安全证据；
5. 定量评估使用 headless validation；
6. JSONL 日志应保留 solver、commands、LLM output，方便审计。

## 21. 已知限制

- 当前 GUI 是演示原型，不是认证级 ATC 工具；
- ACDATA 在短 demo 中可能延迟，因此存在 synthetic fallback；
- 搜索器有时间预算和节点预算，极复杂冲突图可能进入 recovery 或 blocked；
- LLM API 当前只作为解释层，真实模型接入后仍需保持 verifier-first 结构；
- 大规模验证日志未纳入 GitHub 仓库，需要单独保存或以 release artifact 方式发布。
