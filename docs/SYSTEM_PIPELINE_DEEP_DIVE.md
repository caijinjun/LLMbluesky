# BlueSky 间隔管理可视化系统流程深度说明

本文档面向准备接手或扩展本项目的成员，按系统运行时的数据流、控制流和安全边界，详细解释 BlueSky 间隔管理可视化系统各环节的输入、处理、输出、关键参数和可扩展位置。

如果只想快速了解整体设计，可先读：

- `README.md`
- `README_DEMO.md`
- `docs/LLM_VISUAL_DEMO_DESIGN.md`
- `docs/SYSTEM_FLOW_DETAILED.md`

本文档更偏工程交接和二次开发。

---

## 1. 系统目标与边界

### 1.1 系统目标

本系统实现一个 BlueSky GUI 内嵌的人机协同 ATC 冲突解脱演示闭环：

1. 在动态扇区中持续生成航空器；
2. 实时采集航空器状态；
3. 预测未来最接近点（CPA）和间隔风险；
4. 把冲突航空器组织成冲突图；
5. 生成可解释的候选解脱动作；
6. 用本地前向验证器筛掉不安全动作；
7. 搜索一组联合安全动作；
8. 下发 BlueSky 可执行命令；
9. 通过 HMI 显示冲突状态、指令、执行状态和解释；
10. 把每次事件写入 JSONL，支持复盘、审计和训练数据构建。

### 1.2 系统不做什么

系统当前不是认证级 ATC 自动化工具，也不是让 LLM 直接控制飞机的 agent。当前安全边界明确为：

- LLM 不能直接产生最终 BlueSky 命令；
- LLM 不能绕过 verifier；
- GUI fallback 不能替代 headless safety validation；
- 没有 verified action 时，系统宁可 blocked，也不下发猜测指令；
- 真实模型接入后仍只能解释 verified decision payload。

---

## 2. 总体运行闭环

系统运行闭环可以理解为 10 个阶段：

```text
[1] GUI 启动与 AI 面板挂载
    ↓
[2] Reset sector 初始化扇区和内部状态
    ↓
[3] Start auto traffic 启动交通生成与检测 timer
    ↓
[4] BlueSky ACDATA / synthetic fallback 获取飞机状态
    ↓
[5] CPA lookahead 检测预测冲突
    ↓
[6] 构建冲突图并生成候选动作
    ↓
[7] 前向验证每个动作组合
    ↓
[8] 离散约束搜索 / fallback / recovery
    ↓
[9] LLM wrapper 生成解释，BlueSky stack 下发 verified commands
    ↓
[10] HMI 更新 + JSONL 日志 + 执行状态追踪
```

核心实现集中在：

```text
bluesky_project/bluesky/ui/qtgl/aiassist.py
```

主类：

```python
class AiAssistPanel(QWidget):
```

---

## 3. 阶段 1：GUI 启动与 AI 面板挂载

### 3.1 启动入口

常用启动脚本：

```powershell
.\RUN_BLUESKY_GUI.ps1
```

带 mock LLM：

```powershell
.\RUN_MOCK_LLM.ps1
.\RUN_BLUESKY_GUI_WITH_MOCK_LLM.ps1
```

脚本会进入 `bluesky_project/` 并启动 BlueSky GUI。

### 3.2 面板识别

如果加载成功，GUI 底部会出现 AI tab，标题为：

```text
AI Decision Assist - dynamic sector conflict detection
```

这是当前系统最新版本的重要识别标志。

### 3.3 网络流连接

AI 面板通过：

```python
_ensure_stream_connection()
```

连接 BlueSky net stream：

```python
net.stream_received.connect(self.on_simstream_received)
```

它主要关心：

```python
streamname == b"ACDATA"
```

---

## 4. 阶段 2：Reset sector 初始化

### 4.1 用户动作

在 AI tab 点击：

```text
Reset sector
```

调用：

```python
reset_sector()
```

### 4.2 BlueSky 命令

Reset 会调用：

```python
base_sector_commands()
```

当前命令包括：

```text
RESET
HOLD
PAN <CENTER_LAT>,<CENTER_LON>
ZOOM <ZOOM>
BOX HMI_SECTOR,...
COLOR HMI_SECTOR,0,180,80
```

### 4.3 内部状态清空

Reset 会清空：

```python
active_meta
latest_acdata
resolved_pairs
tracked_conflicts
assigned_aircraft
last_targets
last_speeds
last_aircraft_colors
issued_commands
command_records
last_command_by_aircraft
command_queue
```

同时重置：

```python
detect_cycles = 0
conflict_events = 0
active_conflict_count = 0
last_llm_status = "idle"
```

### 4.4 日志初始化

Reset 会新建日志文件：

```python
_new_log_file()
```

输出路径：

```text
bluesky_project/output/hmi_dynamic_logs/dynamic_sector_YYYYmmdd_HHMMSS.jsonl
```

第一条日志通常是：

```json
{"event": "log_started", "scenario": "ATC_HMI_DYNAMIC_14AC_SECTOR"}
```

---

## 5. 阶段 3：动态交通生成

### 5.1 用户动作

点击：

```text
Start auto traffic
```

调用：

```python
start_auto_traffic()
```

### 5.2 启动定时器

系统启动两个核心 timer：

```python
spawn_timer  -> spawn_random_aircraft
```

周期：

```python
SPAWN_INTERVAL_MS = 30000
```

```python
detect_timer -> detect_and_resolve
```

周期：

```python
DETECT_INTERVAL_MS = 4000
```

### 5.3 初始交通波

如果当前无飞机，系统调用：

```python
spawn_initial_wave()
```

初始注入 4 架飞机，覆盖东西向和南北向主要航路。

### 5.4 航路网络

当前动态扇区有 8 条 route：

| Route | 方向 | 入口 | 出口 | 航向 | FL 候选 | 速度区间 |
|---|---|---|---|---:|---|---|
| R1-EW | 西向东 | W_IN | E_IN | 90 | 320/340/360 | 290–320 |
| R1-WE | 东向西 | E_IN | W_IN | 270 | 320/340/360 | 290–320 |
| R2-NS | 北向南 | N_IN | S_IN | 180 | 330/350/370 | 280–310 |
| R2-SN | 南向北 | S_IN | N_IN | 0 | 330/350/370 | 280–310 |
| R3-SWNE | 西南向东北 | SW_IN | NE_IN | 45 | 310/330/350 | 280–310 |
| R3-NESW | 东北向西南 | NE_IN | SW_IN | 225 | 310/330/350 | 280–310 |
| R3-NWSE | 西北向东南 | NW_IN | SE_IN | 135 | 300/340/380 | 270–300 |
| R3-SENW | 东南向西北 | SE_IN | NW_IN | 315 | 300/340/380 | 270–300 |

### 5.5 生成飞机命令

每架飞机用 BlueSky stack 命令生成：

```text
CRE DYNxxx,<type>,<lat>,<lon>,<hdg>,FL<fl>,<speed>
COLOR DYNxxx,0,255,0
ADDWPT DYNxxx <exit_lat> <exit_lon> FL<fl> <speed>
DYNxxx LNAV ON
```

相关函数：

```python
spawn_aircraft()
```

### 5.6 Entry gate

生成前会调用：

```python
_spawn_candidate_is_safe(route, fl, speed)
```

用于避免刚生成就冲突。

验证参数：

```python
ENTRY_LOOKAHEAD_MIN = 2.5
ENTRY_VERIFY_DT_SEC = 10
SPAWN_RETRY_LIMIT = 16
```

如果候选入口在 2.5 分钟内与已有飞机低于间隔，会拒绝或延迟该生成。

---

## 6. 阶段 4：状态采集

### 6.1 首选来源：BlueSky ACDATA

函数：

```python
on_simstream_received(streamname, data, sender_id)
```

当 `streamname == b"ACDATA"` 时，调用：

```python
_set_latest_acdata(ACDataEvent(data))
```

读取并转换：

| ACDATA 字段 | 内部字段 | 说明 |
|---|---|---|
| id | id | callsign |
| lat | lat | 纬度 |
| lon | lon | 经度 |
| alt | alt_ft | 米转英尺 |
| trk | trk | 航迹角 |
| gs | gs_mps | 地速 m/s |
| cas | cas_kt | 指示空速 kt |

### 6.2 RadarWidget 刷新

如果 net stream 不及时，系统也尝试从 radarwidget 读取：

```python
_refresh_from_radarwidget()
```

### 6.3 Synthetic fallback

短时 GUI demo 可能出现 ACDATA 稀疏。系统用：

```python
_synthetic_states_from_meta()
```

根据飞机生成记录推算位置：

```text
entry point + heading + speed + elapsed time
```

注意：fallback 只用于演示，不作为定量安全验证证据。

---

## 7. 阶段 5：CPA 冲突预测

### 7.1 检测入口

定时器调用：

```python
detect_and_resolve()
```

内部执行：

```python
_detect_and_resolve_impl()
```

### 7.2 飞机筛选

只处理动态演示飞机：

```python
acid.startswith("DYN")
```

### 7.3 CPA 计算

函数：

```python
_cpa(a, b)
```

使用：

- 当前经纬度投影到本地 NM 平面；
- 根据 track 和 ground speed 得到速度向量；
- 计算未来 `LOOKAHEAD_MIN` 内最近点。

关键参数：

```python
LOOKAHEAD_MIN = 20.0
PREDICT_GATE_NM = 20.0
```

### 7.4 冲突进入条件

一对飞机进入 detections，当满足：

```python
hsep < _predict_gate_nm()
and not _current_targets_are_safe(a, b)
```

其中 `_current_targets_are_safe` 会考虑当前已经下发但尚未完成的目标高度/速度。

### 7.5 最小间隔

GUI 中 `Min sep NM` 默认来自：

```python
HSEP_NM = 5.0
```

可在界面中动态调整。

垂直验证阈值：

```python
VERIFY_VSEP_FT = 1000.0
```

---

## 8. 阶段 6：冲突图构建

函数：

```python
_build_resolution_plan(state_by_id, detections)
```

把 detections 转成 conflict graph：

```python
graph[acid_a].add(acid_b)
graph[acid_b].add(acid_a)
```

图的含义：

- 节点：涉及冲突的航空器；
- 边：两架航空器在预测 horizon 内存在不安全关系。

系统还记录每个节点的 urgency：

```python
urgency[acid] = min(tcpa)
```

搜索顺序：

```python
order = sorted(graph, key=lambda acid: (-degree, urgency, acid))
```

也就是优先处理冲突边多、时间更紧迫的飞机。

---

## 9. 阶段 7：候选动作空间

### 9.1 候选动作函数

```python
_candidate_actions(state, allow_alt_reversal=False)
```

每个冲突飞机都有一个候选动作列表。

### 9.2 Hold action

默认第一个动作是 hold：

```json
{
  "kind": "hold",
  "target_fl": effective_fl,
  "target_speed": current_speed,
  "command": null
}
```

### 9.3 Altitude actions

高度动作来自：

```python
ALT_DELTAS_FL = [10, 20, 30]
SAFE_LEVELS = list(range(270, 391, 10))
VS_FPM = 2000
```

生成命令：

```text
ALT DYN001,FL350,2000
ALT DYN001,FL330,-2000
```

### 9.4 Speed actions

速度动作来自：

```python
SPEED_DELTAS_KT = [-20, 20, -30, 30]
MIN_SPEED_KT = 250
MAX_SPEED_KT = 330
SPEED_ACCEL_KT_PER_SEC = 1.0
```

生成命令：

```text
SPD DYN001,290
```

### 9.5 Preference 影响

界面 `Preference` 只影响动作排序，不改变 verifier：

```python
speed_first: hold -> speed -> altitude
altitude_first: hold -> altitude -> speed
```

这意味着 Preference 是控制员偏好，不是安全放行。

---

## 10. 阶段 8：前向安全验证

### 10.1 预测状态

函数：

```python
_predicted_state(state, action, t_sec)
```

它根据动作预测未来位置和高度。

速度变化采用线性加速：

```python
SPEED_ACCEL_KT_PER_SEC = 1.0
```

高度变化采用固定垂直率：

```python
VS_FPM = 2000
```

### 10.2 动作对验证

函数：

```python
_action_pair_is_safe(a, action_a, b, action_b)
```

验证过程：

```text
for t = 0 到 LOOKAHEAD_MIN，每 VERIFY_DT_SEC 秒：
    预测 a 的位置/高度
    预测 b 的位置/高度
    计算 hsep, vsep
    如果 hsep < MinSep 且 vsep < VerifyVsep：
        return False
return True
```

参数：

```python
LOOKAHEAD_MIN = 20.0
VERIFY_DT_SEC = 5
VERIFY_VSEP_FT = 1000.0
```

### 10.3 验证含义

一个 action 不是单独安全，而是相对所有已分配邻居安全。

例如：

```text
DYN001 speed -20kt 对 DYN002 安全，
但可能对 DYN003 不安全。
```

所以 solver 必须搜索联合动作组合。

---

## 11. 阶段 9：离散约束搜索

### 11.1 搜索目标

寻找每个冲突节点一个 action，使每条冲突边上的 action pair 都通过前向验证。

### 11.2 搜索过程

`_build_resolution_plan` 内部定义：

```python
compatible(acid, action, assigned)
search(assigned)
```

每次为一个飞机选择候选动作，并检查它与已分配邻居是否兼容。

### 11.3 剪枝

系统使用：

- 最小可行动作数优先；
- pair safety cache；
- 最大搜索节点；
- 时间预算。

参数：

```python
MAX_SOLVER_NODES = 6000
SOLVER_TIME_BUDGET_SEC = 0.55
```

### 11.4 输出

返回：

```python
actions, solver
```

`actions` 是需要下发命令的动作，不包含 hold。

`solver` 记录审计信息：

```json
{
  "method": "discrete_constraint_search",
  "preference": "speed_first",
  "num_conflict_aircraft": 4,
  "num_conflict_pairs": 2,
  "search_nodes": 123,
  "pair_checks": 45,
  "used_alt_reversal_fallback": false,
  "timed_out": false,
  "selected_actions": {
    "DYN001": "speed:290kt",
    "DYN002": "hold"
  },
  "success": true
}
```

---

## 12. 阶段 10：fallback 与 recovery

### 12.1 为什么需要 fallback

动态多机冲突可能导致：

- 当前动作空间内找不到联合安全动作；
- 搜索超时；
- 某些飞机已有 pending altitude command，不允许反向高度；
- 图结构较复杂。

### 12.2 Altitude reversal fallback

第一轮搜索默认：

```python
allow_alt_reversal=False
```

若失败，第二轮允许：

```python
allow_alt_reversal=True
```

并记录：

```json
"used_alt_reversal_fallback": true
```

### 12.3 Recovery altitude graph coloring

如果离散搜索失败，系统调用：

```python
_build_recovery_altitude_plan
```

思想：把冲突图中相邻飞机分配到不同高度层，优先选接近当前高度的安全层。

方法名：

```text
altitude_recovery_graph_coloring
```

### 12.4 Blocked

如果 recovery 也失败：

- HMI 状态显示 `Blocked`；
- 命令表显示：`No verified action; hold for controller review`；
- 日志写入：`conflict_detected_no_verified_action`；
- 不下发任何未验证命令。

---

## 13. 阶段 11：LLM wrapper

### 13.1 调用位置

LLM wrapper 在 solver 之后：

```python
llm_output = self._llm_wrap_decision(detections, actions, solver)
```

也就是说输入的 `actions` 已经是 verifier 通过的动作。

### 13.2 输出契约

输出包含：

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

### 13.3 template_explainer

默认本地模板，优点：

- 离线可运行；
- 输出稳定；
- 便于 HMI demo；
- 不依赖网络。

### 13.4 openai_compatible_api

如果选择该模式，系统调用：

```python
_call_llm_api(decision_payload)
```

环境变量：

```text
ATC_LLM_API_URL=http://127.0.0.1:8000/v1/chat/completions
ATC_LLM_MODEL=qwen3-4b
ATC_LLM_API_KEY=optional_key
```

发送给模型的核心约束：

```text
Only explain verified ATC actions. Do not change commands.
```

如果 API 失败，系统只记录错误，不影响 verified command。

### 13.5 off

关闭解释层，只保留本地决策和命令显示。

---

## 14. 阶段 12：BlueSky 命令下发

### 14.1 下发函数

```python
_stack(command)
```

优先通过：

```python
bs.net.send_event(b"STACKCMD", command, target=target)
```

如果 net 不可用，回退到 console stack。

### 14.2 命令类型

当前自动下发的命令只包括 verified action 的：

```text
ALT DYN001,FL350,2000
SPD DYN002,290
```

不会下发：

- LLM 自由生成命令；
- 未通过 verifier 的 fallback 命令；
- 无 action id 来源的命令。

### 14.3 去重

系统用：

```python
issued_commands
```

避免同一命令重复下发。

---

## 15. 阶段 13：HMI 展示

### 15.1 当前冲突表

表名：

```text
Current conflicts - updated in place
```

列：

| 列 | 含义 |
|---|---|
| Pair | 冲突飞机对 |
| CPA time | 预计 CPA 时间 |
| CPA sep | CPA 时水平/垂直间隔 |
| Now sep | 当前水平/垂直间隔 |
| State | Monitoring / Issued / Executing / Loss / Blocked |
| Command | 相关 BlueSky 命令 |

### 15.2 Issued commands 表

列：

| 列 | 含义 |
|---|---|
| Time | 下发时间 |
| Aircraft | 飞机 |
| Type | altitude / speed / alert |
| BlueSky cmd | 实际命令 |
| Instruction/Reason | 标准话术或原因 |
| Execution | 执行状态 |

### 15.3 状态栏

状态栏包含：

- Aircraft：当前 DYN 飞机数；
- Cycles：检测周期数；
- Active conflicts：活跃冲突数；
- Commands：已发命令数；
- Execution：命令达到目标数量；
- Loss：当前真实间隔 loss 数；
- LLM：template / api_ok / api_error / off；
- Log：当前 JSONL 文件名。

### 15.4 颜色

函数：

```python
_update_aircraft_alert_colors
```

颜色含义：

| 颜色 | 含义 |
|---|---|
| green | 当前无预测冲突 |
| yellow | 存在预测冲突 |
| red | 当前已经 loss |

---

## 16. 阶段 14：执行状态追踪

### 16.1 记录命令

下发 altitude / speed 命令后，系统调用：

```python
_register_command_monitor(row, action)
```

记录：

- aircraft id；
- action kind；
- target FL；
- target speed；
- issued time；
- reached 标记。

### 16.2 判断 reached

函数：

```python
_update_command_execution_statuses
```

高度命令：

```text
abs(current_alt_ft - target_alt_ft) <= 150 ft
```

速度命令：

```text
abs(current_speed - target_speed) <= 5 kt
```

执行状态会显示在 Issued commands 表和状态栏。

---

## 17. 阶段 15：JSONL 日志

### 17.1 日志位置

```text
bluesky_project/output/hmi_dynamic_logs/dynamic_sector_*.jsonl
```

该目录被 `.gitignore` 排除，因为它是运行输出。

### 17.2 日志写入函数

```python
_append_log(record)
```

每条记录自动加：

```json
"wall_time": "YYYY-MM-DDTHH:MM:SS"
```

### 17.3 常见事件

| event | 说明 |
|---|---|
| log_started | 新日志开始 |
| sector_reset | 扇区重置 |
| auto_traffic_started | 自动交通开始 |
| aircraft_spawned | 飞机生成 |
| spawn_delayed_entry_gate | 入口安全门延迟生成 |
| detect_cycle_clear | 检测周期无冲突 |
| conflicts_detected_and_resolved | 检测并下发 verified actions |
| conflict_monitoring_no_new_command | 已有目标安全，无需新命令 |
| preventive_solver_failed_recovery_issued | 预防 solver 失败，recovery 下发 |
| conflict_detected_no_verified_action | 无 verified action，blocked |
| min_separation_changed | 用户调整最小间隔 |

### 17.4 resolved 事件结构

典型结构：

```json
{
  "event": "conflicts_detected_and_resolved",
  "detections": [
    {
      "pair": ["DYN001", "DYN002"],
      "tcpa_min": 2.5,
      "hsep_nm": 4.8,
      "vsep_ft": 0
    }
  ],
  "solver": {
    "method": "discrete_constraint_search",
    "preference": "speed_first",
    "selected_actions": {
      "DYN001": "speed:290kt",
      "DYN002": "hold"
    },
    "success": true
  },
  "commands": ["SPD DYN001,290"],
  "llm_output": {
    "provider": "template_explainer",
    "standard_instructions": [
      "DYN001, adjust indicated airspeed to 290 knots."
    ],
    "explanation": "..."
  }
}
```

---

## 18. Headless validation 与 GUI 的关系

GUI 是 HMI 展示与交互系统。定量安全证据来自：

```text
headless_validation/
```

运行：

```powershell
.\RUN_HEADLESS_SMOKE.ps1
```

Headless 验证通常关注：

- 是否出现 loss；
- fallback 调用次数；
- 冲突是否成功解除；
- 是否有未验证动作；
- 多场景 batch 统计。

GUI 里的 synthetic fallback、界面颜色和 table 状态是展示层，不应替代 headless 统计。

---

## 19. 模块扩展指南

### 19.1 增加新的 HMI 控件

修改：

```python
_build_ui
```

通常需要同步更新：

```python
_update_status_labels
_append_log
```

### 19.2 改冲突检测阈值

参数：

```python
LOOKAHEAD_MIN
HSEP_NM
PREDICT_GATE_NM
VERIFY_VSEP_FT
VERIFY_DT_SEC
```

注意 GUI 的 `Min sep NM` 会动态覆盖水平间隔。

### 19.3 增加新动作类型

需要同时修改：

```python
_candidate_actions
_predicted_state
_action_pair_is_safe
_llm_wrap_decision
_register_command_monitor
_update_command_execution_statuses
```

例如要加 heading vectoring，就必须定义：

- heading 候选集合；
- heading 动力学；
- BlueSky 命令格式；
- verifier 中的未来轨迹预测；
- HMI 展示和 reached 判据。

### 19.4 接入真实 LLM

修改：

```python
_call_llm_api
```

但必须保持：

- 输入只包含 verified decision payload；
- 输出只用于解释；
- 不允许模型新增 action；
- 不允许模型修改 BlueSky command。

### 19.5 增加模型训练输出

如果将未来微调模型接入为 decision selector，建议仍维持两层结构：

```text
LLM 产生候选 action id / ranking
    ↓
本地 verifier 复核
    ↓
只下发 verified action
```

不要把 LLM 直接接到 BlueSky stack。

---

## 20. 典型调试路径

### 20.1 GUI 没有 AI tab

检查：

- `bluesky_project/bluesky/ui/qtgl/aiassist.py` 是否存在；
- `mainwindow.py` 是否挂载 AiAssistPanel；
- 启动脚本是否进入正确的 `bluesky_project/`。

### 20.2 没有飞机生成

检查：

- 是否点击 `Reset sector`；
- 是否点击 `Start auto traffic`；
- entry gate 是否持续拒绝；
- command queue 是否积压；
- BlueSky 是否处于 OP 状态。

### 20.3 有飞机但无冲突

可能原因：

- 交通还没进入交叉区域；
- `Min sep NM` 设得太低；
- flight levels 已天然分离；
- 当前目标计划已安全。

可点击：

```text
Detect now
Fast 2 min
Spawn one
```

### 20.4 有冲突但没有命令

查看 HMI State：

- `Monitoring`：仍在跟踪；
- `Issued`：可能已下发过命令；
- `Executing`：等待飞机执行；
- `Blocked`：无 verified action；
- `Loss`：当前已经进入 loss。

同时查看 JSONL 日志中的 solver 字段。

### 20.5 LLM API 不工作

检查环境变量：

```text
ATC_LLM_API_URL
ATC_LLM_MODEL
ATC_LLM_API_KEY
```

可以先运行：

```powershell
.\RUN_MOCK_LLM.ps1
```

然后选择 `openai_compatible_api` 验证接口链路。

---

## 21. 交接给项目组时的推荐阅读顺序

1. `README.md`：快速运行；
2. `README_DEMO.md`：demo 操作；
3. `docs/SYSTEM_FLOW_DETAILED.md`：系统级流程说明；
4. `docs/SYSTEM_PIPELINE_DEEP_DIVE.md`：工程级流程深挖；
5. `bluesky_project/bluesky/ui/qtgl/aiassist.py`：核心实现；
6. `headless_validation/`：验证脚本。

---

## 22. 一句话总结

当前系统是一个 verifier-first 的 BlueSky ATC HMI 原型：

```text
动态交通 → CPA 冲突图 → 候选动作 → 前向安全验证 → 离散搜索 → verified BlueSky 命令 → LLM 解释 → HMI/日志
```

它的核心不是让 LLM 直接控制，而是把 LLM 放在安全验证之后，用于解释、标准话术和人机交互增强。
