"""Dynamic ATC human-machine decision panel for BlueSky QtGL.

The panel loads a real-sector route map, spawns aircraft at route entry fixes,
monitors live BlueSky ACDATA, detects predicted separation loss, and issues
verified controller commands with a persistent decision log.
"""
from datetime import datetime
import json
import os
from math import cos, radians, sin, sqrt
from pathlib import Path
import random
from time import monotonic
from urllib import request, error

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QTextCursor
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit,
    QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView, QComboBox,
    QApplication, QSizePolicy, QDoubleSpinBox, QFrame, QFormLayout
)
import bluesky as bs
from bluesky.ui.qtgl.customevents import ACDataEvent


FT_PER_METER = 3.280839895
NM_PER_METER = 1.0 / 1852.0
KT_PER_MPS = 1.9438444924406


def _load_real_sector():
    path = Path(__file__).resolve().parents[3] / "scenario_data" / "chengdu_chongqing_real_sector_v1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    waypoints = {}
    routes = []
    all_lats = []
    all_lons = []
    for route_id in data["route_ids"]:
        source = data["routes"][route_id]
        route_points = []
        for point in source["waypoints"]:
            lat = float(point["lat"])
            lon = float(point["lon"])
            name = str(point["name"])
            waypoints[name] = (lat, lon)
            route_points.append((lat, lon))
            all_lats.append(lat)
            all_lons.append(lon)
        min_fl = int(source.get("route_min_required_fl", 0))
        flight_levels = [fl for fl in range(280, 381, 20) if fl >= min_fl]
        routes.append({
            "route": route_id,
            "entry": source["entry"],
            "exit": source["exit"],
            "hdg": int(round(float(source["initial_bearing_deg"]))) % 360,
            "fls": flight_levels,
            "speed": (280, 320),
            "waypoints": route_points,
            "route_code": source.get("source_route_code", route_id),
        })
    bounds = (min(all_lats), min(all_lons), max(all_lats), max(all_lons))
    center = ((bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0)
    return data["sector_id"], waypoints, routes, bounds, center


REAL_SECTOR_ID, REAL_WAYPOINTS, REAL_ROUTES, REAL_BOUNDS, REAL_CENTER = _load_real_sector()

DEMO_WAYPOINTS = {
    "W_IN": (30.7000, 102.7500),
    "E_IN": (30.7000, 105.4500),
    "N_IN": (31.8500, 104.1000),
    "S_IN": (29.5500, 104.1000),
    "SW_IN": (29.7500, 103.1500),
    "NE_IN": (31.6500, 105.0500),
    "NW_IN": (31.6500, 103.1500),
    "SE_IN": (29.7500, 105.0500),
}

DEMO_ROUTES = [
    {"route": "R1-EW", "entry": "W_IN", "exit": "E_IN", "hdg": 90, "fls": [320, 340, 360], "speed": (290, 320)},
    {"route": "R1-WE", "entry": "E_IN", "exit": "W_IN", "hdg": 270, "fls": [320, 340, 360], "speed": (290, 320)},
    {"route": "R2-NS", "entry": "N_IN", "exit": "S_IN", "hdg": 180, "fls": [330, 350, 370], "speed": (280, 310)},
    {"route": "R2-SN", "entry": "S_IN", "exit": "N_IN", "hdg": 0, "fls": [330, 350, 370], "speed": (280, 310)},
    {"route": "R3-SWNE", "entry": "SW_IN", "exit": "NE_IN", "hdg": 45, "fls": [310, 330, 350], "speed": (280, 310)},
    {"route": "R3-NESW", "entry": "NE_IN", "exit": "SW_IN", "hdg": 225, "fls": [310, 330, 350], "speed": (280, 310)},
    {"route": "R3-NWSE", "entry": "NW_IN", "exit": "SE_IN", "hdg": 135, "fls": [300, 340, 380], "speed": (270, 300)},
    {"route": "R3-SENW", "entry": "SE_IN", "exit": "NW_IN", "hdg": 315, "fls": [300, 340, 380], "speed": (270, 300)},
]
for demo_route in DEMO_ROUTES:
    demo_route["waypoints"] = [
        DEMO_WAYPOINTS[demo_route["entry"]],
        DEMO_WAYPOINTS[demo_route["exit"]],
    ]
    demo_route["route_code"] = demo_route["route"]

MAP_CONFIGS = {
    "three_route_demo": {
        "label": "Three-route demo",
        "scenario_name": "ATC_HMI_DYNAMIC_14AC_SECTOR",
        "sector_id": "three_route_demo_sector",
        "waypoints": DEMO_WAYPOINTS,
        "routes": DEMO_ROUTES,
        "bounds": (29.5500, 102.7500, 31.8500, 105.4500),
        "center": (30.7000, 104.1000),
        "zoom": 0.22,
        "visual_scenario": "ATC_HMI_MAP_THREE_ROUTE",
        "description": "Three-route demonstration sector",
    },
    "chengdu_chongqing_real": {
        "label": "Chengdu-Chongqing real routes",
        "scenario_name": "ATC_HMI_REAL_CHENGDU_CHONGQING_SECTOR",
        "sector_id": REAL_SECTOR_ID,
        "waypoints": REAL_WAYPOINTS,
        "routes": REAL_ROUTES,
        "bounds": REAL_BOUNDS,
        "center": REAL_CENTER,
        "zoom": 0.12,
        "visual_scenario": "ATC_HMI_MAP_CHENGDU_ROUTES",
        "description": "Chengdu-Chongqing real-route sector",
    },
}


class AiAssistPanel(QWidget):
    """Dynamic controller-assist panel embedded in the BlueSky bottom tabs."""

    DEFAULT_MAP_KEY = "chengdu_chongqing_real"
    SCENARIO_NAME = MAP_CONFIGS[DEFAULT_MAP_KEY]["scenario_name"]
    CENTER_LAT = MAP_CONFIGS[DEFAULT_MAP_KEY]["center"][0]
    CENTER_LON = MAP_CONFIGS[DEFAULT_MAP_KEY]["center"][1]
    ZOOM = MAP_CONFIGS[DEFAULT_MAP_KEY]["zoom"]
    MAX_AIRCRAFT = 14
    LOOKAHEAD_MIN = 20.0
    HSEP_NM = 5.0
    VSEP_FT = 1000.0
    PREDICT_GATE_NM = 20.0
    VERIFY_VSEP_FT = 1000.0
    VERIFY_DT_SEC = 5
    ALT_DELTAS_FL = [10, 20, 30]
    VS_FPM = 2000
    SPEED_DELTAS_KT = [-20, 20, -30, 30]
    MIN_SPEED_KT = 250
    MAX_SPEED_KT = 330
    SPEED_ACCEL_KT_PER_SEC = 1.0
    SAFE_LEVELS = list(range(270, 391, 10))
    SPAWN_INTERVAL_MS = 30000
    DETECT_INTERVAL_MS = 4000
    RESET_SETTLE_MS = 2500
    ENTRY_LOOKAHEAD_MIN = 2.5
    ENTRY_VERIFY_DT_SEC = 10
    SPAWN_RETRY_LIMIT = 16
    MAX_SPAWN_COMMAND_BACKLOG = 8
    COMMAND_INTERVAL_MS = 300
    STATUS_UPDATE_MIN_SEC = 0.5
    EXECUTION_UPDATE_MIN_SEC = 0.8
    ALT_REACHED_TOLERANCE_FT = 150.0
    SPEED_REACHED_TOLERANCE_KT = 5.0
    EXECUTION_START_ALT_DELTA_FT = 50.0
    EXECUTION_START_SPEED_DELTA_KT = 2.0
    TRACKED_SAFE_CHECK_INTERVAL_SEC = 2.0
    LOG_TEXT_MAX_LINES = 80
    MAX_SOLVER_NODES = 6000
    SOLVER_TIME_BUDGET_SEC = 0.55
    MAX_TRACKED_CONFLICTS = 28
    LOG_DIR = Path(__file__).resolve().parents[3] / "output" / "hmi_dynamic_logs"

    WAYPOINTS = MAP_CONFIGS[DEFAULT_MAP_KEY]["waypoints"]
    ROUTES = MAP_CONFIGS[DEFAULT_MAP_KEY]["routes"]
    SECTOR_BOUNDS = MAP_CONFIGS[DEFAULT_MAP_KEY]["bounds"]
    REAL_SECTOR_ID = MAP_CONFIGS[DEFAULT_MAP_KEY]["sector_id"]

    AIRCRAFT_TYPES = ["A320", "B738", "A319", "E190"]

    def __init__(self, console, radarwidget=None, parent=None):
        super(AiAssistPanel, self).__init__(parent)
        self.console = console
        self.radarwidget = radarwidget
        self.rng = random.Random(20260703)
        self.current_map_key = self.DEFAULT_MAP_KEY
        self._apply_map_config(self.current_map_key)
        self.spawn_timer = QTimer(self)
        self.detect_timer = QTimer(self)
        self.command_timer = QTimer(self)
        self.spawn_timer.timeout.connect(self.spawn_random_aircraft)
        self.detect_timer.timeout.connect(self.detect_and_resolve)
        self.command_timer.timeout.connect(self._drain_command_queue)
        self.command_timer.setInterval(self.COMMAND_INTERVAL_MS)
        self._autoload_done = False
        self.command_queue = []
        self.spawn_index = 0
        self.active_meta = {}
        self.latest_acdata = {}
        self.resolved_pairs = set()
        self.tracked_conflicts = {}
        self.assigned_aircraft = set()
        self.last_targets = {}
        self.last_speeds = {}
        self.issued_commands = set()
        self.command_records = []
        self.event_rows = []
        self.detect_cycles = 0
        self.conflict_events = 0
        self.active_conflict_count = 0
        self.last_command_by_aircraft = {}
        self.last_llm_status = "idle"
        self.detect_busy = False
        self.log_path = None
        self.latest_sim_time = None
        self.last_detect_wall_ts = 0.0
        self._connected_net = None
        self._last_status_update_ts = 0.0
        self._last_execution_update_ts = 0.0
        self._last_conflict_table_signature = None
        self._last_command_status_by_row = {}
        self._status_tone_cache = {}
        self._last_aircraft_table_signature = None
        self._tracked_safe_cache = {}
        self.reset_pending_until = 0.0
        self.pending_start = False
        self._build_ui()
        self._new_log_file()
        self._write_idle_summary()
        self._ensure_stream_connection()

    def _build_ui(self):
        self.setObjectName("aiAssistDashboard")
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("""
            QWidget#aiAssistDashboard { background: #091017; }
            QWidget { background: #091017; color: #d7e0e8; font-size: 12px; }
            QFrame#topBar { background: #0b141d; border-bottom: 1px solid #263746; }
            QFrame#sideRail, QFrame#bottomRail { background: #0c151d; border: 1px solid #263746; }
            QLabel { background: transparent; color: #c8d3dc; }
            QLabel#sectionTitle { color: #dce8f0; font-size: 13px; font-weight: 700; }
            QLabel#fieldValue {
                color: #e2ebf2; background: #101b24; border: 1px solid #2a3b49;
                border-radius: 2px; padding: 5px 7px;
            }
            QFrame#decisionStage { background: #0d1820; border: 1px solid #263a48; }
            QLabel#stageTitle { color: #d5e2e9; font-weight: 650; }
            QLabel#stageDetail { color: #8fa5b3; }
            QGroupBox {
                color: #dce7ef; border: 1px solid #2a3a47; border-radius: 3px;
                margin-top: 10px; padding: 9px 7px 7px 7px; font-weight: 650;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 9px; padding: 0 4px; }
            QPushButton {
                color: #dbe5ec; background: #17232d; border: 1px solid #344958;
                border-radius: 3px; padding: 6px 9px; min-height: 20px;
            }
            QPushButton:hover { background: #20313d; border-color: #4f7083; }
            QPushButton:pressed { background: #101920; }
            QPushButton#primaryAction { background: #087da3; border-color: #29a9cf; color: white; font-weight: 650; }
            QPushButton#dangerAction { color: #ffaaa4; border-color: #70413e; }
            QComboBox, QDoubleSpinBox {
                color: #dbe5ec; background: #101a23; border: 1px solid #344957;
                border-radius: 2px; padding: 5px 7px; min-height: 20px;
            }
            QComboBox::drop-down { border: 0; width: 18px; }
            QTableWidget {
                background: #0b131a; alternate-background-color: #0e1820;
                color: #d4dee6; border: 0; gridline-color: #263642;
                selection-background-color: #173c4d; selection-color: white;
            }
            QHeaderView::section {
                background: #14202a; color: #9eb0bf; border: 0;
                border-right: 1px solid #263642; border-bottom: 1px solid #344957;
                padding: 6px 4px; font-weight: 650;
            }
            QScrollBar:vertical { background: #0b131a; width: 10px; }
            QScrollBar::handle:vertical { background: #3a4c5b; min-height: 24px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        """)

        topbar = QFrame(self)
        topbar.setObjectName("topBar")
        topbar.setFixedHeight(52)
        top_layout = QHBoxLayout(topbar)
        top_layout.setContentsMargins(14, 6, 14, 6)
        brand = QLabel("智能空管")
        brand.setStyleSheet("color: #59cef1; font-size: 16px; font-weight: 750;")
        product = QLabel("冲突解脱验证平台")
        product.setStyleSheet("color: #f0f5f8; font-size: 15px; font-weight: 700;")
        current_view = QLabel("  态势监视  ")
        current_view.setStyleSheet(
            "color: #8ee3ff; background: #102b37; border-bottom: 2px solid #36b9df; "
            "padding: 8px 14px; font-weight: 650;"
        )
        flow = QLabel("冲突检测   ›   决策生成   ›   指令执行   ›   安全验证")
        flow.setStyleSheet("color: #718597; padding-left: 18px;")
        self.header_clock = QLabel("UTC --:--:--")
        self.header_clock.setStyleSheet("color: #9eb0bd;")
        self.system_state = QLabel("● 就绪")
        top_layout.addWidget(brand)
        top_layout.addWidget(product)
        top_layout.addSpacing(28)
        top_layout.addWidget(current_view)
        top_layout.addWidget(flow)
        top_layout.addStretch(1)
        top_layout.addWidget(self.header_clock)
        top_layout.addSpacing(12)
        top_layout.addWidget(self.system_state)
        root.addWidget(topbar)

        body = QHBoxLayout()
        body.setContentsMargins(6, 6, 6, 5)
        body.setSpacing(6)

        left = QFrame(self)
        left.setObjectName("sideRail")
        left.setFixedWidth(260)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 5, 8, 7)
        left_layout.setSpacing(5)

        scene_group = QGroupBox("场景与仿真")
        scene_layout = QVBoxLayout(scene_group)
        scene_layout.setSpacing(6)
        scene_layout.addWidget(QLabel("地图场景"))
        self.map_combo = QComboBox(self)
        self.map_combo.addItem("三航路演示", "three_route_demo")
        self.map_combo.addItem("成都-重庆真实航路", "chengdu_chongqing_real")
        self.map_combo.setCurrentIndex(self.map_combo.findData(self.current_map_key))
        scene_layout.addWidget(self.map_combo)
        scene_meta = QHBoxLayout()
        scene_meta.addWidget(QLabel("航空器数量上限"))
        scene_meta.addStretch(1)
        max_ac = QLabel(str(self.MAX_AIRCRAFT))
        max_ac.setObjectName("fieldValue")
        scene_meta.addWidget(max_ac)
        scene_layout.addLayout(scene_meta)
        self.load_map_btn = QPushButton("加载场景")
        scene_layout.addWidget(self.load_map_btn)
        sim_buttons = QHBoxLayout()
        self.start_btn = QPushButton("启动")
        self.start_btn.setObjectName("primaryAction")
        self.hold_btn = QPushButton("暂停")
        self.reset_btn = QPushButton("重置")
        sim_buttons.addWidget(self.start_btn)
        sim_buttons.addWidget(self.hold_btn)
        sim_buttons.addWidget(self.reset_btn)
        scene_layout.addLayout(sim_buttons)
        self.stop_btn = QPushButton("停止自动交通")
        self.stop_btn.setObjectName("dangerAction")
        scene_layout.addWidget(self.stop_btn)
        left_layout.addWidget(scene_group)

        param_group = QGroupBox("决策参数")
        form = QFormLayout(param_group)
        form.setContentsMargins(8, 10, 8, 7)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(7)
        self.preference_combo = QComboBox(self)
        self.preference_combo.addItems(["altitude_first", "speed_first"])
        form.addRow("解脱偏好", self.preference_combo)
        self.min_sep_spin = QDoubleSpinBox(self)
        self.min_sep_spin.setRange(3.0, 20.0)
        self.min_sep_spin.setDecimals(1)
        self.min_sep_spin.setSingleStep(0.5)
        self.min_sep_spin.setValue(self.HSEP_NM)
        self.min_sep_spin.setSuffix(" NM")
        form.addRow("水平间隔", self.min_sep_spin)
        lookahead = QLabel("20 min")
        lookahead.setObjectName("fieldValue")
        form.addRow("预测窗口", lookahead)
        vsep = QLabel("1000 ft")
        vsep.setObjectName("fieldValue")
        form.addRow("垂直间隔", vsep)
        self.llm_combo = QComboBox(self)
        self.llm_combo.addItems(["template_explainer", "openai_compatible_api", "off"])
        form.addRow("LLM 包装器", self.llm_combo)
        left_layout.addWidget(param_group)

        tool_group = QGroupBox("快捷工具")
        tool_layout = QVBoxLayout(tool_group)
        tool_row1 = QHBoxLayout()
        self.spawn_btn = QPushButton("放行一架")
        self.detect_btn = QPushButton("立即检测")
        tool_row1.addWidget(self.spawn_btn)
        tool_row1.addWidget(self.detect_btn)
        tool_row2 = QHBoxLayout()
        self.op_btn = QPushButton("继续运行")
        self.fast_btn = QPushButton("快进 2 分钟")
        tool_row2.addWidget(self.op_btn)
        tool_row2.addWidget(self.fast_btn)
        tool_layout.addLayout(tool_row1)
        tool_layout.addLayout(tool_row2)
        left_layout.addWidget(tool_group)
        left_layout.addStretch(1)

        center = QFrame(self)
        center.setObjectName("sideRail")
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)
        kpi_bar = QFrame(center)
        kpi_bar.setObjectName("bottomRail")
        kpi_layout = QHBoxLayout(kpi_bar)
        kpi_layout.setContentsMargins(9, 5, 9, 5)
        kpi_layout.setSpacing(6)
        self.status_aircraft = QLabel("航空器 0/14")
        self.status_conflicts = QLabel("活动冲突 0")
        self.status_commands = QLabel("指令总数 0")
        self.status_execution = QLabel("执行中 0/0")
        self.status_safety = QLabel("间隔丧失 0")
        self.status_llm = QLabel("解释层 idle")
        for status in [self.status_aircraft, self.status_conflicts, self.status_commands,
                       self.status_execution, self.status_safety, self.status_llm]:
            status.setAlignment(Qt.AlignCenter)
            status.setMinimumWidth(100)
            kpi_layout.addWidget(status, 1)
        center_layout.addWidget(kpi_bar)
        if self.radarwidget is not None:
            self.radarwidget.setParent(center)
            self.radarwidget.setMinimumSize(640, 390)
            center_layout.addWidget(self.radarwidget, 1)

        right = QFrame(self)
        right.setObjectName("sideRail")
        right.setFixedWidth(390)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(7, 5, 7, 7)
        right_layout.setSpacing(6)

        conflict_group = QGroupBox("当前冲突")
        conflict_layout = QVBoxLayout(conflict_group)
        self.conflict_table = QTableWidget(0, 4, self)
        self.conflict_table.setHorizontalHeaderLabels(["航空器对", "CPA / TCPA", "当前间隔", "状态"])
        self.conflict_table.setAlternatingRowColors(True)
        self.conflict_table.verticalHeader().setVisible(False)
        self.conflict_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.conflict_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.conflict_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.conflict_table.setMinimumHeight(170)
        conflict_layout.addWidget(self.conflict_table)
        right_layout.addWidget(conflict_group, 3)

        process_group = QGroupBox("决策过程")
        process_layout = QVBoxLayout(process_group)
        process_layout.setContentsMargins(7, 8, 7, 8)
        process_layout.setSpacing(5)
        self.decision_stage_widgets = {}

        def add_decision_stage(key, number, title):
            frame = QFrame(self)
            frame.setObjectName("decisionStage")
            stage_layout = QVBoxLayout(frame)
            stage_layout.setContentsMargins(8, 5, 8, 5)
            stage_layout.setSpacing(2)
            header = QHBoxLayout()
            title_label = QLabel("%s  %s" % (number, title))
            title_label.setObjectName("stageTitle")
            status_label = QLabel("等待")
            status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            status_label.setStyleSheet("color: #718493;")
            header.addWidget(title_label)
            header.addStretch(1)
            header.addWidget(status_label)
            detail_label = QLabel("等待新的冲突决策")
            detail_label.setObjectName("stageDetail")
            detail_label.setWordWrap(True)
            stage_layout.addLayout(header)
            stage_layout.addWidget(detail_label)
            process_layout.addWidget(frame)
            self.decision_stage_widgets[key] = (status_label, detail_label)

        add_decision_stage("input", "①", "冲突输入")
        add_decision_stage("candidates", "②", "候选方案")
        add_decision_stage("verification", "③", "验证与选择")
        right_layout.addWidget(process_group, 4)

        decision_group = QGroupBox("决策与验证")
        decision_layout = QFormLayout(decision_group)
        decision_layout.setHorizontalSpacing(7)
        decision_layout.setVerticalSpacing(5)

        def decision_field(text="-"):
            label = QLabel(text)
            label.setObjectName("fieldValue")
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            return label

        self.decision_aircraft_value = decision_field()
        self.decision_action_value = decision_field()
        self.decision_target_value = decision_field()
        self.decision_command_value = decision_field()
        self.decision_verify_value = decision_field("等待决策")
        decision_layout.addRow("目标航空器", self.decision_aircraft_value)
        decision_layout.addRow("机动类型", self.decision_action_value)
        decision_layout.addRow("目标参数", self.decision_target_value)
        decision_layout.addRow("仿真指令", self.decision_command_value)
        decision_layout.addRow("前向验证", self.decision_verify_value)
        self.decision_summary = QLabel("当前无待执行解脱指令")
        self.decision_summary.setWordWrap(True)
        self.decision_summary.setStyleSheet(
            "color: #9eb1bf; background: #0d1820; border-left: 3px solid #2b8cab; padding: 6px;"
        )
        decision_layout.addRow("决策理由", self.decision_summary)
        self.reverify_btn = QPushButton("重新检测并验证")
        decision_layout.addRow(self.reverify_btn)
        right_layout.addWidget(decision_group, 4)

        body.addWidget(left)
        body.addWidget(center, 1)
        body.addWidget(right)
        root.addLayout(body, 1)

        bottom = QFrame(self)
        bottom.setObjectName("bottomRail")
        bottom.setFixedHeight(225)
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(7, 3, 7, 6)
        bottom_layout.setSpacing(7)

        aircraft_group = QGroupBox("航班状态 (0)")
        self.aircraft_group = aircraft_group
        aircraft_layout = QVBoxLayout(aircraft_group)
        self.aircraft_table = QTableWidget(0, 6, self)
        self.aircraft_table.setHorizontalHeaderLabels(["呼号", "航路", "高度", "速度", "航向", "状态"])
        self.aircraft_table.setAlternatingRowColors(True)
        self.aircraft_table.verticalHeader().setVisible(False)
        self.aircraft_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.aircraft_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.aircraft_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        aircraft_layout.addWidget(self.aircraft_table)
        bottom_layout.addWidget(aircraft_group, 5)

        command_group = QGroupBox("指令执行记录")
        command_layout = QVBoxLayout(command_group)
        self.command_table = QTableWidget(0, 6, self)
        self.command_table.setHorizontalHeaderLabels(["时间", "航空器", "类型", "仿真指令", "规范指令 / 决策理由", "执行状态"])
        self.command_table.setAlternatingRowColors(True)
        self.command_table.verticalHeader().setVisible(False)
        self.command_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.command_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.command_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.command_table.setColumnWidth(0, 75)
        self.command_table.setColumnWidth(1, 75)
        self.command_table.setColumnWidth(2, 65)
        self.command_table.setColumnWidth(3, 170)
        self.command_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.command_table.setColumnWidth(5, 165)
        command_layout.addWidget(self.command_table)
        bottom_layout.addWidget(command_group, 8)
        root.addWidget(bottom)

        self.text = QTextEdit(self)
        self.text.setReadOnly(True)
        self.text.setMinimumHeight(72)
        self.text.setMaximumHeight(100)
        self.text.setLineWrapMode(QTextEdit.NoWrap)
        self.text.setStyleSheet(
            "background: #070c11; color: #9fb2c0; border: 1px solid #263746; "
            "font-family: Consolas, monospace; font-size: 11px;"
        )
        self.text.setVisible(False)
        root.addWidget(self.text)

        footer = QFrame(self)
        footer.setObjectName("bottomRail")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(8, 3, 8, 3)
        self.log_toggle_btn = QPushButton("系统日志")
        self.console_toggle_btn = QPushButton("仿真控制台")
        self.status_cycles = QLabel("检测周期 0")
        self.status_log = QLabel("日志 -")
        footer_layout.addWidget(self.log_toggle_btn)
        footer_layout.addWidget(self.console_toggle_btn)
        footer_layout.addStretch(1)
        footer_layout.addWidget(self.status_cycles)
        footer_layout.addSpacing(12)
        footer_layout.addWidget(self.status_log)
        root.addWidget(footer)

        self.map_info = QLabel("")
        self.map_info.setVisible(False)
        self.reset_btn.clicked.connect(self.reset_sector)
        self.start_btn.clicked.connect(self.start_auto_traffic)
        self.stop_btn.clicked.connect(self.stop_auto_traffic)
        self.spawn_btn.clicked.connect(self.spawn_random_aircraft)
        self.detect_btn.clicked.connect(self.detect_and_resolve)
        self.reverify_btn.clicked.connect(self.detect_and_resolve)
        self.op_btn.clicked.connect(self.operate_simulation)
        self.fast_btn.clicked.connect(self.fast_forward_demo)
        self.hold_btn.clicked.connect(lambda: self._stack("HOLD"))
        self.load_map_btn.clicked.connect(self.load_selected_map)
        self.min_sep_spin.valueChanged.connect(self.on_min_separation_changed)
        self.log_toggle_btn.clicked.connect(self._toggle_system_log)
        self.console_toggle_btn.clicked.connect(self._toggle_bluesky_console)
        self._update_map_info()

    def _build_ui_legacy(self):
        self.setObjectName("aiAssistPanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 7, 10, 7)
        layout.setSpacing(6)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(320)
        self.setStyleSheet("""
            QWidget#aiAssistPanel { background: #0b1118; }
            QWidget { background: #0b1118; color: #d8e1ea; font-size: 12px; }
            QLabel { background: transparent; color: #c8d2dc; }
            QGroupBox {
                color: #dce6ef; border: 1px solid #2b3947; border-radius: 3px;
                margin-top: 9px; padding-top: 8px; font-weight: 600;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 9px; padding: 0 4px; }
            QPushButton {
                color: #dce6ef; background: #18232e; border: 1px solid #344655;
                border-radius: 3px; padding: 5px 10px; min-height: 18px;
            }
            QPushButton:hover { background: #22313d; border-color: #4b667a; }
            QPushButton:pressed { background: #111a22; }
            QPushButton#primaryAction { background: #087ea4; border-color: #20a7cf; color: #ffffff; }
            QPushButton#dangerAction { color: #ffb2ad; border-color: #70413f; }
            QComboBox, QDoubleSpinBox {
                color: #dce6ef; background: #111a23; border: 1px solid #344655;
                border-radius: 2px; padding: 4px 7px; min-height: 20px;
            }
            QComboBox::drop-down { border: 0; width: 18px; }
            QTableWidget {
                background: #0e151d; alternate-background-color: #111b24;
                color: #d4dde6; border: 0; gridline-color: #263440;
                selection-background-color: #173b4b; selection-color: #ffffff;
            }
            QHeaderView::section {
                background: #151f29; color: #9fb0c0; border: 0;
                border-right: 1px solid #263440; border-bottom: 1px solid #344655;
                padding: 5px; font-weight: 600;
            }
            QScrollBar:vertical { background: #0e151d; width: 10px; margin: 0; }
            QScrollBar::handle:vertical { background: #3a4b5a; min-height: 24px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        """)

        header_row = QHBoxLayout()
        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        title = QLabel("智能空管冲突解脱验证平台")
        title.setStyleSheet("font-weight: 700; color: #f2f6fa; font-size: 14px;")
        subtitle = QLabel("动态扇区冲突检测 / 决策 / 执行验证")
        subtitle.setStyleSheet("color: #73879a; font-size: 11px;")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header_row.addLayout(title_box)
        header_row.addStretch(1)
        self.system_state = QLabel("● 就绪")
        self.system_state.setStyleSheet(
            "color: #8fa2b3; background: #111a23; border: 1px solid #344655; "
            "border-radius: 3px; padding: 5px 10px; font-weight: 600;"
        )
        header_row.addWidget(self.system_state)
        layout.addLayout(header_row)

        btnrow = QHBoxLayout()
        btnrow.setSpacing(5)
        self.reset_btn = QPushButton("重置场景")
        self.start_btn = QPushButton("启动交通")
        self.start_btn.setObjectName("primaryAction")
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setObjectName("dangerAction")
        self.spawn_btn = QPushButton("放行一架")
        self.detect_btn = QPushButton("立即检测")
        self.op_btn = QPushButton("运行")
        self.fast_btn = QPushButton("快进 2 分钟")
        self.hold_btn = QPushButton("暂停")
        for btn in [self.reset_btn, self.start_btn, self.stop_btn, self.spawn_btn, self.detect_btn, self.op_btn, self.fast_btn, self.hold_btn]:
            btnrow.addWidget(btn)
        btnrow.addStretch(1)
        layout.addLayout(btnrow)

        option_row = QHBoxLayout()
        option_row.setSpacing(6)
        option_row.addWidget(QLabel("场景地图"))
        self.map_combo = QComboBox(self)
        self.map_combo.addItem("三航路演示", "three_route_demo")
        self.map_combo.addItem("成都-重庆真实航路", "chengdu_chongqing_real")
        self.map_combo.setCurrentIndex(self.map_combo.findData(self.current_map_key))
        option_row.addWidget(self.map_combo)
        self.load_map_btn = QPushButton("加载地图")
        option_row.addWidget(self.load_map_btn)
        option_row.addWidget(QLabel("决策偏好"))
        self.preference_combo = QComboBox(self)
        self.preference_combo.addItems(["altitude_first", "speed_first"])
        option_row.addWidget(self.preference_combo)
        option_row.addWidget(QLabel("最小间隔"))
        self.min_sep_spin = QDoubleSpinBox(self)
        self.min_sep_spin.setRange(3.0, 20.0)
        self.min_sep_spin.setDecimals(1)
        self.min_sep_spin.setSingleStep(0.5)
        self.min_sep_spin.setValue(self.HSEP_NM)
        self.min_sep_spin.setSuffix(" NM")
        self.min_sep_spin.setToolTip("Required horizontal separation in nautical miles. The detector and verifier use this value immediately.")
        option_row.addWidget(self.min_sep_spin)
        option_row.addWidget(QLabel("解释层"))
        self.llm_combo = QComboBox(self)
        self.llm_combo.addItems(["template_explainer", "openai_compatible_api", "off"])
        option_row.addWidget(self.llm_combo)
        option_row.addStretch(1)
        layout.addLayout(option_row)

        status_row = QHBoxLayout()
        status_row.setSpacing(5)
        self.status_aircraft = QLabel("航空器 0/14")
        self.status_cycles = QLabel("检测周期 0")
        self.status_conflicts = QLabel("活动冲突 0")
        self.status_commands = QLabel("决策指令 0")
        self.status_execution = QLabel("执行进度 0/0")
        self.status_safety = QLabel("失去间隔 0")
        self.status_llm = QLabel("解释层 idle")
        self.status_log = QLabel("日志 -")
        for item in [
            self.status_aircraft,
            self.status_conflicts,
            self.status_commands,
            self.status_execution,
            self.status_safety,
            self.status_llm,
        ]:
            item.setStyleSheet(
                "color: #b8c5d0; background: #111a23; border: 1px solid #2e3d4a; "
                "border-radius: 3px; padding: 4px 9px;"
            )
            status_row.addWidget(item)
        status_row.addStretch(1)
        layout.addLayout(status_row)

        self.decision_summary = QLabel("最新决策  |  当前无待执行解脱指令")
        self.decision_summary.setWordWrap(False)
        self.decision_summary.setStyleSheet(
            "color: #b8c7d4; background: #101922; border-left: 3px solid #2c91b3; "
            "padding: 5px 9px;"
        )
        layout.addWidget(self.decision_summary)

        self.reset_btn.clicked.connect(self.reset_sector)
        self.start_btn.clicked.connect(self.start_auto_traffic)
        self.stop_btn.clicked.connect(self.stop_auto_traffic)
        self.spawn_btn.clicked.connect(self.spawn_random_aircraft)
        self.detect_btn.clicked.connect(self.detect_and_resolve)
        self.op_btn.clicked.connect(self.operate_simulation)
        self.fast_btn.clicked.connect(self.fast_forward_demo)
        self.hold_btn.clicked.connect(lambda: self._stack("HOLD"))
        self.load_map_btn.clicked.connect(self.load_selected_map)
        self.min_sep_spin.valueChanged.connect(self.on_min_separation_changed)
        self._update_map_info()

        data_row = QHBoxLayout()
        data_row.setSpacing(7)

        conflict_group = QGroupBox("当前冲突 · 状态持续更新")
        conflict_layout = QVBoxLayout(conflict_group)
        self.conflict_table = QTableWidget(0, 6, self)
        self.conflict_table.setHorizontalHeaderLabels(["航空器对", "CPA 时间", "CPA 间隔", "当前间隔", "状态", "关联指令"])
        self.conflict_table.setMinimumHeight(145)
        self.conflict_table.setMaximumHeight(205)
        self.conflict_table.setAlternatingRowColors(True)
        self.conflict_table.verticalHeader().setVisible(False)
        self.conflict_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.conflict_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.conflict_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.conflict_table.setColumnWidth(0, 125)
        self.conflict_table.setColumnWidth(1, 78)
        self.conflict_table.setColumnWidth(2, 112)
        self.conflict_table.setColumnWidth(3, 112)
        self.conflict_table.setColumnWidth(4, 82)
        self.conflict_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        conflict_layout.addWidget(self.conflict_table)
        data_row.addWidget(conflict_group, 5)

        command_group = QGroupBox("指令执行记录")
        command_layout = QVBoxLayout(command_group)
        self.command_table = QTableWidget(0, 6, self)
        self.command_table.setHorizontalHeaderLabels(["时间", "航空器", "类型", "仿真指令", "规范指令 / 决策理由", "执行状态"])
        self.command_table.setMinimumHeight(145)
        self.command_table.setMaximumHeight(205)
        self.command_table.setAlternatingRowColors(True)
        self.command_table.verticalHeader().setVisible(False)
        self.command_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.command_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.command_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.command_table.setColumnWidth(0, 72)
        self.command_table.setColumnWidth(1, 72)
        self.command_table.setColumnWidth(2, 62)
        self.command_table.setColumnWidth(3, 145)
        self.command_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.command_table.setColumnWidth(5, 160)
        command_layout.addWidget(self.command_table)
        data_row.addWidget(command_group, 7)
        layout.addLayout(data_row)

        self.text = QTextEdit(self)
        self.text.setReadOnly(True)
        self.text.setMinimumHeight(72)
        self.text.setMaximumHeight(90)
        self.text.setLineWrapMode(QTextEdit.NoWrap)
        self.text.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.text.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.text.setStyleSheet(
            "background: #080d12; color: #9fb2c0; border: 1px solid #263440; "
            "font-family: Consolas, monospace; font-size: 11px;"
        )
        self.text.setVisible(False)
        layout.addWidget(self.text)

        footer_row = QHBoxLayout()
        footer_row.setSpacing(8)
        self.log_toggle_btn = QPushButton("显示系统日志")
        self.console_toggle_btn = QPushButton("显示仿真控制台")
        self.log_toggle_btn.clicked.connect(self._toggle_system_log)
        self.console_toggle_btn.clicked.connect(self._toggle_bluesky_console)
        footer_row.addWidget(self.log_toggle_btn)
        footer_row.addWidget(self.console_toggle_btn)
        footer_row.addWidget(self.status_cycles)
        footer_row.addWidget(self.status_log, 1)
        layout.addLayout(footer_row)

    def _toggle_system_log(self):
        visible = not self.text.isVisible()
        self.text.setVisible(visible)
        self.log_toggle_btn.setText("隐藏系统日志" if visible else "显示系统日志")

    def _toggle_bluesky_console(self):
        if self.console is None or self.window() is None:
            return
        dock = getattr(self.window(), "dockWidget", None)
        if dock is None:
            return
        visible = not dock.isVisible()
        dock.setVisible(visible)
        self.console.setVisible(True)
        self.console_toggle_btn.setText("隐藏仿真控制台" if visible else "显示仿真控制台")

    @staticmethod
    def _queued_command(item):
        return item.get("command", "") if isinstance(item, dict) else str(item)

    def _stack(self, command, monitor_record=None):
        now_wall = datetime.now().timestamp()
        item = {
            "command": command,
            "queued_wall_time": now_wall,
            "queued_monotonic": monotonic(),
            "monitor_record": monitor_record,
        }
        head = command.strip().split(" ", 1)[0].upper()
        if head in {"ALT", "SPD"}:
            insert_at = 0
            while insert_at < len(self.command_queue):
                queued_head = self._queued_command(self.command_queue[insert_at]).strip().split(" ", 1)[0].upper()
                if queued_head not in {"ALT", "SPD"}:
                    break
                insert_at += 1
            self.command_queue.insert(insert_at, item)
        else:
            self.command_queue.append(item)
        if monitor_record is not None:
            monitor_record["queued_time"] = now_wall
            monitor_record["status"] = "queued"
            self._set_command_status(monitor_record["row"], "queued")
            self._append_log({
                "event": "command_queued",
                "aircraft": monitor_record["acid"],
                "command": command,
                "sim_time": self.latest_sim_time,
                "queue_len": len(self.command_queue),
            })
        if not self.command_timer.isActive():
            self.command_timer.start()
        return item

    def _drain_command_queue(self):
        if not self.command_queue:
            self.command_timer.stop()
            return
        item = self.command_queue.pop(0)
        command = self._queued_command(item)
        monitor_record = item.get("monitor_record") if isinstance(item, dict) else None
        sent = False
        if getattr(bs, "net", None):
            target = bs.net.actnode()
            if target:
                started = monotonic()
                try:
                    bs.net.send_event(b"STACKCMD", command, target=target)
                    sent = True
                except Exception as exc:
                    self._append_log({"event": "command_send_failed", "command": command, "error": str(exc)})
                    if monitor_record is not None:
                        monitor_record["status"] = "send_failed"
                        self._set_command_status(monitor_record["row"], "send failed")
                elapsed = monotonic() - started
                if elapsed > 0.25:
                    self._append_log({
                        "event": "slow_command_send",
                        "command": command,
                        "elapsed_sec": round(elapsed, 3),
                        "queue_len": len(self.command_queue),
                    })
        if self.console is not None and not sent:
            self.console.stack(command)
            sent = True
        if sent and monitor_record is not None:
            sent_wall = datetime.now().timestamp()
            queued_wall = item.get("queued_wall_time", sent_wall) if isinstance(item, dict) else sent_wall
            monitor_record["sent_time"] = sent_wall
            monitor_record["sent_sim_time"] = self.latest_sim_time
            monitor_record["status"] = "sent"
            self._set_command_status(monitor_record["row"], "sent; waiting execution")
            self._append_log({
                "event": "command_sent",
                "aircraft": monitor_record["acid"],
                "command": command,
                "sim_time": self.latest_sim_time,
                "queue_delay_sec": round(sent_wall - queued_wall, 3),
                "queue_len": len(self.command_queue),
            })
        if self.console is not None and sent and self._should_echo_command(command) and hasattr(self.console, "echo"):
            self.console.echo("[%s] SENT CMD: %s" % (self._now(), command))

    def _should_echo_command(self, command):
        head = command.strip().split(" ", 1)[0].upper()
        return head in {"ALT", "SPD"}

    def _ensure_stream_connection(self):
        net = getattr(bs, "net", None)
        if not net or net is self._connected_net:
            return
        if hasattr(net, "stream_received"):
            net.stream_received.connect(self.on_simstream_received)
            self._connected_net = net

    def _now(self):
        return datetime.now().strftime("%H:%M:%S")

    def _new_log_file(self):
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = self.LOG_DIR / ("dynamic_sector_%s.jsonl" % stamp)
        self._append_log({"event": "log_started", "scenario": self.SCENARIO_NAME})

    def _append_log(self, record):
        if self.log_path is None:
            return
        record = dict(record)
        record.setdefault("wall_time", datetime.now().isoformat(timespec="seconds"))
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    def _add_row(self, event, aircraft, cpa, decision, command, status):
        if event in {"spawn", "sector", "control"}:
            return
        self._add_command_row(aircraft, event, command, decision, status)

    def _set_decision_stage(self, key, status, detail, tone="info"):
        widgets = getattr(self, "decision_stage_widgets", {}).get(key)
        if not widgets:
            return
        status_label, detail_label = widgets
        colors = {
            "pending": "#718493",
            "info": "#58c7e8",
            "success": "#69d39a",
            "warning": "#f2c15f",
            "danger": "#ff7770",
        }
        status_label.setText(str(status))
        status_label.setStyleSheet("color: %s; font-weight: 650;" % colors.get(tone, colors["info"]))
        detail_label.setText(str(detail))
        detail_label.setToolTip(str(detail))

    def _reset_decision_process(self):
        for key in ("input", "candidates", "verification"):
            self._set_decision_stage(key, "等待", "等待新的冲突决策", "pending")

    @staticmethod
    def _candidate_display_text(candidate):
        acid = str(candidate.get("aircraft", "-"))
        kind = str(candidate.get("kind", "hold"))
        if kind == "altitude":
            return "%s→FL%d" % (acid, int(candidate.get("target_fl", 0)))
        if kind == "speed":
            return "%s→%d kt" % (acid, int(candidate.get("target_speed", 0)))
        return "%s保持" % acid

    def _update_decision_process(self, detections, actions, solver):
        if not detections:
            return
        tcpa, hsep, vsep, a, b, _pair = detections[0]
        self._set_decision_stage(
            "input",
            "已识别",
            "%s↔%s｜CPA %.1f NM / %.0f ft｜TCPA %.1f min｜阈值 %.1f NM / %.0f ft"
            % (a["id"], b["id"], hsep, vsep, tcpa, self._min_hsep_nm(), self.VERIFY_VSEP_FT),
            "warning",
        )

        candidates = list(solver.get("candidate_actions", []))
        if not candidates:
            candidates = [dict(action) for action in actions]
        selected = solver.get("selected_actions", {})
        candidates.sort(key=lambda item: (
            0 if selected.get(item.get("aircraft")) == item.get("label") else 1,
            1 if item.get("kind") == "hold" else 0,
        ))
        preview = "；".join(self._candidate_display_text(item) for item in candidates[:3])
        candidate_count = int(solver.get("candidate_count", len(candidates) or len(actions)))
        if not preview:
            preview = "未生成可执行候选"
        self._set_decision_stage(
            "candidates",
            "已生成" if candidate_count else "无候选",
            "共 %d 个候选｜%s" % (candidate_count, preview),
            "info" if candidate_count else "danger",
        )

        preference = "高度优先" if solver.get("preference") == "altitude_first" else "速度优先"
        checks = int(solver.get("pair_checks", 0))
        rejected = int(solver.get("rejected_pair_checks", 0))
        if solver.get("success"):
            chosen = "；".join(self._candidate_display_text(action) for action in actions)
            if not chosen:
                chosen = "保持当前已验证方案"
            self._set_decision_stage(
                "verification",
                "已通过",
                "%s｜验证 %d 组，淘汰 %d 组｜选择 %s" % (preference, checks, rejected, chosen),
                "success",
            )
        else:
            self._set_decision_stage(
                "verification",
                "未通过",
                "%s｜验证 %d 组｜未找到满足约束的方案" % (preference, checks),
                "danger",
            )

    def _add_command_row(self, aircraft, command_type, command, instruction, status):
        row = self.command_table.rowCount()
        self.command_table.insertRow(row)
        values = [self._now(), aircraft, command_type, command, instruction, status]
        for col, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            item.setTextAlignment(Qt.AlignCenter)
            item.setToolTip(str(value))
            self.command_table.setItem(row, col, item)
        if hasattr(self, "decision_summary"):
            self.decision_aircraft_value.setText(str(aircraft))
            command_head = str(command).strip().split(" ", 1)[0].upper()
            if command_head == "ALT":
                action_text = "高度调整"
                parts = str(command).split(",")
                target_text = parts[1] if len(parts) > 1 else "-"
            elif command_head == "SPD":
                action_text = "速度调整"
                parts = str(command).split(",")
                target_text = (parts[1] + " kt") if len(parts) > 1 else "-"
            else:
                action_text = str(command_type)
                target_text = "-"
            self.decision_action_value.setText(action_text)
            self.decision_target_value.setText(target_text)
            self.decision_command_value.setText(str(command))
            self.decision_command_value.setToolTip(str(command))
            self.decision_verify_value.setText("前向验证通过")
            self.decision_verify_value.setStyleSheet(
                "color: #75d9a5; background: #14271f; border: 1px solid #2d664b; "
                "border-radius: 2px; padding: 5px 7px; font-weight: 650;"
            )
            self.decision_summary.setText(str(instruction))
            self.decision_summary.setToolTip(str(instruction))
        self.command_table.scrollToBottom()
        self._update_status_labels()
        return row

    def _register_command_monitor(self, row, action):
        if row is None or action.get("kind") not in {"altitude", "speed"}:
            return None
        now_wall = datetime.now().timestamp()
        for previous in self.command_records:
            if (
                previous.get("acid") == action["acid"]
                and previous.get("kind") == action["kind"]
                and previous.get("status") not in {"reached", "superseded"}
            ):
                previous["status"] = "superseded"
                previous["superseded_time"] = now_wall
                self._set_command_status(previous["row"], "superseded by newer command")
                self._append_log({
                    "event": "command_superseded",
                    "aircraft": previous["acid"],
                    "old_command": previous.get("command"),
                    "new_command": action.get("command"),
                    "sim_time": self.latest_sim_time,
                })
        state = self.latest_acdata.get(action["acid"], {})
        record = {
            "row": row,
            "acid": action["acid"],
            "kind": action["kind"],
            "target_fl": action.get("target_fl"),
            "target_speed": action.get("target_speed"),
            "command": action.get("command"),
            "decision_time": now_wall,
            "decision_sim_time": self.latest_sim_time,
            "initial_alt_ft": state.get("alt_ft"),
            "initial_speed_kt": self._speed_kt(state) if state else None,
            "last_alt_ft": state.get("alt_ft"),
            "reached": False,
            "status": "planned",
        }
        self.command_records.append(record)
        return record

    def _set_command_status(self, row, status):
        if not hasattr(self, "command_table") or row >= self.command_table.rowCount():
            return
        if self._last_command_status_by_row.get(row) == status:
            return
        item = QTableWidgetItem(status)
        item.setTextAlignment(Qt.AlignCenter)
        item.setToolTip(status)
        status_lower = str(status).lower()
        if "reached" in status_lower or "verified" in status_lower or "完成" in status_lower:
            item.setForeground(QColor("#63d69a"))
        elif "execut" in status_lower or "waiting" in status_lower or "sent" in status_lower:
            item.setForeground(QColor("#52bde2"))
        elif "failed" in status_lower or "blocked" in status_lower:
            item.setForeground(QColor("#ff7770"))
        elif "superseded" in status_lower:
            item.setForeground(QColor("#7f8d99"))
        self.command_table.setItem(row, 5, item)
        self._last_command_status_by_row[row] = status

    def _update_command_execution_statuses(self):
        if not hasattr(self, "command_table"):
            return
        now_ts = monotonic()
        if now_ts - self._last_execution_update_ts < self.EXECUTION_UPDATE_MIN_SEC:
            return
        self._last_execution_update_ts = now_ts
        if not self.command_records:
            if hasattr(self, "status_execution"):
                self.status_execution.setText("执行进度 0/0")
                self._set_status_tone(self.status_execution, "neutral")
            return
        reached = 0
        tracked = 0
        for record in self.command_records:
            if record.get("status") == "superseded":
                continue
            state = self.latest_acdata.get(record["acid"])
            if state is None:
                self._set_command_status(record["row"], "waiting ACDATA")
                continue
            tracked += 1
            if not record.get("sent_time"):
                self._set_command_status(record["row"], record.get("status", "queued"))
                continue
            just_started = False
            just_reached = False
            if record["kind"] == "altitude":
                target_alt_ft = float(record["target_fl"]) * 100.0
                current_fl = state["alt_ft"] / 100.0
                diff_ft = abs(state["alt_ft"] - target_alt_ft)
                initial_alt_ft = record.get("initial_alt_ft")
                moved_ft = abs(state["alt_ft"] - initial_alt_ft) if initial_alt_ft is not None else 0.0
                toward_target = (target_alt_ft - state["alt_ft"]) * state.get("vs_fpm", 0.0) >= 0.0
                if not record.get("started_time") and (
                    moved_ft >= self.EXECUTION_START_ALT_DELTA_FT
                    or (abs(state.get("vs_fpm", 0.0)) >= 100.0 and toward_target)
                ):
                    record["started_time"] = datetime.now().timestamp()
                    record["started_sim_time"] = self.latest_sim_time
                    record["status"] = "executing"
                    just_started = True
                previous_alt_ft = record.get("last_alt_ft")
                crossed_target = (
                    previous_alt_ft is not None
                    and (previous_alt_ft - target_alt_ft) * (state["alt_ft"] - target_alt_ft) <= 0.0
                    and abs(previous_alt_ft - state["alt_ft"]) > 0.0
                )
                if diff_ft <= self.ALT_REACHED_TOLERANCE_FT or crossed_target:
                    record["reached"] = True
                    just_reached = record.get("status") != "reached"
                    record["status"] = "reached"
                record["last_alt_ft"] = state["alt_ft"]
                phase = "reached" if record["reached"] else ("executing" if record.get("started_time") else "sent")
                status = "FL%.1f->%d %s" % (current_fl, record["target_fl"], phase)
            else:
                current_speed = self._speed_kt(state)
                diff_kt = abs(current_speed - int(record["target_speed"]))
                initial_speed = record.get("initial_speed_kt")
                moved_kt = abs(current_speed - initial_speed) if initial_speed is not None else 0.0
                if not record.get("started_time") and moved_kt >= self.EXECUTION_START_SPEED_DELTA_KT:
                    record["started_time"] = datetime.now().timestamp()
                    record["started_sim_time"] = self.latest_sim_time
                    record["status"] = "executing"
                    just_started = True
                if diff_kt <= self.SPEED_REACHED_TOLERANCE_KT:
                    record["reached"] = True
                    just_reached = record.get("status") != "reached"
                    record["status"] = "reached"
                phase = "reached" if record["reached"] else ("executing" if record.get("started_time") else "sent")
                status = "CAS%d->%d %s" % (current_speed, record["target_speed"], phase)
            if just_started:
                self._append_log({
                    "event": "command_execution_started",
                    "aircraft": record["acid"],
                    "command": record["command"],
                    "sim_time": self.latest_sim_time,
                    "send_to_start_sec": round(record["started_time"] - record["sent_time"], 3),
                })
            if just_reached:
                record["reached_time"] = datetime.now().timestamp()
                record["reached_sim_time"] = self.latest_sim_time
                if record["kind"] == "altitude" and self.last_targets.get(record["acid"]) == record["target_fl"]:
                    self.last_targets.pop(record["acid"], None)
                if record["kind"] == "speed" and self.last_speeds.get(record["acid"]) == record["target_speed"]:
                    self.last_speeds.pop(record["acid"], None)
                self._append_log({
                    "event": "command_reached",
                    "aircraft": record["acid"],
                    "command": record["command"],
                    "sim_time": self.latest_sim_time,
                    "send_to_reached_sec": round(record["reached_time"] - record["sent_time"], 3),
                })
            if record["reached"]:
                reached += 1
            self._set_command_status(record["row"], status)
        if hasattr(self, "status_execution"):
            self.status_execution.setText("执行进度 %d/%d" % (reached, tracked))
            self._set_status_tone(self.status_execution, "success" if tracked and reached == tracked else "info")

    def _current_separation_summary(self):
        aircraft = [
            state for state in self.latest_acdata.values()
            if str(state.get("id", "")).startswith("DYN")
        ]
        if len(aircraft) < 2:
            return {"loss_count": 0, "min_hsep": None, "min_vsep": None}
        loss_count = 0
        min_hsep = None
        min_vsep = None
        required_hsep_nm = self._min_hsep_nm()
        for i, a in enumerate(aircraft):
            for b in aircraft[i + 1:]:
                ax, ay = self._xy_nm(a["lat"], a["lon"])
                bx, by = self._xy_nm(b["lat"], b["lon"])
                hsep = sqrt((bx - ax) * (bx - ax) + (by - ay) * (by - ay))
                vsep = abs(a["alt_ft"] - b["alt_ft"])
                min_hsep = hsep if min_hsep is None else min(min_hsep, hsep)
                min_vsep = vsep if min_vsep is None else min(min_vsep, vsep)
                if hsep < required_hsep_nm and vsep < self.VERIFY_VSEP_FT:
                    loss_count += 1
        return {"loss_count": loss_count, "min_hsep": min_hsep, "min_vsep": min_vsep}

    def _aircraft_has_pending_command(self, acid):
        for record in self.command_records:
            if (
                record.get("acid") == acid
                and not record.get("reached")
                and record.get("status") != "superseded"
            ):
                return True
        return False

    def _command_is_active(self, command):
        return any(
            record.get("command") == command
            and record.get("status") not in {"reached", "superseded", "send_failed"}
            for record in self.command_records
        )

    def _pair_has_pending_command(self, acid_a, acid_b):
        return self._aircraft_has_pending_command(acid_a) or self._aircraft_has_pending_command(acid_b)

    def _sync_tracked_conflicts(self, aircraft, detections, actions=None, default_state="Monitoring"):
        actions = actions or []
        state_by_id = {state["id"]: state for state in aircraft}
        detected_pairs = set()
        for tcpa, hsep, vsep, a, b, pair in detections:
            detected_pairs.add(pair)
            self.tracked_conflicts[pair] = {
                "ids": pair,
                "first_seen": self.tracked_conflicts.get(pair, {}).get("first_seen", self._now()),
                "last_state": default_state,
            }

        rows = []
        live_ids = set(state_by_id.keys())
        related_action_ids = {action["acid"] for action in actions if action.get("command")}
        now_ts = monotonic()
        for pair in list(self.tracked_conflicts.keys()):
            acid_a, acid_b = pair
            if acid_a not in live_ids or acid_b not in live_ids:
                self.tracked_conflicts.pop(pair, None)
                self._tracked_safe_cache.pop(pair, None)
                continue
            a = state_by_id[acid_a]
            b = state_by_id[acid_b]
            tcpa, hsep, vsep = self._cpa(a, b)
            ax, ay = self._xy_nm(a["lat"], a["lon"])
            bx, by = self._xy_nm(b["lat"], b["lon"])
            current_h = sqrt((bx - ax) * (bx - ax) + (by - ay) * (by - ay))
            current_v = abs(a["alt_ft"] - b["alt_ft"])
            current_loss = current_h < self._min_hsep_nm() and current_v < self.VERIFY_VSEP_FT
            pending = self._pair_has_pending_command(acid_a, acid_b)
            has_new_action = acid_a in related_action_ids or acid_b in related_action_ids

            if current_loss:
                row_state = "Loss"
            elif has_new_action:
                row_state = "Issued"
            elif pending:
                row_state = "Executing"
            elif pair in detected_pairs:
                row_state = default_state
            else:
                cached = self._tracked_safe_cache.get(pair)
                if cached and now_ts - cached[0] < self.TRACKED_SAFE_CHECK_INTERVAL_SEC:
                    future_safe = cached[1]
                else:
                    future_safe = self._current_targets_are_safe(a, b)
                    self._tracked_safe_cache[pair] = (now_ts, future_safe)
                if future_safe:
                    self.tracked_conflicts.pop(pair, None)
                    self._tracked_safe_cache.pop(pair, None)
                    continue
                row_state = "Monitoring"

            self.tracked_conflicts[pair]["last_state"] = row_state
            rows.append((tcpa, hsep, vsep, a, b, pair, row_state))
        rows.sort(key=lambda x: (x[0], x[1], x[5]))
        if len(rows) > self.MAX_TRACKED_CONFLICTS:
            keep_pairs = {row[5] for row in rows[:self.MAX_TRACKED_CONFLICTS]}
            for pair in list(self.tracked_conflicts.keys()):
                if pair not in keep_pairs:
                    self.tracked_conflicts.pop(pair, None)
            rows = rows[:self.MAX_TRACKED_CONFLICTS]
        return rows

    def _refresh_conflict_table(self, detections, actions=None, state="monitoring"):
        actions = actions or []
        action_by_aircraft = {action["acid"]: action["command"] for action in actions if action.get("command")}
        table_rows = []
        for item in detections:
            if len(item) >= 7:
                tcpa, hsep, vsep, a, b, _pair, row_state = item
            else:
                tcpa, hsep, vsep, a, b, _pair = item
                row_state = state
            acid_a = a["id"]
            acid_b = b["id"]
            pair_text = "%s - %s" % (acid_a, acid_b)
            ax, ay = self._xy_nm(a["lat"], a["lon"])
            bx, by = self._xy_nm(b["lat"], b["lon"])
            current_h = sqrt((bx - ax) * (bx - ax) + (by - ay) * (by - ay))
            current_v = abs(a["alt_ft"] - b["alt_ft"])
            related = []
            for acid in [acid_a, acid_b]:
                cmd = action_by_aircraft.get(acid) or self.last_command_by_aircraft.get(acid)
                if cmd and cmd not in related:
                    related.append(cmd)
            values = [
                pair_text,
                "%.1f NM\n%.1f min" % (hsep, tcpa),
                "%.1f NM\n%.0f ft" % (current_h, current_v),
                row_state,
            ]
            active_loss = current_h < self._min_hsep_nm() and current_v < self.VSEP_FT
            table_rows.append((values, active_loss))
        signature = tuple((tuple(values), active_loss) for values, active_loss in table_rows)
        self.active_conflict_count = len(table_rows)
        if signature == self._last_conflict_table_signature:
            return
        self._last_conflict_table_signature = signature
        self.conflict_table.setRowCount(0)
        for row, (values, active_loss) in enumerate(table_rows):
            self.conflict_table.insertRow(row)
            self.conflict_table.setRowHeight(row, 46)
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)
                item.setToolTip(str(value))
                if col in {0, 3}:
                    item.setForeground(QColor("#ff7770" if active_loss else "#f0b85a"))
                self.conflict_table.setItem(row, col, item)
        self._update_status_labels()

    def _refresh_aircraft_table(self):
        if not hasattr(self, "aircraft_table"):
            return
        states = [
            state for state in self.latest_acdata.values()
            if str(state.get("id", "")).startswith("DYN")
        ]
        states.sort(key=lambda state: str(state.get("id", "")))
        conflict_aircraft = set()
        for pair in self.tracked_conflicts:
            conflict_aircraft.update(pair)
        rows = []
        for state in states[:self.MAX_AIRCRAFT]:
            acid = str(state.get("id", "-"))
            meta = self.active_meta.get(acid, {})
            route = str(meta.get("route", "-"))
            if len(route) > 15:
                route = route[:14] + "…"
            values = [
                acid,
                route,
                "FL%.0f" % (float(state.get("alt_ft", 0.0)) / 100.0),
                "%d kt" % self._speed_kt(state),
                "%03d°" % (int(round(float(state.get("trk", state.get("hdg", 0.0))))) % 360),
                "冲突监控" if acid in conflict_aircraft else "正常运行",
            ]
            rows.append(values)
        signature = tuple(tuple(values) for values in rows)
        if signature == self._last_aircraft_table_signature:
            return
        self._last_aircraft_table_signature = signature
        self.aircraft_table.setRowCount(0)
        for row, values in enumerate(rows):
            self.aircraft_table.insertRow(row)
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)
                item.setToolTip(str(value))
                if col == 0:
                    item.setForeground(QColor("#59cef1"))
                elif col == 5:
                    item.setForeground(QColor("#f0b85a" if value == "冲突监控" else "#75d9a5"))
                self.aircraft_table.setItem(row, col, item)
        self.aircraft_group.setTitle("航班状态 (%d)" % len(rows))

    def _set_status_tone(self, label, tone):
        key = id(label)
        if self._status_tone_cache.get(key) == tone:
            return
        tones = {
            "neutral": ("#b8c5d0", "#111a23", "#2e3d4a"),
            "info": ("#8dd8ef", "#102631", "#28657a"),
            "warning": ("#f3c36b", "#2b2416", "#74571f"),
            "danger": ("#ff8b85", "#301919", "#7a3431"),
            "success": ("#75d9a5", "#14271f", "#2d664b"),
        }
        fg, bg, border = tones.get(tone, tones["neutral"])
        label.setStyleSheet(
            "color: %s; background: %s; border: 1px solid %s; "
            "border-radius: 3px; padding: 4px 9px; font-weight: 600;" % (fg, bg, border)
        )
        self._status_tone_cache[key] = tone

    def _update_status_labels(self):
        if not hasattr(self, "status_aircraft"):
            return
        now_ts = monotonic()
        if now_ts - self._last_status_update_ts < self.STATUS_UPDATE_MIN_SEC:
            return
        self._last_status_update_ts = now_ts
        live_count = len(self._live_dyn_ids())
        scheduled_count = len(self.active_meta)
        self._refresh_aircraft_table()
        sim_seconds = int(float(self.latest_sim_time or 0.0))
        sim_h = (sim_seconds // 3600) % 24
        sim_m = (sim_seconds % 3600) // 60
        sim_s = sim_seconds % 60
        self.header_clock.setText("仿真 UTC %02d:%02d:%02d" % (sim_h, sim_m, sim_s))
        self.status_aircraft.setText("航空器 %d/%d · 队列%d" % (live_count, self.MAX_AIRCRAFT, scheduled_count))
        self.status_cycles.setText("检测周期 %d" % self.detect_cycles)
        self.status_conflicts.setText("活动冲突 %d" % self.active_conflict_count)
        self.status_commands.setText("决策指令 %d" % len(self.command_records))
        self._set_status_tone(self.status_aircraft, "info" if live_count else "neutral")
        self._set_status_tone(self.status_conflicts, "warning" if self.active_conflict_count else "success")
        self._set_status_tone(self.status_commands, "info" if self.command_records else "neutral")
        summary = self._current_separation_summary()
        if summary["min_hsep"] is None:
            self.status_safety.setText("失去间隔 0")
            self.status_safety.setToolTip("No DYN aircraft pair available yet.")
            self._set_status_tone(self.status_safety, "success")
        else:
            self.status_safety.setText("失去间隔 %d" % summary["loss_count"])
            self.status_safety.setToolTip(
                "Current minimum separation among visible DYN aircraft: %.1f NM / %.0f ft"
                % (summary["min_hsep"], summary["min_vsep"])
            )
            self._set_status_tone(self.status_safety, "danger" if summary["loss_count"] else "success")
        self.status_llm.setText("解释层 %s" % self.last_llm_status)
        self._set_status_tone(self.status_llm, "info" if self.last_llm_status not in {"idle", "off"} else "neutral")
        log_text = self.log_path.name if self.log_path else "-"
        self.status_log.setText("日志 %s" % log_text)
        self.status_log.setToolTip(str(self.log_path) if self.log_path else "-")
        running = self.spawn_timer.isActive() or self.detect_timer.isActive()
        self.system_state.setText("● 运行中" if running else "● 就绪")
        self._set_status_tone(self.system_state, "success" if running else "neutral")

    def _min_hsep_nm(self):
        if hasattr(self, "min_sep_spin"):
            return float(self.min_sep_spin.value())
        return float(self.HSEP_NM)

    def _predict_gate_nm(self):
        return max(float(self.PREDICT_GATE_NM), self._min_hsep_nm() * 3.0)

    def _hsep_nm(self, a, b):
        ax, ay = self._xy_nm(a["lat"], a["lon"])
        bx, by = self._xy_nm(b["lat"], b["lon"])
        return sqrt((bx - ax) * (bx - ax) + (by - ay) * (by - ay))

    def on_min_separation_changed(self, value):
        self._tracked_safe_cache.clear()
        self._last_status_update_ts = 0.0
        self._append_log({"event": "min_separation_changed", "hsep_nm": float(value)})
        self._log_text("Min separation changed to %.1f NM; detector/verifier will use it from next cycle." % float(value))
        self._update_status_labels()
        if self.detect_timer.isActive():
            QTimer.singleShot(100, self.detect_and_resolve)

    def _apply_map_config(self, map_key):
        config = MAP_CONFIGS[map_key]
        self.current_map_key = map_key
        self.SCENARIO_NAME = config["scenario_name"]
        self.REAL_SECTOR_ID = config["sector_id"]
        self.WAYPOINTS = config["waypoints"]
        self.ROUTES = config["routes"]
        self.SECTOR_BOUNDS = config["bounds"]
        self.CENTER_LAT, self.CENTER_LON = config["center"]
        self.ZOOM = config["zoom"]

    def _update_map_info(self):
        if not hasattr(self, "map_info"):
            return
        config = MAP_CONFIGS[self.current_map_key]
        self.map_info.setText(
            "%s. %d directed routes, max %d aircraft. CPA conflicts and issued commands are shown below."
            % (config["description"], len(config["routes"]), self.MAX_AIRCRAFT)
        )

    def load_selected_map(self):
        map_key = self.map_combo.currentData()
        if map_key not in MAP_CONFIGS:
            return
        self._apply_map_config(map_key)
        self._update_map_info()
        self.reset_sector()

    def _write_idle_summary(self):
        config = MAP_CONFIGS[self.current_map_key]
        self.text.setPlainText(
            "Map ready: %s (%s).\n"
            "Routes: %d directed routes; route geometry is drawn on the radar display.\n"
            "Solver: discrete constraint search over altitude/speed actions, then LLM-style instruction/explanation wrapper.\n"
            "Safety rule: predicted CPA within %.0f min, HSEP < %.1f NM and VSEP < %.0f ft triggers resolution.\n"
            "Press Reset sector, then Start auto traffic. Logs: %s" % (
                config["label"], self.REAL_SECTOR_ID, len(self.ROUTES), self.LOOKAHEAD_MIN,
                self._min_hsep_nm(), self.VERIFY_VSEP_FT, self.log_path
            )
        )
        self._update_status_labels()

    def _log_text(self, line):
        if not hasattr(self, "text"):
            return
        self.text.append("[%s] %s" % (self._now(), line))
        document = self.text.document()
        if document.blockCount() > self.LOG_TEXT_MAX_LINES:
            cursor = self.text.textCursor()
            cursor.movePosition(QTextCursor.Start)
            for _ in range(document.blockCount() - self.LOG_TEXT_MAX_LINES):
                cursor.select(QTextCursor.BlockUnderCursor)
                cursor.removeSelectedText()
                cursor.deleteChar()
        bar = self.text.verticalScrollBar()
        bar.setValue(bar.maximum())

    def arm_autoload(self, delay_ms=500):
        self._autoload_attempts = 0
        QTimer.singleShot(delay_ms, self.auto_load_sector)

    def auto_load_sector(self):
        if self._autoload_done:
            return
        self._autoload_attempts += 1
        active = bool(getattr(bs, "net", None) and bs.net.actnode())
        if not active and self._autoload_attempts < 40:
            QTimer.singleShot(500, self.auto_load_sector)
            return
        if not active:
            self._log_text("仿真节点尚未就绪，请稍后重置扇区。")
            return
        self._autoload_done = True
        self._ensure_stream_connection()
        self._log_text("仿真节点已就绪，可重置扇区并加载动态场景。")

    def base_sector_commands(self):
        # Keep visual initialization light. Extra route polylines and waypoint labels
        # trigger QtGL buffer updates and make each command feel slow.
        south, west, north, east = self.SECTOR_BOUNDS
        commands = [
            "RESET",
            "HOLD",
            "PAN %.4f,%.4f" % (self.CENTER_LAT, self.CENTER_LON),
            "ZOOM %.2f" % self.ZOOM,
            "BOX HMI_SECTOR,%.4f,%.4f,%.4f,%.4f" % (south, west, north, east),
            "COLOR HMI_SECTOR,0,180,80",
        ]
        visual_scenario = MAP_CONFIGS[self.current_map_key].get("visual_scenario")
        if visual_scenario:
            commands.append("PCALL %s" % visual_scenario)
        return commands

    def reset_sector(self):
        self._ensure_stream_connection()
        self.stop_auto_traffic(stack_hold=False)
        self.spawn_index = 0
        self.active_meta.clear()
        self.latest_acdata.clear()
        self.resolved_pairs.clear()
        self.tracked_conflicts.clear()
        self.assigned_aircraft.clear()
        self.last_targets.clear()
        self.last_speeds.clear()
        self.issued_commands.clear()
        self.command_records.clear()
        self.last_command_by_aircraft.clear()
        self.command_queue.clear()
        self.command_timer.stop()
        self._last_status_update_ts = 0.0
        self._last_execution_update_ts = 0.0
        self._last_conflict_table_signature = None
        self._last_command_status_by_row.clear()
        self._tracked_safe_cache.clear()
        self.latest_sim_time = None
        self.last_detect_wall_ts = 0.0
        self.reset_pending_until = datetime.now().timestamp() + (self.RESET_SETTLE_MS / 1000.0)
        self.pending_start = False
        self.detect_cycles = 0
        self.conflict_events = 0
        self.active_conflict_count = 0
        self.last_llm_status = "idle"
        self.conflict_table.setRowCount(0)
        self.command_table.setRowCount(0)
        if hasattr(self, "aircraft_table"):
            self.aircraft_table.setRowCount(0)
            self._last_aircraft_table_signature = None
        for label in [self.decision_aircraft_value, self.decision_action_value,
                      self.decision_target_value, self.decision_command_value]:
            label.setText("-")
        self.decision_verify_value.setText("等待决策")
        self.decision_summary.setText("当前无待执行解脱指令")
        self.decision_summary.setToolTip("")
        self._reset_decision_process()
        self._new_log_file()
        for command in self.base_sector_commands():
            self._stack(command)
        self._write_idle_summary()
        self._add_row("sector", "-", "-", "Reset dynamic sector", "RESET", "ready")
        self._append_log({
            "event": "sector_reset",
            "max_aircraft": self.MAX_AIRCRAFT,
            "sector_id": self.REAL_SECTOR_ID,
            "map_key": self.current_map_key,
            "route_count": len(self.ROUTES),
        })

    def start_auto_traffic(self):
        self._ensure_stream_connection()
        now_ts = datetime.now().timestamp()
        if self.command_queue or now_ts < self.reset_pending_until:
            if not self.pending_start:
                self.pending_start = True
                self._log_text("场景正在重置，命令队列完成后将自动启动交通流。")
            QTimer.singleShot(800, self.start_auto_traffic)
            return
        self.pending_start = False
        self._trim_excess_live_aircraft()
        self._stack("OP")
        if not self.active_meta and not self.latest_acdata:
            self.spawn_initial_wave()
        self.spawn_timer.start(self.SPAWN_INTERVAL_MS)
        self.detect_timer.start(self.DETECT_INTERVAL_MS)
        self._last_status_update_ts = 0.0
        self._update_status_labels()
        QTimer.singleShot(6000, self.detect_and_resolve)
        self._add_row("control", "-", "-", "Auto traffic started", "OP", "running")
        self._append_log({"event": "auto_traffic_started", "spawn_interval_ms": self.SPAWN_INTERVAL_MS})

    def operate_simulation(self):
        self._stack("OP")
        if (self.active_meta or self.latest_acdata) and not self.detect_timer.isActive():
            self.detect_timer.start(self.DETECT_INTERVAL_MS)
            QTimer.singleShot(100, self.detect_and_resolve)
            self._append_log({"event": "conflict_monitor_resumed", "source": "operate_button"})

    def stop_auto_traffic(self, stack_hold=True):
        self.spawn_timer.stop()
        self.detect_timer.stop()
        self._last_status_update_ts = 0.0
        self._update_status_labels()
        if stack_hold:
            self._stack("HOLD")
            self._add_row("control", "-", "-", "Auto traffic stopped", "HOLD", "hold")
            self._append_log({"event": "auto_traffic_stopped"})

    def fast_forward_demo(self):
        self.detect_and_resolve()
        self._stack("FF 0:2:0")
        self._add_row("control", "-", "-", "Fast-forward two sim minutes", "FF 0:2:0", "running")
        self._append_log({"event": "fast_forward_requested", "command": "FF 0:2:0"})

    def _live_dyn_ids(self):
        self._refresh_from_radarwidget()
        return sorted([acid for acid in self.latest_acdata.keys() if str(acid).startswith("DYN")])

    def _can_spawn_more(self):
        live_count = len(self._live_dyn_ids()) if self.latest_acdata else 0
        return len(self.active_meta) < self.MAX_AIRCRAFT and live_count < self.MAX_AIRCRAFT

    def _trim_excess_live_aircraft(self):
        live_ids = self._live_dyn_ids()
        if len(live_ids) <= self.MAX_AIRCRAFT:
            return
        for acid in live_ids[self.MAX_AIRCRAFT:]:
            self._stack("DEL %s" % acid)
            self.active_meta.pop(acid, None)
        self._append_log({"event": "excess_aircraft_deleted", "deleted": live_ids[self.MAX_AIRCRAFT:]})

    def active_count(self):
        live_ids = self._live_dyn_ids()
        if live_ids:
            return len(live_ids)
        return len(self.active_meta)

    def spawn_initial_wave(self):
        if self.current_map_key == "three_route_demo":
            initial = [
                (self.ROUTES[0], 340),
                (self.ROUTES[1], 340),
                (self.ROUTES[2], 330),
                (self.ROUTES[3], 350),
            ]
            for route, fl in initial:
                self.spawn_aircraft(route, fl=fl, enforce_gate=False)
            return
        route_by_id = {route["route"]: route for route in self.ROUTES}
        initial = [
            ("DYN001", "A320", route_by_id["V69_F001_C01"], 340, 300),
            ("DYN002", "B738", route_by_id["V69_F002_C01"], 340, 300),
            ("DYN003", "E190", route_by_id["W179_F001_C01"], 320, 290),
            ("DYN004", "A319", route_by_id["W231_F001_C01"], 360, 300),
        ]
        now_ts = datetime.now().timestamp()
        for acid, actype, route, fl, speed in initial:
            entry_lat, entry_lon = self.WAYPOINTS[route["entry"]]
            self.active_meta[acid] = {
                "id": acid,
                "type": actype,
                "route": route["route"],
                "route_code": route.get("route_code", route["route"]),
                "hdg": route["hdg"],
                "entry_lat": entry_lat,
                "entry_lon": entry_lon,
                "entry": route["entry"],
                "exit": route["exit"],
                "fl": fl,
                "speed": speed,
                "spawn_ts": now_ts,
                "spawn_time": self._now(),
            }
            self._add_row(
                "spawn", acid, "-", "%s FL%d %dkt" % (route["route"], fl, speed),
                "PCALL ATC_HMI_REAL_CHENGDU_INITIAL", "loading",
            )
        self.spawn_index = len(initial)
        self._stack("PCALL ATC_HMI_REAL_CHENGDU_INITIAL")
        self._append_log({
            "event": "real_sector_initial_wave_queued",
            "scenario_file": "ATC_HMI_REAL_CHENGDU_INITIAL.scn",
            "aircraft": [item[0] for item in initial],
        })
        self._update_status_labels()

    def spawn_random_aircraft(self):
        if not self._can_spawn_more():
            return
        if self._traffic_busy_for_spawn():
            self._append_log({"event": "spawn_delayed_busy_traffic", "active_conflicts": self.active_conflict_count})
            return
        for _attempt in range(self.SPAWN_RETRY_LIMIT):
            route = self.rng.choice(self.ROUTES)
            fl = self.rng.choice(route["fls"])
            speed = self.rng.randint(route["speed"][0], route["speed"][1])
            ok, reason = self._spawn_candidate_is_safe(route, fl, speed)
            if ok:
                self.spawn_aircraft(route, fl=fl, speed=speed, enforce_gate=False)
                return
        self._append_log({"event": "spawn_delayed_entry_gate", "reason": reason})
        self._log_text(
            "Spawn delayed by entry gate: no safe boundary entry in %d trials under %.1f NM / %.0f ft."
            % (self.SPAWN_RETRY_LIMIT, self._min_hsep_nm(), self.VERIFY_VSEP_FT)
        )
        self._update_status_labels()

    def spawn_aircraft(self, route, fl=None, speed=None, enforce_gate=True):
        if not self._can_spawn_more():
            return None
        fl = fl if fl is not None else self.rng.choice(route["fls"])
        speed = speed if speed is not None else self.rng.randint(route["speed"][0], route["speed"][1])
        if enforce_gate:
            ok, reason = self._spawn_candidate_is_safe(route, fl, speed)
            if not ok:
                self._append_log({"event": "manual_spawn_rejected_entry_gate", "route": route["route"], "fl": fl, "speed": speed, "reason": reason})
                self._log_text("Spawn rejected by entry gate: %s" % reason)
                return None
        self.spawn_index += 1
        acid = "DYN%03d" % self.spawn_index
        actype = self.rng.choice(self.AIRCRAFT_TYPES)
        entry_lat, entry_lon = self.WAYPOINTS[route["entry"]]
        commands = [
            "CRE %s,%s,%.6f,%.6f,%d,FL%d,%d" % (acid, actype, entry_lat, entry_lon, route["hdg"], fl, speed),
        ]
        for waypoint_lat, waypoint_lon in route.get("waypoints", [])[1:]:
            commands.append(
                "ADDWPT %s %.6f %.6f FL%d %d" % (acid, waypoint_lat, waypoint_lon, fl, speed)
            )
        if len(commands) == 1:
            exit_lat, exit_lon = self.WAYPOINTS[route["exit"]]
            commands.append("ADDWPT %s %.6f %.6f FL%d %d" % (acid, exit_lat, exit_lon, fl, speed))
        commands.append("%s LNAV ON" % acid)
        for command in commands:
            self._stack(command)
        self.active_meta[acid] = {
            "id": acid,
            "type": actype,
            "route": route["route"],
            "route_code": route.get("route_code", route["route"]),
            "hdg": route["hdg"],
            "entry_lat": entry_lat,
            "entry_lon": entry_lon,
            "entry": route["entry"],
            "exit": route["exit"],
            "fl": fl,
            "speed": speed,
            "spawn_ts": datetime.now().timestamp(),
            "spawn_time": self._now(),
        }
        self._add_row("spawn", acid, "-", "%s FL%d %dkt" % (route["route"], fl, speed), commands[0], "created")
        self._append_log({"event": "aircraft_spawned", "aircraft": self.active_meta[acid], "commands": commands})
        self._update_status_labels()
        return acid

    def _traffic_busy_for_spawn(self):
        summary = self._current_separation_summary()
        return (
            summary.get("loss_count", 0) > 0
            or len(self.command_queue) > self.MAX_SPAWN_COMMAND_BACKLOG
        )

    def _spawn_gate_states(self):
        live = [
            state for state in self.latest_acdata.values()
            if str(state.get("id", "")).startswith("DYN")
        ]
        live_ids = {state["id"] for state in live}
        scheduled = [
            state for state in self._synthetic_states_from_meta()
            if state["id"] not in live_ids
        ]
        return live + scheduled

    def _candidate_spawn_state(self, route, fl, speed):
        entry_lat, entry_lon = self.WAYPOINTS[route["entry"]]
        return {
            "id": "CANDIDATE",
            "lat": entry_lat,
            "lon": entry_lon,
            "alt_ft": float(fl) * 100.0,
            "trk": float(route["hdg"]),
            "gs_mps": float(speed) / KT_PER_MPS,
            "cas_kt": float(speed),
        }

    def _project_xy_nm(self, state, t_sec):
        x0, y0 = self._xy_nm(state["lat"], state["lon"])
        distance_nm = self._speed_kt(state) * float(t_sec) / 3600.0
        rad = radians(state["trk"])
        return x0 + distance_nm * sin(rad), y0 + distance_nm * cos(rad)

    def _spawn_candidate_is_safe(self, route, fl, speed):
        existing = self._spawn_gate_states()
        if not existing:
            return True, "no existing traffic"
        candidate = self._candidate_spawn_state(route, fl, speed)
        horizon_sec = int(self.ENTRY_LOOKAHEAD_MIN * 60)
        required_hsep_nm = self._min_hsep_nm()
        for other in existing:
            vsep = abs(candidate["alt_ft"] - other["alt_ft"])
            if vsep >= self.VERIFY_VSEP_FT:
                continue
            for t_sec in range(0, horizon_sec + self.ENTRY_VERIFY_DT_SEC, self.ENTRY_VERIFY_DT_SEC):
                cx, cy = self._project_xy_nm(candidate, t_sec)
                ox, oy = self._project_xy_nm(other, t_sec)
                hsep = sqrt((ox - cx) * (ox - cx) + (oy - cy) * (oy - cy))
                if hsep < required_hsep_nm:
                    return False, (
                        "candidate %s FL%d %dkt conflicts with %s in %.1f min: %.1f NM / %.0f ft"
                        % (route["route"], fl, speed, other["id"], t_sec / 60.0, hsep, vsep)
                    )
        return True, "safe"

    def on_simstream_received(self, streamname, data, sender_id):
        if streamname != b"ACDATA":
            return
        self._set_latest_acdata(ACDataEvent(data))

    def _refresh_from_radarwidget(self):
        radar = getattr(self.parent(), "radarwidget", None)
        if radar is None and self.window() is not None:
            radar = getattr(self.window(), "radarwidget", None)
        if radar is None:
            for widget in QApplication.topLevelWidgets():
                radar = getattr(widget, "radarwidget", None)
                if radar is not None:
                    break
        acdata = getattr(radar, "acdata", None)
        if acdata is not None and len(getattr(acdata, "id", [])) > 0:
            self._set_latest_acdata(acdata)

    def _set_latest_acdata(self, acdata):
        sim_time = getattr(acdata, "simt", None)
        if sim_time is not None:
            try:
                self.latest_sim_time = float(sim_time)
            except (TypeError, ValueError):
                pass
        ids = list(getattr(acdata, "id", []))
        lat = list(getattr(acdata, "lat", []))
        lon = list(getattr(acdata, "lon", []))
        alt = list(getattr(acdata, "alt", []))
        trk = list(getattr(acdata, "trk", []))
        gs = list(getattr(acdata, "gs", getattr(acdata, "tas", [])))
        cas = list(getattr(acdata, "cas", []))
        vs = list(getattr(acdata, "vs", []))
        self.latest_acdata = {}
        for idx, acid in enumerate(ids):
            try:
                acid = str(acid)
                cas_kt = float(cas[idx]) * KT_PER_MPS if idx < len(cas) else float(gs[idx]) * KT_PER_MPS
                self.latest_acdata[acid] = {
                    "id": acid,
                    "lat": float(lat[idx]),
                    "lon": float(lon[idx]),
                    "alt_ft": float(alt[idx]) * FT_PER_METER,
                    "trk": float(trk[idx]),
                    "gs_mps": float(gs[idx]),
                    "cas_kt": cas_kt,
                    "vs_fpm": float(vs[idx]) * FT_PER_METER * 60.0 if idx < len(vs) else 0.0,
                }
            except (IndexError, TypeError, ValueError):
                continue
        self._update_command_execution_statuses()

    def _xy_nm(self, lat, lon):
        x = (lon - self.CENTER_LON) * 60.0 * cos(radians(self.CENTER_LAT))
        y = (lat - self.CENTER_LAT) * 60.0
        return x, y

    def _velocity_nm_min(self, trk_deg, gs_mps):
        rad = radians(trk_deg)
        speed_nm_min = gs_mps * 60.0 * NM_PER_METER
        return speed_nm_min * sin(rad), speed_nm_min * cos(rad)

    def _synthetic_states_from_meta(self):
        states = []
        now_ts = datetime.now().timestamp()
        for meta in self.active_meta.values():
            elapsed_sec = max(0.0, now_ts - float(meta.get("spawn_ts", now_ts)))
            speed_kt = float(meta.get("speed", 440))
            distance_nm = speed_kt * elapsed_sec / 3600.0
            trk = float(meta.get("hdg", 0.0))
            rad = radians(trk)
            dx_nm = distance_nm * sin(rad)
            dy_nm = distance_nm * cos(rad)
            lat = float(meta.get("entry_lat", self.CENTER_LAT)) + dy_nm / 60.0
            lon = float(meta.get("entry_lon", self.CENTER_LON)) + dx_nm / (60.0 * cos(radians(self.CENTER_LAT)))
            states.append({
                "id": meta["id"],
                "lat": lat,
                "lon": lon,
                "alt_ft": float(meta.get("fl", 330)) * 100.0,
                "trk": trk,
                "gs_mps": speed_kt / KT_PER_MPS,
                "cas_kt": speed_kt,
                "source": "synthetic_meta",
            })
        return states

    def _cpa(self, a, b):
        ax, ay = self._xy_nm(a["lat"], a["lon"])
        bx, by = self._xy_nm(b["lat"], b["lon"])
        avx, avy = self._velocity_nm_min(a["trk"], a["gs_mps"])
        bvx, bvy = self._velocity_nm_min(b["trk"], b["gs_mps"])
        rx, ry = bx - ax, by - ay
        vx, vy = bvx - avx, bvy - avy
        vv = vx * vx + vy * vy
        tcpa = 0.0 if vv <= 1e-9 else max(0.0, min(self.LOOKAHEAD_MIN, -((rx * vx + ry * vy) / vv)))
        dx, dy = rx + vx * tcpa, ry + vy * tcpa
        hsep = sqrt(dx * dx + dy * dy)
        vsep = abs(a["alt_ft"] - b["alt_ft"])
        return tcpa, hsep, vsep

    def _speed_kt(self, state):
        raw = state.get("cas_kt", state["gs_mps"] * KT_PER_MPS)
        return int(round(raw))

    def _effective_fl(self, state):
        acid = state["id"]
        target = self.last_targets.get(acid)
        if target is not None:
            return int(target)
        current_fl = state["alt_ft"] / 100.0
        return int(min(self.SAFE_LEVELS, key=lambda level: (abs(level - current_fl), level)))

    def _effective_speed(self, state):
        return int(self.last_speeds.get(state["id"], self._speed_kt(state)))

    def _pending_alt_direction(self, state):
        target = self.last_targets.get(state["id"])
        if target is None:
            return 0
        diff_ft = target * 100.0 - state["alt_ft"]
        if abs(diff_ft) <= self.ALT_REACHED_TOLERANCE_FT:
            return 0
        return 1 if diff_ft > 0.0 else -1

    def _candidate_actions(self, state, allow_alt_reversal=False):
        acid = state["id"]
        effective_fl = self._effective_fl(state)
        current_speed = self._effective_speed(state)
        pending_alt_direction = self._pending_alt_direction(state)
        actions = [{
            "acid": acid,
            "kind": "hold",
            "target_fl": effective_fl,
            "target_speed": current_speed,
            "command": None,
            "label": "hold",
        }]

        altitude_levels = [
            level for level in self.SAFE_LEVELS
            if abs(level - effective_fl) in self.ALT_DELTAS_FL
        ]
        for target_fl in sorted(set(altitude_levels), key=lambda fl: (abs(fl - effective_fl), fl)):
            vs = self.VS_FPM if target_fl * 100.0 > state["alt_ft"] else -self.VS_FPM
            target_direction = 1 if target_fl * 100.0 > state["alt_ft"] else -1
            if pending_alt_direction and target_direction != pending_alt_direction and not allow_alt_reversal:
                continue
            actions.append({
                "acid": acid,
                "kind": "altitude",
                "target_fl": target_fl,
                "target_speed": current_speed,
                "command": "ALT %s,FL%d,%d" % (acid, target_fl, vs),
                "label": "altitude:FL%d" % target_fl,
            })

        seen_speeds = set()
        for delta in self.SPEED_DELTAS_KT:
            target_speed = max(self.MIN_SPEED_KT, min(self.MAX_SPEED_KT, current_speed + delta))
            if target_speed == current_speed or target_speed in seen_speeds:
                continue
            seen_speeds.add(target_speed)
            actions.append({
                "acid": acid,
                "kind": "speed",
                "target_fl": effective_fl,
                "target_speed": target_speed,
                "command": "SPD %s,%d" % (acid, target_speed),
                "label": "speed:%dkt" % target_speed,
            })

        preference = self.preference_combo.currentText() if hasattr(self, "preference_combo") else "speed_first"
        order = {"hold": 0, "speed": 1, "altitude": 2} if preference == "speed_first" else {"hold": 0, "altitude": 1, "speed": 2}
        return sorted(actions, key=lambda action: (
            order.get(action["kind"], 9),
            abs(action["target_fl"] - effective_fl),
            abs(action["target_speed"] - current_speed),
        ))

    def _action_id(self, action):
        if action["kind"] == "hold":
            return "%s_hold" % action["acid"]
        if action["kind"] == "speed":
            return "%s_spd_%d" % (action["acid"], int(round(action["target_speed"])))
        if action["kind"] == "altitude":
            return "%s_alt_%d" % (action["acid"], int(round(action["target_fl"])))
        return "%s_%s" % (action["acid"], action["kind"])

    def _action_deviation_cost(self, state, action):
        return int(
            abs(int(round(action["target_fl"])) - self._effective_fl(state))
            + abs(int(round(action["target_speed"])) - self._effective_speed(state))
        )

    def _level_occupancy(self, acid, action, state_by_id):
        if action["kind"] in {"hold", "speed"}:
            return "current", 0
        probe_state = state_by_id[acid]
        conflicts = 0
        occupied = 0
        for other_id, other in state_by_id.items():
            if other_id == acid:
                continue
            other_hold = {
                "acid": other_id,
                "kind": "hold",
                "target_fl": self._effective_fl(other),
                "target_speed": self._effective_speed(other),
                "command": None,
                "label": "current_target",
            }
            if abs(action["target_fl"] - other_hold["target_fl"]) < 10:
                occupied += 1
            if not self._action_pair_is_safe(probe_state, action, other, other_hold):
                conflicts += 1
        if conflicts:
            return "crossing_risk", conflicts
        if occupied:
            return "occupied", occupied
        return "free", 0

    def _encode_model_actions(self, graph_ids, state_by_id, allow_alt_reversal=False):
        encoded = {}
        lookup = {}
        for acid in sorted(graph_ids):
            rows = []
            for action in self._candidate_actions(state_by_id[acid], allow_alt_reversal=allow_alt_reversal):
                level_status, level_count = self._level_occupancy(acid, action, state_by_id)
                aid = self._action_id(action)
                rows.append([
                    aid,
                    action["kind"],
                    int(round(action["target_fl"])),
                    int(round(action["target_speed"])),
                    self._action_deviation_cost(state_by_id[acid], action),
                    level_status,
                    level_count,
                ])
                lookup[aid] = action
            encoded[acid] = rows
        return encoded, lookup

    def _edge_geometry(self, a, b):
        angle = abs(((a["trk"] - b["trk"] + 180.0) % 360.0) - 180.0)
        if angle >= 150.0:
            return "head_on"
        if angle <= 30.0:
            return "parallel"
        if angle >= 60.0:
            return "crossing"
        return "converging"

    def _risk_bucket(self, tcpa, hsep):
        if hsep < 1.0 or tcpa < 2.0:
            return "critical"
        if hsep < 3.0 or tcpa < 5.0:
            return "high"
        return "medium"

    def _build_action_model_input(self, graph_ids, state_by_id, detections, encoded_actions):
        aircraft_rows = []
        for acid in sorted(graph_ids):
            state = state_by_id[acid]
            route = self.spawned_meta.get(acid, {}).get("route", "unknown")
            aircraft_rows.append([
                acid,
                route,
                int(round(state["alt_ft"] / 100.0)),
                int(round(state["trk"])),
                self._effective_speed(state),
                self._effective_fl(state),
                self._effective_speed(state),
                "none" if not self._pending_alt_direction(state) else ("up" if self._pending_alt_direction(state) > 0 else "down"),
            ])
        edge_rows = []
        for tcpa, hsep, vsep, a, b, _pair in detections:
            current_hsep = self._hsep_nm(a, b)
            current_vsep = abs(a["alt_ft"] - b["alt_ft"])
            angle = abs(((a["trk"] - b["trk"] + 180.0) % 360.0) - 180.0)
            edge_rows.append([
                a["id"],
                b["id"],
                self._edge_geometry(a, b),
                round(tcpa, 1),
                round(hsep, 1),
                int(round(vsep)),
                round(current_hsep, 1),
                int(round(current_vsep)),
                int(round(angle)),
                self._risk_bucket(tcpa, hsep),
            ])
        return {
            "schema": "qwen_conflict_choice_input_v1_1",
            "pref": self.preference_combo.currentText() if hasattr(self, "preference_combo") else "speed_first",
            "limits": [float(self.LOOKAHEAD_MIN), float(self._min_hsep_nm()), int(self.VERIFY_VSEP_FT)],
            "aircraft": aircraft_rows,
            "edges": edge_rows,
            "actions": encoded_actions,
        }

    def _parse_action_model_text(self, raw):
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`").strip()
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end >= start:
            raw = raw[start:end + 1]
        return json.loads(raw)

    def _call_action_model(self, model_input):
        url = os.environ.get("ATC_ACTION_API_URL")
        if not url:
            return {"ok": False, "skipped": True, "error": "ATC_ACTION_API_URL is not set"}
        timeout = float(os.environ.get("ATC_ACTION_TIMEOUT_SEC", "3.0"))
        instruction = (
            "You are an ATC conflict-resolution decision selector. Given a compact conflict graph and available "
            "candidate actions, select action IDs that resolve predicted conflicts while respecting safety limits "
            "and controller preference. Return JSON only with fields: schema, status, actions, reason_codes. "
            "Do not invent action IDs. Do not output BlueSky commands."
        )
        if url.endswith("/predict") or os.environ.get("ATC_ACTION_API_MODE", "").lower() == "predict":
            body = {
                "instruction": instruction,
                "input": json.dumps(model_input, ensure_ascii=False, separators=(",", ":")),
                "max_new_tokens": int(os.environ.get("ATC_ACTION_MAX_TOKENS", "160")),
                "return_raw": True,
            }
        else:
            body = {
                "model": os.environ.get("ATC_ACTION_MODEL", os.environ.get("ATC_LLM_MODEL", "qwen3-4b")),
                "messages": [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": instruction + "\n\nInput:\n" + json.dumps(model_input, ensure_ascii=False, separators=(",", ":"))},
                ],
                "temperature": 0,
                "max_tokens": int(os.environ.get("ATC_ACTION_MAX_TOKENS", "160")),
            }
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("ATC_ACTION_API_KEY", os.environ.get("ATC_LLM_API_KEY", ""))
        if api_key:
            headers["Authorization"] = "Bearer " + api_key
        req = request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
        started = monotonic()
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except (error.URLError, TimeoutError, OSError) as exc:
            return {"ok": False, "error": str(exc), "latency_sec": round(monotonic() - started, 3)}
        try:
            parsed = json.loads(raw)
            if "output" in parsed:
                output = parsed["output"]
            else:
                content = parsed["choices"][0]["message"]["content"]
                output = self._parse_action_model_text(content)
            return {"ok": True, "output": output, "raw": raw[:1200], "latency_sec": round(monotonic() - started, 3)}
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return {"ok": False, "error": "parse_error: %s" % exc, "raw": raw[:1200], "latency_sec": round(monotonic() - started, 3)}

    def _verify_model_solution(self, graph, state_by_id, action_lookup, selected_ids):
        selected = {}
        invalid = []
        for aid in selected_ids:
            action = action_lookup.get(aid)
            if action is None:
                invalid.append(aid)
                continue
            selected[action["acid"]] = action
        for acid in graph:
            if acid not in selected:
                hold_id = "%s_hold" % acid
                hold = action_lookup.get(hold_id)
                if hold is None:
                    invalid.append(hold_id)
                else:
                    selected[acid] = hold
        if invalid:
            return False, [], {"invalid_action_ids": invalid, "missing_aircraft": []}
        failed_pairs = []
        for acid, neighbors in graph.items():
            for neighbor in neighbors:
                if acid >= neighbor:
                    continue
                if not self._action_pair_is_safe(state_by_id[acid], selected[acid], state_by_id[neighbor], selected[neighbor]):
                    failed_pairs.append([acid, neighbor])
        if failed_pairs:
            return False, [], {"invalid_action_ids": [], "failed_pairs": failed_pairs}
        commands = []
        for acid, action in sorted(selected.items()):
            if action["kind"] == "hold" or not action["command"] or self._command_is_active(action["command"]):
                continue
            commands.append(action)
        return True, commands, {"invalid_action_ids": [], "failed_pairs": []}

    def _try_model_resolution_plan(self, graph, state_by_id, detections):
        encoded_actions, action_lookup = self._encode_model_actions(graph, state_by_id, allow_alt_reversal=False)
        model_input = self._build_action_model_input(graph, state_by_id, detections, encoded_actions)
        api_result = self._call_action_model(model_input)
        if not api_result.get("ok"):
            return None, {
                "method": "llm_action_selector",
                "success": False,
                "fallback_reason": api_result.get("error", "model_call_failed"),
                "model_api": api_result,
                "model_input": model_input,
            }
        output = api_result.get("output") or {}
        selected_ids = output.get("actions") or []
        safe, commands, verification = self._verify_model_solution(graph, state_by_id, action_lookup, selected_ids)
        candidate_actions = [
            {
                "aircraft": acid,
                "kind": action["kind"],
                "target_fl": action["target_fl"],
                "target_speed": action["target_speed"],
                "label": action["label"],
            }
            for acid in order
            for action in actions_by_acid[acid]
        ]
        solver = {
            "method": "llm_action_selector_verified" if safe else "llm_action_selector_rejected",
            "preference": self.preference_combo.currentText() if hasattr(self, "preference_combo") else "speed_first",
            "num_conflict_aircraft": len(graph),
            "num_conflict_pairs": len(detections),
            "selected_action_ids": selected_ids,
            "model_status": output.get("status"),
            "model_reason_codes": output.get("reason_codes"),
            "model_latency_sec": api_result.get("latency_sec"),
            "verification": verification,
            "model_input": model_input,
            "success": safe,
        }
        if safe:
            solver["selected_actions"] = {action["acid"]: action["label"] for action in commands}
            return commands, solver
        return None, solver

    def _speed_distance_nm(self, start_speed_kt, target_speed_kt, t_sec):
        delta = target_speed_kt - start_speed_kt
        if abs(delta) <= 1e-9:
            return start_speed_kt * t_sec / 3600.0
        direction = 1.0 if delta > 0 else -1.0
        ramp_time = abs(delta) / self.SPEED_ACCEL_KT_PER_SEC
        if t_sec <= ramp_time:
            end_speed = start_speed_kt + direction * self.SPEED_ACCEL_KT_PER_SEC * t_sec
            return ((start_speed_kt + end_speed) / 2.0) * t_sec / 3600.0
        ramp_distance = ((start_speed_kt + target_speed_kt) / 2.0) * ramp_time / 3600.0
        cruise_distance = target_speed_kt * (t_sec - ramp_time) / 3600.0
        return ramp_distance + cruise_distance

    def _predicted_state(self, state, action, t_sec):
        x0, y0 = self._xy_nm(state["lat"], state["lon"])
        start_command_speed = max(1.0, float(self._effective_speed(state)))
        start_ground_speed = float(self._speed_kt(state))
        target_ground_speed = start_ground_speed * (float(action["target_speed"]) / start_command_speed)
        distance_nm = self._speed_distance_nm(start_ground_speed, target_ground_speed, t_sec)
        rad = radians(state["trk"])
        x = x0 + distance_nm * sin(rad)
        y = y0 + distance_nm * cos(rad)
        target_alt = action["target_fl"] * 100.0
        if abs(target_alt - state["alt_ft"]) <= 1e-6:
            alt = target_alt
        else:
            direction = 1.0 if target_alt > state["alt_ft"] else -1.0
            delta = direction * self.VS_FPM * (t_sec / 60.0)
            alt = min(target_alt, state["alt_ft"] + delta) if direction > 0 else max(target_alt, state["alt_ft"] + delta)
        return x, y, alt

    def _action_pair_is_safe(self, a, action_a, b, action_b):
        horizon_sec = int(self.LOOKAHEAD_MIN * 60)
        required_hsep_nm = self._min_hsep_nm()
        for t_sec in range(0, horizon_sec + self.VERIFY_DT_SEC, self.VERIFY_DT_SEC):
            ax, ay, aalt = self._predicted_state(a, action_a, t_sec)
            bx, by, balt = self._predicted_state(b, action_b, t_sec)
            hsep = sqrt((bx - ax) * (bx - ax) + (by - ay) * (by - ay))
            vsep = abs(aalt - balt)
            if hsep < required_hsep_nm and vsep < self.VERIFY_VSEP_FT:
                return False
        return True

    def _current_targets_are_safe(self, a, b):
        action_a = {
            "acid": a["id"], "kind": "hold", "target_fl": self._effective_fl(a),
            "target_speed": self._effective_speed(a), "command": None, "label": "current_target",
        }
        action_b = {
            "acid": b["id"], "kind": "hold", "target_fl": self._effective_fl(b),
            "target_speed": self._effective_speed(b), "command": None, "label": "current_target",
        }
        return self._action_pair_is_safe(a, action_a, b, action_b)

    def _build_resolution_plan(self, state_by_id, detections):
        graph = {}
        urgency = {}
        for tcpa, _hsep, _vsep, a, b, _pair in detections:
            graph.setdefault(a["id"], set()).add(b["id"])
            graph.setdefault(b["id"], set()).add(a["id"])
            urgency[a["id"]] = min(urgency.get(a["id"], 999.0), tcpa)
            urgency[b["id"]] = min(urgency.get(b["id"], 999.0), tcpa)

        if os.environ.get("ATC_ACTION_API_URL"):
            model_actions, model_solver = self._try_model_resolution_plan(graph, state_by_id, detections)
            if model_actions is not None:
                return model_actions, model_solver

        actions_by_acid = {acid: self._candidate_actions(state_by_id[acid], allow_alt_reversal=False) for acid in graph}
        order = sorted(graph, key=lambda acid: (-len(graph[acid]), urgency.get(acid, 999.0), acid))
        pair_cache = {}
        checked_nodes = 0
        used_alt_reversal_fallback = False
        deadline = monotonic() + self.SOLVER_TIME_BUDGET_SEC
        timed_out = False

        def action_index(acid, action):
            return actions_by_acid[acid].index(action)

        def compatible(acid, action, assigned):
            for neighbor in graph[acid]:
                if neighbor not in assigned:
                    continue
                left, right = sorted([acid, neighbor])
                if acid == left:
                    a_id, a_action = acid, action
                    b_id, b_action = neighbor, assigned[neighbor]
                else:
                    a_id, a_action = neighbor, assigned[neighbor]
                    b_id, b_action = acid, action
                key = (left, action_index(left, a_action), right, action_index(right, b_action))
                if key not in pair_cache:
                    pair_cache[key] = self._action_pair_is_safe(state_by_id[a_id], a_action, state_by_id[b_id], b_action)
                if not pair_cache[key]:
                    return False
            return True

        def search(assigned):
            nonlocal checked_nodes, timed_out
            checked_nodes += 1
            if checked_nodes > self.MAX_SOLVER_NODES or monotonic() > deadline:
                timed_out = True
                return None
            if len(assigned) >= len(order):
                return dict(assigned)
            best_acid = None
            best_actions = None
            for acid in order:
                if acid in assigned:
                    continue
                feasible = [action for action in actions_by_acid[acid] if compatible(acid, action, assigned)]
                if best_actions is None or len(feasible) < len(best_actions):
                    best_acid = acid
                    best_actions = feasible
                if best_actions is not None and len(best_actions) <= 1:
                    break
            if best_acid is None or best_actions is None:
                return dict(assigned)
            for action in best_actions:
                if compatible(best_acid, action, assigned):
                    assigned[best_acid] = action
                    result = search(assigned)
                    if result is not None:
                        return result
                    assigned.pop(best_acid, None)
            return None

        solution = search({})
        if solution is None:
            used_alt_reversal_fallback = True
            actions_by_acid = {acid: self._candidate_actions(state_by_id[acid], allow_alt_reversal=True) for acid in graph}
            pair_cache = {}
            checked_nodes = 0
            deadline = monotonic() + self.SOLVER_TIME_BUDGET_SEC
            timed_out = False
            solution = search({})
        solver = {
            "method": "discrete_constraint_search",
            "preference": self.preference_combo.currentText() if hasattr(self, "preference_combo") else "speed_first",
            "num_conflict_aircraft": len(order),
            "num_conflict_pairs": len(detections),
            "candidate_count": sum(len(items) for items in actions_by_acid.values()),
            "candidate_actions": candidate_actions[:3],
            "search_nodes": checked_nodes,
            "pair_checks": len(pair_cache),
            "rejected_pair_checks": sum(1 for safe in pair_cache.values() if not safe),
            "used_alt_reversal_fallback": used_alt_reversal_fallback,
            "timed_out": timed_out,
            "selected_actions": {},
            "success": solution is not None,
        }
        if os.environ.get("ATC_ACTION_API_URL"):
            solver["llm_action_selector_fallback"] = True
        if solution is None:
            return [], solver

        commands = []
        for acid, action in sorted(solution.items()):
            solver["selected_actions"][acid] = action["label"]
            if action["kind"] == "hold" or not action["command"]:
                continue
            if self._command_is_active(action["command"]):
                continue
            commands.append(action)
        candidate_actions.sort(key=lambda item: (
            0 if solver["selected_actions"].get(item["aircraft"]) == item["label"] else 1,
            1 if item["kind"] == "hold" else 0,
        ))
        solver["candidate_actions"] = candidate_actions[:3]
        return commands, solver

    def _build_recovery_altitude_plan(self, state_by_id, detections):
        graph = {}
        urgency = {}
        for tcpa, _hsep, _vsep, a, b, _pair in detections:
            graph.setdefault(a["id"], set()).add(b["id"])
            graph.setdefault(b["id"], set()).add(a["id"])
            urgency[a["id"]] = min(urgency.get(a["id"], 999.0), tcpa)
            urgency[b["id"]] = min(urgency.get(b["id"], 999.0), tcpa)
        order = sorted(graph, key=lambda acid: (-len(graph[acid]), urgency.get(acid, 999.0), acid))
        assigned = {}
        selected_actions = {}
        for acid in order:
            state = state_by_id[acid]
            effective_fl = self._effective_fl(state)
            current_speed = self._effective_speed(state)
            levels = sorted(
                self.SAFE_LEVELS,
                key=lambda fl: (abs(fl - effective_fl), abs(fl - int(round(state["alt_ft"] / 100.0))), fl),
            )
            chosen = None
            for level in levels:
                if all(abs(level - assigned[neighbor]) >= 10 for neighbor in graph[acid] if neighbor in assigned):
                    chosen = level
                    break
            if chosen is None:
                chosen = max(
                    self.SAFE_LEVELS,
                    key=lambda fl: min([abs(fl - assigned[neighbor]) for neighbor in graph[acid] if neighbor in assigned] or [999]),
                )
            assigned[acid] = chosen
            if chosen == effective_fl:
                selected_actions[acid] = "hold"
                continue
            selected_actions[acid] = "recovery_altitude:FL%d" % chosen
        actions = []
        for acid in order:
            label = selected_actions.get(acid)
            if not label or label == "hold":
                continue
            state = state_by_id[acid]
            target_fl = assigned[acid]
            current_speed = self._effective_speed(state)
            vs = self.VS_FPM if target_fl * 100.0 > state["alt_ft"] else -self.VS_FPM
            command = "ALT %s,FL%d,%d" % (acid, target_fl, vs)
            if self._command_is_active(command):
                continue
            actions.append({
                "acid": acid,
                "kind": "altitude",
                "target_fl": target_fl,
                "target_speed": current_speed,
                "command": command,
                "label": label,
            })
        solver = {
            "method": "altitude_recovery_graph_coloring",
            "preference": "safety_recovery",
            "num_conflict_aircraft": len(order),
            "num_conflict_pairs": len(detections),
            "selected_actions": selected_actions,
            "success": bool(actions),
        }
        return actions, solver

    def _llm_wrap_decision(self, detections, actions, solver):
        conflicts = []
        for tcpa, hsep, vsep, a, b, _pair in detections:
            conflicts.append({
                "aircraft": [a["id"], b["id"]],
                "tcpa_min": round(tcpa, 2),
                "predicted_hsep_nm": round(hsep, 2),
                "current_vsep_ft": round(vsep, 0),
            })
        structured_actions = []
        phrases = []
        for action in actions:
            if action["kind"] == "altitude":
                verb = "descend" if action.get("command", "").rsplit(",", 1)[-1].startswith("-") else "climb"
                phrase = "%s, %s and maintain flight level %d." % (action["acid"], verb, action["target_fl"])
            elif action["kind"] == "speed":
                phrase = "%s, adjust indicated airspeed to %d knots." % (action["acid"], action["target_speed"])
            else:
                phrase = "%s, maintain current clearance." % action["acid"]
            phrases.append(phrase)
            structured_actions.append({
                "aircraft": action["acid"],
                "maneuver": action["kind"],
                "target_fl": action["target_fl"],
                "target_speed_kt": action["target_speed"],
                "bluesky_command": action["command"],
                "instruction": phrase,
            })
        mode = self.llm_combo.currentText() if hasattr(self, "llm_combo") else "template_explainer"
        reason = (
            "The verifier searches altitude and speed candidates over a %.0f-minute horizon; "
            "accepted actions keep predicted separation above %.1f NM or %.0f ft."
        ) % (self.LOOKAHEAD_MIN, self._min_hsep_nm(), self.VERIFY_VSEP_FT)
        payload = {
            "provider": mode,
            "prompt_contract": "conflict_state + controller_preference -> structured_actions + standard_phrase + rationale",
            "preference": solver.get("preference"),
            "conflicts": conflicts,
            "structured_actions": structured_actions,
            "standard_instructions": phrases,
            "explanation": reason,
        }
        if mode == "openai_compatible_api":
            api_result = self._call_llm_api(payload)
            if api_result.get("ok"):
                self.last_llm_status = "api_ok"
                payload["provider"] = api_result.get("provider", mode)
                payload["model_text"] = api_result.get("text", "")
                if api_result.get("text"):
                    payload["explanation"] = api_result["text"]
            else:
                self.last_llm_status = "api_error"
                payload["api_error"] = api_result.get("error", "unknown_error")
        elif mode == "off":
            self.last_llm_status = "off"
        else:
            self.last_llm_status = "template"
        return payload

    def _call_llm_api(self, decision_payload):
        url = os.environ.get("ATC_LLM_API_URL")
        if not url:
            return {"ok": False, "error": "ATC_LLM_API_URL is not set"}
        model = os.environ.get("ATC_LLM_MODEL", "qwen3-4b")
        api_key = os.environ.get("ATC_LLM_API_KEY", "")
        prompt = (
            "You are an air-traffic controller decision assistant. "
            "Rewrite the verified conflict-resolution plan into concise standard ATC instructions "
            "and a short safety rationale. Do not invent new actions. JSON input:\n"
            + json.dumps(decision_payload, ensure_ascii=True)
        )
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Only explain verified ATC actions. Do not change commands."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = "Bearer " + api_key
        data = json.dumps(body).encode("utf-8")
        req = request.Request(url, data=data, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=2.0) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except (error.URLError, TimeoutError, OSError) as exc:
            return {"ok": False, "error": str(exc)}
        try:
            parsed = json.loads(raw)
            text = parsed["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            text = raw[:1200]
        return {"ok": True, "provider": "openai_compatible_api:%s" % model, "text": text}

    def detect_and_resolve(self):
        if self.detect_busy:
            return
        self.detect_busy = True
        self.last_detect_wall_ts = monotonic()
        try:
            self._detect_and_resolve_impl()
        except Exception as exc:
            self._append_log({
                "event": "detect_cycle_error",
                "cycle": self.detect_cycles,
                "error": repr(exc),
                "sim_time": self.latest_sim_time,
            })
            self._log_text("Conflict monitor error in cycle %d: %s" % (self.detect_cycles, exc))
        finally:
            self.detect_busy = False

    def _detect_and_resolve_impl(self):
        self.detect_cycles += 1
        self._refresh_from_radarwidget()
        self._update_status_labels()
        aircraft = [
            state for state in self.latest_acdata.values()
            if str(state.get("id", "")).startswith("DYN")
        ]
        if len(aircraft) < 2:
            self.tracked_conflicts.clear()
            self._sync_tracked_conflicts(aircraft, [], [])
            self._refresh_conflict_table([])
            return
        state_by_id = {state["id"]: state for state in aircraft}
        detections = []
        for i, a in enumerate(aircraft):
            for b in aircraft[i + 1:]:
                pair = tuple(sorted([a["id"], b["id"]]))
                tcpa, hsep, vsep = self._cpa(a, b)
                if hsep < self._predict_gate_nm() and not self._current_targets_are_safe(a, b):
                    detections.append((tcpa, hsep, vsep, a, b, pair))
        detections.sort(key=lambda x: (x[0], x[1]))
        if not detections:
            display_rows = self._sync_tracked_conflicts(aircraft, [], [])
            self._refresh_conflict_table(display_rows)
            if self.detect_cycles % 10 == 0:
                self._append_log({"event": "detect_cycle_clear", "cycle": self.detect_cycles, "aircraft": len(aircraft)})
            return
        actions, solver = self._build_resolution_plan(state_by_id, detections)
        llm_output = self._llm_wrap_decision(detections, actions, solver)
        self.conflict_events += 1
        aircraft_text = ",".join(sorted({item[3]["id"] for item in detections} | {item[4]["id"] for item in detections}))
        first = detections[0]
        cpa_text = "%.1f min %.1f NM %.0f ft" % (first[0], first[1], first[2])
        if not actions:
            if solver.get("success"):
                display_rows = self._sync_tracked_conflicts(aircraft, detections, [], default_state="Issued")
                self._refresh_conflict_table(display_rows, [], state="Issued")
                self._append_log({
                    "event": "conflict_monitoring_no_new_command",
                    "detections": len(detections),
                    "solver": solver,
                    "llm_output": llm_output,
                })
                self._update_status_labels()
                return
            recovery_actions, recovery_solver = self._build_recovery_altitude_plan(state_by_id, detections)
            if recovery_actions:
                actions = recovery_actions
                solver = recovery_solver
                llm_output = self._llm_wrap_decision(detections, actions, solver)
                self._append_log({
                    "event": "preventive_solver_failed_recovery_issued",
                    "detections": len(detections),
                    "recovery_solver": solver,
                    "commands": [action["command"] for action in actions],
                    "llm_output": llm_output,
                })
            else:
                display_rows = self._sync_tracked_conflicts(aircraft, detections, [], default_state="Blocked")
                self._refresh_conflict_table(display_rows, [], state="Blocked")
                self._update_decision_process(detections, [], solver)
                self._add_command_row(aircraft_text, "alert", "-", "No verified action; hold for controller review", "blocked")
                self._append_log({
                    "event": "conflict_detected_no_verified_action",
                    "detections": len(detections),
                    "solver": solver,
                    "llm_output": llm_output,
                })
                self._update_status_labels()
                return

        self._update_decision_process(detections, actions, solver)
        instruction_by_aircraft = {
            item["aircraft"]: item["instruction"]
            for item in llm_output.get("structured_actions", [])
        }
        for action in actions:
            command = action["command"]
            self.assigned_aircraft.add(action["acid"])
            self.last_command_by_aircraft[action["acid"]] = command
            if action["kind"] == "altitude":
                self.last_targets[action["acid"]] = action["target_fl"]
            if action["kind"] == "speed":
                self.last_speeds[action["acid"]] = action["target_speed"]
            row = self._add_command_row(
                action["acid"],
                action["kind"],
                command,
                instruction_by_aircraft.get(action["acid"], "Verified conflict-resolution command"),
                "planned",
            )
            monitor_record = self._register_command_monitor(row, action)
            self._stack(command, monitor_record=monitor_record)
            self.issued_commands.add(command)

        decision_text = "%s via %s" % (solver["preference"], solver["method"])
        command_text = "; ".join(action["command"] for action in actions)
        instruction_text = " ".join(llm_output["standard_instructions"])
        display_rows = self._sync_tracked_conflicts(aircraft, detections, actions, default_state="Issued")
        self._refresh_conflict_table(display_rows, actions, state="Issued")
        self._log_text(
            "Decision %s | Aircraft %s | CPA %s | CMD %s | Instruction %s"
            % (decision_text, aircraft_text, cpa_text, command_text, instruction_text)
        )
        for _tcpa, _hsep, _vsep, a, b, pair in detections:
            self.resolved_pairs.add(pair)
        self._append_log({
            "event": "conflicts_detected_and_resolved",
            "detections": [
                {
                    "pair": [a["id"], b["id"]],
                    "tcpa_min": tcpa,
                    "hsep_nm": hsep,
                    "vsep_ft": vsep,
                }
                for tcpa, hsep, vsep, a, b, _pair in detections
            ],
            "solver": solver,
            "commands": [action["command"] for action in actions],
            "llm_output": llm_output,
        })
        self._update_status_labels()
