"""Headless validation for the dynamic BlueSky ATC sector.

This script runs without the QtGL GUI. It creates a three-route sector, spawns
up to 14 aircraft at boundary fixes, monitors predicted CPA, issues altitude
resolution commands, and writes a JSONL event log plus a summary JSON.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parent
DEFAULT_BLUESKY_ROOT = WORKSPACE_ROOT.parent / "bluesky_project"
BLUESKY_ROOT = Path(os.environ.get("BLUESKY_ROOT", str(DEFAULT_BLUESKY_ROOT)))
if str(BLUESKY_ROOT) not in sys.path:
    sys.path.insert(0, str(BLUESKY_ROOT))
os.chdir(BLUESKY_ROOT)

import bluesky as bs  # noqa: E402
from bluesky import stack  # noqa: E402
from bluesky.tools.aero import ft, fpm  # noqa: E402


CENTER_LAT = 30.7000
CENTER_LON = 104.1000
MAX_AIRCRAFT = 14
LOOKAHEAD_MIN = 20.0
HSEP_NM = 5.0
VSEP_FT = 1000.0
VSEP_EPS_FT = 1.0
VERIFY_HSEP_NM = 7.0
VERIFY_VSEP_FT = 1000.0
ALT_DELTAS_FL = [10, 20, 30]
VS_FPM = 2000
SIM_DT = 1.0
SIM_DURATION_SEC = float(os.environ.get("ATC_SIM_DURATION_SEC", str(20 * 60)))
MONITOR_INTERVAL_SEC = 2
SPAWN_INTERVAL_SEC = float(os.environ.get("ATC_SPAWN_INTERVAL_SEC", "150"))
RNG_SEED = int(os.environ.get("ATC_RNG_SEED", "20260703"))
LOG_DIR = Path(os.environ.get("ATC_LOG_DIR", WORKSPACE_ROOT / "headless_dynamic_logs"))
ENTRY_GATE_NM = 12.0
PREDICT_GATE_NM = 20.0
SPAWN_ENTRY_GATE_NM = 20.0
SPAWN_PREDICT_GATE_NM = 20.0
ENTRY_VERTICAL_GATE_FT = 5000.0
PREDICT_VERTICAL_GATE_FT = 5000.0
SAFE_LEVELS = list(range(270, 391, 10))
MIN_TARGET_FL_GAP = 10
ALT_TARGET_LOCK_FT = 500.0
SPEED_DELTAS_KT = [-20, 20, -30, 30]
MIN_SPEED_KT = 250
MAX_SPEED_KT = 330
VERIFY_DT_SEC = 2
SPEED_ACCEL_KT_PER_SEC = 1.0
RESOLUTION_PREFERENCE = os.environ.get("ATC_RESOLUTION_PREFERENCE", "speed_first")
ALLOW_SPEED_ACTIONS = os.environ.get("ATC_ALLOW_SPEED_ACTIONS", "1") == "1"
MAX_SEARCH_NODES = int(os.environ.get("ATC_MAX_SEARCH_NODES", "100000"))
SEARCH_TIME_LIMIT_SEC = float(os.environ.get("ATC_SEARCH_TIME_LIMIT_SEC", "5.0"))
ENABLE_UNVERIFIED_FALLBACK = os.environ.get("ATC_ENABLE_UNVERIFIED_FALLBACK", "0") == "1"


WAYPOINTS = {
    "W_IN": (30.7000, 102.7500),
    "E_IN": (30.7000, 105.4500),
    "N_IN": (31.8500, 104.1000),
    "S_IN": (29.5500, 104.1000),
    "SW_IN": (29.7500, 103.1500),
    "NE_IN": (31.6500, 105.0500),
    "NW_IN": (31.6500, 103.1500),
    "SE_IN": (29.7500, 105.0500),
}

ROUTES = [
    {"name": "R1-EW", "entry": "W_IN", "exit": "E_IN", "hdg": 90, "fls": [320, 340, 360], "speed": (290, 320)},
    {"name": "R1-WE", "entry": "E_IN", "exit": "W_IN", "hdg": 270, "fls": [320, 340, 360], "speed": (290, 320)},
    {"name": "R2-NS", "entry": "N_IN", "exit": "S_IN", "hdg": 180, "fls": [330, 350, 370], "speed": (280, 310)},
    {"name": "R2-SN", "entry": "S_IN", "exit": "N_IN", "hdg": 0, "fls": [330, 350, 370], "speed": (280, 310)},
    {"name": "R3-SWNE", "entry": "SW_IN", "exit": "NE_IN", "hdg": 45, "fls": [310, 330, 350], "speed": (280, 310)},
    {"name": "R3-NESW", "entry": "NE_IN", "exit": "SW_IN", "hdg": 225, "fls": [310, 330, 350], "speed": (280, 310)},
]

AIRCRAFT_TYPES = ["A320", "B738", "A319", "E190"]


@dataclass
class AircraftState:
    acid: str
    lat: float
    lon: float
    alt_ft: float
    trk: float
    gs_mps: float


@dataclass(frozen=True)
class CandidateAction:
    acid: str
    kind: str
    target_fl: int
    target_speed_kt: int
    command: str | None
    label: str


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def xy_nm(lat: float, lon: float) -> tuple[float, float]:
    x = (lon - CENTER_LON) * 60.0 * math.cos(math.radians(CENTER_LAT))
    y = (lat - CENTER_LAT) * 60.0
    return x, y


def velocity_nm_min(trk_deg: float, gs_mps: float) -> tuple[float, float]:
    rad = math.radians(trk_deg)
    speed_nm_min = gs_mps * 60.0 / 1852.0
    return speed_nm_min * math.sin(rad), speed_nm_min * math.cos(rad)


def cpa(a: AircraftState, b: AircraftState) -> tuple[float, float, float]:
    ax, ay = xy_nm(a.lat, a.lon)
    bx, by = xy_nm(b.lat, b.lon)
    avx, avy = velocity_nm_min(a.trk, a.gs_mps)
    bvx, bvy = velocity_nm_min(b.trk, b.gs_mps)
    rx, ry = bx - ax, by - ay
    vx, vy = bvx - avx, bvy - avy
    vv = vx * vx + vy * vy
    tcpa = 0.0 if vv <= 1e-9 else max(0.0, min(LOOKAHEAD_MIN, -((rx * vx + ry * vy) / vv)))
    dx, dy = rx + vx * tcpa, ry + vy * tcpa
    return tcpa, math.hypot(dx, dy), abs(a.alt_ft - b.alt_ft)


def current_hsep_nm(a: AircraftState, b: AircraftState) -> float:
    ax, ay = xy_nm(a.lat, a.lon)
    bx, by = xy_nm(b.lat, b.lon)
    return math.hypot(bx - ax, by - ay)


class HeadlessSectorRunner:
    def __init__(self) -> None:
        self.rng = random.Random(RNG_SEED)
        self.spawn_index = 0
        self.active_meta: dict[str, dict] = {}
        self.resolved_pairs: set[tuple[str, str]] = set()
        self.last_targets: dict[str, int] = {}
        self.last_speed_targets: dict[str, int] = {}
        self.commands_issued: list[str] = []
        self.issued_commands: set[str] = set()
        self.solver_stats: list[dict] = []
        self.min_hsep_nm = float("inf")
        self.min_vsep_ft_when_hloss = float("inf")
        self.loss_events: list[dict] = []
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_token = f"seed{RNG_SEED}_pid{os.getpid()}_{stamp}"
        self.run_id = f"dynamic_sector_headless_{run_token}"
        self.sample_index = 0
        self.log_path = LOG_DIR / f"dynamic_sector_headless_{run_token}.jsonl"
        self.summary_path = LOG_DIR / f"dynamic_sector_headless_{run_token}_summary.json"

    def log(self, event: str, **payload) -> None:
        record = {"time": now_iso(), "simt": float(getattr(bs.sim, "simt", 0.0)), "event": event, **payload}
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    def stack(self, command: str) -> None:
        stack.stack(command)

    def flush_stack(self, n: int = 3) -> None:
        for _ in range(n):
            stack.process()

    def init_bluesky(self) -> None:
        print("Initializing BlueSky detached...", flush=True)
        bs.init(mode="sim-detached", scnfile="")
        bs.sim.setdt(SIM_DT)
        bs.sim.fastforward(None)
        self.stack("RESET")
        self.stack("HOLD")
        self.flush_stack()
        self.log("scenario_initialized", sim_dt=SIM_DT)

    def candidate_is_safe(self, route: dict, fl: int, speed_kt: int) -> tuple[bool, str]:
        entry_lat, entry_lon = WAYPOINTS[route["entry"]]
        candidate = AircraftState(
            acid="CAND",
            lat=entry_lat,
            lon=entry_lon,
            alt_ft=fl * 100.0,
            trk=float(route["hdg"]),
            gs_mps=speed_kt * 0.514444,
        )
        for state in self.get_states():
            hnow = current_hsep_nm(candidate, state)
            vnow = abs(candidate.alt_ft - self.effective_alt_ft(state))
            if hnow < SPAWN_ENTRY_GATE_NM and vnow < ENTRY_VERTICAL_GATE_FT:
                return False, f"entry_gate:{state.acid}:h={hnow:.2f}:v={vnow:.0f}"
            tcpa, hsep, vsep = cpa(candidate, state)
            target_vsep = abs(candidate.alt_ft - self.effective_alt_ft(state))
            if hsep < SPAWN_PREDICT_GATE_NM and target_vsep < PREDICT_VERTICAL_GATE_FT:
                return False, f"predict_gate:{state.acid}:tcpa={tcpa:.2f}:h={hsep:.2f}:v={target_vsep:.0f}"
        return True, "ok"

    def spawn_aircraft(self, route: dict, fl: int | None = None, require_safe: bool = True) -> str | None:
        if len(self.active_meta) >= MAX_AIRCRAFT:
            return None
        selected_route = route
        selected_fl = fl
        selected_speed = None
        if require_safe:
            candidate_routes = [route] + self.rng.sample(ROUTES, len(ROUTES))
            attempts = 0
            last_reason = "not_checked"
            for candidate_route in candidate_routes:
                candidate_levels = [selected_fl] if selected_fl is not None else list(SAFE_LEVELS)
                self.rng.shuffle(candidate_levels)
                for candidate_fl in candidate_levels:
                    attempts += 1
                    speed = self.rng.randint(candidate_route["speed"][0], candidate_route["speed"][1])
                    safe, reason = self.candidate_is_safe(candidate_route, candidate_fl, speed)
                    last_reason = reason
                    if safe:
                        selected_route = candidate_route
                        selected_fl = candidate_fl
                        selected_speed = speed
                        break
                if selected_speed is not None:
                    break
            if selected_speed is None:
                self.log("spawn_skipped_by_entry_gate", requested_route=route["name"], attempts=attempts, reason=last_reason)
                return None
        self.spawn_index += 1
        acid = f"DYN{self.spawn_index:03d}"
        actype = self.rng.choice(AIRCRAFT_TYPES)
        route = selected_route
        entry_lat, entry_lon = WAYPOINTS[route["entry"]]
        exit_lat, exit_lon = WAYPOINTS[route["exit"]]
        fl = selected_fl if selected_fl is not None else self.rng.choice(route["fls"])
        speed = selected_speed if selected_speed is not None else self.rng.randint(route["speed"][0], route["speed"][1])
        commands = [
            f"CRE {acid},{actype},{entry_lat:.6f},{entry_lon:.6f},{route['hdg']},FL{fl},{speed}",
            f"ADDWPT {acid} {exit_lat:.6f} {exit_lon:.6f} FL{fl} {speed}",
            f"{acid} LNAV ON",
        ]
        for command in commands:
            self.stack(command)
        self.flush_stack()
        self.active_meta[acid] = {
            "type": actype,
            "route": route["name"],
            "fl": fl,
            "speed": speed,
            "entry": route["entry"],
            "exit": route["exit"],
        }
        self.log("aircraft_spawned", acid=acid, route=route["name"], fl=fl, speed=speed, commands=commands)
        return acid

    def spawn_initial_wave(self) -> None:
        print("Spawning initial wave...", flush=True)
        initial_conflict_specs = [
            (ROUTES[0], ROUTES[1], 340),
            (ROUTES[2], ROUTES[3], 370),
        ]
        for route_a, route_b, fl in initial_conflict_specs:
            self.spawn_aircraft(route_a, fl, require_safe=False)
            self.spawn_aircraft(route_b, fl, require_safe=False)
        for route, fl in [(ROUTES[4], 310), (ROUTES[5], 390)]:
            self.spawn_aircraft(route, fl, require_safe=True)
        self.stack("OP")
        self.flush_stack()
        bs.sim.op()
        bs.sim.fastforward(None)
        self.monitor_and_resolve()
        self.log("initial_wave_spawned", ntraf=len(bs.traf.id))
        print(f"Initial wave ready: commanded={len(self.active_meta)}, ntraf={len(bs.traf.id)}", flush=True)

    def get_states(self) -> list[AircraftState]:
        states: list[AircraftState] = []
        for i, acid in enumerate(bs.traf.id):
            states.append(
                AircraftState(
                    acid=str(acid),
                    lat=float(bs.traf.lat[i]),
                    lon=float(bs.traf.lon[i]),
                    alt_ft=float(bs.traf.alt[i]) / ft,
                    trk=float(bs.traf.trk[i]),
                    gs_mps=float(bs.traf.gs[i]),
                )
            )
        return states

    def effective_alt_ft(self, state: AircraftState) -> float:
        target_fl = self.last_targets.get(state.acid)
        return target_fl * 100.0 if target_fl is not None else state.alt_ft

    def effective_speed_kt(self, state: AircraftState) -> int:
        target_speed = self.last_speed_targets.get(state.acid)
        if target_speed is not None:
            return target_speed
        meta_speed = self.route_meta(state.acid).get("speed")
        if meta_speed is not None:
            return int(round(meta_speed))
        return int(round(state.gs_mps / 0.514444))

    def current_ground_speed_kt(self, state: AircraftState) -> float:
        return float(state.gs_mps / 0.514444)

    def nearest_safe_level(self, state: AircraftState) -> int:
        current_fl = int(round(state.alt_ft / 100.0))
        return min(SAFE_LEVELS, key=lambda fl: abs(fl - current_fl))

    def generate_candidate_actions(self, state: AircraftState) -> list[CandidateAction]:
        current_fl = int(round(state.alt_ft / 100.0))
        effective_fl = int(round(self.effective_alt_ft(state) / 100.0))
        current_speed = self.effective_speed_kt(state)
        active_target_fl = self.last_targets.get(state.acid)
        altitude_locked = active_target_fl is not None
        candidates: list[CandidateAction] = [
            CandidateAction(
                acid=state.acid,
                kind="hold",
                target_fl=effective_fl,
                target_speed_kt=current_speed,
                command=None,
                label="hold",
            )
        ]

        if not altitude_locked:
            altitude_levels = sorted(
                [fl for fl in SAFE_LEVELS if fl != effective_fl],
                key=lambda fl: (abs(fl - effective_fl), abs(fl - current_fl)),
            )
            for fl in altitude_levels:
                vs = VS_FPM if fl * 100.0 > state.alt_ft else -VS_FPM
                candidates.append(
                    CandidateAction(
                        acid=state.acid,
                        kind="altitude",
                        target_fl=fl,
                        target_speed_kt=current_speed,
                        command=f"ALT {state.acid},FL{fl},{vs}",
                        label=f"altitude:FL{fl}",
                    )
                )

        if ALLOW_SPEED_ACTIONS:
            seen_speeds: set[int] = set()
            for delta in SPEED_DELTAS_KT:
                target_speed = max(MIN_SPEED_KT, min(MAX_SPEED_KT, current_speed + delta))
                if target_speed == current_speed or target_speed in seen_speeds:
                    continue
                seen_speeds.add(target_speed)
                candidates.append(
                    CandidateAction(
                        acid=state.acid,
                        kind="speed",
                        target_fl=effective_fl,
                        target_speed_kt=target_speed,
                        command=f"SPD {state.acid},{target_speed}",
                        label=f"speed:{target_speed}",
                    )
                )

        if RESOLUTION_PREFERENCE == "speed_first":
            order = {"hold": 0, "speed": 1, "altitude": 2}
        else:
            order = {"hold": 0, "altitude": 1, "speed": 2}
        return sorted(
            candidates,
            key=lambda action: (
                order.get(action.kind, 9),
                abs(action.target_fl - effective_fl),
                abs(action.target_speed_kt - current_speed),
            ),
        )

    def speed_distance_nm(self, start_speed_kt: float, target_speed_kt: float, t_sec: float) -> float:
        delta = target_speed_kt - start_speed_kt
        if abs(delta) <= 1e-9:
            return start_speed_kt * t_sec / 3600.0
        direction = 1.0 if delta > 0 else -1.0
        ramp_time = abs(delta) / SPEED_ACCEL_KT_PER_SEC
        if t_sec <= ramp_time:
            end_speed = start_speed_kt + direction * SPEED_ACCEL_KT_PER_SEC * t_sec
            return ((start_speed_kt + end_speed) / 2.0) * t_sec / 3600.0
        ramp_distance = ((start_speed_kt + target_speed_kt) / 2.0) * ramp_time / 3600.0
        cruise_distance = target_speed_kt * (t_sec - ramp_time) / 3600.0
        return ramp_distance + cruise_distance

    def predicted_state(self, state: AircraftState, action: CandidateAction, t_sec: float) -> tuple[float, float, float]:
        x0, y0 = xy_nm(state.lat, state.lon)
        start_command_speed_kt = max(1.0, float(self.effective_speed_kt(state)))
        start_ground_speed_kt = self.current_ground_speed_kt(state)
        target_ground_speed_kt = start_ground_speed_kt * (float(action.target_speed_kt) / start_command_speed_kt)
        distance_nm = self.speed_distance_nm(start_ground_speed_kt, target_ground_speed_kt, t_sec)
        rad = math.radians(state.trk)
        x = x0 + distance_nm * math.sin(rad)
        y = y0 + distance_nm * math.cos(rad)
        target_alt = action.target_fl * 100.0
        if abs(target_alt - state.alt_ft) <= 1e-6:
            alt = target_alt
        else:
            direction = 1.0 if target_alt > state.alt_ft else -1.0
            delta = direction * VS_FPM * (t_sec / 60.0)
            if direction > 0:
                alt = min(target_alt, state.alt_ft + delta)
            else:
                alt = max(target_alt, state.alt_ft + delta)
        return x, y, alt

    def action_pair_is_safe(
        self,
        a: AircraftState,
        action_a: CandidateAction,
        b: AircraftState,
        action_b: CandidateAction,
    ) -> bool:
        horizon_sec = int(LOOKAHEAD_MIN * 60)
        for t_sec in range(0, horizon_sec + VERIFY_DT_SEC, VERIFY_DT_SEC):
            ax, ay, aalt = self.predicted_state(a, action_a, t_sec)
            bx, by, balt = self.predicted_state(b, action_b, t_sec)
            hsep = math.hypot(bx - ax, by - ay)
            vsep = abs(aalt - balt)
            if hsep < VERIFY_HSEP_NM and vsep < VERIFY_VSEP_FT - VSEP_EPS_FT:
                return False
        return True

    def current_targets_are_safe(self, a: AircraftState, b: AircraftState) -> bool:
        action_a = CandidateAction(
            acid=a.acid,
            kind="hold",
            target_fl=int(round(self.effective_alt_ft(a) / 100.0)),
            target_speed_kt=self.effective_speed_kt(a),
            command=None,
            label="current_target",
        )
        action_b = CandidateAction(
            acid=b.acid,
            kind="hold",
            target_fl=int(round(self.effective_alt_ft(b) / 100.0)),
            target_speed_kt=self.effective_speed_kt(b),
            command=None,
            label="current_target",
        )
        return self.action_pair_is_safe(a, action_a, b, action_b)

    def route_meta(self, acid: str) -> dict:
        return self.active_meta.get(acid, {})

    def pending_alt_dir(self, state: AircraftState) -> str:
        target_fl = self.last_targets.get(state.acid)
        if target_fl is None:
            return "none"
        diff_ft = target_fl * 100.0 - state.alt_ft
        if abs(diff_ft) <= ALT_TARGET_LOCK_FT:
            return "none"
        return "climb" if diff_ft > 0 else "descend"

    def action_id(self, action: CandidateAction) -> str:
        if action.kind == "hold":
            return f"{action.acid}_hold"
        if action.kind == "speed":
            return f"{action.acid}_spd_{action.target_speed_kt}"
        if action.kind == "altitude":
            return f"{action.acid}_alt_{action.target_fl}"
        return f"{action.acid}_{action.kind}_{action.target_fl}_{action.target_speed_kt}"

    def heading_delta_deg(self, a: float, b: float) -> int:
        return int(round(abs((a - b + 180.0) % 360.0 - 180.0)))

    def edge_geometry(self, a: AircraftState, b: AircraftState) -> str:
        delta = self.heading_delta_deg(a.trk, b.trk)
        if delta >= 150:
            return "head_on"
        if delta <= 20:
            return "overtaking" if current_hsep_nm(a, b) < PREDICT_GATE_NM else "parallel"
        if 55 <= delta <= 125:
            return "crossing"
        return "converging"

    def risk_bucket(self, tcpa: float, cpa_hsep: float, current_hsep: float) -> str:
        if current_hsep < HSEP_NM or cpa_hsep < 1.0 or tcpa <= 2.0:
            return "critical"
        if cpa_hsep < 3.0 or tcpa <= 5.0:
            return "high"
        if cpa_hsep < HSEP_NM or tcpa <= 8.0:
            return "medium"
        return "low"

    def rounded_vsep(self, value_ft: float) -> int:
        return int(round(value_ft / 100.0) * 100)

    def action_deviation_cost(self, state: AircraftState, action: CandidateAction) -> int:
        effective_fl = int(round(self.effective_alt_ft(state) / 100.0))
        effective_speed = self.effective_speed_kt(state)
        if action.kind == "altitude":
            return abs(action.target_fl - effective_fl)
        if action.kind == "speed":
            return abs(action.target_speed_kt - effective_speed)
        return 0

    def level_occupancy(self, state: AircraftState, action: CandidateAction, state_by_id: dict[str, AircraftState]) -> tuple[str, int]:
        if action.kind in {"hold", "speed"}:
            target_fl = int(round(self.effective_alt_ft(state) / 100.0))
            default_status = "current"
        else:
            target_fl = action.target_fl
            default_status = "free"

        status = default_status
        conflict_count = 0
        target_alt_ft = target_fl * 100.0
        for other in state_by_id.values():
            if other.acid == state.acid:
                continue
            other_alt_ft = self.effective_alt_ft(other)
            if abs(target_alt_ft - other_alt_ft) >= VERIFY_VSEP_FT:
                continue
            conflict_count += 1
            hnow = current_hsep_nm(state, other)
            tcpa, hsep, _vsep = cpa(state, other)
            if hnow < VERIFY_HSEP_NM:
                status = "blocked"
            elif hsep < PREDICT_GATE_NM:
                if status not in {"blocked", "crossing_risk"}:
                    status = "crossing_risk"
            elif status not in {"blocked", "crossing_risk"}:
                status = "occupied"

        return status, conflict_count

    def model_candidate_actions(
        self,
        acid: str,
        state_by_id: dict[str, AircraftState],
        selected_label: str | None = None,
    ) -> tuple[list[list], dict[str, CandidateAction]]:
        state = state_by_id[acid]
        effective_fl = int(round(self.effective_alt_ft(state) / 100.0))
        allowed_alt_levels = {
            effective_fl + sign * delta
            for delta in ALT_DELTAS_FL
            for sign in (-1, 1)
            if min(SAFE_LEVELS) <= effective_fl + sign * delta <= max(SAFE_LEVELS)
        }
        encoded: list[list] = []
        action_map: dict[str, CandidateAction] = {}
        for action in self.generate_candidate_actions(state):
            keep = action.kind in {"hold", "speed"} or action.target_fl in allowed_alt_levels or action.label == selected_label
            if not keep:
                continue
            level_status, level_conflict_count = self.level_occupancy(state, action, state_by_id)
            if level_status == "blocked" and action.label != selected_label:
                continue
            aid = self.action_id(action)
            encoded.append(
                [
                    aid,
                    action.kind,
                    action.target_fl,
                    action.target_speed_kt,
                    self.action_deviation_cost(state, action),
                    "occupied" if level_status == "blocked" else level_status,
                    level_conflict_count,
                ]
            )
            action_map[aid] = action
        return encoded, action_map

    def verify_selected_actions(
        self,
        detections: list[tuple],
        selected_actions: dict[str, CandidateAction],
    ) -> dict:
        if not selected_actions:
            return {
                "schema": "forward_verification_v1",
                "safe": False,
                "verifier": "bluesky_forward_sim",
                "lookahead_min": LOOKAHEAD_MIN,
                "dt_sec": VERIFY_DT_SEC,
                "min_hsep_nm": None,
                "min_vsep_ft_when_hsep_below_verify": None,
                "loss_events": 0,
                "invalid_action_ids": [],
                "fallback_used": False,
            }
        min_hsep = float("inf")
        min_vsep_when_hloss = float("inf")
        loss_events = 0
        horizon_sec = int(LOOKAHEAD_MIN * 60)
        for _tcpa, _hsep, _vsep, a, b, _pair in detections:
            action_a = selected_actions.get(a.acid)
            action_b = selected_actions.get(b.acid)
            if action_a is None or action_b is None:
                loss_events += 1
                continue
            for t_sec in range(0, horizon_sec + VERIFY_DT_SEC, VERIFY_DT_SEC):
                ax, ay, aalt = self.predicted_state(a, action_a, t_sec)
                bx, by, balt = self.predicted_state(b, action_b, t_sec)
                hsep = math.hypot(bx - ax, by - ay)
                vsep = abs(aalt - balt)
                min_hsep = min(min_hsep, hsep)
                if hsep < VERIFY_HSEP_NM:
                    min_vsep_when_hloss = min(min_vsep_when_hloss, vsep)
                if hsep < VERIFY_HSEP_NM and vsep < VERIFY_VSEP_FT - VSEP_EPS_FT:
                    loss_events += 1
        return {
            "schema": "forward_verification_v1",
            "safe": loss_events == 0,
            "verifier": "bluesky_forward_sim",
            "lookahead_min": LOOKAHEAD_MIN,
            "dt_sec": VERIFY_DT_SEC,
            "min_hsep_nm": None if min_hsep == float("inf") else round(min_hsep, 2),
            "min_vsep_ft_when_hsep_below_verify": None
            if min_vsep_when_hloss == float("inf")
            else self.rounded_vsep(min_vsep_when_hloss),
            "loss_events": loss_events,
            "invalid_action_ids": [],
            "fallback_used": False,
        }

    def build_training_sample(
        self,
        state_by_id: dict[str, AircraftState],
        detections: list[tuple],
        solver_info: dict,
    ) -> dict:
        self.sample_index += 1
        graph_ids = sorted({a.acid for *_prefix, a, b, _pair in detections} | {b.acid for *_prefix, a, b, _pair in detections})
        selected_labels = solver_info.get("selected_actions", {})
        candidate_actions: dict[str, list[list]] = {}
        action_lookup: dict[str, CandidateAction] = {}
        selected_action_ids: list[str] = []
        selected_actions: dict[str, CandidateAction] = {}
        invalid_action_ids: list[str] = []

        for acid in graph_ids:
            encoded, action_map = self.model_candidate_actions(acid, state_by_id, selected_labels.get(acid))
            candidate_actions[acid] = encoded
            action_lookup.update(action_map)
            label = selected_labels.get(acid)
            if label is None:
                continue
            matched = next((aid for aid, action in action_map.items() if action.label == label), None)
            if matched is None:
                invalid_action_ids.append(f"{acid}:{label}")
                continue
            selected_action_ids.append(matched)
            selected_actions[acid] = action_map[matched]

        aircraft_rows = []
        aircraft_full = []
        for acid in graph_ids:
            state = state_by_id[acid]
            meta = self.route_meta(acid)
            route = meta.get("route", "unknown")
            target_fl = int(round(self.effective_alt_ft(state) / 100.0))
            speed_kt = self.effective_speed_kt(state)
            target_speed = self.effective_speed_kt(state)
            pending_dir = self.pending_alt_dir(state)
            aircraft_rows.append(
                [
                    acid,
                    route,
                    int(round(state.alt_ft / 100.0)),
                    int(round(state.trk)),
                    speed_kt,
                    target_fl,
                    target_speed,
                    pending_dir,
                ]
            )
            aircraft_full.append(
                {
                    "id": acid,
                    "type": meta.get("type", "unknown"),
                    "route": route,
                    "entry": meta.get("entry"),
                    "exit": meta.get("exit"),
                    "lat": round(state.lat, 6),
                    "lon": round(state.lon, 6),
                    "fl": int(round(state.alt_ft / 100.0)),
                    "alt_ft": round(state.alt_ft, 1),
                    "trk_deg": round(state.trk, 1),
                    "speed_kt": speed_kt,
                    "ground_speed_kt": int(round(state.gs_mps / 0.514444)),
                    "target_fl": target_fl,
                    "target_speed_kt": target_speed,
                    "pending_alt_dir": pending_dir,
                }
            )

        edge_rows = []
        edge_full = []
        for tcpa, hsep, vsep, a, b, pair in detections:
            hnow = current_hsep_nm(a, b)
            vnow = abs(a.alt_ft - b.alt_ft)
            geometry = self.edge_geometry(a, b)
            angle = self.heading_delta_deg(a.trk, b.trk)
            risk = self.risk_bucket(tcpa, hsep, hnow)
            edge_rows.append(
                [
                    a.acid,
                    b.acid,
                    geometry,
                    round(tcpa, 1),
                    round(hsep, 1),
                    self.rounded_vsep(vsep),
                    round(hnow, 1),
                    self.rounded_vsep(vnow),
                    angle,
                    risk,
                ]
            )
            edge_full.append(
                {
                    "id": "-".join(pair),
                    "a": a.acid,
                    "b": b.acid,
                    "geometry": geometry,
                    "tcpa_min": round(tcpa, 2),
                    "cpa_hsep_nm": round(hsep, 2),
                    "cpa_vsep_ft": self.rounded_vsep(vsep),
                    "current_hsep_nm": round(hnow, 2),
                    "current_vsep_ft": self.rounded_vsep(vnow),
                    "crossing_angle_deg": angle,
                    "risk": risk,
                    "status": "active_loss" if hnow < HSEP_NM and vnow < VSEP_FT else "predicted_conflict",
                }
            )

        if not selected_action_ids:
            status = "no_verified_solution"
        elif all(action_lookup[action_id].kind == "hold" for action_id in selected_action_ids):
            status = "already_safe"
        else:
            status = "resolved"
        reason_codes = []
        for action_id in selected_action_ids:
            action = action_lookup[action_id]
            if action.kind == "hold":
                code = "hold_current_clearance"
            elif action.kind == "speed":
                code = "speed_preference_deconflict" if RESOLUTION_PREFERENCE == "speed_first" else "minimal_deviation"
            else:
                code = "altitude_preference_deconflict" if RESOLUTION_PREFERENCE == "altitude_first" else "multi_edge_vertical_deconflict"
            reason_codes.append([action_id, code])
        if status == "already_safe":
            reason_codes = [[action_id, "current_targets_already_safe"] for action_id in selected_action_ids]
        elif not reason_codes:
            reason_codes = [["none", "no_candidate_passed_verification"]]

        verification = self.verify_selected_actions(detections, selected_actions)
        verification["invalid_action_ids"] = invalid_action_ids
        verification["fallback_used"] = bool(solver_info.get("fallback_used"))

        tags = {RESOLUTION_PREFERENCE}
        tags.add("single_pair" if len(detections) == 1 else "multi_pair")
        degrees = {acid: 0 for acid in graph_ids}
        for _tcpa, _hsep, _vsep, a, b, _pair in detections:
            degrees[a.acid] += 1
            degrees[b.acid] += 1
            tags.add(self.edge_geometry(a, b))
        if any(degree > 1 for degree in degrees.values()):
            tags.add("multi_edge_aircraft")
        for _tcpa, hsep, _vsep, _a, _b, _pair in detections:
            if hsep < 1.0:
                tags.add("cpa_lt_1")
            elif hsep < 3.0:
                tags.add("cpa_1_3")
            elif hsep < HSEP_NM:
                tags.add("cpa_3_5")
        if status == "no_verified_solution":
            tags.add("no_verified_solution")
        for action_id in selected_action_ids:
            action = action_lookup[action_id]
            if action.kind == "speed":
                tags.add("speed_solution")
            elif action.kind == "altitude":
                tags.add("altitude_solution")
            elif action.kind == "hold":
                tags.add("hold_solution")

        state_full = {
            "schema": "bluesky_conflict_state_v1",
            "scenario": {
                "name": "ATC_HMI_DYNAMIC_14AC_SECTOR",
                "sim_time_sec": round(float(getattr(bs.sim, "simt", 0.0)), 1),
                "route_set_id": "chengdu_like_6route_v1",
                "traffic_count": len(state_by_id),
            },
            "context": {
                "preference": RESOLUTION_PREFERENCE,
                "allow_altitude_reversal": False,
            },
            "constraints": {
                "lookahead_min": LOOKAHEAD_MIN,
                "detect_hsep_nm": HSEP_NM,
                "detect_vsep_ft": VSEP_FT,
                "predict_gate_nm": PREDICT_GATE_NM,
                "verify_hsep_nm": VERIFY_HSEP_NM,
                "verify_vsep_ft": VERIFY_VSEP_FT,
                "safe_fl_min": min(SAFE_LEVELS),
                "safe_fl_max": max(SAFE_LEVELS),
                "safe_fl_step": 10,
                "speed_min_kt": MIN_SPEED_KT,
                "speed_max_kt": MAX_SPEED_KT,
                "vertical_rate_fpm": VS_FPM,
                "speed_accel_kt_per_sec": SPEED_ACCEL_KT_PER_SEC,
            },
            "aircraft": aircraft_full,
            "conflict_edges": edge_full,
            "candidate_actions": candidate_actions,
            "graph_summary": {
                "num_conflict_aircraft": len(graph_ids),
                "num_conflict_edges": len(detections),
                "max_degree": max(degrees.values()) if degrees else 0,
                "min_tcpa_min": round(min(item[0] for item in detections), 2),
                "min_cpa_hsep_nm": round(min(item[1] for item in detections), 2),
            },
        }

        sample_id = f"{self.run_id}_s{self.sample_index:04d}_t{int(round(float(getattr(bs.sim, 'simt', 0.0)))):04d}"
        return {
            "sample_id": sample_id,
            "split_group": f"route_set_v1_seed{RNG_SEED}",
            "sample_tags": sorted(tags),
            "state_full": state_full,
            "model_input": {
                "schema": "qwen_conflict_choice_input_v1_1",
                "pref": RESOLUTION_PREFERENCE,
                "limits": [LOOKAHEAD_MIN, VERIFY_HSEP_NM, int(VERIFY_VSEP_FT)],
                "aircraft": aircraft_rows,
                "edges": edge_rows,
                "actions": candidate_actions,
            },
            "teacher_output": {
                "schema": "qwen_conflict_choice_output_v1",
                "status": status,
                "actions": selected_action_ids,
                "reason_codes": reason_codes,
            },
            "verification": verification,
            "provenance": {
                "source": "headless_dynamic_sector_validation.py",
                "run_id": self.run_id,
                "seed": RNG_SEED,
                "solver_method": solver_info.get("method"),
                "solver": solver_info,
                "log_path": str(self.log_path),
            },
        }

    def monitor_and_resolve(self) -> None:
        states = self.get_states()
        detections = []
        state_by_id = {state.acid: state for state in states}
        for i, a in enumerate(states):
            for b in states[i + 1 :]:
                hnow = current_hsep_nm(a, b)
                vnow = abs(a.alt_ft - b.alt_ft)
                self.min_hsep_nm = min(self.min_hsep_nm, hnow)
                if hnow < HSEP_NM:
                    self.min_vsep_ft_when_hloss = min(self.min_vsep_ft_when_hloss, vnow)
                if hnow < HSEP_NM and vnow < VSEP_FT - VSEP_EPS_FT:
                    event = {"pair": [a.acid, b.acid], "hsep_nm": hnow, "vsep_ft": vnow}
                    self.loss_events.append(event)
                    self.log("loss_of_separation", **event)

                pair = tuple(sorted([a.acid, b.acid]))
                tcpa, hsep, vsep = cpa(a, b)
                target_vsep = abs(self.effective_alt_ft(a) - self.effective_alt_ft(b))
                current_vsep = abs(a.alt_ft - b.alt_ft)
                if hsep < PREDICT_GATE_NM and (target_vsep < VERIFY_VSEP_FT or current_vsep < VERIFY_VSEP_FT):
                    if not self.current_targets_are_safe(a, b):
                        self.last_targets.pop(a.acid, None)
                        self.last_targets.pop(b.acid, None)
                    detections.append((tcpa, hsep, vsep, a, b, pair))

        detections.sort(key=lambda item: (item[0], item[1]))
        if not detections:
            return

        pre_last_targets = dict(self.last_targets)
        pre_last_speed_targets = dict(self.last_speed_targets)
        commands, solver_info = self.build_resolution_plan(state_by_id, detections)
        post_last_targets = dict(self.last_targets)
        post_last_speed_targets = dict(self.last_speed_targets)
        self.last_targets = pre_last_targets
        self.last_speed_targets = pre_last_speed_targets
        training_sample = self.build_training_sample(state_by_id, detections, solver_info)
        self.last_targets = post_last_targets
        self.last_speed_targets = post_last_speed_targets
        self.log("conflict_training_sample", **training_sample)
        self.solver_stats.append(solver_info)
        issued_now = []
        for command in commands:
            if command in self.issued_commands:
                continue
            self.stack(command)
            self.commands_issued.append(command)
            self.issued_commands.add(command)
            issued_now.append(command)
        self.flush_stack()

        conflicts = []
        for tcpa, hsep, vsep, a, b, pair in detections:
            self.resolved_pairs.add(pair)
            conflicts.append(
                {
                    "pair": list(pair),
                    "tcpa_min": tcpa,
                    "predicted_hsep_nm": hsep,
                    "current_vsep_ft": abs(a.alt_ft - b.alt_ft),
                    "target_vsep_ft": abs(self.effective_alt_ft(a) - self.effective_alt_ft(b)),
                }
            )
        self.log("conflicts_detected_and_resolved", conflicts=conflicts, commands=issued_now, solver=solver_info)

    def build_resolution_plan(self, state_by_id: dict[str, AircraftState], detections: list[tuple]) -> tuple[list[str], dict]:
        graph: dict[str, set[str]] = {}
        urgency: dict[str, float] = {}
        for tcpa, _hsep, _vsep, a, b, _pair in detections:
            graph.setdefault(a.acid, set()).add(b.acid)
            graph.setdefault(b.acid, set()).add(a.acid)
            urgency[a.acid] = min(urgency.get(a.acid, float("inf")), tcpa)
            urgency[b.acid] = min(urgency.get(b.acid, float("inf")), tcpa)

        actions_by_acid = {acid: self.generate_candidate_actions(state_by_id[acid]) for acid in graph}
        order = sorted(graph, key=lambda acid: (-len(graph[acid]), urgency.get(acid, float("inf")), acid))
        checked_nodes = 0
        search_limited = False
        search_started = time.perf_counter()
        pair_cache: dict[tuple[str, int, str, int], bool] = {}
        action_index = {
            acid: {action: idx for idx, action in enumerate(actions)}
            for acid, actions in actions_by_acid.items()
        }

        def compatible(acid: str, action: CandidateAction, assigned: dict[str, CandidateAction]) -> bool:
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
                key = (
                    left,
                    action_index[left][a_action],
                    right,
                    action_index[right][b_action],
                )
                if key not in pair_cache:
                    pair_cache[key] = self.action_pair_is_safe(state_by_id[a_id], a_action, state_by_id[b_id], b_action)
                if not pair_cache[key]:
                    return False
            return True

        def search(assigned: dict[str, CandidateAction]) -> dict[str, CandidateAction] | None:
            nonlocal checked_nodes, search_limited
            checked_nodes += 1
            if checked_nodes > MAX_SEARCH_NODES or time.perf_counter() - search_started > SEARCH_TIME_LIMIT_SEC:
                search_limited = True
                return None
            if len(assigned) >= len(order):
                return dict(assigned)
            best_acid = None
            best_actions = None
            for acid in order:
                if acid in assigned:
                    continue
                feasible_actions = [action for action in actions_by_acid[acid] if compatible(acid, action, assigned)]
                if best_actions is None or len(feasible_actions) < len(best_actions):
                    best_acid = acid
                    best_actions = feasible_actions
                if best_actions is not None and len(best_actions) <= 1:
                    break
            if best_acid is None or best_actions is None:
                return dict(assigned)
            if not best_actions:
                return None
            for action in best_actions:
                if not compatible(best_acid, action, assigned):
                    continue
                assigned[best_acid] = action
                result = search(assigned)
                if result is not None:
                    return result
                assigned.pop(best_acid, None)
                if search_limited:
                    return None
            return None

        solution = search({})
        if solution is not None:
            commands = self.commands_from_solution(solution, state_by_id)
            info = {
                "method": "discrete_constraint_search",
                "preference": RESOLUTION_PREFERENCE,
                "num_conflict_aircraft": len(order),
                "num_conflict_pairs": len(detections),
                "search_nodes": checked_nodes,
                "pair_checks": len(pair_cache),
                "search_limited": search_limited,
                "search_time_limit_sec": SEARCH_TIME_LIMIT_SEC,
                "max_search_nodes": MAX_SEARCH_NODES,
                "selected_actions": {acid: action.label for acid, action in sorted(solution.items())},
                "fallback_used": False,
            }
            return commands, info

        if ENABLE_UNVERIFIED_FALLBACK:
            commands = self.build_altitude_graph_fallback(state_by_id, detections)
            method = "altitude_graph_fallback"
            fallback_used = True
        else:
            commands = []
            method = "no_safe_solution_hold"
            fallback_used = False
        info = {
            "method": method,
            "preference": RESOLUTION_PREFERENCE,
            "num_conflict_aircraft": len(order),
            "num_conflict_pairs": len(detections),
            "search_nodes": checked_nodes,
            "pair_checks": len(pair_cache),
            "search_limited": search_limited,
            "search_time_limit_sec": SEARCH_TIME_LIMIT_SEC,
            "max_search_nodes": MAX_SEARCH_NODES,
            "selected_actions": {},
            "fallback_used": fallback_used,
        }
        return commands, info

    def commands_from_solution(self, solution: dict[str, CandidateAction], state_by_id: dict[str, AircraftState]) -> list[str]:
        commands = []
        for acid, action in sorted(solution.items()):
            state = state_by_id[acid]
            current_fl = int(round(state.alt_ft / 100.0))
            current_speed = self.effective_speed_kt(state)
            if action.kind == "hold":
                if action.target_fl != current_fl:
                    self.last_targets[acid] = action.target_fl
                continue
            if action.kind == "altitude":
                if self.last_targets.get(acid) == action.target_fl:
                    continue
                commands.append(action.command)
                self.last_targets[acid] = action.target_fl
            elif action.kind == "speed":
                if self.last_speed_targets.get(acid) == action.target_speed_kt or action.target_speed_kt == current_speed:
                    continue
                commands.append(action.command)
                self.last_speed_targets[acid] = action.target_speed_kt
        return [command for command in commands if command]

    def build_altitude_graph_fallback(self, state_by_id: dict[str, AircraftState], detections: list[tuple]) -> list[str]:
        graph: dict[str, set[str]] = {}
        urgency: dict[str, float] = {}
        for tcpa, _hsep, _vsep, a, b, _pair in detections:
            graph.setdefault(a.acid, set()).add(b.acid)
            graph.setdefault(b.acid, set()).add(a.acid)
            urgency[a.acid] = min(urgency.get(a.acid, float("inf")), tcpa)
            urgency[b.acid] = min(urgency.get(b.acid, float("inf")), tcpa)

        targets: dict[str, int] = {
            acid: target for acid, target in self.last_targets.items() if acid in state_by_id
        }
        order = sorted(graph, key=lambda acid: (-len(graph[acid]), urgency.get(acid, float("inf")), acid))
        for acid in order:
            state = state_by_id[acid]
            preferred = targets.get(acid, self.nearest_safe_level(state))
            level_order = sorted(SAFE_LEVELS, key=lambda fl: (fl != preferred, abs(fl - preferred), abs(fl - state.alt_ft / 100.0)))
            chosen = None
            for level in level_order:
                if all(abs(level - targets[nb]) >= MIN_TARGET_FL_GAP for nb in graph[acid] if nb in targets):
                    chosen = level
                    break
            if chosen is None:
                chosen = max(
                    SAFE_LEVELS,
                    key=lambda fl: min([abs(fl - targets[nb]) for nb in graph[acid] if nb in targets] or [999]),
                )
            targets[acid] = chosen

        commands = []
        for acid in order:
            target_fl = targets[acid]
            previous_target = self.last_targets.get(acid)
            current_fl = int(round(state_by_id[acid].alt_ft / 100.0))
            if previous_target == target_fl:
                continue
            if target_fl == current_fl:
                self.last_targets[acid] = target_fl
                continue
            vs = VS_FPM if target_fl > current_fl else -VS_FPM
            command = f"ALT {acid},FL{target_fl},{vs}"
            commands.append(command)
            self.last_targets[acid] = target_fl
        return commands

    def build_resolution(self, a: AircraftState, b: AircraftState) -> list[str]:
        commands, _info = self.build_resolution_plan(
            {a.acid: a, b.acid: b},
            [(0.0, 0.0, 0.0, a, b, tuple(sorted([a.acid, b.acid])))],
        )
        return commands

    def run(self) -> dict:
        self.init_bluesky()
        self.spawn_initial_wave()
        next_spawn = SPAWN_INTERVAL_SEC
        next_monitor = 0
        while float(bs.sim.simt) < SIM_DURATION_SEC:
            simt = float(bs.sim.simt)
            if simt >= next_spawn and len(self.active_meta) < MAX_AIRCRAFT:
                self.spawn_aircraft(self.rng.choice(ROUTES))
                next_spawn += SPAWN_INTERVAL_SEC
            if simt >= next_monitor:
                self.monitor_and_resolve()
                next_monitor += MONITOR_INTERVAL_SEC
            bs.sim.step()
            if int(simt) > 0 and int(simt) % 300 == 0:
                print(f"simt={simt:.0f}s ntraf={len(bs.traf.id)} commands={len(self.commands_issued)} los={len(self.loss_events)}", flush=True)

        self.monitor_and_resolve()
        summary = {
            "success": len(self.loss_events) == 0,
            "sim_duration_sec": SIM_DURATION_SEC,
            "ntraf_final": len(bs.traf.id),
            "commands_issued": self.commands_issued,
            "num_commands": len(self.commands_issued),
            "resolved_pairs": [list(p) for p in sorted(self.resolved_pairs)],
            "num_loss_events": len(self.loss_events),
            "loss_events": self.loss_events[:20],
            "min_hsep_nm": self.min_hsep_nm,
            "min_vsep_ft_when_hsep_lt_5nm": None if self.min_vsep_ft_when_hloss == float("inf") else self.min_vsep_ft_when_hloss,
            "solver_method": "discrete_constraint_search",
            "solver_calls": len(self.solver_stats),
            "fallback_calls": sum(1 for item in self.solver_stats if item.get("fallback_used")),
            "online_verify_hsep_nm": VERIFY_HSEP_NM,
            "online_verify_vsep_ft": VERIFY_VSEP_FT,
            "online_verify_dt_sec": VERIFY_DT_SEC,
            "speed_accel_kt_per_sec": SPEED_ACCEL_KT_PER_SEC,
            "allow_speed_actions": ALLOW_SPEED_ACTIONS,
            "log_path": str(self.log_path),
        }
        self.summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
        self.log("summary", **summary)
        return summary


def main() -> int:
    runner = HeadlessSectorRunner()
    summary = runner.run()
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    print(f"SUMMARY_PATH={runner.summary_path}")
    return 0 if summary["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
