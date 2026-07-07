import os
import sys
import shutil
import time
import math
import random
import numpy as np
import torch
import bluesky as bs
from bluesky import stack, traf, tools

# ==========================================
# 0. 导入依赖
# ==========================================
from plugins.Multi_Agent.DDPG import DDPG_TRAIN, ReplayBuffer
from plugins.Multi_Agent.Normalizer_GAT10 import AircraftStateNormalizer

# ==========================================
# 1. 配置类 (Self-Prioritizing DDPG)
# ==========================================
class Config:
    def __init__(self):
        self.mode = 'train'

        # --- 路径配置 ---
        self.base_path = r"D:\pythonprogram\Autonomous-ATC-N_Closest-master"

        # 输出路径
        self.output_path = os.path.join(self.base_path, "output", "result", "DDPG_Self-Priority-4")
        self.txt_path = os.path.join(self.output_path, "result.txt")
        self.npy_path = os.path.join(self.output_path, "result.npy")

        self.scn_source = os.path.join(self.base_path, "scenario", "DQN_3D.scn")
        self.route_file = "./routes/case_study_init_4.npy"

        os.makedirs(self.output_path, exist_ok=True)
        self.model_save_path = os.path.join(self.output_path, "DDPG_Agent")
        self.scn_log_dir = os.path.join(self.output_path, f"{self.mode}_scn")
        os.makedirs(self.scn_log_dir, exist_ok=True)

        # --- 仿真参数 ---
        self.dt = 6.0
        self.num_intruders = 5

        # === 动作空间 ===
        # [spd, alt, hdg, priority]
        self.action_dim = 4 
        
        self.max_speed_delta = 4.5
        self.max_alt_delta = 300.0
        self.max_hdg_delta = 5.0 
        
        # 优先级过滤器参数
        self.max_priority_actions = 3  # (Deprecated by Group Filter)
        self.debug_priority = False     # 开启优先级日志输出

        # 物理限制
        self.limits = {
            'speed': (200, 300),
            'alt': (20000, 21000)
        }
        self.flight_levels = [alt for alt in range(20000, 21000, 300)]

        # --- RL ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.buffer_size = 50000
        self.batch_size = 1024
        self.gamma = 0.99
        self.curriculum = [(float('inf'), 18, 2000, 50)]
        
        # === 奖励权重 ===
        self.reward_weights = {
            # --- 安全 ---
            'collision': -1000.0,

            # --- 轨迹与效率 ---
            'track_error': -1.0,
            'track_sq_penalty': -0.1,
            'heading_error': -1.0,
            'progress': 1.0,
            'pos_progress': 0.5,

            # --- 稳定性 ---
            'alt_hold': -1.0,
            'action_smooth': -0.5,

            # --- 终止与边界 ---
            'boundary': -50.0,
            'goal_arrival': 200.0,
            
            # --- 优先级惩罚 (New) ---
            'priority_penalty_coeff': 0.1
        }

        self.reward_params = {
            'accept_xtk': 2.0,
            'accept_alt_err': 50.0,
            'arrival_dist': 5.0,
            'max_track_width': 20.0,
            'safe_dist': 10.0,

            # 碰撞判定阈值
            'collision_h_km': 2.0,
            'collision_v_m': 300.0,
        }


# ==========================================
# 2. 核心控制器
# ==========================================
class ACREnvironmentController:
    def __init__(self):
        self.cfg = Config()
        self.normalizer = AircraftStateNormalizer(num_intruders=self.cfg.num_intruders)
        self.replay_buffer = ReplayBuffer(self.cfg.buffer_size)

        self.agent = DDPG_TRAIN(
            ego_dim=20,
            neighbor_dim=15 * self.cfg.num_intruders,
            hidden_dim=128,
            action_dim=self.cfg.action_dim, # 4: [spd, alt, hdg, priority]
            device=self.cfg.device
        )
        
        self.active_experience = {}
        self.route_assignments = {}
        self.assigned_levels = {}
        self.routes = np.load(self.cfg.route_file)
        self.spawn_choices = [60, 70, 80]
        self.last_expert_selection = {}

        self.episode_count = 1
        self.step_count = 0
        self.num_ac_generated_total = 0
        self.win_count = 0
        self.flight_stats = {}
        self.efficiency_history = []
        self.effort_history = []
        self.safety_stats = {'los_count': 0, 'nmac_count': 0, 'min_sep_dist': 50.0}
        self.route_timers = [random.choice(self.spawn_choices) for _ in range(len(self.routes))]
        self.curr_max_conc, self.curr_max_steps, self.curr_total = self.cfg.curriculum[0][1:]
        
        # Stats for Shielding
        self.ep_total_actions = 0
        self.ep_shielded_actions = 0

        # === [统计模块] 全局累积数据 ===
        self.global_stats = {
            'total_aircraft': [],
            'success_count': [],
            'collision_count': [],
            'boundary_count': [],
            'total_extra_dist_pct': [],
            'success_sample_count': [],
            'total_min_sep': [],
            'total_warn_frames': [],
            'total_cost': [],
            'cost_sample_count': [],
            'shield_pct': [] # New: Shielded Action Percentage
        }
        self.episode_outcomes = []

        self.log_file_path = None

        print(f"\n=== Episode {self.episode_count} Started (Self-Prioritizing DDPG) ===")


    def reset(self):
        self.episode_count += 1
        if self.episode_count == 1000:
            stack.stack("STOP")
        self.step_count = 0
        self.num_ac_generated_total = 0
        self.win_count = 0
        self.active_experience = {}
        self.route_assignments = {}
        self.assigned_levels = {}
        self.flight_stats = {}
        self.last_expert_selection = {}
        self.efficiency_history = []
        self.effort_history = []
        self.safety_stats = {'los_count': 0, 'nmac_count': 0, 'min_sep_dist': 50.0}
        self.episode_outcomes = []
        self.route_timers = [random.choice(self.spawn_choices) for _ in range(len(self.routes))]
        
        # Reset Shield Stats
        self.ep_total_actions = 0
        self.ep_shielded_actions = 0

        if self.cfg.mode == 'train' and self.replay_buffer.size() > self.cfg.batch_size:
            self._update_agent(10)

        print(f"\n=== Episode {self.episode_count} ===")
        self._init_log_file()
        stack.stack('IC DQN_3D.scn')
        if self.cfg.mode == 'train' and self.episode_count % 20 == 0:
            self.agent.save(self.cfg.model_save_path)
            print(f"💾 Model saved at episode {self.episode_count}")

    # ==========================================
    # 2. 优先级过滤器 (Group-Based Priority Filter)
    # ==========================================
    def priority_filter(self, raw_agent_outputs, neighbor_map):
        """
        Input: 
            raw_agent_outputs = {acid: [spd, alt, hdg, priority]}
            neighbor_map = {acid: [[lat, lon, spd, alt, hdg, neighbor_id], ...]}
        Logic:
          1. Build Conflict Graph: Connect agents if dist < detection_range (e.g. 15km)
          2. Find Connected Components (Conflict Groups)
          3. For EACH group, select the agent with highest priority (if p > 0)
          4. Others in the group are silenced.
        """
        filtered_actions = {}
        acids = list(raw_agent_outputs.keys())
        parent = {acid: acid for acid in acids}

        def find(i):
            if parent[i] == i: return i
            parent[i] = find(parent[i])
            return parent[i]

        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j

        # 1. Build Graph & Union-Find
        # neighbor_map contains neighbors within 50km (from _get_neighbors_info_with_id)
        # We can use a tighter threshold for "Conflict Group", e.g., 20km
        GROUP_DIST_THRESHOLD = 20.0 

        for acid in acids:
            neighbors = neighbor_map.get(acid, [])
            idx_ego = traf.id2idx(acid)
            lat_ego, lon_ego = traf.lat[idx_ego], traf.lon[idx_ego]
            
            for neigh in neighbors:
                neigh_id = neigh[5] # [lat, lon, spd, alt, hdg, id]
                if neigh_id in raw_agent_outputs: # Ensure neighbor is also an active agent
                    # Calculate distance
                    dist = MathUtils.calculate_haversine_distance(lat_ego, lon_ego, neigh[0], neigh[1])
                    if dist < GROUP_DIST_THRESHOLD:
                        union(acid, neigh_id)

        # 2. Group Agents by Root
        groups = {}
        for acid in acids:
            root = find(acid)
            if root not in groups: groups[root] = []
            groups[root].append(acid)

        # 3. Process Each Group
        selected_acids = set()
        
        if self.cfg.debug_priority and len(groups) > 0:
            # Only print if there are actual groups (size > 1) or just to show activity
            interesting_groups = [g for g in groups.values() if len(g) > 1]
            if interesting_groups:
                print(f"\n--- Priority Filter Step {self.step_count} ---")
        
        for root, group_members in groups.items():
            # Sort members by priority (descending)
            sorted_members = sorted(group_members, key=lambda a: raw_agent_outputs[a][3], reverse=True)
            
            top_agent = sorted_members[0]
            top_priority = raw_agent_outputs[top_agent][3]
            
            # Check silence condition
            if len(group_members) == 1 or top_priority > 0:
                filtered_actions[top_agent] = raw_agent_outputs[top_agent][:3]
                selected_acids.add(top_agent)
                if self.cfg.debug_priority and len(group_members) > 1:
                    print(f"  Winner: {top_agent}")
            else:
                if self.cfg.debug_priority and len(group_members) > 1:
                    print(f"  All Silent (Top p={top_priority:.2f} <= 0)")
                pass

        # 4. Force others to 0
        for acid, raw_act in raw_agent_outputs.items():
            if acid not in selected_acids:
                filtered_actions[acid] = np.array([0.0, 0.0, 0.0])
        
        # --- Update Shield Stats ---
        n_total = len(raw_agent_outputs)
        n_selected = len(selected_acids)
        self.ep_total_actions += n_total
        self.ep_shielded_actions += (n_total - n_selected)
                
        return filtered_actions

    def step(self):
        self.step_count += 1
        all_spawned = (self.num_ac_generated_total >= self.curr_total)
        all_cleared = (len(traf.id) == 0)

        if self.step_count >= self.curr_max_steps or (all_spawned and all_cleared and self.step_count > 10):
            self._conclude_episode()
            self.reset()
            return

        self._spawn_traffic()
        if len(traf.id) == 0: return

        # 1. 获取邻居
        neighbor_map = self._get_neighbors_info_with_id()

        # 2. 收集状态
        gat_inputs = self._collect_gat_states(neighbor_map)
        self._update_safety_stats(gat_inputs)

        # === Phase 1: Collect Raw Actions (with Priority) ===
        raw_outputs = {} # {acid: [spd, alt, hdg, priority]}
        
        for acid, inputs in gat_inputs.items():
            noise = 0.2 if self.cfg.mode == 'train' else 0.0
            # Output is now 4D: [spd, alt, hdg, priority]
            raw_out = self.agent.take_action(inputs['ego'], inputs['neigh_flatten'], inputs['mask'],
                                             noise_sigma=noise)
            raw_outputs[acid] = raw_out

        # === Phase 2: Priority Filter ===
        # Group-based filtering
        execution_actions = self.priority_filter(raw_outputs, neighbor_map)

        rewards = {}
        dones = {}
        infos = {}

        # === Phase 3: Execution & Storage ===
        for acid, inputs in gat_inputs.items():
            # Execute filtered action
            exec_act = execution_actions[acid]
            self._apply_incremental_action(acid, exec_act)
            
            if acid in self.flight_stats: 
                self.flight_stats[acid]['fuel_proxy'] += np.linalg.norm(exec_act)

            # Compute Reward using RAW output (to penalize priority usage)
            raw_out = raw_outputs[acid]
            r, d, i = self._compute_reward_phase2(acid, raw_out, inputs)
            rewards[acid] = r
            dones[acid] = d
            infos[acid] = i

        self._update_flight_metrics()

        if self.cfg.mode == 'train':
            for acid in self.active_experience:
                if acid in gat_inputs and acid in rewards:
                    last = self.active_experience[acid]
                    self.replay_buffer.add(
                        last['state']['ego'], last['state']['neigh_flatten'], last['state']['mask'],
                        last['action'], rewards[acid], # Store RAW action (4D)
                        gat_inputs[acid]['ego'], gat_inputs[acid]['neigh_flatten'], gat_inputs[acid]['mask'],
                        dones.get(acid, False)
                    )

        self.active_experience = {}
        for acid in gat_inputs:
            if dones.get(acid, False):
                self._remove_aircraft(acid, infos.get(acid, ""))
                if acid in self.last_expert_selection: del self.last_expert_selection[acid]
                if acid in self.assigned_levels: del self.assigned_levels[acid]
            else:
                # Store RAW action for replay buffer
                self.active_experience[acid] = {'state': gat_inputs[acid], 'action': raw_outputs[acid]}

        if self.step_count % 100 == 0: self._print_status()

    # ==========================================
    # Reward Function (Updated for Priority)
    # ==========================================
    def _compute_reward_phase2(self, acid, raw_action, inputs):
        """
        raw_action: [spd, alt, hdg, priority]
        """
        w = self.cfg.reward_weights
        p_cfg = self.cfg.reward_params
        idx = traf.id2idx(acid)
        route_idx = self.route_assignments.get(acid, 0)
        route = self.routes[route_idx]

        # Extract Action and Priority
        # action_ctrl: [spd, alt, hdg]
        action_ctrl = raw_action[:3]
        priority = raw_action[3]

        last_lat, last_lon = self.flight_stats[acid]['prev_pos']
        lat, lon = traf.lat[idx], traf.lon[idx]
        target_alt = self.assigned_levels.get(acid, 20000.0)
        curr_alt_ft = traf.alt[idx] * 3.28084

        r, done, info = 0.0, False, ""

        # --- 0. Priority Penalty (New) ---
        # if p > 0: reward -= 0.1 * (p + |a|)
        if priority > 0:
            penalty = w['priority_penalty_coeff'] * (priority + np.linalg.norm(action_ctrl))
            r -= penalty

        # --- 1. 碰撞惩罚 ---
        if inputs['collision_flag'] > 0.5:
            r += w['collision']
            info = "COLLISION"
            done = True

        # --- 2. 轨迹跟踪 ---
        xtk = MathUtils.calculate_distance_to_line(lat, lon, route[0], route[1], route[2], route[3])
        r += w['track_error'] * xtk
        r += w['track_sq_penalty'] * (xtk ** 2)

        route_heading = route[4]
        dist_from_start = MathUtils.calculate_haversine_distance(route[0], route[1], lat, lon)
        bearing_from_start = MathUtils.calculate_bearing(route[0], route[1], lat, lon)
        angle_diff = math.radians(bearing_from_start - route_heading)
        xtk_signed = dist_from_start * math.sin(angle_diff)

        intercept_angle = math.degrees(math.atan(-5.0 * xtk_signed))
        desired_heading = (route_heading + intercept_angle) % 360
        curr_hdg = traf.hdg[idx]
        vf_hdg_err = abs((desired_heading - curr_hdg + 180) % 360 - 180)
        r += w['heading_error'] * (vf_hdg_err / 180.0)

        # Progress
        along_track_dist = dist_from_start * math.cos(angle_diff)
        last_dist_from_start = MathUtils.calculate_haversine_distance(route[0], route[1], last_lat, last_lon)
        last_bearing_from_start = MathUtils.calculate_bearing(route[0], route[1], last_lat, last_lon)
        last_angle_diff = math.radians(last_bearing_from_start - route_heading)
        last_along_track_dist = last_dist_from_start * math.cos(last_angle_diff)
        dist_improv = along_track_dist - last_along_track_dist
        if dist_improv > 0:
            r += min(dist_improv * w['progress'], 5.0)
        else:
            r -= abs(dist_improv) * w['pos_progress']

        # Stability
        alt_diff = abs(curr_alt_ft - target_alt)
        r += w['alt_hold'] * (alt_diff / 100.0)
        r += w['action_smooth'] * np.mean(np.abs(action_ctrl))

        # Termination
        dist_to_goal = MathUtils.calculate_haversine_distance(lat, lon, route[2], route[3])
        if xtk > p_cfg['max_track_width']:
            r += w['boundary']
            done = True
            info = "BOUNDARY"
        elif dist_to_goal < p_cfg['arrival_dist']:
            r += w['goal_arrival']
            done = True
            info = "GOAL"
            self.win_count += 1

        r = max(-500.0, min(r, 200.0))
        return r, done, info

    # ==========================================
    # 状态收集 (Implemented Cylinder Check)
    # ==========================================
    def _collect_gat_states(self, neighbor_map):
        states = {}
        raw_states = self._collect_raw_states()

        for acid in raw_states:
            route_idx = self.route_assignments.get(acid, 0)
            route = self.routes[route_idx]
            route_info = (route[0], route[1], route[2], route[3])
            target_alt = self.assigned_levels.get(acid, 20000.0)

            # --- 碰撞检测核心逻辑 ---
            idx = traf.id2idx(acid)
            own_lat, own_lon = traf.lat[idx], traf.lon[idx]
            own_alt_ft = traf.alt[idx] * 3.28084

            # 标志位：是否处于圆柱体碰撞区
            is_in_cylinder = 0.0
            min_dist_val = 50.0

            raw_neighs = neighbor_map.get(acid, [])

            for i, neigh in enumerate(raw_neighs):
                n_lat, n_lon = neigh[0], neigh[1]
                n_alt_ft = neigh[3]

                # 计算物理距离
                h_dist_km = MathUtils.calculate_haversine_distance(own_lat, own_lon, n_lat, n_lon)
                v_dist_ft = abs(own_alt_ft - n_alt_ft)

                if h_dist_km < min_dist_val and v_dist_ft < self.cfg.reward_params['collision_v_m']: min_dist_val = h_dist_km

                # A. 严格碰撞判定: H < 2km AND V < 300m (984ft)
                if h_dist_km < self.cfg.reward_params['collision_h_km'] and v_dist_ft < 300.0:
                    is_in_cylinder = 1.0

                # B. GAT Feature Flag (感知层面的接近)
                if v_dist_ft < 300.0:
                    raw_neighs[i].append(1.0)
                else:
                    raw_neighs[i].append(0.0)

            # Normalization
            norm_ego, route_heading_ref = self.normalizer.normalize_own_state_with_route(
                raw_states[acid], route_info, target_alt, 0  # ego flag unused
            )
            norm_neighs_flat = self.normalizer.normalize_other_aircraft(
                acid, raw_neighs, raw_states[acid], route_heading_ref
            )

            neigh_matrix = np.array(norm_neighs_flat).reshape(self.cfg.num_intruders, 15)
            mask = np.zeros(self.cfg.num_intruders)
            mask[:len(raw_neighs)] = 1.0

            states[acid] = {
                'ego': np.array(norm_ego),
                'neigh': neigh_matrix,
                'neigh_flatten': neigh_matrix.flatten(),
                'mask': mask,
                'min_dist_km': min_dist_val,
                'collision_flag': is_in_cylinder
            }
        return states

    def _get_neighbors_info_with_id(self):
        if len(traf.id) == 0: return {}
        neighbor_map = {}
        lats, lons, spds, alts, hdgs = traf.lat, traf.lon, traf.cas, traf.alt, traf.hdg
        for i, acid_i in enumerate(traf.id):
            dist_list = []
            for j, acid_j in enumerate(traf.id):
                if i == j: continue
                d = MathUtils.calculate_haversine_distance(lats[i], lons[i], lats[j], lons[j])
                if d < 50.0: dist_list.append((d, j))
            dist_list.sort(key=lambda x: x[0])
            top_n = dist_list[:self.cfg.num_intruders]
            n_data = []
            for _, k in top_n: n_data.append(
                [lats[k], lons[k], spds[k] * 1.9439, alts[k] * 3.28084, hdgs[k], traf.id[k]])
            neighbor_map[acid_i] = n_data
        return neighbor_map

    def _collect_raw_states(self):
        states = {}
        for i, acid in enumerate(traf.id):
            idx = traf.id2idx(acid)
            route_idx = self.route_assignments.get(acid, 0)
            route = self.routes[route_idx]
            lat_now, lon_now = traf.lat[idx], traf.lon[idx]
            xtk = MathUtils.calculate_distance_to_line(lat_now, lon_now, route[0], route[1], route[2], route[3])
            dist = MathUtils.calculate_haversine_distance(lat_now, lon_now, route[2], route[3])
            b_goal = MathUtils.calculate_bearing(lat_now, lon_now, route[2], route[3])
            hdg_err = (b_goal - traf.hdg[idx] + 180) % 360 - 180
            s = [lat_now, lon_now, traf.cas[idx] * 1.9439, traf.alt[idx] * 3.28084, traf.hdg[idx], xtk, dist, b_goal,
                 hdg_err]
            states[acid] = s
        return states

    def _apply_incremental_action(self, acid, action):
        idx = traf.id2idx(acid)
        d_spd = action[0] * self.cfg.max_speed_delta
        d_alt = action[1] * self.cfg.max_alt_delta
        d_hdg = action[2] * self.cfg.max_hdg_delta
        curr_spd = traf.cas[idx] * 1.9439
        curr_alt = traf.alt[idx] * 3.28084
        curr_hdg = traf.hdg[idx]
        target_spd = MathUtils.clamp(curr_spd + d_spd, *self.cfg.limits['speed'])
        target_alt = MathUtils.clamp(curr_alt + d_alt, *self.cfg.limits['alt'])
        target_hdg = (curr_hdg + d_hdg) % 360
        stack.stack(f'SPD {acid} {target_spd}')
        stack.stack(f'ALT {acid} {target_alt}')
        stack.stack(f'HDG {acid} {target_hdg}')
        self._write_scn_log(f'SPD {acid} {target_spd:.1f}')
        self._write_scn_log(f'ALT {acid} {target_alt:.0f}')
        self._write_scn_log(f'HDG {acid} {target_hdg:.1f}')

    def _spawn_traffic(self):
        if self.num_ac_generated_total >= self.curr_total: return
        if len(traf.id) == 0:
            for i in range(len(self.routes)):
                if len(traf.id) >= self.curr_max_conc: break
                if self.num_ac_generated_total >= self.curr_total: break
                self._create_aircraft(i)
                self.route_timers[i] = self.step_count + random.choice(self.spawn_choices)
        else:
            for k in range(len(self.route_timers)):
                if self.step_count == self.route_timers[k]:
                    if len(traf.id) >= self.curr_max_conc:
                        self.route_timers[k] += 10;
                        continue
                    self._create_aircraft(k)
                    self.route_timers[k] = self.step_count + random.choice(self.spawn_choices)
                    if self.num_ac_generated_total >= self.curr_total: break

    def _create_aircraft(self, route_idx):
        lat, lon, glat, glon, _ = self.routes[route_idx]
        acid = f"KL{self.num_ac_generated_total}"
        init_alt = random.choice(self.cfg.flight_levels)
        init_hdg = MathUtils.calculate_bearing(lat, lon, glat, glon)
        stack.stack(f'CRE {acid}, B737, {lat}, {lon}, {init_hdg}, {init_alt}, 250')
        stack.stack(f'ADDWPT {acid} {glat}, {glon}')
        stack.stack(f'VNAV {acid} ON')
        self._write_scn_log(f'CRE {acid}, B737, {lat}, {lon}, {init_hdg}, {init_alt}, 250')
        self._write_scn_log(f'ADDWPT {acid} {glat}, {glon}')
        self._write_scn_log(f'VNAV {acid} ON')
        self.route_assignments[acid] = route_idx
        self.assigned_levels[acid] = float(init_alt)
        self.num_ac_generated_total += 1
        ideal_dist = MathUtils.calculate_haversine_distance(lat, lon, glat, glon)
        self.flight_stats[acid] = {
            'dist_flown': 0.0, 'ideal_dist': ideal_dist, 'fuel_proxy': 0.0, 'prev_pos': (lat, lon),
            'min_sep': 50.0, 'warn_frames': 0, "ori": (lat, lon), "goal": (glat, glon), "counts": 0
        }

    def _update_flight_metrics(self):
        for i, acid in enumerate(traf.id):
            if acid in self.flight_stats:
                curr_lat, curr_lon = traf.lat[i], traf.lon[i]
                prev_lat, prev_lon = self.flight_stats[acid]['prev_pos']
                self.flight_stats[acid]['dist_flown'] += MathUtils.calculate_haversine_distance(prev_lat, prev_lon,
                                                                                                curr_lat, curr_lon)
                self.flight_stats[acid]['prev_pos'] = (curr_lat, curr_lon)
                self.flight_stats[acid]['counts'] += 1

    def _remove_aircraft(self, acid, reason):
        stack.stack(f'DEL {acid}')
        self._write_scn_log(f'DEL {acid}')
        if acid in self.flight_stats:
            idx = traf.id2idx(acid)
            stats = self.flight_stats[acid]
            te = stats['ideal_dist'] / max(stats['dist_flown'], 0.1)
            extra_dist_pct = 0.0
            if reason == 'GOAL':
                _lat, _lon = stats['ori']
                _glat, _glon = stats['goal']
                lat = np.linspace(_lat, _glat, stats['counts'])
                lon = np.linspace(_lon, _glon, stats['counts'])
                ideal_dist = 0
                for i in range(stats['counts'] - 1):
                    ideal_dist += MathUtils.calculate_haversine_distance(lat[i], lon[i], lat[i + 1], lon[i + 1])
                diff = stats['dist_flown'] - ideal_dist + MathUtils.calculate_haversine_distance(traf.lat[idx],
                                                                                                 traf.lon[idx], _glat,
                                                                                                 _glon)
                extra_dist_pct = (max(0.0, diff) / stats['ideal_dist']) * 100

            self.efficiency_history.append(te);
            self.effort_history.append(stats['fuel_proxy'])

            # [统计模块] 记录详细结果
            self.episode_outcomes.append({
                'acid': acid,
                'result': reason,
                'cost': stats['fuel_proxy'],
                'warn_frames': stats['warn_frames'],
                'min_sep': stats['min_sep'],
                'extra_dist_pct': extra_dist_pct
            })

            if reason == "GOAL":
                print(f"✈️ {acid} Arrived. TE={te:.3f}")
            elif reason == "COLLISION":
                print(f"💥 {acid} Collided!")
            elif reason == "BOUNDARY":
                print(f"❌ {acid} Out of Bound!")
            del self.flight_stats[acid]

    def _update_safety_stats(self, gat_inputs):
        # Reset per-step counters if needed, or accumulate
        # Here we track global min separation for the episode
        step_min_dist = 50.0
        
        for acid, data in gat_inputs.items():
            dist = data['min_dist_km']
            if dist < step_min_dist: step_min_dist = dist
            
            # Check for LOS (Loss of Separation) < 10km (example threshold)
            if dist < self.cfg.reward_params['safe_dist']:
                if acid in self.flight_stats:
                    self.flight_stats[acid]['warn_frames'] += 1
                
        if step_min_dist < self.safety_stats['min_sep_dist']:
            self.safety_stats['min_sep_dist'] = step_min_dist

    def _conclude_episode(self):
        # 1. Calculate Episode Stats
        avg_te = np.mean(self.efficiency_history) if self.efficiency_history else 0.0
        total_cost = np.sum(self.effort_history) if self.effort_history else 0.0
        
        # Count outcomes
        n_coll = sum(1 for o in self.episode_outcomes if o['result'] == 'COLLISION')
        n_bound = sum(1 for o in self.episode_outcomes if o['result'] == 'BOUNDARY')
        
        # Calculate avg warn frames
        total_warn = sum(o['warn_frames'] for o in self.episode_outcomes)
        
        # Calculate Shield Pct
        shield_pct = (self.ep_shielded_actions / self.ep_total_actions * 100) if self.ep_total_actions > 0 else 0.0
        
        # 2. Update Global Stats
        self.global_stats['total_aircraft'].append(self.num_ac_generated_total)
        self.global_stats['success_count'].append(self.win_count)
        self.global_stats['collision_count'].append(n_coll)
        self.global_stats['boundary_count'].append(n_bound)
        
        self.global_stats['total_min_sep'].append(self.safety_stats['min_sep_dist'])
        self.global_stats['total_extra_dist_pct'].append(avg_te) 
        self.global_stats['total_cost'].append(total_cost)
        
        self.global_stats['success_sample_count'].append(len(self.efficiency_history))
        self.global_stats['total_warn_frames'].append(total_warn)
        self.global_stats['cost_sample_count'].append(len(self.effort_history))
        self.global_stats['shield_pct'].append(shield_pct)

        # 3. Print Summary
        self._print_episode_stats()
        # Print cumulative every episode
        self._print_cumulative_stats(window=100)
        
        # 4. Save to NPY
        self.save_global_stats_to_npy()
        
        # 5. Curriculum Update (Optional)
        if self.win_count / max(1, self.num_ac_generated_total) > 0.9:
            # Increase difficulty if needed
            pass

    def _print_cumulative_stats(self, window=100):
        n = len(self.global_stats['total_aircraft'])
        start_idx = max(0, n - window)
        
        # Slice data
        g = self.global_stats
        total_gen = sum(g['total_aircraft'][start_idx:])
        total_succ = sum(g['success_count'][start_idx:])
        total_coll = sum(g['collision_count'][start_idx:])
        total_bound = sum(g['boundary_count'][start_idx:])
        
        # Averages
        avg_min_sep = np.mean(g['total_min_sep'][start_idx:]) if n > 0 else 0.0
        
        # Weighted averages for efficiency (based on sample counts)
        succ_samples = sum(g['success_sample_count'][start_idx:])
        cost_samples = sum(g['cost_sample_count'][start_idx:])
        
        # Reconstruct weighted sums
        w_te_sum = sum(np.array(g['total_extra_dist_pct'][start_idx:]) * np.array(g['success_sample_count'][start_idx:]))
        avg_te = w_te_sum / succ_samples if succ_samples > 0 else 0.0
        
        total_cost_sum = sum(g['total_cost'][start_idx:])
        avg_cost = total_cost_sum / cost_samples if cost_samples > 0 else 0.0
        
        # Warn frames
        total_warn = sum(g['total_warn_frames'][start_idx:])
        avg_warn = total_warn / total_gen if total_gen > 0 else 0.0
        
        # Shield Pct
        avg_shield = np.mean(g['shield_pct'][start_idx:]) if n > 0 else 0.0

        # Rates
        succ_rate = (total_succ / total_gen * 100) if total_gen > 0 else 0.0
        coll_rate = (total_coll / total_gen * 100) if total_gen > 0 else 0.0
        bound_rate = (total_bound / total_gen * 100) if total_gen > 0 else 0.0

        stats_output = f"""
        {"=" * 65}
        📊 CUMULATIVE STATS (Episodes {start_idx+1}-{n})
           Total Aircraft Gen : {total_gen}
           --------------------------------------------------
           [Performance]
           1. Success Rate    : {succ_rate:.2f}%
           2. Collision Rate  : {coll_rate:.2f}%
           3. Boundary Rate   : {bound_rate:.2f}%
           --------------------------------------------------
           [Efficiency & Cost]
           4. Extra Dist %    : {avg_te:.2f}% (Success Flights Only)
           5. Avg Op. Cost    : {avg_cost:.4f} (Action Norm Sum)
           --------------------------------------------------
           [Safety Margins]
           6. Avg Min Sep     : {avg_min_sep:.3f} km (Global Safety Margin)
           7. Avg Warn Duration: {avg_warn:.2f} frames (< {self.cfg.reward_params['safe_dist']}km)
           --------------------------------------------------
           [Priority Filter]
           8. Avg Shielded Act: {avg_shield:.2f}% (Silenced Actions)
        {"=" * 65}
        """
        print(stats_output)
        
        # Save to log file
        with open(self.cfg.txt_path, "a", encoding="utf-8") as f:
            f.write(stats_output)
            f.write("\n")

    def _print_episode_stats(self):
        avg_te = np.mean(self.efficiency_history) if self.efficiency_history else 0.0
        avg_effort = np.mean(self.effort_history) if self.effort_history else 0.0
        print(
            f"\n📊 Episode {self.episode_count} Summary: Gen {self.num_ac_generated_total}/{self.curr_total} | Success {self.win_count} | TE {avg_te:.4f} | MinSep {self.safety_stats['min_sep_dist']:.3f}")

    def _print_status(self):
        print(
            f"Step {self.step_count}: AC {len(traf.id)} | Gen {self.num_ac_generated_total} | MinSep {self.safety_stats['min_sep_dist']:.2f}")

    def _init_log_file(self):
        if self.cfg.mode == 'eval' or (self.cfg.mode == 'train' and self.episode_count % 100 == 0):
            self.log_file_path = os.path.join(self.cfg.scn_log_dir, f"Ep{self.episode_count}.scn")
            shutil.copy2(self.cfg.scn_source, self.log_file_path)
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(f"\n# Episode {self.episode_count} Log\n")
        else:
            self.log_file_path = None

    def _write_scn_log(self, text):
        if self.log_file_path:
            t_str = time.strftime('%H:%M:%S.00', time.gmtime(bs.sim.simt))
            with open(self.log_file_path, 'a', encoding='utf-8') as f: f.write(f"{t_str}>{text}\n")

    def _update_agent(self, round=2):
        for _ in range(round):
            batch = self.replay_buffer.sample(self.cfg.batch_size)
            self.agent.update(
                {'ego': batch[0], 'neigh': batch[1], 'mask': batch[2], 'action': batch[3], 'reward': batch[4],
                 'next_ego': batch[5], 'next_neigh': batch[6], 'next_mask': batch[7], 'done': batch[8]})

    def save_global_stats_to_npy(self):
        """
        将global_stats字典保存为结构化numpy数组
        """
        # 创建结构化数据类型
        dtype = [
            ('episode', 'int32'),
            ('total_aircraft', 'int32'),
            ('success_count', 'int32'),
            ('collision_count', 'int32'),
            ('boundary_count', 'int32'),
            ('total_extra_dist_pct', 'float32'),
            ('success_sample_count', 'int32'),
            ('total_min_sep', 'float32'),
            ('total_warn_frames', 'int32'),
            ('total_cost', 'float32'),
            ('cost_sample_count', 'int32'),
            ('shield_pct', 'float32')
        ]

        # 获取数据长度（以最长的列表为准）
        n_episodes = len(self.global_stats['total_aircraft'])

        # 创建结构化数组
        structured_array = np.zeros(n_episodes, dtype=dtype)

        # 填充数据
        structured_array['episode'] = np.arange(n_episodes)
        structured_array['total_aircraft'] = np.array(self.global_stats['total_aircraft'], dtype='int32')
        structured_array['success_count'] = np.array(self.global_stats['success_count'], dtype='int32')
        structured_array['collision_count'] = np.array(self.global_stats['collision_count'], dtype='int32')
        structured_array['boundary_count'] = np.array(self.global_stats['boundary_count'], dtype='int32')
        structured_array['total_extra_dist_pct'] = np.array(self.global_stats['total_extra_dist_pct'], dtype='float32')
        structured_array['success_sample_count'] = np.array(self.global_stats['success_sample_count'], dtype='int32')
        structured_array['total_min_sep'] = np.array(self.global_stats['total_min_sep'], dtype='float32')
        structured_array['total_warn_frames'] = np.array(self.global_stats['total_warn_frames'], dtype='int32')
        structured_array['total_cost'] = np.array(self.global_stats['total_cost'], dtype='float32')
        structured_array['cost_sample_count'] = np.array(self.global_stats['cost_sample_count'], dtype='int32')
        structured_array['shield_pct'] = np.array(self.global_stats['shield_pct'], dtype='float32')

        # 保存到文件
        np.save(self.cfg.npy_path, structured_array)

        return structured_array


class MathUtils:
    @staticmethod
    def calculate_haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(
            dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    @staticmethod
    def calculate_bearing(lat1, lon1, lat2, lon2):
        lat1, lat2 = math.radians(lat1), math.radians(lat2)
        dlon = math.radians(lon2 - lon1)
        y = math.sin(dlon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        return (math.degrees(math.atan2(y, x)) + 360) % 360

    @staticmethod
    def calculate_distance_to_line(plat, plon, lat1, lon1, lat2, lon2):
        d13 = MathUtils.calculate_haversine_distance(lat1, lon1, plat, plon) / 6371.0
        brng13 = math.radians(MathUtils.calculate_bearing(lat1, lon1, plat, plon))
        brng12 = math.radians(MathUtils.calculate_bearing(lat1, lon1, lat2, lon2))
        return abs(math.asin(math.sin(d13) * math.sin(brng13 - brng12))) * 6371.0

    @staticmethod
    def clamp(val, min_v, max_v):
        return max(min_v, min(val, max_v))

    @staticmethod
    def calculate_cpa(p_rel_n, p_rel_e, v_rel_n, v_rel_e):
        v_rel_sq = v_rel_n ** 2 + v_rel_e ** 2
        if v_rel_sq < 1e-6: return 0.0, math.sqrt(p_rel_n ** 2 + p_rel_e ** 2)
        t_cpa = -(p_rel_n * v_rel_n + p_rel_e * v_rel_e) / v_rel_sq
        if t_cpa <= 0:
            d_cpa = math.sqrt(p_rel_n ** 2 + p_rel_e ** 2)
        else:
            p_cpa_n = p_rel_n + v_rel_n * t_cpa
            p_cpa_e = p_rel_e + v_rel_e * t_cpa
            d_cpa = math.sqrt(p_cpa_n ** 2 + p_cpa_e ** 2)
        return t_cpa, d_cpa


# Plugin Entry
controller = None


def init_plugin():
    global controller
    config = {'plugin_name': 'case_step_DDPG', 'plugin_type': 'sim', 'update_interval': 6.0,
              'update': update}
    controller = ACREnvironmentController()
    return config, {}


def update():
    if controller: controller.step()