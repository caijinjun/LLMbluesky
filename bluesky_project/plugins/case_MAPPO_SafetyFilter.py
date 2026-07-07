"""
MAPPO + Safety Filter 插件for BlueSky

基于 case_step_DDPG.py 修改,集成:
1. MAPPO策略 (Multi-Agent Proximal Policy Optimization)
2. Safety Filter (Hamilton-Jacobi Reachability-based)

两阶段训练:
 - Phase 1: Warmstart (use_safety_filter=False)
 - Phase 2: Fine-tune with Safety Filter (use_safety_filter=True)
"""
import sys
import os
# 添加onpolicy包路径
_plugin_dir = os.path.dirname(os.path.abspath(__file__))
_layered_dir = os.path.join(_plugin_dir, 'LayeredSafeMARL')
if _layered_dir not in sys.path:
    sys.path.insert(0, _layered_dir)

import shutil
import time
import math
import random
import numpy as np
import torch
import bluesky as bs
from bluesky import stack, traf, tools

# ==========================================
# 导入依赖
# ==========================================
from plugins.Multi_Agent.MAPPOPolicy import R_MAPPOPolicy
from plugins.LayeredSafeMARL.onpolicy.algorithms.mappo import R_MAPPO
from plugins.LayeredSafeMARL.onpolicy.utils.shared_buffer import SharedReplayBuffer
# from plugins.Multi_Agent.safety_filter import KinematicVehicleSafetyHandle, HjDataHandle  # 暂时禁用
from plugins.Multi_Agent.BlueSkyAdapter import BlueSkyMAPPOAdapter
from plugins.Multi_Agent.Normalizer_GAT10 import AircraftStateNormalizer

# ... (中间代码保持不变) ...


# from plugins.LayeredSafeMARL.multiagent.config import AirTaxiConfig  # 暂时禁用


# ==========================================
# 配置类
# ==========================================
class Config:
    def __init__(self):
        self.mode = 'train'  # 'train' or 'eval'
        self.use_safety_filter = False  # Phase1: False, Phase2: True
        
        # 路径配置
        self.base_path = r"D:\pythonprogram\Autonomous-ATC-N_Closest-master"
        self.output_path = os.path.join(self.base_path, "output", "result", "MAPPO_SafetyFilter")
        self.txt_path = os.path.join(self.output_path, "result.txt")
        self.npy_path = os.path.join(self.output_path, "result.npy")
        
        self.scn_source = os.path.join(self.base_path, "scenario", "DQN_3D.scn")
        self.route_file = "./routes/case_study_init.npy"
        
        os.makedirs(self.output_path, exist_ok=True)
        self.model_save_path = os.path.join(self.output_path, "MAPPO_Agent")
        self.scn_log_dir = os.path.join(self.output_path, f"{self.mode}_scn")
        os.makedirs(self.scn_log_dir, exist_ok=True)
        
        # 仿真参数
        self.dt = 6.0
        self.num_intruders = 5
        
        # 动作空间
        self.action_dim = 2  # [heading_delta, speed_delta]
        self.obs_dim = 30  # 9 (ego+xtk) + 3 * 7 (intruders)
        
        # 物理限制
        self.flight_levels = [alt for alt in range(20000, 21000, 300)]
        self.limits = {
            'speed': (200, 300),
            'alt': (20000, 21000)
        }
        
        # RL参数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = 0.99
        self.curriculum = [(float('inf'), 18, 2000, 50)]
        
        # MAPPO特定参数
        self.lr = 1e-4 # 降低学习率
        self.critic_lr = 1e-4
        self.opti_eps = 1e-5
        self.weight_decay = 0
        self.hidden_size = 64
        self.use_recurrent_policy = False
        
        # Safety Filter参数 (如果启用)
        self.hj_value_function_file = "plugins/HJ/airtaxi_value_function.pkl"  # HJ数据路径
        
        # 奖励权重 (与case_step_DDPG.py完全一致)
        # 奖励权重 (缩小100倍以适应PPO)
        self.reward_weights = {
            # 安全
            'collision': -10.0,
            'shield_intervention': -0.05,  # Safety Filter干预
            
            # 轨迹与效率
            'track_error': -0.1, # 增大偏航惩罚
            'track_sq_penalty': -0.01,
            'heading_error': -0.05, # 增大航向误差惩罚
            'progress': 0.01,
            'pos_progress': 0.005,
            
            # 稳定性
            'alt_hold': -0.01,
            'action_smooth': -0.005,
            
            # 状态与MoE
            'noop_bonus': 0.002,
            'expert_switch': -0.005,
            
            # 终止与边界
            'boundary': -5.0, # 大幅增大出界惩罚
            'goal_arrival': 2.0
        }
        
        self.reward_params = {
            'accept_xtk': 2.0,
            'accept_alt_err': 50.0,
            'arrival_dist': 5.0,
            'max_track_width': 20.0,
            'safe_dist': 10.0,
            'collision_h_km': 2.0,
            'collision_v_m': 300.0,
            'lookahead_time': 120.0,
            'min_sep_h': 9260.0,
            'min_sep_v': 1000.0
        }


# ==========================================
# 核心控制器
# ==========================================
class MAPPOSafetyFilterController:
    def __init__(self):
        self.cfg = Config()
        self.adapter = BlueSkyMAPPOAdapter(
            num_intruders=self.cfg.num_intruders,
            use_safety_filter=self.cfg.use_safety_filter
        )
        self.normalizer = AircraftStateNormalizer(num_intruders=self.cfg.num_intruders)
        
        # 初始化MAPPO策略
        self._init_mappo_policy()
        
        # 初始化Safety Filter (如果启用)
        if self.cfg.use_safety_filter:
            self._init_safety_filter()
        else:
            self.safety_filter = None
        
        # 环境状态
        self.route_assignments = {}
        self.assigned_levels = {}
        self.routes = np.load(self.cfg.route_file)
        self.spawn_choices = [60, 70, 80]
        
        self.episode_count = 1
        self.step_count = 0
        self.num_ac_generated_total = 0
        self.win_count = 0
        self.flight_stats = {}
        
        # 统计
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
            'cost_sample_count': []
        }
        self.episode_outcomes = []
        
        self.route_timers = [random.choice(self.spawn_choices) for _ in range(len(self.routes))]
        self.curr_max_conc, self.curr_max_steps, self.curr_total = self.cfg.curriculum[0][1:]
        
        print(f"\n=== MAPPO + Safety Filter Controller Initialized ===")
        print(f"Safety Filter: {'ENABLED' if self.cfg.use_safety_filter else 'DISABLED'}")
        print(f"Mode: {self.cfg.mode}")
        print(f"=== Episode {self.episode_count} Started ===\n")
    
    def _init_mappo_policy(self):
        """初始化MAPPO策略"""
        import argparse
        import gym
        
        # 创建参数对象
        args = argparse.Namespace()
        args.lr = 1e-4 # 降低学习率
        args.critic_lr = 1e-4
        args.opti_eps = self.cfg.opti_eps
        args.weight_decay = self.cfg.weight_decay
        args.hidden_size = self.cfg.hidden_size
        args.use_recurrent_policy = self.cfg.use_recurrent_policy
        args.recurrent_N = 1
        args.use_naive_recurrent_policy = False
        args.use_orthogonal = True
        args.gain = 0.01
        args.use_policy_active_masks = True # 开启Mask,忽略不存在的飞机
        args.use_feature_normalization = True
        args.use_ReLU = True
        args.stacked_frames = 1
        args.layer_N = 1
        args.use_popart = False
        args.use_valuenorm = True
        args.use_feature_popart = False
        
        # Trainer参数
        args.clip_param = 0.2
        args.ppo_epoch = 10 # 增加训练轮数
        args.num_mini_batch = 2 # 增加mini-batch
        args.data_chunk_length = 10
        args.value_loss_coef = 1
        args.entropy_coef = 0.01
        args.max_grad_norm = 10.0
        args.huber_delta = 10.0
        args.use_max_grad_norm = True
        args.use_clipped_value_loss = True
        args.use_huber_loss = True
        args.use_value_active_masks = True # 开启Mask
        args.episode_length = 2000 # 恢复正常的buffer容量,因为现在是并行存储
        args.n_rollout_threads = 1 # 单线程
        args.gamma = 0.99
        args.gae_lambda = 0.95
        args.use_gae = True
        args.use_proper_time_limits = False
        
        self.max_agents = 20  # 最大支持同时存在的飞机数
        
        # 定义观测和动作空间
        obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, 
                                    shape=(self.cfg.obs_dim,), dtype=np.float32)
        cent_obs_space = obs_space
        act_space = gym.spaces.Box(low=-1.0, high=1.0, 
                                    shape=(self.cfg.action_dim,), dtype=np.float32)
        
        # 1. 初始化Policy
        self.policy = R_MAPPOPolicy(
            args=args,
            obs_space=obs_space,
            cent_obs_space=cent_obs_space,
            act_space=act_space,
            device=self.cfg.device
        )
        
        # 2. 初始化Trainer
        self.trainer = R_MAPPO(args, self.policy, device=self.cfg.device)
        
        # 3. 初始化Buffer
        self.buffer = SharedReplayBuffer(
            args, 
            num_agents=self.max_agents, # 并行存储所有可能的飞机
            obs_space=obs_space,
            cent_obs_space=cent_obs_space,
            act_space=act_space
        )
        
        # RNN states
        self.rnn_states_actor = {}
        self.rnn_states_critic = {}
        
        # Agent索引管理
        self.agent_index_map = {} # acid -> index
        self.available_indices = list(range(self.max_agents))

        
    def _init_safety_filter(self):
        """初始化Safety Filter (暂时禁用)"""
        print("⚠️ Safety Filter disabled (需要安装jax等依赖)")
        self.safety_filter = None
        self.cfg.use_safety_filter = False
    
    def reset(self):
        """重置episode"""
        self.episode_count += 1
        if self.episode_count == 1000:
            stack.stack("STOP")
        
        self.step_count = 0
        self.num_ac_generated_total= 0
        self.win_count = 0
        self.route_assignments = {}
        self.assigned_levels = {}
        self.flight_stats = {}
        self.episode_outcomes = []
        self.rnn_states_actor = {}
        self.rnn_states_critic = {}
        
        # 重置索引管理
        self.agent_index_map = {}
        self.available_indices = list(range(self.max_agents))
        
        self.route_timers = [random.choice(self.spawn_choices) for _ in range(len(self.routes))]
        
        # TODO: MAPPO更新逻辑 (需要experience buffer)
        
        print(f"\n=== Episode {self.episode_count} ===")
        stack.stack('IC DQN_3D.scn')
        
        # 定期保存模型
        if self.cfg.mode == 'train' and self.episode_count % 20 == 0:
            self._save_model()
    
    def step(self):
        """主控制循环"""
        self.step_count += 1
        all_spawned = (self.num_ac_generated_total >= self.curr_total)
        all_cleared = (len(traf.id) == 0)
        
        if self.step_count >= self.curr_max_steps or (all_spawned and all_cleared and self.step_count > 10):
            self._conclude_episode()
            self.reset()
            return
        
        self._spawn_traffic()
        if len(traf.id) == 0:
            return
        
        # 获取邻居信息
        neighbor_map = self._get_neighbors_info_with_id()
        
        # === 主要决策循环 ===
        rewards = {}
        dones = {}
        infos = {}
        collision_info = {}
        
        # === 准备Batch数据容器 ===
        batch_obs = np.zeros((1, self.max_agents, self.cfg.obs_dim), dtype=np.float32)
        batch_cent_obs = np.zeros((1, self.max_agents, self.cfg.obs_dim), dtype=np.float32)
        batch_actions = np.zeros((1, self.max_agents, self.cfg.action_dim), dtype=np.float32)
        batch_log_probs = np.zeros((1, self.max_agents, self.cfg.action_dim), dtype=np.float32)
        batch_values = np.zeros((1, self.max_agents, 1), dtype=np.float32)
        batch_rewards = np.zeros((1, self.max_agents, 1), dtype=np.float32)
        batch_masks = np.ones((1, self.max_agents, 1), dtype=np.float32) # Default done=False (mask=1)
        batch_active_masks = np.zeros((1, self.max_agents, 1), dtype=np.float32) # Default inactive
        batch_rnn_actor = np.zeros((1, self.max_agents, 1, self.cfg.hidden_size), dtype=np.float32)
        batch_rnn_critic = np.zeros((1, self.max_agents, 1, self.cfg.hidden_size), dtype=np.float32)

        for acid in list(traf.id):
            if acid not in self.route_assignments:
                continue
            
            # 分配Agent Index
            if acid not in self.agent_index_map:
                if len(self.available_indices) > 0:
                    idx = self.available_indices.pop(0)
                    self.agent_index_map[acid] = idx
                else:
                    print(f"⚠️ Max agents reached, skipping {acid}")
                    continue
            
            idx = self.agent_index_map[acid]
            batch_active_masks[0, idx] = 1.0
            
            # 1. 获取状态
            obs, cent_obs = self.adapter.bluesky_to_mappo_state(
                acid, traf, self.route_assignments
            )
            
            # 初始化RNN states
            if acid not in self.rnn_states_actor:
                self.rnn_states_actor[acid] = np.zeros((1, 1, self.cfg.hidden_size))
                self.rnn_states_critic[acid] = np.zeros((1, 1, self.cfg.hidden_size))
            
            # 填入Batch (用于Buffer存储旧RNN状态)
            batch_obs[0, idx] = obs
            batch_cent_obs[0, idx] = cent_obs
            batch_rnn_actor[0, idx] = self.rnn_states_actor[acid][0]
            batch_rnn_critic[0, idx] = self.rnn_states_critic[acid][0]
            
            # 2. MAPPO决策
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.cfg.device)
            cent_obs_tensor = torch.FloatTensor(cent_obs).unsqueeze(0).to(self.cfg.device)
            rnn_actor = torch.FloatTensor(self.rnn_states_actor[acid]).to(self.cfg.device)
            rnn_critic = torch.FloatTensor(self.rnn_states_critic[acid]).to(self.cfg.device)
            masks = torch.ones(1, 1).to(self.cfg.device)
            
            values, action, action_log_probs, rnn_actor_new, rnn_critic_new = self.policy.get_actions(
                cent_obs_tensor, obs_tensor, rnn_actor, rnn_critic, masks,
                deterministic=(self.cfg.mode == 'eval')
            )
            
            action_np = action.cpu().detach().numpy()[0]
            values_np = values.cpu().detach().numpy()
            action_log_probs_np = action_log_probs.cpu().detach().numpy()
            
            self.rnn_states_actor[acid] = rnn_actor_new.cpu().detach().numpy()
            self.rnn_states_critic[acid] = rnn_critic_new.cpu().detach().numpy()
            
            # 填入Batch
            batch_actions[0, idx] = action_np
            batch_log_probs[0, idx] = action_log_probs_np
            batch_values[0, idx] = values_np
            
            # 3. Safety Filter (如果启用)
            shield_triggered = 0
            if self.cfg.use_safety_filter and self.safety_filter is not None:
                filtered_action = self._apply_safety_filter(acid, action_np)
                if not np.allclose(filtered_action, action_np, atol=1e-3):
                    shield_triggered = 1
                action_np = filtered_action
            
            # 4. 执行动作
            hdg_cmd, alt_cmd, spd_cmd = self.adapter.mappo_action_to_bluesky(
               acid, action_np, traf
            )
            stack.stack(f"HDG {acid} {hdg_cmd}")
            stack.stack(f"SPD {acid} {spd_cmd}")
            # stack.stack(f"ALT {acid} {alt_cmd}")  # 可选
            
            # 记录动作消耗
            if acid in self.flight_stats:
                self.flight_stats[acid]['fuel_proxy'] += np.linalg.norm(action_np)
            
            # 5. 碰撞检测
            collision_flag, min_dist_km = self._check_collision(acid)
            collision_info[acid] = {'min_dist_km': min_dist_km}
            
            # 6. 计算Reward
            r, d, i = self._compute_reward_phase2(acid, action_np, collision_flag, shield_triggered)
            rewards[acid] = r
            dones[acid] = d
            infos[acid] = i
            
            # 填入Batch
            batch_rewards[0, idx] = r
            if d:
                batch_masks[0, idx] = 0.0
            
        # 7. 存入Buffer (仅在训练模式)
        if self.cfg.mode == 'train':
            self.buffer.insert(
                share_obs=batch_cent_obs,
                obs=batch_obs,
                rnn_states_actor=batch_rnn_actor,
                rnn_states_critic=batch_rnn_critic,
                actions=batch_actions,
                action_log_probs=batch_log_probs,
                value_preds=batch_values,
                rewards=batch_rewards,
                masks=batch_masks,
                active_masks=batch_active_masks
            )
        
        # 更新飞行指标
        self._update_flight_metrics()
        
        # 更新安全统计
        self._update_safety_stats(collision_info)
        
        # 处理Done飞机
        for acid in list(dones.keys()):
            if dones.get(acid, False):
                self._remove_aircraft(acid, infos.get(acid, ""))
                # 清理RNN states
                if acid in self.rnn_states_actor:
                    del self.rnn_states_actor[acid]
                if acid in self.rnn_states_critic:
                    del self.rnn_states_critic[acid]
        
        if self.step_count % 100 == 0:
            print(f"Step {self.step_count}: AC {len(traf.id)} | Gen {self.num_ac_generated_total}")
    
    def _apply_safety_filter(self, acid, action_np):
        """应用Safety Filter"""
        # TODO: 实现完整的Safety Filter逻辑
        # 需要获取周围飞机状态,调用safety_filter.apply_safety_filter
        return action_np
    
    def _spawn_traffic(self):
        """生成飞机"""
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
                        self.route_timers[k] += 10
                        continue
                    self._create_aircraft(k)
                    self.route_timers[k] = self.step_count + random.choice(self.spawn_choices)
                    if self.num_ac_generated_total >= self.curr_total: break
    
    def _create_aircraft(self, route_idx):
        """创建单架飞机"""
        lat, lon, glat, glon, _ = self.routes[route_idx]
        acid = f"KL{self.num_ac_generated_total}"
        init_alt = random.choice(self.cfg.flight_levels if hasattr(self.cfg, 'flight_levels') else [20000, 20300, 20600, 20900])
        init_hdg = MathUtils.calculate_bearing(lat, lon, glat, glon)
        stack.stack(f'CRE {acid}, B737, {lat}, {lon}, {init_hdg}, {init_alt}, 250')
        stack.stack(f'ADDWPT {acid} {glat}, {glon}')
        stack.stack(f'VNAV {acid} ON')
        self.route_assignments[acid] = {'route_idx': route_idx, 'start': (lat, lon), 'target': (glat, glon)}
        self.assigned_levels[acid] = float(init_alt)
        self.num_ac_generated_total += 1
        ideal_dist = MathUtils.calculate_haversine_distance(lat, lon, glat, glon)
        self.flight_stats[acid] = {
            'dist_flown': 0.0, 'ideal_dist': ideal_dist, 'fuel_proxy': 0.0, 'prev_pos': (lat, lon),
            'min_sep': 50.0, 'warn_frames': 0, "ori": (lat, lon), "goal": (glat, glon), "counts": 0
        }
    
    def _conclude_episode(self):
        """Episode结束统计"""
        # 1. 处理在本回合结束时仍在飞行中的飞机 (视为超时/出界)
        active_acids = list(self.flight_stats.keys())
        for acid in active_acids:
            self._remove_aircraft(acid, "BOUNDARY")
        
        # 2. 将本回合 outcomes 汇总到全局统计 global_stats
        ep_total = len(self.episode_outcomes)
        if ep_total == 0: return
        
        self.global_stats['total_aircraft'].append(ep_total)
        
        warn_frames = 0.0
        min_sep = 0.0
        success_count = 0
        success_sample_count = 0
        extra_dist_pct = 0.0
        cost = 0.0
        cost_sample_count = 0
        collision_count = 0
        boundary_count = 0
        
        for outcome in self.episode_outcomes:
            res = outcome['result']
            
            # 安全指标累加 (所有飞机)
            warn_frames += outcome['warn_frames']
            min_sep += outcome['min_sep']
            
            # 分类统计
            if res == 'GOAL':
                success_count += 1
                success_sample_count += 1
                # 效率指标 (仅统计成功飞机)
                extra_dist_pct += outcome['extra_dist_pct']
                # 操作消耗
                cost += outcome['cost']
                cost_sample_count += 1
            elif res == 'COLLISION':
                collision_count += 1
            else:  # BOUNDARY or TIMEOUT
                boundary_count += 1
        
        self.global_stats['total_warn_frames'].append(warn_frames)
        self.global_stats['total_min_sep'].append(min_sep)
        self.global_stats['success_count'].append(success_count)
        self.global_stats['success_sample_count'].append(success_sample_count)
        self.global_stats['total_extra_dist_pct'].append(extra_dist_pct)
        self.global_stats['total_cost'].append(cost)
        self.global_stats['cost_sample_count'].append(cost_sample_count)
        self.global_stats['collision_count'].append(collision_count)
        self.global_stats['boundary_count'].append(boundary_count)
        
        # 3. 计算累计平均指标
        total_ac = np.sum(self.global_stats['total_aircraft'][-100:])
        success_ac = np.sum(self.global_stats['success_sample_count'][-100:])
        
        # A. 基础三率
        avg_success = (np.sum(self.global_stats['success_count'][-100:]) / total_ac) * 100 if total_ac > 0 else 0
        avg_col = (np.sum(self.global_stats['collision_count'][-100:]) / total_ac) * 100 if total_ac > 0 else 0
        avg_bound = (np.sum(self.global_stats['boundary_count'][-100:]) / total_ac) * 100 if total_ac > 0 else 0
        
        # B. 效率指标 (Extra Distance %)
        avg_extra_dist = 0.0
        if success_ac > 0:
            avg_extra_dist = (np.sum(self.global_stats['total_extra_dist_pct'][-100:]) / success_ac)
        
        # C. 安全指标 (Avg Min Sep)
        avg_min_sep = np.sum(self.global_stats['total_min_sep'][-100:]) / total_ac if total_ac > 0 else 0
        
        # D. 警告时间 (Warn Frames)
        avg_warn = np.sum(self.global_stats['total_warn_frames'][-100:]) / total_ac if total_ac > 0 else 0
        
        # E. 操作消耗
        avg_cost = 0.0
        cost_samples = np.sum(self.global_stats['cost_sample_count'][-100:])
        if cost_samples > 0:
            avg_cost = np.sum(self.global_stats['total_cost'][-100:]) / cost_samples
        
        # === PPO Update (仅在训练模式) ===
        train_infos = {}
        if self.cfg.mode == 'train' and self.buffer.step > 0:
            print("🔄 Updating Policy...")
            self.trainer.prep_training()
            
            # 计算Returns (假设最后一步value为0)
            next_value = self.buffer.value_preds[self.buffer.step]
            self.buffer.compute_returns(next_value, self.trainer.value_normalizer)
            
            # 执行训练
            train_infos = self.trainer.train(self.buffer)
            
            # 重置Buffer
            self.buffer.after_update()
            self.buffer.step = 0 # 强制重置step索引
            
            print(f"📈 Loss: Policy={train_infos['policy_loss']:.4f} | Value={train_infos['value_loss']:.4f}")

        # 构建打印信息的字符串
        stats_output = f"""
        {"=" * 65}
        📊 CUMULATIVE STATS (Episodes {max(0, self.episode_count - 99)}-{self.episode_count})
           Total Aircraft Gen : {total_ac}
           --------------------------------------------------
           [Performance]
           1. Success Rate    : {avg_success:.2f}%
           2. Collision Rate  : {avg_col:.2f}%
           3. Boundary Rate   : {avg_bound:.2f}%
           --------------------------------------------------
           [Efficiency & Cost]
           4. Extra Dist %    : {avg_extra_dist:.2f}% (Success Flights Only)
           5. Avg Op. Cost    : {avg_cost:.4f} (Action Norm Sum)
           --------------------------------------------------
           [Safety Margins]
           6. Avg Min Sep     : {avg_min_sep:.3f} km (Global Safety Margin)
           7. Avg Warn Duration: {avg_warn:.2f} frames (< {self.cfg.reward_params['safe_dist']}km)
           --------------------------------------------------
           [Training]
           8. Policy Loss     : {train_infos.get('policy_loss', 0.0):.4f}
           9. Value Loss      : {train_infos.get('value_loss', 0.0):.4f}
        {"=" * 65}
        """
        print(stats_output)
        
        # 保存数据
        self.save_global_stats_to_npy()
        
        # 写入日志文件
        with open(self.cfg.txt_path, "a", encoding="utf-8") as f:
            f.write(stats_output)
            f.write("\n")
    
    def _remove_aircraft(self, acid, reason):
        """移除飞机并记录统计"""
        stack.stack(f'DEL {acid}')
        if acid in self.flight_stats:
            # 释放Agent Index
            if acid in self.agent_index_map:
                idx = self.agent_index_map[acid]
                self.available_indices.append(idx)
                self.available_indices.sort() # 保持有序
                del self.agent_index_map[acid]

            stats = self.flight_stats[acid]
            extra_dist_pct = 0.0
            if reason == 'GOAL':
                extra_dist_pct = max(0.0, (stats['dist_flown'] - stats['ideal_dist']) / stats['ideal_dist'] * 100)
            
            self.episode_outcomes.append({
                'acid': acid,
                'result': reason,
                'cost': stats['fuel_proxy'],
                'warn_frames': stats['warn_frames'],
                'min_sep': stats['min_sep'],
                'extra_dist_pct': extra_dist_pct
            })
            
            if reason == "GOAL":
                print(f"✈️ {acid} Arrived.")
            elif reason == "COLLISION":
                print(f"💥 {acid} Collided!")
            elif reason == "BOUNDARY":
                print(f"❌ {acid} Out of Bound!")
            del self.flight_stats[acid]
    
    def _get_neighbors_info_with_id(self):
        """获取邻居信息(与DDPG一致)"""
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
            for _, k in top_n:
                n_data.append([lats[k], lons[k], spds[k] * 1.9439, alts[k] * 3.28084, hdgs[k], traf.id[k]])
            neighbor_map[acid_i] = n_data
        return neighbor_map
    
    def _update_flight_metrics(self):
        """更新飞机飞行指标"""
        for i, acid in enumerate(traf.id):
            if acid in self.flight_stats:
                curr_lat, curr_lon = traf.lat[i], traf.lon[i]
                prev_lat, prev_lon = self.flight_stats[acid]['prev_pos']
                self.flight_stats[acid]['dist_flown'] += MathUtils.calculate_haversine_distance(
                    prev_lat, prev_lon, curr_lat, curr_lon
                )
                self.flight_stats[acid]['prev_pos'] = (curr_lat, curr_lon)
                self.flight_stats[acid]['counts'] += 1
    
    def _update_safety_stats(self, collision_info):
        """更新安全统计"""
        for acid, info in collision_info.items():
            min_dist = info.get('min_dist_km', 50.0)
            if acid in self.flight_stats:
                self.flight_stats[acid]['min_sep'] = min(self.flight_stats[acid]['min_sep'], min_dist)
                if min_dist < self.cfg.reward_params['safe_dist']:
                    self.flight_stats[acid]['warn_frames'] += 1
    
    def _check_collision(self, acid):
        """检查碰撞(圆柱体检测)"""
        idx = traf.id2idx(acid)
        my_lat, my_lon = traf.lat[idx], traf.lon[idx]
        my_alt = traf.alt[idx]
        
        collision_flag = 0.0
        min_dist_km = 50.0
        
        for i, other_acid in enumerate(traf.id):
            if other_acid == acid:
                continue
            other_lat, other_lon = traf.lat[i], traf.lon[i]
            other_alt = traf.alt[i]
            
            h_dist = MathUtils.calculate_haversine_distance(my_lat, my_lon, other_lat, other_lon)
            v_dist = abs(my_alt - other_alt)
            
            if h_dist < min_dist_km:
                min_dist_km = h_dist
            
            # 碰撞判定: H < 2km AND V < 300m
            if h_dist < self.cfg.reward_params['collision_h_km'] and v_dist < self.cfg.reward_params['collision_v_m']:
                collision_flag = 1.0
                break
        
        return collision_flag, min_dist_km
    
    def _compute_reward_phase2(self, acid, action, collision_flag, shield_triggered):
        """计算reward(与DDPG完全一致)"""
        w = self.cfg.reward_weights
        p = self.cfg.reward_params
        idx = traf.id2idx(acid)
        route_info = self.route_assignments.get(acid, {})
        route_idx = route_info.get('route_idx', 0)
        route = self.routes[route_idx]
        
        last_lat, last_lon = self.flight_stats[acid]['prev_pos']
        lat, lon = traf.lat[idx], traf.lon[idx]
        target_alt = self.assigned_levels.get(acid, 20000.0)
        curr_alt_ft = traf.alt[idx] * 3.28084
        
        r, done, info = 0.0, False, ""
        
        # 1. 干预惩罚
        if shield_triggered:
            r += w['shield_intervention']
        
        # 2. 碰撞惩罚
        if collision_flag > 0.5:
            r += w['collision']
            info = "COLLISION"
            done = True
        
        # 3. 轨迹跟踪
        # 使用Haversine公式计算真实的偏航距离(km)
        xtk = MathUtils.calculate_cross_track_distance(lat, lon, route[0], route[1], route[2], route[3])
        r += w['track_error'] * abs(xtk) # 使用绝对值
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
        r += w['action_smooth'] * np.mean(np.abs(action))
        
        # Termination
        dist_to_goal = MathUtils.calculate_haversine_distance(lat, lon, route[2], route[3])
        if xtk > p['max_track_width']:
            r += w['boundary']
            done = True
            info = "BOUNDARY"
        elif dist_to_goal < p['arrival_dist']:
            r += w['goal_arrival']
            done = True
            info = "GOAL"
            self.win_count += 1
        
        r = max(-500.0, min(r, 200.0))
        return r, done, info
    
    def _save_model(self):
        """保存MAPPO模型"""
        save_path = self.cfg.model_save_path
        torch.save(self.policy.actor.state_dict(), f"{save_path}_actor.pth")
        torch.save(self.policy.critic.state_dict(), f"{save_path}_critic.pth")
        print(f"💾 Model saved: Episode {self.episode_count}")

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
            ('cost_sample_count', 'int32')
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

        # 保存到文件
        np.save(self.cfg.npy_path, structured_array)

        return structured_array

# ==========================================
# BlueSky插件接口
# ==========================================
controller = None


def init_plugin():
    global controller
    config = {
        'plugin_name': 'case_MAPPO_SafetyFilter',
        'plugin_type': 'sim',
        'update_interval': 6.0,
        'update': update
    }
    controller = MAPPOSafetyFilterController()
    return config, {}


def update():
    if controller:
        controller.step()


# ==========================================
# 工具类
# ==========================================
class MathUtils:
    @staticmethod
    def calculate_haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    @staticmethod
    def calculate_bearing(lat1, lon1, lat2, lon2):
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dlambda = math.radians(lon2 - lon1)
        x = math.sin(dlambda) * math.cos(phi2)
        y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
        return (math.degrees(math.atan2(x, y)) + 360) % 360

    @staticmethod
    def calculate_distance_to_line(plat, plon, lat1, lon1, lat2, lon2):
        d = MathUtils.calculate_haversine_distance(lat1, lon1, lat2, lon2)
        dp = MathUtils.calculate_haversine_distance(lat1, lon1, plat, plon)
        brg1 = math.radians(MathUtils.calculate_bearing(lat1, lon1, lat2, lon2))
        brg2 = math.radians(MathUtils.calculate_bearing(lat1, lon1, plat, plon))
        return abs(dp * math.sin(brg2 - brg1))

    @staticmethod
    def calculate_cross_track_distance(plat, plon, lat1, lon1, lat2, lon2):
        """计算偏航距离 (Cross Track Distance)"""
        return MathUtils.calculate_distance_to_line(plat, plon, lat1, lon1, lat2, lon2)
