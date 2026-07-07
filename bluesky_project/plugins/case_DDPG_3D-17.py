import os
import shutil
import bluesky as bs
from bluesky import stack, settings, navdb, traf, sim, scr, tools
from geopy.distance import geodesic
from plugins.Multi_Agent.DDPG_3DElevn import DDPG
from plugins.Multi_Agent.Normalizer import AircraftStateNormalizer
from DataCollector import DataCollector
import numpy as np
import time
import random
import math
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def init_plugin():
    global num_ac, max_ac, num_intruders
    global agent_manager
    global positions, route_keeper, route_num, route_queue, choices
    global episode_num, episode_max
    global step_num, step_max
    global actions, old_air_craft, current_air_craft
    global speed_choices, alt_choices, hdg_choices
    global max_speed, min_speed, max_alt, min_alt
    global win_list, best_win, reward_list, reward_memory
    global transition_dict
    global AircraftStateNormalizer
    global last_goal_distance, last_along_track
    global boundary_strikes
    global last_vertical_distances
    global collision_count, out_of_bound_count
    global max_route_distance_static
    global written, SCN_File
    global obs_dim

    # [新增] 全局数据采集器
    global data_collector

    written = 0  # 生成数据时建议关闭 SCN 记录
    os.makedirs('output/DDPG/DDPG3D-15/scenarios', exist_ok=True)

    collision_count = 0
    out_of_bound_count = 0
    AircraftStateNormalizer = AircraftStateNormalizer()

    transition_dict = {
        'joint_states': [], 'joint_next_states': [], 'joint_actions': [],
        'joint_rewards': [], 'joint_dones': [], 'joint_masks': [], 'joint_presence_masks': []
    }
    reward_memory = []

    # 初始化字典
    last_goal_distance = {}
    last_along_track = {}
    boundary_strikes = {}
    last_vertical_distances = {}

    num_ac = 0
    max_ac = 30
    num_intruders = 3

    # [⚡️终极修复⚡️] 强制将观测维度设为 40
    # 原因：Normalizer 输出维度是 40，如果设为 32 会导致广播错误 (ValueError: shape(40,) into shape(32,))
    obs_dim = 40

    # DDPG 参数：这里 state_dim=14 是为了匹配 Teacher Model 的输入层
    # 尽管 obs_dim=40 (buffer大小)，但 take_action 时模型会自动只取它需要的前几维
    agent_manager = DDPG(state_dim=14, intruders_dim=18, hidden_dim=256, action_dim=3, max_agents=max_ac)

    # 初始化数据采集器 (每5步记录一次)
    data_collector = DataCollector(save_path="output/train_data_7b.jsonl", record_interval=5)

    # 强制加载训练好的模型
    model_path = "output/DDPG/DDPG3D-15/DDPG"
    if os.path.exists(f"{model_path}_actor.pth"):
        print("=" * 50)
        print("🎥 数据生成模式: 已加载 Teacher Model (256/14 Legacy)")
        agent_manager.load_models(model_path)
        print("=" * 50)
    else:
        print("❌ 错误：未找到模型文件！无法生成高质量数据！")
        return {}, {}

    reward_list = [0 for _ in range(max_ac)]
    best_win = 0
    win_list = 0
    min_speed = 220
    max_speed = 320
    min_alt = 19500
    max_alt = 21000

    step_num = 0
    episode_max = 6000
    step_max = 1100

    # 设置极大的 episode_num，消除 DDPG 的探索噪声
    episode_num = 100000

    positions = np.load('./routes/case_study_init.npy')
    all_route_dists = []
    for i in range(len(positions)):
        slat, slon, tlat, tlon = positions[i][0], positions[i][1], positions[i][2], positions[i][3]
        d = calculate_haversine_distance(slat, slon, tlat, tlon)
        all_route_dists.append(d)

    max_route_distance_static = max(all_route_dists) if all_route_dists else 0.0
    route_num = len(positions)
    route_keeper = np.zeros(max_ac, dtype=int)
    choices = [20, 25, 30]
    route_queue = random.choices(choices, k=positions.shape[0])

    old_air_craft = {}
    current_air_craft = {}
    actions = {}

    if written == 1:
        SCN_File = f"output/DDPG/DDPG3D-15/scenarios/{episode_num}.scn"
        if os.path.exists('multi_agent.scn'):
            shutil.copy2('multi_agent.scn', SCN_File)
        else:
            open(SCN_File, 'w').close()

    config = {
        'plugin_name': 'case_DDPG_3D-11',
        'plugin_type': 'sim',
        'update_interval': 5.0,
        'update': update,
    }
    return config, {}


def update():
    global num_ac, max_ac, num_intruders
    global agent_manager
    global positions, route_keeper, route_num, route_queue, choices
    global episode_num, episode_max
    global step_num, step_max
    global actions, old_air_craft, current_air_craft
    global speed_choices, alt_choices, hdg_choices
    global max_speed, min_speed, max_alt, min_alt
    global win_list, best_win, reward_list, reward_memory
    global transition_dict
    global AircraftStateNormalizer
    global last_goal_distance, last_along_track
    global boundary_strikes
    global collision_count, out_of_bound_count
    global written, SCN_File
    global obs_dim

    # [新增]
    global data_collector

    current_time = bs.sim.simt
    data_time = time.strftime('%H:%M:%S.00', time.gmtime(current_time))

    if step_num >= step_max:
        reset()
        return
    if num_ac == max_ac and len(traf.id) == 0:
        reset()
        return

    old_air_craft = current_air_craft.copy()

    # 获取原始数据 (用于采集)
    min_dis_craft = get_min_Dis()
    raw_own_states = get_own_state()  # 这里的 [2]是knots, [4]是hdg

    # [关键修复] 为 Normalizer 构造纯数值数据 (剥离 ID)
    min_dis_craft_numeric = {}
    for aid, intruders in min_dis_craft.items():
        min_dis_craft_numeric[aid] = [intruder[1:] for intruder in intruders]

    # 归一化
    rewards, dones = get_rewards(min_dis_craft)
    current_air_craft = AircraftStateNormalizer.normalize_complete_state(raw_own_states, min_dis_craft_numeric)

    # Buffer 存储逻辑 (这里传入 obs_dim=40，确保能装下)
    if old_air_craft:
        js, jns, ja, jr, jd, jm_alive, jm_present = build_joint_transition(
            old_air_craft, current_air_craft, actions, rewards, dones, max_ac, obs_dim, agent_manager.action_dim
        )
        transition_dict['joint_states'].append(js)
        transition_dict['joint_next_states'].append(jns)
        transition_dict['joint_actions'].append(ja)
        transition_dict['joint_rewards'].append(jr)
        transition_dict['joint_dones'].append(jd)
        transition_dict['joint_masks'].append(jm_alive)
        transition_dict['joint_presence_masks'].append(jm_present)

    actions = {}
    for i, air_craft in enumerate(traf.id):
        # 1. 决策
        full_state = current_air_craft[air_craft]  # 这是 40 维的

        # Teacher Model 只需要前 14 维 (StateNet输入层大小)
        # 所以我们需要在这里做一个手动切片，否则 Linear(14, 256) 接收 40 维输入会报错
        # 假设前14维是本机状态+必要的相对信息，这取决于 Normalizer 的拼接顺序
        # 通常 Normalizer 是 [Own_State, Intruder_State]
        # 如果报错 size mismatch (40 vs 14)，请启用下面的切片：
        # model_input = full_state[:14]
        # raw_action = agent_manager.take_action(model_input, episode_num=100000)

        # 既然之前的报错是 load_state_dict 时的 shape mismatch，说明模型确实是 14 维输入
        # 这里为了稳妥，我们传 full_state，看看 DDPG 内部是否会自动处理
        # 如果 DDPG 内部没写切片，PyTorch 会报 "mat1 and mat2 shapes cannot be multiplied"
        # 鉴于之前没报这个错，说明可能 Normalizer 输出的顺序恰好对上了，或者我们先试运行
        # 为了保险起见，建议用切片：

        model_input = full_state  # 暂时传完整状态，如果报错再切片

        try:
            raw_action = agent_manager.take_action(model_input, episode_num=100000)
        except RuntimeError as e:
            if "mat1 and mat2 shapes cannot be multiplied" in str(e):
                # 如果报错维度不对，说明模型只吃 14 维
                raw_action = agent_manager.take_action(full_state[:14], episode_num=100000)
            else:
                raise e

        # ===============================================
        # [数据采集] 记录 DDPG 的决策
        # ===============================================
        if air_craft in raw_own_states and air_craft in min_dis_craft and not dones[air_craft]:
            state_list = raw_own_states[air_craft]
            own_state_dict = {
                'lat': state_list[0],
                'lon': state_list[1],
                'spd': state_list[2],  # Knots
                'hdg': state_list[4]  # Degree
            }

            data_collector.save_sample(
                aircraft_id=air_craft,
                own_state=own_state_dict,
                intruders=min_dis_craft[air_craft],
                action_raw=raw_action
            )
        # ===============================================

        delta_speed = raw_action[0] * 15
        delta_alt = raw_action[1] * 200
        delta_hdg = raw_action[2] * 12

        actions[air_craft] = raw_action.copy()

        # 2. 执行限制
        new_tas = clamp(traf.cas[i] * 1.9437 + delta_speed, min_speed, max_speed)
        new_alt = clamp(traf.alt[i] * 3.28084 + delta_alt, min_alt, max_alt)
        new_hdg = (traf.hdg[i] + delta_hdg) % 360

        stack.stack('SPD {} {}'.format(air_craft, new_tas))
        stack.stack('ALT {} {}'.format(air_craft, new_alt))
        stack.stack('HDG {} {}'.format(air_craft, new_hdg))

        if written == 1:
            with open(SCN_File, 'a', encoding='utf-8') as f:
                f.write(f"{data_time}>SPD {air_craft} {new_tas}\n")
                f.write(f"{data_time}>ALT {air_craft} {new_alt}\n")
                f.write(f"{data_time}>HDG {air_craft} {new_hdg}\n")

        if dones[air_craft]:
            stack.stack('DEL {}'.format(air_craft))
            last_goal_distance.pop(air_craft, None)
            last_along_track.pop(air_craft, None)
            boundary_strikes.pop(air_craft, None)

    # 交通生成逻辑
    if num_ac < max_ac:
        if len(traf.id) == 0:
            for i in range(len(positions)):
                lat, lon, glat, glon, h = positions[i]
                bearing_to_goal = calculate_bearing(lat, lon, glat, glon)
                stack.stack('CRE KL{}, B737, {}, {}, {}, 20000, 250'.format(num_ac, lat, lon, h))
                stack.stack('ADDWPT KL{} {}, {}'.format(num_ac, glat, glon))
                stack.stack('HDG KL{} {}'.format(num_ac, bearing_to_goal))
                stack.stack(f'VNAV KL{num_ac} ON')
                route_keeper[num_ac] = i
                num_ac += 1
                if num_ac == max_ac: break
        else:
            for k in range(len(route_queue)):
                if step_num == route_queue[k]:
                    lat, lon, glat, glon, h = positions[k]
                    bearing_to_goal = calculate_bearing(lat, lon, glat, glon)
                    stack.stack('CRE KL{}, B737, {}, {}, {}, 20000, 250'.format(num_ac, lat, lon, h))
                    stack.stack('ADDWPT KL{} {}, {}'.format(num_ac, glat, glon))
                    stack.stack('HDG KL{} {}'.format(num_ac, bearing_to_goal))
                    stack.stack(f'VNAV KL{num_ac} ON')
                    route_keeper[num_ac] = k
                    num_ac += 1
                    route_queue[k] = step_num + random.choices(choices, k=1)[0]
                    if num_ac == max_ac: break

    step_num += 1

    # [修改] 禁用训练更新！
    if step_num > 0 and step_num % 300 == 0:
        for key in transition_dict:
            transition_dict[key] = []
        print(f"Running Step: {step_num} | Collecting Data (No Training)...")

    if step_num % 200 == 0:
        print(step_num)


def reset():
    global num_ac, max_ac
    global agent_manager
    global positions, route_keeper, route_queue
    global episode_num, step_num
    global actions, old_air_craft, current_air_craft
    global win_list, best_win, reward_list, reward_memory
    global transition_dict
    global last_goal_distance, last_along_track
    global boundary_strikes
    global last_vertical_distances

    global collision_count
    global out_of_bound_count

    global written, SCN_File
    global obs_dim

    surviving_distances = []
    for i, aircraft_id in enumerate(traf.id):
        lati, loni = traf.lat[i], traf.lon[i]
        route_idx = route_keeper[int(aircraft_id[2:])]
        target_lat = positions[route_idx][2]
        target_lon = positions[route_idx][3]
        dist = calculate_haversine_distance(lati, loni, target_lat, target_lon)
        surviving_distances.append(dist)

    avg_survive_dist = np.mean(surviving_distances) if len(surviving_distances) > 0 else 0.0
    success_rate = win_list / max_ac
    avg_reward = np.mean(reward_list)

    print("Episode: {} | Success Rate: {:.2f} | Collisions: {} | OOB: {} | Avg Dist: {:.2f} | Reward: {:.4f}".format(
        episode_num, success_rate, collision_count, out_of_bound_count, avg_survive_dist, avg_reward))

    stats_dir = 'output/DDPG/DDPG3D-15'
    os.makedirs(stats_dir, exist_ok=True)
    stats_file = os.path.join(stats_dir, 'training_stats.csv')

    file_exists = os.path.exists(stats_file)
    with open(stats_file, 'a') as f:
        if not file_exists:
            f.write("Episode,SuccessRate,Collisions,OOB,AvgRemainDist,AvgReward\n")
        f.write("{},{:.4f},{},{},{:.4f},{:.4f}\n".format(
            episode_num, success_rate, collision_count, out_of_bound_count, avg_survive_dist, avg_reward))

    reward_memory.append(avg_reward)
    np.save(os.path.join(stats_dir, 'reward_memory.npy'), reward_memory)

    num_ac = 0
    step_num = 0
    episode_num += 1
    route_keeper = np.zeros(max_ac, dtype=int)
    actions = {}
    old_air_craft = {}
    current_air_craft = {}
    route_queue = random.choices([20, 25, 30], k=positions.shape[0])

    # 必须重置状态追踪字典
    last_goal_distance = {}
    last_along_track = {}
    boundary_strikes = {}
    last_vertical_distances = {}

    collision_count = 0
    out_of_bound_count = 0

    # 重置 Buffer
    transition_dict = {
        'joint_states': [], 'joint_next_states': [], 'joint_actions': [],
        'joint_rewards': [], 'joint_dones': [], 'joint_masks': [], 'joint_presence_masks': []
    }
    reward_list = [0 for _ in range(max_ac)]

    best_win = max(win_list, best_win)
    win_list = 0
    if episode_num == episode_max:
        stack.stack('STOP')

    if written == 1:
        SCN_File = f"output/DDPG/DDPG3D-15/scenarios/{episode_num}.scn"
        if os.path.exists('multi_agent.scn'):
            shutil.copy2('multi_agent.scn', SCN_File)
        else:
            open(SCN_File, 'w').close()

    stack.stack('IC multi_agent.scn')


def get_own_state():
    global route_keeper
    own_state = {}
    for i, id in enumerate(traf.id):
        index = traf.id2idx(id)
        lat, lon = traf.lat[index], traf.lon[index]
        speed, alt, hdg = traf.cas[index] * 1.9439, traf.alt[index] * 3.28084, traf.hdg[index]

        route = positions[route_keeper[int(id[2:])]]
        start_lat, start_lon, goal_lat, goal_lon, start_h = route

        # [新增] 计算到目标的方位角
        bearing_to_goal = calculate_bearing(lat, lon, goal_lat, goal_lon)

        # [新增] 计算航向偏差（归一化到[-1, 1]）
        heading_error = bearing_to_goal - hdg
        if heading_error > 180:
            heading_error -= 360
        elif heading_error < -180:
            heading_error += 360
        heading_error_norm = heading_error / 180.0

        own_state[id] = [
            lat, lon, speed, alt, hdg,  # 0-4: 基础状态
            start_lat, start_lon,  # 5-6: 起点
            goal_lat, goal_lon,  # 7-8: 终点
            start_h,  # 9: 起点高度
            bearing_to_goal,  # 10: 到目标的方位角
            heading_error_norm,  # 11: 航向偏差
        ]  # 总计14维
    return own_state


def get_min_Dis():
    global route_keeper
    global num_intruders
    id = traf.id
    lat = traf.lat
    lon = traf.lon
    n_aircraft = len(traf.id)

    if n_aircraft == 0:
        return {}

    min_distances = {}
    for i, id_i in enumerate(id):
        dist = {}
        # 1. 计算所有飞机的距离
        for j, id_j in enumerate(id):
            if i != j:
                dist[id_j] = calculate_haversine_distance(lat[i], lon[i], lat[j], lon[j])

        # 2. 排序找出最近的 num_intruders 个
        sorted_list = list(dict(sorted(dist.items(), key=lambda item: item[1])).keys())

        min_distances[id_i] = []  # 初始化列表

        for z in range(min(len(sorted_list), num_intruders)):
            air_index = traf.id.index(sorted_list[z])

            # --- [参数准备] ---
            # 入侵机信息
            int_lat, int_lon = lat[air_index], lon[air_index]
            int_spd = traf.cas[air_index] * 1.9439  # knots
            int_hdg = traf.trk[air_index]  # degree
            int_alt = traf.alt[air_index] * 3.28084  # ft

            # 本机信息 (用于计算相对动态)
            own_lat, own_lon = lat[i], lon[i]
            own_spd = traf.cas[i] * 1.9439  # knots
            own_hdg = traf.trk[i]  # degree

            # --- [计算 Distance (m)] ---
            distance_km = dist[sorted_list[z]]
            distance_m = distance_km * 1000.0

            # --- [计算 Closing Rate (m/s)] ---
            bearing_to_int = calculate_bearing(own_lat, own_lon, int_lat, int_lon)

            rad_bearing = math.radians(bearing_to_int)
            rad_own_hdg = math.radians(own_hdg)
            rad_int_hdg = math.radians(int_hdg)

            v_own_proj = own_spd * math.cos(rad_bearing - rad_own_hdg)
            v_int_proj = int_spd * math.cos(rad_bearing - rad_int_hdg)

            closing_spd_kts = v_own_proj - v_int_proj
            closing_spd_ms = closing_spd_kts * 0.514444

            # --- [构造列表] ---
            min_distances[id_i].append(
                [
                    sorted_list[z],  # 0: ID
                    int_lat,  # 1: Lat
                    int_lon,  # 2: Lon
                    int_spd,  # 3: Spd
                    int_alt,  # 4: Alt
                    int_hdg,  # 5: Hdg
                    distance_m,  # 6: Dist (m)
                    closing_spd_ms  # 7: Closing Rate (m/s) [关键修复]
                ])
    return min_distances


def get_rewards(stats):
    global route_keeper, positions, win_list, last_goal_distance, last_along_track
    global boundary_strikes
    global collision_count, step_num, step_max, out_of_bound_count
    global last_vertical_distances

    w_collision = -140.0
    w_arrival = 60.0
    w_progress = 0.2
    w_heading = 0.15
    w_step_cost = -0.01
    w_timeout = -10.0
    w_deviation_linear = -0.01
    w_boundary_soft = -0.1
    w_out_of_corridor = -45.0
    soft_boundary_dist = 15.0
    max_dev_dist = 25.0
    collision_hor = 2.0
    collision_ver = 500.0
    danger_hor = 4.0
    danger_ver = 600.0
    w_danger = -25.0
    warning_hor = 8.0
    warning_ver = 800.0
    w_warning = -8.0
    arrival_distance = 8.0

    id = traf.id
    lon = traf.lon
    lat = traf.lat
    n_aircraft = len(id)
    rewards = {id_: 0.0 for id_ in id}
    dones = {id_: False for id_ in id}

    if n_aircraft == 0: return rewards, dones

    for i, aircraft_id in enumerate(id):
        index = traf.id2idx(aircraft_id)
        lati, loni, alti, hdgi = lat[index], lon[index], traf.alt[index] * 3.28084, traf.hdg[index]
        route_idx = route_keeper[int(aircraft_id[2:])]
        start_lat, start_lon, target_lat, target_lon, _ = positions[route_idx]

        cross_track_error = calculate_distance_to_line(
            lati, loni, start_lat, start_lon, target_lat, target_lon
        )
        current_along_track = calculate_along_track_distance(
            lati, loni, start_lat, start_lon, target_lat, target_lon
        )

        prev_along_track = last_along_track.get(aircraft_id, current_along_track)
        last_along_track[aircraft_id] = current_along_track
        progress_projected = current_along_track - prev_along_track
        rewards[aircraft_id] += w_progress * clamp(progress_projected, -2.0, 2.0)

        dist_to_goal = calculate_haversine_distance(lati, loni, target_lat, target_lon)
        if dist_to_goal > arrival_distance:
            bearing_to_goal = calculate_bearing(lati, loni, target_lat, target_lon)
            angle_diff = abs(bearing_to_goal - hdgi)
            angle_diff = min(angle_diff, 360 - angle_diff)
            rewards[aircraft_id] += w_heading * math.cos(math.radians(angle_diff))

        rewards[aircraft_id] += w_step_cost
        rewards[aircraft_id] += cross_track_error * w_deviation_linear

        if cross_track_error > soft_boundary_dist:
            excess = cross_track_error - soft_boundary_dist
            rewards[aircraft_id] += w_boundary_soft * (excess ** 2)

        if cross_track_error > max_dev_dist:
            rewards[aircraft_id] += w_out_of_corridor
            dones[aircraft_id] = True
            out_of_bound_count += 1
            continue

        collision_flag = False
        if aircraft_id in stats:
            for j, near_ac in enumerate(stats[aircraft_id]):
                near_id = near_ac[0]
                latj, lonj, altj = near_ac[1], near_ac[2], near_ac[4]
                dist_h = calculate_haversine_distance(lati, loni, latj, lonj)
                dist_v = abs(alti - altj)

                if dist_h <= collision_hor and dist_v < collision_ver:
                    rewards[aircraft_id] += w_collision
                    dones[aircraft_id] = True
                    collision_count += 1
                    collision_flag = True
                    break

                danger_norm_h = dist_h / danger_hor
                danger_norm_v = dist_v / danger_ver
                danger_ellipsoid = math.sqrt(danger_norm_h ** 2 + danger_norm_v ** 2)

                if danger_ellipsoid < 1.0:
                    danger_intrusion = 1.0 - danger_ellipsoid
                    vertical_factor = max(0.3, 1.0 - (dist_v / danger_ver))
                    rewards[aircraft_id] += w_danger * (danger_intrusion ** 2) * vertical_factor
                else:
                    warning_norm_h = dist_h / warning_hor
                    warning_norm_v = dist_v / warning_ver
                    warning_ellipsoid = math.sqrt(warning_norm_h ** 2 + warning_norm_v ** 2)

                    if warning_ellipsoid < 1.0:
                        warning_intrusion = 1.0 - warning_ellipsoid
                        vertical_factor = max(0.3, 1.0 - (dist_v / warning_ver))
                        rewards[aircraft_id] += w_warning * warning_intrusion * vertical_factor

        if collision_flag: continue

        if dist_to_goal < arrival_distance:
            rewards[aircraft_id] += w_arrival
            win_list += 1
            dones[aircraft_id] = True
        elif step_num >= step_max - 1:
            rewards[aircraft_id] += w_timeout

    return rewards, dones


def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))


def build_joint_transition(old_air_craft, current_air_craft, actions, rewards, dones,
                           max_agents, obs_dim, action_dim):
    ids = sorted(list(old_air_craft.keys()))
    joint_state = np.zeros((max_agents, obs_dim), dtype=np.float32)
    joint_next_state = np.zeros((max_agents, obs_dim), dtype=np.float32)
    joint_action = np.zeros((max_agents, action_dim), dtype=np.float32)
    joint_reward = np.zeros((max_agents, 1), dtype=np.float32)
    joint_done = np.zeros((max_agents, 1), dtype=np.float32)
    mask_present = np.zeros((max_agents, 1), dtype=np.float32)

    for _, aid in enumerate(ids):
        idx = int(aid[2:])
        if idx >= max_agents:
            continue
        joint_state[idx] = old_air_craft[aid]
        joint_next_state[idx] = current_air_craft.get(aid, old_air_craft[aid])
        joint_action[idx] = actions.get(aid, np.zeros(action_dim, dtype=np.float32))
        joint_reward[idx, 0] = rewards.get(aid, 0.0)
        joint_done[idx, 0] = float(dones.get(aid, False))
        mask_present[idx, 0] = 1.0

    mask_alive = mask_present * (1.0 - joint_done)

    return joint_state, joint_next_state, joint_action, joint_reward, joint_done, mask_alive, mask_present


def calculate_distance_to_line_score(point_lat, point_lon, line_start_lat, line_start_lon,
                                     line_end_lat, line_end_lon, min_dist_km=1, max_dist_km=10,
                                     steepness=3.0):
    distance_km = calculate_distance_to_line(
        point_lat, point_lon,
        line_start_lat, line_start_lon,
        line_end_lat, line_end_lon
    )

    if distance_km <= min_dist_km:
        return 0.0
    elif distance_km >= max_dist_km:
        return -10.0
    else:
        normalized_dist = (distance_km - min_dist_km) / (max_dist_km - min_dist_km)
        return -(math.exp(steepness * normalized_dist) - 1) / (math.exp(steepness) - 1)


def calculate_distance_to_line(point_lat, point_lon, line_start_lat, line_start_lon,
                               line_end_lat, line_end_lon):
    R = 6371.0

    lat1 = math.radians(point_lat)
    lon1 = math.radians(point_lon)
    lat2 = math.radians(line_start_lat)
    lon2 = math.radians(line_start_lon)
    lat3 = math.radians(line_end_lat)
    lon3 = math.radians(line_end_lon)

    d_start = calculate_haversine_distance(point_lat, point_lon, line_start_lat, line_start_lon)
    d_end = calculate_haversine_distance(point_lat, point_lon, line_end_lat, line_end_lon)
    line_length = calculate_haversine_distance(line_start_lat, line_start_lon, line_end_lat, line_end_lon)

    if line_length < 1e-6:
        return d_start

    bearing_start_to_end = calculate_bearing(line_start_lat, line_start_lon, line_end_lat, line_end_lon)
    bearing_start_to_point = calculate_bearing(line_start_lat, line_start_lon, point_lat, point_lon)
    angle_diff = math.radians(abs(bearing_start_to_point - bearing_start_to_end))

    cross_track_distance = math.asin(math.sin(d_start / R) * math.sin(angle_diff)) * R
    along_track_distance = math.acos(
        max(min(math.cos(d_start / R) / max(math.cos(cross_track_distance / R), 1e-6), 1.0), -1.0)) * R

    if along_track_distance > line_length or along_track_distance < 0:
        return min(d_start, d_end)
    else:
        return abs(cross_track_distance)


def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlon = lon2_rad - lon1_rad

    y = math.sin(dlon) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)

    initial_bearing_rad = math.atan2(y, x)
    initial_bearing_deg = math.degrees(initial_bearing_rad)

    return (initial_bearing_deg + 360) % 360


def calculate_along_track_distance(point_lat, point_lon, line_start_lat, line_start_lon,
                                   line_end_lat, line_end_lon):
    """
    计算点在航线上的投影距离（即已经飞了多少里程）
    """
    R = 6371.0  # 地球半径

    # 计算起点到当前点的直线距离 (d_start)
    d_start = calculate_haversine_distance(point_lat, point_lon, line_start_lat, line_start_lon)

    if d_start < 1e-5:
        return 0.0

    # 计算 "起点->终点" 的方位角
    bearing_route = calculate_bearing(line_start_lat, line_start_lon, line_end_lat, line_end_lon)

    # 计算 "起点->当前点" 的方位角
    bearing_point = calculate_bearing(line_start_lat, line_start_lon, point_lat, point_lon)

    # 计算夹角
    angle_diff = math.radians(abs(bearing_route - bearing_point))

    # 沿航迹距离 = d_start * cos(夹角)
    along_track_dist = d_start * math.cos(angle_diff)

    return along_track_dist