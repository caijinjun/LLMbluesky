import os
import shutil
import bluesky as bs
from bluesky import stack, settings, navdb, traf, sim, scr, tools
from geopy.distance import geodesic
from plugins.Multi_Agent.DDPG_3DEight import DDPG
from plugins.Multi_Agent.Normalizer import AircraftStateNormalizer
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
    # 新增 last_along_track
    global last_goal_distance, last_along_track
    global boundary_strikes
    global last_vertical_distances  # 新增：记录上一步的垂直距离
    global collision_count, out_of_bound_count
    global max_route_distance_static
    global written, SCN_File
    global obs_dim

    written = 0  # [训练模式] 1 = 记录场景文件, 0 = 关闭 (关闭以显著提升速度)

    os.makedirs('output/DDPG/DDPG3D-11/scenarios', exist_ok=True)

    collision_count = 0
    out_of_bound_count = 0

    AircraftStateNormalizer = AircraftStateNormalizer()

    transition_dict = {
        'joint_states': [],
        'joint_next_states': [],
        'joint_actions': [],
        'joint_rewards': [],
        'joint_dones': [],
        'joint_masks': [],
        'joint_presence_masks': []
    }
    reward_memory = []

    # 初始化状态字典
    last_goal_distance = {}
    last_along_track = {}
    boundary_strikes = {}
    last_vertical_distances = {}  # 新增：初始化垂直距离记录

    num_ac = 0
    max_ac = 30
    num_intruders = 3
    obs_dim = 15 + 18  # [修正] 归一化后：自身15维（sin/cos编码扩展）+ 入侵者18维 = 33维
    agent_manager = DDPG(state_dim=14, intruders_dim=18, hidden_dim=256, action_dim=3, max_agents=max_ac)  # [修改] state_dim: 12->14, hidden_dim: 128->256
    
    # [训练模式] 从头开始训练
    print("="*50)
    print("🚀 训练模式: 从头开始训练新模型")
    print("="*50)
        
    reward_list = [0 for _ in range(max_ac)]

    best_win = 0
    win_list = 0
    min_speed = 220
    max_speed = 320
    min_alt = 19500
    max_alt = 21000

    step_num = 0
    episode_max = 6000  # [训练模式] 完整训练轮数
    step_max = 1100

    positions = np.load('./routes/case_study_init.npy')

    all_route_dists = []
    for i in range(len(positions)):
        slat, slon, tlat, tlon = positions[i][0], positions[i][1], positions[i][2], positions[i][3]
        d = calculate_haversine_distance(slat, slon, tlat, tlon)
        all_route_dists.append(d)

    max_route_distance_static = max(all_route_dists) if all_route_dists else 0.0
    avg_route_distance_static = np.mean(all_route_dists) if all_route_dists else 0.0

    route_num = len(positions)
    route_keeper = np.zeros(max_ac, dtype=int)
    choices = [20, 25, 30]
    route_queue = random.choices(choices, k=positions.shape[0])
    episode_num = 0
    old_air_craft = {}
    current_air_craft = {}
    actions = {}

    if written == 1:
        SCN_File = f"output/DDPG/DDPG3D-11/scenarios/{episode_num}.scn"

        if os.path.exists('multi_agent.scn'):
            shutil.copy2('multi_agent.scn', SCN_File)
        else:
            open(SCN_File, 'w').close()

    config = {
        'plugin_name': 'case_DDPG_3D-8',
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

    global collision_count
    global out_of_bound_count

    global written, SCN_File
    global obs_dim

    current_time = bs.sim.simt
    data_time = time.strftime('%H:%M:%S.00', time.gmtime(current_time))

    if step_num >= step_max:
        reset()
        return
    if num_ac == max_ac and len(traf.id) == 0:
        reset()
        return

    old_air_craft = current_air_craft.copy()
    min_dis_craft = get_min_Dis()
    own_state = get_own_state()
    rewards, dones = get_rewards(min_dis_craft)
    current_air_craft = AircraftStateNormalizer.normalize_complete_state(own_state, min_dis_craft)
    for id in traf.id:
        index = int(id[2:])
        reward_list[index] = reward_list[index] * 0.9 + rewards[id]

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
        state = current_air_craft[air_craft]

        raw_action = agent_manager.take_action(state, episode_num)  # [训练模式] 使用episode_num添加探索噪声
        # sp_sig = ternary_bucket(raw_action[0])
        # alt_sig = ternary_bucket(raw_action[1])
        # hdg_sig = ternary_bucket(raw_action[2])

        delta_speed = raw_action[0] * 15  # [修改] 从10增加到15
        delta_alt = raw_action[1] * 200    
        # 航向权限 15
        delta_hdg = raw_action[2] * 12    

        actions[air_craft] = raw_action.copy()

        # apply limits before sending commands
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
            # 清理状态记录
            last_goal_distance.pop(air_craft, None)
            last_along_track.pop(air_craft, None)
            boundary_strikes.pop(air_craft, None)

    if num_ac < max_ac:
        if len(traf.id) == 0:
            for i in range(len(positions)):
                lat, lon, glat, glon, h = positions[i]
                bearing_to_goal = calculate_bearing(lat, lon, glat, glon)
                stack.stack('CRE KL{}, B737, {}, {}, {}, 20000, 250'.format(num_ac, lat, lon, h))
                stack.stack('ADDWPT KL{} {}, {}'.format(num_ac, glat, glon))
                stack.stack('HDG KL{} {}'.format(num_ac, bearing_to_goal))
                stack.stack(f'VNAV KL{num_ac} ON')

                if written == 1:
                    with open(SCN_File, 'a', encoding='utf-8') as f:
                        f.write(f"{data_time}>CRE KL{num_ac}, B737, {lat}, {lon}, {h}, 20000, 250\n")
                        f.write(f"{data_time}>ADDWPT KL{num_ac} {glat}, {glon}\n")
                        f.write(f"{data_time}>HDG KL{num_ac} {bearing_to_goal}\n")
                        f.write(f"{data_time}>VNAV KL{num_ac} ON\n")

                route_keeper[num_ac] = i
                num_ac += 1
                if num_ac == max_ac:
                    break
        else:
            for k in range(len(route_queue)):
                if step_num == route_queue[k]:
                    lat, lon, glat, glon, h = positions[k]
                    bearing_to_goal = calculate_bearing(lat, lon, glat, glon)
                    stack.stack('CRE KL{}, B737, {}, {}, {}, 20000, 250'.format(num_ac, lat, lon, h))
                    stack.stack('ADDWPT KL{} {}, {}'.format(num_ac, glat, glon))
                    stack.stack('HDG KL{} {}'.format(num_ac, bearing_to_goal))
                    stack.stack(f'VNAV KL{num_ac} ON')

                    if written == 1:
                        with open(SCN_File, 'a', encoding='utf-8') as f:
                            f.write(f"{data_time}>CRE KL{num_ac}, B737, {lat}, {lon}, {h}, 20000, 250\n")
                            f.write(f"{data_time}>ADDWPT KL{num_ac} {glat}, {glon}\n")
                            f.write(f"{data_time}>HDG KL{num_ac} {bearing_to_goal}\n")
                            f.write(f"{data_time}>VNAV KL{num_ac} ON\n")

                    route_keeper[num_ac] = k
                    num_ac += 1
                    route_queue[k] = step_num + random.choices(choices, k=1)[0]
                    if num_ac == max_ac:
                        break

    step_num += 1

    # [修改] 每步进行一次更新训练
    if step_num > 0 and step_num % 300 == 0:
        agent_manager.update(transition_dict, episode_num)
        # [关键] 清空字典
        for key in transition_dict:
            transition_dict[key] = []

    if step_num % 100 == 0:
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
    global last_vertical_distances  # 新增

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

    stats_dir = 'output/DDPG/DDPG3D-11'
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

    # [修改] 处理本局剩余的尾部数据
    if len(transition_dict['joint_states']) > 0:
        agent_manager.update(transition_dict, episode_num)

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
    last_vertical_distances = {}  # 新增：重置垂直距离记录

    collision_count = 0
    out_of_bound_count = 0

    # 重置 Buffer 积累字典
    transition_dict = {
        'joint_states': [],
        'joint_next_states': [],
        'joint_actions': [],
        'joint_rewards': [],
        'joint_dones': [],
        'joint_masks': [],
        'joint_presence_masks': []
    }
    reward_list = [0 for _ in range(max_ac)]
    if episode_num % 10 == 0:
        agent_manager.save_models(f"output/DDPG/DDPG3D-11/DDPG")

    best_win = max(win_list, best_win)
    win_list = 0
    if episode_num == episode_max:
        stack.stack('STOP')

    if written == 1:
        SCN_File = f"output/DDPG/DDPG3D-11/scenarios/{episode_num}.scn"

        if os.path.exists('multi_agent.scn'):
            shutil.copy2('multi_agent.scn', SCN_File)
        else:
            open(SCN_File, 'w').close()

    stack.stack('IC multi_agent.scn')


def get_own_state():
    global route_keeper
    own_state = {}
    for i, id in enumerate(traf.id):
        index = i  # [优化] 直接使用循环索引
        lat, lon = traf.lat[index], traf.lon[index]
        speed, alt, hdg = traf.cas[index] * 1.9439, traf.alt[index] * 3.28084, traf.hdg[index]
        
        route = positions[route_keeper[int(id[2:])]]
        start_lat, start_lon, goal_lat, goal_lon, start_h = route
        
        # [新增] 计算到目标的方位角
        bearing_to_goal = calculate_bearing(lat, lon, goal_lat, goal_lon)
        
        # [新增] 计算航向偏差（归一化到[-1, 1]）
        heading_error = bearing_to_goal - hdg
        # 处理角度跨越360度的情况
        if heading_error > 180:
            heading_error -= 360
        elif heading_error < -180:
            heading_error += 360
        heading_error_norm = heading_error / 180.0  # 归一化到[-1, 1]
        
        own_state[id] = [
            lat, lon, speed, alt, hdg,           # 0-4: 基础状态
            start_lat, start_lon,                 # 5-6: 起点
            goal_lat, goal_lon,                   # 7-8: 终点
            start_h,                              # 9: 起点高度
            bearing_to_goal,                      # 10: 到目标的方位角 [新增]
            heading_error_norm,                   # 11: 航向偏差（归一化）[新增]
        ]  # 总计12维 -> 14维
    return own_state


def get_min_Dis():
    global num_intruders
    
    # 获取所有飞机状态
    ids = traf.id
    n_aircraft = len(ids)
    if n_aircraft == 0:
        return {}

    # 转换为 numpy 数组进行批量计算
    lats = np.array(traf.lat)
    lons = np.array(traf.lon)
    cas = np.array(traf.cas) * 1.9439
    alts = np.array(traf.alt) * 3.28084
    trks = np.array(traf.trk)

    # 利用广播机制计算距离矩阵 [N, N]
    # 注意：这里为了速度使用欧氏距离近似（在小范围内误差可接受），或者调用向量化的 haversine
    # 为了保持精度，我们调用之前向量化的 calculate_haversine_distance
    # lat1: [N, 1], lat2: [1, N] -> result: [N, N]
    dist_matrix = calculate_haversine_distance(
        lats[:, np.newaxis], lons[:, np.newaxis],
        lats[np.newaxis, :], lons[np.newaxis, :]
    )
    
    # 将对角线（自己到自己）设为无穷大，避免选到自己
    np.fill_diagonal(dist_matrix, np.inf)

    min_distances = {}
    
    # 对每一行（每架飞机），找到最近的 k 个飞机的索引
    # argsort 默认从小到大排序
    sorted_indices = np.argsort(dist_matrix, axis=1)[:, :num_intruders]

    for i, aircraft_id in enumerate(ids):
        intruder_list = []
        for idx in sorted_indices[i]:
            # 如果距离是无穷大（说明飞机数量少于 num_intruders），则跳过或补零
            if dist_matrix[i, idx] == np.inf:
                continue
                
            intruder_list.append([
                ids[idx],
                lats[idx],
                lons[idx],
                cas[idx],
                alts[idx],
                trks[idx]
            ])
        min_distances[aircraft_id] = intruder_list

    return min_distances


def get_rewards(stats):
    global route_keeper, positions, win_list, last_goal_distance, last_along_track
    global boundary_strikes
    global collision_count, step_num, step_max, out_of_bound_count
    global last_vertical_distances  # 新增

    # ==========================================
    # 1. 基础权重
    # ==========================================
    w_collision = -140.0
    w_arrival = 60.0
    w_progress = 0.2
    w_heading = 0.15
    w_step_cost = -0.01
    w_timeout = -10.0

    # ==========================================
    # 2. 航线与边界
    # ==========================================
    w_deviation_linear = -0.01
    w_boundary_soft = -0.1
    w_out_of_corridor = -45.0
    soft_boundary_dist = 15.0
    max_dev_dist = 25.0

    # ==========================================
    # 3. 避撞参数 - 分层警告区设计
    # ==========================================
    collision_hor = 2.0  # km
    collision_ver = 500.0  # ft

    danger_hor = 4.0  # km
    danger_ver = 600.0  # ft
    w_danger = -25.0

    warning_hor = 8.0  # km
    warning_ver = 800.0  # ft
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
        index = i  # [优化] 直接使用循环索引，避免 O(N) 查找
        lati, loni, alti, hdgi = lat[index], lon[index], traf.alt[index] * 3.28084, traf.hdg[index]
        route_idx = route_keeper[int(aircraft_id[2:])]
        start_lat, start_lon, target_lat, target_lon, _ = positions[route_idx]

        # 1. 偏航计算
        cross_track_error = calculate_distance_to_line(
            lati, loni, start_lat, start_lon, target_lat, target_lon
        )

        # 2. 投影进度计算
        current_along_track = calculate_along_track_distance(
            lati, loni, start_lat, start_lon, target_lat, target_lon
        )

        prev_along_track = last_along_track.get(aircraft_id, current_along_track)
        last_along_track[aircraft_id] = current_along_track

        progress_projected = current_along_track - prev_along_track
        rewards[aircraft_id] += w_progress * clamp(progress_projected, -2.0, 2.0)

        # 3. 航向对齐
        dist_to_goal = calculate_haversine_distance(lati, loni, target_lat, target_lon)
        if dist_to_goal > arrival_distance:
            bearing_to_goal = calculate_bearing(lati, loni, target_lat, target_lon)
            angle_diff = abs(bearing_to_goal - hdgi)
            angle_diff = min(angle_diff, 360 - angle_diff)
            rewards[aircraft_id] += w_heading * math.cos(math.radians(angle_diff))

        # 4. 基础消耗
        rewards[aircraft_id] += w_step_cost

        # 5. 偏航惩罚
        rewards[aircraft_id] += cross_track_error * w_deviation_linear

        if cross_track_error > soft_boundary_dist:
            excess = cross_track_error - soft_boundary_dist
            rewards[aircraft_id] += w_boundary_soft * (excess ** 2)

        # 硬性出界
        if cross_track_error > max_dev_dist:
            rewards[aircraft_id] += w_out_of_corridor
            dones[aircraft_id] = True
            out_of_bound_count += 1
            continue

        # 6. 避撞逻辑
        collision_flag = False
        if aircraft_id in stats:
            for j, near_ac in enumerate(stats[aircraft_id]):
                near_id = near_ac[0]
                latj, lonj, altj = near_ac[1], near_ac[2], near_ac[4]
                dist_h = calculate_haversine_distance(lati, loni, latj, lonj)
                dist_v = abs(alti - altj)

                # 6.1 碰撞检测
                if dist_h <= collision_hor and dist_v < collision_ver:
                    rewards[aircraft_id] += w_collision
                    dones[aircraft_id] = True
                    collision_count += 1
                    collision_flag = True
                    break


                # 6.3 内层危险区 (2-4 km)
                danger_norm_h = dist_h / danger_hor
                danger_norm_v = dist_v / danger_ver
                danger_ellipsoid = math.sqrt(danger_norm_h ** 2 + danger_norm_v ** 2)

                if danger_ellipsoid < 1.0:
                    danger_intrusion = 1.0 - danger_ellipsoid
                    vertical_factor = max(0.3, 1.0 - (dist_v / danger_ver))
                    rewards[aircraft_id] += w_danger * (danger_intrusion ** 2) * vertical_factor
                else:
                    # 6.4 外层警告区 (4-8 km)
                    warning_norm_h = dist_h / warning_hor
                    warning_norm_v = dist_v / warning_ver
                    warning_ellipsoid = math.sqrt(warning_norm_h ** 2 + warning_norm_v ** 2)

                    if warning_ellipsoid < 1.0:
                        warning_intrusion = 1.0 - warning_ellipsoid
                        vertical_factor = max(0.3, 1.0 - (dist_v / warning_ver))
                        rewards[aircraft_id] += w_warning * warning_intrusion * vertical_factor

        if collision_flag: continue

        # 7. 到达检测
        if dist_to_goal < arrival_distance:
            rewards[aircraft_id] += w_arrival
            win_list += 1
            dones[aircraft_id] = True
        elif step_num >= step_max - 1:
            rewards[aircraft_id] += w_timeout

    return rewards, dones


def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))


# [关键修改] mask_alive 逻辑修复
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
        idx = int(aid[2:])  # 固定槽位：KLxx -> xx
        if idx >= max_agents:
            continue
        joint_state[idx] = old_air_craft[aid]
        joint_next_state[idx] = current_air_craft.get(aid, old_air_craft[aid])
        joint_action[idx] = actions.get(aid, np.zeros(action_dim, dtype=np.float32))
        joint_reward[idx, 0] = rewards.get(aid, 0.0)
        joint_done[idx, 0] = float(dones.get(aid, False))
        mask_present[idx, 0] = 1.0

    # 仅对仍然存活/未终止的槽位置 1
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
    """
    Vectorized Haversine distance calculation.
    Supports both scalar and numpy array inputs.
    """
    R = 6371.0
    
    # Ensure inputs are numpy arrays if they are lists
    if isinstance(lat1, list): lat1 = np.array(lat1)
    if isinstance(lon1, list): lon1 = np.array(lon1)
    if isinstance(lat2, list): lat2 = np.array(lat2)
    if isinstance(lon2, list): lon2 = np.array(lon2)

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    
    # Clip to ensure value is within [0, 1] to avoid numerical errors
    a = np.clip(a, 0, 1)
    
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Vectorized bearing calculation.
    """
    # Ensure inputs are numpy arrays
    if isinstance(lat1, list): lat1 = np.array(lat1)
    if isinstance(lon1, list): lon1 = np.array(lon1)
    if isinstance(lat2, list): lat2 = np.array(lat2)
    if isinstance(lon2, list): lon2 = np.array(lon2)

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlon = lon2_rad - lon1_rad

    y = np.sin(dlon) * np.cos(lat2_rad)
    x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)

    initial_bearing_rad = np.arctan2(y, x)
    initial_bearing_deg = np.degrees(initial_bearing_rad)

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


