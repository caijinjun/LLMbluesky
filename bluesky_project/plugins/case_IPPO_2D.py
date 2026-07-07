import os
import bluesky as bs
from bluesky import stack, settings, navdb, traf, sim, scr, tools
from geopy.distance import geodesic
from plugins.Multi_Agent.DDPG_3D import DDPG
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
    global action_list, speed_list, alt_list
    global max_speed, min_speed, max_alt, min_alt
    global win_list, best_win, reward_list, reward_memory
    global transition_dict
    global AircraftStateNormalizer
    global last_goal_distance

    AircraftStateNormalizer = AircraftStateNormalizer()

    transition_dict = {
        'states': [],
        'actions': [],
        'next_states': [],
        'rewards': [],
        'dones': []
    }
    reward_memory = []
    last_goal_distance = {}

    num_ac = 0
    max_ac = 30
    num_intruders = 3
    agent_manager = DDPG(state_dim=12, intruders_dim=18, hidden_dim=128, action_dim=3)
    reward_list = [0 for _ in range(max_ac)]

    best_win = 0
    win_list = 0
    # 合理上下限，防止 clamp 失效
    min_speed = 200
    max_speed = 320
    min_alt = 20000
    max_alt = 21000
    speed_list = [-20, 0, 20]
    alt_list = [-100, 0, 100]

    step_num = 0
    episode_max = 50000
    step_max = 2000

    positions = np.load('./routes/case_study_init.npy')
    route_num = len(positions)
    route_keeper = np.zeros(max_ac, dtype=int)
    choices = [20, 25, 30]  # 4 minutes, 5 minutes, 6 minutes
    route_queue = random.choices(choices, k=positions.shape[0])
    episode_num = 0
    old_air_craft = {}
    current_air_craft = {}
    actions = {}

    config = {
        'plugin_name': 'case_IPPO_2D',
        'plugin_type': 'sim',
        'update_interval': 12.0,
        'update': update,
    }

    return config, {}


def update():
    # ... (全局变量保持不变) ...
    global num_ac, max_ac, num_intruders
    global agent_manager
    global positions, route_keeper, route_num, route_queue, choices
    global episode_num, episode_max
    global step_num, step_max
    global actions, old_air_craft, current_air_craft
    global action_list, speed_list, alt_list
    global max_speed, min_speed, max_alt, min_alt
    global win_list, best_win, reward_list, reward_memory
    global transition_dict
    global AircraftStateNormalizer
    global last_goal_distance
    global collision_count
    global out_of_bound_count

    # === 混合控制参数 ===
    SAFE_RADIUS = 15.0  # (km)

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

    for i, air_craft in enumerate(traf.id):
        if air_craft in old_air_craft.keys():
            transition_dict['states'].append(old_air_craft[air_craft])
            transition_dict['actions'].append(actions.get(air_craft, np.zeros(2)))
            transition_dict['next_states'].append(current_air_craft[air_craft])
            transition_dict['rewards'].append(rewards[air_craft])
            transition_dict['dones'].append(dones[air_craft])

    # === 核心修改：生成动作与混合控制 ===
    actions = {}
    for i, air_craft in enumerate(traf.id):
        state = current_air_craft[air_craft]

        # 1. 先让 DDPG 算一个动作 (用于 buffer 记录)
        rl_action = agent_manager.take_action(state)
        actions[air_craft] = rl_action

        lati, loni = traf.lat[i], traf.lon[i]
        alti, hdgi = traf.alt[i] * 3.28084, traf.hdg[i]

        route_idx = route_keeper[int(air_craft[2:])]
        target_lat = positions[route_idx][2]
        target_lon = positions[route_idx][3]

        # === 混合控制逻辑判断 ===
        is_dangerous = False
        nearby_aircrafts = min_dis_craft.get(air_craft, [])

        if len(nearby_aircrafts) > 0:
            nearest = nearby_aircrafts[0]
            near_lat, near_lon = nearest[0], nearest[1]
            dist_to_intruder = calculate_haversine_distance(lati, loni, near_lat, near_lon)

            if dist_to_intruder < SAFE_RADIUS:
                is_dangerous = True

        # === 执行控制 ===
        if is_dangerous:
            # >>> 危险模式：DDPG 接管所有控制 <<<
            delta_speed = rl_action[0] * 15
            delta_alt = rl_action[0] * 150

            raw_hdg = rl_action[1]
            if raw_hdg < -0.33:
                delta_hdg = -3.0
            elif raw_hdg > 0.33:
                delta_hdg = 3.0
            else:
                delta_hdg = 0.0

            # 计算新状态
            new_tas = clamp(traf.cas[i] * 1.9437 + delta_speed, min_speed, max_speed)
            new_alt = clamp(traf.alt[i] * 3.28084 + delta_alt, min_alt, max_alt)
            new_hdg = (traf.hdg[i] + delta_hdg) % 360

        else:
            # >>> 安全模式：规则导航 <<<
            # 1. 航向控制：指向终点
            desired_bearing = calculate_bearing(lati, loni, target_lat, target_lon)
            hdg_diff = (desired_bearing - hdgi + 180) % 360 - 180

            if abs(hdg_diff) > 3.0:
                delta_hdg = 3.0 * np.sign(hdg_diff)
            else:
                delta_hdg = hdg_diff
            new_hdg = (traf.hdg[i] + delta_hdg) % 360

            # 2. 速度和高度：【修改点】保持当前值不变
            # 获取当前值并转换单位 (m/s -> kts, m -> ft)
            current_tas = traf.cas[i] * 1.9437
            current_alt = traf.alt[i] * 3.28084

            # 直接赋给新值 (加 clamp 是为了防止之前的操作导致数值越界，保持稳定性)
            new_tas = clamp(current_tas, min_speed, max_speed)
            new_alt = clamp(current_alt, min_alt, max_alt)

        # === 发送指令给 BlueSky ===
        stack.stack('SPD {} {}'.format(air_craft, new_tas))
        stack.stack('ALT {} {}'.format(air_craft, new_alt))
        stack.stack('HDG {} {}'.format(air_craft, new_hdg))

        # 奖励 Shaping
        rewards[air_craft] -= 0.01 * (abs(rl_action[0]) + abs(rl_action[1]))

        if dones[air_craft]:
            stack.stack('DEL {}'.format(air_craft))
            last_goal_distance.pop(air_craft, None)

    if num_ac < max_ac:
        if len(traf.id) == 0:
            for i in range(len(positions)):
                lat, lon, glat, glon, h = positions[i]
                stack.stack('CRE KL{}, B737, {}, {}, {}, 20000, 250'.format(num_ac, lat, lon, h))
                stack.stack('ADDWPT KL{} {}, {}'.format(num_ac, glat, glon))
                stack.stack(f'VNAV KL{num_ac} ON')
                route_keeper[num_ac] = i
                num_ac += 1
                if num_ac == max_ac:
                    break
        else:
            for k in range(len(route_queue)):
                if step_num == route_queue[k]:
                    lat, lon, glat, glon, h = positions[k]
                    stack.stack('CRE KL{}, B737, {}, {}, {}, 20000, 250'.format(num_ac, lat, lon, h))
                    stack.stack('ADDWPT KL{} {}, {}'.format(num_ac, glat, glon))
                    stack.stack(f'VNAV KL{num_ac} ON')
                    route_keeper[num_ac] = k
                    num_ac += 1
                    route_queue[k] = step_num + random.choices(choices, k=1)[0]
                    if num_ac == max_ac:
                        break

    step_num += 1
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
    global last_goal_distance
    num_ac = 0
    step_num = 0
    episode_num += 1
    route_keeper = np.zeros(max_ac, dtype=int)
    actions = {}
    old_air_craft = {}
    current_air_craft = {}
    route_queue = random.choices([20, 25, 30], k=positions.shape[0])
    last_goal_distance = {}

    print("Episode: {} | Win: {} | Best Win: {} | Reward: {}".format(episode_num, win_list, best_win, np.mean(reward_list)))
    reward_memory.append(np.mean(reward_list))
    os.makedirs('output/DDPG/DDPG2D', exist_ok=True)
    np.save('output/DDPG/DDPG2D/reward_memory.npy', reward_memory)

    agent_manager.update(transition_dict)
    transition_dict = {
        'states': [],
        'actions': [],
        'next_states': [],
        'rewards': [],
        'dones': []
    }
    reward_list = [0 for _ in range(max_ac)]
    if episode_num % 10 == 0:
        agent_manager.save_models(f"output\\DDPG\\DDPG2D\\DDPG")

    best_win = max(win_list, best_win)
    win_list = 0
    if episode_num == episode_max:
        stack.stack('STOP')

    stack.stack('IC multi_agent.scn')


def get_own_state():
    """获取自己的状态特征"""
    global route_keeper
    own_state = {}
    for i, id in enumerate(traf.id):
        index = traf.id2idx(id)
        own_state[id] = [traf.lat[index], traf.lon[index], traf.cas[index] * 1.9439, traf.alt[index] * 3.28084, traf.hdg[index]]
        route = positions[route_keeper[int(id[2:])]]
        own_state[id].extend(route)
    return own_state


def get_min_Dis():
    """获取最近无人机的状态"""
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
        min_distances[id_i] = []
        for j, id_j in enumerate(id):
            if i != j:
                dist[id_j] = calculate_haversine_distance(lat[i], lon[i], lat[j], lon[j])
        sorted_list = list(dict(sorted(dist.items(), key=lambda item: item[1])).keys())
        for z in range(len(sorted_list)):
            air_index = traf.id.index(sorted_list[z])
            min_distances[id_i].append([lat[air_index], lon[air_index], traf.cas[air_index] * 1.9439, traf.alt[air_index] * 3.28084, traf.trk[air_index]])
    return min_distances


def get_rewards(stats):
    """计算奖励，加入前进/航迹/航向/时间/平滑性 shaping"""
    global route_keeper
    global positions
    global win_list
    global last_goal_distance
    collision_distance = 10  # km
    warning_distance = 20    # km
    arrival_distance = 5     # km
    height_distance = 300

    collision_penalty = -10
    arrival_reward = 1

    id = traf.id
    lon = traf.lon
    lat = traf.lat
    n_aircraft = len(id)

    rewards = {id_: 0.0 for id_ in id}
    dones = {id_: False for id_ in id}

    if n_aircraft == 0:
        return rewards, dones

    for i, aircraft_id in enumerate(id):
        index = traf.id2idx(aircraft_id)
        lati, loni, alti, hdgi = lat[index], lon[index], traf.alt[index] * 3.28084, traf.hdg[index]
        # 进度：距离目标减少量
        start_lat, start_lon, target_lat, target_lon, hdg_ori = positions[route_keeper[int(aircraft_id[2:])]]
        dist_to_goal = calculate_haversine_distance(lati, loni, target_lat, target_lon)
        prev_dist = last_goal_distance.get(aircraft_id, dist_to_goal)
        progress = prev_dist - dist_to_goal
        rewards[aircraft_id] += 0.05 * progress  # 每 km 前进奖励
        rewards[aircraft_id] -= 0.01  # 时间惩罚
        last_goal_distance[aircraft_id] = dist_to_goal

        # 航迹保持
        path_score = calculate_distance_to_line_score(lati, loni, start_lat, start_lon, target_lat, target_lon, min_dist_km=1, max_dist_km=10, steepness=3.0)
        rewards[aircraft_id] += 0.02 * path_score  # path_score in [-10,0]

        # 航向误差惩罚
        bearing_to_goal = calculate_bearing(lati, loni, target_lat, target_lon)
        ang_err = abs(bearing_to_goal - hdgi)
        ang_err = min(ang_err, 360 - ang_err)
        rewards[aircraft_id] -= 0.02 * (ang_err / 180.0)

        # 碰撞/接近惩罚
        for j in range(len(stats[aircraft_id])):
            near_aircraft = stats[aircraft_id][j]
            latj, lonj, altj, hdgj = near_aircraft[0], near_aircraft[1], near_aircraft[3], near_aircraft[4]
            dist = calculate_haversine_distance(lati, loni, latj, lonj)
            if dist <= collision_distance and abs(alti - altj) < height_distance:
                rewards[aircraft_id] = collision_penalty
                dones[aircraft_id] = True
                break
            elif dist < warning_distance and abs(alti - altj) < height_distance:
                rewards[aircraft_id] += -2 + 0.1 * dist
                break

        if not dones[aircraft_id]:
            if dist_to_goal < arrival_distance:
                rewards[aircraft_id] += arrival_reward
                win_list += 1
                dones[aircraft_id] = True

    return rewards, dones


def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))


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
    R = 6371.0  # km

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
    along_track_distance = math.acos(max(min(math.cos(d_start / R) / max(math.cos(cross_track_distance / R), 1e-6), 1.0), -1.0)) * R

    if along_track_distance > line_length or along_track_distance < 0:
        return min(d_start, d_end)
    else:
        return abs(cross_track_distance)


def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0  # km

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
