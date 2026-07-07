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
    global collision_count
    global out_of_bound_count
    global max_route_distance_static

    collision_count = 0
    out_of_bound_count = 0

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
    
    # === 关键修改: 创建模型并加载已训练权重 ===
    agent_manager = DDPG(state_dim=12, intruders_dim=18, hidden_dim=128, action_dim=2)
    
    # 加载模型权重 (修改为您的模型路径)
    model_path = "output\\DDPG\\DDPG3D\\DDPG"
    print(f"Loading model from: {model_path}")
    agent_manager.load_models(model_path)
    print("Model loaded successfully!")
    
    reward_list = [0 for _ in range(max_ac)]

    best_win = 0
    win_list = 0
    min_speed = 200
    max_speed = 320
    min_alt = 20000
    max_alt = 21000
    speed_list = [-20, 0, 20]
    alt_list = [-100, 0, 100]

    step_num = 0
    episode_max = 100  # 测试100轮
    step_max = 800

    # 加载路线数据
    positions = np.load('./routes/case_study_init.npy')

    # 计算所有路线的初始最大距离
    all_route_dists = []
    for i in range(len(positions)):
        slat, slon, tlat, tlon = positions[i][0], positions[i][1], positions[i][2], positions[i][3]
        d = calculate_haversine_distance(slat, slon, tlat, tlon)
        all_route_dists.append(d)

    max_route_distance_static = max(all_route_dists) if all_route_dists else 0.0
    avg_route_distance_static = np.mean(all_route_dists) if all_route_dists else 0.0

    print("-" * 30)
    print(f"Pure DDPG Testing Mode (No Rules)")
    print(f"Total Routes: {len(positions)}")
    print(f"Max Route Distance: {max_route_distance_static:.2f} km")
    print(f"Avg Route Distance: {avg_route_distance_static:.2f} km")
    print("-" * 30)

    route_num = len(positions)
    route_keeper = np.zeros(max_ac, dtype=int)
    choices = [20, 25, 30]
    route_queue = random.choices(choices, k=positions.shape[0])
    episode_num = 0
    old_air_craft = {}
    current_air_craft = {}
    actions = {}

    config = {
        'plugin_name': 'case_DDPG_3D_test',
        'plugin_type': 'sim',
        'update_interval': 10.0,
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
    global action_list, speed_list, alt_list
    global max_speed, min_speed, max_alt, min_alt
    global win_list, best_win, reward_list, reward_memory
    global transition_dict
    global AircraftStateNormalizer
    global last_goal_distance
    global collision_count
    global out_of_bound_count

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

    # === 纯 DDPG 控制 (无规则) ===
    actions = {}
    for i, air_craft in enumerate(traf.id):
        state = current_air_craft[air_craft]

        # 使用 DDPG 生成动作 (无噪声,因为是测试模式)
        action = agent_manager.take_action(state, episode_num=-1)
        actions[air_craft] = action

        # 能量控制
        delta_speed = action[0] * 15
        delta_alt = action[0] * 150

        # 航向动作离散化
        raw_hdg = action[1]
        if raw_hdg < -0.33:
            delta_hdg = -3.0
        elif raw_hdg > 0.33:
            delta_hdg = 3.0
        else:
            delta_hdg = 0.0

        # 应用物理限制
        new_tas = clamp(traf.cas[i] * 1.9437 + delta_speed, min_speed, max_speed)
        new_alt = clamp(traf.alt[i] * 3.28084 + delta_alt, min_alt, max_alt)
        new_hdg = (traf.hdg[i] + delta_hdg) % 360

        # 发送指令
        stack.stack('SPD {} {}'.format(air_craft, new_tas))
        stack.stack('ALT {} {}'.format(air_craft, new_alt))
        stack.stack('HDG {} {}'.format(air_craft, new_hdg))

        if dones[air_craft]:
            stack.stack('DEL {}'.format(air_craft))
            last_goal_distance.pop(air_craft, None)

    # 创建航空器逻辑
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
    global collision_count
    global out_of_bound_count

    # 计算存活飞机的平均剩余距离
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

    print("=" * 60)
    print(f"TEST Episode: {episode_num}")
    print(f"Success Rate: {success_rate:.2%} ({win_list}/{max_ac})")
    print(f"Collisions: {collision_count}")
    print(f"Out of Bound: {out_of_bound_count}")
    print(f"Avg Remaining Distance: {avg_survive_dist:.2f} km")
    print(f"Avg Reward: {avg_reward:.4f}")
    print("=" * 60)

    # 保存测试结果
    stats_dir = 'output/DDPG/DDPG3D_test'
    os.makedirs(stats_dir, exist_ok=True)
    stats_file = os.path.join(stats_dir, 'test_results.csv')

    file_exists = os.path.exists(stats_file)
    with open(stats_file, 'a') as f:
        if not file_exists:
            f.write("Episode,SuccessRate,Wins,Collisions,OOB,AvgRemainDist,AvgReward\\n")
        f.write("{},{:.4f},{},{},{},{:.4f},{:.4f}\\n".format(
            episode_num, success_rate, win_list, collision_count, out_of_bound_count, avg_survive_dist, avg_reward))

    # 注意: 测试模式下不调用 agent_manager.update() (不训练)

    # 重置变量
    num_ac = 0
    step_num = 0
    episode_num += 1
    route_keeper = np.zeros(max_ac, dtype=int)
    actions = {}
    old_air_craft = {}
    current_air_craft = {}
    route_queue = random.choices([20, 25, 30], k=positions.shape[0])
    last_goal_distance = {}
    collision_count = 0
    out_of_bound_count = 0

    transition_dict = {
        'states': [],
        'actions': [],
        'next_states': [],
        'rewards': [],
        'dones': []
    }
    reward_list = [0 for _ in range(max_ac)]

    best_win = max(win_list, best_win)
    win_list = 0
    
    if episode_num == episode_max:
        print("\\n" + "=" * 60)
        print("TESTING COMPLETED!")
        print(f"Best Success Rate: {best_win}/{max_ac}")
        print("=" * 60)
        stack.stack('STOP')

    stack.stack('IC multi_agent.scn')


def get_own_state():
    global route_keeper
    own_state = {}
    for i, id in enumerate(traf.id):
        index = traf.id2idx(id)
        own_state[id] = [traf.lat[index], traf.lon[index], traf.cas[index] * 1.9439, traf.alt[index] * 3.28084,
                         traf.hdg[index]]
        route = positions[route_keeper[int(id[2:])]]
        own_state[id].extend(route)
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
        min_distances[id_i] = []
        for j, id_j in enumerate(id):
            if i != j:
                dist[id_j] = calculate_haversine_distance(lat[i], lon[i], lat[j], lon[j])
        sorted_list = list(dict(sorted(dist.items(), key=lambda item: item[1])).keys())
        for z in range(len(sorted_list)):
            air_index = traf.id.index(sorted_list[z])
            min_distances[id_i].append(
                [lat[air_index], lon[air_index], traf.cas[air_index] * 1.9439, traf.alt[air_index] * 3.28084,
                 traf.trk[air_index]])
    return min_distances


def get_rewards(stats):
    global route_keeper
    global positions
    global win_list
    global last_goal_distance
    global collision_count
    global step_num, step_max
    global out_of_bound_count

    # 参数设置
    collision_distance = 3
    warning_distance = 10
    arrival_distance = 10
    height_distance = 300

    # 权重设置 (与训练时一致)
    w_collision = -5.0
    w_arrival = 5.0
    w_progress = 0.1
    w_heading = 0.05
    w_timeout = -2.0

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

        route_idx = route_keeper[int(aircraft_id[2:])]
        start_lat, start_lon, target_lat, target_lon, hdg_ori = positions[route_idx]

        dist_to_goal = calculate_haversine_distance(lati, loni, target_lat, target_lon)
        prev_dist = last_goal_distance.get(aircraft_id, dist_to_goal)

        # 1. 前进奖励
        progress = prev_dist - dist_to_goal
        last_goal_distance[aircraft_id] = dist_to_goal
        progress = clamp(progress, -2.0, 2.0)
        rewards[aircraft_id] += w_progress * progress

        # 2. 航向引导奖励
        bearing_to_goal = calculate_bearing(lati, loni, target_lat, target_lon)
        angle_diff = abs(bearing_to_goal - hdgi)
        angle_diff = min(angle_diff, 360 - angle_diff)
        heading_reward = math.cos(math.radians(angle_diff))
        rewards[aircraft_id] += w_heading * heading_reward

        # 3. 存在性惩罚
        rewards[aircraft_id] -= 0.005

        # 4. 碰撞检测
        collision_flag = False
        for j in range(len(stats[aircraft_id])):
            near_aircraft = stats[aircraft_id][j]
            latj, lonj, altj, hdgj = near_aircraft[0], near_aircraft[1], near_aircraft[3], near_aircraft[4]
            dist = calculate_haversine_distance(lati, loni, latj, lonj)

            if dist <= collision_distance and abs(alti - altj) < height_distance:
                rewards[aircraft_id] += w_collision
                dones[aircraft_id] = True
                collision_count += 1
                collision_flag = True
                break
            elif dist < warning_distance and abs(alti - altj) < height_distance:
                ratio = (dist - collision_distance) / (warning_distance - collision_distance)
                penalty = -1.0 * (1.0 - ratio)
                rewards[aircraft_id] += penalty

        if collision_flag:
            continue

        # 5. 到达检测
        if dist_to_goal < arrival_distance:
            rewards[aircraft_id] += w_arrival
            win_list += 1
            dones[aircraft_id] = True

        # 6. 超时检测
        elif step_num >= step_max - 1:
            rewards[aircraft_id] += w_timeout

        # 7. 防逃逸机制
        if dist_to_goal > 650:
            rewards[aircraft_id] += -3
            dones[aircraft_id] = True
            out_of_bound_count += 1

    return rewards, dones


def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))


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
