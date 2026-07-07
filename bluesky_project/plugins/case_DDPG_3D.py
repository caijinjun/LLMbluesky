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

    # 全局计数变量
    global collision_count
    global out_of_bound_count

    # === 新增：全局变量存储最大航线距离，供防逃逸判断使用 ===
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
    agent_manager = DDPG(state_dim=12, intruders_dim=18, hidden_dim=128, action_dim=2)
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
    episode_max = 50000
    step_max = 800

    # 加载路线数据
    positions = np.load('./routes/case_study_init.npy')

    # === 修改点：计算所有路线的初始最大距离 ===
    all_route_dists = []
    for i in range(len(positions)):
        # positions 格式: [start_lat, start_lon, target_lat, target_lon, hdg...]
        slat, slon, tlat, tlon = positions[i][0], positions[i][1], positions[i][2], positions[i][3]
        d = calculate_haversine_distance(slat, slon, tlat, tlon)
        all_route_dists.append(d)

    max_route_distance_static = max(all_route_dists) if all_route_dists else 0.0
    avg_route_distance_static = np.mean(all_route_dists) if all_route_dists else 0.0

    print("-" * 30)
    print(f"Route Analysis:")
    print(f"Total Routes: {len(positions)}")
    print(f"Max Route Distance: {max_route_distance_static:.2f} km")
    print(f"Avg Route Distance: {avg_route_distance_static:.2f} km")
    print(f"Current OOB Threshold: 600 km")
    if max_route_distance_static > 600:
        print("WARNING: Max route distance > 650km! Increase your OOB threshold!")
    print("-" * 30)
    # ==========================================

    route_num = len(positions)
    route_keeper = np.zeros(max_ac, dtype=int)
    choices = [20, 25, 30]
    route_queue = random.choices(choices, k=positions.shape[0])
    episode_num = 0
    old_air_craft = {}
    current_air_craft = {}
    actions = {}

    config = {
        'plugin_name': 'case_DDPG_3D',
        'plugin_type': 'sim',
        'update_interval': 10.0,
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

            if abs(hdg_diff) > 5.0:
                delta_hdg = 5.0 * np.sign(hdg_diff)
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
    # ### 修改点: 引用全局变量
    global collision_count
    global out_of_bound_count  # 引用

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

    # === 修改点 1: 计算成功率 ===
    # max_ac 是预设的飞机总数 (30), win_list 是到达终点的数量
    success_rate = win_list / max_ac
    avg_reward = np.mean(reward_list)

    # === 修改点 2: 打印日志 (Win -> Success Rate) ===
    print("Episode: {} | Success Rate: {:.2f} | Collisions: {} | OOB: {} | Avg Dist: {:.2f} | Reward: {:.4f}".format(
        episode_num, success_rate, collision_count, out_of_bound_count, avg_survive_dist, avg_reward))

    # === 修改点 3: 保存详细数据到文件 (CSV格式，方便后续画图) ===
    stats_dir = 'output/DDPG/DDPG3D'
    os.makedirs(stats_dir, exist_ok=True)
    stats_file = os.path.join(stats_dir, 'training_stats.csv')

    # 如果文件不存在，先写入表头
    file_exists = os.path.exists(stats_file)
    with open(stats_file, 'a') as f:
        if not file_exists:
            f.write("Episode,SuccessRate,Collisions,OOB,AvgRemainDist,AvgReward\n")
        # 写入本轮数据
        f.write("{},{:.4f},{},{},{:.4f},{:.4f}\n".format(
            episode_num, success_rate, collision_count, out_of_bound_count, avg_survive_dist, avg_reward))

    # 保存 Reward Memory (保持原样)
    reward_memory.append(avg_reward)
    np.save(os.path.join(stats_dir, 'reward_memory.npy'), reward_memory)

    agent_manager.update(transition_dict)

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
    out_of_bound_count = 0  # 重置 OOB 计数

    transition_dict = {
        'states': [],
        'actions': [],
        'next_states': [],
        'rewards': [],
        'dones': []
    }
    reward_list = [0 for _ in range(max_ac)]
    if episode_num % 10 == 0:
        agent_manager.save_models(f"output\\DDPG\\DDPG3D\\DDPG")

    best_win = max(win_list, best_win)
    win_list = 0
    if episode_num == episode_max:
        stack.stack('STOP')

    stack.stack('IC multi_agent.scn')


# ... (get_own_state, get_min_Dis 保持不变) ...
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
    # ### 修改点: 引用全局变量
    global out_of_bound_count

    # === 距离参数设置 ===
    collision_distance = 3.0  # 实际碰撞半径 (km)
    warning_distance = 10.0  # 感知/预警半径 (km)
    arrival_distance = 10.0  # 到达判定半径 (km)
    height_distance = 300  # 高度差判定 (ft)

    # === 权重设置 (Reward Weights) ===
    w_collision = -5.0  # 撞机惩罚 (大)
    w_arrival = 5.0  # 到达奖励 (大)
    w_progress = 0.1  # 前进奖励 (持续引导)
    w_heading = 0.05  # 航向对齐 (关键：由0.005提升到0.05，增强指向性)
    w_off_course = 0.1  # 偏航惩罚系数 (新增：配合 score 函数使用)
    w_timeout = -2.0  # 超时惩罚
    w_exist = -0.005  # 存在步数惩罚 (逼迫尽快完成)

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
        # 获取当前状态
        lati, loni = lat[index], lon[index]
        alti, hdgi = traf.alt[index] * 3.28084, traf.hdg[index]

        # 获取目标航路信息
        route_idx = route_keeper[int(aircraft_id[2:])]
        start_lat, start_lon, target_lat, target_lon, hdg_ori = positions[route_idx]

        # 计算距离
        dist_to_goal = calculate_haversine_distance(lati, loni, target_lat, target_lon)
        prev_dist = last_goal_distance.get(aircraft_id, dist_to_goal)

        # ----------------------------------------------------------------
        # 1. 前进奖励 (Progress Reward)
        # ----------------------------------------------------------------
        progress = prev_dist - dist_to_goal
        last_goal_distance[aircraft_id] = dist_to_goal
        # 限制单步奖励范围，防止跳变
        progress = clamp(progress, -2.0, 2.0)
        rewards[aircraft_id] += w_progress * progress

        # ----------------------------------------------------------------
        # 2. 航向引导奖励 (Heading Alignment) - 增强版
        # ----------------------------------------------------------------
        bearing_to_goal = calculate_bearing(lati, loni, target_lat, target_lon)
        angle_diff = abs(bearing_to_goal - hdgi)
        angle_diff = min(angle_diff, 360 - angle_diff)
        # 使用 Cosine 映射：0度->+1.0, 90度->0.0, 180度->-1.0
        heading_reward = math.cos(math.radians(angle_diff))
        rewards[aircraft_id] += w_heading * heading_reward

        # ----------------------------------------------------------------
        # 3. 航路保持/偏航惩罚 (Cross-Track Error) - 新增关键点
        # ----------------------------------------------------------------
        # calculate_distance_to_line_score 返回值范围通常是 [-10.0, 0.0]
        # 0.0 表示在航线上，-10.0 表示偏离很远
        path_score = calculate_distance_to_line_score(
            lati, loni,
            start_lat, start_lon,
            target_lat, target_lon,
            min_dist_km=2.0,  # 允许左右 2km 的自由机动空间用于避障
            max_dist_km=20.0,  # 超过 20km 就算严重偏航
            steepness=2.0  # 惩罚曲线陡度
        )
        # score 是负数，所以直接加
        rewards[aircraft_id] += w_off_course * path_score

        # ----------------------------------------------------------------
        # 4. 存在性惩罚 (Time Penalty)
        # ----------------------------------------------------------------
        rewards[aircraft_id] += w_exist

        # ----------------------------------------------------------------
        # 5. 碰撞与冲突检测 (Collision Avoidance)
        # ----------------------------------------------------------------
        collision_flag = False
        if aircraft_id in stats:
            for j in range(len(stats[aircraft_id])):
                near_aircraft = stats[aircraft_id][j]
                # near_aircraft: [lat, lon, tas, alt, trk]
                latj, lonj, altj = near_aircraft[0], near_aircraft[1], near_aircraft[3]

                dist = calculate_haversine_distance(lati, loni, latj, lonj)
                alt_diff = abs(alti - altj)

                # A. 发生碰撞 (强惩罚，结束回合)
                if dist <= collision_distance and alt_diff < height_distance:
                    rewards[aircraft_id] += w_collision
                    dones[aircraft_id] = True
                    collision_count += 1
                    collision_flag = True
                    break  # 撞了一个就算输，不用算撞了几个

                # B. 进入预警区 (线性惩罚)
                elif dist < warning_distance and alt_diff < height_distance:
                    # 距离越近，惩罚越大 (从 0 到 -1.0)
                    # dist=3 -> penalty=-1.0
                    # dist=10 -> penalty=0.0
                    penalty = -1.0 * (1.0 - (dist - collision_distance) / (warning_distance - collision_distance))
                    rewards[aircraft_id] += penalty

        if collision_flag:
            continue  # 已撞毁，跳过后续判断

        # ----------------------------------------------------------------
        # 6. 到达检测 (Arrival)
        # ----------------------------------------------------------------
        if dist_to_goal < arrival_distance:
            rewards[aircraft_id] += w_arrival
            win_list += 1
            dones[aircraft_id] = True

        # ----------------------------------------------------------------
        # 7. 超时与防逃逸 (Timeout & OOB)
        # ----------------------------------------------------------------
        elif step_num >= step_max - 1:
            rewards[aircraft_id] += w_timeout

        # 动态防逃逸机制
        elif dist_to_goal > 650:
            # 惩罚逃逸，但不宜过大，以免它学会“为了不逃逸而自杀(撞机)”
            rewards[aircraft_id] += -3.0
            dones[aircraft_id] = True
            out_of_bound_count += 1

    return rewards, dones


# ... (后续工具函数保持不变) ...
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