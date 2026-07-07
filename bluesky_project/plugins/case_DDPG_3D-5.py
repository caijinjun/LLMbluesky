import os
import shutil
import bluesky as bs
from bluesky import stack, settings, navdb, traf, sim, scr, tools
from geopy.distance import geodesic
from plugins.Multi_Agent.DDPG_3DFive import DDPG
from plugins.Multi_Agent.Normalizer import AircraftStateNormalizer
import numpy as np
import time
import random
import math
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def ternary_bucket(x, thresh=0.33):
    """Map continuous [-1,1] to {-1,0,1} buckets."""
    if x < -thresh:
        return -1
    if x > thresh:
        return 1
    return 0


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
    global last_goal_distance
    global collision_count, out_of_bound_count
    global max_route_distance_static
    global written, SCN_File

    written = 0  # 1 = record on, 0 = off

    os.makedirs('output/DDPG/DDPG3D-5/scenarios', exist_ok=True)

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
    agent_manager = DDPG(state_dim=12, intruders_dim=18, hidden_dim=128, action_dim=3)
    reward_list = [0 for _ in range(max_ac)]

    best_win = 0
    win_list = 0
    min_speed = 200
    max_speed = 320
    min_alt = 15000
    max_alt = 25000
    speed_choices = [-20.0, 0.0, 20.0]
    alt_choices = [-300.0, 0.0, 300.0]
    hdg_choices = [-3.0, 0.0, 3.0]

    step_num = 0
    episode_max = 8000
    step_max = 800

    positions = np.load('./routes/case_study_init.npy')

    all_route_dists = []
    for i in range(len(positions)):
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
    route_num = len(positions)
    route_keeper = np.zeros(max_ac, dtype=int)
    choices = [20, 25, 30]
    route_queue = random.choices(choices, k=positions.shape[0])
    episode_num = 0
    old_air_craft = {}
    current_air_craft = {}
    actions = {}

    if written == 1:
        SCN_File = f"output/DDPG/DDPG3D-5/scenarios/{episode_num}.scn"

        if os.path.exists('multi_agent.scn'):
            shutil.copy2('multi_agent.scn', SCN_File)
        else:
            open(SCN_File, 'w').close()

    config = {
        'plugin_name': 'case_DDPG_3D-5',
        'plugin_type': 'sim',
        'update_interval': 8.0,
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
    global last_goal_distance

    global collision_count
    global out_of_bound_count

    global written, SCN_File

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

    for i, air_craft in enumerate(traf.id):
        if air_craft in old_air_craft.keys():
            transition_dict['states'].append(old_air_craft[air_craft])
            transition_dict['actions'].append(actions.get(air_craft, np.zeros(3)))
            transition_dict['next_states'].append(current_air_craft[air_craft])
            transition_dict['rewards'].append(rewards[air_craft])
            transition_dict['dones'].append(dones[air_craft])

    actions = {}
    for i, air_craft in enumerate(traf.id):
        state = current_air_craft[air_craft]

        raw_action = agent_manager.take_action(state)
        sp_sig = ternary_bucket(raw_action[0])
        alt_sig = ternary_bucket(raw_action[1])
        hdg_sig = ternary_bucket(raw_action[2])

        delta_speed = speed_choices[sp_sig + 1]
        delta_alt = alt_choices[alt_sig + 1]
        delta_hdg = hdg_choices[hdg_sig + 1]

        actions[air_craft] = np.array([sp_sig, alt_sig, hdg_sig], dtype=np.float32)

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
            last_goal_distance.pop(air_craft, None)

            if written == 1:
                with open(SCN_File, 'a', encoding='utf-8') as f:
                    f.write(f"{data_time}>DEL {air_craft}\n")

    if num_ac < max_ac:
        if len(traf.id) == 0:
            for i in range(len(positions)):
                lat, lon, glat, glon, h = positions[i]
                stack.stack('CRE KL{}, B737, {}, {}, {}, 20000, 250'.format(num_ac, lat, lon, h))
                stack.stack('ADDWPT KL{} {}, {}'.format(num_ac, glat, glon))
                stack.stack(f'VNAV KL{num_ac} ON')

                if written == 1:
                    with open(SCN_File, 'a', encoding='utf-8') as f:
                        f.write(f"{data_time}>CRE KL{num_ac}, B737, {lat}, {lon}, {h}, 20000, 250\n")
                        f.write(f"{data_time}>ADDWPT KL{num_ac} {glat}, {glon}\n")
                        f.write(f"{data_time}>VNAV KL{num_ac} ON\n")

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

                    if written == 1:
                        with open(SCN_File, 'a', encoding='utf-8') as f:
                            f.write(f"{data_time}>CRE KL{num_ac}, B737, {lat}, {lon}, {h}, 20000, 250\n")
                            f.write(f"{data_time}>ADDWPT KL{num_ac} {glat}, {glon}\n")
                            f.write(f"{data_time}>VNAV KL{num_ac} ON\n")

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

    global written, SCN_File

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

    stats_dir = 'output/DDPG/DDPG3D-5'
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

    agent_manager.update(transition_dict)

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
    if episode_num % 10 == 0:
        agent_manager.save_models(f"output\\DDPG\\DDPG3D-5\\DDPG")

    best_win = max(win_list, best_win)
    win_list = 0
    if episode_num == episode_max:
        stack.stack('STOP')

    if written == 1:

        SCN_File = f"output/DDPG/DDPG3D-5/scenarios/{episode_num}.scn"

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
    global route_keeper, positions, win_list, last_goal_distance
    global collision_count, step_num, step_max, out_of_bound_count

    # ==========================================
    # 1. 基础权重 (针对漏洞二：大幅提升到达奖励)
    # ==========================================
    w_collision = -50.0

    # 【修复2】提升到达奖励，确保抵消过程中的侵入惩罚
    # 即使在警告区飞了5步(-75)，到达(+100)依然是赚的，保证它有动力冲线
    w_arrival = 100.0

    w_progress = 0.2
    w_heading = 0.005
    w_timeout = -10.0
    w_step_cost = -0.01

    # ==========================================
    # 2. 航线与边界
    # ==========================================
    w_deviation_linear = -0.01
    max_dev_dist = 30.0
    w_out_of_corridor = -45.0
    soft_boundary_dist = 20.0

    # ==========================================
    # 3. 避撞参数 (保持 8s/3km 配置)
    # ==========================================
    collision_hor = 3.0
    collision_ver = 900.0

    warning_hor = 12.0
    warning_ver = 2000.0
    w_intrusion = -15.0

    # 【修复3】稍微扩大到达判定，缓解终点拥堵
    arrival_distance = 6.0

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

        # 偏航衰减因子 (只作用于 Progress)
        dev_factor = max(0.0, 1.0 - (cross_track_error / soft_boundary_dist))

        # --- 1. 任务进度 ---
        dist_to_goal = calculate_haversine_distance(lati, loni, target_lat, target_lon)
        prev_dist = last_goal_distance.get(aircraft_id, dist_to_goal)
        last_goal_distance[aircraft_id] = dist_to_goal

        # Progress 依然受 dev_factor 限制，偏离了就不给前进分
        raw_progress = clamp(prev_dist - dist_to_goal, -2.0, 2.0)
        rewards[aircraft_id] += w_progress * raw_progress * dev_factor

        # --- 2. 航向对齐 ---
        # 【修复1】解绑 Heading 和 dev_factor
        # 即使在 25km 处 (dev_factor=0)，我们也希望它因“指向终点”而获得正反馈
        # 这能引导它主动调头，而不是被动等待惩罚
        bearing_to_goal = calculate_bearing(lati, loni, target_lat, target_lon)
        angle_diff = abs(bearing_to_goal - hdgi)
        angle_diff = min(angle_diff, 360 - angle_diff)

        # 无论在哪里，只要指向目标，就给微小的奖励 (0.005)
        rewards[aircraft_id] += w_heading * math.cos(math.radians(angle_diff))

        # --- 3. 基础消耗 ---
        rewards[aircraft_id] += w_step_cost

        # --- 4. 偏航惩罚 (组合拳) ---
        # 线性惩罚
        rewards[aircraft_id] += cross_track_error * w_deviation_linear

        # 边界势场 (Soft Boundary)
        if cross_track_error > soft_boundary_dist:
            excess = cross_track_error - soft_boundary_dist
            boundary_penalty = -0.1 * (excess ** 2)
            rewards[aircraft_id] += boundary_penalty

        # 硬性出界
        if cross_track_error > max_dev_dist:
            rewards[aircraft_id] += w_out_of_corridor
            dones[aircraft_id] = True
            out_of_bound_count += 1
            continue

            # --- 5. 避撞逻辑 ---
        collision_flag = False
        if aircraft_id in stats:
            for near_ac in stats[aircraft_id]:
                latj, lonj, altj = near_ac[0], near_ac[1], near_ac[3]
                dist_h = calculate_haversine_distance(lati, loni, latj, lonj)
                dist_v = abs(alti - altj)

                # 硬性碰撞
                if dist_h <= collision_hor and dist_v < collision_ver:
                    rewards[aircraft_id] += w_collision
                    dones[aircraft_id] = True
                    collision_count += 1
                    collision_flag = True
                    break

                    # 柔性警告
                norm_h = dist_h / warning_hor
                norm_v = dist_v / warning_ver
                ellipsoid_dist = math.sqrt(norm_h ** 2 + norm_v ** 2)
                if ellipsoid_dist < 1.0:
                    intrusion = 1.0 - ellipsoid_dist
                    rewards[aircraft_id] += w_intrusion * intrusion
        if collision_flag: continue

        # --- 6. 到达检测 ---
        if dist_to_goal < arrival_distance:
            rewards[aircraft_id] += w_arrival
            win_list += 1
            dones[aircraft_id] = True
        elif step_num >= step_max - 1:
            rewards[aircraft_id] += w_timeout

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



