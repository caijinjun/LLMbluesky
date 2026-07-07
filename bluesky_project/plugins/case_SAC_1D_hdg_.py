import os
import shutil
import bluesky as bs
from pyparsing import actions
from bluesky import stack, settings, navdb, traf, sim, scr, tools
from geopy.distance import geodesic
# from plugins.Multi_Agent.DAM_1 import Config, DynamicAgentManager
from plugins.Multi_Agent.DDPG import DDPG
from plugins.Multi_Agent.Normalizer import AircraftStateNormalizer
import numpy as np
import time
import random
import math
from typing import Dict, List, Tuple
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 运行命令"python BlueSky.py --sim --detached --scenfile DQN_3D.scn"


def init_plugin():
    # Addtional initilisation code
    # Configuration parameters
    global num_ac, max_ac, num_intruders
    global agent_manager
    global positions, route_keeper, route_num, route_queue, choices
    global episode_num, episode_max
    global step_num, step_max
    global actions, old_air_craft, current_air_craft
    global action_list, speed_list, alt_list, hdg_list
    global max_speed, min_speed, max_alt, min_alt
    global win_list, best_win, reward_list, reward_memory
    global transition_dict
    global AircraftStateNormalizer
    global written, SCN_File

    written = 0
    AircraftStateNormalizer = AircraftStateNormalizer(num_intruders=1)
    transition_dict = {
        'states': [],
        'actions': [],
        'next_states': [],
        'rewards': [],
        'dones': []
    }
    reward_memory = []

    num_ac = 0
    max_ac = 2
    num_intruders = 1
    agent_manager = DDPG(state_dim=12, intruders_dim=6, hidden_dim=128, action_dim=9, actor_lr=1e-4, critic_lr=1e-3,)
    if written == 1:
        agent_manager.load_models(f"D:\\2025\\项目+调研\\2025.6.19客机RL\\code\\Autonomous-ATC-N_Closest\\output\\DDPG\\DDPG_single\\SAC")
    reward_list = [0 for _ in range(max_ac)]

    best_win = 0
    win_list = 0
    max_speed = 300
    min_speed = 200
    max_alt = 21000
    min_alt = 20000
    speed_list = 20
    alt_list = 200
    hdg_list = 3

    step_num = 0
    episode_max = 50000
    step_max = 2000

    positions = np.load('./routes/case_study_contra.npy')
    route_num = len(positions)
    route_keeper = np.zeros(max_ac, dtype=int)
    # choices = [20, 25, 30]  # 4 minutes, 5 minutes, 6 minutes
    choices = [40, 45, 50]
    route_queue = random.choices(choices, k=positions.shape[0])
    episode_num = 0
    old_air_craft = {}
    current_air_craft = {}
    actions = {}

    if written == 1:
        SCN_File = f"D:\\2025\\项目+调研\\2025.6.19客机RL\\code\\Autonomous-ATC-N_Closest\\output\\DDPG\\DDPG_single\\scn\\{episode_num}.scn"
        shutil.copy2(
            'D:\\2025\\项目+调研\\2025.6.19客机RL\\code\\Autonomous-ATC-N_Closest\\scenario\\DQN_3D.scn',
            SCN_File)

    config = {
        # The name of your plugin
        'plugin_name': 'case_SAC_1D_hdg_',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type': 'sim',

        # Update interval in seconds. By default, your plugin's update function(s)
        # are called every timestep of the simulation. If your plugin needs less
        # frequent updates provide an update interval.
        'update_interval': 6.0,

        # The update function is called after traffic is updated. Use this if you
        # want to do things as a result of what happens in traffic. If you need to
        # something before traffic is updated please use preupdate.

        'update': update}

    # If your plugin has a state, you will probably need a reset function to
    # clear the state in between simulations.
    # 'reset':         reset
    # }

    stackfunctions = {
    }
    return config, stackfunctions


def update():
    global num_ac, max_ac, num_intruders
    global agent_manager
    global positions, route_keeper, route_num, route_queue, choices
    global episode_num, episode_max
    global step_num, step_max
    global actions, old_air_craft, current_air_craft
    global action_list, speed_list, alt_list, hdg_list
    global max_speed, min_speed, max_alt, min_alt
    global win_list, best_win, reward_list, reward_memory
    global transition_dict
    global AircraftStateNormalizer
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
        reward_list[index] = reward_list[index] * 0.99 + rewards[id]

    for i, air_craft in enumerate(traf.id):
        if air_craft in old_air_craft.keys():
            if actions[air_craft] <= 0.05:
                rewards[air_craft] = rewards[air_craft] + 0.05
            transition_dict['states'].append(old_air_craft[air_craft])
            transition_dict['actions'].append(actions[air_craft])
            transition_dict['next_states'].append(current_air_craft[air_craft])
            transition_dict['rewards'].append(rewards[air_craft])
            transition_dict['dones'].append(dones[air_craft])

    actions = {}
    for i, air_craft in enumerate(traf.id):
        state = current_air_craft[air_craft]
        actions[air_craft] = agent_manager.take_action(state)

        new_hdg = (traf.hdg[i] + actions[air_craft]*hdg_list) % 360  # 确保航向角在0-360范围内
        stack.stack('HDG {} {}'.format(air_craft, new_hdg))
        if written == 1:
            with open(SCN_File, 'a', encoding='utf-8') as f:
                f.writelines(f"{data_time}>HDG {air_craft} {new_hdg}\n")
        if dones[air_craft] == True:
            stack.stack('DEL {}'.format(air_craft))
            if written == 1:
                with open(SCN_File, 'a', encoding='utf-8') as f:
                    f.writelines(f"{data_time}>DEL {air_craft}\n")


    if num_ac < max_ac:  ## maybe spawn a/c based on time, not based on this update interval

        if len(traf.id) == 0:
            for i in range(len(positions)):
                lat,lon,glat,glon,h = positions[i]
                stack.stack('CRE KL{}, B737, {}, {}, {}, 20000, 250'.format(num_ac,lat,lon,h))
                stack.stack('ADDWPT KL{} {}, {}'.format(num_ac,glat,glon))
                stack.stack(f'VNAV KL{num_ac} ON')
                if written == 1:
                    with open(SCN_File, 'a', encoding='utf-8') as f:
                        f.writelines(f"{data_time}>CRE KL{num_ac}, B737, {lat}, {lon}, {h}, 20000, 250\n")
                        f.writelines(f"{data_time}>ADDWPT KL{num_ac} {glat}, {glon}\n")
                        f.writelines(f"{data_time}>VNAV KL{num_ac} ON\n")
                route_keeper[num_ac] = i
                num_ac += 1
                if num_ac == max_ac:
                    break

        # else:
        #     for k in range(len(route_queue)):
        #         if step_num == route_queue[k]:
        #             lat,lon,glat,glon,h = positions[k]
        #             stack.stack('CRE KL{}, B737, {}, {}, {}, 20000, 250'.format(num_ac,lat,lon,h))
        #             stack.stack('ADDWPT KL{} {}, {}'.format(num_ac,glat,glon))
        #             stack.stack(f'VNAV KL{num_ac} ON')
        #             if written == 1:
        #                 with open(SCN_File, 'a', encoding='utf-8') as f:
        #                     f.writelines(f"{data_time}>CRE KL{num_ac}, B737, {lat}, {lon}, {h}, 20000, 250\n")
        #                     f.writelines(f"{data_time}>ADDWPT KL{num_ac} {glat}, {glon}\n")
        #                     f.writelines(f"{data_time}>VNAV KL{num_ac} ON\n")
        #             route_keeper[num_ac] = k
        #             num_ac += 1
        #             route_queue[k] = step_num + random.choices(choices,k=1)[0]
        #
        #             if num_ac == max_ac:
        #                 break

    step_num += 1
    if step_num % 100 == 0:
        print(step_num)
        # for i, air in enumerate(traf.id):
        #     start_lat, start_lon, target_lat, target_lon, hdg_ori = positions[route_keeper[int(air[2:])]]
        #     print(f"飞机{air}现在经纬度为{traf.lat[i]},{traf.lon[i]},目的地经纬度{target_lat},{target_lon}")


def reset():
    global num_ac, max_ac
    global agent_manager
    global positions, route_keeper, route_queue, choices
    global episode_num, step_num
    global actions, old_air_craft, current_air_craft
    global win_list, best_win, reward_list, reward_memory
    global transition_dict
    global written, SCN_File
    num_ac = 0
    step_num = 0
    # episode_rewards.append(episode_reward)
    episode_num += 1
    route_keeper = np.zeros(max_ac, dtype=int)
    actions = {}
    old_air_craft = {}
    current_air_craft = {}
    route_queue = random.choices(choices, k=positions.shape[0])

    print("Episode: {} | Win: {} | Best Win: {} | Reward: {}".format(episode_num, win_list, best_win, np.mean(reward_list)))
    # reward_memory.append(np.mean(reward_list))
    # np.save(f"D:\\2025\\项目+调研\\2025.6.19客机RL\\code\\Autonomous-ATC-N_Closest\\output\\SAC\\SAC_hdg\\reward_memory.npy", reward_memory)
    if written == 0:
        agent_manager.update(transition_dict)
    if episode_num % 10 == 0 and written == 0:
        agent_manager.save_models(f"D:\\2025\\项目+调研\\2025.6.19客机RL\\code\\Autonomous-ATC-N_Closest\\output\\DDPG\\DDPG_single\\SAC")
    # 定期保存模型
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
        stack.stack('STOP')

    if written == 1:
        SCN_File = f"D:\\2025\\项目+调研\\2025.6.19客机RL\\code\\Autonomous-ATC-N_Closest\\output\\DDPG\\DDPG_single\\scn\\{episode_num}.scn"
        shutil.copy2(
            'D:\\2025\\项目+调研\\2025.6.19客机RL\\code\\Autonomous-ATC-N_Closest\\scenario\\DQN_3D.scn',
            SCN_File)

    stack.stack('IC DQN_3D.scn')

def get_own_state():
    global route_keeper
    own_state = {}
    for i, id in enumerate(traf.id):
        index = traf.id2idx(id)
        own_state[id] = [traf.lat[index], traf.lon[index], traf.cas[index] * 1.9439, traf.alt[index] * 3.28084, traf.hdg[index]]
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
    # 只处理有效飞机
    for i, id_i in enumerate(id):
        dist = {}
        min_distances[id_i] = []
        for j, id_j in enumerate(id):
            if i != j:
                dist[id_j] = calculate_haversine_distance(lat[i], lon[i], lat[j], lon[j])
        sorted_list = list(dict(sorted(dist.items(), key=lambda item: item[1])).keys())
        for z in range(len(sorted_list)):
            air_index = traf.id.index(sorted_list[z])
            min_distances[id_i].append([lat[air_index], lon[air_index], traf.cas[air_index] * 1.9439, traf.alt[air_index] * 3.28084, traf.hdg[air_index]])
    return min_distances


def get_rewards(stats):
    global route_keeper
    global positions
    global win_list
    # 距离阈值（公里）
    collision_distance = 10  # 碰撞距离
    warning_distance = 20  # 警告距离
    arrival_distance = 5  # 到达目标距离
    height_distance = 300

    # 奖励值
    collision_penalty = -10  # 碰撞惩罚
    arrival_reward = 100  # 到达奖励

    id = traf.id
    lon = traf.lon
    lat = traf.lat
    n_aircraft = len(id)

    # 确保返回的字典键顺序与输入idx一致
    rewards = {id_: 0.0 for id_ in id}
    dones = {id_: False for id_ in id}

    # 如果没有飞机，直接返回
    if n_aircraft == 0:
        return rewards, dones

    for i, aircraft_id in enumerate(id):
        index = traf.id2idx(aircraft_id)
        lati, loni, alti, hdgi = lat[index], lon[index], traf.alt[index] * 3.28084, traf.hdg[index]
        for j in range(len(stats[aircraft_id])):
            near_aircraft = stats[aircraft_id][j]
            latj, lonj, altj, hdgj = near_aircraft[0], near_aircraft[1], near_aircraft[3], near_aircraft[4]
            dist = calculate_haversine_distance(lati, loni, latj, lonj)
            if dist <= collision_distance and abs(alti - altj) < height_distance:
                rewards[aircraft_id] += collision_penalty
                dones[aircraft_id] = True
                break
            elif dist < warning_distance and abs(alti - altj) < height_distance:
                normalized_dist = (warning_distance - dist) / (warning_distance - collision_distance)
                # 使用指数函数计算分数，距离越大分数越负
                rewards[aircraft_id] -= (math.exp(3 * normalized_dist) - 1) / (math.exp(3) - 1)*10
                break
        start_lat, start_lon, target_lat, target_lon, hdg_ori = positions[route_keeper[int(aircraft_id[2:])]]
        init_distance = calculate_haversine_distance(start_lat, start_lon, target_lat, target_lon)
        distance_to_target = calculate_haversine_distance(lati, loni, target_lat, target_lon)
        if not dones[aircraft_id]:
            hdg_ = calculate_bearing(lati, loni, target_lat, target_lon)
            if abs(hdg_ - hdgi)<0.1:
                rewards[aircraft_id] += 0.5
            score = calculate_distance_to_line_score(lati, loni, start_lat, start_lon, target_lat, target_lon,
                                                     max_dist_km=100, steepness=3.0)  # 降低陡度
            rewards[aircraft_id] += score
            if score == -10:
                dones[aircraft_id] = True
        if not dones[aircraft_id]:
            if distance_to_target < arrival_distance:
                rewards[aircraft_id] += arrival_reward
                win_list += 1
                dones[aircraft_id] = True
            else:
                rewards[aircraft_id] -= (distance_to_target/init_distance)*3
    return rewards, dones

def clamp(value, min_val, max_val):
    """将值限制在最小值和最大值之间"""
    return max(min_val, min(value, max_val))


def calculate_distance_to_line_score(point_lat, point_lon, line_start_lat, line_start_lon,
                                     line_end_lat, line_end_lon, min_dist_km=0, max_dist_km=50,
                                     steepness=3.0):
    """
    计算点到航线距离的分数

    参数:
        point_lat, point_lon: 点的经纬度
        line_start_lat, line_start_lon: 航线起点的经纬度
        line_end_lat, line_end_lon: 航线终点的经纬度
        min_dist_km: 最小距离阈值(km)，小于此距离分数为0
        max_dist_km: 最大距离阈值(km)，大于等于此距离分数为-1
        steepness: 陡度参数，越大曲线越陡峭

    返回:
        score: 距离分数，范围[-1, 0]
    """
    # 计算点到航线的距离(km)
    distance_km = calculate_distance_to_line(
        point_lat, point_lon,
        line_start_lat, line_start_lon,
        line_end_lat, line_end_lon
    )

    # 应用分数函数
    if distance_km <= min_dist_km:
        return 0.0
    elif distance_km >= max_dist_km:
        return -10.0
    else:
        # 归一化距离到[0,1]范围
        normalized_dist = (distance_km - min_dist_km) / (max_dist_km - min_dist_km)
        # 使用指数函数计算分数，距离越大分数越负
        return -((math.exp(steepness * normalized_dist) - 1) / (math.exp(steepness) - 1)) * 10


def calculate_distance_to_line(point_lat, point_lon, line_start_lat, line_start_lon,
                               line_end_lat, line_end_lon):
    """
    计算点到航线的最短距离(km)

    使用球面几何计算点到大圆航线的最短距离
    """
    # 地球半径(km)
    R = 6371.0

    # 将角度转换为弧度
    lat1 = math.radians(point_lat)
    lon1 = math.radians(point_lon)
    lat2 = math.radians(line_start_lat)
    lon2 = math.radians(line_start_lon)
    lat3 = math.radians(line_end_lat)
    lon3 = math.radians(line_end_lon)

    # 计算点到航线起点的距离
    d_start = calculate_haversine_distance(point_lat, point_lon, line_start_lat, line_start_lon)

    # 计算点到航线终点的距离
    d_end = calculate_haversine_distance(point_lat, point_lon, line_end_lat, line_end_lon)

    # 计算航线长度
    line_length = calculate_haversine_distance(line_start_lat, line_start_lon, line_end_lat, line_end_lon)

    # 如果航线长度接近0，直接返回到起点的距离
    if line_length < 1e-6:
        return d_start

    # 计算点到航线的垂直距离
    # 使用球面三角公式计算跨轨距离

    # 计算方位角
    bearing_start_to_end = calculate_bearing(line_start_lat, line_start_lon, line_end_lat, line_end_lon)
    bearing_start_to_point = calculate_bearing(line_start_lat, line_start_lon, point_lat, point_lon)

    # 计算角度差
    angle_diff = math.radians(abs(bearing_start_to_point - bearing_start_to_end))

    # 计算垂直距离
    cross_track_distance = math.asin(math.sin(d_start / R) * math.sin(angle_diff)) * R

    # 计算沿航线的距离
    along_track_distance = math.acos(math.cos(d_start / R) / math.cos(cross_track_distance / R)) * R

    # 检查垂足是否在线段上
    if along_track_distance > line_length or along_track_distance < 0:
        # 垂足不在线段上，返回到最近端点的距离
        return min(d_start, d_end)
    else:
        return abs(cross_track_distance)


def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    """
    计算两点间的大圆距离(km)
    """
    R = 6371.0  # 地球半径(km)

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
    """
    计算从点1到点2的真方位角(度)
    """
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


import math


def angle_diff_basic(angle1, angle2):
    """
    计算两个角度之间的最小夹角差（基础方法）

    参数:
        angle1, angle2: 角度值（度）

    返回:
        最小夹角差（度）
    """
    angle1 = (angle1+360) % 360
    angle2 = (angle2+360) % 360
    # 计算直接差值
    diff = abs(angle1 - angle2) % 360

    # 考虑角度循环性，取最小夹角
    return min(diff, 360 - diff)
