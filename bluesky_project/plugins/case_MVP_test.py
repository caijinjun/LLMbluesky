"""
MVP算法性能测试插件
用于测试BlueSky内置的MVP（Modified Voltage Potential）冲突解决算法
在与DDPG相同的场景下运行，以便进行性能对比

使用方法:
1. 确保在settings.cfg中加载此插件
2. 运行BlueSky后启动仿真

指标对比:
- 成功率（到达目标的飞机比例）
- 碰撞次数
- 出界次数
- 平均剩余距离
"""

import os
import bluesky as bs
from bluesky import stack, settings, navdb, traf, sim, scr, tools
from geopy.distance import geodesic
import numpy as np
import time
import random
import math


def init_plugin():
    global num_ac, max_ac
    global positions, route_keeper, route_num, route_queue, choices
    global episode_num, episode_max
    global step_num, step_max
    global win_list, best_win
    global collision_count, out_of_bound_count
    global max_route_distance_static
    global last_positions  # 用于碰撞检测
    global arrival_distance
    global mvp_enabled
    global sim_started
    global min_speed, max_speed, min_alt, max_alt

    # 输出目录
    os.makedirs('output/MVP_test', exist_ok=True)

    # 碰撞参数
    collision_hor = 2.0  # km - 水平碰撞距离
    collision_ver = 500.0  # ft - 垂直碰撞距离
    arrival_distance = 8.0  # km - 到达判定距离

    collision_count = 0
    out_of_bound_count = 0
    last_positions = {}

    num_ac = 0
    max_ac = 30  # 与DDPG训练相同的飞机数量

    best_win = 0
    win_list = 0

    # [与DDPG相同] 速度和高度限制
    min_speed = 220  # 节
    max_speed = 320  # 节
    min_alt = 19500  # 英尺
    max_alt = 21000  # 英尺

    step_num = 0
    episode_max = 200  # 测试200个episode
    step_max = 1100  # 与DDPG训练相同的最大步数

    # 加载相同的航线数据
    positions = np.load('./routes/case_study_init.npy')

    # 计算最大航线距离
    all_route_dists = []
    for i in range(len(positions)):
        slat, slon, tlat, tlon = positions[i][0], positions[i][1], positions[i][2], positions[i][3]
        d = calculate_haversine_distance(slat, slon, tlat, tlon)
        all_route_dists.append(d)

    max_route_distance_static = max(all_route_dists) if all_route_dists else 0.0

    route_num = len(positions)
    route_keeper = np.zeros(max_ac, dtype=int)
    choices = [20, 25, 30]  # 与DDPG相同的飞机生成间隔
    route_queue = random.choices(choices, k=positions.shape[0])
    episode_num = 0

    mvp_enabled = False  # 标记MVP是否已启用

    print("=" * 60)
    print("🛫 MVP算法性能测试")
    print("=" * 60)
    print(f"场景参数:")
    print(f"  - 最大飞机数量: {max_ac}")
    print(f"  - 最大步数: {step_max}")
    print(f"  - 测试轮数: {episode_max}")
    print(f"  - 碰撞阈值: {collision_hor} km (水平), {collision_ver} ft (垂直)")
    print(f"  - 到达阈值: {arrival_distance} km")
    print("=" * 60)

    sim_started = False  # 标记仿真是否已启动

    config = {
        'plugin_name': 'case_MVP_test',
        'plugin_type': 'sim',
        'update_interval': 5.0,  # 与DDPG相同的更新间隔
        'update': update,
        'preupdate': preupdate,  # 预更新函数，用于启动仿真
    }

    return config, {}


def preupdate():
    """预更新函数 - 在每个仿真步骤前调用"""
    global sim_started
    
    # 只在第一次调用时启动仿真
    if not sim_started:
        stack.stack('FF')  # 快进模式
        stack.stack('DTMULT 20')  # 时间加速
        sim_started = True
        print("✅ 仿真已启动 (FF + DTMULT 20)")


def update():
    global num_ac, max_ac
    global positions, route_keeper, route_num, route_queue, choices
    global episode_num, episode_max
    global step_num, step_max
    global win_list, best_win
    global collision_count, out_of_bound_count
    global last_positions
    global arrival_distance
    global mvp_enabled
    global min_speed, max_speed, min_alt, max_alt

    current_time = bs.sim.simt

    # 在第一步启用MVP算法
    if not mvp_enabled and step_num == 0:
        stack.stack('ASAS ON')  # 启用ASAS系统
        stack.stack('RESO MVP')  # 使用MVP解决冲突
        stack.stack('RMETHH BOTH')  # 同时使用速度和航向进行水平解决
        stack.stack('RMETHV V/S')  # 使用垂直速度进行垂直解决
        mvp_enabled = True
        print("✅ ASAS + MVP 已启用")

    # 检查是否需要重置
    if step_num >= step_max:
        reset()
        return

    if num_ac == max_ac and len(traf.id) == 0:
        reset()
        return

    # 检测碰撞、出界和到达
    check_aircraft_status()

    # 创建飞机
    if num_ac < max_ac:
        if len(traf.id) == 0:
            # 初始创建所有飞机
            for i in range(len(positions)):
                lat, lon, glat, glon, h = positions[i]
                bearing_to_goal = calculate_bearing(lat, lon, glat, glon)
                # 创建飞机
                stack.stack('CRE KL{}, B737, {}, {}, {}, 20000, 250'.format(num_ac, lat, lon, h))
                stack.stack('ADDWPT KL{} {}, {}'.format(num_ac, glat, glon))
                stack.stack('HDG KL{} {}'.format(num_ac, bearing_to_goal))
                stack.stack('VNAV KL{} ON'.format(num_ac))
                stack.stack('LNAV KL{} ON'.format(num_ac))

                route_keeper[num_ac] = i
                num_ac += 1
                if num_ac == max_ac:
                    break
        else:
            # 按计划创建新飞机
            for k in range(len(route_queue)):
                if step_num == route_queue[k]:
                    lat, lon, glat, glon, h = positions[k]
                    bearing_to_goal = calculate_bearing(lat, lon, glat, glon)
                    stack.stack('CRE KL{}, B737, {}, {}, {}, 20000, 250'.format(num_ac, lat, lon, h))
                    stack.stack('ADDWPT KL{} {}, {}'.format(num_ac, glat, glon))
                    stack.stack('HDG KL{} {}'.format(num_ac, bearing_to_goal))
                    stack.stack('VNAV KL{} ON'.format(num_ac))
                    stack.stack('LNAV KL{} ON'.format(num_ac))

                    route_keeper[num_ac] = k
                    num_ac += 1
                    route_queue[k] = step_num + random.choices(choices, k=1)[0]
                    if num_ac == max_ac:
                        break

    # [与DDPG相同] 强制约束高度和速度在边界内
    for i, aircraft_id in enumerate(traf.id):
        # 检查高度是否超出边界
        current_alt = traf.alt[i] * 3.28084  # 转换为英尺
        if current_alt < min_alt:
            stack.stack('ALT {} {}'.format(aircraft_id, min_alt))
        elif current_alt > max_alt:
            stack.stack('ALT {} {}'.format(aircraft_id, max_alt))
        
        # 检查速度是否超出边界
        current_speed = traf.cas[i] * 1.9437  # 转换为节
        if current_speed < min_speed:
            stack.stack('SPD {} {}'.format(aircraft_id, min_speed))
        elif current_speed > max_speed:
            stack.stack('SPD {} {}'.format(aircraft_id, max_speed))

    step_num += 1

    # 每200步打印进度
    if step_num % 200 == 0:
        print(f"Episode {episode_num} - Step {step_num}/{step_max} - Aircraft: {len(traf.id)}")


def check_aircraft_status():
    """检查所有飞机的状态：碰撞、出界、到达"""
    global win_list, collision_count, out_of_bound_count
    global route_keeper, positions
    global arrival_distance, last_positions

    # 碰撞参数
    collision_hor = 2.0  # km
    collision_ver = 500.0  # ft

    # 偏航边界
    max_dev_dist = 25.0  # km

    aircraft_to_delete = []

    for i, aircraft_id in enumerate(traf.id):
        index = traf.id2idx(aircraft_id)
        lati, loni = traf.lat[index], traf.lon[index]
        alti = traf.alt[index] * 3.28084  # 转换为英尺

        route_idx = route_keeper[int(aircraft_id[2:])]
        start_lat, start_lon, target_lat, target_lon, _ = positions[route_idx]

        # 1. 检查到达目标
        dist_to_goal = calculate_haversine_distance(lati, loni, target_lat, target_lon)
        if dist_to_goal < arrival_distance:
            win_list += 1
            aircraft_to_delete.append(aircraft_id)
            continue

        # 2. 检查偏航出界
        cross_track_error = calculate_distance_to_line(
            lati, loni, start_lat, start_lon, target_lat, target_lon
        )
        if cross_track_error > max_dev_dist:
            out_of_bound_count += 1
            aircraft_to_delete.append(aircraft_id)
            continue

        # 3. 检查碰撞
        for j in range(i + 1, len(traf.id)):
            other_id = traf.id[j]
            other_index = traf.id2idx(other_id)
            latj, lonj = traf.lat[other_index], traf.lon[other_index]
            altj = traf.alt[other_index] * 3.28084

            dist_h = calculate_haversine_distance(lati, loni, latj, lonj)
            dist_v = abs(alti - altj)

            if dist_h < collision_hor and dist_v < collision_ver:
                collision_count += 1
                if aircraft_id not in aircraft_to_delete:
                    aircraft_to_delete.append(aircraft_id)
                if other_id not in aircraft_to_delete:
                    aircraft_to_delete.append(other_id)

    # 删除需要移除的飞机
    for ac_id in aircraft_to_delete:
        stack.stack('DEL {}'.format(ac_id))


def reset():
    global num_ac, max_ac
    global positions, route_keeper, route_queue
    global episode_num, step_num
    global win_list, best_win
    global collision_count, out_of_bound_count
    global last_positions
    global mvp_enabled
    global sim_started

    # 计算生存飞机的平均剩余距离
    surviving_distances = []
    for i, aircraft_id in enumerate(traf.id):
        lati, loni = traf.lat[i], traf.lon[i]
        route_idx = route_keeper[int(aircraft_id[2:])]
        target_lat = positions[route_idx][2]
        target_lon = positions[route_idx][3]
        dist = calculate_haversine_distance(lati, loni, target_lat, target_lon)
        surviving_distances.append(dist)

    avg_survive_dist = np.mean(surviving_distances) if len(surviving_distances) > 0 else 0.0

    # 计算成功率
    success_rate = win_list / max_ac if max_ac > 0 else 0

    print("=" * 60)
    print("Episode: {} | Success Rate: {:.2%} | Collisions: {} | OOB: {} | Avg Dist: {:.2f} km".format(
        episode_num, success_rate, collision_count, out_of_bound_count, avg_survive_dist))
    print("=" * 60)

    # 保存统计数据
    stats_dir = 'output/MVP_test'
    stats_file = os.path.join(stats_dir, 'test_stats.csv')

    file_exists = os.path.exists(stats_file)
    with open(stats_file, 'a') as f:
        if not file_exists:
            f.write("Episode,SuccessRate,Collisions,OOB,AvgRemainDist,TotalAircraft\n")
        f.write("{},{:.4f},{},{},{:.4f},{}\n".format(
            episode_num, success_rate, collision_count, out_of_bound_count, avg_survive_dist, max_ac))

    # 重置状态
    num_ac = 0
    step_num = 0
    episode_num += 1
    route_keeper = np.zeros(max_ac, dtype=int)
    route_queue = random.choices([20, 25, 30], k=positions.shape[0])
    last_positions = {}

    collision_count = 0
    out_of_bound_count = 0

    best_win = max(win_list, best_win)
    win_list = 0
    mvp_enabled = False
    sim_started = False  # 重置仿真启动标志

    # 检查是否完成所有测试
    if episode_num >= episode_max:
        print("\n" + "=" * 60)
        print("🏁 MVP测试完成！")
        print(f"结果已保存到: {stats_file}")
        print("=" * 60)
        stack.stack('STOP')
    else:
        # 重新加载场景（与DDPG相同使用multi_agent.scn）
        stack.stack('IC multi_agent.scn')


# ============== 辅助函数 ==============

def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    """计算两点之间的Haversine距离（公里）"""
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
    """计算从点1到点2的方位角"""
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


def calculate_distance_to_line(point_lat, point_lon, line_start_lat, line_start_lon,
                               line_end_lat, line_end_lon):
    """计算点到航线的垂直距离（公里）"""
    R = 6371.0

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
