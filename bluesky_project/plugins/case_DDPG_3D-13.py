import os
import shutil
import bluesky as bs
from bluesky import stack, settings, navdb, traf, sim, scr, tools
from geopy.distance import geodesic
from plugins.Multi_Agent.DDPG_3DElevn import DDPG
from plugins.Multi_Agent.OptimizedNormalizer import OptimizedAircraftNormalizer
import numpy as np
import time
import random
import math
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# [Off-Policy] 全局变量：追踪已传递给DDPG的数据位置
last_transmitted_idx = 0


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
    global normalizer  # [\u4f18\u5316] \u65b0\u7684\u4f18\u5316Normalizer
    # \u65b0\u589e last_along_track
    global last_goal_distance, last_along_track
    global boundary_strikes
    global last_vertical_distances  # 新增：记录上一步的垂直距离
    global collision_count, out_of_bound_count
    global max_route_distance_static
    global written, SCN_File
    global obs_dim
    global route_assignment_counter  # [优化] 航线轮询分配计数器

    written = 0  # [训练模式] 0=关闭场景记录以提高训练速度

    os.makedirs('output/DDPG/DDPG3D-22/scenarios', exist_ok=True)

    collision_count = 0
    out_of_bound_count = 0

    # [优化] 使用新的优化Normalizer (29维输出)
    normalizer = OptimizedAircraftNormalizer(num_intruders=3)

    transition_dict = {
        'joint_states': [],
        'joint_next_states': [],
        'joint_actions': [],
        'joint_rewards': [],
        'joint_dones': [],
        'joint_masks': [],
        'joint_presence_masks': []
    }
    
    # [Off-Policy] 追踪已传递位置，避免重复传递历史数据
    last_transmitted_idx = 0
    
    reward_memory = []

    # 初始化状态字典
    last_goal_distance = {}
    last_along_track = {}
    boundary_strikes = {}
    last_vertical_distances = {}  # 新增：初始化垂直距离记录

    num_ac = 0
    max_ac = 40
    num_intruders = 3
    # [优化] 新的状态空间: 自我+目标8维 + 入侵者21维 = 29维
    obs_dim = 29
    agent_manager = DDPG(state_dim=6, intruders_dim=18, hidden_dim=256, action_dim=3,
                         max_agents=max_ac)  # state_dim=6原始, intruders_dim=18原始(3×6)

    # [训练模式] 从头训练新模型（航向引导改进版）
    # model_path = "output/DDPG/DDPG3D-22/DDPG_ep_latest"  # 如需继续训练，取消注释
    
    # 从头训练
    if False:  # 改为True可加载模型继续训练
        agent_manager.load_models(model_path)
        print("="*50)
        print("🔄 继续训练模式")
        print(f"📥 模型路径: {model_path}")
        print("📊 状态空间: 8维（航向引导改进）")
        print("💾 场景记录: 已开启")
        print("="*50)
    else:
        print("="*50)
        print("⚠️  未找到模型文件,从头开始训练")
        print("📊 状态空间: 29维 (自我+目标8维 + 入侵者21维)")
        print("✨ 特性: 机体坐标系 + 极坐标 + 相对化")
        print("="*50)
        
    reward_list = [0 for _ in range(max_ac)]

    best_win = 0
    win_list = 0
    min_speed = 220
    max_speed = 320
    min_alt = 19000
    max_alt = 21500

    step_num = 0
    episode_max = 5000  # [训练模式] 训练5000个episode
    step_max = 1000

    # ==========================================
    # [新设计] 矩形空域航线生成
    # ==========================================
    def generate_rectangular_routes(center_lat=32.5, center_lon=117.5,
                                    width_km=300, height_km=200,
                                    num_routes=5, min_distance_km=200):
        """
        生成基于矩形空域的航线
        
        参数:
            center_lat, center_lon: 矩形中心坐标
            width_km: 矩形宽度 (东西方向, km)
            height_km: 矩形高度 (南北方向, km)
            num_routes: 航线数量 (默认5条)
            min_distance_km: 最小航线距离 (km)
        
        返回:
            boundary_points: 边界点数组 [(纬度, 经度), ...]
            routes: 航线数组 [起点纬度, 起点经度, 终点纬度, 终点经度, 初始航向]
        """
        
        # 1. 计算矩形的4个角点
        # 北边界
        north_lat, _ = calculate_destination(center_lat, center_lon, 0, height_km / 2)
        # 南边界
        south_lat, _ = calculate_destination(center_lat, center_lon, 180, height_km / 2)
        # 东边界
        _, east_lon = calculate_destination(center_lat, center_lon, 90, width_km / 2)
        # 西边界
        _, west_lon = calculate_destination(center_lat, center_lon, 270, width_km / 2)
        
        # 2. 在矩形的4条边上生成采样点
        points_per_edge = 5  # 每条边5个点
        boundary_points = []
        
        # 北边 (从西到东)
        for i in range(points_per_edge):
            ratio = i / (points_per_edge - 1)
            lon = west_lon + ratio * (east_lon - west_lon)
            boundary_points.append((north_lat, lon, 'N'))
        
        # 东边 (从北到南)
        for i in range(1, points_per_edge):  # 跳过第一个点(已在北边)
            ratio = i / (points_per_edge - 1)
            lat = north_lat + ratio * (south_lat - north_lat)
            boundary_points.append((lat, east_lon, 'E'))
        
        # 南边 (从东到西)
        for i in range(1, points_per_edge):
            ratio = i / (points_per_edge - 1)
            lon = east_lon + ratio * (west_lon - east_lon)
            boundary_points.append((south_lat, lon, 'S'))
        
        # 西边 (从南到北)
        for i in range(1, points_per_edge - 1):  # 跳过首尾(已在南/北边)
            ratio = i / (points_per_edge - 1)
            lat = south_lat + ratio * (north_lat - south_lat)
            boundary_points.append((lat, west_lon, 'W'))
        
        # 3. 交叉点计算辅助函数
        def calculate_line_intersection(p1, p2, p3, p4):
            """计算两条线段是否相交"""
            x1, y1 = p1[1], p1[0]  # 经度,纬度
            x2, y2 = p2[1], p2[0]
            x3, y3 = p3[1], p3[0]
            x4, y4 = p4[1], p4[0]
            
            denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
            if abs(denom) < 1e-10:
                return None  # 平行或重合
            
            t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
            u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / denom
            
            if 0 < t < 1 and 0 < u < 1:  # 线段内部相交
                ix = x1 + t*(x2-x1)
                iy = y1 + t*(y2-y1)
                return (iy, ix)  # 返回纬度,经度
            return None
        
        def count_intersections(routes_list):
            """计算所有航线的交叉点数量"""
            count = 0
            for i in range(len(routes_list)):
                for j in range(i+1, len(routes_list)):
                    r1 = routes_list[i]
                    r2 = routes_list[j]
                    intersection = calculate_line_intersection(
                        (r1[0], r1[1]), (r1[2], r1[3]),
                        (r2[0], r2[1]), (r2[2], r2[3])
                    )
                    if intersection:
                        count += 1
            return count
        
        # 4. 生成满足条件的航线 (4-10个交叉点)
        target_min = 4
        target_max = 10
        max_attempts = 200
        best_routes = None
        best_count = 0
        
        for attempt in range(max_attempts):
            routes = []
            used_points = set()
            
            # 随机生成num_routes条航线
            for _ in range(num_routes):
                attempts_per_route = 50
                for _ in range(attempts_per_route):
                    # 随机选择起点和终点
                    start_idx = np.random.randint(0, len(boundary_points))
                    end_idx = np.random.randint(0, len(boundary_points))
                    
                    # 避免同一个点或重复使用
                    if start_idx == end_idx or start_idx in used_points or end_idx in used_points:
                        continue
                    
                    start_lat, start_lon, _ = boundary_points[start_idx]
                    end_lat, end_lon, _ = boundary_points[end_idx]
                    
                    # 检查距离
                    dist = calculate_haversine_distance(start_lat, start_lon, end_lat, end_lon)
                    
                    if dist >= min_distance_km:
                        initial_hdg = calculate_bearing(start_lat, start_lon, end_lat, end_lon)
                        routes.append([start_lat, start_lon, end_lat, end_lon, initial_hdg])
                        used_points.add(start_idx)
                        used_points.add(end_idx)
                        break
            
            # 如果没有生成足够的航线,跳过
            if len(routes) < num_routes:
                continue
            
            # 计算交叉点数量
            num_intersections = count_intersections(routes)
            
            # 检查是否在目标范围
            if target_min <= num_intersections <= target_max:
                best_routes = routes
                best_count = num_intersections
                print(f"  ✓ 找到满足条件的方案: {num_intersections}个交叉点")
                break
            
            # 记录最接近的方案
            if best_routes is None or abs(num_intersections - 7) < abs(best_count - 7):
                best_routes = routes
                best_count = num_intersections
        
        # 如果没找到完美方案,使用最接近的
        routes = best_routes if best_routes else []
        
        if len(routes) < num_routes:
            print(f"⚠️  警告: 仅生成 {len(routes)} 条航线")
        
        # 统计信息
        if routes:
            route_lengths = [calculate_haversine_distance(r[0], r[1], r[2], r[3]) for r in routes]
            num_intersections = count_intersections(routes)
            print(f"  📊 矩形空域统计:")
            print(f"     尺寸: {width_km}km × {height_km}km")
            print(f"     航线数: {len(routes)}条")
            print(f"     交叉点: {num_intersections}个")
            print(f"     航线长度: 平均{np.mean(route_lengths):.1f}km, "
                  f"最短{np.min(route_lengths):.1f}km, 最长{np.max(route_lengths):.1f}km")
        
        # 返回边界点(仅纬度经度)和航线
        boundary_coords = [(lat, lon) for lat, lon, _ in boundary_points]
        return np.array(boundary_coords), np.array(routes)


    # 生成矩形空域航线
    _, positions = generate_rectangular_routes(
        center_lat=32.5, 
        center_lon=117.5,
        width_km=300,
        height_km=200,
        num_routes=5,
        min_distance_km=200
    )
    print(f"✈️  生成了 {len(positions)} 条测试航线")


    all_route_dists = []
    for i in range(len(positions)):
        slat, slon, tlat, tlon = positions[i][0], positions[i][1], positions[i][2], positions[i][3]
        d = calculate_haversine_distance(slat, slon, tlat, tlon)
        all_route_dists.append(d)


    max_route_distance_static = max(all_route_dists) if all_route_dists else 0.0
    avg_route_distance_static = np.mean(all_route_dists) if all_route_dists else 0.0
    print(f"📏 航线距离: 平均 {avg_route_distance_static:.2f} km, 最大 {max_route_distance_static:.2f} km")

    route_num = len(positions)
    route_keeper = np.zeros(max_ac, dtype=int)
    choices = [20, 25, 30]
    route_queue = random.choices(choices, k=positions.shape[0])
    episode_num = 0
    old_air_craft = {}
    current_air_craft = {}
    actions = {}
    
    # [新增] 初始化固定场景变量(用于课程学习)
    fixed_positions = positions.copy()  # 保存初始场景作为固定场景
    
    # [优化] 航线轮询分配器 - 确保飞机均匀分配到每条航线（30架→5条航线，每条6架）
    route_assignment_counter = 0

    if written == 1:
        SCN_File = f"output/DDPG/DDPG3D-22/scenarios/{episode_num}.scn"

        if os.path.exists('multi_agent.scn'):
            shutil.copy2('multi_agent.scn', SCN_File)
        else:
            open(SCN_File, 'w').close()

    config = {
        'plugin_name': 'case_DDPG_3D-13',
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
    global normalizer  #
    global last_goal_distance, last_along_track
    global boundary_strikes
    global last_transmitted_idx  # [Off-Policy] 追踪已传递位置

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
    current_air_craft = normalizer.normalize_complete_state(own_state, min_dis_craft)
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

        # [训练模式] 使用episode_num启用探索噪声
        raw_action = agent_manager.take_action(state, episode_num)
        actions[air_craft] = raw_action.copy()

        delta_speed = raw_action[0] * 15  # [修改] 从10增加到15
        delta_alt = raw_action[1] * 200
        # 航向权限 15
        delta_hdg = raw_action[2] * 12

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

    # [修改] 间隔创建飞机，避免起点拥挤
    if num_ac < max_ac:
        for k in range(len(route_queue)):
            if step_num == route_queue[k]:
                route_idx = np.random.randint(0, len(positions))
                lat, lon, glat, glon, h = positions[route_idx]
                bearing_to_goal = calculate_bearing(lat, lon, glat, glon)
                
                # 随机初始高度和速度（增加训练多样性）
                init_alt = np.random.randint(min_alt, max_alt + 1)  # 19000-21500 ft
                init_speed = np.random.randint(min_speed, max_speed + 1)  # 220-320 kts
                
                stack.stack('CRE KL{}, B737, {}, {}, {}, {}, {}'.format(num_ac, lat, lon, h, init_alt, init_speed))
                stack.stack('ADDWPT KL{} {}, {}'.format(num_ac, glat, glon))
                stack.stack('HDG KL{} {}'.format(num_ac, bearing_to_goal))
                stack.stack(f'VNAV KL{num_ac} ON')

                if written == 1:
                    with open(SCN_File, 'a', encoding='utf-8') as f:
                        f.write(f"{data_time}>CRE KL{num_ac}, B737, {lat}, {lon}, {h}, {init_alt}, {init_speed}\n")
                        f.write(f"{data_time}>ADDWPT KL{num_ac} {glat}, {glon}\n")
                        f.write(f"{data_time}>HDG KL{num_ac} {bearing_to_goal}\n")
                        f.write(f"{data_time}>VNAV KL{num_ac} ON\n")

                route_keeper[num_ac] = route_idx
                num_ac += 1
                route_queue[k] = step_num + random.choices(choices, k=1)[0]
                if num_ac == max_ac:
                    break

    step_num += 1


    if step_num > 0 and step_num % 200 == 0:
         if len(transition_dict['joint_states']) >= 256:
             agent_manager.update(transition_dict, episode_num)

    if step_num % 200 == 0:
        buffer_size = len(transition_dict['joint_states'])
        print(f"Step {step_num} | 外部收集: {buffer_size} steps")


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
    global route_assignment_counter  # [优化] 航线分配计数器

    global collision_count
    global out_of_bound_count

    global written, SCN_File
    global obs_dim
    global fixed_positions  # [新增] 课程学习固定场景

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

    stats_dir = 'output/DDPG/DDPG3D-22'
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

    # [Off-Policy] 已在update()中每5步训练,此处不再重复调用
    # (经验会持续积累,不需要在episode结束时特殊处理)

    num_ac = 0
    step_num = 0
    episode_num += 1
    route_keeper = np.zeros(max_ac, dtype=int)
    actions = {}
    old_air_craft = {}
    current_air_craft = {}
    route_queue = [np.random.randint(10, 41) for _ in range(positions.shape[0])]
    # 随机间隔：10-40步 = 50-200秒，增加场景多样性

    # 必须重置状态追踪字典
    last_goal_distance = {}
    last_along_track = {}
    boundary_strikes = {}
    last_vertical_distances = {}  # 新增：重置垂直距离记录


    collision_count = 0
    out_of_bound_count = 0

    # [Off-Policy] 不清空经验池,完全依赖DDPG内部的500000步buffer管理
    # transition_dict持续积累,DDPG.update()会自动存入内部buffer并管理
    
    reward_list = [0 for _ in range(max_ac)]
    
    # [训练模式] 定期保存模型
    if episode_num % 300 == 0 and episode_num > 0:
        model_save_path = f"output/DDPG/DDPG3D-22/DDPG_ep{episode_num}"
        agent_manager.save_models(model_save_path)
        print(f"💾 模型已保存: {model_save_path}")

    best_win = max(win_list, best_win)
    win_list = 0
    if episode_num == episode_max:
        stack.stack('STOP')

    if written == 1:
        SCN_File = f"output/DDPG/DDPG3D-22/scenarios/{episode_num}.scn"

        if os.path.exists('multi_agent.scn'):
            shutil.copy2('multi_agent.scn', SCN_File)
        else:
            open(SCN_File, 'w').close()

    # ==========================================
    # [新增] 课程学习: 前500轮固定场景,之后随机场景
    # ==========================================
    def generate_intelligent_crossing_routes(center_lat=32.5, center_lon=117.5, 
                                             width_km=300, height_km=200, num_routes=5):
        """
        智能交叉航线生成（矩形空域版本）：确保4-10个交叉点
        
        设计理念：
        - 矩形空域，从边界随机选择起点和终点
        - 控制交叉点数量在4-10个之间
        - 支持全球随机中心坐标
        """
        
        # 1. 生成矩形空域（全球随机中心）
        # [优化] 中心点完全随机（位置无关训练）
        actual_center_lat = np.random.uniform(-60, 60)   # 避免极地
        actual_center_lon = np.random.uniform(-180, 180)  # 全球任意经度
        
        # 计算矩形的4个边界
        north_lat, _ = calculate_destination(actual_center_lat, actual_center_lon, 0, height_km / 2)
        south_lat, _ = calculate_destination(actual_center_lat, actual_center_lon, 180, height_km / 2)
        _, east_lon = calculate_destination(actual_center_lat, actual_center_lon, 90, width_km / 2)
        _, west_lon = calculate_destination(actual_center_lat, actual_center_lon, 270, width_km / 2)
        
        # 2. 在矩形的4条边上生成采样点
        points_per_edge = 5
        boundary_points = []
        
        # 北边 (从西到东)
        for i in range(points_per_edge):
            ratio = i / (points_per_edge - 1)
            lon = west_lon + ratio * (east_lon - west_lon)
            boundary_points.append((north_lat, lon))
        
        # 东边 (从北到南)
        for i in range(1, points_per_edge):
            ratio = i / (points_per_edge - 1)
            lat = north_lat + ratio * (south_lat - north_lat)
            boundary_points.append((lat, east_lon))
        
        # 南边 (从东到西)
        for i in range(1, points_per_edge):
            ratio = i / (points_per_edge - 1)
            lon = east_lon + ratio * (west_lon - east_lon)
            boundary_points.append((south_lat, lon))
        
        # 西边 (从南到北)
        for i in range(1, points_per_edge - 1):
            ratio = i / (points_per_edge - 1)
            lat = south_lat + ratio * (north_lat - south_lat)
            boundary_points.append((lat, west_lon))
        
        # 3. 交叉点计算辅助函数
        def calculate_line_intersection(p1, p2, p3, p4):
            """计算两条线段是否相交"""
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            x4, y4 = p4
            
            denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
            if abs(denom) < 1e-10:
                return None  # 平行或重合
            
            t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
            u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / denom
            
            if 0 < t < 1 and 0 < u < 1:  # 线段内部相交
                ix = x1 + t*(x2-x1)
                iy = y1 + t*(y2-y1)
                return (ix, iy)
            return None
        
        def count_intersections(routes_list):
            """计算所有航线的交叉点数量"""
            intersections = []
            for i in range(len(routes_list)):
                for j in range(i+1, len(routes_list)):
                    r1 = routes_list[i]
                    r2 = routes_list[j]
                    intersection = calculate_line_intersection(
                        (r1[1], r1[0]), (r1[3], r1[2]),  # 经纬度转xy
                        (r2[1], r2[0]), (r2[3], r2[2])
                    )
                    if intersection:
                        intersections.append(intersection)
            return len(intersections), intersections
        
        # 4. 尝试生成满足条件的航线（4-10个交叉点）
        max_attempts = 100
        best_routes = None
        best_count = 0
        target_min = 4
        target_max = 10
        min_distance_km = 200
        
        n = len(boundary_points)
        
        for attempt in range(max_attempts):
            routes = []
            used_points = set()
            
            # 随机生成num_routes条航线
            for _ in range(num_routes):
                attempts_per_route = 50
                for _ in range(attempts_per_route):
                    start_idx = np.random.randint(0, n)
                    end_idx = np.random.randint(0, n)
                    
                    # 避免同一个点或重复使用
                    if start_idx == end_idx or start_idx in used_points or end_idx in used_points:
                        continue
                    
                    route_start = boundary_points[start_idx]
                    route_end = boundary_points[end_idx]
                    
                    # 检查航线长度
                    route_length = calculate_haversine_distance(
                        route_start[0], route_start[1],
                        route_end[0], route_end[1]
                    )
                    
                    # 如果航线满足最小距离要求
                    if route_length >= min_distance_km:
                        routes.append([route_start[0], route_start[1], 
                                      route_end[0], route_end[1],
                                      calculate_bearing(route_start[0], route_start[1], 
                                                      route_end[0], route_end[1])])
                        used_points.add(start_idx)
                        used_points.add(end_idx)
                        break
            
            # 如果没有生成完整num_routes条航线，跳过这次尝试
            if len(routes) < num_routes:
                continue
            
            # 计算交叉点
            num_intersections, intersections = count_intersections(routes)
            
            # 如果在目标范围内，直接使用
            if target_min <= num_intersections <= target_max:
                best_routes = routes
                best_count = num_intersections
                break
            
            # 记录最接近目标的方案
            if best_routes is None or abs(num_intersections - 7) < abs(best_count - 7):
                best_routes = routes
                best_count = num_intersections
        
        # 如果100次尝试都不满足，使用最接近的方案
        routes = best_routes if best_routes else []
        
        # ==========================================
        # 统计信息打印
        # ==========================================
        
        # 1. 计算航线长度
        route_lengths = []
        for route in routes:
            length = calculate_haversine_distance(route[0], route[1], route[2], route[3])
            route_lengths.append(length)
        
        # 2. 计算交叉点信息
        num_intersections, intersections = count_intersections(routes)
        
        # 打印统计信息
        print(f"  📊 场景统计:")
        print(f"     交叉点: {num_intersections}个")
        
        if route_lengths:
            print(f"     航线长度: 平均{np.mean(route_lengths):.1f}km, "
                  f"最短{np.min(route_lengths):.1f}km, "
                  f"最长{np.max(route_lengths):.1f}km")
        
        return np.array(boundary_points), np.array(routes)
        
        return np.array(boundary_points), np.array(routes)

    
    # 课程学习策略 - 两阶段设计（简化版）
    CURRICULUM_FIXED_END = 100       # 固定场景结束
    # Episode 1-100: 固定场景
    # Episode 101+: 完全随机场景
    
    if episode_num == 1:
        # Episode 1: 生成初始固定场景并保存
        _, positions = generate_intelligent_crossing_routes(
            width_km=300,
            height_km=200
        )
        fixed_positions = positions  # 保存到全局变量
        print(f"📚 课程学习阶段1: Episode 1-{CURRICULUM_FIXED_END} 固定场景")
        print(f"   ✈️  5条矩形空域航线 (预计4-10个交叉点)")
        
    elif episode_num <= CURRICULUM_FIXED_END:
        # 阶段1: 使用固定场景
        positions = fixed_positions
        if episode_num % 50 == 0:
            print(f"  📌 固定场景训练中 ({episode_num}/{CURRICULUM_FIXED_END})")
            
    else:
        # 阶段2: 完全随机场景
        if episode_num == CURRICULUM_FIXED_END + 1:
            print(f"🎓 课程学习完成! Episode {CURRICULUM_FIXED_END+1}+ 完全随机场景")
            print(f"   🌍 全球随机中心 + 矩形空域航线")
        
        _, positions = generate_intelligent_crossing_routes(
            width_km=300,
            height_km=200
        )
        if episode_num % 100 == 0:
            route_dists = [calculate_haversine_distance(p[0], p[1], p[2], p[3]) for p in positions]
            print(f"  🎲 随机场景 Episode {episode_num}: {len(positions)}条航线, 平均{np.mean(route_dists)/1.852:.1f}nm")

    stack.stack('IC multi_agent.scn')


def get_own_state():
    """
    优化的状态空间设计（航向引导改进版）
    返回: 每架飞机6维原始状态 → 8维（角度转sin/cos）
    
    核心改进：使用航线方向引导，而非直接指向目标
    """
    global route_keeper, min_alt, max_alt, min_speed, max_speed
    own_state = {}
    
    for i, id in enumerate(traf.id):
        index = traf.id2idx(id)
        lat, lon = traf.lat[index], traf.lon[index]
        speed = traf.cas[index] * 1.9439  # knots
        alt = traf.alt[index] * 3.28084   # ft
        hdg = traf.hdg[index]              # degrees
        
        route = positions[route_keeper[int(id[2:])]]
        start_lat, start_lon, goal_lat, goal_lon, _ = route
        
        # ==========================================
        # 核心改进：计算航线标准方向
        # ==========================================
        route_bearing = calculate_bearing(start_lat, start_lon, goal_lat, goal_lon)
        
        # ==========================================
        # 状态1：速度归一化
        # ==========================================
        speed_normalized = (speed - min_speed) / (max_speed - min_speed)
        
        # ==========================================
        # 状态2：航向误差（相对于航线方向，而非目标）
        # ==========================================
        heading_error_to_route = route_bearing - hdg
        # 处理角度跨越360度
        if heading_error_to_route > 180:
            heading_error_to_route -= 360
        elif heading_error_to_route < -180:
            heading_error_to_route += 360
        # Normalizer会转sin/cos
        
        # ==========================================
        # 状态3：横向偏离（Cross Track Error）
        # ==========================================
        cross_track_error = calculate_distance_to_line(
            lat, lon, start_lat, start_lon, goal_lat, goal_lon
        )
        
        # ==========================================
        # 状态4：归一化高度
        # ==========================================
        altitude_normalized = (alt - min_alt) / (max_alt - min_alt)
        
        # ==========================================
        # 状态5：航线标准方向
        # ==========================================
        # route_bearing已计算，Normalizer会转sin/cos
        
        # ==========================================
        # 状态6：当前航向（绝对值）
        # ==========================================
        # hdg已获取，Normalizer会转sin/cos
        
        own_state[id] = [
            speed_normalized,           # 0: [0, 1]
            heading_error_to_route,     # 1: [-180, 180] → Normalizer转sin/cos
            cross_track_error,          # 2: km
            altitude_normalized,        # 3: [0, 1]
            route_bearing,              # 4: [0, 360] → Normalizer转sin/cos
            hdg,                        # 5: [0, 360] → Normalizer转sin/cos
        ]  # 总计6维原始 → Normalizer后8维 (3个角度各转sin/cos)
    
    return own_state


def get_min_Dis():
    """
    第三层：入侵者感知 (Intruder State) - 机体坐标系
    返回: 每架飞机最近3个入侵者的相对状态（机体坐标系）
           每个入侵者6维: [x_body, y_body, z_body, relative_hdg, distance, closing_rate]
    """
    global route_keeper, num_intruders
    
    id_list = traf.id
    n_aircraft = len(id_list)
    
    if n_aircraft == 0:
        return {}
    
    min_distances_body = {}
    
    for i, own_id in enumerate(id_list):
        # 自身状态
        own_lat = traf.lat[i]
        own_lon = traf.lon[i]
        own_alt = traf.alt[i] * 3.28084  # ft
        own_hdg = traf.hdg[i]
        own_speed = traf.cas[i] * 1.9439  # knots
        
        # 计算到所有其他飞机的距离
        distances = {}
        for j, other_id in enumerate(id_list):
            if i != j:
                dist = calculate_haversine_distance(
                    traf.lat[i], traf.lon[i],
                    traf.lat[j], traf.lon[j]
                )
                distances[other_id] = dist
        
        # 选择最近的3个
        if len(distances) == 0:
            min_distances_body[own_id] = []
            continue
        
        sorted_ids = sorted(distances.keys(), key=lambda x: distances[x])
        nearest_ids = sorted_ids[:min(num_intruders, len(sorted_ids))]
        
        # 计算机体坐标系下的相对状态
        min_distances_body[own_id] = []
        
        for intruder_id in nearest_ids:
            j = id_list.index(intruder_id)
            
            intruder_lat = traf.lat[j]
            intruder_lon = traf.lon[j]
            intruder_alt = traf.alt[j] * 3.28084  # ft
            intruder_hdg = traf.hdg[j]
            intruder_speed = traf.cas[j] * 1.9439  # knots
            
            # ==========================================
            # 机体坐标系转换
            # ==========================================
            
            # 1. 计算相对位置（地心坐标系）
            rel_north = (intruder_lat - own_lat) * 111.32  # km (纬度1度≈111.32km)
            rel_east = (intruder_lon - own_lon) * 111.32 * math.cos(math.radians(own_lat))  # km
            
            # 2. 旋转到机体坐标系
            # theta = own_hdg (0度=正北，顺时针增加)
            theta_rad = math.radians(own_hdg)
            
            # 机体坐标系：x轴=机头方向(前正后负), y轴=右翼方向(右正左负)
            x_body = rel_north * math.cos(theta_rad) + rel_east * math.sin(theta_rad)
            y_body = -rel_north * math.sin(theta_rad) + rel_east * math.cos(theta_rad)
            
            # 3. 相对高度（z轴：上正下负）
            z_body = intruder_alt - own_alt
            
            # 4. 入侵者相对航向
            relative_hdg = intruder_hdg - own_hdg
            if relative_hdg > 180:
                relative_hdg -= 360
            elif relative_hdg < -180:
                relative_hdg += 360
            
            # 5. 距离
            distance = distances[intruder_id]
            
            # 6. 接近速度 (closing rate)
            # 计算相对速度在连线方向上的分量
            own_vx = own_speed * math.sin(theta_rad)
            own_vy = own_speed * math.cos(theta_rad)
            
            intruder_theta_rad = math.radians(intruder_hdg)
            intruder_vx = intruder_speed * math.sin(intruder_theta_rad)
            intruder_vy = intruder_speed * math.cos(intruder_theta_rad)
            
            # 相对速度
            rel_vx = intruder_vx - own_vx
            rel_vy = intruder_vy - own_vy
            
            # 投影到连线方向
            if distance > 0.001:
                dx = rel_east
                dy = rel_north
                norm = math.sqrt(dx**2 + dy**2)
                if norm > 0.001:
                    unit_x = dx / norm
                    unit_y = dy / norm
                    closing_rate = -(rel_vx * unit_x + rel_vy * unit_y)
                else:
                    closing_rate = 0.0
            else:
                closing_rate = 0.0
            
            min_distances_body[own_id].append([
                x_body,         # 0: 机头方向距离（前为正）
                y_body,         # 1: 右翼方向距离（右为正）
                z_body,         # 2: 相对高度（上为正）
                relative_hdg,   # 3: 相对航向
                distance,       # 4: 3D距离
                closing_rate,   # 5: 接近速度
            ])
    
    return min_distances_body


def get_rewards(stats):
    global route_keeper, positions, win_list, last_goal_distance, last_along_track
    global boundary_strikes
    global collision_count, step_num, step_max, out_of_bound_count
    global last_vertical_distances  # 新增

    # ==========================================
    # 1. 基础权重
    # ==========================================
    w_collision = -120.0
    w_arrival = 60.0
    w_progress = 0.2
    w_heading = 0.15
    w_step_cost = -0.01
    w_timeout = -10.0

    # ==========================================
    # 2. 航线与边界 - [修复] 放宽边界适应全球随机场景
    # ==========================================
    w_deviation_linear = -0.01
    w_boundary_soft = -0.1
    w_out_of_corridor = -45.0
    soft_boundary_dist = 20.0  # 15→30 km软边界
    max_dev_dist = 35.0  # 25→50 km硬性OOB [关键修复]

    # ==========================================
    # 3. 避撞参数 - 分层警告区设计
    # ==========================================
    collision_hor = 5.0  # km
    collision_ver = 500.0  # ft

    danger_hor = 8.0  # km
    danger_ver = 8000.0  # ft
    w_danger = -25.0

    warning_hor = 10.0  # km
    warning_ver = 1000.0  # ft
    w_warning = -8.0

    arrival_distance = 3.0

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

        # 3. 航向对齐（相对于航线方向）
        dist_to_goal = calculate_haversine_distance(lati, loni, target_lat, target_lon)
        if dist_to_goal > arrival_distance:
            # 使用航线方向而非目标方位
            route_bearing = calculate_bearing(start_lat, start_lon, target_lat, target_lon)
            angle_diff = abs(route_bearing - hdgi)
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


def calculate_destination(lat, lon, bearing, distance_km):
    """
    根据起点、方向和距离计算终点坐标
    
    参数:
        lat: 起点纬度 (度)
        lon: 起点经度 (度) 
        bearing: 方向角 (度, 0-360, 0为正北)
        distance_km: 距离 (km)
    
    返回:
        (end_lat, end_lon): 终点坐标
    """
    R = 6371.0  # 地球半径 (km)
    
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    bearing_rad = math.radians(bearing)
    
    # 计算终点纬度
    end_lat_rad = math.asin(
        math.sin(lat_rad) * math.cos(distance_km / R) +
        math.cos(lat_rad) * math.sin(distance_km / R) * math.cos(bearing_rad)
    )
    
    # 计算终点经度
    end_lon_rad = lon_rad + math.atan2(
        math.sin(bearing_rad) * math.sin(distance_km / R) * math.cos(lat_rad),
        math.cos(distance_km / R) - math.sin(lat_rad) * math.sin(end_lat_rad)
    )
    
    end_lat = math.degrees(end_lat_rad)
    end_lon = math.degrees(end_lon_rad)
    
    return end_lat, end_lon

