import os
import shutil
import bluesky as bs
from bluesky import stack, settings, navdb, traf, sim, scr, tools
from geopy.distance import geodesic
from plugins.Multi_Agent.DDPG_3DTen import DDPG
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

    written = 0  # [训练模式] 1 = 记录场景文件, 0 = 关闭

    os.makedirs('output/DDPG/DDPG3D-18/scenarios', exist_ok=True)

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
    obs_dim = 13 + 27  # [修复] 归一化后：自身13维 + 入侵者27维(3×9) = 40维
    agent_manager = DDPG(state_dim=13, intruders_dim=27, hidden_dim=256, action_dim=3,
                         max_agents=max_ac)  # state_dim=13归一化后, intruders_dim=27归一化后(3×9)

    # [训练模式] 注释掉自动加载模型，从头开始训练或者手动加载
    # model_path = "output\\DDPG\\DDPG3D-7\\DDPG"
    # if os.path.exists(f"{model_path}_actor.pth"):
    #     print("="*50)
    #     print("🔄 继续训练: 加载已有模型")
    #     agent_manager.load_models(model_path)
    #     print("✅ 模型加载成功！")
    #     print("="*50)
    # else:
    print("=" * 50)
    print("🚀 训练模式: 从头开始训练新模型")
    print("=" * 50)
    reward_list = [0 for _ in range(max_ac)]

    best_win = 0
    win_list = 0
    min_speed = 220
    max_speed = 320
    min_alt = 19000
    max_alt = 21000

    step_num = 0
    episode_max = 6000  # [训练模式] 完整训练轮数
    step_max = 1300  # [修复2] 1100→1300,给更多时间完成航线

    # ==========================================
    # [新设计] 不规则多边形空域航线生成
    # ==========================================
    def generate_irregular_polygon_routes(center_lat=32.5, center_lon=117.5, 
                                          radius_nm=150, num_boundary_points=12,
                                          num_routes=12, min_distance_nm=250):
        """
        生成基于不规则多边形的航线,避免所有航线聚集在中心
        
        参数:
            center_lat, center_lon: 空域中心坐标
            radius_nm: 多边形半径 (海里)
            num_boundary_points: 边界点数量
            num_routes: 生成的航线数量
            min_distance_nm: 最小航线距离 (海里)
        
        返回:
            vertices: 边界点数组 [(纬度, 经度), ...]
            routes: 航线数组 [起点纬度, 起点经度, 终点纬度, 终点经度, 初始航向]
        """
        # 转换单位: 1 nm = 1.852 km
        radius_km = radius_nm * 1.852
        min_distance_km = min_distance_nm * 1.852
        
        # 1. 生成不规则多边形的边界点
        vertices = []
        angles = np.linspace(0, 360, num_boundary_points, endpoint=False)
        
        for angle in angles:
            # 半径添加 ±15% 随机扰动
            r = radius_km * np.random.uniform(0.85, 1.15)
            lat, lon = calculate_destination(center_lat, center_lon, angle, r)
            vertices.append((lat, lon))
        
        # 2. 从边界点生成航线,限制对角线数量
        routes = []
        used_pairs = set()
        diagonal_count = 0  # 对角线航线计数
        max_diagonals = max(1, num_routes // 3)  # 最多1/3对角线
        
        attempts = 0
        max_attempts = 1000
        
        def get_angle_difference(i, j, num_points):
            """计算两点在圆周上的角度差"""
            angle_per_point = 360.0 / num_points
            angle_i = i * angle_per_point
            angle_j = j * angle_per_point
            diff = abs(angle_j - angle_i)
            if diff > 180:
                diff = 360 - diff
            return diff
        
        while len(routes) < num_routes and attempts < max_attempts:
            # 随机选择两个不同的边界点
            i = np.random.randint(0, num_boundary_points)
            j = np.random.randint(0, num_boundary_points)
            
            if i == j or (i, j) in used_pairs or (j, i) in used_pairs:
                attempts += 1
                continue
            
            # 计算角度差,判断是否为对角线
            angle_diff = get_angle_difference(i, j, num_boundary_points)
            is_diagonal = angle_diff > 140  # 大于140度视为对角线
            
            # 如果是对角线且已达上限,跳过
            if is_diagonal and diagonal_count >= max_diagonals:
                attempts += 1
                continue
            
            start_lat, start_lon = vertices[i]
            end_lat, end_lon = vertices[j]
            
            # 计算距离
            dist = calculate_haversine_distance(start_lat, start_lon, end_lat, end_lon)
            
            # 检查是否满足最小距离要求
            if dist >= min_distance_km:
                initial_hdg = calculate_bearing(start_lat, start_lon, end_lat, end_lon)
                routes.append([start_lat, start_lon, end_lat, end_lon, initial_hdg])
                used_pairs.add((i, j))
                if is_diagonal:
                    diagonal_count += 1
            
            attempts += 1
        
        # 如果无法生成足够的航线,放宽限制
        if len(routes) < num_routes:
            print(f"⚠️  警告: 仅生成 {len(routes)} 条满足要求的航线")
            print(f"   正在放宽限制生成剩余航线...")
            
            while len(routes) < num_routes and attempts < max_attempts * 2:
                i = np.random.randint(0, num_boundary_points)
                j = np.random.randint(0, num_boundary_points)
                
                if i == j or (i, j) in used_pairs or (j, i) in used_pairs:
                    attempts += 1
                    continue
                
                start_lat, start_lon = vertices[i]
                end_lat, end_lon = vertices[j]
                
                # 放宽距离限制到80%
                dist = calculate_haversine_distance(start_lat, start_lon, end_lat, end_lon)
                if dist >= min_distance_km * 0.8:
                    initial_hdg = calculate_bearing(start_lat, start_lon, end_lat, end_lon)
                    routes.append([start_lat, start_lon, end_lat, end_lon, initial_hdg])
                    used_pairs.add((i, j))
                
                attempts += 1
        
        print(f"✈️  生成 {len(routes)} 条航线 (对角线: {diagonal_count}, 斜角: {len(routes)-diagonal_count})")
        return np.array(vertices), np.array(routes)


    # 生成不规则多边形航线 - [简化配置]
    _, positions = generate_irregular_polygon_routes(
        center_lat=32.5, 
        center_lon=117.5,
        radius_nm=150,           # 150 海里半径
        num_boundary_points=6,   # [简化] 6个边界点(六边形)
        num_routes=5,            # [简化] 5条航线
        min_distance_nm=150      # [简化] 最小150海里
    )
    print(f"✈️  生成了 {len(positions)} 条航线 (简化不规则空域)")

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
    route_queue = random.choices(choices, k=max_ac)  # [修复] 与max_ac一致,30个时间槽
    episode_num = 0
    old_air_craft = {}
    current_air_craft = {}
    actions = {}
    
    # [恢复] 按用户要求移除固定分配
    # fixed_route_assignment = [i % len(positions) for i in range(max_ac)]
    # print(f"🎯 固定航线分配: {max_ac}架飞机分配到{len(positions)}条航线")
    
    # [新增] 初始化固定场景变量(用于课程学习)
    fixed_positions = positions.copy()  # 保存初始场景作为固定场景

    if written == 1:
        SCN_File = f"output/DDPG/DDPG3D-18/scenarios/{episode_num}.scn"

        if os.path.exists('multi_agent.scn'):
            shutil.copy2('multi_agent.scn', SCN_File)
        else:
            open(SCN_File, 'w').close()

    config = {
        'plugin_name': 'case_DDPG_3D-12',
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

    # [修复] 间隔创建飞机，避免起点拥挤
    if num_ac < max_ac:
        for k in range(len(route_queue)):
            if step_num == route_queue[k]:
                # 随机选择一条航线
                route_idx = np.random.randint(0, len(positions))
                
                lat, lon, glat, glon, h = positions[route_idx]
                bearing_to_goal = calculate_bearing(lat, lon, glat, glon)
                
                # [修复3] 随机高度 19000-21000英尺,以500英尺为间隔
                init_alt = np.random.choice([19000, 19500, 20000, 20500, 21000])
                
                stack.stack('CRE KL{}, B737, {}, {}, {}, {}, 250'.format(num_ac, lat, lon, h, init_alt))
                stack.stack('ADDWPT KL{} {}, {}'.format(num_ac, glat, glon))
                stack.stack('HDG KL{} {}'.format(num_ac, bearing_to_goal))
                stack.stack(f'VNAV KL{num_ac} ON')

                if written == 1:
                    with open(SCN_File, 'a', encoding='utf-8') as f:
                        f.write(f"{data_time}>CRE KL{num_ac}, B737, {lat}, {lon}, {h}, {init_alt}, 250\n")
                        f.write(f"{data_time}>ADDWPT KL{num_ac} {glat}, {glon}\n")
                        f.write(f"{data_time}>HDG KL{num_ac} {bearing_to_goal}\n")
                        f.write(f"{data_time}>VNAV KL{num_ac} ON\n")

                route_keeper[num_ac] = route_idx
                num_ac += 1
                route_queue[k] = step_num + random.choices(choices, k=1)[0]
                if num_ac == max_ac:
                    break

    step_num += 1

    # [修改] 每 150 步进行一次更新训练
    if step_num > 0 and step_num % 300 == 0:
        agent_manager.update(transition_dict, episode_num)
        # [关键] 清空字典
        for key in transition_dict:
            transition_dict[key] = []

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
    global last_vertical_distances  # 新增

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

    stats_dir = 'output/DDPG/DDPG3D-18'
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
    route_queue = random.choices([20, 25, 30], k=max_ac)  # [修复] 30个时间槽

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
        agent_manager.save_models(f"output/DDPG/DDPG3D-18/DDPG")

    best_win = max(win_list, best_win)
    win_list = 0
    if episode_num == episode_max:
        stack.stack('STOP')

    if written == 1:
        SCN_File = f"output/DDPG/DDPG3D-18/scenarios/{episode_num}.scn"

        if os.path.exists('multi_agent.scn'):
            shutil.copy2('multi_agent.scn', SCN_File)
        else:
            open(SCN_File, 'w').close()

    # ==========================================
    # [新增] 课程学习: 前500轮固定场景,之后随机场景
    # ==========================================
    def generate_irregular_polygon_routes_reset(center_lat=32.5, center_lon=117.5, 
                                                radius_nm=150, num_boundary_points=12,
                                                num_routes=12, min_distance_nm=250):
        """重新生成航线(与init_plugin版本相同)"""
        radius_km = radius_nm * 1.852
        min_distance_km = min_distance_nm * 1.852
        
        # 生成不规则多边形边界点
        vertices = []
        angles = np.linspace(0, 360, num_boundary_points, endpoint=False)
        for angle in angles:
            r = radius_km * np.random.uniform(0.85, 1.15)
            lat, lon = calculate_destination(center_lat, center_lon, angle, r)
            vertices.append((lat, lon))
        
        # 从边界点生成航线
        routes = []
        used_pairs = set()
        attempts = 0
        max_attempts = 1000
        
        while len(routes) < num_routes and attempts < max_attempts:
            i = np.random.randint(0, num_boundary_points)
            j = np.random.randint(0, num_boundary_points)
            
            if i == j or (i, j) in used_pairs or (j, i) in used_pairs:
                attempts += 1
                continue
            
            start_lat, start_lon = vertices[i]
            end_lat, end_lon = vertices[j]
            dist = calculate_haversine_distance(start_lat, start_lon, end_lat, end_lon)
            
            if dist >= min_distance_km:
                initial_hdg = calculate_bearing(start_lat, start_lon, end_lat, end_lon)
                routes.append([start_lat, start_lon, end_lat, end_lon, initial_hdg])
                used_pairs.add((i, j))
            
            attempts += 1
        
        # 放宽限制生成剩余航线
        if len(routes) < num_routes:
            while len(routes) < num_routes and attempts < max_attempts * 2:
                i = np.random.randint(0, num_boundary_points)
                j = np.random.randint(0, num_boundary_points)
                
                if i == j or (i, j) in used_pairs or (j, i) in used_pairs:
                    attempts += 1
                    continue
                
                start_lat, start_lon = vertices[i]
                end_lat, end_lon = vertices[j]
                dist = calculate_haversine_distance(start_lat, start_lon, end_lat, end_lon)
                
                if dist >= min_distance_km * 0.8:
                    initial_hdg = calculate_bearing(start_lat, start_lon, end_lat, end_lon)
                    routes.append([start_lat, start_lon, end_lat, end_lon, initial_hdg])
                    used_pairs.add((i, j))
                
                attempts += 1
        
        return np.array(vertices), np.array(routes)
    
    # ==========================================
    # 课程学习策略 - 平滑过渡版本
    # ==========================================
    FIXED_PHASE_END = 400        # Episode 1-400: 100%固定场景
    TRANSITION_END = 700         # Episode 701+: 100%随机场景
    
    if episode_num == 1:
        # Episode 1: 生成初始固定场景并保存
        _, positions = generate_irregular_polygon_routes_reset(
            center_lat=32.5,
            center_lon=117.5,
            radius_nm=150,
            num_boundary_points=6,
            num_routes=5,
            min_distance_nm=150
        )
        fixed_positions = positions
        print(f"  🎯 生成初始固定场景 (将用于前{FIXED_PHASE_END}轮,然后平滑过渡至Episode {TRANSITION_END})")
    
    elif episode_num <= FIXED_PHASE_END:
        # Episode 2-400: 100%使用固定场景
        positions = fixed_positions
        if episode_num % 100 == 0:
            print(f"  📌 固定场景阶段 (Episode {episode_num}/{FIXED_PHASE_END})")
    
    elif episode_num <= TRANSITION_END:
        # Episode 401-700: 混合过渡期,线性增加随机场景比例
        progress = (episode_num - FIXED_PHASE_END) / (TRANSITION_END - FIXED_PHASE_END)
        random_ratio = progress  # 从0%线性增长到100%
        
        if np.random.random() < random_ratio:
            # 使用随机场景
            _, positions = generate_irregular_polygon_routes_reset(
                center_lat=32.5,
                center_lon=117.5,
                radius_nm=150,
                num_boundary_points=6,
                num_routes=5,
                min_distance_nm=150
            )
            if episode_num % 50 == 0:
                print(f"  🔄 过渡期-随机 ({random_ratio*100:.0f}%随机率, Episode {episode_num})")
        else:
            # 使用固定场景
            positions = fixed_positions
            if episode_num % 50 == 0:
                print(f"  🔄 过渡期-固定 ({random_ratio*100:.0f}%随机率, Episode {episode_num})")
    
    else:
        # Episode 701+: 100%随机场景
        _, positions = generate_irregular_polygon_routes_reset(
            center_lat=32.5,
            center_lon=117.5,
            radius_nm=150,
            num_boundary_points=6,
            num_routes=5,
            min_distance_nm=150
        )
        if episode_num == TRANSITION_END + 1:
            print(f"  🎓 课程学习完成! 进入完全随机场景阶段")
        elif episode_num % 100 == 0:
            route_dists = [calculate_haversine_distance(p[0], p[1], p[2], p[3]) for p in positions]
            print(f"  🎲 随机场景: {len(positions)}条航线, 平均{np.mean(route_dists)/1.852:.1f}nm")

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

        # [坐标相对化] 提升泛化能力
        relative_to_start_lat = lat - start_lat
        relative_to_start_lon = lon - start_lon
        relative_to_goal_lat = goal_lat - lat  
        relative_to_goal_lon = goal_lon - lon
        
        # 计算到目标的方位角（明确方向信号）
        bearing_to_goal = calculate_bearing(lat, lon, goal_lat, goal_lon)
        
        # 计算航向偏差（action-oriented）
        heading_error = bearing_to_goal - hdg
        if heading_error > 180:
            heading_error -= 360
        elif heading_error < -180:
            heading_error += 360
        heading_error_norm = heading_error / 180.0

        own_state[id] = [
            # 相对坐标（泛化）
            relative_to_start_lat, relative_to_start_lon,  # 0-1: 已飞多远
            relative_to_goal_lat, relative_to_goal_lon,    # 2-3: 还差多远
            # 基础状态
            speed, alt, hdg,                                # 4-6
            # 任务信息
            start_h,                                        # 7: 起始航向
            bearing_to_goal,                                # 8: 目标方位
            heading_error_norm,                             # 9: 航向偏差
        ]  # 总计10维 → 归一化后12维 (start_h和bearing用sin/cos)
    return own_state


def get_min_Dis():
    """
    获取每架飞机最近的3个入侵者的相对状态
    返回: {aircraft_id: [[7维相对状态] * 3]}
    """
    global route_keeper, num_intruders
    
    id_list = traf.id
    n_aircraft = len(id_list)
    
    if n_aircraft == 0:
        return {}
    
    min_distances_relative = {}
    
    for i, own_id in enumerate(id_list):
        # 自身状态
        own_lat = traf.lat[i]
        own_lon = traf.lon[i]
        own_speed = traf.cas[i] * 1.9439
        own_alt = traf.alt[i] * 3.28084
        own_hdg = traf.hdg[i]
        
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
            min_distances_relative[own_id] = []
            continue
            
        sorted_ids = sorted(distances.keys(), key=lambda x: distances[x])
        nearest_ids = sorted_ids[:min(num_intruders, len(sorted_ids))]
        
        # 计算相对状态
        min_distances_relative[own_id] = []
        
        for intruder_id in nearest_ids:
            j = id_list.index(intruder_id)
            
            intruder_lat = traf.lat[j]
            intruder_lon = traf.lon[j]
            intruder_speed = traf.cas[j] * 1.9439
            intruder_alt = traf.alt[j] * 3.28084
            intruder_hdg = traf.hdg[j]
            
            # 1. 相对位置（泛化）
            rel_lat = intruder_lat - own_lat
            rel_lon = intruder_lon - own_lon
            
            # 2. 相对高度
            rel_alt = intruder_alt - own_alt
            
            # 3. 相对方位（关键！）
            bearing_to_intruder = calculate_bearing(own_lat, own_lon, intruder_lat, intruder_lon)
            relative_bearing = bearing_to_intruder - own_hdg
            if relative_bearing > 180:
                relative_bearing -= 360
            elif relative_bearing < -180:
                relative_bearing += 360
            
            # 4. 入侵者绝对航向（预测轨迹）
            # 保留绝对值，让模型学习判断
            
            # 5. 距离
            distance = distances[intruder_id]
            
            # 6. 接近速度 (closing rate)
            # 计算相对速度在连线方向上的分量
            # V_closing = -d(distance)/dt
            import math
            
            # 自身速度矢量
            own_vx = own_speed * math.sin(math.radians(own_hdg))
            own_vy = own_speed * math.cos(math.radians(own_hdg))
            
            # 入侵者速度矢量
            intruder_vx = intruder_speed * math.sin(math.radians(intruder_hdg))
            intruder_vy = intruder_speed * math.cos(math.radians(intruder_hdg))
            
            # 相对速度矢量
            rel_vx = intruder_vx - own_vx
            rel_vy = intruder_vy - own_vy
            
            # 连线方向单位矢量
            if distance > 0.001:  # 避免除以0
                dx = rel_lon * 111.32 * math.cos(math.radians(own_lat))  # km
                dy = rel_lat * 111.32  # km
                norm = math.sqrt(dx**2 + dy**2)
                if norm > 0.001:
                    unit_x = dx / norm
                    unit_y = dy / norm
                    # 投影: 相对速度在连线方向上的分量
                    # 负值表示接近，正值表示远离
                    closing_rate = -(rel_vx * unit_x + rel_vy * unit_y)  # knots
                else:
                    closing_rate = 0.0
            else:
                closing_rate = 0.0
            
            min_distances_relative[own_id].append([
                rel_lat,            # 0: 相对纬度
                rel_lon,            # 1: 相对经度
                rel_alt,            # 2: 相对高度
                relative_bearing,   # 3: 相对方位
                intruder_hdg,       # 4: 入侵者航向
                distance,           # 5: 距离
                closing_rate,       # 6: 接近速度
            ])
    
    return min_distances_relative


def get_rewards(stats):
    global route_keeper, positions, win_list, last_goal_distance, last_along_track
    global boundary_strikes
    global collision_count, step_num, step_max, out_of_bound_count
    global last_vertical_distances  # 新增

    # ==========================================
    # 1. 基础权重
    # ==========================================
    # [修复5] 增强奖励信号
    w_collision = -140.0
    w_arrival = 100.0        # 60→10,增加到达奖励
    w_progress = 2.0          # 0.2→2.0,10倍提升
    w_heading = 1.5           # 0.15→1.5,10倍提升
    w_step_cost = -0.05       # -0.01→-0.05,增加时间压力
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

        # 硬性出界 - 横向边界
        if cross_track_error > max_dev_dist:
            rewards[aircraft_id] += w_out_of_corridor
            dones[aircraft_id] = True
            out_of_bound_count += 1
            continue
        
        # [新增] 硬性出界 - 纵向边界
        # 计算航线总长度
        route_length = calculate_haversine_distance(start_lat, start_lon, target_lat, target_lon)
        
        # 允许范围: 起点向后50km 到 终点向前50km
        longitudinal_margin = 50.0  # km
        min_along_track = -longitudinal_margin
        max_along_track = route_length + longitudinal_margin
        
        # 检查是否超出纵向边界
        if current_along_track < min_along_track or current_along_track > max_along_track:
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

