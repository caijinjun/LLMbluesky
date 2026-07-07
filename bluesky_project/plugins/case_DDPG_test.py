"""
DDPG模型性能测试插件
加载已训练的模型,在不规则多边形空域场景中测试性能
"""

from bluesky import stack, traf
from bluesky.tools.aero import ft, nm
import numpy as np
import torch
import os
import random
import math
from geopy.distance import geodesic

# 导入DDPG模型 - 修正导入路径
from plugins.Multi_Agent.DDPG_3DTen import DDPG
from plugins.Multi_Agent.Normalizer import AircraftStateNormalizer

def init_plugin():
    global max_ac, episode_num, test_episodes
    global positions, route_keeper, route_queue
    global collision_count, out_of_bound_count, success_count
    global agent_manager
    
    print("=" * 80)
    print("🧪 DDPG模型测试模式")
    print("=" * 80)
    
    # 测试参数
    max_ac = 30
    test_episodes = 100  # 测试100个episode
    episode_num = 0
    
    collision_count = 0
    out_of_bound_count = 0
    success_count = 0
    
    # 生成不规则多边形航线
    def generate_irregular_polygon_routes(center_lat=32.5, center_lon=117.5, 
                                          radius_nm=150, num_boundary_points=12,
                                          num_routes=12, min_distance_nm=250):
        radius_km = radius_nm * 1.852
        min_distance_km = min_distance_nm * 1.852
        
        # 生成边界点
        vertices = []
        angles = np.linspace(0, 360, num_boundary_points, endpoint=False)
        for angle in angles:
            r = radius_km * np.random.uniform(0.85, 1.15)
            lat, lon = calculate_destination(center_lat, center_lon, angle, r)
            vertices.append((lat, lon))
        
        # 生成航线
        routes = []
        used_pairs = set()
        attempts = 0
        
        while len(routes) < num_routes and attempts < 1000:
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
        
        return np.array(vertices), np.array(routes)
    
    # 生成测试场景
    _, positions = generate_irregular_polygon_routes()
    print(f"✈️  生成了 {len(positions)} 条测试航线 (不规则多边形空域)")
    
    route_keeper = np.zeros(max_ac, dtype=int)
    route_queue = random.choices([20, 25, 30], k=positions.shape[0])
    
    # 加载训练好的模型
    model_path = "output/DDPG/DDPG3D-15/DDPG"  # 修改为您的模型路径
    
    if not os.path.exists(model_path + "_actor.pth"):
        print(f"❌ 错误: 找不到模型文件 {model_path}")
        print("请修改model_path指向您训练好的模型")
        return
    
    # 初始化DDPG agent
    state_dim = 18
    action_dim = 3
    max_agents = 30
    
    agent_manager = DDPGAgentManager(
        state_dim=state_dim,
        action_dim=action_dim,
        max_agents=max_agents,
        hidden_dim=256,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        buffer_size=100000,
        batch_size=256
    )
    
    # 加载模型
    agent_manager.load_models(model_path)
    print(f"✅ 成功加载模型: {model_path}")
    print(f"📊 测试设置: {test_episodes} episodes, {max_ac} aircraft/episode")
    print("=" * 80)
    
    return {
        'plugin_name': 'case_DDPG_Test',
        'plugin_type': 'sim',
        'update_interval': 5.0,
        'update': update,
        'preupdate': preupdate,
        'reset': reset
    }, {}


def preupdate():
    pass


def update():
    global episode_num, collision_count, out_of_bound_count, success_count
    
    # 简化版update,只执行动作不训练
    if len(traf.id) == 0:
        return
    
    # 获取状态并执行动作
    for i, aircraft_id in enumerate(traf.id):
        state = get_aircraft_state(aircraft_id, i)
        
        # 使用模型预测动作(无探索噪声)
        action = agent_manager.take_action(state, episode_num=-1)  # -1表示测试模式
        
        # 应用动作
        delta_speed = action[0] * 15
        delta_alt = action[1] * 200
        delta_hdg = action[2] * 12
        
        new_tas = np.clip(traf.cas[i] * 1.9437 + delta_speed, 220, 320)
        new_alt = np.clip(traf.alt[i] * 3.28084 + delta_alt, 19000, 21000)
        new_hdg = (traf.hdg[i] + delta_hdg) % 360
        
        stack.stack(f'SPD {aircraft_id} {new_tas}')
        stack.stack(f'ALT {aircraft_id} {new_alt}')
        stack.stack(f'HDG {aircraft_id} {new_hdg}')
    
    # 检查终止条件
    _, dones = get_rewards()
    
    for aircraft_id in list(dones.keys()):
        if dones[aircraft_id]:
            stack.stack(f'DEL {aircraft_id}')


def reset():
    global episode_num, collision_count, out_of_bound_count, success_count
    global positions, route_keeper, route_queue
    
    # 计算统计
    total = max_ac
    timeout = total - success_count - collision_count - out_of_bound_count
    success_rate = success_count / total
    
    print(f"Episode {episode_num:3d} | 成功:{success_count:2d} | 碰撞:{collision_count:2d} | 出界:{out_of_bound_count:2d} | 超时:{timeout:2d} | 成功率:{success_rate:.2%}")
    
    # 保存统计
    stats_dir = 'output/DDPG/DDPG-15-Testcomplexy'
    os.makedirs(stats_dir, exist_ok=True)
    stats_file = os.path.join(stats_dir, 'test_stats.csv')
    
    if episode_num == 0:
        with open(stats_file, 'w') as f:
            f.write("Episode,Success,Collision,OOB,Timeout,SuccessRate\n")
    
    with open(stats_file, 'a') as f:
        f.write(f"{episode_num},{success_count},{collision_count},{out_of_bound_count},{timeout},{success_rate:.4f}\n")
    
    # 重置计数
    collision_count = 0
    out_of_bound_count = 0
    success_count = 0
    
    episode_num += 1
    
    if episode_num >= test_episodes:
        print("=" * 80)
        print(f"✅ 测试完成! 共 {test_episodes} episodes")
        print(f"📊 结果已保存到: {stats_file}")
        print("=" * 80)
        stack.stack('STOP')
        return
    
    # 生成新的随机场景
    def generate_irregular_polygon_routes_reset(center_lat=32.5, center_lon=117.5, 
                                                radius_nm=150, num_boundary_points=12,
                                                num_routes=12, min_distance_nm=250):
        radius_km = radius_nm * 1.852
        min_distance_km = min_distance_nm * 1.852
        
        vertices = []
        angles = np.linspace(0, 360, num_boundary_points, endpoint=False)
        for angle in angles:
            r = radius_km * np.random.uniform(0.85, 1.15)
            lat, lon = calculate_destination(center_lat, center_lon, angle, r)
            vertices.append((lat, lon))
        
        routes = []
        used_pairs = set()
        attempts = 0
        
        while len(routes) < num_routes and attempts < 1000:
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
        
        return np.array(vertices), np.array(routes)
    
    _, positions = generate_irregular_polygon_routes_reset()
    route_keeper = np.zeros(max_ac, dtype=int)
    route_queue = random.choices([20, 25, 30], k=positions.shape[0])
    
    stack.stack('IC multi_agent.scn')


def get_aircraft_state(aircraft_id, index):
    """获取飞机状态 - 简化版"""
    route_idx = route_keeper[int(aircraft_id[2:])]
    target_lat, target_lon = positions[route_idx][2], positions[route_idx][3]
    
    lat, lon = traf.lat[index], traf.lon[index]
    alt = traf.alt[index] * 3.28084
    hdg = traf.hdg[index]
    cas = traf.cas[index] * 1.9437
    
    dist_to_goal = calculate_haversine_distance(lat, lon, target_lat, target_lon)
    bearing_to_goal = calculate_bearing(lat, lon, target_lat, target_lon)
    
    # 简化状态:18维
    state = np.zeros(18)
    state[0] = lat / 90.0
    state[1] = lon / 180.0
    state[2] = alt / 50000.0
    state[3] = hdg / 360.0
    state[4] = cas / 500.0
    state[5] = dist_to_goal / 1000.0
    state[6] = bearing_to_goal / 360.0
    
    return state


def get_rewards():
    """简化的奖励计算"""
    global collision_count, out_of_bound_count, success_count
    
    rewards = {id_: 0.0 for id_ in traf.id}
    dones = {id_: False for id_ in traf.id}
    
    for i, aircraft_id in enumerate(traf.id):
        route_idx = route_keeper[int(aircraft_id[2:])]
        start_lat, start_lon = positions[route_idx][0], positions[route_idx][1]
        target_lat, target_lon = positions[route_idx][2], positions[route_idx][3]
        
        lat, lon = traf.lat[i], traf.lon[i]
        
        # 检查到达
        dist_to_goal = calculate_haversine_distance(lat, lon, target_lat, target_lon)
        if dist_to_goal < 8.0:
            dones[aircraft_id] = True
            success_count += 1
            continue
        
        # 检查偏航
        cross_track = calculate_distance_to_line(lat, lon, start_lat, start_lon, target_lat, target_lon)
        if cross_track > 25.0:
            dones[aircraft_id] = True
            out_of_bound_count += 1
            continue
        
        # 检查碰撞(简化)
        for j in range(i+1, len(traf.id)):
            dist_h = calculate_haversine_distance(lat, lon, traf.lat[j], traf.lon[j])
            dist_v = abs(traf.alt[i] - traf.alt[j]) * 3.28084
            
            if dist_h <= 2.0 and dist_v < 500.0:
                dones[aircraft_id] = True
                collision_count += 1
                break
    
    return rewards, dones


# 辅助函数
def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad)*math.cos(lat2_rad)*math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c


def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    dlon = lon2_rad - lon1_rad
    y = math.sin(dlon) * math.cos(lat2_rad)
    x = math.cos(lat1_rad)*math.sin(lat2_rad) - math.sin(lat1_rad)*math.cos(lat2_rad)*math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def calculate_destination(lat, lon, bearing, distance_km):
    R = 6371.0
    lat_rad, lon_rad = math.radians(lat), math.radians(lon)
    bearing_rad = math.radians(bearing)
    end_lat_rad = math.asin(math.sin(lat_rad)*math.cos(distance_km/R) + 
                            math.cos(lat_rad)*math.sin(distance_km/R)*math.cos(bearing_rad))
    end_lon_rad = lon_rad + math.atan2(math.sin(bearing_rad)*math.sin(distance_km/R)*math.cos(lat_rad),
                                       math.cos(distance_km/R) - math.sin(lat_rad)*math.sin(end_lat_rad))
    return math.degrees(end_lat_rad), math.degrees(end_lon_rad)


def calculate_distance_to_line(point_lat, point_lon, line_start_lat, line_start_lon, line_end_lat, line_end_lon):
    R = 6371.0
    d_start = calculate_haversine_distance(point_lat, point_lon, line_start_lat, line_start_lon)
    d_end = calculate_haversine_distance(point_lat, point_lon, line_end_lat, line_end_lon)
    line_length = calculate_haversine_distance(line_start_lat, line_start_lon, line_end_lat, line_end_lon)
    
    if line_length < 1e-6:
        return d_start
    
    bearing_start_to_end = calculate_bearing(line_start_lat, line_start_lon, line_end_lat, line_end_lon)
    bearing_start_to_point = calculate_bearing(line_start_lat, line_start_lon, point_lat, point_lon)
    angle_diff = math.radians(abs(bearing_start_to_point - bearing_start_to_end))
    
    cross_track_distance = math.asin(math.sin(d_start/R) * math.sin(angle_diff)) * R
    return abs(cross_track_distance)
