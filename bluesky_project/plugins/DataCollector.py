import json
import math
import os
import random


class DataCollector:
    def __init__(self, save_path="output/train_data_7b.jsonl", record_interval=5):
        self.save_path = save_path
        self.record_interval = record_interval  # 降采样：每隔多少步记录一次
        self.step_counter = {}  # 记录每个飞机的步数

        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # 以追加模式打开文件
        self.file_handle = open(self.save_path, 'a', encoding='utf-8')

    def close(self):
        self.file_handle.close()

    def calculate_bearing(self, lat1, lon1, lat2, lon2):
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        dlon = lon2_rad - lon1_rad
        y = math.sin(dlon) * math.cos(lat2_rad)
        x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
        return (math.degrees(math.atan2(y, x)) + 360) % 360

    def generate_rule_reasoning(self, own_hdg, intruder_bearing, rel_heading_diff):
        """生成规则解释 (包含规则类型判定)"""
        rule_type = "UNKNOWN"
        rule_text = ""
        analysis_text = ""

        # 航向差接近180度 (160-200) -> 对头
        if 160 <= rel_heading_diff <= 200:
            rule_type = "HEAD_ON"
            rule_text = "【规则引用】根据 ICAO 对头相遇规则：当两航空器在正对面相遇时，双方均应向右转弯。"
            analysis_text = "【分析】检测到对头冲突。为建立左侧安全间隔，必须向右改变航向。"
        # 航向差接近0度 -> 追越
        elif 0 <= rel_heading_diff <= 20:
            rule_type = "OVERTAKE"
            rule_text = "【规则引用】根据 ICAO 追越规则：从后方超越前机的航空器，应向右避让。"
            analysis_text = "【分析】正在追越前机。为避免尾随碰撞，应向右转弯。"
        else:
            if 0 < intruder_bearing < 180:
                rule_type = "CROSSING_RIGHT"  # 右侧有飞机
                rule_text = "【规则引用】根据 ICAO 交叉相遇规则：右侧有航空器时，应给对方让路。"
                analysis_text = f"【分析】入侵机位于右舷 {int(intruder_bearing)} 度。本机无路权，需主动机动。"
            else:
                rule_type = "CROSSING_LEFT"  # 左侧有飞机
                rule_text = "【规则引用】根据 ICAO 交叉相遇规则：本机位于入侵机右侧，拥有优先路权。"
                analysis_text = "【分析】本机有路权，但建议根据态势微调以确保安全。"

        return rule_type, rule_text, analysis_text

    def save_sample(self, aircraft_id, own_state, intruders, action_raw):
        """
        保存样本，包含三重过滤机制
        """
        # --- 过滤器 1: 降采样 (防止数据太密) ---
        if aircraft_id not in self.step_counter:
            self.step_counter[aircraft_id] = 0
        self.step_counter[aircraft_id] += 1

        if self.step_counter[aircraft_id] % self.record_interval != 0:
            return

        if not intruders: return
        nearest = intruders[0]  # nearest结构需对应: [id, lat, lon, spd, alt, trk, dist, closing_spd]

        # 构造 Input Prompt
        bearing_to_intruder = self.calculate_bearing(own_state['lat'], own_state['lon'], nearest[1], nearest[2])

        relative_bearing = (bearing_to_intruder - own_state['hdg'] + 360) % 360
        if relative_bearing > 180: relative_bearing -= 360

        rel_hdg_diff = abs(own_state['hdg'] - nearest[5])

        # --- [修复点 1] 还原所有动作真实值 (包括高度) ---
        d_spd = action_raw[0] * 15.0  # Speed (kt)
        d_alt = action_raw[1] * 200.0  # Altitude (ft) - 假设主程序里缩放因子是200
        d_hdg = action_raw[2] * 12.0  # Heading (deg)

        # --- [修复点 2] 过滤器增加高度检测 ---
        # 只有动作幅度够大才记录 (航向>1.5度 OR 速度>3节 OR 高度>10英尺)
        is_significant_action = abs(d_hdg) > 1.5 or abs(d_spd) > 3.0 or abs(d_alt) > 10.0

        if not is_significant_action:
            if random.random() > 0.1:
                return

        # 获取规则
        rule_type, rule_text, analysis_text = self.generate_rule_reasoning(own_state['hdg'], relative_bearing,
                                                                           rel_hdg_diff)

        # --- 过滤器 3: 逻辑一致性校验 ---
        # 规则是对头(必须右转)，但 DDPG 却左转了 -> 丢弃
        if rule_type == "HEAD_ON" and d_hdg < -1.0: return
        if rule_type == "OVERTAKE" and d_hdg < -1.0: return
        if rule_type == "CROSSING_RIGHT" and d_hdg < -1.0: return

        # 构造 Output
        prompt = f"【本机状态】航向: {int(own_state['hdg'])}度, 速度: {int(own_state['spd'])}kt。\n" \
                 f"【入侵机】相对方位 {int(relative_bearing)} 度，航向 {int(nearest[5])} 度。\n" \
                 f"【任务】请给出避撞指令。"

        # --- [修复点 3] 输出文本包含高度指令 ---
        action_desc = f"【指令】航向改变 {d_hdg:.1f} 度，速度改变 {d_spd:.1f} kt，高度改变 {d_alt:.1f} ft。"

        full_output = f"{rule_text}\n{analysis_text}\n{action_desc}"

        data_line = {
            "instruction": "你是一个专业的空管AI，负责根据ICAO规则进行防撞决策。",
            "input": prompt,
            "output": full_output
        }

        self.file_handle.write(json.dumps(data_line, ensure_ascii=False) + "\n")
        self.file_handle.flush()