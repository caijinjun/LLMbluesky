# BlueSky插件系统路径补丁
import sys
import os

# 添加plugins目录到Python路径
plugins_dir = os.path.dirname(os.path.abspath(__file__))
layered_safeMARL_dir = os.path.join(plugins_dir, 'LayeredSafe MARL')

if layered_safeMARL_dir not in sys.path:
    sys.path.insert(0, layered_safeMARL_dir)
