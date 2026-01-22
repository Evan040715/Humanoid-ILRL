import joblib
import numpy as np
import os
from pathlib import Path

# === 配置 ===
# 输入文件路径
INPUT_FILE = "resources/feedforward_data/violin_left15_poses.pkl"
# 输出文件路径 (自动保存到 outputs 目录)
OUTPUT_FILE = "resources/motions/output/from_pkl/violin_h1_19dof.npy"

def convert():
    # 1. 路径处理
    root_dir = Path(__file__).resolve().parent.parent
    in_path = root_dir / INPUT_FILE
    out_path = root_dir / OUTPUT_FILE
    
    if not in_path.exists():
        print(f"❌ 找不到输入文件: {in_path}")
        return

    print(f"🔄 正在加载: {in_path}")
    data_dict = joblib.load(in_path)
    
    # 2. 剥洋葱：获取内层核心数据
    # 根据你刚才的输出，外层 key 是文件名 'violin_left15_poses.npz'
    # 我们用 list(keys)[0] 动态获取它，防止文件名变了代码报错
    outer_key = list(data_dict.keys())[0]
    core_data = data_dict[outer_key]
    
    print(f"🔑 提取核心数据 Key: {outer_key}")
    
    # 3. 提取关键字段
    # (Frames, 19)
    dof_pos = core_data['dof'] 
    fps = core_data['fps']
    dt = 1.0 / fps
    
    print(f"📊 关节数据 Shape: {dof_pos.shape}")
    print(f"⏱️ 帧率: {fps} (dt={dt:.4f}s)")

    # 4. 计算关节速度 (Finite Difference)
    # 速度 = (位置_后一帧 - 位置_当前帧) / dt
    # 既然是模仿学习，我们可以简单的用差分计算目标速度
    dof_vel = np.zeros_like(dof_pos)
    # 前 N-1 帧
    dof_vel[:-1] = (dof_pos[1:] - dof_pos[:-1]) / dt
    # 最后一帧速度保持不变 (复制倒数第二帧)
    dof_vel[-1] = dof_vel[-2]

    # 5. 组装数据
    final_dict = {
        "dof_pos": dof_pos,   # (N, 19)
        "dof_vel": dof_vel,   # (N, 19)
        "dt": dt              # float
    }
    
    # 6. 保存
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, final_dict)
    
    print(f"✅ 转换成功！已保存至: {out_path}")
    print("👉 现在去修改 h1_config.py 中的 reference_motion_file 指向这个新文件吧！")

if __name__ == "__main__":
    convert()