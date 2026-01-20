import numpy as np
import joblib
import torch
from scipy.spatial.transform import Rotation as R
import os

# === 配置路径 ===
# 修改这里为你下载的 npz 文件路径
INPUT_PATH = "resources/motions/input/01/01_01_poses.npz" 
# 输出路径
OUTPUT_PATH = "resources/motions/output/01/h1_cmu_jump_19dof.npy"

# === H1 关节配置 (19 DoF) ===
# 顺序必须与 URDF 文件中的关节顺序一致！
# URDF顺序 (从 resources/robots/h1/urdf/h1.urdf):
# 0-4:   [L_Hip_Yaw, L_Hip_Roll, L_Hip_Pitch, L_Knee, L_Ankle]
# 5-9:   [R_Hip_Yaw, R_Hip_Roll, R_Hip_Pitch, R_Knee, R_Ankle]
# 10:    [Torso]
# 11-14: [L_Shoulder_Pitch, L_Shoulder_Roll, L_Shoulder_Yaw, L_Elbow]
# 15-18: [R_Shoulder_Pitch, R_Shoulder_Roll, R_Shoulder_Yaw, R_Elbow]
# 
# 此顺序必须与 h1_config.py 中 default_joint_angles 字典的顺序一致
# 目标帧率
TARGET_DT = 0.02 

def convert_cmu_to_h1():
    print(f"Loading AMASS data from {INPUT_PATH}...")
    try:
        data = np.load(INPUT_PATH)
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {INPUT_PATH}，请确认路径是否正确。")
        return

    # AMASS 数据通常包含: 'poses', 'trans', 'dmpls'...
    # poses shape: (Frames, 156) -> 52个关节 * 3轴角
    # FPS 通常是 60 或 120，我们需要重采样到 dt=0.02 (50Hz)
    poses = data['poses'] 
    mocap_fps = data['mocap_framerate']
    num_frames = poses.shape[0]
    duration = num_frames / mocap_fps
    
    print(f"原始数据: {num_frames} 帧, 帧率 {mocap_fps} Hz, 时长 {duration:.2f} 秒")

    # === 1. 时间重采样 (Resampling) ===
    # 我们的目标是 50Hz (dt=0.02)
    target_num_frames = int(duration / TARGET_DT)
    x_original = np.linspace(0, duration, num_frames)
    x_target = np.linspace(0, duration, target_num_frames)
    
    # 简单的线性插值函数
    def resample(original_data):
        # original_data: (Frames, D)
        resampled = np.zeros((target_num_frames, original_data.shape[1]))
        for i in range(original_data.shape[1]):
            resampled[:, i] = np.interp(x_target, x_original, original_data[:, i])
        return resampled

    poses_resampled = resample(poses)
    
    # === 2. 关节映射 (Mapping) ===
    # SMPL 关节索引 (参考 SMPL 文档):
    # 0: Pelvis (Root), 1: L_Hip, 2: R_Hip, 4: L_Knee, 5: R_Knee, 7: L_Ankle, 8: R_Ankle
    # 12: L_Shoulder, 13: R_Shoulder, 16: L_Elbow, 17: R_Elbow
    # 每个关节有 3 个值 (轴角 Axis-Angle)
    
    # 创建 H1 的动作容器 (Frames, 19)
    h1_dof_pos = np.zeros((target_num_frames, 19))
    
    # --- 辅助函数: 轴角转欧拉角 ---
    # SMPL 主要是以 X 轴为弯曲轴 (Pitch)
    # H1 的关节定义比较复杂，但主要也是 Pitch
    def get_euler(axis_angle, seq='xyz'):
        r = R.from_rotvec(axis_angle)
        return r.as_euler(seq, degrees=False)

    print("正在转换关节角度...")
    for f in range(target_num_frames):
        # 提取当前帧的 SMPL 关节数据
        # 格式: poses_resampled[f, start_idx : end_idx]
        
        # --- 左腿 (Left Leg) ---
        # 1. Left Hip (SMPL idx 1 -> indices 3:6)
        l_hip_aa = poses_resampled[f, 3:6] 
        l_hip_euler = get_euler(l_hip_aa, 'xyz') # 假设顺序，主要取 Pitch
        # H1 Left Hip: Yaw(0), Roll(1), Pitch(2)
        # 这是一个简化的映射，通常只需 Pitch 就能走起来
        # SMPL 的 X 轴通常对应 Pitch
        h1_dof_pos[f, 0] = l_hip_euler[2] * 0.5  # Yaw (大幅减小，防止乱扭)
        h1_dof_pos[f, 1] = l_hip_euler[1] * 0.5  # Roll
        h1_dof_pos[f, 2] = l_hip_euler[0] - 0.3 # Pitch (关键! 减去0.3是补偿H1的初始弯曲)

        # 2. Left Knee (SMPL idx 4 -> indices 12:15)
        l_knee_aa = poses_resampled[f, 12:15]
        l_knee_euler = get_euler(l_knee_aa, 'xyz')
        # H1 Knee: 只有 Pitch (正值代表弯曲)
        # SMPL Knee 也是正值弯曲
        h1_dof_pos[f, 3] = l_knee_euler[0] + 0.3 # 加上初始弯曲补偿

        # 3. Left Ankle (SMPL idx 7 -> indices 21:24)
        l_ankle_aa = poses_resampled[f, 21:24]
        l_ankle_euler = get_euler(l_ankle_aa, 'xyz')
        h1_dof_pos[f, 4] = l_ankle_euler[0] - 0.1 # 微调踝关节

        # --- 右腿 (Right Leg) ---
        # 1. Right Hip (SMPL idx 2 -> indices 6:9)
        r_hip_aa = poses_resampled[f, 6:9]
        r_hip_euler = get_euler(r_hip_aa, 'xyz')
        h1_dof_pos[f, 5] = r_hip_euler[2] * 0.5 # Yaw
        h1_dof_pos[f, 6] = r_hip_euler[1] * 0.5 # Roll
        h1_dof_pos[f, 7] = r_hip_euler[0] - 0.3 # Pitch

        # 2. Right Knee (SMPL idx 5 -> indices 15:18)
        r_knee_aa = poses_resampled[f, 15:18]
        r_knee_euler = get_euler(r_knee_aa, 'xyz')
        h1_dof_pos[f, 8] = r_knee_euler[0] + 0.3

        # 3. Right Ankle (SMPL idx 8 -> indices 24:27)
        r_ankle_aa = poses_resampled[f, 24:27]
        r_ankle_euler = get_euler(r_ankle_aa, 'xyz')
        h1_dof_pos[f, 9] = r_ankle_euler[0] - 0.1

        # --- 躯干 (Torso) ---
        # SMPL Pelvis (idx 0 -> indices 0:3) 的旋转用于 torso
        # 通常 torso 主要是 Yaw (Z轴旋转)
        pelvis_aa = poses_resampled[f, 0:3]  # Root joint
        pelvis_euler = get_euler(pelvis_aa, 'xyz')
        h1_dof_pos[f, 10] = pelvis_euler[2] * 0.3  # Torso Yaw (减小幅度)

        # --- 左臂 (Left Arm) ---
        # 1. Left Shoulder (SMPL idx 12 -> indices 36:39)
        l_shoulder_aa = poses_resampled[f, 36:39]
        l_shoulder_euler = get_euler(l_shoulder_aa, 'xyz')
        h1_dof_pos[f, 11] = l_shoulder_euler[1] * 0.8  # Pitch (前后摆动)
        h1_dof_pos[f, 12] = l_shoulder_euler[0] * 0.5  # Roll (侧向)
        h1_dof_pos[f, 13] = l_shoulder_euler[2] * 0.3  # Yaw (旋转)

        # 2. Left Elbow (SMPL idx 16 -> indices 48:51)
        l_elbow_aa = poses_resampled[f, 48:51]
        l_elbow_euler = get_euler(l_elbow_aa, 'xyz')
        h1_dof_pos[f, 14] = l_elbow_euler[1] * 0.8  # Elbow (主要弯曲)

        # --- 右臂 (Right Arm) ---
        # 1. Right Shoulder (SMPL idx 13 -> indices 39:42)
        r_shoulder_aa = poses_resampled[f, 39:42]
        r_shoulder_euler = get_euler(r_shoulder_aa, 'xyz')
        h1_dof_pos[f, 15] = r_shoulder_euler[1] * 0.8  # Pitch
        h1_dof_pos[f, 16] = r_shoulder_euler[0] * 0.5  # Roll
        h1_dof_pos[f, 17] = r_shoulder_euler[2] * 0.3  # Yaw

        # 2. Right Elbow (SMPL idx 17 -> indices 51:54)
        r_elbow_aa = poses_resampled[f, 51:54]
        r_elbow_euler = get_euler(r_elbow_aa, 'xyz')
        h1_dof_pos[f, 18] = r_elbow_euler[1] * 0.8  # Elbow

    # === 3. 保存为 .npy ===
    data_dict = {
        "dof_pos": h1_dof_pos,
        "dof_vel": np.zeros_like(h1_dof_pos), # 速度设为0，让RL自己推导或忽略
        "dt": TARGET_DT
    }
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.save(OUTPUT_PATH, data_dict)
    print(f"✅ 转换完成! 文件已保存至: {OUTPUT_PATH}")
    print(f"数据形状: {h1_dof_pos.shape}")

if __name__ == "__main__":
    convert_cmu_to_h1()