import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import os

# ================= 配置区域 =================
# 你想看的 pkl 文件路径
# INPUT_FILE = "resources/feedforward_data/wave_right15_poses.pkl"
INPUT_FILE = "resources/feedforward_data/violin_left15_poses.pkl"
# 播放速度 (1.0 = 正常速度, 0.5 = 慢放)
PLAYBACK_SPEED = 0.5
# ===========================================

# H1 机器人简易骨骼长度 (单位: 米) - 估算值，用于可视化足够了
LINK_LENGTHS = {
    'thigh': 0.40,
    'shin': 0.40,
    'torso': 0.20,
    'shoulder_width': 0.15, # 单侧宽度
    'upper_arm': 0.30,
    'forearm': 0.30
}

def get_h1_fk(dof_pos, root_pos, root_rot):
    """
    简易正向运动学 (FK)：根据关节角度计算 H1 关键点坐标
    dof_pos: (19,)
    root_pos: (3,)
    root_rot: (4,) [x, y, z, w]
    """
    # 1. 基础旋转矩阵 (根节点)
    base_r = R.from_quat(root_rot).as_matrix() # (3, 3)
    base_p = root_pos # (3,)

    # 定义关键点字典
    joints = {}
    joints['pelvis'] = base_p

    # 辅助函数：计算局部点在世界坐标系的位置
    # parent_pos: 父节点世界坐标
    # parent_rot: 父节点旋转矩阵 (为了简单，这里我们只考虑根节点旋转和简单的局部偏移)
    # 真实的 FK 需要每一级连乘，这里为了简化，假设所有肢体主要受根节点方向控制 + 自身摆动
    # 这种近似对于可视化动作意图（比如抬手、迈腿）是足够的
    
    def apply_rot(vec, rot_matrix):
        return rot_matrix @ vec

    # --- 🦵 下半身 (Legs) ---
    # 19 DoF 顺序: 
    # [0-4] Left Leg: Hip(Yaw, Roll, Pitch), Knee, Ankle
    # [5-9] Right Leg
    # [10] Torso
    # [11-14] Left Arm
    # [15-18] Right Arm
    
    # 左腿
    l_hip_offset = np.array([0, 0.1, -0.05]) # 髋关节相对于骨盆的偏移
    joints['l_hip'] = base_p + apply_rot(l_hip_offset, base_r)
    
    # 简化计算：膝盖位置 = 髋 + 大腿向量(受Hip Pitch控制)
    # 这里做了一个非常简化的假设：主要看 Hip Pitch (idx 2) 和 Knee (idx 3)
    l_thigh_vec = np.array([0, 0, -LINK_LENGTHS['thigh']])
    # 绕 Y 轴旋转 (Pitch)
    l_pitch = dof_pos[2]
    r_pitch_mat = R.from_euler('y', l_pitch).as_matrix()
    joints['l_knee'] = joints['l_hip'] + apply_rot(r_pitch_mat @ l_thigh_vec, base_r)
    
    l_shin_vec = np.array([0, 0, -LINK_LENGTHS['shin']])
    l_knee_angle = dof_pos[3]
    r_knee_mat = R.from_euler('y', l_pitch + l_knee_angle).as_matrix()
    joints['l_ankle'] = joints['l_knee'] + apply_rot(r_knee_mat @ l_shin_vec, base_r)

    # 右腿
    r_hip_offset = np.array([0, -0.1, -0.05])
    joints['r_hip'] = base_p + apply_rot(r_hip_offset, base_r)
    
    r_pitch = dof_pos[7]
    r_pitch_mat = R.from_euler('y', r_pitch).as_matrix()
    r_thigh_vec = np.array([0, 0, -LINK_LENGTHS['thigh']])
    joints['r_knee'] = joints['r_hip'] + apply_rot(r_pitch_mat @ r_thigh_vec, base_r)
    
    r_knee_angle = dof_pos[8]
    r_knee_mat = R.from_euler('y', r_pitch + r_knee_angle).as_matrix()
    r_shin_vec = np.array([0, 0, -LINK_LENGTHS['shin']])
    joints['r_ankle'] = joints['r_knee'] + apply_rot(r_knee_mat @ r_shin_vec, base_r)

    # --- 👕 躯干 (Torso) ---
    torso_vec = np.array([0, 0, LINK_LENGTHS['torso']])
    # 简单假设 Torso Joint (idx 10, Yaw) 影响不大，直接向上
    joints['torso'] = base_p + apply_rot(torso_vec, base_r)
    
    # --- 💪 手臂 (Arms) ---
    # Left Arm [11-14]: Shoulder(Pitch, Roll, Yaw), Elbow
    l_shoulder_offset = np.array([0, LINK_LENGTHS['shoulder_width'], 0.1])
    joints['l_shoulder'] = joints['torso'] + apply_rot(l_shoulder_offset, base_r)
    
    # 左臂 FK
    # Pitch(11, Y轴), Roll(12, X轴), Yaw(13, Z轴)
    l_s_pitch, l_s_roll, l_s_yaw = dof_pos[11], dof_pos[12], dof_pos[13]
    # 复合旋转
    l_arm_rot = R.from_euler('yxz', [l_s_pitch, l_s_roll, l_s_yaw]).as_matrix()
    l_upper_vec = np.array([0, 0, -LINK_LENGTHS['upper_arm']]) # 假设初始向下
    joints['l_elbow'] = joints['l_shoulder'] + apply_rot(l_arm_rot @ l_upper_vec, base_r)
    
    # Elbow (14, Pitch)
    l_elbow_angle = dof_pos[14]
    l_fore_rot = R.from_euler('yxz', [l_s_pitch + l_elbow_angle, l_s_roll, l_s_yaw]).as_matrix()
    l_fore_vec = np.array([0, 0, -LINK_LENGTHS['forearm']])
    joints['l_hand'] = joints['l_elbow'] + apply_rot(l_fore_rot @ l_fore_vec, base_r)

    # Right Arm [15-18]
    r_shoulder_offset = np.array([0, -LINK_LENGTHS['shoulder_width'], 0.1])
    joints['r_shoulder'] = joints['torso'] + apply_rot(r_shoulder_offset, base_r)
    
    r_s_pitch, r_s_roll, r_s_yaw = dof_pos[15], dof_pos[16], dof_pos[17]
    r_arm_rot = R.from_euler('yxz', [r_s_pitch, r_s_roll, r_s_yaw]).as_matrix()
    r_upper_vec = np.array([0, 0, -LINK_LENGTHS['upper_arm']])
    joints['r_elbow'] = joints['r_shoulder'] + apply_rot(r_arm_rot @ r_upper_vec, base_r)
    
    r_elbow_angle = dof_pos[18]
    r_fore_rot = R.from_euler('yxz', [r_s_pitch + r_elbow_angle, r_s_roll, r_s_yaw]).as_matrix()
    r_fore_vec = np.array([0, 0, -LINK_LENGTHS['forearm']])
    joints['r_hand'] = joints['r_elbow'] + apply_rot(r_fore_rot @ r_fore_vec, base_r)

    return joints

def run_visualization():
    # 1. 健壮的路径处理
    root_dir = Path(__file__).resolve().parent.parent
    file_path = root_dir / INPUT_FILE
    
    if not file_path.exists():
        print(f"❌ 找不到文件: {file_path}")
        return

    print(f"🔄 加载数据: {file_path}")
    data_dict = joblib.load(file_path)
    
    # 提取核心数据
    key = list(data_dict.keys())[0]
    core = data_dict[key]
    
    dof_seq = core['dof']          # (T, 19)
    root_pos_seq = core['root_trans_offset'] # (T, 3)
    root_rot_seq = core['root_rot'] # (T, 4)
    fps = core['fps']
    
    print(f"📊 数据帧数: {len(dof_seq)}, FPS: {fps}")

    # 2. Matplotlib 设置
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"H1 Motion: {os.path.basename(INPUT_FILE)}")
    
    # 定义连线关系
    bones = [
        ('pelvis', 'l_hip'), ('l_hip', 'l_knee'), ('l_knee', 'l_ankle'), # 左腿
        ('pelvis', 'r_hip'), ('r_hip', 'r_knee'), ('r_knee', 'r_ankle'), # 右腿
        ('pelvis', 'torso'), # 脊柱
        ('torso', 'l_shoulder'), ('l_shoulder', 'l_elbow'), ('l_elbow', 'l_hand'), # 左臂
        ('torso', 'r_shoulder'), ('r_shoulder', 'r_elbow'), ('r_elbow', 'r_hand')  # 右臂
    ]
    
    lines = [ax.plot([], [], [], 'o-', lw=2, markersize=4)[0] for _ in range(len(bones))]

    # 设置视野范围 (根据根节点轨迹自动调整)
    mid_x = np.mean(root_pos_seq[:, 0])
    mid_y = np.mean(root_pos_seq[:, 1])
    mid_z = np.mean(root_pos_seq[:, 2]) + 0.5
    range_w = 1.0
    
    ax.set_xlim(mid_x - range_w, mid_x + range_w)
    ax.set_ylim(mid_y - range_w, mid_y + range_w)
    ax.set_zlim(0, 2.0) # 高度通常在 0-2米
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    def update(frame):
        # 降采样播放速度
        idx = int(frame) % len(dof_seq)
        
        # 计算当前帧的骨架位置
        joints = get_h1_fk(dof_seq[idx], root_pos_seq[idx], root_rot_seq[idx])
        
        for line, (start_joint, end_joint) in zip(lines, bones):
            p1 = joints[start_joint]
            p2 = joints[end_joint]
            
            line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
            line.set_3d_properties([p1[2], p2[2]])
            
            # 颜色装饰: 左红右蓝
            if 'l_' in start_joint or 'l_' in end_joint:
                line.set_color('red')
            elif 'r_' in start_joint or 'r_' in end_joint:
                line.set_color('blue')
            else:
                line.set_color('black')
                
        return lines

    ani = animation.FuncAnimation(fig, update, frames=len(dof_seq), 
                                  interval=(1000/fps)/PLAYBACK_SPEED, blit=False)
    
    print("🎥 窗口已弹出，正在播放...")
    plt.show()

if __name__ == "__main__":
    run_visualization()