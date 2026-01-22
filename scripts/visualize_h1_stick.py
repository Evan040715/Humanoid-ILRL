import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R
import os

# === 配置 ===
# 这里填你刚刚生成的 .npy 文件路径
INPUT_FILE = "resources/motions/output/from_pkl/violin_h1_19dof.npy"

# === H1 机器人近似尺寸 (单位: 米) ===
# 这些是硬编码的近似值，只为画火柴人，不代表物理真实
LINK_LENS = {
    'torso': 0.25,
    'thigh': 0.4,
    'shin': 0.4,
    'shoulder_width': 0.2, # 单侧宽度
    'upper_arm': 0.3,
    'forearm': 0.3
}

def forward_kinematics_approx(dof_pos):
    """
    简易正向运动学：根据 19个关节角度计算关键关节点 (x,y,z)
    注意：这是近似计算，假设了标准的旋转顺序，仅用于预览动作意图。
    """
    # 关节索引映射 (参考你的 convert 脚本)
    # 0-4:   L_Leg [Yaw, Roll, Pitch, Knee, Ankle]
    # 5-9:   R_Leg [Yaw, Roll, Pitch, Knee, Ankle]
    # 10:    Torso [Yaw]
    # 11-14: L_Arm [S_Pitch, S_Roll, S_Yaw, Elbow]
    # 15-18: R_Arm [S_Pitch, S_Roll, S_Yaw, Elbow]

    # 初始化所有关节点坐标
    joints = {}
    
    # 1. 根节点 (Pelvis) - 假设固定在空中某个高度，方便观察
    root_pos = np.array([0.0, 0.0, 1.0]) 
    joints['root'] = root_pos

    # 辅助函数：根据欧拉角创建旋转矩阵
    def get_rot(angles, seq='xyz'):
        return R.from_euler(seq, angles).as_matrix()

    # --- 躯干 (Torso) ---
    # 只有 Yaw (idx 10)
    # H1 Torso joint 主要是 Yaw
    r_torso = get_rot([0, 0, dof_pos[10]]) 
    # 躯干向上延伸
    neck_pos = root_pos + r_torso @ np.array([0, 0, LINK_LENS['torso']])
    joints['neck'] = neck_pos

    # --- 左腿 (Left Leg) ---
    # Hip: Yaw(0), Roll(1), Pitch(2)
    r_l_hip = get_rot([dof_pos[1], dof_pos[2], dof_pos[0]], 'yxz') # 顺序近似
    l_hip_base = root_pos + np.array([0, 0.1, 0]) # 髋部稍微偏左
    # Knee
    l_knee_pos = l_hip_base + r_l_hip @ np.array([0, 0, -LINK_LENS['thigh']])
    joints['l_knee'] = l_knee_pos
    # Ankle (Knee joint idx 3)
    r_l_knee = r_l_hip @ get_rot([0, dof_pos[3], 0], 'xyz') # 膝盖只有 Pitch
    l_ankle_pos = l_knee_pos + r_l_knee @ np.array([0, 0, -LINK_LENS['shin']])
    joints['l_ankle'] = l_ankle_pos

    # --- 右腿 (Right Leg) ---
    r_r_hip = get_rot([dof_pos[6], dof_pos[7], dof_pos[5]], 'yxz')
    r_hip_base = root_pos + np.array([0, -0.1, 0]) 
    r_knee_pos = r_hip_base + r_r_hip @ np.array([0, 0, -LINK_LENS['thigh']])
    joints['r_knee'] = r_knee_pos
    r_r_knee = r_r_hip @ get_rot([0, dof_pos[8], 0], 'xyz')
    r_ankle_pos = r_knee_pos + r_r_knee @ np.array([0, 0, -LINK_LENS['shin']])
    joints['r_ankle'] = r_ankle_pos

    # --- 左臂 (Left Arm) ---
    # Shoulder: Pitch(11), Roll(12), Yaw(13)
    # 初始手臂向下
    l_shoulder_base = neck_pos + np.array([0, LINK_LENS['shoulder_width'], 0])
    joints['l_shoulder'] = l_shoulder_base
    
    r_l_shoulder = get_rot([dof_pos[12], dof_pos[11], dof_pos[13]], 'yxz')
    l_elbow_pos = l_shoulder_base + r_l_shoulder @ np.array([0, 0, -LINK_LENS['upper_arm']])
    joints['l_elbow'] = l_elbow_pos
    
    # Elbow (idx 14)
    r_l_elbow = r_l_shoulder @ get_rot([0, dof_pos[14], 0], 'xyz')
    l_hand_pos = l_elbow_pos + r_l_elbow @ np.array([0, 0, -LINK_LENS['forearm']])
    joints['l_hand'] = l_hand_pos

    # --- 右臂 (Right Arm) ---
    # Shoulder: Pitch(15), Roll(16), Yaw(17)
    r_shoulder_base = neck_pos + np.array([0, -LINK_LENS['shoulder_width'], 0])
    joints['r_shoulder'] = r_shoulder_base
    
    r_r_shoulder = get_rot([dof_pos[16], dof_pos[15], dof_pos[17]], 'yxz')
    r_elbow_pos = r_shoulder_base + r_r_shoulder @ np.array([0, 0, -LINK_LENS['upper_arm']])
    joints['r_elbow'] = r_elbow_pos
    
    # Elbow (idx 18)
    r_r_elbow = r_r_shoulder @ get_rot([0, dof_pos[18], 0], 'xyz')
    r_hand_pos = r_elbow_pos + r_r_elbow @ np.array([0, 0, -LINK_LENS['forearm']])
    joints['r_hand'] = r_hand_pos

    return joints

def run_visualization():
    # 1. 加载数据
    # 获取绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    file_path = os.path.join(project_root, INPUT_FILE)

    if not os.path.exists(file_path):
        print(f"❌ 找不到文件: {file_path}")
        return

    print(f"Loading: {file_path}")
    data = np.load(file_path, allow_pickle=True).item()
    dof_pos_seq = data['dof_pos'] # (Frames, 19)
    num_frames = dof_pos_seq.shape[0]
    dt = data['dt']

    print(f"Total Frames: {num_frames}, DT: {dt}")

    # 2. 设置 Matplotlib 3D 绘图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"H1 Motion Preview: {os.path.basename(INPUT_FILE)}")

    # 设置视角和范围
    ax.set_xlim3d([-0.8, 0.8])
    ax.set_ylim3d([-0.8, 0.8])
    ax.set_zlim3d([0.0, 1.8])
    ax.set_xlabel('X (Forward)')
    ax.set_ylabel('Y (Side)')
    ax.set_zlabel('Z (Up)')

    # 初始化线段 (火柴人的骨架)
    # 我们定义几条连接线: 
    # 左腿链, 右腿链, 脊柱链, 左臂链, 右臂链
    lines = [ax.plot([], [], [], 'o-', lw=2)[0] for _ in range(5)]
    # 颜色区分: 腿(蓝), 躯干(黑), 左臂(红), 右臂(绿)
    lines[0].set_color('blue')   # 左腿
    lines[1].set_color('cyan')   # 右腿
    lines[2].set_color('black')  # 躯干 (Root -> Neck)
    lines[3].set_color('red')    # 左臂
    lines[4].set_color('green')  # 右臂

    # 3. 动画更新函数
    def update(frame):
        # 获取当前帧的关节角度
        current_dof = dof_pos_seq[frame]
        
        # 计算坐标
        j = forward_kinematics_approx(current_dof)
        
        # 定义连线逻辑
        # 线0: Root -> L_Hip -> L_Knee -> L_Ankle
        lines[0].set_data_3d(
            [j['root'][0], j['root'][0], j['l_knee'][0], j['l_ankle'][0]],
            [j['root'][1], j['root'][1]+0.1, j['l_knee'][1], j['l_ankle'][1]],
            [j['root'][2], j['root'][2], j['l_knee'][2], j['l_ankle'][2]]
        )
        
        # 线1: Root -> R_Hip -> R_Knee -> R_Ankle
        lines[1].set_data_3d(
            [j['root'][0], j['root'][0], j['r_knee'][0], j['r_ankle'][0]],
            [j['root'][1], j['root'][1]-0.1, j['r_knee'][1], j['r_ankle'][1]],
            [j['root'][2], j['root'][2], j['r_knee'][2], j['r_ankle'][2]]
        )

        # 线2: Root -> Neck
        lines[2].set_data_3d(
            [j['root'][0], j['neck'][0]],
            [j['root'][1], j['neck'][1]],
            [j['root'][2], j['neck'][2]]
        )

        # 线3: Neck -> L_Shoulder -> L_Elbow -> L_Hand
        lines[3].set_data_3d(
            [j['neck'][0], j['l_shoulder'][0], j['l_elbow'][0], j['l_hand'][0]],
            [j['neck'][1], j['l_shoulder'][1], j['l_elbow'][1], j['l_hand'][1]],
            [j['neck'][2], j['l_shoulder'][2], j['l_elbow'][2], j['l_hand'][2]]
        )

        # 线4: Neck -> R_Shoulder -> R_Elbow -> R_Hand
        lines[4].set_data_3d(
            [j['neck'][0], j['r_shoulder'][0], j['r_elbow'][0], j['r_hand'][0]],
            [j['neck'][1], j['r_shoulder'][1], j['r_elbow'][1], j['r_hand'][1]],
            [j['neck'][2], j['r_shoulder'][2], j['r_elbow'][2], j['r_hand'][2]]
        )

        return lines

    # 创建动画
    # interval 根据 dt 设定，但为了观看通常设慢一点 (比如 50ms)
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)

    print("🎥 播放动画中... (请查看弹出的窗口)")
    plt.show()

if __name__ == "__main__":
    run_visualization()