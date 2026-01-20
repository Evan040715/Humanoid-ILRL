import torch
import numpy as np
import smplx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# ================= 配置区域 =================
# 1. 原始数据路径 (你想看的那个文件)
NPZ_FILE = "resources/motions/amass_raw/CMU/07/07_01_poses.npz" 

# 2. SMPL 模型路径 (必须存在!)
MODEL_PATH = "resources/smpl/SMPL_NEUTRAL.pkl" 

# ===========================================

def visualize_smpl_data():
    # --- 1. 检查文件 ---
    if not os.path.exists(NPZ_FILE):
        print(f"❌ 错误: 找不到数据文件 {NPZ_FILE}")
        return
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 找不到模型文件 {MODEL_PATH}")
        print("请下载 SMPL_NEUTRAL.pkl 并修改脚本中的路径！")
        return

    # --- 2. 加载数据 ---
    print(f"正在加载数据: {NPZ_FILE}")
    data = np.load(NPZ_FILE)
    
    # 提取关键数据
    # AMASS 数据通常是 (Frames, 156) 或者 (Frames, 72)
    # 只要取前 72 个 (24关节 * 3) 即可驱动标准 SMPL
    poses = torch.tensor(data['poses'][:, :72], dtype=torch.float32)
    trans = torch.tensor(data['trans'], dtype=torch.float32) # 根节点位移
    betas = torch.tensor(data['betas'][:10], dtype=torch.float32).unsqueeze(0) # 体型
    
    n_frames = poses.shape[0]
    print(f"帧数: {n_frames}, 正在计算骨骼位置 (Forward Kinematics)...")

    # --- 3. 加载 SMPL 模型并计算关节位置 ---
    # 使用 smplx 库自动计算正向运动学
    smpl_layer = smplx.create(
        model_path="resources",
        model_type='smpl',
        gender='neutral',
        use_pca=False,
        batch_size=1
    )

    # 我们需要批量计算，或者逐帧计算。为了显存安全，我们逐帧计算并将关节位置存下来。
    # joints_seq: [Frames, 24, 3]
    joints_seq = []
    
    # 降采样：如果帧数太多，每隔几帧取一帧，加快可视化
    skip = 2 
    frames_to_show = range(0, n_frames, skip)
    
    with torch.no_grad():
        for i in frames_to_show:
            output = smpl_layer(
                betas=betas,
                global_orient=poses[i:i+1, :3], # Root rotation
                body_pose=poses[i:i+1, 3:72],   # Body rotation
                transl=trans[i:i+1]             # Root translation
            )
            # output.joints 通常有 45 个点，前 24 个是标准 SMPL 骨架
            joints_seq.append(output.joints[0, :24, :].numpy())
            
    joints_seq = np.array(joints_seq) # Shape: (T, 24, 3)

    # --- 4. 定义 SMPL 骨架连接关系 (用于画线) ---
    # 格式: (Parent, Child) 索引
    kinematic_tree = [
        (0, 1), (0, 2), (0, 3),       # Pelvis -> L_Hip, R_Hip, Spine1
        (1, 4), (2, 5), (3, 6),       # Hips -> Knees, Spine1 -> Spine2
        (4, 7), (5, 8), (6, 9),       # Knees -> Ankles, Spine2 -> Spine3
        (7, 10), (8, 11), (9, 12),    # Ankles -> Toes, Spine3 -> Neck
        (12, 13), (12, 14), (12, 15), # Neck -> Head, L_Collar, R_Collar
        (14, 16), (15, 17),           # Collars -> Shoulders
        (16, 18), (17, 19),           # Shoulders -> Elbows
        (18, 20), (19, 21),           # Elbows -> Wrists
        (20, 22), (21, 23)            # Wrists -> Hands
    ]

    # --- 5. Matplotlib 可视化 ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"Source SMPL Motion: {os.path.basename(NPZ_FILE)}")

    # 自动设置视野范围
    all_x = joints_seq[:, :, 0].flatten()
    all_y = joints_seq[:, :, 1].flatten()
    all_z = joints_seq[:, :, 2].flatten()
    ax.set_xlim3d([np.min(all_x)-0.5, np.max(all_x)+0.5])
    ax.set_ylim3d([np.min(all_y)-0.5, np.max(all_y)+0.5])
    ax.set_zlim3d([np.min(all_z)-0.5, np.max(all_z)+0.5])
    
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    # 初始化线条
    lines = [ax.plot([], [], [], 'o-', lw=2, markersize=3)[0] for _ in range(len(kinematic_tree))]
    
    # 颜色装饰
    # 左边(索引包含1,4,7...): 红色 / 右边: 绿色 / 中间: 黑色
    # 简单起见，统一蓝色
    for line in lines:
        line.set_color('blue')

    def update(frame_idx):
        current_joints = joints_seq[frame_idx]
        
        for i, (parent, child) in enumerate(kinematic_tree):
            # 获取两个点的坐标
            p1 = current_joints[parent]
            p2 = current_joints[child]
            
            lines[i].set_data_3d(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                [p1[2], p2[2]]
            )
        return lines

    ani = animation.FuncAnimation(fig, update, frames=len(joints_seq), interval=30, blit=False)
    
    print("🎥 播放动画中...")
    plt.show()

if __name__ == "__main__":
    visualize_smpl_data()