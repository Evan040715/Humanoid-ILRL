# import numpy as np
# import joblib
# import torch
# from scipy.spatial.transform import Rotation as R
# import os

# # === 配置路径 ===
# # 修改这里为你下载的 npz 文件路径
# INPUT_PATH = "resources/motions/amass_raw/CMU/141/141_16_poses.npz" 
# # 输出路径
# OUTPUT_PATH = "resources/motions/output/h1_141_16_wavehello_19dof.npy"

# # === H1 关节配置 (19 DoF) ===
# # 顺序必须与 URDF 文件中的关节顺序一致！
# # URDF顺序 (从 resources/robots/h1/urdf/h1.urdf):
# # 0-4:   [L_Hip_Yaw, L_Hip_Roll, L_Hip_Pitch, L_Knee, L_Ankle]
# # 5-9:   [R_Hip_Yaw, R_Hip_Roll, R_Hip_Pitch, R_Knee, R_Ankle]
# # 10:    [Torso]
# # 11-14: [L_Shoulder_Pitch, L_Shoulder_Roll, L_Shoulder_Yaw, L_Elbow]
# # 15-18: [R_Shoulder_Pitch, R_Shoulder_Roll, R_Shoulder_Yaw, R_Elbow]
# # 
# # 此顺序必须与 h1_config.py 中 default_joint_angles 字典的顺序一致
# # 目标帧率
# TARGET_DT = 0.02 

# def convert_cmu_to_h1():
#     print(f"Loading AMASS data from {INPUT_PATH}...")
#     try:
#         data = np.load(INPUT_PATH)
#     except FileNotFoundError:
#         print(f"❌ 错误: 找不到文件 {INPUT_PATH}，请确认路径是否正确。")
#         return

#     # AMASS 数据通常包含: 'poses', 'trans', 'dmpls'...
#     # poses shape: (Frames, 156) -> 52个关节 * 3轴角
#     # FPS 通常是 60 或 120，我们需要重采样到 dt=0.02 (50Hz)
#     poses = data['poses'] 
#     mocap_fps = data['mocap_framerate']
#     num_frames = poses.shape[0]
#     duration = num_frames / mocap_fps
    
#     print(f"原始数据: {num_frames} 帧, 帧率 {mocap_fps} Hz, 时长 {duration:.2f} 秒")

#     # === 1. 时间重采样 (Resampling) ===
#     # 我们的目标是 50Hz (dt=0.02)
#     target_num_frames = int(duration / TARGET_DT)
#     x_original = np.linspace(0, duration, num_frames)
#     x_target = np.linspace(0, duration, target_num_frames)
    
#     # 简单的线性插值函数
#     def resample(original_data):
#         # original_data: (Frames, D)
#         resampled = np.zeros((target_num_frames, original_data.shape[1]))
#         for i in range(original_data.shape[1]):
#             resampled[:, i] = np.interp(x_target, x_original, original_data[:, i])
#         return resampled

#     poses_resampled = resample(poses)
    
#     # === 2. 关节映射 (Mapping) ===
#     # SMPL 关节索引 (参考 SMPL 文档):
#     # 0: Pelvis (Root), 1: L_Hip, 2: R_Hip, 4: L_Knee, 5: R_Knee, 7: L_Ankle, 8: R_Ankle
#     # 12: L_Shoulder, 13: R_Shoulder, 16: L_Elbow, 17: R_Elbow
#     # 每个关节有 3 个值 (轴角 Axis-Angle)
    
#     # 创建 H1 的动作容器 (Frames, 19)
#     h1_dof_pos = np.zeros((target_num_frames, 19))
    
#     # --- 辅助函数: 轴角转欧拉角 ---
#     # SMPL 主要是以 X 轴为弯曲轴 (Pitch)
#     # H1 的关节定义比较复杂，但主要也是 Pitch
#     def get_euler(axis_angle, seq='xyz'):
#         r = R.from_rotvec(axis_angle)
#         return r.as_euler(seq, degrees=False)

#     print("正在转换关节角度...")
#     for f in range(target_num_frames):
#         # 提取当前帧的 SMPL 关节数据
#         # 格式: poses_resampled[f, start_idx : end_idx]
        
#         # --- 左腿 (Left Leg) ---
#         # 1. Left Hip (SMPL idx 1 -> indices 3:6)
#         l_hip_aa = poses_resampled[f, 3:6] 
#         l_hip_euler = get_euler(l_hip_aa, 'xyz') # 假设顺序，主要取 Pitch
#         # H1 Left Hip: Yaw(0), Roll(1), Pitch(2)
#         # 这是一个简化的映射，通常只需 Pitch 就能走起来
#         # SMPL 的 X 轴通常对应 Pitch
#         h1_dof_pos[f, 0] = l_hip_euler[2] * 0.5  # Yaw (大幅减小，防止乱扭)
#         h1_dof_pos[f, 1] = l_hip_euler[1] * 0.5  # Roll
#         h1_dof_pos[f, 2] = l_hip_euler[0] - 0.3 # Pitch (关键! 减去0.3是补偿H1的初始弯曲)

#         # 2. Left Knee (SMPL idx 4 -> indices 12:15)
#         l_knee_aa = poses_resampled[f, 12:15]
#         l_knee_euler = get_euler(l_knee_aa, 'xyz')
#         # H1 Knee: 只有 Pitch (正值代表弯曲)
#         # SMPL Knee 也是正值弯曲
#         h1_dof_pos[f, 3] = l_knee_euler[0] + 0.3 # 加上初始弯曲补偿

#         # 3. Left Ankle (SMPL idx 7 -> indices 21:24)
#         l_ankle_aa = poses_resampled[f, 21:24]
#         l_ankle_euler = get_euler(l_ankle_aa, 'xyz')
#         h1_dof_pos[f, 4] = l_ankle_euler[0] - 0.1 # 微调踝关节

#         # --- 右腿 (Right Leg) ---
#         # 1. Right Hip (SMPL idx 2 -> indices 6:9)
#         r_hip_aa = poses_resampled[f, 6:9]
#         r_hip_euler = get_euler(r_hip_aa, 'xyz')
#         h1_dof_pos[f, 5] = r_hip_euler[2] * 0.5 # Yaw
#         h1_dof_pos[f, 6] = r_hip_euler[1] * 0.5 # Roll
#         h1_dof_pos[f, 7] = r_hip_euler[0] - 0.3 # Pitch

#         # 2. Right Knee (SMPL idx 5 -> indices 15:18)
#         r_knee_aa = poses_resampled[f, 15:18]
#         r_knee_euler = get_euler(r_knee_aa, 'xyz')
#         h1_dof_pos[f, 8] = r_knee_euler[0] + 0.3

#         # 3. Right Ankle (SMPL idx 8 -> indices 24:27)
#         r_ankle_aa = poses_resampled[f, 24:27]
#         r_ankle_euler = get_euler(r_ankle_aa, 'xyz')
#         h1_dof_pos[f, 9] = r_ankle_euler[0] - 0.1

#         # --- 躯干 (Torso) ---
#         # SMPL Pelvis (idx 0 -> indices 0:3) 的旋转用于 torso
#         # 通常 torso 主要是 Yaw (Z轴旋转)
#         pelvis_aa = poses_resampled[f, 0:3]  # Root joint
#         pelvis_euler = get_euler(pelvis_aa, 'xyz')
#         h1_dof_pos[f, 10] = pelvis_euler[2] * 0.3  # Torso Yaw (减小幅度)

#         # --- 左臂 (Left Arm) ---
#         # 1. Left Shoulder (SMPL idx 12 -> indices 36:39)
#         l_shoulder_aa = poses_resampled[f, 36:39]
#         l_shoulder_euler = get_euler(l_shoulder_aa, 'xyz')
#         h1_dof_pos[f, 11] = l_shoulder_euler[1] * 0.8  # Pitch (前后摆动)
#         h1_dof_pos[f, 12] = l_shoulder_euler[0] * 0.5  # Roll (侧向)
#         h1_dof_pos[f, 13] = l_shoulder_euler[2] * 0.3  # Yaw (旋转)

#         # 2. Left Elbow (SMPL idx 16 -> indices 48:51)
#         l_elbow_aa = poses_resampled[f, 48:51]
#         l_elbow_euler = get_euler(l_elbow_aa, 'xyz')
#         h1_dof_pos[f, 14] = l_elbow_euler[1] * 0.8  # Elbow (主要弯曲)

#         # --- 右臂 (Right Arm) ---
#         # 1. Right Shoulder (SMPL idx 13 -> indices 39:42)
#         r_shoulder_aa = poses_resampled[f, 39:42]
#         r_shoulder_euler = get_euler(r_shoulder_aa, 'xyz')
#         h1_dof_pos[f, 15] = r_shoulder_euler[1] * 0.8  # Pitch
#         h1_dof_pos[f, 16] = r_shoulder_euler[0] * 0.5  # Roll
#         h1_dof_pos[f, 17] = r_shoulder_euler[2] * 0.3  # Yaw

#         # 2. Right Elbow (SMPL idx 17 -> indices 51:54)
#         r_elbow_aa = poses_resampled[f, 51:54]
#         r_elbow_euler = get_euler(r_elbow_aa, 'xyz')
#         h1_dof_pos[f, 18] = r_elbow_euler[1] * 0.8  # Elbow

#     # === 3. 保存为 .npy ===
#     data_dict = {
#         "dof_pos": h1_dof_pos,
#         "dof_vel": np.zeros_like(h1_dof_pos), # 速度设为0，让RL自己推导或忽略
#         "dt": TARGET_DT
#     }
    
#     os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
#     np.save(OUTPUT_PATH, data_dict)
#     print(f"✅ 转换完成! 文件已保存至: {OUTPUT_PATH}")
#     print(f"数据形状: {h1_dof_pos.shape}")

# if __name__ == "__main__":
#     convert_cmu_to_h1()





import numpy as np
import torch
import smplx
import os
from tqdm import tqdm

# === 配置路径 ===
# 输入：原始 AMASS 数据 (Z-Up)
INPUT_PATH = "resources/motions/amass_raw/CMU/07/07_01_poses.npz" 
# 输出：H1 格式动作
OUTPUT_PATH = "resources/motions/output/h1_07_01_walk_19dof.npy"
# SMPL 模型路径
SMPL_MODEL_PATH = "resources/smpl/SMPL_NEUTRAL.pkl"

# === H1 物理参数 (近似值, 单位: 米) ===
# 建议根据 URDF 微调这些值
H1_LINKS = {
    'torso_height': 0.42999,
    'shoulder_width': 0.31070,
    'upper_arm': 0.19886,
    'forearm': 0.30,   # 目前URDF里缺少可用于估计的wrist joint，先保留近似
    'thigh': 0.40,
    'shin': 0.40
}

# 目标帧率
TARGET_DT = 0.02 

class H1Kinematics(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 定义轴向 (根据 H1 URDF 定义)
        # 假设: Pitch(Y), Roll(X), Yaw(Z)
        self.axis_x = torch.tensor([1, 0, 0], dtype=torch.float32)
        self.axis_y = torch.tensor([0, 1, 0], dtype=torch.float32)
        self.axis_z = torch.tensor([0, 0, 1], dtype=torch.float32)

    def euler_to_rot_mat(self, angle, axis):
        """生成旋转矩阵 (Batch, 3, 3)"""
        # angle: (B,)
        # axis: (3,)
        B = angle.shape[0]
        c = torch.cos(angle)
        s = torch.sin(angle)
        
        # 罗德里格斯公式简化版 / 或者直接构建矩阵
        # 这里为了简单，针对 XYZ 单轴构建
        mat = torch.eye(3).repeat(B, 1, 1).to(angle.device)
        
        if torch.equal(axis, self.axis_x):
            mat[:, 1, 1] = c; mat[:, 1, 2] = -s
            mat[:, 2, 1] = s; mat[:, 2, 2] = c
        elif torch.equal(axis, self.axis_y):
            mat[:, 0, 0] = c; mat[:, 0, 2] = s
            mat[:, 2, 0] = -s; mat[:, 2, 2] = c
        elif torch.equal(axis, self.axis_z):
            mat[:, 0, 0] = c; mat[:, 0, 1] = -s
            mat[:, 1, 0] = s; mat[:, 1, 1] = c
            
        return mat

    def forward_arm(self, q_pitch, q_roll, q_yaw, q_elbow, side='left'):
        """
        简化的手臂 FK: 计算手腕相对于躯干中心(Torso)的位置
        """
        B = q_pitch.shape[0]
        device = q_pitch.device
        
        # 1. Torso -> Shoulder Base
        # 左肩向左(+Y), 右肩向右(-Y) (假设 Y 是左右, X 是前)
        # Wait: H1 URDF 通常 X向前, Y向左, Z向上
        y_sign = 1.0 if side == 'left' else -1.0
        offset_shoulder = torch.tensor([0, y_sign * H1_LINKS['shoulder_width'], H1_LINKS['torso_height']], device=device)
        
        # 2. Shoulder Rotation (Pitch -> Roll -> Yaw)
        R_pitch = self.euler_to_rot_mat(q_pitch, self.axis_y) 
        R_roll  = self.euler_to_rot_mat(q_roll,  self.axis_x) 
        R_yaw   = self.euler_to_rot_mat(q_yaw,   self.axis_z)
        
        R_shoulder = torch.bmm(R_pitch, torch.bmm(R_roll, R_yaw))
        
        # 3. Upper Arm 向量 (假设初始向下 -Z)
        vec_upper = torch.tensor([0, 0, -H1_LINKS['upper_arm']], device=device).repeat(B, 1).unsqueeze(-1)
        elbow_pos_rel = torch.bmm(R_shoulder, vec_upper).squeeze(-1)
        
        # 4. Elbow Rotation (Pitch)
        R_elbow = self.euler_to_rot_mat(q_elbow, self.axis_y)
        R_total = torch.bmm(R_shoulder, R_elbow)
        
        # 5. Forearm 向量
        vec_fore = torch.tensor([0, 0, -H1_LINKS['forearm']], device=device).repeat(B, 1).unsqueeze(-1)
        wrist_pos_rel = elbow_pos_rel + torch.bmm(R_total, vec_fore).squeeze(-1)
        
        return offset_shoulder + wrist_pos_rel

def run_ik_retargeting():
    # 1. 路径修复
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    full_input = os.path.join(project_root, INPUT_PATH)
    full_output = os.path.join(project_root, OUTPUT_PATH)
    full_smpl = os.path.join(project_root, SMPL_MODEL_PATH)

    # 2. 加载 SMPL 数据
    print("⏳ Loading SMPL Data...")
    smpl_raw = np.load(full_input)
    poses = torch.tensor(smpl_raw['poses'][:, :72], dtype=torch.float32)
    trans = torch.tensor(smpl_raw['trans'], dtype=torch.float32)
    betas = torch.tensor(smpl_raw['betas'][:10], dtype=torch.float32).unsqueeze(0)
    
    # 3. 计算 SMPL 关键点 (Ground Truth Targets)
    print("🦴 Calculating SMPL Keypoints...")
    smpl_layer = smplx.create(
        model_path=os.path.join(project_root, "resources"),
        model_type='smpl', gender='neutral', use_pca=False, batch_size=len(poses)
    )
    with torch.no_grad():
        output = smpl_layer(betas=betas.repeat(len(poses),1), body_pose=poses[:, 3:72], global_orient=poses[:, :3], transl=trans)
        joints = output.joints # (Frames, 45, 3)
        
        # 提取目标：相对于骨盆(idx 0)的手腕位置
        # SMPL idx: 20(Left Wrist), 21(Right Wrist), 0(Pelvis)
        # 注意：这里我们用相对位置，消除身高/全局位移的差异
        target_l_wrist = joints[:, 20, :] - joints[:, 0, :]
        target_r_wrist = joints[:, 21, :] - joints[:, 0, :]
        
        # 坐标系转换：SMPL(Z-up) -> H1(Z-up)
        # 通常不需要大改，但如果方向不对可能需要交换 XY
        # 这里假设直接匹配
    
    # 4. 初始化 IK 优化器
    print("🔧 Starting IK Optimization (This may take a minute)...")
    device = torch.device("cpu") # 简单的IK用CPU足够，GPU可能有overhead
    kinematics = H1Kinematics().to(device)
    
    # 初始化待优化的关节参数 (Frames, 4)
    # [Pitch, Roll, Yaw, Elbow]
    l_arm_params = torch.zeros(len(poses), 4, requires_grad=True, device=device)
    r_arm_params = torch.zeros(len(poses), 4, requires_grad=True, device=device)
    
    optimizer = torch.optim.Adam([l_arm_params, r_arm_params], lr=0.05)
    
    # 5. 优化循环
    target_l = target_l_wrist.to(device)
    target_r = target_r_wrist.to(device)
    
    iterations = 200 # 迭代次数
    pbar = tqdm(range(iterations))
    
    for i in pbar:
        optimizer.zero_grad()
        
        # 正向计算当前 H1 手腕位置
        pred_l = kinematics.forward_arm(
            l_arm_params[:, 0], l_arm_params[:, 1], l_arm_params[:, 2], l_arm_params[:, 3], side='left'
        )
        pred_r = kinematics.forward_arm(
            r_arm_params[:, 0], r_arm_params[:, 1], r_arm_params[:, 2], r_arm_params[:, 3], side='right'
        )
        
        # Loss 1: 位置误差 (Position Error)
        loss_pos = torch.mean((pred_l - target_l)**2) + torch.mean((pred_r - target_r)**2)
        
        # Loss 2: 正则化 (Regularization) - 防止关节扭成麻花
        # 鼓励关节保持在 0 附近 (Energy minimization)
        loss_reg = 0.01 * (torch.mean(l_arm_params**2) + torch.mean(r_arm_params**2))
        
        # Loss 3: 肘部约束 (Elbow Constraint)
        # H1 肘部只能单向弯曲 (0 ~ 2.6 rad), 不能反向折断
        loss_lim = torch.sum(torch.relu(-l_arm_params[:, 3])) + torch.sum(torch.relu(-r_arm_params[:, 3]))
        
        loss = loss_pos + loss_reg + loss_lim
        loss.backward()
        optimizer.step()
        
        pbar.set_description(f"Loss: {loss.item():.4f}")

    # 6. 组装最终数据
    # IK 算出了 Arm，腿部我们直接用 Heuristic (通常腿部用直接映射+Offset效果就不错)
    # 或者你也写一个 Leg FK 做 IK，但这里为了简单先混合使用
    
    print("📦 Assembling Data...")
    num_frames = len(poses)
    
    # 重采样到 50Hz (如果原始不是50Hz)
    mocap_fps = smpl_raw['mocap_framerate']
    target_len = int(num_frames / mocap_fps / TARGET_DT)
    
    # 简单的插值函数
    def resample_tensor(data):
        # data: (N, D)
        original_idx = np.linspace(0, num_frames-1, num_frames)
        target_idx = np.linspace(0, num_frames-1, target_len)
        res = np.zeros((target_len, data.shape[1]))
        for d in range(data.shape[1]):
            res[:, d] = np.interp(target_idx, original_idx, data[:, d].detach().numpy())
        return res

    # 提取优化后的手臂数据
    l_arm_opt = resample_tensor(l_arm_params)
    r_arm_opt = resample_tensor(r_arm_params)
    
    # 提取腿部数据 (使用之前的简单映射逻辑)
    # 这里需要重新读取原始 poses 进行重采样
    def get_euler_np(aa):
        from scipy.spatial.transform import Rotation as R
        return R.from_rotvec(aa).as_euler('xyz')
        
    # 我们需要对原始 SMPL 关节数据也做重采样才能匹配长度
    # 为了简化代码，这里直接处理原始 poses 数组
    poses_np = poses.numpy()
    poses_resampled = np.zeros((target_len, 72))
    for i in range(72):
         poses_resampled[:, i] = np.interp(np.linspace(0, num_frames, target_len), np.linspace(0, num_frames, num_frames), poses_np[:, i])

    h1_dof_pos = np.zeros((target_len, 19))
    
    for f in range(target_len):
        # --- Legs (Heuristic Mapping) ---
        # 沿用之前的经验公式，因为腿部直接映射通常比较稳
        l_hip_e = get_euler_np(poses_resampled[f, 3:6])
        h1_dof_pos[f, 0:3] = [l_hip_e[2]*0.5, l_hip_e[1]*0.5, l_hip_e[0]-0.2]
        
        l_knee_e = get_euler_np(poses_resampled[f, 12:15])
        h1_dof_pos[f, 3] = l_knee_e[0] + 0.2
        
        l_ankle_e = get_euler_np(poses_resampled[f, 21:24])
        h1_dof_pos[f, 4] = l_ankle_e[0] - 0.1
        
        r_hip_e = get_euler_np(poses_resampled[f, 6:9])
        h1_dof_pos[f, 5:8] = [r_hip_e[2]*0.5, r_hip_e[1]*0.5, r_hip_e[0]-0.2]
        
        r_knee_e = get_euler_np(poses_resampled[f, 15:18])
        h1_dof_pos[f, 8] = r_knee_e[0] + 0.2
        
        r_ankle_e = get_euler_np(poses_resampled[f, 24:27])
        h1_dof_pos[f, 9] = r_ankle_e[0] - 0.1
        
        # --- Torso ---
        pelvis_e = get_euler_np(poses_resampled[f, 0:3])
        h1_dof_pos[f, 10] = pelvis_e[2] * 0.5
        
        # --- Arms (IK Result) ---
        # 填入我们辛苦优化出来的 IK 结果
        # Left Arm: Pitch, Roll, Yaw, Elbow
        h1_dof_pos[f, 11:15] = l_arm_opt[f]
        # Right Arm
        h1_dof_pos[f, 15:19] = r_arm_opt[f]

    # 保存
    os.makedirs(os.path.dirname(full_output), exist_ok=True)
    np.save(full_output, {"dof_pos": h1_dof_pos, "dof_vel": np.zeros_like(h1_dof_pos), "dt": TARGET_DT})
    print(f"✅ IK Optimization Done! Saved to: {OUTPUT_PATH}")
    print("👉 Now check with visualize_h1_stick.py")

if __name__ == "__main__":
    run_ik_retargeting()