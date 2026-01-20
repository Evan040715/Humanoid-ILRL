import os
import numpy as np
from scipy.spatial.transform import Rotation as R

# Convert AMASS/SMPL(.npz) -> G1 23DoF reference motion (.npy)
#
# Output format matches what `legged_gym/envs/g1/g1_env.py` expects:
#   {"dof_pos": (T,23), "dof_vel": (T,23), "dt": float}
#
# Joint order MUST match `resources/robots/g1_description/g1_23dof.urdf` revolute order:
# 0  left_hip_pitch_joint
# 1  left_hip_roll_joint
# 2  left_hip_yaw_joint
# 3  left_knee_joint
# 4  left_ankle_pitch_joint
# 5  left_ankle_roll_joint
# 6  right_hip_pitch_joint
# 7  right_hip_roll_joint
# 8  right_hip_yaw_joint
# 9  right_knee_joint
# 10 right_ankle_pitch_joint
# 11 right_ankle_roll_joint
# 12 waist_yaw_joint
# 13 left_shoulder_pitch_joint
# 14 left_shoulder_roll_joint
# 15 left_shoulder_yaw_joint
# 16 left_elbow_joint
# 17 left_wrist_roll_joint
# 18 right_shoulder_pitch_joint
# 19 right_shoulder_roll_joint
# 20 right_shoulder_yaw_joint
# 21 right_elbow_joint
# 22 right_wrist_roll_joint

INPUT_PATH = "resources/motions/amass_raw/CMU/07/07_01_poses.npz"
OUTPUT_PATH = "resources/motions/output/g1_07_01_walk_23dof.npy"
TARGET_DT = 0.02


def _resample_linear(x_original, x_target, data_2d):
    out = np.zeros((len(x_target), data_2d.shape[1]), dtype=np.float64)
    for i in range(data_2d.shape[1]):
        out[:, i] = np.interp(x_target, x_original, data_2d[:, i])
    return out


def _aa_to_euler(axis_angle, seq="xyz"):
    return R.from_rotvec(axis_angle).as_euler(seq, degrees=False)


def convert_cmu_to_g1_23dof():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    in_path = os.path.join(project_root, INPUT_PATH)
    out_path = os.path.join(project_root, OUTPUT_PATH)

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"AMASS npz not found: {in_path}")

    data = np.load(in_path)
    poses = data["poses"]  # (Frames, 156)
    mocap_fps = float(data["mocap_framerate"])
    num_frames = poses.shape[0]
    duration = num_frames / mocap_fps

    target_num_frames = int(duration / TARGET_DT)
    x_original = np.linspace(0, duration, num_frames)
    x_target = np.linspace(0, duration, target_num_frames)
    poses_resampled = _resample_linear(x_original, x_target, poses)

    dof_pos = np.zeros((target_num_frames, 23), dtype=np.float32)

    # NOTE:
    # This is a *heuristic* mapping. It is good enough to bootstrap imitation learning,
    # but for high-quality retargeting you typically need full kinematic retargeting.
    for f in range(target_num_frames):
        # ---- legs ----
        # SMPL: pelvis(0), l_hip(1), r_hip(2), l_knee(4), r_knee(5), l_ankle(7), r_ankle(8)
        l_hip_e = _aa_to_euler(poses_resampled[f, 3:6], "xyz")
        r_hip_e = _aa_to_euler(poses_resampled[f, 6:9], "xyz")
        l_knee_e = _aa_to_euler(poses_resampled[f, 12:15], "xyz")
        r_knee_e = _aa_to_euler(poses_resampled[f, 15:18], "xyz")
        l_ankle_e = _aa_to_euler(poses_resampled[f, 21:24], "xyz")
        r_ankle_e = _aa_to_euler(poses_resampled[f, 24:27], "xyz")

        # g1 hip order is pitch, roll, yaw
        dof_pos[f, 0] = l_hip_e[0] - 0.2
        dof_pos[f, 1] = l_hip_e[1] * 0.5
        dof_pos[f, 2] = l_hip_e[2] * 0.5
        dof_pos[f, 3] = l_knee_e[0] + 0.2
        dof_pos[f, 4] = l_ankle_e[0] - 0.1
        dof_pos[f, 5] = l_ankle_e[1] * 0.3

        dof_pos[f, 6] = r_hip_e[0] - 0.2
        dof_pos[f, 7] = r_hip_e[1] * 0.5
        dof_pos[f, 8] = r_hip_e[2] * 0.5
        dof_pos[f, 9] = r_knee_e[0] + 0.2
        dof_pos[f, 10] = r_ankle_e[0] - 0.1
        dof_pos[f, 11] = r_ankle_e[1] * 0.3

        # ---- waist yaw ----
        pelvis_e = _aa_to_euler(poses_resampled[f, 0:3], "xyz")
        dof_pos[f, 12] = pelvis_e[2] * 0.3

        # ---- arms (SMPL: l_shoulder 12, r_shoulder 13, l_elbow 16, r_elbow 17) ----
        l_sh_e = _aa_to_euler(poses_resampled[f, 36:39], "xyz")
        r_sh_e = _aa_to_euler(poses_resampled[f, 39:42], "xyz")
        l_el_e = _aa_to_euler(poses_resampled[f, 48:51], "xyz")
        r_el_e = _aa_to_euler(poses_resampled[f, 51:54], "xyz")

        # shoulder pitch/roll/yaw
        dof_pos[f, 13] = l_sh_e[1] * 0.8
        dof_pos[f, 14] = l_sh_e[0] * 0.5
        dof_pos[f, 15] = l_sh_e[2] * 0.3
        dof_pos[f, 16] = l_el_e[1] * 0.8
        dof_pos[f, 17] = 0.0  # wrist roll (not in SMPL) -> keep neutral

        dof_pos[f, 18] = r_sh_e[1] * 0.8
        dof_pos[f, 19] = r_sh_e[0] * 0.5
        dof_pos[f, 20] = r_sh_e[2] * 0.3
        dof_pos[f, 21] = r_el_e[1] * 0.8
        dof_pos[f, 22] = 0.0  # wrist roll

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, {"dof_pos": dof_pos, "dof_vel": np.zeros_like(dof_pos), "dt": float(TARGET_DT)})
    print(f"✅ Saved: {out_path}")
    print(f"dof_pos shape: {dof_pos.shape}, dt={TARGET_DT}")


if __name__ == "__main__":
    convert_cmu_to_g1_23dof()


