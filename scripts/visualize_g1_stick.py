import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R

# Quick stick-figure preview for G1 23DoF reference motion files.
INPUT_FILE = "resources/motions/output/g1_07_01_walk_23dof.npy"

# Approximate lengths for visualization only (meters)
LINK_LENS = {
    "torso": 0.30,
    "thigh": 0.45,
    "shin": 0.45,
    "shoulder_width": 0.10,  # half-width
    "upper_arm": 0.25,
    "forearm": 0.22,
}


def _rot(angles, seq="xyz"):
    return R.from_euler(seq, angles, degrees=False).as_matrix()


def fk_approx(q):
    # q: (23,)
    j = {}
    root = np.array([0.0, 0.0, 1.0])
    j["root"] = root

    # waist yaw (12)
    r_torso = _rot([0, 0, q[12]])
    neck = root + r_torso @ np.array([0, 0, LINK_LENS["torso"]])
    j["neck"] = neck

    # legs: use hip pitch/roll/yaw indices (0..5, 6..11)
    # left hip base offset +Y
    l_hip_base = root + np.array([0, 0.10, 0])
    r_l_hip = _rot([q[1], q[0], q[2]], "yxz")  # approx
    l_knee = l_hip_base + r_l_hip @ np.array([0, 0, -LINK_LENS["thigh"]])
    r_l_knee = r_l_hip @ _rot([0, q[3], 0])
    l_ankle = l_knee + r_l_knee @ np.array([0, 0, -LINK_LENS["shin"]])
    j["l_knee"] = l_knee
    j["l_ankle"] = l_ankle

    r_hip_base = root + np.array([0, -0.10, 0])
    r_r_hip = _rot([q[7], q[6], q[8]], "yxz")
    r_knee = r_hip_base + r_r_hip @ np.array([0, 0, -LINK_LENS["thigh"]])
    r_r_knee = r_r_hip @ _rot([0, q[9], 0])
    r_ankle = r_knee + r_r_knee @ np.array([0, 0, -LINK_LENS["shin"]])
    j["r_knee"] = r_knee
    j["r_ankle"] = r_ankle

    # arms: shoulder pitch/roll/yaw + elbow (wrist ignored)
    l_sh = neck + np.array([0, LINK_LENS["shoulder_width"], 0])
    r_sh = neck + np.array([0, -LINK_LENS["shoulder_width"], 0])
    j["l_shoulder"] = l_sh
    j["r_shoulder"] = r_sh

    r_l_sh = _rot([q[14], q[13], q[15]], "yxz")
    l_elbow = l_sh + r_l_sh @ np.array([0, 0, -LINK_LENS["upper_arm"]])
    r_l_el = r_l_sh @ _rot([0, q[16], 0])
    l_hand = l_elbow + r_l_el @ np.array([0, 0, -LINK_LENS["forearm"]])
    j["l_elbow"] = l_elbow
    j["l_hand"] = l_hand

    r_r_sh = _rot([q[19], q[18], q[20]], "yxz")
    r_elbow = r_sh + r_r_sh @ np.array([0, 0, -LINK_LENS["upper_arm"]])
    r_r_el = r_r_sh @ _rot([0, q[21], 0])
    r_hand = r_elbow + r_r_el @ np.array([0, 0, -LINK_LENS["forearm"]])
    j["r_elbow"] = r_elbow
    j["r_hand"] = r_hand

    return j


def run():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    path = os.path.join(project_root, INPUT_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")

    data = np.load(path, allow_pickle=True).item()
    q_seq = data["dof_pos"]
    dt = float(data["dt"])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"G1 Motion Preview: {os.path.basename(INPUT_FILE)}")
    ax.set_xlim3d([-0.8, 0.8])
    ax.set_ylim3d([-0.8, 0.8])
    ax.set_zlim3d([0.0, 1.8])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    lines = [ax.plot([], [], [], "o-", lw=2)[0] for _ in range(5)]
    lines[0].set_color("blue")   # left leg
    lines[1].set_color("cyan")   # right leg
    lines[2].set_color("black")  # torso
    lines[3].set_color("red")    # left arm
    lines[4].set_color("green")  # right arm

    def update(i):
        j = fk_approx(q_seq[i])
        # left leg
        lines[0].set_data_3d(
            [j["root"][0], j["root"][0], j["l_knee"][0], j["l_ankle"][0]],
            [j["root"][1], j["root"][1] + 0.1, j["l_knee"][1], j["l_ankle"][1]],
            [j["root"][2], j["root"][2], j["l_knee"][2], j["l_ankle"][2]],
        )
        # right leg
        lines[1].set_data_3d(
            [j["root"][0], j["root"][0], j["r_knee"][0], j["r_ankle"][0]],
            [j["root"][1], j["root"][1] - 0.1, j["r_knee"][1], j["r_ankle"][1]],
            [j["root"][2], j["root"][2], j["r_knee"][2], j["r_ankle"][2]],
        )
        # torso
        lines[2].set_data_3d(
            [j["root"][0], j["neck"][0]],
            [j["root"][1], j["neck"][1]],
            [j["root"][2], j["neck"][2]],
        )
        # left arm
        lines[3].set_data_3d(
            [j["neck"][0], j["l_shoulder"][0], j["l_elbow"][0], j["l_hand"][0]],
            [j["neck"][1], j["l_shoulder"][1], j["l_elbow"][1], j["l_hand"][1]],
            [j["neck"][2], j["l_shoulder"][2], j["l_elbow"][2], j["l_hand"][2]],
        )
        # right arm
        lines[4].set_data_3d(
            [j["neck"][0], j["r_shoulder"][0], j["r_elbow"][0], j["r_hand"][0]],
            [j["neck"][1], j["r_shoulder"][1], j["r_elbow"][1], j["r_hand"][1]],
            [j["neck"][2], j["r_shoulder"][2], j["r_elbow"][2], j["r_hand"][2]],
        )
        return lines

    # IMPORTANT: keep a reference to the animation object, otherwise it may get garbage-collected
    # before rendering (you'll see: "Animation was deleted without rendering anything.")
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=q_seq.shape[0],
        interval=max(int(dt * 1000), 20),
        blit=False,
    )

    plt.show()
    return ani


if __name__ == "__main__":
    _ani = run()


