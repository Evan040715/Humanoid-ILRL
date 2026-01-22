import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import isaacgym
from isaacgym import gymapi, gymutil
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import tkinter as tk
from tkinter import ttk
import threading

# ================= GUI 类定义 =================
class RobotControlGUI:
    def __init__(self, dof_names, default_angles, action_scale, limits=None):
        self.root = tk.Tk()
        self.root.title("H1 Robot Joint Control")
        self.root.geometry("400x800")
        
        self.action_scale = action_scale
        self.default_angles = default_angles # dict or list
        self.sliders = {}
        self.dof_names = dof_names
        self.values = {} # 存储当前的目标弧度值

        # 创建滚动区域
        canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 标题
        ttk.Label(self.scrollable_frame, text="Joint Targets (Radians)", font=("Arial", 12, "bold")).pack(pady=10)
        
        # 重置按钮
        ttk.Button(self.scrollable_frame, text="Reset to Default", command=self.reset_sliders).pack(pady=5)

        # 为每个关节创建滑块
        for i, name in enumerate(dof_names):
            frame = ttk.Frame(self.scrollable_frame)
            frame.pack(fill='x', padx=5, pady=2)
            
            # 标签
            lbl = ttk.Label(frame, text=f"{name}:", width=25, anchor='w')
            lbl.pack(side='left')
            
            # 获取默认值
            default_val = 0.0
            if isinstance(default_angles, dict):
                # 尝试模糊匹配或直接匹配
                default_val = default_angles.get(name, 0.0)
            elif isinstance(default_angles, (list, np.ndarray, torch.Tensor)):
                default_val = float(default_angles[i])

            self.values[name] = default_val

            # 设置范围 (如果没有提供 limits，默认 -3.14 到 3.14)
            lower = -3.14
            upper = 3.14
            if limits is not None:
                lower = float(limits[i][0])
                upper = float(limits[i][1])

            # 滑块
            scale = tk.Scale(frame, from_=lower, to=upper, orient='horizontal', 
                             resolution=0.01, length=150, showvalue=True)
            scale.set(default_val)
            scale.pack(side='right')
            
            # 绑定事件
            scale.config(command=lambda val, n=name: self.update_value(n, val))
            self.sliders[name] = scale

    def update_value(self, name, val):
        self.values[name] = float(val)

    def reset_sliders(self):
        for i, name in enumerate(self.dof_names):
            default_val = 0.0
            if isinstance(self.default_angles, dict):
                default_val = self.default_angles.get(name, 0.0)
            elif isinstance(self.default_angles, (list, np.ndarray, torch.Tensor)):
                default_val = float(self.default_angles[i])
            
            self.sliders[name].set(default_val)
            self.values[name] = default_val

    def get_actions(self, device='cpu'):
        """
        计算 Action。
        公式: Target = Action * scale + Default
        所以: Action = (Target - Default) / scale
        """
        actions = []
        for i, name in enumerate(self.dof_names):
            target = self.values[name]
            
            default_val = 0.0
            if isinstance(self.default_angles, dict):
                default_val = self.default_angles.get(name, 0.0)
            elif isinstance(self.default_angles, (list, np.ndarray, torch.Tensor)):
                default_val = float(self.default_angles[i])
            
            # 计算 action
            act = (target - default_val) / self.action_scale
            actions.append(act)
        
        return torch.tensor(actions, dtype=torch.float, device=device).unsqueeze(0) # [1, num_dofs]

    def update(self):
        self.root.update()

# ================= 主程序 =================

def play_with_gui(args):
    # 1. 加载配置和环境
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # 强制修改配置以适应 GUI 调试
    env_cfg.env.num_envs = 1  # 只开一个环境
    env_cfg.domain_rand.push_robots = False #哪怕你配置里写了推机器人，调试时也关掉
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False

    # 准备环境
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    # 重置环境
    obs = env.reset()

    # 2. 获取关节信息
    dof_names = env.dof_names
    default_dof_pos = env.default_dof_pos[0] # tensor
    action_scale = env_cfg.control.action_scale
    dof_limits = env.dof_pos_limits # [num_dof, 2]

    print(f"Loaded {len(dof_names)} joints.")
    print(f"Action Scale: {action_scale}")

    # 3. 初始化 GUI
    # 将 Tensor 转为 CPU 列表方便 GUI 使用
    gui = RobotControlGUI(
        dof_names=dof_names, 
        default_angles=default_dof_pos.cpu().numpy(), 
        action_scale=action_scale,
        limits=dof_limits.cpu().numpy()
    )

    print("\n✅ GUI 已启动！请在弹出的窗口中拖动滑块来控制机器人关节。\n")

    # 4. 仿真循环
    while True:
        # A. 刷新 GUI (处理鼠标点击事件)
        gui.update()

        # B. 从 GUI 获取当前的 Action
        # 我们这里直接覆盖 Policy，用 GUI 的值作为 action
        gui_actions = gui.get_actions(device=env.device)

        # C. 仿真一步
        # obs, _, rews, dones, infos = env.step(gui_actions)
        
        # 注意：env.step 通常包含 internal physics step + observation computation
        # 这里的 gui_actions 已经是处理过的 (target - default) / scale
        obs, privileged_obs, rewards, dofs, infos = env.step(gui_actions)

        # 如果需要，可以在这里打印 obs 查看数值
        # print(env.dof_pos[0]) 

if __name__ == '__main__':
    # 解析参数 (使用 legged_gym 标准参数)
    args = get_args()
    # 强制指定 task 为 h1 (或者你可以通过命令行传参)
    
    args.task = "h1" 
    
    play_with_gui(args)