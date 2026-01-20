
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import os
import numpy as np
import torch

class H1Robot(LeggedRobot):
    
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions
        noise_vec[9+3*self.num_actions:9+3*self.num_actions+2] = 0. # sin/cos phase
        
        return noise_vec

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)
        
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()

        # === IMITATION LEARNING: Load Motion Data ===
        self.ref_dof_pos = None
        self.ref_dof_vel = None
        self.motion_len = 0
        self.motion_dt = None

        ref_file = getattr(self.cfg.env, "reference_motion_file", None)
        if ref_file:
            # allow relative paths (relative to repo root)
            ref_path = ref_file
            if not os.path.isabs(ref_path):
                ref_path = os.path.join(LEGGED_GYM_ROOT_DIR, ref_path)

            print(f"Loading reference motion from {ref_path}")
            loaded_data = np.load(ref_path, allow_pickle=True).item()

            # Expect dict keys: "dof_pos", "dof_vel", "dt"
            self.ref_dof_pos = torch.tensor(loaded_data["dof_pos"], device=self.device, dtype=torch.float)
            self.ref_dof_vel = torch.tensor(loaded_data["dof_vel"], device=self.device, dtype=torch.float)
            self.motion_len = int(self.ref_dof_pos.shape[0])
            self.motion_dt = float(loaded_data["dt"])

            if self.motion_len <= 0:
                raise ValueError(f"Reference motion has no frames: {ref_path}")
            if self.motion_dt <= 0:
                raise ValueError(f"Reference motion dt must be > 0, got {self.motion_dt}: {ref_path}")

            print(f"Motion loaded: {self.motion_len} frames, dt={self.motion_dt}")

    def _get_ref_state(self):
        """Return reference DOF position/velocity for current sim time for each env.

        Returns:
            ref_pos: (num_envs, num_dof)
            ref_vel: (num_envs, num_dof)
        """
        if self.ref_dof_pos is None or self.ref_dof_vel is None or self.motion_len <= 0 or self.motion_dt is None:
            # Fallback: no reference loaded
            return self.dof_pos, self.dof_vel

        # current time per env [s]
        phase_t = self.episode_length_buf.to(dtype=torch.float) * self.dt
        motion_idx = (phase_t / self.motion_dt).to(dtype=torch.long)

        if getattr(self.cfg.env, "reference_loop", True):
            motion_idx = motion_idx % self.motion_len
        else:
            motion_idx = torch.clamp(motion_idx, 0, self.motion_len - 1)

        return self.ref_dof_pos[motion_idx], self.ref_dof_vel[motion_idx]

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _post_physics_step_callback(self):
        self.update_feet_state()

        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
        
        return super()._post_physics_step_callback()
    
    
    def compute_observations(self):
        """ Computes observations
        """
        sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1)
        self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        
    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res
    
    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact
        return torch.sum(pos_error, dim=(1))
    
    def _reward_alive(self):
        # Reward for staying alive
        return 1.0
    
    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))
    
    def _reward_hip_pos(self):
        # 全身17 DOF: 索引0,1是左髋yaw/roll, 5,6是右髋yaw/roll
        # 如果num_actions=17，这些索引仍然有效
        if self.num_actions >= 7:
            return torch.sum(torch.square(self.dof_pos[:,[0,1,5,6]]), dim=1)
        else:
            # 10 DOF模式：只有腿部关节
            return torch.sum(torch.square(self.dof_pos[:,[0,1,5,6]]), dim=1)

    def _reward_tracking_joint_pos(self):
        """Imitation reward: track reference joint positions with Gaussian kernel."""
        ref_pos, _ = self._get_ref_state()
        # sum squared error over joints
        error = torch.sum(torch.square(self.dof_pos - ref_pos), dim=1)
        sigma = float(getattr(self.cfg.rewards, "tracking_sigma", 0.25))
        # avoid divide-by-zero
        sigma = max(sigma, 1e-6)
        return torch.exp(-error / sigma)

    # Optional: if you later add scale `tracking_lin_vel` and want to use it as imitation term
    def _reward_tracking_lin_vel(self):
        """Imitation reward: track reference base linear velocity if reference provides it."""
        # Not available in current reference format; keep as safe fallback.
        return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
    