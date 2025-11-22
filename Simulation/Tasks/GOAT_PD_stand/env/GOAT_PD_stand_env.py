import torch

from __future__ import annotations
from isaaclab.utils.math import sample_uniform
from .GOAT_PD_stand_env_cfg import GOATPDStandEnvCfg
from lib.env.GOAT_base_env import GOATBaseEnv
from lib.low_level_controller.pd_controller import PD_Controller


class GOATIKStandEnv(GOATBaseEnv):
    cfg: GOATPDStandEnvCfg

    def __init__(self, cfg: GOATPDStandEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Curriculum level initialization
        self.curriculum_level = 0
        self.max_curriculum_level = cfg.max_curriculum_level - 1
        self.base_angular_vel_noise_per = torch.linspace(start=0, end=cfg.max_base_angular_vel_noise_per, steps=self.max_curriculum_level)
        self.gravity_vector_noise_per = torch.linspace(start=0, end=cfg.max_gravity_vector_noise_per, steps=self.max_curriculum_level)
        self.joint_pos_noise_per = torch.linspace(start=0, end=cfg.max_joint_pos_noise_per, steps=self.max_curriculum_level)
        self.joint_vel_noise_per = torch.linspace(start=0, end=cfg.max_joint_vel_noise_per, steps=self.max_curriculum_level)

        # Space initialization
        self.observation = torch.zeros((self.num_envs, self.cfg.observation_space), dtype=torch.float32, device=self.device)
        self.state = torch.zeros((self.num_envs, self.cfg.state_space), dtype=torch.float32, device=self.device)

        self.leg_controller = PD_Controller(kp=self.cfg.kp,
                                            kd=self.cfg.kd,
                                            num_envs=self.num_envs,
                                            num_dof=self.cfg.leg_dof,
                                            device=self.device,
                                            dt=self.cfg.sim_dt)

    def _reset_idx(self, env_ids: torch.Tensor):
        pos_noise = sample_uniform(
            -0.125, 0.125,
            (len(env_ids), self.num_active_joints),
            self.device,)
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_pos = joint_pos[:, :-2] + pos_noise
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)

        # Publish to simulator
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        Preprocessor that helps applying policy's action to simulation

        Args:
            actions (torch.Tensor): Joint pos command (angle), wheel's velocity for each legs in shape (num_envs, 2, 4)
        """
        # Refine command
        self.actions = actions.clone()
        self.joint_pos_cmd = self.actions[:, :, :3]
        self.wheel_cmd_vel = self.actions[:, :, 3:]

        # Current state
        self.joint_pos = self._robot.data.joint_pos
        self.joint_vel = self._robot.data.joint_vel

        # Domain Randomization
        self.joint_pos = self._add_gaussian_noise(self.joint_pos, )

        # HW limits
        self.joint_limits = self._robot.data.joint_pos_limits
        self.torque_limits = self._robot.data.joint_effort_limits

    def _apply_action(self):
        self.torque_cmd = self.leg_controller.compute_torque(joint_pos=self.joint_pos,
                                                             joint_vel=self.joint_vel,
                                                             joint_pos_cmd=self.joint_pos_cmd,
                                                             joint_limits=self.joint_limits,
                                                             torque_limits=self.torque_limits)

        # TODO: wheel controller
        self._robot.set_joint_effort_target(self.torque_cmd)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self.observation
        self.state
        return {"actor": self.observation, "critic": self.state}
    
    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
    
    def _get_dones(self):
        return torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
    
    def _add_gaussian_noise(data: torch.Tensor, noise_percentage: int):
        """
        Add (noise_percentage)% noise to all components of data
        """
        noise_ratio = noise_percentage / 100.0
        
        # Standard normal distribution 
        noise = torch.randn_like(data)
        
        noisy_data = data * (1 + noise_ratio * noise)
        
        return noisy_data
