import torch

from __future__ import annotations
from isaaclab.utils.math import sample_uniform
from isaaclab.controllers import DifferentialIKControllerCfg
from .GOAT_stand_env_cfg import GOATStandEnvCfg
from lib.env.GOAT_base_env import GOATBaseEnv
from lib.low_level_controller.ik_pd_controller import IK_PD_Controller


class GOATStandEnv(GOATBaseEnv):
    cfg: GOATStandEnvCfg

    def __init__(self, cfg: GOATStandEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.observation = torch.zeros((self.num_envs, self.cfg.observation_space), dtype=torch.float32, device=self.device)
        self.state = torch.zeros((self.num_envs, self.cfg.state_space), dtype=torch.float32, device=self.device)

        diff_ik_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls")
        self.leg_controller = IK_PD_Controller(diff_ik_cfg=diff_ik_cfg, 
                                               kp=self.cfg.kp,
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
            actions (torch.Tensor): Foot delta position, wheel's velocity for each legs in shape (N, 2, 4)
        """
        # Refine command
        self.actions = actions.clone()
        self.delta_foot_pos = self.actions[:, :, :3]
        self.wheel_cmd_vel = self.actions[:, :, 3:]
        self.link_pose = self._robot.data.body_link_pose_w
        self.target_pos = self.link_pose[:, 7:, :3] + self.delta_foot_pos

        # Current state
        self.joint_pos = self._robot.data.joint_pos
        self.joint_vel = self._robot.data.joint_vel
        self.jacobian = self._robot.root_physx_view.get_jacobians()

        # HW limits
        self.joint_limits = self._robot.data.joint_pos_limits
        self.torque_limits = self._robot.data.joint_effort_limits

    def _apply_action(self):
        self.torque_cmd, _ = self.leg_controller.compute_torque(link_pose=self.link_pose,
                                                                joint_pos=self.joint_pos,
                                                                joint_vel=self.joint_vel,
                                                                foot_cmd=self.target_pos,
                                                                joint_limits=self.joint_limits,
                                                                torque_limits=self.torque_limits,
                                                                jacobian=self.jacobian)

        # TODO: wheel controller
        self._robot.set_joint_effort_target(self.torque_cmd)

    def _get_observations(self) -> dict[str, torch.Tensor]:
         
        return {"actor": self.observation, "critic": self.state}
    
    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
    
    def _get_dones(self):
        return torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        
