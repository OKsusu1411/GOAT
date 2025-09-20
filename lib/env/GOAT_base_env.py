# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

#from __future__ import annotations

import torch

from abc import abstractmethod
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.controllers.joint_impedance import JointImpedanceController
from isaaclab.controllers import DifferentialIKController
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform
from .GOAT_base_env_cfg import GOATBaseEnvCfg

class GOATBaseEnv(DirectRLEnv):
    # Load config file
    cfg: GOATBaseEnvCfg

    def __init__(self, cfg: GOATBaseEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total env ids
        self.total_env_ids = torch.arange(self.num_envs, device=self.device)
        
        # Joint Index
        self.joint_idx = self._robot.find_joints(".*_Joint")[0]

        # Joint & Link Limits
        self.robot_dof_lower_limits = self._robot.data.joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.joint_pos_limits[0, :, 1].to(device=self.device)
        
        # unit tensors
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float32, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float32, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.device).repeat((self.num_envs, 1))

    # Create scene
    def _setup_scene(self):
        # GOAT Robot
        self._robot = Articulation(self.cfg.GOAT_cfg)
        self.scene.articulations["robot"] = self._robot

        # Ground Plane
        spawn_ground_plane(prim_path=self.cfg.plane.prim_path, cfg=GroundPlaneCfg(), translation=self.cfg.plane.init_state.pos)

        # Light
        light_cfg = self.cfg.dome_light.spawn
        light_cfg.func(self.cfg.dome_light.prim_path, light_cfg)


    # Reset Env
    def _reset_idx(self, env_ids: torch.Tensor):
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)

        # Publish to simulator
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


    ## =============== RL main methods ================ ##

    @abstractmethod
    # Before physics step
    def _pre_physics_step(self, actions: torch.Tensor):
        pass

    @abstractmethod
    # Apply action
    def _apply_action(self):
        pass

    @abstractmethod
    # Get observation
    def _get_observations(self) -> dict[str, torch.Tensor]:
        pass

    @abstractmethod
    # Get reward
    def _get_rewards(self) -> torch.Tensor:
        pass



        


