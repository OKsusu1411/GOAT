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

    # Create scene
    def _setup_scene(self):
        # GOAT
        self.GOAT = Articulation(self.cfg.GOAT_cfg)
        self.scene.articuations["GOAT"] = self.GOAT

        # Ground Plane
        spawn_ground_plane(prim_path=self.cfg.plane.prim_path, cfg=GroundPlaneCfg(), translation=self.cfg.plane.init_state.pos)

        # Light
        light_cfg = self.cfg.dome_light.spawn
        light_cfg.func(self.cfg.dome_light.prim_path, light_cfg)


    ## =============== RL main methods ================ ##
    @abstractmethod
    # Reset envs
    def _reset_idx(self, env_ids: torch.Tensor):
        pass

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



        


