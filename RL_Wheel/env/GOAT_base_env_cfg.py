# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import isaaclab.sim as sim_utils

from __future__ import annotations
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.controllers.joint_impedance import JointImpedanceControllerCfg

current_dir = os.path.dirname(__file__)
TRON_usdpath = os.path.join(current_dir, "../../TRON_description/WF_TRON1A/WF_TRON1A.usd")
GOAT_usdpath = os.path.join(current_dir, "../../")

@configclass
class GOATBaseEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s: int
    decimation: int
    action_space: int
    observation_space: int
    state_space: int

    # ground plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0]),
        spawn=GroundPlaneCfg(),
    )

    # light
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)

# ================= TODO: 로봇 cfg 따로 만들기==========================
    # GOAT cfg
    GOAT_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"../../TRON_description/WF_TRON1A/WF_TRON1A.usd",
            translation=[0.0, 0.0, 0.0],
            orientation=[0.0, 0.0, 0.0],
            scale=1.0,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
          ),
          articulation_props=sim_utils.ArticulationRootPropertiesCfg(
              enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
          ),
        ),

        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "hip_L_Joint": 0.0,
                "hip_R_Joint": 0.0,
                "thigh_L_Joint": 0.0,
                "thigh_R_Joint": 0.0,
                "knee_L_Joint": 0.0,
                "knee_R_Joint": 0.0,
                "wheel_L_Joint": 0.0,
                "wheel_R_Joint": 0.0,
            },
            joint_vel={".*": 0.0},
        ),

        actuators={
            "leg": DCMotorCfg(
                joint_names_expr=["hip_.*", "thigh_.*", "knee_.*",],       # parameter reference from TRON
                effort_limit=4.5,
                saturation_effort=4.5,
                velocity_limit=15.0,
                stiffness=40.0,
                damping=2.5,
                friction=0.0,

            ),
            "wheel": DCMotorCfg(
                joint_names_expr=["wheel_.*"],
                effort_limit=2.5,
                saturation_effort=2.5,
                velocity_limit=15.0,
                stiffness=0.0,
                damping=0.8,
                friction=0.0,
            )
        }
    )

    imp_controller: JointImpedanceControllerCfg = JointImpedanceControllerCfg(
        command_type="p_abs",
        impedance_mode="variable",
        stiffness=300.0,
        damping_ratio=0.5,
        stiffness_limits=(30, 300),
        damping_ratio_limits=(0, 1),
        inertial_compensation=True,
        gravity_compensation=True,)