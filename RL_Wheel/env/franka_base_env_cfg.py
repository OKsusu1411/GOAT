# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.controllers.joint_impedance import JointImpedanceControllerCfg


OBJECT_DIR = {
    "table":{
        "url": "/table.usd"
    },

    "stand": {
        "url": "/stand.usd"
    }
}


@configclass
class FrankaBaseEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s: int
    decimation: int
    action_space: int
    observation_space: int
    state_space: int

    # ground plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -0.612]),
        spawn=GroundPlaneCfg(),
    )

    # light
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)

    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=12, solver_velocity_iteration_count=0),
        # collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            "panda_finger_joint.*": 0.04,
        }
        ))
    # Impedance Controller를 사용하는 경우, 액추에이터 PD제어 모델 사용 X (중복 토크 계산)
    # 액추에이터에 Impedance Controller가 붙음으로써 최하단 제어기의 역할을 하게 되는 개념.
    robot.actuators["panda_shoulder"].stiffness = 0.0
    robot.actuators["panda_shoulder"].damping = 0.0
    robot.actuators["panda_forearm"].stiffness = 0.0
    robot.actuators["panda_forearm"].damping = 0.0

    goal_pos_marker_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="Visuals/goal_marker",
        markers={
        "frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.1, 0.1, 0.1),
            )
        }
    )

    # TCP marker
    tcp_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="Visuals/TCP_current",
        markers={
        "frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.1, 0.1, 0.1),
            )
        }
    )

    # stand
    stand = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Stand",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[1.0, 0, 0, 0.0]),
        spawn=sim_utils.UsdFileCfg(usd_path=os.path.join(os.getcwd(), "Dataset", "mydata") + OBJECT_DIR["stand"]["url"],
                                   scale=(1.2, 1.2, 1.2),
                                   ),
    )

    # Table
    table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.67, 0.0, -0.3], rot=[1.0, 0, 0, 0.0]),
        spawn=sim_utils.UsdFileCfg(usd_path=os.path.join(os.getcwd(), "Dataset", "mydata") + OBJECT_DIR["table"]["url"],
                                   collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                                   scale=(0.7, 1.0, 0.6),
                                   semantic_tags=[("class", "table")]
                                   ),
    )
    
    # Joint Impedance controller
    imp_controller: JointImpedanceControllerCfg = JointImpedanceControllerCfg(
        command_type="p_abs",
        impedance_mode="variable",
        stiffness=300.0,
        damping_ratio=0.5,
        stiffness_limits=(30, 300),
        damping_ratio_limits=(0, 1),
        inertial_compensation=True,
        gravity_compensation=True,)

    
    # Scene entities
    robot_entity: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])

    # Joint Control Action scale
    loc_res_scale = 0.1 
    rot_res_scale = 0.1
    joint_res_clipping = 0.2
    stiffness_scale = imp_controller.stiffness_limits[1]
    damping_scale = imp_controller.damping_ratio_limits[1]
    gripper_scale = 0.04

    # target point reset
    reset_position_noise_x = 0.1
    reset_position_noise_y = 0.2