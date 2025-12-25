# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import isaaclab.sim as sim_utils

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import DCMotorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg


# Robot asset paths
current_dir = os.path.dirname(__file__)
TRON_ASSET = {
    "urdf_path": os.path.join(current_dir, "../assets/TRON/WF_TRON1A/urdf/WF_TRON.urdf"),
    "usd_path": os.path.join(current_dir, "../assets/TRON/WF_TRON1A/usd/WF_TRON.usd"),
    "usd_place": os.path.join(current_dir, "../assets/TRON/WF_TRON1A/usd/"),
    "usd_filename": "WF_TRON.usd"
}
GOAT_ASSET = {
    "urdf_path": os.path.join(current_dir, "../assets/GOAT/WF_GOAT/urdf/WF_GOAT.urdf"),
    "usd_path": os.path.join(current_dir, "../assets/GOAT/WF_GOAT/usd/WF_GOAT.usd"),
    "usd_place": os.path.join(current_dir, "../assets/GOAT/WF_GOAT/usd/"),
    "usd_filename": "WF_GOAT.usd"
}

# URDF to USD conversion
urdf_cfg: sim_utils.UrdfConverterCfg = sim_utils.UrdfConverterCfg(
    root_link_name = "base_Link",
    asset_path = GOAT_ASSET["urdf_path"],
    usd_dir = GOAT_ASSET["usd_place"],
    usd_file_name = GOAT_ASSET["usd_filename"],
    fix_base=False,
    joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
        drive_type="force",
        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
    ),
)
urdf_converter = sim_utils.UrdfConverter(cfg = urdf_cfg)

# URDF conversion check
if urdf_converter.usd_path == GOAT_ASSET["usd_path"]:
    print("urdf conversion success!")
else:
    print("urdf conversion failed!")


GOAT_Cfg: ArticulationCfg = ArticulationCfg(
    # prim_path="{ENV_REGEX_NS}/Robot",               # Path for Interactivescene's clone_environemnts
    prim_path="/World/envs/env_.*/Robot",               # Path for DirectRLEnv
    spawn=sim_utils.UsdFileCfg(
        usd_path=urdf_converter.usd_path,
        scale=(1.0, 1.0, 1.0),
        activate_contact_sensors=True,
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
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            fix_root_link= False
        ),
    ),

    # Link, Joint list in Isaac sim
    # Link = ['base_Link', 'hip_L_Link', 'hip_R_Link', 'thigh_L_Link', 'thigh_R_Link', 'calf_L_Link', 'calf_R_Link', 'wheel_L_Link', 'wheel_R_Link']
    # Joint = ['hip_L_Joint', 'hip_R_Joint', 'thigh_L_Joint', 'thigh_R_Joint', 'knee_L_Joint', 'knee_R_Joint', 'wheel_L_Joint', 'wheel_R_Joint']
    
    # Initial Joint pos and vel
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
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
        ),

    # Actuators cfg
    actuators={
        "hip": DCMotorCfg(
            joint_names_expr=["hip_.*",],
            effort_limit=4.5,
            saturation_effort=4.5,
            velocity_limit=15.0,
            stiffness=0.0,                          # Internal PD controller not used
            damping=0.0,                            # Internal PD controller not used
            friction=0.0033,                        # Static friction coefficient
            dynamic_friction=5.646268e-02,          # Dynamic friction coefficient 
            viscous_friction=3.190248e-01,          # Viscous friction coefficient
        ),

        "thigh": DCMotorCfg(
            joint_names_expr=["thigh_.*",],
            effort_limit=4.5,
            saturation_effort=4.5,
            velocity_limit=15.0,
            stiffness=0.0,
            damping=0.32,
            friction=0.0033,
            dynamic_friction=5.646268e-02,
            viscous_friction=3.190248e-01,
        ),

        "knee": DCMotorCfg(
            joint_names_expr=["knee_.*",],
            effort_limit=4.5,
            saturation_effort=4.5,
            velocity_limit=15.0,
            stiffness=0.0,
            damping=0.0,
            friction=0.0235,
            dynamic_friction=4.432008e-01,
            viscous_friction=2.993308e-01,
        ),
        
        "wheel": DCMotorCfg(
            joint_names_expr=["wheel_.*"],
            effort_limit=2.5,
            saturation_effort=2.5,
            velocity_limit=15.0,
            stiffness=0.0,
            damping=0.0,
            friction=None,
            dynamic_friction=4.432008e-01,
            viscous_friction=2.993308e-01,
        )
    }
)

@configclass
class GOATBaseEnvCfg(DirectRLEnvCfg):
    # Env
    episode_length_s: int = 10       # Episode length in seconds
    sim_dt: float = 0.01             # Simulation(low-level controller) frequency
    decimation: int = 3              # Policy frequency = sim_freq / decimation
    action_space: int = 0            # Dimension of action space vector
    observation_space: int = 0       # Dimension of observation space vector
    state_space: int = 0             # Dimension of state space vector for privileged RL

    # Terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",        # Should apply terrain generator later
        env_spacing=3.0,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=0.8,
            restitution=0.0
        ),
        debug_vis=False
    )

    # Light
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=True)

    # GOAT cfg
    GOAT_cfg: ArticulationCfg = GOAT_Cfg