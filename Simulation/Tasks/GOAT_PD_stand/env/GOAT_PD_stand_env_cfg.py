import isaaclab.sim as sim_utils
import gymnasium
import torch

from isaaclab.sim import SimulationCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import DCMotorCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from lib.env.GOAT_base_env_cfg import GOATBaseEnvCfg
from gymnasium.spaces import Dict

@configclass
class GOATPDStandEnvCfg(GOATBaseEnvCfg):
    ## ==================== Environment parameters ==================== ##
    episode_length_s = 10.0
    sim_dt = 0.005                              # 200Hz torque controller
    decimation = 2                              # 100Hz policy
    action_space = 8                            # [L + R, joint pos + wheel velocity]
    observation_space = 29                      # Observation space
    state_space = 41                            # Inclde privilege state information

    ## ==================== Controller gain ==================== ##
    joint_kp=torch.tensor([[0.33, 0.27, 1.4]])
    joint_kd=torch.tensor([[0.01, 0.01, 0.001]])
    wheel_kp=torch.tensor([[0.1]])
    wheel_ki=torch.tensor([[0.1]])
    joint_action_weight = 10
    wheel_action_weight = 35
    
    ## ==================== Robot configuration ==================== ##
    leg_dof = 3                                 # Hip, Thigh, Knee
    num_leg = 2                                 # Bipedal
    n_leg_j = leg_dof * num_leg
    num_total_joints = n_leg_j + num_leg        # Whee per legs
    torque_limits = torch.tensor([4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 2.5, 2.5])
    joint_input_limits = torch.tensor([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
    

    ## ==================== Curriculum parameters ==================== ##
    total_DR_curriculum_level = 5               # Domain Randomization curriculum level
    total_task_curriculum_level = ["balancing", "recovery"]
    success_rate_buffer_len = 500

    max_base_acceleration_noise_per = 10        # Noise percentage (%)
    max_base_angular_vel_noise_per = 20
    max_gravity_vector_noise_per = 5
    max_base_quaternion_noise_per = 5
    max_joint_pos_noise_per = 3
    max_joint_vel_noise_per = 150

    max_terrain_friction_random_per = 50        # Friction randomization (%)
    max_terrain_restitution_random_per = 50     # Restitution randomization (%)
    default_terrain_static_friction = 0.7       # Default frictions
    default_terrain_dynamic_friction = 0.5
    default_terrain_restitution = 0.4

    max_episode_length = 10/sim_dt #5*60/sim_dt            # 5 minutes for truncated    
    ## ==================== Terminal condition ==================== ##
    height_reset_condition = 0.3

    ## ==================== Reward Shaping ==================== ##
    target_height = 0.45                        # meter (m)
    upright_threshold = 5                       # degree
    height_threshold = 0.1                      # meter (m)
    curriculum_level_up_threshold = 0.8         # success rate
    curriculum_level_down_threshold = 0.2

    r_orient_weight = 1.2
    r_height_weight = 1.2
    r_vel_lin_weight = 0.4
    r_vel_ang_weight = 0.4
    r_vel_joint_weight = 0.0
    r_effort_weight = 0.01
    r_terminated_weight = 3.0

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=sim_dt,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=0.8,
            restitution=0.0,
        ),
    )
    
    # Terrain
    terrain_importer_cfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        env_spacing=3.0,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=default_terrain_static_friction,
            dynamic_friction=default_terrain_dynamic_friction,
            restitution=default_terrain_restitution         # Collision
        ),
        debug_vis=False
    )

    contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*wheel.*",
        history_length=0,
        update_period=0.0                       # Update every period
    )