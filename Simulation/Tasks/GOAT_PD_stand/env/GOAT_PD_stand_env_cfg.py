import isaaclab.sim as sim_utils
import gymnasium
import torch

from isaaclab.sim import SimulationCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import DCMotorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.noise.noise_cfg import GaussianNoiseCfg, NoiseModelCfg
from lib.env.GOAT_base_env_cfg import GOATBaseEnvCfg
from gymnasium.spaces import Dict

@configclass
class GOATPDStandEnvCfg(GOATBaseEnvCfg):
    ## ==================== Environment parameters ==================== ##
    episode_length_s = 10.0
    sim_dt = 0.005                              # 200Hz torque controller
    decimation = 2                              # 100Hz policy
    action_space = [2, 4]                       # [L + R, joint pos + wheel velocity]
    observation_space = 35                      # Observation space
    state_space = 47                            # Privilege state information

    ## ==================== Controller gain ==================== ##
    kp=torch.tensor([[1.0, 1.0, 1.0]])
    kd=torch.tensor([[0.1, 0.1, 0.1]])
    
    ## ==================== Robot configuration ==================== ##
    leg_dof = 3                                 # Hip, Thigh, Knee
    n_leg_j = leg_dof * 2
    num_total_joints = n_leg_j + 2              # Two wheels

    ## ==================== Curriculum parameters ==================== ##
    max_curriculum_level = 5                    # Total curriculum level

    max_base_acceleration_noise_per = 10        # Noise percentage (%)
    max_base_angular_vel_noise_per = 20
    max_gravity_vector_noise_per = 5
    max_base_quaternion_noise_per = 5
    max_joint_pos_noise_per = 3
    max_joint_vel_noise_per = 150

    max_terrain_friction_random_per = 50        # Friction randomization (%)
    default_terrain_static_friction = 0.7       # Default frictions
    default_terrain_dynamic_friction = 0.5 

    max_episode_length = 5*60/sim_dt            # 5 minutes for truncated    

    ## ==================== Reward weight ==================== ##


    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=sim_dt,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=0.8,
            restitution=0.0,
        ),
    )
    
    # Terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        env_spacing=3.0,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=default_terrain_static_friction,
            dynamic_friction=default_terrain_dynamic_friction,
            restitution=0.0                                 # Collision
        ),
        debug_vis=False
    )