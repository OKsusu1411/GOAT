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
class GOATStandEnvCfg(GOATBaseEnvCfg):
    # Environment parameters
    episode_length_s = 10.0
    sim_dt = 0.005                  # 200Hz torque controller
    decimation = 2                  # 100Hz policy
    action_space = [2, 4]           # [L + R, foot delta position + wheel velocity]
    observation_space = 1           # TODO
    state_space = 0                 # Privilege state information
    observation_noise_std = 1.0     # Observation's Gaussian noise deviation
    action_noise_std = 1.0          # Action's Gaussian noise deviation

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
            static_friction=1.0,
            dynamic_friction=0.8,
            restitution=0.0
        ),
        debug_vis=False
    )

    # Noise for Domain Randomization
    observation_noise_model = NoiseModelCfg(
        noise_cfg=GaussianNoiseCfg(
            mean=0.0,
            std=observation_noise_std,
            operation='add'
            )
        )

    action_noise_model = NoiseModelCfg(
        noise_cfg=GaussianNoiseCfg(
            mean=0.0,
            std=action_noise_std,
            operation='add'
            )
        )

    # Hyperparameters
    kp=torch.tensor([[400.0, 400.0, 300.0]])
    kd=torch.tensor([[20.0, 20.0, 15.0]])
    
    leg_dof = 3
    n_leg_j = leg_dof * 2
    num_total_joints = n_leg_j + 2