from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import  RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim import SimulationCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from lib.env.GOAT_base_env_cfg import GOATBaseEnvCfg


@configclass
class GOATSpawnEnvCfg(GOATBaseEnvCfg):
    # env
    episode_length_s = 10.0
    decimation = 2
    action_space = 1
    observation_space = 1
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )