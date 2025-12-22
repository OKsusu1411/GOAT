import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="RL train for recovery and balancing task")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to spawn.")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import torch.nn as nn

from .agent import 
from .env.GOAT_PD_stand_env import GOATPDStandEnv
from .env.GOAT_PD_stand_env_cfg import GOATPDStandEnvCfg
from .trainer.sequential import SequentialTrainer
from lib.agent.ppo.ppo import PPO
from lib.memory.random import RandomMemory
from lib.wrapper.isaaclab_wrapper import IsaacLabWrapper
from skrl.utils import set_seed

def main():
    set_seed(args_cli.seed)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()