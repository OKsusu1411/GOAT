# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")

parser.add_argument("--disable_fabric", 
                    action="store_true", 
                    default=False, 
                    help="Disable fabric and use USD I/O operations.")

parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="GOAT-stand-v0", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")

parser.add_argument("--algorithm",
                    type=str,
                    default="PPO",
                    choices=["PPO", "SAC", "TD3"],
                    help="The RL algorithm used for training the skrl agent.")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# Always headless
args_cli.headless = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym

from lib.agent.PPO.ppo import PPO
from lib.memory.random import RandomMemory
from Simulation.Tasks.GOAT_PD_stand.model.asymmetric_actor_critic import Asymmetric_Actor, Asymmetric_Critic
from Simulation.Tasks.GOAT_PD_stand.trainer.sequential import SequentialTrainer

from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
from lib.wrapper.isaaclab_wrapper import IsaacLabWrapper

# config shortcuts
algorithm = args_cli.algorithm.lower()

def main():
    # parse configuration
    try:
        env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
        rl_cfg = load_cfg_from_registry(args_cli.task, f"rl_{algorithm}_cfg_entry_point")
    except ValueError as e:
        print(e)
        return

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", rl_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    # get checkpoint path
    if args_cli.checkpoint is not None:
        resume_path = os.path.abspath(args_cli.checkpoint)
        log_dir = os.path.dirname(os.path.dirname(resume_path))
    else:
        print("[INFO] Unfortunately a pre-trained checkpoint is not found for this task.")
        resume_path = None

    # create isaac environment
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap around environment
    env = IsaacLabWrapper(env)  

    # configure and instantiate the skrl runner
    model_cfg = rl_cfg["model"]
    agent_cfg = rl_cfg["agent"]
    trainer_cfg = rl_cfg["trainer"]

    # TODO: Runner 만들어 감싸기
    model = {}
    model["policy"] = Asymmetric_Actor(observation_space=env.observation_space,
                                       action_space=env.action_space,
                                       device=env.device,
                                       cfg=model_cfg["actor"])
    
    model["value"] = Asymmetric_Critic(state_space=env.state_space,
                                       action_space=env.action_space,
                                       device=env.device,
                                       cfg=model_cfg["critic"])
    
    memory = RandomMemory(memory_size=agent_cfg["rollouts"],
                          num_envs=env.num_envs,
                          device=env.device)
    
    agent = PPO(models=model,
                memory=memory,
                observation_space=env.observation_space,
                state_space=env.state_space,
                action_space=env.action_space,
                device=env.device,
                cfg=agent_cfg)
    
    trainer = SequentialTrainer(env=env,
                                agents=agent,
                                cfg=trainer_cfg)

    if resume_path is not None:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        agent.load(resume_path)
    
    timestep = 0

    # simulate environment
    trainer.train()

    if args_cli.video:
        timestep += 1
        # exit the play loop after recording one video
        # if timestep == args_cli.video_length:
        #     break

        # time delay for real-time evaluation
        # sleep_time = dt - (time.time() - start_time)
        # if args_cli.real_time and sleep_time > 0:
        #     time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
