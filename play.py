"""
Visual Evaluation Script for GOAT PD Stand
"""
import argparse
import torch
import os

# [중요] Isaac Sim 앱 실행기 (가장 먼저 임포트)
from isaaclab.app import AppLauncher

# 인자 파싱 (실행 시 checkpoint 경로를 받기 위함)
parser = argparse.ArgumentParser(description="Play trained policy")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to the saved .pt file")
parser.add_argument("--task", type=str, default="GOAT-stand-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to visualize")

parser.add_argument("--algorithm",
                    type=str,
                    default="PPO",
                    choices=["PPO", "SAC", "TD3"],
                    help="The RL algorithm used for training the skrl agent.")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# GUI 모드로 실행 (headless=False)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Isaac Lab & SKRL Imports ---
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
    # 1. 환경 설정 (Visual 모드)
    try:
        env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
        rl_cfg = load_cfg_from_registry(args_cli.task, f"rl_{algorithm}_cfg_entry_point")
    except ValueError as e:
        print(e)
        return
    
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # wrap around environment
    env = IsaacLabWrapper(env)  

    # [중요] 렌더링 활성화
    # env_cfg.sim.render_interval = 1  # 매 스텝마다 렌더링
    
    model_cfg = rl_cfg["model"]
    agent_cfg = rl_cfg["agent"]

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
    
    # get checkpoint path
    if args_cli.checkpoint is not None:
        print(f"Loading model from: {args_cli.checkpoint}")
        agent.load(args_cli.checkpoint)
    else:
        print("[INFO] Unfortunately a pre-trained checkpoint is not found for this task.")
        args_cli.checkpoint = None

    # Evaluation
    agent.enable_training_mode(False)

    obs, _ = env.reset()
    
    print("Start Simulation...")
    while simulation_app.is_running():
        with torch.no_grad():
            # Action
            actions = agent.act(obs, None, timestep=0, timesteps=0)[0]
            
            # Step
            obs, rewards, terminated, truncated, info = env.step(actions)

if __name__ == "__main__":
    main()
    simulation_app.close()