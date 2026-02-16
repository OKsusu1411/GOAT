import gymnasium as gym
# from .env.GOAT_PD_stand_env_cfg import GOATBaseEnvCfg

gym.register(
    id="GOAT-stand-v0", 
    entry_point=f"{__name__}.env.GOAT_PD_stand_env:GOATPDStandEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.env.GOAT_PD_stand_env_cfg:GOATPDStandEnvCfg",
        "rl_ppo_cfg_entry_point": f"{__name__}.cfg:rl_ppo_cfg.yaml",
    }
)

print(f"Registration is Complete.")