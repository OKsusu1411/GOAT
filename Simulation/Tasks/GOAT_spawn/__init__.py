import gymnasium as gym
from . import cfg

gym.register(
    id="GOAT-Spawn-v0", 
    entry_point=f"{__name__}.GOAT_spawn_env:GOATSpawnEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.GOAT_spawn_env_cfg:GOATSpawnEnvCfg",
        "rl_ppo_cfg_entry_point": f"{cfg.__name__}:ppo_cfg.yaml",
        "rl_sac_cfg_entry_point": f"{cfg.__name__}:sac_cfg.yaml",
        "rl_td3_cfg_entry_point": f"{cfg.__name__}:td3_cfg.yaml",
    }
)



print(f"Registration is Complete.")