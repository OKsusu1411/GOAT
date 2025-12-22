import gymnasium as gym
from .env.GOAT_PD_stand_env_cfg import GOATBaseEnvCfg

gym.register(
    id="GOAT-stand-v0", 
    entry_point=f"{__name__}.GOAT_PD_stand_env:GOATPDStandEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.GOAT_PD_stand_env_cfg:GOATPDStandEnvCfg",
        "rl_ppo_cfg_entry_point": f"{GOATBaseEnvCfg.__name__}:ppo_cfg.yaml",
        "rl_sac_cfg_entry_point": f"{GOATBaseEnvCfg.__name__}:sac_cfg.yaml",
        "rl_td3_cfg_entry_point": f"{GOATBaseEnvCfg.__name__}:td3_cfg.yaml",
    }
)

print(f"Registration is Complete.")