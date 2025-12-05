import gymnasium as gym
from .env.GOAT_IK_stand_env_cfg import GOATIKStandEnvCfg

gym.register(
    id="GOAT-IK-stand-v0", 
    entry_point=f"{__name__}.GOAT_IK_stand_env:GOATIKStandEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.GOAT_IK_stand_env_cfg:GOATIKStandEnvCfg",
        "rl_ppo_cfg_entry_point": f"{GOATIKStandEnvCfg.__name__}:ppo_cfg.yaml",
        "rl_sac_cfg_entry_point": f"{GOATIKStandEnvCfg.__name__}:sac_cfg.yaml",
        "rl_td3_cfg_entry_point": f"{GOATIKStandEnvCfg.__name__}:td3_cfg.yaml",
    }
)

print(f"Registration is Complete.")