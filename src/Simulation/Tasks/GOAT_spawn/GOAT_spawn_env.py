
from __future__ import annotations

import torch

from .GOAT_spawn_env_cfg import GOATSpawnEnvCfg
from lib.env.GOAT_base_env import GOATBaseEnv


class GOATSpawnEnv(GOATBaseEnv):
    cfg: GOATSpawnEnvCfg

    def __init__(self, cfg: GOATSpawnEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        pass
    
    def _apply_action(self):
        pass

    def _get_observations(self) -> dict[str, torch.Tensor]:
        obs = torch.zeros((self.num_envs, self.cfg.observation_space), dtype=torch.float32, device=self.device)

        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
    
    def _get_dones(self):
        return torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
    
    def _reset_idx(self, env_ids):
        return super()._reset_idx(env_ids)
        
