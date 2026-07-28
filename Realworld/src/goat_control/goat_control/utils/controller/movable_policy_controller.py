from __future__ import annotations

from typing import Any

import numpy as np

from .policy_controller import PolicyController


class MovableBasePolicyController(PolicyController):
    """Normal experiment: free base, legs + wheels actuated.

    Observation: [base_ang_vel(3), base_quat(4), base_command(3),
                  joint_pos[legs], joint_vel(all), previous_action(all)].
    Action     : full joint action (legs + wheels), scaled per joint.

    Tracking mode: Base Velocity. The 3-dim ``_base_command`` [v_x, v_y, w_z]
    """

    MODE = "movable"
    HAS_WHEELS = True
    COMMAND_TYPE = "base_velocity"

    def __init__(self, cfg: dict, logger: Any | None) -> None:
        super().__init__(cfg, logger)

        # --- Base Velocity Tracking command UX state ---
        self._vx_step = float(cfg.get("policy_command_vx_step", 0.1))
        self._wz_step = float(cfg.get("policy_command_wz_step", 0.01))
        self._vx_limit = float(cfg.get("policy_command_vx_limit", 1.0))
        self._wz_limit = float(cfg.get("policy_command_wz_limit", 0.5))

    def _build_observation(self,
                           base_lin_vel: np.ndarray,
                           base_ang_vel: np.ndarray,
                           base_quat: np.ndarray,
                           joint_pos: np.ndarray,
                           joint_vel: np.ndarray) -> np.ndarray:

        return np.hstack([base_ang_vel,
                          base_quat,
                          self._base_command,
                          joint_pos[self._joint_indices] - self._natural_pos[self._joint_indices],
                          joint_vel,
                          self.previous_action]).reshape(1, -1).astype(np.float32)

    def _decode_action(self, raw_action: np.ndarray) -> None:
        policy_action = raw_action * self.policy_action_scale_factor
        self._delta_pos = policy_action[self._joint_indices]
        self._wheel_speed_ref = policy_action[self._wheel_indices]

    # ------------------------------------------------------------------
    # Keyboard command interface (Base Velocity Tracking)
    # ------------------------------------------------------------------
    def handle_key(self, key: str) -> str | None:
        if key == "UP":
            self._base_command[0] = float(np.clip(self._base_command[0] + self._vx_step,
                                                  -self._vx_limit, self._vx_limit))
        elif key == "DOWN":
            self._base_command[0] = float(np.clip(self._base_command[0] - self._vx_step,
                                                  -self._vx_limit, self._vx_limit))
        elif key == "RIGHT":
            self._base_command[2] = float(np.clip(self._base_command[2] + self._wz_step,
                                                  -self._wz_limit, self._wz_limit))
        elif key == "LEFT":
            self._base_command[2] = float(np.clip(self._base_command[2] - self._wz_step,
                                                  -self._wz_limit, self._wz_limit))
        elif key == " ":
            self._base_command[:] = 0.0
            return "Base command reset to zero\r"
        else:
            return None

        return (f"Base cmd vx={self._base_command[0]:.2f} "
                f"wz={self._base_command[2]:.2f}\r")

    def command_help(self) -> list[str]:
        return [
            "--- Base Velocity Command ---",
            "'UP'/'DOWN': v_x +/-  |  'RIGHT'/'LEFT': w_z +/-",
            "'space': reset base command\r",
        ]
