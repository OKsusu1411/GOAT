from __future__ import annotations

from typing import Any

import numpy as np

from .policy_controller import PolicyController


class FixedBasePolicyController(PolicyController):
    """Jig experiment: base is fixed, legs only (no wheel actuation).

    Observation: [joint_pos[legs], joint_vel[legs], previous_action(legs),
                  joint_command(legs)].
    Action     : leg-only action, scaled per leg joint.

    Tracking mode: Joint Position. The 6-dim ``_joint_command`` (leg joint targets) 
    """

    MODE = "fixed"
    HAS_WHEELS = False
    COMMAND_TYPE = "joint_position"

    #: Keyboard class selector -> (name, left_idx, right_idx) into the 6-dim
    #: leg joint_command, ordered [hip_L, hip_R, thigh_L, thigh_R, knee_L, knee_R].
    _CLASS_TABLE = {
        1: ("hip", 0, 1),
        2: ("thigh", 2, 3),
        3: ("knee", 4, 5),
    }

    def __init__(self, cfg: dict, logger: Any | None) -> None:
        super().__init__(cfg, logger)

        # --- Joint Position Tracking command UX state ---
        self._selected_class = 1  # default: hip

        # Step size: config value is in degrees, converted to radians here.
        step_deg = float(cfg.get("policy_command_joint_step", 5.0))
        self._joint_step = float(np.radians(step_deg))

        # Per-leg-joint clip limits from joint_pos_limit ([min, max] pairs for
        # all 8 joints, in joint_names order).
        limit = np.asarray(cfg["joint_pos_limit"], dtype=float).reshape(-1, 2)
        leg_limit = limit[self._joint_indices]  # (num_leg_joints, 2)
        self._joint_cmd_min = leg_limit[:, 0]
        self._joint_cmd_max = leg_limit[:, 1]

        # NOTE: Hip joint is only rotated toward outside direction 
        self._joint_cmd_max[0] = 0 # Left Hip 
        self._joint_cmd_min[1] = 0 # Right Hip


    def _build_observation(self,
                           base_lin_vel: np.ndarray,
                           base_ang_vel: np.ndarray,
                           base_quat: np.ndarray,
                           joint_pos: np.ndarray,
                           joint_vel: np.ndarray) -> np.ndarray:

        return np.hstack([joint_pos[self._joint_indices],
                          joint_vel[self._joint_indices],
                          self.previous_action,
                          self._joint_command]).reshape(1, -1).astype(np.float32)

    def _decode_action(self, raw_action: np.ndarray) -> None:
        # Raw action and scale factor are both leg-only (no wheels on the jig).
        policy_action = raw_action * self.policy_action_scale_factor
        self._delta_pos = policy_action
        self._wheel_speed_ref[:] = 0.0

    # ------------------------------------------------------------------
    # Keyboard command interface (Joint Position Tracking)
    # ------------------------------------------------------------------
    def handle_key(self, key: str) -> str | None:
        # Stage 1: select joint class.
        if key in ("1", "2", "3"):
            self._selected_class = int(key)
            name, _, _ = self._CLASS_TABLE[self._selected_class]
            return f"Joint class selected: [{key}] {name}\r"

        # Stage 2: increment/decrement selected joint (left as reference,
        # right joint mirrored with opposite sign).
        if key in ("UP", "DOWN"):
            name, l_idx, r_idx = self._CLASS_TABLE[self._selected_class]
            sign = 1.0 if key == "UP" else -1.0
            self._joint_command[l_idx] += sign * self._joint_step
            self._joint_command[r_idx] -= sign * self._joint_step
            # Clip each side into its joint position limit.
            self._joint_command[l_idx] = float(np.clip(self._joint_command[l_idx],
                                                       self._joint_cmd_min[l_idx],
                                                       self._joint_cmd_max[l_idx]))
            self._joint_command[r_idx] = float(np.clip(self._joint_command[r_idx],
                                                       self._joint_cmd_min[r_idx],
                                                       self._joint_cmd_max[r_idx]))
            return (f"Joint cmd [{name}] L={self._joint_command[l_idx]:.3f} "
                    f"R={self._joint_command[r_idx]:.3f}\r")

        # Reset to default pose (zeros).
        if key == " ":
            self._joint_command[:] = 0.0
            return "Joint command reset to zero\r"

        if key == "n":
            self._joint_command = np.array([0.0, 0.0, 0.738, -0.738, 1.462, -1.462], dtype=np.float32)

        return None

    def command_help(self) -> list[str]:
        return [
            "--- Joint Position Command ---",
            "'1'/'2'/'3': select hip/thigh/knee",
            "Up/Down arrow: selected joint +/-",
            "'n': Natural standing position",
            "'space': reset joint command\r",
        ]
