from __future__ import annotations

import numpy as np

from .policy_controller import PolicyController


class MovableBasePolicyController(PolicyController):
    """Normal experiment: free base, legs + wheels actuated.

    Observation: [base_ang_vel(3), base_quat(4), joint_pos[legs], joint_vel(all),
                  previous_action(all)].
    Action     : full joint action (legs + wheels), scaled per joint.
    """

    MODE = "movable"
    HAS_WHEELS = True

    def _build_observation(self,
                           base_lin_vel: np.ndarray,
                           base_ang_vel: np.ndarray,
                           base_quat: np.ndarray,
                           joint_pos: np.ndarray,
                           joint_vel: np.ndarray) -> np.ndarray:
        
        return np.hstack([base_ang_vel,
                          base_quat,
                          joint_pos[self._joint_indices],
                          joint_vel,
                          self.previous_action]).reshape(1, -1).astype(np.float32)

    def _decode_action(self, raw_action: np.ndarray) -> None:
        policy_action = raw_action * self.policy_action_scale_factor
        self._delta_pos = policy_action[self._joint_indices]
        self._wheel_speed_ref = policy_action[self._wheel_indices]
        self.previous_action = raw_action
