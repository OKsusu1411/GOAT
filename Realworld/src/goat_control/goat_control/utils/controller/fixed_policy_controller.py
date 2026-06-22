from __future__ import annotations

import numpy as np

from .policy_controller import PolicyController


class FixedBasePolicyController(PolicyController):
    """Jig experiment: base is fixed, legs only (no wheel actuation).

    Observation: [joint_pos[legs], joint_vel[legs], previous_action(legs),
                  joint_command(legs)].
    Action     : leg-only action, scaled per leg joint.

    Note: ``_joint_command`` is currently a zero placeholder (no setter wired
    yet); the user adjusts this separately.
    """

    MODE = "fixed"
    HAS_WHEELS = False

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
        self.previous_action = raw_action
