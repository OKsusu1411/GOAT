from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from motor_interfaces.msg import ImuState
from sensor_msgs.msg import JointState


class BaseController(ABC):
    """Abstract base class for all torque controllers.

    Contract:
        - compute() receives ROS2 sensor messages directly from the node callback
          and returns a raw torque command (num_joints,).
        - Each controller is responsible for extracting the fields it needs
          from JointState and ImuState internally.
        - Raw torque must NOT have any safety filtering applied inside.
          SafetyLimiter is the sole owner of all post-computation safety checks.
        - reset() clears all internal state (integrators, session flags, etc.).
          Must be called when switching away from this controller.
    """

    @abstractmethod
    def compute(
        self,
        joint_state: JointState,
        base_state: ImuState,
        dt_sec: float) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Compute raw torque command for the current control cycle.

        Args:
            joint_state: Sensor message containing joint positions [rad] and
                         velocities [rad/s]. Calibration offset already applied
                         by the node callback.
            base_state:  IMU sensor message containing orientation, angular
                         velocity, and linear acceleration.
            dt_sec:      Elapsed time since the last compute() call [s].

        Returns:
            raw_torque: np.ndarray of shape (num_joints,) [Nm].
                        No safety filtering applied.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset all internal state.

        Called when this controller is deactivated (e.g., mode switch)
        or when the system is re-initialized.
        """