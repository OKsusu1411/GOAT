# goat_control/core/control/inverse_kinematics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np


IkMode = Literal["translation", "pose"]


@dataclass
class IKResult:
    """Inverse kinematics result."""
    joint_position_command: np.ndarray  # shape: (n, 1)
    delta_joint_position: np.ndarray    # shape: (n, 1)
    task_error: np.ndarray              # translation: (3,1), pose: (6,1)


class InverseKinematicsSolver:
    """ROS-independent inverse kinematics solver (DLS / damped least squares).

    - mode="translation": solve only linear position (use Jacobian linear part)
    - mode="pose": solve rotation + translation (use full 6xN Jacobian)

    Convention:
    - target_pose is a 7x1 vector: [qw, qx, qy, qz, x, y, z]^T  (wxyz + position)
      If your pipeline uses [qx,qy,qz,qw], convert before calling.
    """

    def __init__(self, damping_constant: float = 0.01):
        self.damping_constant = float(damping_constant)
        if self.damping_constant <= 0.0:
            raise ValueError("damping_constant must be > 0.")

    # ------------------------- math helpers ------------------------- #
    @staticmethod
    def quaternion_wxyz_to_rotation_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
        """Quaternion (w, x, y, z) -> Rotation Matrix (3x3)."""
        quaternion_wxyz = np.asarray(quaternion_wxyz, dtype=float).reshape(4,)
        w, x, y, z = quaternion_wxyz

        rotation_matrix = np.array(
            [
                [1 - 2 * (y**2 + z**2),     2 * (x*y - z*w),         2 * (x*z + y*w)],
                [2 * (x*y + z*w),           1 - 2 * (x**2 + z**2),   2 * (y*z - x*w)],
                [2 * (x*z - y*w),           2 * (y*z + x*w),         1 - 2 * (x**2 + y**2)],
            ],
            dtype=float,
        )
        return rotation_matrix

    @staticmethod
    def _vee(skew_matrix: np.ndarray) -> np.ndarray:
        """vee operator: so(3) -> R^3"""
        return np.array(
            [skew_matrix[2, 1], skew_matrix[0, 2], skew_matrix[1, 0]],
            dtype=float
        ).reshape(3, 1)

    @staticmethod
    def rotation_log_map(rotation_matrix: np.ndarray) -> np.ndarray:
        """Log map of SO(3): returns rotation error vector (3x1).

        Given R (3x3), returns phi such that exp([phi]x) = R.
        Robust for small angles.
        """
        rotation_matrix = np.asarray(rotation_matrix, dtype=float).reshape(3, 3)

        trace_value = float(np.trace(rotation_matrix))
        cos_theta = (trace_value - 1.0) * 0.5
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = float(np.arccos(cos_theta))

        if theta < 1e-8:
            # Small-angle approximation: log(R) ~ 0.5*(R - R^T)
            skew_part = 0.5 * (rotation_matrix - rotation_matrix.T)
            return InverseKinematicsSolver._vee(skew_part)

        sin_theta = float(np.sin(theta))
        skew_part = (rotation_matrix - rotation_matrix.T) * (0.5 * theta / sin_theta)
        return InverseKinematicsSolver._vee(skew_part)

    def damped_least_squares_pseudoinverse(self, matrix: np.ndarray) -> np.ndarray:
        """Damped Least Squares pseudoinverse: J^T (J J^T + λ^2 I)^-1"""
        matrix = np.asarray(matrix, dtype=float)
        matrix_transpose = matrix.T

        row_dimension = matrix.shape[0]
        damping_matrix = (self.damping_constant ** 2) * np.eye(row_dimension, dtype=float)

        inverse_term = np.linalg.inv(matrix @ matrix_transpose + damping_matrix)
        pseudoinverse_matrix = matrix_transpose @ inverse_term
        return pseudoinverse_matrix

    # ------------------------- IK core ------------------------- #
    def solve(
        self,
        target_pose_wxyz_xyz: np.ndarray,
        current_position_xyz: np.ndarray,
        current_rotation_matrix: np.ndarray,
        jacobian_6xn: np.ndarray,
        joint_position_n: np.ndarray,
        mode: IkMode = "translation",
    ) -> IKResult:
        """Solve IK for one step.

        Args:
            target_pose_wxyz_xyz: shape (7,1) or (7,) -> [qw,qx,qy,qz,x,y,z]
            current_position_xyz: shape (3,1) or (3,)
            current_rotation_matrix: shape (3,3)
            jacobian_6xn: shape (6,n)  [angular; linear] stacked
            joint_position_n: shape (n,1) or (n,)
            mode: "translation" or "pose"

        Returns:
            IKResult with joint_position_command (n,1)
        """
        target_pose_wxyz_xyz = np.asarray(target_pose_wxyz_xyz, dtype=float).reshape(7, 1)
        current_position_xyz = np.asarray(current_position_xyz, dtype=float).reshape(3, 1)
        current_rotation_matrix = np.asarray(current_rotation_matrix, dtype=float).reshape(3, 3)
        jacobian_6xn = np.asarray(jacobian_6xn, dtype=float)
        joint_position_n = np.asarray(joint_position_n, dtype=float).reshape(-1, 1)

        target_quaternion_wxyz = target_pose_wxyz_xyz[0:4, 0].reshape(4,)
        target_position_xyz = target_pose_wxyz_xyz[4:7, 0].reshape(3, 1)

        if mode == "pose":
            target_rotation_matrix = self.quaternion_wxyz_to_rotation_matrix(target_quaternion_wxyz)

            # Rotation error: log( R_current^T * R_target )
            rotation_error_matrix = current_rotation_matrix.T @ target_rotation_matrix
            rotation_error_vector = self.rotation_log_map(rotation_error_matrix)  # (3,1)

            position_error_vector = target_position_xyz - current_position_xyz   # (3,1)
            task_error = np.vstack([rotation_error_vector, position_error_vector])  # (6,1)

            pseudoinverse_jacobian = self.damped_least_squares_pseudoinverse(jacobian_6xn)
            delta_joint_position = pseudoinverse_jacobian @ task_error
            joint_position_command = joint_position_n + delta_joint_position

            return IKResult(
                joint_position_command=joint_position_command,
                delta_joint_position=delta_joint_position,
                task_error=task_error,
            )

        # mode == "translation"
        linear_jacobian_3xn = jacobian_6xn[3:6, :]  # use linear part only
        position_error_vector = target_position_xyz - current_position_xyz  # (3,1)

        pseudoinverse_linear_jacobian = self.damped_least_squares_pseudoinverse(linear_jacobian_3xn)
        delta_joint_position = pseudoinverse_linear_jacobian @ position_error_vector
        joint_position_command = joint_position_n + delta_joint_position

        return IKResult(
            joint_position_command=joint_position_command,
            delta_joint_position=delta_joint_position,
            task_error=position_error_vector,
        )
