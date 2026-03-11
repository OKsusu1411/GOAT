# goat_control/core/control/pd_controller.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
from sensor_msgs.msg import JointState
from motor_interfaces.msg import BaseStates

import numpy as np


@dataclass
class CalibrationManagerConfig:
    """Calibration manager configuration"""
    joint_offsets: np.ndarray  # shape: (num_joints,)
    imu_offsets: np.ndarray


class CalibrationManager:
    """Position PD controller for specified joint indices.

    - Input/Output units:
        target_position_rad: [rad]
        current_position_rad: [rad]
        current_velocity_rad_per_sec: [rad/s]
        desired_velocity_rad_per_sec: [rad/s] (optional, default 0)
        output_torque_command: controller effort (your system interprets it as torque)
    """

    def __init__(self, config: CalibrationManagerConfig):
        self.config = config

        # Load offset
        self.joint_offsets = np.asarray(config.joint_offsets, dtype=float).flatten()
        self.imu_offsets = np.asarray(config.imu_offsets, dtype=float).flatten()

    def apply_joint_offset(self, joint_state_msg: JointState):
        """"Execute in JointState topic subscriber"""
        joint_position_rad=np.array(joint_state_msg.position)
        joint_position_rad -= self.joint_offsets

        # Overload msg
        joint_state_msg.position = joint_position_rad.tolist()
        return joint_state_msg
    
    def apply_imu_offset(self, imu_msg: BaseStates):
        """Execute in BaseStates(imu) topic subscriber"""
        orientation_quat_w=float(imu_msg.quat.w)
        orientation_quat_x=float(imu_msg.quat.x)
        orientation_quat_y=float(imu_msg.quat.y)
        orientation_quat_z=float(imu_msg.quat.z)

        # NOTE: currently not used 
        return None