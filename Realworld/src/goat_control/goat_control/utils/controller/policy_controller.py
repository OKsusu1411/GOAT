from __future__ import annotations

from typing import List, Any

import os
import numpy as np
import onnxruntime as ort
from motor_interfaces.msg import ImuState
from sensor_msgs.msg import JointState

from .base_controller import BaseController


class PolicyController(BaseController):
    """PD (legs) + PI (wheels) torque controller driven by policy actions.

    Action space (set via set_targets()):
        delta_pos:    Delta joint position [rad], shape (num_leg_joints,).
                      Added to natural_joint_position to form the PD reference.
        wheel_speed:  Desired wheel speed [rad/s], shape (num_wheel_joints,).
                      PI controller tracks this on wheel_indices only.

    YAML keys consumed:
        joint_names, joint_indices, wheel_indices, natural_joint_position,
        policy_leg_proportional_gain, policy_leg_derivative_gain,
        policy_wheel_proportional_gain
        integrator_state_limit
    """

    def __init__(self, cfg: dict, logger: Any | None) -> None:
        self.logger = logger
        self.num_joints: int = len(cfg["joint_names"])
        self._joint_indices: List[int] = list(cfg["joint_indices"])
        self._wheel_indices: List[int] = list(cfg["wheel_indices"])
        self.num_leg_joints = len(self._joint_indices)
        self.num_wheel_joints = len(self._wheel_indices)

        # Natural info
        self.gear_ratio = np.asarray(cfg["motor_gear_ratio"], dtype=float).flatten()
        self._natural_pos = np.asarray(cfg["natural_joint_position"], dtype=float).flatten()

        # --- PD gains (legs) ---
        self._kp = np.asarray(cfg["policy_leg_proportional_gain"], dtype=float).flatten() # [n_leg]
        self._kd = np.asarray(cfg["policy_leg_derivative_gain"], dtype=float).flatten() # [n_leg]

        # --- P gains (wheels) ---
        self._kp_wheel = np.asarray(cfg["policy_wheel_proportional_gain"], dtype=float).flatten() # [n_wheel]

        # --- Policy-related information ---
        self.policy_action_scale_factor = np.asarray(cfg["policy_action_scale_factor"], dtype=float).flatten()
        self.policy_observation_info = dict(cfg["policy_observation_info"])
        self.providers = self._resolve_providers(str(cfg["policy_device"]))
        self.checkpoint_path = cfg["policy_checkpoint_path"]
        self.decimation = int(cfg["policy_decimation"])

        # --- ONNX Runtime I/O names (populated by _load_agent) ---
        self._input_name: str | None = None
        self._output_name: str | None = None

        self.logger.info(f"[Policy Controller] Observation Info \r")
        self.policy_observation_name = []
        self.policy_observation_dim = 0
        for k, v in self.policy_observation_info.items():
            self.policy_observation_name.append(k)
            self.policy_observation_dim += v
            self.logger.info(f"Name : {k} | Dim : {v}\r")

        self.agent = self._load_agent(self.checkpoint_path, self.providers)

        # --- Validate lengths ---
        for name, arr in [
            ("natural_joint_position", self._natural_pos),
        ]:
            if arr.size != self.num_joints:
                raise ValueError(f"{name} length must equal num_joints ({self.num_joints}).")

        # --- Internal state ---
        self._delta_pos = np.zeros(self.num_leg_joints, dtype=float)
        self._wheel_speed_ref = np.zeros(self.num_wheel_joints, dtype=float)
        self._base_command = np.zeros(3, dtype=float)  # [v_x, v_y, w_z]

        # --- Count for decimation processing ---
        self.decimation_count = 0

        # --- History information ---
        self.previous_action = np.zeros(self.num_joints, dtype=float)
        # self.joint_vel_hist_length = int(cfg["policy_joint_vel_hist_length"])
        # self.joint_vel_hist = np.zeros((self.joint_vel_hist_length, self.num_joints), dtype=float)


    # ------------------------------------------------------------------
    # Initialization Functions
    # ------------------------------------------------------------------

    def _resolve_providers(self, device_name: str) -> list[str]:
        available = ort.get_available_providers()
        if device_name.startswith("cuda"):
            if "CUDAExecutionProvider" in available:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.logger.info(f"[Policy Controller] CUDA execution provider unavailable. Falling back to CPU.\r")
        return ["CPUExecutionProvider"]

    def _load_agent(self, checkpoint_path: str, providers: list[str]):
        if not checkpoint_path:
            self.logger.info(f"[Policy Controller] No policy checkpoint provided; publishing zero actions.\r")
            return None

        try:
            path = os.path.abspath(checkpoint_path)
            session = ort.InferenceSession(path, providers=providers)
            self._input_name = session.get_inputs()[0].name
            self._output_name = session.get_outputs()[0].name

            # Validate input dimension (batch axis may be dynamic/str, so check last dim only)
            in_shape = session.get_inputs()[0].shape
            if isinstance(in_shape[-1], int) and in_shape[-1] != self.policy_observation_dim:
                raise ValueError(
                    f"ONNX input dim ({in_shape[-1]}) differs from observation dim ({self.policy_observation_dim})."
                )

            self.logger.info(f"[Policy Controller] Loaded ONNX policy from '{path}' on {session.get_providers()}.\r")

            test_obs = np.random.randn(1, self.policy_observation_dim).astype(np.float32)
            test_act = session.run([self._output_name], {self._input_name: test_obs})[0].reshape(-1)
            self.logger.info(f"[Policy Controller] Random test input : {test_obs}\r")
            self.logger.info(f"[Policy Controller] Random test result : {test_act}\r")
            return session
        except Exception as exc:
            self.logger.info(f"[Policy Controller] Failed to load ONNX policy '{checkpoint_path}': {exc}\r")
            return None


    # ------------------------------------------------------------------
    # Main Functions
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset PI integrator and target buffers."""
        self._delta_pos[:] = 0.0
        self._wheel_speed_ref[:] = 0.0
        self._base_command[:] = 0.0
        self.decimation_count = 0

    def set_command(self, command: np.ndarray) -> None:
        """Update base_command for next policy inference.

        Args:
            command: [v_x, v_y, w_z] shape (3,). v_y should be 0 (non-holonomic).
        """
        self._base_command[:] = command

    def set_targets(self,
                    base_lin_vel: np.ndarray,
                    base_ang_vel: np.ndarray,
                    base_quat: np.ndarray,
                    joint_pos: np.ndarray,
                    joint_vel: np.ndarray) -> None:
        """Inject policy action before compute().

        Args:
            base_lin_vel:  Base linear velocity             [m/s], shape (3,).
            base_ang_vel:  Base angular velocity            [rad/s], shape (3,).
            base_quat:     Base quaternion                  [w, x, y, z], shape (4,).
            joint_pos:     Joint angle (all joints)         [rad], shape (J,).
            joint_vel:     Joint velocity                   [rad/s], shape (J,).
        """
        # Observation setting
        previous_action = self.previous_action.copy()
        base_command = self._base_command.copy()
        # NOTE: Non-holonomic command 이므로, v_y는 항상 0
        # NOTE: joint_pos: legs only (6), joint_vel: all joints (8)
        observation = np.hstack([base_ang_vel, 
                                 base_quat, 
                                 joint_pos[self._joint_indices], 
                                 joint_vel, 
                                 previous_action]).reshape(1, -1) # [1, N]
        if self.policy_observation_dim != observation.shape[1]:
            raise ValueError(f"Observation dimension differs from pre-defined setting: ({self.policy_observation_dim} / {observation.shape[1]})")

        # Action processing
        obs = observation.astype(np.float32)  # already shape (1, N)
        policy_action_np_raw = self.agent.run([self._output_name], {self._input_name: obs})[0].reshape(-1)
        policy_action_np = policy_action_np_raw * self.policy_action_scale_factor # ACTION_{JOINT}

        # if self.decimation_count <= 10:
        #     obs_str = ", ".join(f"{x}" for x in observation.reshape(-1))
        #     act_str = ", ".join(f"{x}" for x in policy_action_np)

        #     self.logger.info(f"[step {self.decimation_count}] policy observation : [{obs_str}]\r")
        #     self.logger.info(f"[step {self.decimation_count}] policy action : [{act_str}]\r")

        self._delta_pos = policy_action_np[self._joint_indices]
        self._wheel_speed_ref = policy_action_np[self._wheel_indices]
        self.previous_action = policy_action_np_raw

    def compute(self,
                joint_state: JointState,
                base_state: ImuState,
                dt_sec: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute raw torque: PD on legs + P on wheels."""
        # Data processing
        base_lin_vel = np.asarray([base_state.vel.x, base_state.vel.y, base_state.vel.z])
        base_ang_vel = np.asarray([base_state.gyro.x, base_state.gyro.y, base_state.gyro.z])
        base_quat = np.asarray([base_state.quat.w, base_state.quat.x, base_state.quat.y, base_state.quat.z]) # NOTE: Isaacsim quaternion convention
        joint_pos = np.asarray(joint_state.position, dtype=float).flatten()
        joint_vel = np.asarray(joint_state.velocity, dtype=float).flatten()
        # Leg extraction
        default_leg_pos = self._natural_pos[self._joint_indices]
        joint_leg_pos = joint_pos[self._joint_indices]
        joint_leg_vel = joint_vel[self._joint_indices]
        # wheel extraction
        joint_wheel_vel = joint_vel[self._wheel_indices]

        # Computed torque
        tau_cmd = np.zeros(self.num_joints, dtype=float)
        target_pos = np.zeros(self.num_joints, dtype=float)

        if self.agent is None:
            return tau_cmd, self._natural_pos.copy(), np.zeros(len(self._wheel_indices))

        # --- Reference Generation (decimation) ---
        if self.decimation_count % self.decimation == 0:
            self.set_targets(base_lin_vel, base_ang_vel, base_quat, joint_pos, joint_vel)

        # --- Error Calculation (Motor Space) ---
        target_leg_pos = default_leg_pos + self._delta_pos
        leg_pos_err = target_leg_pos - joint_leg_pos
        leg_vel_err = -joint_leg_vel
        leg_pos_err /= self.gear_ratio[self._joint_indices] # Joint -> Motor
        leg_vel_err /= self.gear_ratio[self._joint_indices] # Joint -> Motor

        wheel_vel_err = self._wheel_speed_ref - joint_wheel_vel
        wheel_vel_err /= self.gear_ratio[self._wheel_indices] # Joint -> Motor
        
        # --- Leg PD (Joint Space) ---
        tau_leg = self._kp * leg_pos_err + self._kd * leg_vel_err
        tau_leg /= self.gear_ratio[self._joint_indices] # Motor -> Joint


        # --- Wheel P (Joint Space) ---
        tau_wheel = self._kp_wheel * wheel_vel_err
        tau_wheel /= self.gear_ratio[self._wheel_indices] # Motor -> Joint

        # --- Data inserting ---
        tau_cmd[self._joint_indices] = tau_leg # Joint torque
        tau_cmd[self._wheel_indices] = tau_wheel # Joint torque

        target_pos[self._joint_indices] = target_leg_pos
        
        # Update decimation step
        self.decimation_count += 1

        # if self.decimation_count <= 1:
        #     target_str = ", ".join(f"{x}" for x in target_leg_pos.reshape(-1))
        #     torque_str = ", ".join(f"{x}" for x in tau_cmd.reshape(-1))
        #     self.logger.info(f"[step {self.decimation_count}] target : [{target_str}]\r")
        #     self.logger.info(f"[step {self.decimation_count}] torque : [{torque_str}]\r")

        return tau_cmd, target_pos, self._wheel_speed_ref.copy()
