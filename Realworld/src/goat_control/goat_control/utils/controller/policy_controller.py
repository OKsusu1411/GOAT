from __future__ import annotations

from typing import List, Any

import os
import numpy as np
import torch
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
        policy_wheel_proportional_gain, policy_wheel_integral_gain,
        integrator_state_limit
    """

    def __init__(self, cfg: dict, logger: Any | None) -> None:
        self.logger = logger
        self.num_joints: int = len(cfg["joint_names"])
        self._joint_indices: List[int] = list(cfg["joint_indices"])
        self._wheel_indices: List[int] = list(cfg["wheel_indices"])
        self.num_leg_joints = len(self._joint_indices)
        self.num_wheel_joints = len(self._wheel_indices)

        # Natural (default) joint position
        self._natural_pos = np.asarray(cfg["natural_joint_position"], dtype=float).flatten()

        # --- PD gains (legs) ---
        self._kp = np.asarray(cfg["policy_leg_proportional_gain"], dtype=float).flatten() # [n_leg]
        self._kd = np.asarray(cfg["policy_leg_derivative_gain"], dtype=float).flatten() # [n_leg]

        # --- PI gains (wheels) ---
        self._kp_wheel = np.asarray(cfg["policy_wheel_proportional_gain"], dtype=float).flatten() # [n_wheel]
        self._ki_wheel = np.asarray(cfg["policy_wheel_integral_gain"], dtype=float).flatten() # [n_wheel]

        # --- Policy-related information ---
        self.leg_joint_limit_factor = cfg["policy_pos_margin_factor"]
        self.leg_joint_limit = np.asarray(cfg["policy_leg_joint_limit"]).reshape(self.num_leg_joints, 2)
        self.policy_observation_info = dict(cfg["policy_observation_info"])
        self.device = self._resolve_device(str(cfg["policy_device"]))
        self.checkpoint_path = cfg["policy_checkpoint_path"]
        self.decimation = int(cfg["policy_decimation"])

        self.logger.info(f"[Policy Controller] Observation Info \r")
        self.policy_observation_name = []
        self.policy_observation_dim = 0
        for k, v in self.policy_observation_info.items():
            self.policy_observation_name.append(k)
            self.policy_observation_dim += v
            self.logger.info(f"Name : {k} | Dim : {v}\r")

        self.agent = self._load_agent(self.checkpoint_path, self.device)

        # --- Validate lengths ---
        for name, arr in [
            ("natural_joint_position", self._natural_pos),
        ]:
            if arr.size != self.num_joints:
                raise ValueError(f"{name} length must equal num_joints ({self.num_joints}).")

        # --- Internal state ---
        self._integrator = np.zeros(self.num_joints, dtype=float)
        self._delta_pos = np.zeros(self.num_leg_joints, dtype=float)
        self._wheel_speed_ref = np.zeros(self.num_wheel_joints, dtype=float)
        self._base_command = np.zeros(3, dtype=float)  # [v_x, v_y, w_z]

        # --- Count for decimation processing ---
        self.decimation_count = 0

        # --- History information ---
        self.joint_vel_hist_length = int(cfg["policy_joint_vel_hist_length"])
        self.joint_vel_hist = np.zeros((self.joint_vel_hist_length, self.num_joints), dtype=float)
        self.previous_action = np.zeros(self.num_joints, dtype=float)

        # --- Augmented joint limit ---
        self.leg_augmented_joint_limit = np.zeros((self.num_leg_joints, 2))
        self.leg_augmented_joint_limit[:, 0] = self.leg_joint_limit_factor * self.leg_joint_limit[:, 0]
        self.leg_augmented_joint_limit[:, 1] = self.leg_joint_limit_factor * self.leg_joint_limit[:, 1]


    # ------------------------------------------------------------------
    # Initialization Functions
    # ------------------------------------------------------------------

    def _resolve_device(self, device_name: str) -> torch.device:
        try:
            device = torch.device(device_name)
        except Exception:
            self.logger.info(f"[PolicyController] Invalid torch_device '{device_name}'. Falling back to CPU.\r")
            return torch.device("cpu")

        if device.type == "cuda" and not torch.cuda.is_available():
            self.logger.info(f"[Policy Controller] CUDA requested for torch_device but unavailable. Falling back to CPU.\r")
            return torch.device("cpu")

        return device

    def _load_agent(self, checkpoint_path: str, device: torch.device):
        if not checkpoint_path:
            self.logger.info(f"[Policy Controller] No policy checkpoint provided; publishing zero actions.\r")
            return None

        try:
            path = os.path.abspath(checkpoint_path)
            model = torch.jit.load(path).to(device)
            model.eval()
            self.logger.info(f"[Policy Controller] Loaded policy checkpoint from '{path}' on {device}.\r")

            test_obs = torch.randn([1, self.policy_observation_dim], device=device)
            test_act = model(test_obs).detach().cpu().numpy().reshape(-1)
            self.logger.info(f"[Policy Controller] Zero test input : {test_obs}\r")
            self.logger.info(f"[Policy Controller] Zero test result : {test_act}\r")
            return model
        except Exception as exc:
            self.logger.info(f"[Policy Controller] Failed to load policy checkpoint '{path}': {exc}\r")
            return None


    # ------------------------------------------------------------------
    # Main Functions
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset PI integrator and target buffers."""
        self._integrator[:] = 0.0
        self._delta_pos[:] = 0.0
        self._wheel_speed_ref[:] = 0.0
        self._base_command[:] = 0.0
        self.decimation_count = 0
        self.joint_vel_hist = np.zeros((self.joint_vel_hist_length, self.num_joints), dtype=float)

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
        default_joint_pos = self._natural_pos.copy()
        previous_action = self.previous_action.copy()
        joint_vel_hist = self.joint_vel_hist.copy().reshape(-1)
        base_command = self._base_command.copy()
        stop_sign = (np.linalg.norm(base_command) < 1e-6).astype(float)
        # NOTE: Non-holonomic command 이므로, v_y는 항상 0
        # joint_pos: legs only (6), joint_vel: all joints (8)
        observation = np.hstack([base_ang_vel, 
                                 base_quat, 
                                 base_command, 
                                 default_joint_pos[self._joint_indices],
                                 stop_sign,
                                 joint_pos[self._joint_indices], 
                                 joint_vel, 
                                 previous_action, 
                                 joint_vel_hist]).reshape(1, -1) # [1, N]
        if self.policy_observation_dim != observation.shape[1]:
            raise ValueError(f"Observation dimension differs from pre-defined setting: ({self.policy_observation_dim} / {observation.shape[1]})")

        # Action processing
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        policy_action = self.agent(obs_tensor, deterministic=True)
        policy_action_np = policy_action.detach().cpu().numpy().reshape(-1)

        # obs_str = ", ".join(f"{x}" for x in observation.reshape(-1))
        # act_str = ", ".join(f"{x}" for x in policy_action_np)

        # self.logger.info(f"[step {self.decimation_count}] policy observation : [{obs_str}]\r")
        # self.logger.info(f"[step {self.decimation_count}] policy action : [{act_str}]\r")

        self._delta_pos = policy_action_np[self._joint_indices]
        self._wheel_speed_ref = policy_action_np[self._wheel_indices]
        self.previous_action = policy_action_np

    def compute(self,
                joint_state: JointState,
                base_state: ImuState,
                dt_sec: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute raw torque: PD on legs + PI on wheels."""
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

        # --- Leg PD ---
        target_leg_pos = np.clip(default_leg_pos + self._delta_pos, 
                                 self.leg_augmented_joint_limit[:, 0],
                                 self.leg_augmented_joint_limit[:, 1])
        pos_err = target_leg_pos - joint_leg_pos
        vel_err = -joint_leg_vel  # desired velocity = 0
        tau_leg = self._kp * pos_err + self._kd * vel_err

        # --- Wheel P ---
        speed_err = self._wheel_speed_ref - joint_wheel_vel
        tau_wheel = self._kp_wheel * speed_err
        
        # --- Data inserting ---
        tau_cmd[self._joint_indices] = tau_leg
        tau_cmd[self._wheel_indices] = tau_wheel

        target_pos[self._joint_indices] = target_leg_pos

        # Update joint velocity history
        self.joint_vel_hist[int(self.decimation_count % self.joint_vel_hist_length), :] = joint_vel

        # torque_str = ", ".join(f"{x}" for x in tau_cmd.reshape(-1))
        # self.logger.info(f"[step {self.decimation_count}] torque : [{torque_str}]\r")
        
        # Update decimation step
        self.decimation_count += 1

        return tau_cmd, target_pos, self._wheel_speed_ref.copy()
