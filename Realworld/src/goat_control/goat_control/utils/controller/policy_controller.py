from __future__ import annotations

from abc import abstractmethod
from typing import List, Any

import os
import numpy as np
import onnxruntime as ort
from motor_interfaces.msg import ImuState
from sensor_msgs.msg import JointState

from .base_controller import BaseController


class PolicyController(BaseController):
    """PD (legs) + P (wheels) torque controller driven by policy actions.

    Abstract base. The two experiment setups differ only in how the policy
    observation is assembled and how the raw action is decoded, so subclasses
    implement just those two hooks:

        - fixed-base  (jig)   : legs only, no base/wheel observation.
        - movable-base(normal): base observation + wheels.
    """

    #: Mode key (set by subclass). Selects the ``policy_<MODE>`` config block.
    MODE: str = ""
    #: Whether this setup actuates the wheels.
    HAS_WHEELS: bool = True
    #: Tracking command semantics this controller consumes (set by subclass):
    #: "joint_position" (fixed) or "base_velocity" (movable).
    COMMAND_TYPE: str = ""

    def __init__(self, cfg: dict, logger: Any | None) -> None:
        self.logger = logger
        self.num_joints: int = len(cfg["joint_names"])
        self._joint_indices: List[int] = list(cfg["joint_indices"])
        self._wheel_indices: List[int] = list(cfg["wheel_indices"])
        self.num_leg_joints = len(self._joint_indices)
        self.num_wheel_joints = len(self._wheel_indices)

        # Natural info
        self.gear_ratio = np.asarray(cfg["motor_gear_ratio"], dtype=float).flatten()

        # --- PD gains (legs) ---
        self._kp = np.asarray(cfg["policy_leg_proportional_gain"], dtype=float).flatten() # [n_leg]
        self._kd = np.asarray(cfg["policy_leg_derivative_gain"], dtype=float).flatten() # [n_leg]

        # --- P gains (wheels) ---
        self._kp_wheel = np.asarray(cfg["policy_wheel_proportional_gain"], dtype=float).flatten() # [n_wheel]

        # --- Mode-specific config block (policy_fixed / policy_movable) ---
        mode_cfg = dict(cfg[f"policy_{self.MODE}"])
        self.policy_observation_info = dict(mode_cfg["observation_info"])
        self.policy_action_scale_factor = np.asarray(mode_cfg["action_scale_factor"], dtype=float).flatten()
        self._natural_pos = np.asarray(mode_cfg["natural_joint_position"], dtype=float).flatten()
        self._effective_joint_indices: List[int] = list(mode_cfg["effective_joint_indices"])
        self._effective_wheel_indices: List[int] = list(mode_cfg["effective_wheel_indices"])
        self.num_effective_leg_joints = len(self._effective_joint_indices)
        self.num_effective_wheel_joints = len(self._effective_wheel_indices)

        # --- Common policy-related information ---
        self.num_traj_points = cfg["num_traj_points"]
        self.providers = self._resolve_providers(str(cfg["policy_device"]))
        self.checkpoint_path = cfg["policy_checkpoint_path"]
        self.decimation = int(cfg["policy_decimation"])
        self.q_ref_traj = None

        # --- ONNX Runtime I/O names (populated by _load_agent) ---
        self._input_name: str | None = None
        self._output_name: str | None = None

        self.logger.info(f"[Policy Controller] Observation Info (mode={self.MODE}) \r")
        self.policy_observation_name = []
        self.policy_observation_dim = 0
        for k, v in self.policy_observation_info.items():
            self.policy_observation_name.append(k)
            self.policy_observation_dim += v
            self.logger.info(f"Name : {k} | Dim : {v}\r")

        self.agent = self._load_agent(self.checkpoint_path, self.providers)

        # --- Validate lengths ---
        for name, arr in [("natural_joint_position", self._natural_pos)]:
            if arr.size != self.num_joints:
                raise ValueError(f"{name} length must equal num_joints ({self.num_joints}).")

        # --- Internal state ---
        # Raw action dim is defined by the (mode-specific) scale factor length.
        self.observation = np.zeros((1, self.policy_observation_dim), dtype=np.float32)
        self._action_dim = int(self.policy_action_scale_factor.size)
        self._delta_pos = np.zeros(self.num_leg_joints, dtype=float)
        self._wheel_speed_ref = np.zeros(self.num_wheel_joints, dtype=float)
        self._base_command = np.zeros(4, dtype=float)  # [v_x, v_y, w_z, h]
        self._joint_command = np.zeros(self.num_leg_joints, dtype=float) # [L_hip, R_hip, L_thigh, R_thigh, L_knee, R_knee]

        # --- Count for decimation processing ---
        self._start = False
        self.count = 0

        # --- History information ---
        self.previous_action = np.zeros(self._action_dim, dtype=float)

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
            raise RuntimeError(f"Checkpoint is None.")

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
            raise RuntimeError(f"[Policy Controller] Failed to load ONNX policy '{checkpoint_path}': {exc}\r")

    # ------------------------------------------------------------------
    # Mode-specific hooks (implemented by subclasses)
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_observation(self,
                           base_lin_vel: np.ndarray,
                           base_ang_vel: np.ndarray,
                           base_quat: np.ndarray,
                           joint_pos: np.ndarray,
                           joint_vel: np.ndarray) -> np.ndarray:
        """Assemble the policy observation, shape (1, observation_dim) float32."""

    @abstractmethod
    def _decode_action(self, raw_action: np.ndarray) -> None:
        """Decode raw policy output into ``_delta_pos`` / ``_wheel_speed_ref``
        and update ``previous_action``."""

    # ------------------------------------------------------------------
    # Main Functions
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset target buffers and action history."""
        self._delta_pos[:] = 0.0
        self._wheel_speed_ref[:] = 0.0
        self._base_command[:] = 0.0
        self._joint_command[:] = 0.0
        self.previous_action[:] = 0.0
        self.count = 0

    def get_gravity_orientation(self, quaternion: np.ndarray) -> np.ndarray:
        qw = quaternion[0]
        qx = quaternion[1]
        qy = quaternion[2]
        qz = quaternion[3]

        gravity_orientation = np.zeros(3)

        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

        return gravity_orientation

    # ------------------------------------------------------------------
    # Keyboard command interface (interpreted per subclass)
    # ------------------------------------------------------------------
    # Key token convention: the node reads raw terminal input and forwards a
    # canonical token. Arrow keys are normalized to "UP"/"DOWN"/"LEFT"/"RIGHT";
    # all other keys are passed through as their single-character string.

    def handle_key(self, key: str) -> str | None:
        """Interpret a keyboard token and update this controller's command buffer.

        Returns a log string describing the resulting command, or None if the
        key is not handled by this controller. Default: ignore everything.
        """
        return None

    def command_help(self) -> list[str]:
        """Per-mode key guide lines, used by the node startup menu."""
        return []

    def set_targets(self,
                    base_lin_vel: np.ndarray,
                    base_ang_vel: np.ndarray,
                    base_quat: np.ndarray,
                    joint_pos: np.ndarray,
                    joint_vel: np.ndarray) -> None:
        """Run policy inference and update action targets before compute().

        Args:
            base_lin_vel:  Base linear velocity             [m/s], shape (3,).
            base_ang_vel:  Base angular velocity            [rad/s], shape (3,).
            base_quat:     Base quaternion                  [w, x, y, z], shape (4,).
            joint_pos:     Joint angle (all joints)         [rad], shape (J,).
            joint_vel:     Joint velocity                   [rad/s], shape (J,).
        """
        self.observation = self._build_observation(base_lin_vel, base_ang_vel, base_quat, joint_pos, joint_vel)
        raw_action = self.agent.run([self._output_name], {self._input_name: self.observation})[0].reshape(-1)
        self._decode_action(raw_action)
        self.previous_action = raw_action 

    def compute(self,
                joint_state: JointState,
                base_state: ImuState,
                dt_sec: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute raw torque: PD on legs + P on wheels (wheels only if HAS_WHEELS)."""
        # Data processing
        base_lin_vel = np.asarray([base_state.acc.x, base_state.acc.y, base_state.acc.z])
        base_ang_vel = np.asarray([base_state.gyro.x, base_state.gyro.y, base_state.gyro.z])
        base_quat = np.asarray([base_state.quat.w, base_state.quat.x, base_state.quat.y, base_state.quat.z])
        joint_pos = np.asarray(joint_state.position, dtype=float).flatten()
        joint_vel = np.asarray(joint_state.velocity, dtype=float).flatten()
        # Leg extraction
        default_leg_pos = self._natural_pos[self._joint_indices]
        joint_leg_pos = joint_pos[self._joint_indices]
        joint_leg_vel = joint_vel[self._joint_indices]
        # wheel extraction
        joint_wheel_vel = joint_vel[self._wheel_indices]
        # Computed torque
        joint_cmd = np.zeros(self.num_joints, dtype=float)
        target_pos = np.zeros(self.num_joints, dtype=float)

        # --- Reference Generation (decimation) ---
        if self._start:
            if self.count % self.decimation == 0:
                self.set_targets(base_lin_vel, base_ang_vel, base_quat, joint_pos, joint_vel)
            target_leg_pos = default_leg_pos + self._delta_pos
        else:
            if self.q_ref_traj is None:
                self.q_ref_traj = np.zeros((self.num_traj_points, self.num_leg_joints), dtype=float)
                for i in range(self.num_leg_joints):
                    self.q_ref_traj[:, i] = np.linspace(joint_pos[i], self._natural_pos[i], self.num_traj_points)
            target_leg_pos = self.q_ref_traj[min(self.num_traj_points-1, self.count), :]

        print(f"acc_x : {base_lin_vel[0]:4f} | acc_y : {base_lin_vel[1]:4f} | acc_z : {base_lin_vel[2]:4f}")
        
        # --- Error Calculation (Joint Space) ---
        leg_pos_err = target_leg_pos - joint_leg_pos
        leg_vel_err = -joint_leg_vel

        # --- Leg PD (Joint Space) ---
        tau_leg = self._kp * leg_pos_err + self._kd * leg_vel_err
        joint_cmd[self._joint_indices] = tau_leg

        # --- Wheel P (Joint Space) ---
        if self.HAS_WHEELS:
            wheel_vel_err = self._wheel_speed_ref - joint_wheel_vel
            tau_wheel = self._kp_wheel * wheel_vel_err
            joint_cmd[self._wheel_indices] = tau_wheel

        # --- Data processing ---
        target_pos[self._joint_indices] = target_leg_pos

        # Update decimation step
        self.count += 1

        return joint_cmd, target_pos, self._wheel_speed_ref.copy()