# goat_control/core/model/pipeline_factory.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from ..control.control_pipeline import ControlPipeline
from ..control.pd_controller import PDJointController
from ..control.safety_limiter import (
    ConditionalIntegratorAntiWindup,
    TorqueSafetyLimiter,
)
from ..estimation.state_manager import MotorStateCollector, StateManager
from .goat_model import GoatModel, GoatModelConfig, EffortOutputMode


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return list(value)


def _require(mapping: Dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing required key: '{key}'")
    return mapping[key]


def _get_section(config_dict: Dict[str, Any], *names: str) -> Dict[str, Any]:
    """Return first existing section dict among names, else {}."""
    for name in names:
        section_value = config_dict.get(name, None)
        if isinstance(section_value, dict):
            return section_value
    return {}


def _load_yaml_file(yaml_path: str) -> Dict[str, Any]:
    with open(yaml_path, "r", encoding="utf-8") as file_handle:
        data = yaml.safe_load(file_handle)
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping/dict.")
    return data


def build_goat_model_from_yaml(yaml_path: str) -> GoatModel:
    """Load YAML and build GoatModel."""
    config_dict = _load_yaml_file(yaml_path)

    robot_section = _get_section(config_dict, "robot", "goat", "model")
    pd_section = _get_section(config_dict, "pd", "position_pd")
    wheel_pi_section = _get_section(config_dict, "wheel_pi", "pi", "velocity_pi")
    safety_section = _get_section(config_dict, "safety", "limiter", "torque_limiter")
    estimation_section = _get_section(config_dict, "estimation", "state_manager")

    joint_names = _as_list(_require(robot_section, "joint_names"))
    joint_indices = _as_list(robot_section.get("joint_indices", list(range(0, 6))))
    wheel_indices = _as_list(robot_section.get("wheel_indices", [6, 7]))
    knee_indices = _as_list(robot_section.get("knee_indices", []))

    motor_torque_constant_nm_per_amp = _as_list(_require(robot_section, "motor_torque_constant_nm_per_amp"))
    motor_gear_ratio = _as_list(_require(robot_section, "motor_gear_ratio"))
    motor_direction = _as_list(_require(robot_section, "motor_direction"))

    motor_current_amp_per_lsb = float(estimation_section.get("motor_current_amp_per_lsb", 66.0 / 4096.0))
    angle_deg_per_lsb = float(estimation_section.get("angle_deg_per_lsb", 0.001))
    speed_deg_per_sec_per_lsb = float(estimation_section.get("speed_deg_per_sec_per_lsb", 1.0))

    joint_velocity_lpf_alpha = estimation_section.get("joint_velocity_lpf_alpha", None)
    joint_effort_like_lpf_alpha = estimation_section.get("joint_effort_like_lpf_alpha", None)
    joint_velocity_lpf_alpha = None if joint_velocity_lpf_alpha is None else float(joint_velocity_lpf_alpha)
    joint_effort_like_lpf_alpha = None if joint_effort_like_lpf_alpha is None else float(joint_effort_like_lpf_alpha)

    pd_proportional_gain = pd_section.get("proportional_gain", None)
    pd_derivative_gain = pd_section.get("derivative_gain", None)

    wheel_pi_proportional_gain = wheel_pi_section.get("proportional_gain", None)
    wheel_pi_integral_gain = wheel_pi_section.get("integral_gain", None)

    wheel_integrator_state_limit = float(wheel_pi_section.get("integrator_state_limit", 0.0))
    wheel_output_limit_per_joint = wheel_pi_section.get("output_limit_per_joint", None)

    torque_lpf_alpha_per_joint = safety_section.get("torque_lpf_alpha_per_joint", None)
    max_torque_per_joint = safety_section.get("max_torque_per_joint", None)

    goat_model_config = GoatModelConfig(
        joint_names=[str(name) for name in joint_names],
        joint_indices=[int(index) for index in joint_indices],
        wheel_indices=[int(index) for index in wheel_indices],
        knee_indices=[int(index) for index in knee_indices],
        motor_torque_constant_nm_per_amp=[float(value) for value in motor_torque_constant_nm_per_amp],
        motor_gear_ratio=[float(value) for value in motor_gear_ratio],
        motor_direction=[int(value) for value in motor_direction],
        motor_current_amp_per_lsb=motor_current_amp_per_lsb,
        angle_deg_per_lsb=angle_deg_per_lsb,
        speed_deg_per_sec_per_lsb=speed_deg_per_sec_per_lsb,
        pd_proportional_gain=None if pd_proportional_gain is None else [float(value) for value in _as_list(pd_proportional_gain)],
        pd_derivative_gain=None if pd_derivative_gain is None else [float(value) for value in _as_list(pd_derivative_gain)],
        wheel_pi_proportional_gain=None if wheel_pi_proportional_gain is None else [float(value) for value in _as_list(wheel_pi_proportional_gain)],
        wheel_pi_integral_gain=None if wheel_pi_integral_gain is None else [float(value) for value in _as_list(wheel_pi_integral_gain)],
        wheel_integrator_state_limit=wheel_integrator_state_limit,
        wheel_output_limit_per_joint=None if wheel_output_limit_per_joint is None else [float(value) for value in _as_list(wheel_output_limit_per_joint)],
        torque_lpf_alpha_per_joint=None if torque_lpf_alpha_per_joint is None else [float(value) for value in _as_list(torque_lpf_alpha_per_joint)],
        max_torque_per_joint=None if max_torque_per_joint is None else [float(value) for value in _as_list(max_torque_per_joint)],
        joint_velocity_lpf_alpha=joint_velocity_lpf_alpha,
        joint_effort_like_lpf_alpha=joint_effort_like_lpf_alpha,
    )

    return GoatModel(goat_model_config)


def build_control_pipeline_from_yaml(
    yaml_path: str,
    motor_drivers: Sequence[Any],
    *,
    effort_output_mode: EffortOutputMode = "current_amp",
) -> Tuple[GoatModel, ControlPipeline]:
    """Build (GoatModel, ControlPipeline) from YAML + motor drivers.

    Args:
        yaml_path: path to YAML config file.
        motor_drivers: sequence of MotorDriver objects used by MotorStateCollector.
        effort_output_mode:
            - "current_amp": RobotState.joint_effort_like is current [A]
            - "torque_nm": RobotState.joint_effort_like is torque [Nm]

    Returns:
        goat_model, control_pipeline
    """
    goat_model = build_goat_model_from_yaml(yaml_path)

    # 1) Estimation
    motor_state_collector = MotorStateCollector(list(motor_drivers))
    state_manager_config = goat_model.build_state_manager_config(effort_output_mode=effort_output_mode)
    state_manager = StateManager(state_manager_config)

    # 2) Controllers
    pd_controller_config = goat_model.build_pd_controller_config()
    pd_joint_controller = PDJointController(pd_controller_config)

    # Wheel PI: Use conditional integration helper (anti-windup)
    antiwindup_config = goat_model.build_conditional_integrator_config()
    wheel_antiwindup_controller = ConditionalIntegratorAntiWindup(antiwindup_config)

    wheel_pi_controller_config = goat_model.build_wheel_pi_controller_config()
    # These can be full-length or wheel-only vectors; pipeline builder will expand.
    wheel_proportional_gain = np.asarray(wheel_pi_controller_config.proportional_gain, dtype=float)
    wheel_integral_gain = np.asarray(wheel_pi_controller_config.integral_gain, dtype=float)

    # 3) Safety limiter
    safety_limiter_config = goat_model.build_torque_safety_limiter_config()
    torque_safety_limiter = TorqueSafetyLimiter(safety_limiter_config)

    # 4) Pipeline (expands wheel gains to full length internally)
    control_pipeline = ControlPipeline.build_from_goat_model(
        goat_model=goat_model,
        motor_state_collector=motor_state_collector,
        state_manager=state_manager,
        pd_joint_controller=pd_joint_controller,
        wheel_antiwindup_controller=wheel_antiwindup_controller,
        torque_safety_limiter=torque_safety_limiter,
        wheel_proportional_gain=wheel_proportional_gain,
        wheel_integral_gain=wheel_integral_gain,
    )

    return goat_model, control_pipeline
