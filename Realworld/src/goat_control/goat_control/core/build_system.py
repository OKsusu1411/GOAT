# goat_control/core/build_system.py
from __future__ import annotations

from typing import Any, Sequence, Tuple

from .model.goat_model import GoatModel, EffortOutputMode
from .model.model_builder import build_control_pipeline_from_yaml
from .control.control_pipeline import ControlPipeline


def launch_core_control_system(
    *,
    yaml_path: str,
    motor_drivers: Sequence[Any],
    effort_output_mode: EffortOutputMode = "torque_nm",
) -> Tuple[GoatModel, ControlPipeline]:
    """
    Build GoatModel + ControlPipeline from YAML + MotorDriver list.
    Core-level system assembly entry point (ROS-independent).
    """
    return build_control_pipeline_from_yaml(
        yaml_path=yaml_path,
        motor_drivers=motor_drivers,
        effort_output_mode=effort_output_mode,
    )
