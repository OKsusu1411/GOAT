# goat_control/core/__init__.py
"""goat_control.core

ROS-independent core modules.

This package intentionally avoids importing hardware-heavy builders at import
time (e.g., YAML loading, CAN/serial helpers). Import what you need directly
from submodules, e.g.:

  - goat_control.core.model.build_goat_model_from_yaml
  - goat_control.core.model.build_control_pipeline_from_yaml
"""

__all__ = []
