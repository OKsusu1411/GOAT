# goat_control/core/__init__.py
"""
Core package (ROS-independent).
Keep this lightweight to avoid importing hardware-dependent modules at import time.
"""
from .build_system import launch_core_control_system


__all__ = [
    "launch_core_control_system",
]
