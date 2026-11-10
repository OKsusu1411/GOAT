# goat_control/utils/motor/__init__.py
from .can import CanInterface
from .motor_driver import MotorDriver
from .motor_manager import MotorManager

__all__ = [
    "CanInterface",
    "MotorDriver",
    "set_mg_unit_scales",
    "get_mg_unit_scales",
    "MotorManager",
]
