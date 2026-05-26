# goat_control/utils/motor/__init__.py
from .can import CanInterface
from .motor_driver import MotorDriver, MotorParams
# [DEPRECATED] FirstOrderLowPassFilter is no longer used; torque LPF lives in
# MotorManager.torque_lpf(). format_motor_states references fields removed from
# MotorStatesData and is also unused.
# from .filters import FirstOrderLowPassFilter
from .motor_manager import MotorManager  # , format_motor_states  # [DEPRECATED]
from .protocol import (
    CanIds,
    mg_ids,
    E7,
    MGUnitScales,
    set_mg_unit_scales,
    get_mg_unit_scales,
    pack_iq_from_amp,
    payload_torque_mode_from_amp,
)

__all__ = [
    "CanInterface",
    "MotorDriver",
    "MotorParams",
    "CanIds",
    "mg_ids",
    "E7",
    "pack_iq_from_amp",
    "payload_torque_mode_from_amp",
    "MGUnitScales",
    "set_mg_unit_scales",
    "get_mg_unit_scales",
    # [DEPRECATED] "FirstOrderLowPassFilter",
    "MotorManager",
    # [DEPRECATED] "format_motor_states",
]
