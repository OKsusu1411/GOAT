# goat_control/core/comm/__init__.py
from .can import CanInterface
from .motor_driver import MotorDriver, MotorParams
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

]
