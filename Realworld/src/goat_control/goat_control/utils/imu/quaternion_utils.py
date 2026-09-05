# goat_control/utils/imu/quaternion_utils.py
from __future__ import annotations
from motor_interfaces.msg import ImuState

import numpy as np

def inverse_quat(q) -> ImuState | np.ndarray:
    """Return inverted quaternion [w,x,y,z]"""
    w, x, y, z = _extract_wxyz(q)

    if hasattr(q, 'quat'):
        out = type(q)()
        out.quat.w = w
        out.quat.x = -x
        out.quat.y = -y
        out.quat.z = -z
        return out
    else:
        return np.array([w, -x, -y, -z], dtype=float)
    return q

def multiply_quat(q1, q2) -> ImuState | np.ndarray:
    """Quaternion multipication -> [w, x, y, z]"""
    w1, x1, y1, z1 = _extract_wxyz(q1)
    w2, x2, y2, z2 = _extract_wxyz(q2)

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    if hasattr(q2, 'quat'):
        out = type(q2)()
        out.quat.w = w
        out.quat.x = x
        out.quat.y = y
        out.quat.z = z
        return out
    else:
        return np.array([w, x, y, z], dtype=float)
    
def rotate_vector_by_quat(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector based on quaternion"""
    w = q[0]   # w
    u = q[1:]  # x, y, z
    # v' = v + 2 * w * (u x v) + 2 * (u x (u x v))
    return v + 2.0 * w * np.cross(u, v) + 2.0 * np.cross(u, np.cross(u, v))

def axis_angle_to_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    """Return quaternion with axis, angle"""
    s = np.sin(angle / 2.0)
    
    return np.array([
        np.cos(angle / 2.0),
        axis[0] * s,
        axis[1] * s,
        axis[2] * s
    ], dtype=float)

### ============== Helper function ============== ###
def _extract_wxyz(q) -> tuple[float, float, float, float]:
    if hasattr(q, 'quat'):
        return q.quat.w, q.quat.x, q.quat.y, q.quat.z
    else:
        return float(q[0]), float(q[1]), float(q[2]), float(q[3])