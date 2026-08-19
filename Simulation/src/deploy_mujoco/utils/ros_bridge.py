"""Conversions between MuJoCo state and ROS2 messages.

This is the only place allowed to depend on ROS message types, keeping
``mujoco_sim`` / ``sim_worker`` ROS-free.

Threading: the sim -> ROS converters take a :class:`SimSnapshot`, never live
MjData, so a ROS thread can build messages with no lock held while the sim
thread keeps stepping. The one converter that still takes ``sim``
(:func:`cmd_to_ctrl`) reads only MjModel, which is immutable after load.
"""
from __future__ import annotations

import logging
from typing import Optional

import mujoco
import numpy as np

from builtin_interfaces.msg import Time
from sensor_msgs.msg import Imu, JointState
from nav_msgs.msg import Odometry

from .mujoco_sim import EFFORT, POSITION, VELOCITY, MujocoSim, SimSnapshot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# stdlib logging -> ROS (rosout) bridge
# ---------------------------------------------------------------------- #
class _RosLogBridge(logging.Handler):
    """Forward stdlib logging records to a rclpy node's logger (rosout)."""

    def __init__(self, node) -> None:
        super().__init__()
        self._logger = node.get_logger()

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        level = record.levelno
        if level >= logging.ERROR:
            self._logger.error(msg)
        elif level >= logging.WARNING:
            self._logger.warn(msg)
        elif level >= logging.INFO:
            self._logger.info(msg)
        else:
            self._logger.debug(msg)


# Root logger of this package: "goat" when installed as goat.utils.*, "utils"
# when the node is run in place. Bridging the wrong one silently drops every
# record, so it is derived instead of hardcoded.
_PACKAGE_LOGGER = __name__.split(".")[0]


def install_ros_logging_bridge(node, logger_name: Optional[str] = None,
                               level: int = logging.INFO) -> None:
    """Route stdlib logging under ``logger_name`` into the node's rosout logger.

    ``mujoco_sim`` / ``sim_worker`` / ``ros_bridge`` log via the stdlib logging
    module so they stay ROS-free. Under a ROS launch those records are otherwise dropped: INFO
    is below the last-resort handler's WARNING threshold and nothing reaches
    rosout. Call this once early in a node's ``__init__`` (before the sim is
    built) so their INFO/WARNING -- e.g. the load-time model inspection and the
    actuator-interface resolver warnings -- show up in the ROS log.
    """
    log = logging.getLogger(logger_name or _PACKAGE_LOGGER)
    log.setLevel(level)
    log.propagate = False
    if any(isinstance(h, _RosLogBridge) for h in log.handlers):
        return  # idempotent: don't stack a bridge on re-init
    log.addHandler(_RosLogBridge(node))


# Which JointState field supplies the ctrl scalar for each actuator interface.
# Keyed by the interface constants MujocoSim resolves at load time.
_FIELD_BY_INTERFACE = {
    EFFORT: "effort",
    POSITION: "position",
    VELOCITY: "velocity",
}


# ---------------------------------------------------------------------- #
# sim -> ROS
# ---------------------------------------------------------------------- #
def sim_time_to_msg(sim_time: float) -> Time:
    sec = int(sim_time)
    nanosec = int(round((sim_time - sec) * 1e9))
    # guard against rounding to exactly 1e9; keep nanosec an int (Time.nanosec
    # rejects floats) by subtracting an integer, not the float 1e9.
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    return Time(sec=sec, nanosec=nanosec)


def joint_state_msg(snap: SimSnapshot, stamp: Time) -> JointState:
    """Build a JointState from a snapshot.

    Joints are emitted in the snapshot's order (the controller convention; see
    SimConfig.joint_order), so state output matches the name-based command
    path. Multi-DoF joints (a free base) are not in this array -- they are
    reported through :func:`odom_msg` instead.
    """
    msg = JointState()
    msg.header.stamp = stamp
    msg.name = list(snap.joint_names)
    msg.position = [float(v) for v in snap.qpos]
    msg.velocity = [float(v) for v in snap.qvel]
    return msg


def sim_joint_name(sim: MujocoSim, jid: int) -> str:
    return mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_JOINT, jid)


def imu_msg(snap: SimSnapshot, stamp: Time, frame_id: str = "imu_link") -> Imu:
    """Build a sensor_msgs/Imu from the snapshot's IMU sensor fields.

    ``framequat`` (MuJoCo order w,x,y,z), ``gyro`` and ``accelerometer`` come
    from the first sensor of each type in the model; a model without them
    yields identity / zero. Covariance left at 0 (unknown) per REP-145.
    """
    msg = Imu()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.orientation.w = 1.0  # identity default

    if snap.quat is not None:
        w, x, y, z = (float(v) for v in snap.quat)  # MuJoCo w,x,y,z -> ROS x,y,z,w
        msg.orientation.x = x
        msg.orientation.y = y
        msg.orientation.z = z
        msg.orientation.w = w

    if snap.gyro is not None:
        msg.angular_velocity.x = float(snap.gyro[0])
        msg.angular_velocity.y = float(snap.gyro[1])
        msg.angular_velocity.z = float(snap.gyro[2])

    if snap.accel is not None:
        msg.linear_acceleration.x = float(snap.accel[0])
        msg.linear_acceleration.y = float(snap.accel[1])
        msg.linear_acceleration.z = float(snap.accel[2])

    return msg

def odom_msg(snap: SimSnapshot, stamp: Time, frame_id: str = "odom",
             child_frame_id: str = "base_link") -> Odometry:
    """Build a nav_msgs/Odometry from the snapshot's free-joint (base) state.

    Fixed-base models carry no base fields, so the message stays at zero pose
    and zero twist.
    """
    msg = Odometry()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.child_frame_id = child_frame_id
    msg.pose.pose.orientation.w = 1.0

    if snap.base_pos is not None:
        msg.pose.pose.position.x = float(snap.base_pos[0])
        msg.pose.pose.position.y = float(snap.base_pos[1])
        msg.pose.pose.position.z = float(snap.base_pos[2])

        quat = snap.base_quat  # w, x, y, z
        msg.pose.pose.orientation.x = float(quat[1])
        msg.pose.pose.orientation.y = float(quat[2])
        msg.pose.pose.orientation.z = float(quat[3])
        msg.pose.pose.orientation.w = float(quat[0])

        msg.twist.twist.linear.x = float(snap.base_linvel[0])
        msg.twist.twist.linear.y = float(snap.base_linvel[1])
        msg.twist.twist.linear.z = float(snap.base_linvel[2])

        msg.twist.twist.angular.x = float(snap.base_angvel[0])
        msg.twist.twist.angular.y = float(snap.base_angvel[1])
        msg.twist.twist.angular.z = float(snap.base_angvel[2])

    return msg


# ---------------------------------------------------------------------- #
# ROS -> sim   (command interface = sensor_msgs/JointState)
# ---------------------------------------------------------------------- #
def cmd_to_ctrl(msg: JointState, sim: MujocoSim) -> np.ndarray:
    """Map a JointState command to a full-length ctrl vector.

    Each actuator reads its scalar from the single JointState field its
    interface expects (``sim.actuator_interfaces``, resolved from the model):
    EFFORT -> ``msg.effort``, POSITION -> ``msg.position``, VELOCITY ->
    ``msg.velocity``. So a motor gets a torque, a position actuator a target
    angle, a velocity actuator a target speed -- all written raw into ctrl.

    Two addressing modes:
    - named (``msg.name`` set): each name -> actuator id, value read from that
      actuator's field at the *name index* ``i`` (name[i] pairs with field[i]).
    - unnamed: applied in actuator order; actuator ``aid`` reads its field at
      *index aid* (arrays are treated sparse per actuator index for mixed
      models -- fill each actuator's slot in the array its interface uses).

    Unknown names are warned and ignored; actuators whose expected field has no
    value at the needed index keep ctrl = 0 and are summarized in one warning.

    Thread-safety: only MjModel is read (name lookups, actuator interfaces),
    which never changes after load, so this runs on the ROS callback thread
    with no lock -- the result is handed to ``SimWorker.submit_ctrl``.
    """
    ctrl = np.zeros(sim.nu, dtype=float)
    interfaces = sim.actuator_interfaces
    fields = {
        "effort": list(msg.effort),
        "position": list(msg.position),
        "velocity": list(msg.velocity),
    }
    missing = []  # aggregated so a short/empty cmd warns once, not per actuator

    if not msg.name:
        for aid in range(sim.nu):
            field = _FIELD_BY_INTERFACE[interfaces[aid]]
            arr = fields[field]
            if aid < len(arr):
                ctrl[aid] = arr[aid]
            else:
                missing.append((aid, field))
    else:
        for i, name in enumerate(msg.name):
            aid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                logger.warning("unknown actuator '%s' in cmd; ignored", name)
                continue
            field = _FIELD_BY_INTERFACE[interfaces[aid]]
            arr = fields[field]
            if i < len(arr):
                ctrl[aid] = arr[i]
            else:
                missing.append((name, field))

    if missing:
        logger.warning("cmd missing values for %d actuator(s): %s; those ctrl stay 0",
                       len(missing), missing)
    return ctrl
