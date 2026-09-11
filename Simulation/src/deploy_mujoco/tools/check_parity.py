"""Parity check: standalone Policy vs goat_control MovableBasePolicyController.

Feeds identical random states to both and compares the raw torque vector.
Needs the ROS paths (for goat_control); the standalone script itself does not.
"""
import sys
from pathlib import Path
import numpy as np
import yaml

DEPLOY = Path("/home/grape4314/GOAT/Simulation/src/deploy_mujoco")
sys.path.insert(0, str(DEPLOY))
from deploy_mujoco import Policy, Safety                      # noqa: E402

from goat_control.utils.controller.movable_policy_controller import (  # noqa: E402
    MovableBasePolicyController)
from goat_control.utils.controller.safety_limiter import SafetyLimiter  # noqa: E402
from motor_interfaces.msg import ImuState                     # noqa: E402
from sensor_msgs.msg import JointState                        # noqa: E402


class Log:
    def info(self, *a, **k): pass
    warn = warning = error = info


sim_cfg = yaml.safe_load((DEPLOY / "configs/goat.yaml").read_text())
real_cfg = yaml.safe_load(
    Path("/home/grape4314/GOAT/Realworld/src/goat_control/config/goat_config.yaml").read_text())

ckpt = "/home/grape4314/GOAT/Realworld/src/goat_control/checkpoint/stand.onnx"
real_cfg["policy_checkpoint_path"] = ckpt

mine = Policy(sim_cfg, Path(ckpt))
theirs = MovableBasePolicyController(real_cfg, Log())

my_safety = Safety(sim_cfg)
their_safety = SafetyLimiter(real_cfg, Log())

# --- 1. static config comparison ------------------------------------------
print("=== config parity ===")
checks = [
    ("natural_joint_position", mine.natural, theirs._natural_pos),
    ("action_scale", mine.scale, theirs.policy_action_scale_factor),
    ("leg_kp", mine.kp, theirs._kp),
    ("leg_kd", mine.kd, theirs._kd),
    ("wheel_kp", mine.kp_wheel, theirs._kp_wheel),
    ("leg_indices", mine.leg, np.asarray(theirs._joint_indices)),
    ("wheel_indices", mine.wheel, np.asarray(theirs._wheel_indices)),
    ("decimation", np.array([mine.decimation]), np.array([theirs.decimation])),
    ("obs_dim", np.array([mine.obs_dim]), np.array([theirs.policy_observation_dim])),
]
ok = True
for name, a, b in checks:
    same = np.array_equal(np.asarray(a), np.asarray(b))
    ok &= same
    print(f"  {'OK ' if same else 'MISMATCH'}  {name:24s} {np.asarray(a).tolist()}"
          + ("" if same else f"  !=  {np.asarray(b).tolist()}"))

# safety params
s_checks = [
    ("pos_lower", my_safety.lower, their_safety._pos_lower),
    ("pos_upper", my_safety.upper, their_safety._pos_upper),
    ("vel_estop", np.array([my_safety.vel_max]), np.array([their_safety._estop_threshold])),
    ("estop_idx", my_safety.leg, their_safety._estop_indices),
]
for name, a, b in s_checks:
    same = np.array_equal(np.asarray(a), np.asarray(b))
    ok &= same
    print(f"  {'OK ' if same else 'MISMATCH'}  {name:24s} {np.asarray(a).tolist()}"
          + ("" if same else f"  !=  {np.asarray(b).tolist()}"))

# --- 2. rollout torque comparison -----------------------------------------
print("\n=== torque parity over 200 ticks (random states, same command) ===")
rng = np.random.default_rng(0)
cmd = np.array([0.05, 0.0, -0.03])
mine.command[:] = cmd
theirs.set_command(cmd)

worst = 0.0
for t in range(200):
    gyro = rng.normal(0, 0.3, 3)
    quat = rng.normal(0, 1, 4); quat /= np.linalg.norm(quat)
    pos = mine.natural + rng.normal(0, 0.05, 8)
    vel = rng.normal(0, 0.2, 8)

    tau_mine = mine.torque(gyro, quat, pos, vel)

    js = JointState(); js.position = pos.tolist(); js.velocity = vel.tolist()
    imu = ImuState()
    imu.quat.w, imu.quat.x, imu.quat.y, imu.quat.z = quat
    imu.gyro.x, imu.gyro.y, imu.gyro.z = gyro
    tau_theirs, _, _ = theirs.compute(js, imu, 0.005, True)

    worst = max(worst, float(np.max(np.abs(tau_mine - tau_theirs))))

print(f"  max |tau_standalone - tau_goat_control| over 200 ticks = {worst:.3e}")
ok &= worst < 1e-9
print("\n" + ("PARITY OK" if ok else "PARITY FAILED"))
sys.exit(0 if ok else 1)
