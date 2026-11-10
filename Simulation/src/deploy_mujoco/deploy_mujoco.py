#!/usr/bin/env python3
"""Standalone MuJoCo closed-loop deployment for GOAT."""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
import yaml

HERE = Path(__file__).resolve().parent

KEYS = {
    32: "toggle", 80: "toggle",     # space / P
    82: "reset",  88: "reset",      # R     / X
    81: "quit",                     # Q       (esc closes the window itself)
    265: "vx+",   87: "vx+",        # up    / W
    264: "vx-",   83: "vx-",        # down  / S
    262: "wz+",   68: "wz+",        # right / D
    263: "wz-",   65: "wz-",        # left  / A
    328: "h+",    325: "h-",        # Numpad 8 / 5
    48: "zero",   90: "zero",       # 0     / Z
}

def get_gravity_orientation(quaternion: np.ndarray) -> np.ndarray:
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation

class Policy:
    """ONNX policy driving PD on the legs and P on the wheels.

    obs = [ang_vel(3), gravity(3), command(3),
           leg_pos - natural(6), joint_vel(8), previous_action(8)]

    previous_action is the raw network output, before ``action_scale``.
    """

    def __init__(self, cfg: dict, checkpoint: Path) -> None:
        pol, gains = cfg["policy"], cfg["gains"]
        self.leg = np.asarray(cfg["leg_indices"], dtype=int)
        self.wheel = np.asarray(cfg["wheel_indices"], dtype=int)
        self.natural = np.asarray(pol["natural_joint_position"], dtype=float)
        self.scale = np.asarray(pol["action_scale"], dtype=float)
        self.kp = np.asarray(gains["leg_kp"], dtype=float)
        self.kd = np.asarray(gains["leg_kd"], dtype=float)
        self.kp_wheel = np.asarray(gains["wheel_kp"], dtype=float)
        self.decimation = int(pol["decimation"])
        self.limits = cfg["command"]

        self.session = ort.InferenceSession(str(checkpoint), providers=["CPUExecutionProvider"])
        self.inp = self.session.get_inputs()[0]
        self.out_name = self.session.get_outputs()[0].name

        self.obs_dim = 3 + 3 + 4 + self.leg.size + self.natural.size + self.scale.size
        want = self.inp.shape[-1]
        if isinstance(want, int) and want != self.obs_dim:
            raise SystemExit(f"ONNX wants obs dim {want}, config builds {self.obs_dim}")

        self.reset()

    def reset(self) -> None:
        self.command = np.zeros(4)                       # [v_x, v_y, w_z, h]
        self.command[3] = self.limits["height_lower_limit"]
        self.previous_action = np.zeros(self.scale.size)
        self.delta_pos = np.zeros(self.leg.size)
        self.wheel_ref = np.zeros(self.wheel.size)
        self.tick = 0

    def _infer(self, ang_vel, quat, pos, vel) -> None:
        gravity_vector = get_gravity_orientation(quat)
        obs = np.hstack([ang_vel, gravity_vector, self.command,
                         pos[self.leg] - self.natural[self.leg], vel,
                         self.previous_action]).astype(np.float32)[None]
        action = self.session.run([self.out_name], {self.inp.name: obs})[0].ravel()
        self.previous_action = action
        scaled = action * self.scale
        self.delta_pos = scaled[self.leg]
        self.wheel_ref = scaled[self.wheel]

    def torque(self, ang_vel, quat, pos, vel) -> np.ndarray:
        if self.tick % self.decimation == 0:
            self._infer(ang_vel, quat, pos, vel)
        self.tick += 1

        tau = np.zeros(pos.size)
        target = self.natural[self.leg] + self.delta_pos
        tau[self.leg] = self.kp * (target - pos[self.leg]) - self.kd * vel[self.leg]
        tau[self.wheel] = self.kp_wheel * (self.wheel_ref - vel[self.wheel])
        return tau

    def steer(self, action: str) -> str:
        """Apply a command action from KEYS. Returns a status line."""
        lim = self.limits
        if action == "vx+":
            self.command[0] = min(self.command[0] + lim["vx_step"], lim["vx_limit"])
        elif action == "vx-":
            self.command[0] = max(self.command[0] - lim["vx_step"], -lim["vx_limit"])
        elif action == "wz+":
            self.command[2] = min(self.command[2] + lim["wz_step"], lim["wz_limit"])
        elif action == "wz-":
            self.command[2] = max(self.command[2] - lim["wz_step"], -lim["wz_limit"])
        elif action == "h+":
            self.command[3] = min(self.command[3] + lim["height_step"], lim["height_upper_limit"])
        elif action == "h-":
            self.command[3] = max(self.command[3] - lim["height_step"], lim["height_lower_limit"])
        elif action == "zero":
            self.command[:] = 0.0
            self.command[3] = lim["height_lower_limit"]
        return f"cmd  v_x={self.command[0]:+.2f}  w_z={self.command[2]:+.2f}  h={self.command[3]:+.3f}"


class Safety:
    """Latching kill switch on joint position / leg velocity, plus torque clip."""

    def __init__(self, cfg: dict) -> None:
        limit = np.asarray(cfg["safety"]["joint_pos_limit"], dtype=float)
        self.lower, self.upper = limit[:, 0], limit[:, 1]
        self.leg = np.asarray(cfg["leg_indices"], dtype=int)
        self.vel_max = float(cfg["safety"]["joint_vel_estop"])
        self.tau_max = np.asarray(cfg["safety"]["max_torque"], dtype=float)
        self.blocked = False

    def reset(self) -> None:
        self.blocked = False

    def __call__(self, tau, pos, vel) -> np.ndarray:
        if not self.blocked:
            if np.any(pos < self.lower) or np.any(pos > self.upper):
                self.blocked = True
                print(f"[safety] joint position limit: {np.round(pos, 3).tolist()}", flush=True)
            elif np.any(np.abs(vel[self.leg]) > self.vel_max):
                self.blocked = True
                print(f"[safety] leg velocity estop: {np.round(vel[self.leg], 3).tolist()}", flush=True)
        if self.blocked:
            return np.zeros_like(tau)
        return np.clip(tau, -self.tau_max, self.tau_max)


class Sim:
    """MuJoCo model plus the controller-order <-> MuJoCo-order index maps."""

    def __init__(self, args, cfg: dict, policy: Policy, safety: Safety) -> None:
        self.args = args
        self.policy = policy
        self.safety = safety
        self.model = mujoco.MjModel.from_xml_path(str(args.model))
        self.model.opt.timestep = cfg["timestep"] 
        self.data = mujoco.MjData(self.model)

        # The MJCF declares joints per leg; the policy expects L/R pairs.
        q, qdot, actuator = [], [], []
        for name in cfg["joint_names"]:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if jid < 0 or aid < 0:
                raise SystemExit(f"'{name}' missing from {args.model} (joint={jid}, actuator={aid})")
            if self.model.actuator_biastype[aid] != mujoco.mjtBias.mjBIAS_NONE:
                raise SystemExit(f"actuator '{name}' is not a plain <motor>; the policy sends torque")
            q.append(self.model.jnt_qposadr[jid])
            qdot.append(self.model.jnt_dofadr[jid])
            actuator.append(aid)
        self.q = np.asarray(q, dtype=int)
        self.qdot = np.asarray(qdot, dtype=int)
        self.actuator = np.asarray(actuator, dtype=int)

        self.free_base = any(self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE for j in range(self.model.njnt))
        self.quat = self._sensor(mujoco.mjtSensor.mjSENS_FRAMEQUAT)
        self.gyro = self._sensor(mujoco.mjtSensor.mjSENS_GYRO)

        self.sim_dt = float(self.model.opt.timestep)
        self.control_dt = float(cfg["control_dt"])
        self.n_substeps = round(self.control_dt / self.sim_dt)
        if not np.isclose(self.n_substeps * self.sim_dt, self.control_dt) or self.n_substeps < 1:
            raise SystemExit(f"control_dt {self.control_dt}s is not a positive integer "
                             f"multiple of timestep {self.sim_dt}s")

        self.running = False
        self.reset()

    def _sensor(self, kind) -> slice | None:
        for sid in range(self.model.nsensor):
            if self.model.sensor_type[sid] == kind:
                adr = self.model.sensor_adr[sid]
                return slice(adr, adr + self.model.sensor_dim[sid])
        return None

    def reset(self) -> None:
        key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, self.args.keyframe)
        if key >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key)
        else:
            mujoco.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.policy.reset()
        self.safety.reset()

    def step(self) -> None:
        """One control tick: observe, compute torque, advance n_substeps."""
        pos = self.data.qpos[self.q]
        vel = self.data.qvel[self.qdot]
        sensor = self.data.sensordata
        quat = sensor[self.quat] if self.quat else np.array([1.0, 0.0, 0.0, 0.0])
        gyro = sensor[self.gyro] if self.gyro else np.zeros(3)
        
        if self.running:
            tau = self.policy.torque(gyro, quat, pos, vel)
            self.data.ctrl[self.actuator] = tau
            # self.data.ctrl[self.actuator] = self.safety(tau, pos, vel)
        else:
            self.data.ctrl[self.actuator] = 0.0
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=HERE / "config/xml/goat_on_stand.xml")
    p.add_argument("--config", type=Path, default=HERE / "config/goat.yaml")
    p.add_argument("--keyframe", default="home")
    p.add_argument("--duration", type=float, default=0.0, help="stop after N s of sim time")
    p.add_argument("--stopped", action="store_true", help="begin with the policy paused")
    p.add_argument("--headless", action="store_true", help="no viewer")
    p.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=None, help="pace to wall clock (default: on with viewer, off headless)")
    args = p.parse_args()
    if args.realtime is None:
        args.realtime = not args.headless
    return args

def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    checkpoint = Path(cfg["policy"]["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = (args.config.parent / checkpoint).resolve()

    policy = Policy(cfg, checkpoint)
    sim = Sim(args, cfg, policy, Safety(cfg))
    sim.running = not args.stopped

    print(f"checkpoint: {checkpoint}")
    print(f"physics {sim.sim_dt*1e3:.1f} ms | policy {1/(sim.control_dt*policy.decimation):.0f} Hz")
    print("keys go to the VIEWER window, not this terminal:")
    print("  space/P start-stop | R/X reset | Q quit | up-down/W-S v_x | left-right/A-D w_z | 0/Z zero command")
    print("policy " + ("running" if sim.running else "paused -- press space in the viewer"), flush=True)

    pending: deque[str] = deque()

    def on_key(code: int) -> None:
        action = KEYS.get(code)
        if action is not None:
            pending.append(action)

    # Viewer
    viewer = None if args.headless else mujoco.viewer.launch_passive(sim.model, sim.data, show_left_ui=True, show_right_ui=False, key_callback=on_key)
    camera_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, "track")
    with viewer.lock():
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = camera_id

    try:
        n, quit_requested = 0, False
        while not quit_requested and (viewer is None or viewer.is_running()):
            if args.duration and sim.data.time >= args.duration:
                break

            while pending:
                action = pending.popleft()
                if action == "quit":
                    quit_requested = True
                elif action == "toggle":
                    sim.running = not sim.running
                    print("policy " + ("running" if sim.running else "stopped"), flush=True)
                elif action == "reset":
                    sim.reset()
                    print("reset", flush=True)
                else:
                    print(policy.steer(action), flush=True)

            start = time.perf_counter()
            sim.step()
            n += 1

            if viewer is not None and n % 4 == 0:      # render ~50 Hz
                viewer.sync()
            if args.realtime:
                remaining = sim.control_dt - (time.perf_counter() - start)
                if remaining > 0:
                    time.sleep(remaining)

        print(f"\n{n} ticks, {sim.data.time:.2f} s sim time", flush=True)
    finally:
        if viewer is not None:
            viewer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
