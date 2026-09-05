# goat_sysid/goat_sysid/friction_id.py
import numpy as np
import argparse
import csv
import struct
import time
import yaml
import math

from typing import Any
from datetime import datetime
from pathlib import Path

from goat_control.nodes.motor_io import MotorIO

def set_sin_position_reference(position_range, repeat, num_points):
    lower, upper = np.asarray(position_range, dtype=float)

    center = 0.5 * (lower + upper)
    amplitude = 0.5 * (upper - lower)

    initial_phase = np.arcsin(-center / amplitude)
    phase = (2.0 * np.pi * repeat * np.arange(num_points) / num_points + initial_phase)

    return center + amplitude * np.sin(phase)


def set_sin_velocity_reference(velocity_limit, repeat, num_points):
    phase = 2.0 * np.pi * repeat * np.arange(num_points) / num_points
    return float(velocity_limit) * np.sin(phase)


def leg_control(kp_leg: float, kd_leg: float, q: np.ndarray , q_dot: np.ndarray, q_ref: np.ndarray,  max_torque_per_joint: float) -> None:
    """PD control for torque command."""
    return np.clip(kp_leg * (q_ref - q) + kd_leg * (-q_dot), -max_torque_per_joint, max_torque_per_joint)

def wheel_control(kp_wheel: float, q_dot: np.ndarray, q_dot_ref: np.ndarray, max_torque_per_wheel: float) -> None:
    """P control for torque command."""
    return np.clip(kp_wheel * (q_dot_ref - q_dot), -max_torque_per_wheel, max_torque_per_wheel)


def run(motor_interface: MotorIO, cfg: dict, args: Any, csv_path: Path) -> None:
        # Arguments
        joint_id = args.joint_id
        duration = args.duration
        soft_factor = args.soft_factor
        repeat = args.repeat
        period = 1.0 / args.hz
        num_points = int(duration / period)
        is_leg = True if joint_id < 6 else False

        # Configs
        joint_names = cfg["joint_names"]
        target_joint_name = joint_names[joint_id]
        num_joints = len(cfg["joint_names"])
        kp_leg = cfg["policy_leg_proportional_gain"][0]
        kd_leg = cfg["policy_leg_derivative_gain"][0]
        kp_wheel = cfg["policy_wheel_proportional_gain"][0]
        max_pos_per_joint = np.asarray(cfg["joint_pos_limit"]).reshape(-1, 2) * soft_factor                 # Position amplitude (soft relaxation)
        max_vel_per_joint = 33.0 * soft_factor                                                              # Velocity amplitude
        max_torque_leg = 4                                                                                  # Torque clipping
        max_torque_wheel = 2

        header =  ["time_sec"] 
        header += [f"{name}_pos_rad" for name in joint_names]
        header += [f"{name}_vel_rad/s" for name in joint_names]
        header += [f"{name}_actual_torque" for name in joint_names]
        header += [f"{target_joint_name}_target_ref"]
        header += [f"{target_joint_name}_target_torque"]

        tau = np.zeros(num_joints, dtype=np.float32)
        prev_tau = np.zeros(num_joints, dtype=np.float32)
        elapsed_time = 0.0
        start_time = time.perf_counter()
        next_time = start_time
        with csv_path.open("w", newline="", encoding="utf-8") as files:
            # Header
            writer = csv.writer(files)
            writer.writerow(header)

            # Set target
            if is_leg:
                ref = set_sin_position_reference(max_pos_per_joint[joint_id, :], repeat, num_points)
            else:
                ref = set_sin_velocity_reference(max_vel_per_joint, repeat, num_points)

            # Send and log
            for i in range(num_points):
                now = time.perf_counter()
                elapsed_time = now - start_time
                # duration check
                if elapsed_time >= duration:
                    break

                # Update reference input
                ref_now = ref[i]

                joint_state_msg = motor_interface.read_joint_state()
                q = np.asarray(joint_state_msg.position, dtype=np.float32)
                q_dot = np.asarray(joint_state_msg.velocity, dtype=np.float32)
                q_tau = np.asarray(joint_state_msg.effort, dtype=np.float32)

                # Calculate torque (joint specific)
                if is_leg:
                    tau[joint_id] = leg_control(kp_leg, kd_leg, q[joint_id], q_dot[joint_id], ref_now, max_torque_leg)
                else:
                    tau[joint_id] = wheel_control(kp_wheel, q_dot[joint_id], ref_now, max_torque_wheel)

                # Send torque
                motor_interface.write_motor(tau)

                # CSV logging
                row = [elapsed_time]
                row += [q[i] for i in range(num_joints)]
                row += [q_dot[i] for i in range(num_joints)]
                row += [q_tau[i] for i in range(num_joints)]
                row += [ref_now]
                row += [prev_tau[joint_id]]
                writer.writerow(row)

                next_time += period
                sleep_time = next_time - time.perf_counter()

                prev_tau = tau
                if sleep_time > 0:
                    time.sleep(sleep_time)
    
            files.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--joint_id", type=int, default=4)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--soft_factor", type=float, default=0.4)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--hz", type=float, default=200.0)
    parser.add_argument("--timeout", type=float, default=0.05)
    parser.add_argument("--config", default="src/goat_control/config/goat_config.yaml")

    # Arguments
    args = parser.parse_args()

    # Config
    with open(args.config, "r", encoding="utf-8") as file_handle:
        cfg = yaml.safe_load(file_handle)

    num_joints = len(cfg["joint_names"])
    joint_name = cfg["joint_names"][args.joint_id]

    # CSV setting
    log_dir = Path("src/goat_sysid/logs/fixed_test")
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = log_dir / f"response_{joint_name}_{timestamp}_{args.soft_factor}.csv"

    # Motorio
    motor_io = MotorIO(cfg=cfg, logger=None, can_tx_timeout_sec=float(cfg.get("can_tx_timeout_sec", 0.05)))

    try:
        # Joint specific test
        run(motor_interface=motor_io,
            cfg=cfg, args=args, csv_path=csv_path)

    except KeyboardInterrupt:
        print("\nCurrent test interrupted.")

    finally:
        # Zero torque send
        for _ in range(5):
            motor_io.write_motor(np.zeros(num_joints, dtype=np.float32))
        time.sleep(0.01)
        
        motor_io.close()

    print(f"CSV saved: {csv_path}")