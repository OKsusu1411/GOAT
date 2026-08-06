import argparse
import csv
import struct
import time
import yaml

from typing import Any
from datetime import datetime
from pathlib import Path

from goat_control.utils.motor import CanInterface, MotorDriver

def read_and_print_pid(can_interface: CanInterface, motor_driver: MotorDriver,
                       timeout: float) -> dict:
    """Read and print the motor PID raw parameters using command 0x30."""
    msg = can_interface.txrx(
        tx_id=motor_driver.can_ids.tx_id,
        rx_id=motor_driver.can_ids.rx_id,
        cmd_byte=0x30,
        payload7=b"\x00" * 7,
        timeout=timeout,
        accept_rx_id=True,
        accept_tx_echo_diff=True,
    )

    data = bytes(msg.data)

    pid = {"iq_kp": data[6],
           "iq_ki": data[7]}

    print("\nMotor PI parameters")
    print(f"  Iq Kp    : {pid['iq_kp']}")
    print(f"  Iq Ki    : {pid['iq_ki']}\n")

    return pid

def read_motor_state2(can_interface: CanInterface, motor_driver: MotorDriver,
                      cfg: dict, timeout: float) -> dict:
    """Request motor state 2 with command 0x9C."""
    msg = can_interface.txrx(
        tx_id=motor_driver.can_ids.tx_id,
        rx_id=motor_driver.can_ids.rx_id,
        cmd_byte=0x9C,
        payload7=b"\x00" * 7,
        timeout=timeout,
        accept_rx_id=True,
        accept_tx_echo_diff=True,
    )

    if msg is None or len(msg.data) != 8:
        return {
            "iq_lsb": None,
            "iq_amp": None,
            "speed_dps": None,
            "encoder": None,
        }

    data = bytes(msg.data)

    if data[0] != 0x9C:
        return {
            "iq_lsb": None,
            "iq_amp": None,
            "speed_dps": None,
            "encoder": None,
        }

    iq_lsb = struct.unpack("<h", data[2:4])[0]
    speed_lsb = struct.unpack("<h", data[4:6])[0]

    return {
        "iq_lsb": iq_lsb,
        "iq_amp": iq_lsb * float(cfg["motor_current_amp_per_lsb"]),
        "speed_dps": speed_lsb * float(cfg["speed_deg_per_sec_per_lsb"])
    }

def send_current_and_read(can_interface: CanInterface, motor_driver: MotorDriver,
                          cfg: dict, current_cmd_amp: float, timeout: float) -> dict:
    """Send one 0xA1 current command and parse its 0xA1 response."""
    current_cmd_lsb = int(round(current_cmd_amp / cfg["motor_current_amp_per_lsb"])) # Current to LSB
    current_cmd_lsb = max(min(current_cmd_lsb, cfg["max_current_lsb"]), -cfg["max_current_lsb"]) # Clipping in LSB
    current_cmd_actual_amp = current_cmd_lsb * float(cfg["motor_current_amp_per_lsb"]) # Logging : Current

    out = b"\x00\x00\x00" + int(current_cmd_lsb).to_bytes(2, byteorder="little", signed=True) + b"\x00\x00"

    msg = can_interface.txrx(tx_id=motor_driver.can_ids.tx_id,
                             rx_id=motor_driver.can_ids.rx_id,
                             cmd_byte=0xA1, payload7=out,
                             timeout=timeout, accept_rx_id=True, accept_tx_echo_diff=True)

    if msg is None or len(msg.data) != 8:
        return {
            "iq_cmd_lsb": current_cmd_lsb,
            "iq_cmd_amp": current_cmd_actual_amp,
            "iq_lsb": None,
            "iq_amp": None,
            "iq_error_amp": None,
            "speed_dps": None}

    data = bytes(msg.data)

    iq_lsb = struct.unpack("<h", data[2:4])[0] # LSB
    speed_lsb = struct.unpack("<h", data[4:6])[0] # LSB

    iq_amp = iq_lsb * float(cfg["motor_current_amp_per_lsb"]) # LSB to current
    speed_dps = speed_lsb * float(cfg["speed_deg_per_sec_per_lsb"]) # LSB to deg/s

    return {"iq_cmd_lsb": current_cmd_lsb,
            "iq_cmd_amp": current_cmd_actual_amp,
            "iq_lsb": iq_lsb,
            "iq_amp": iq_amp,
            "iq_error_amp": current_cmd_actual_amp - iq_amp,
            "speed_dps": speed_dps}


def run_current_test(can_interface: CanInterface, motor_driver: MotorDriver,
                     cfg: dict, args: Any, csv_path: Path) -> None:

    target_current_amp = args.current
    zero_sec = args.zero
    apply_sec = args.apply
    apply_cycle = args.recursive
    recovery_sec = args.recovery
    sample_hz = args.hz
    timeout = args.timeout

    total_duration_sec = zero_sec + apply_sec + recovery_sec
    sec_per_apply_cycle = apply_sec / apply_cycle
    sample_period_sec = 1.0 / sample_hz

    fieldnames = ["time_sec",
                  "iq_cmd_requested_amp",
                  "iq_cmd_lsb",
                  "iq_cmd_amp",
                  "iq_lsb",
                  "iq_amp",
                  "iq_error_amp",
                  "speed_dps"]

    start_time = time.perf_counter()
    next_sample_time = start_time

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        while True:
            elapsed_sec = time.perf_counter() - start_time

            if elapsed_sec >= total_duration_sec:
                break

            # Torque
            if elapsed_sec < zero_sec:
                current_cmd_amp = 0.0
            elif elapsed_sec < zero_sec + apply_sec:
                delta = elapsed_sec - zero_sec
                sign = 1 if delta % (sec_per_apply_cycle) < sec_per_apply_cycle/2 else -1.0
                current_cmd_amp = target_current_amp * sign
            else:
                current_cmd_amp = 0.0


            result = send_current_and_read(can_interface=can_interface,
                                           motor_driver=motor_driver,
                                           cfg=cfg, current_cmd_amp=current_cmd_amp, timeout=timeout)
            
            state2_result = read_motor_state2(can_interface=can_interface,
                                              motor_driver=motor_driver,
                                              cfg=cfg, timeout=timeout)

            writer.writerow({
                "time_sec": elapsed_sec,
                "iq_cmd_requested_amp": current_cmd_amp,
                "iq_cmd_lsb": result["iq_cmd_lsb"],
                "iq_cmd_amp": result["iq_cmd_amp"],
                "iq_lsb": state2_result["iq_lsb"],
                "iq_amp": state2_result["iq_amp"],
                "iq_error_amp": result["iq_error_amp"],
                "speed_dps": state2_result["speed_dps"],
            })

            next_sample_time += sample_period_sec
            sleep_sec = next_sample_time - time.perf_counter()

            if sleep_sec > 0.0:
                time.sleep(sleep_sec)
            else:
                next_sample_time = time.perf_counter()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--joint_id", type=int, default=6)
    parser.add_argument("--current", type=float, required=True)

    parser.add_argument("--zero", type=float, default=1.0)
    parser.add_argument("--apply", type=float, default=10.0)
    parser.add_argument("--recursive", type=int, default=100)
    parser.add_argument("--recovery", type=float, default=1.0)

    parser.add_argument("--hz", type=float, default=500.0)
    parser.add_argument("--timeout", type=float, default=0.05)
    parser.add_argument("--config", default="src/goat_control/config/goat_config.yaml")

    # Arguments
    args = parser.parse_args()

    # Config
    with open(args.config, "r", encoding="utf-8") as file_handle:
        cfg = yaml.safe_load(file_handle)

    bus_index = int(cfg["motor_bus_index"][args.joint_id])
    channel = cfg["can_channels"][bus_index]
    node_id = cfg["motor_node_ids"][args.joint_id]

    # CSV setting
    log_dir = Path("src/goat_sysid/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = log_dir / f"iq_response_{channel}_id{node_id}_{timestamp}.csv"

    # CAN & Driver corresponding motor
    can_interface = CanInterface(channel=channel, interface="socketcan")
    can_interface.open()

    motor_driver = MotorDriver(can_interface, node_id=node_id)

    try:

        read_and_print_pid(can_interface=can_interface,
                           motor_driver=motor_driver,
                           timeout=args.timeout)
        
        run_current_test(can_interface=can_interface, 
                         motor_driver=motor_driver,
                         cfg=cfg, args=args, csv_path=csv_path)

    except KeyboardInterrupt:
        print("\nCurrent test interrupted.")

    finally:
        # Zero torque send
        for _ in range(3):
            send_current_and_read(can_interface=can_interface,
                                motor_driver=motor_driver,
                                cfg=cfg, current_cmd_amp=0.0,
                                timeout=args.timeout)
        time.sleep(0.01)

        can_interface.close()

    print(f"CSV saved: {csv_path}")
