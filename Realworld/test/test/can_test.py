#!/usr/bin/env python3
"""
Simple SocketCAN tester for candleLight/gs_usb (can0).

Requires:
    pip install python-can

Usage examples:
    # Read "state1" (0x9A) from node 1 (ID 0x141) on can0
    python3 can_test.py --node 1 read-state1

    # Clear error flags (0x9B)
    python3 can_test.py --node 1 clear-errors

    # Read state2 (0x9C)
    python3 can_test.py --node 1 read-state2

    # Read encoder/angles
    python3 can_test.py --node 1 read-encoder
    python3 can_test.py --node 1 read-angle-multi
    python3 can_test.py --node 1 read-angle-single

    # Start/Stop/Run (device-dependent)
    python3 can_test.py --node 1 run
    python3 can_test.py --node 1 stop
    python3 can_test.py --node 1 power

Notes:
- Bring up can0 beforehand, e.g. 1 Mbps:
    sudo ip link set can0 down 2>/dev/null
    sudo ip link set can0 up type can bitrate 1000000
- This script uses standard 11-bit CAN IDs:
    TX: 0x140 + node_id
    RX: 0x180 + node_id
"""

import argparse
import time
from typing import Optional, Tuple

import can


CMD_STATE1_READ  = 0x9A  # temperature/voltage/current/motorState/errorState
CMD_CLEAR_ERR    = 0x9B
CMD_STATE2_READ  = 0x9C  # temperature, phase current or output, speed, encoder (depends on model)
CMD_ENCODER_READ = 0x90
CMD_ANGLE_MULTI  = 0x92
CMD_ANGLE_SINGLE = 0x94

CMD_POWER        = 0x80  # device-specific
CMD_STOP         = 0x81  # device-specific
CMD_RUN          = 0x88  # device-specific

ERROR_BITS = {
    0: "Under-voltage",
    1: "Over-voltage",
    2: "Driver over-temp",
    3: "Motor over-temp",
    4: "Motor over-current",
    5: "Motor short circuit",
    6: "Commutation error",
    7: "Input signal lost/timeout",
}

def parse_state1(data: bytes):
    """
    data layout (8 bytes), reply to 0x9A:
        [0] = 0x9A
        [1] = temperature (int8, 1°C/LSB)
        [2] = voltage L (uint16, 0.01V/LSB)
        [3] = voltage H
        [4] = current L (uint16, 0.01A/LSB)
        [5] = current H
        [6] = motorState (0x00=ON, 0x10=OFF per doc)
        [7] = errorState (bitfield)
    """
    if len(data) != 8 or data[0] != CMD_STATE1_READ:
        raise ValueError("Unexpected STATE1 payload")
    temp = int.from_bytes((data[1].to_bytes(1, 'little', signed=False)), 'little', signed=True)
    voltage_raw = int.from_bytes(bytes([data[2], data[3]]), 'little', signed=False)
    current_raw = int.from_bytes(bytes([data[4], data[5]]), 'little', signed=False)
    motor_state = data[6]
    error_state = data[7]

    voltage_v = voltage_raw * 0.01
    current_a = current_raw * 0.01

    errors = [name for bit, name in ERROR_BITS.items() if (error_state >> bit) & 1]

    return {
        "temperature_C": temp,
        "voltage_V": voltage_v,
        "current_A": current_a,
        "motor_state": "ON" if motor_state == 0x00 else ("OFF" if motor_state == 0x10 else hex(motor_state)),
        "error_state_hex": f"0x{error_state:02X}",
        "errors": errors or ["None"],
    }


def open_bus(channel="can0"):
    return can.interface.Bus(channel=channel, interface="socketcan")


def tx_rx_once(bus: can.Bus, node_id: int, cmd: int, payload: bytes = b'\x00'*7,
               timeout: float = 0.5) -> Optional[can.Message]:
    tx_id = 0x140 + node_id
    rx_id = 0x180 + node_id
    data = bytes([cmd]) + payload
    if len(data) != 8:
        raise ValueError("Payload must result in 8 bytes total")

    msg = can.Message(arbitration_id=tx_id, data=data, is_extended_id=False)
    bus.send(msg)

    t_end = time.time() + timeout
    while time.time() < t_end:
        rx = bus.recv(timeout=timeout)
        if rx is None:
            continue
        if rx.arbitration_id == rx_id and len(rx.data) == 8 and rx.data[0] == cmd:
            return rx
    return None


def cmd_state1(bus, node_id):
    rx = tx_rx_once(bus, node_id, CMD_STATE1_READ)
    if rx is None:
        print("No reply to STATE1 (0x9A)")
        return 1
    parsed = parse_state1(rx.data)
    print(f"Reply ID=0x{rx.arbitration_id:X} DATA={rx.data.hex(' ').upper()}")
    for k, v in parsed.items():
        print(f"{k}: {v}")
    return 0


def cmd_clear_errors(bus, node_id):
    rx = tx_rx_once(bus, node_id, CMD_CLEAR_ERR)
    if rx is None:
        print("No reply to CLEAR_ERRORS (0x9B)")
        return 1
    print(f"Cleared. Reply ID=0x{rx.arbitration_id:X} DATA={rx.data.hex(' ').upper()}")
    return 0


def cmd_state2(bus, node_id):
    rx = tx_rx_once(bus, node_id, CMD_STATE2_READ)
    if rx is None:
        print("No reply to STATE2 (0x9C)")
        return 1
    print(f"Reply ID=0x{rx.arbitration_id:X} DATA={rx.data.hex(' ').upper()}")
    print("NOTE: STATE2 payload mapping depends on model (MF/MG/MS). Interpret accordingly.")
    return 0


def cmd_simple(bus, node_id, cmd_val: int, name: str):
    rx = tx_rx_once(bus, node_id, cmd_val)
    if rx is None:
        print(f"No reply to {name} (0x{cmd_val:02X})")
        return 1
    print(f"{name} sent. Reply ID=0x{rx.arbitration_id:X} DATA={rx.data.hex(' ').upper()}")
    return 0


def cmd_read(bus, node_id, cmd_val: int, name: str):
    rx = tx_rx_once(bus, node_id, cmd_val)
    if rx is None:
        print(f"No reply to {name} (0x{cmd_val:02X})")
        return 1
    print(f"Reply ID=0x{rx.arbitration_id:X} DATA={rx.data.hex(' ').upper()}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="can0", help="SocketCAN channel (default: can0)")
    ap.add_argument("--node", type=int, required=True, help="Node ID (1..32)")
    ap.add_argument("action", choices=[
        "read-state1", "clear-errors", "read-state2",
        "read-encoder", "read-angle-multi", "read-angle-single",
        "run", "stop", "power",
    ])
    args = ap.parse_args()

    bus = open_bus(args.channel)

    if args.action == "read-state1":
        return_code = cmd_state1(bus, args.node)
    elif args.action == "clear-errors":
        return_code = cmd_clear_errors(bus, args.node)
    elif args.action == "read-state2":
        return_code = cmd_state2(bus, args.node)
    elif args.action == "read-encoder":
        return_code = cmd_read(bus, args.node, CMD_ENCODER_READ, "READ_ENCODER")
    elif args.action == "read-angle-multi":
        return_code = cmd_read(bus, args.node, CMD_ANGLE_MULTI, "READ_ANGLE_MULTI")
    elif args.action == "read-angle-single":
        return_code = cmd_read(bus, args.node, CMD_ANGLE_SINGLE, "READ_ANGLE_SINGLE")
    elif args.action == "run":
        return_code = cmd_simple(bus, args.node, CMD_RUN, "RUN")
    elif args.action == "stop":
        return_code = cmd_simple(bus, args.node, CMD_STOP, "STOP")
    elif args.action == "power":
        return_code = cmd_simple(bus, args.node, CMD_POWER, "POWER")
    else:
        return_code = 1

    bus.shutdown()
    exit(return_code)


if __name__ == "__main__":
    main()
