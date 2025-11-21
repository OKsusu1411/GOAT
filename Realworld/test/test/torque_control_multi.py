#!/usr/bin/env python3
# torque_control_multi.py
# Examples:
#   python3 torque_control_multi.py --series MF --bitrate 1000000 \
#       --cmd 1:2.0 --cmd 2:-1.0 --cmd 3:0.0 --cmd 4:1.5
#
#   python3 torque_control_multi.py --series MG --cmd 1:3.0 --cmd 4:-2.5
#
# Notes:
# - --series applies to all nodes (MF or MG). If your motors mix series, call twice.
# - Bring up can0 beforehand (or let script do it with --bringup).

import argparse
import struct
import time
import os
import subprocess
from typing import Dict, Tuple, List
import can

# A/LSB scale (from docs, based on +/- current range)
SCALE_A_PER_LSB = {
    "MF": 16.5 / 2048.0,   # ≈ 0.00806 A/LSB
    "MG": 33.0 / 2048.0,   # ≈ 0.01611 A/LSB
}

def bringup_can(channel: str, bitrate: int):
    # Try to bring up the CAN interface if requested
    try:
        subprocess.run(["sudo", "ip", "link", "set", channel, "down"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["sudo", "ip", "link", "set", channel, "up", "type", "can", "bitrate", str(bitrate)], check=True)
    except Exception as e:
        print(f"[WARN] Could not bring up {channel}: {e}")

def open_bus(channel="can0"):
    return can.interface.Bus(channel=channel, interface="socketcan")

def parse_cmds(cmd_list: List[str]) -> Dict[int, float]:
    """Parse ['1:2.0', '2:-1.0'] -> {1:2.0, 2:-1.0}"""
    out = {}
    for item in cmd_list:
        try:
            node_s, amp_s = item.split(":")
            node = int(node_s.strip())
            amps = float(amp_s.strip())
            if node < 1 or node > 32:
                raise ValueError
            out[node] = amps
        except Exception:
            raise SystemExit(f"Bad --cmd '{item}'. Use NODE:AMPS (e.g., 1:2.5)")
    return out

def send_and_wait(bus, tx_id, rx_id, data0, payload7=b"\x00"*7, timeout=0.3):
    data = bytes([data0]) + payload7
    msg = can.Message(arbitration_id=tx_id, data=data, is_extended_id=False)
    bus.send(msg)
    t_end = time.time() + timeout
    while time.time() < t_end:
        m = bus.recv(timeout=timeout)
        if not m:
            continue
        if m.arbitration_id == rx_id and len(m.data) == 8 and m.data[0] == data0:
            return m
    return None

def run_enable(bus, node_id):
    tx_id = 0x140 + node_id
    rx_id = 0x180 + node_id
    send_and_wait(bus, tx_id, rx_id, 0x88)  # RUN

def clear_errors(bus, node_id):
    tx_id = 0x140 + node_id
    rx_id = 0x180 + node_id
    send_and_wait(bus, tx_id, rx_id, 0x9B)  # CLEAR ERRORS

def read_state2(bus, node_id):
    tx_id = 0x140 + node_id
    rx_id = 0x180 + node_id
    rep = send_and_wait(bus, tx_id, rx_id, 0x9C)
    return rep

def torque_command(bus, node_id, series, amps):
    scale = SCALE_A_PER_LSB[series]
    iq = int(round(amps / scale))
    if iq > 2048: iq = 2048
    if iq < -2048: iq = -2048

    tx_id = 0x140 + node_id
    rx_id = 0x180 + node_id
    iq_le = struct.pack("<h", iq)
    payload = b"\x00\x00\x00" + iq_le + b"\x00\x00"
    rep = send_and_wait(bus, tx_id, rx_id, 0xA1, payload)

    return iq, rep

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="can0", help="SocketCAN channel")
    ap.add_argument("--bitrate", type=int, default=1000000, help="CAN bitrate if --bringup")
    ap.add_argument("--bringup", action="store_true", help="Try to bring up CAN interface automatically")
    ap.add_argument("--series", choices=["MF","MG"], required=True, help="Motor series for all nodes in this call")
    ap.add_argument("--cmd", action="append", required=True,
                    help="Torque command per node as NODE:AMPS (e.g., --cmd 1:2.0 --cmd 2:-1.0)")
    ap.add_argument("--run", action="store_true", help="Send RUN (0x88) to all listed nodes before torque")
    ap.add_argument("--clear", action="store_true", help="Send CLEAR ERRORS (0x9B) to all listed nodes before torque")
    ap.add_argument("--feedback", action="store_true", help="Print reply of 0x9C (state2) after torque")
    args = ap.parse_args()

    if args.bringup:
        bringup_can(args.channel, args.bitrate)

    cmds = parse_cmds(args.cmd)  # {node: amps}

    bus = open_bus(args.channel)

    # Optional prep
    for node in cmds.keys():
        if args.clear:
            clear_errors(bus, node)
        if args.run:
            run_enable(bus, node)

    # Send torque commands node-by-node
    for node, amps in cmds.items():
        iq_cmd, rep = torque_command(bus, node, args.series, amps)
        status = "OK" if rep else "NO-REPLY"
        print(f"node={node:02d} amps={amps:+.3f} A -> iq={iq_cmd:+d} LSB [{status}]")

    # Optional feedback readback per node
    if args.feedback:
        for node in cmds.keys():
            rep2 = read_state2(bus, node)
            if rep2:
                d = rep2.data
                temp = struct.unpack("<b", d[1:2])[0]
                raw_iq_or_power = struct.unpack("<h", d[2:4])[0]
                speed_dps = struct.unpack("<h", d[4:6])[0]
                enc = struct.unpack("<H", d[6:8])[0]
                iq_res = (33.0/4096.0) if args.series=="MF" else (66.0/4096.0)
                iq_amp = raw_iq_or_power * iq_res
                print(f"feedback node={node:02d}: temp={temp}°C iq={raw_iq_or_power}LSB({iq_amp:.3f}A) speed={speed_dps}dps enc={enc}")
            else:
                print(f"feedback node={node:02d}: NO-REPLY")

    bus.shutdown()

if __name__ == "__main__":
    main()
