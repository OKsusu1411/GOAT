#!/usr/bin/env python3
import argparse
import time
import can
import subprocess
import sys

def bringup_can(channel: str, bitrate: int):
    # 필요할 때만 인터페이스 올림
    try:
        subprocess.run(["sudo", "ip", "link", "set", channel, "down"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["sudo", "ip", "link", "set", channel, "up", "type", "can",
                        "bitrate", str(bitrate)], check=True)
        print(f"[INFO] {channel} up @ {bitrate} bps")
    except Exception as e:
        print(f"[WARN] bringup failed ({channel}): {e}")

def txrx_state2_temp(bus: can.Bus, node_id: int, timeout: float = 0.08):
    """
    0x9C 요청 -> 응답의 data[1] = int8 온도(°C) 반환. 없으면 None.
    """
    tx_id = 0x140 + node_id
    rx_id = 0x180 + node_id
    data = bytes([0x9C]) + b"\x00"*7

    try:
        bus.send(can.Message(arbitration_id=tx_id, data=data, is_extended_id=False))
    except can.CanError as e:
        print(f"[ERR ] send fail (node {node_id}): {e}")
        return None

    deadline = time.time() + timeout
    while time.time() < deadline:
        m = bus.recv(timeout=0.02)
        if m and m.arbitration_id == rx_id and len(m.data) == 8 and m.data[0] == 0x9C:
            # data[1] = int8 temperature in °C
            temp_raw = int.from_bytes(m.data[1:2], byteorder="little", signed=True)
            return float(temp_raw)
    return None

def main():
    ap = argparse.ArgumentParser(description="Read motor temperature via CAN (0x9C).")
    ap.add_argument("--channel", default="can0")
    ap.add_argument("--bitrate", type=int, default=1_000_000)
    ap.add_argument("--bringup", action="store_true", help="Bring up CAN interface automatically")
    ap.add_argument("--nodes", type=int, nargs="+", default=[1], help="Node IDs to query (e.g., --nodes 1 2 3 4)")
    ap.add_argument("--rate", type=float, default=2.0, help="Query rate Hz (use 0 for one-shot)")
    args = ap.parse_args()

    if args.bringup:
        bringup_can(args.channel, args.bitrate)

    try:
        bus = can.interface.Bus(channel=args.channel, interface="socketcan")
    except Exception as e:
        print(f"[ERR ] open bus failed: {e}")
        sys.exit(1)

    period = (1.0 / args.rate) if args.rate > 0 else None

    try:
        if period is None:
            # one-shot
            for nid in args.nodes:
                t = txrx_state2_temp(bus, nid, timeout=0.1)
                print(f"node {nid:02d}: temp = {t if t is not None else 'NO-REPLY'} °C")
        else:
            while True:
                for nid in args.nodes:
                    t = txrx_state2_temp(bus, nid, timeout=0.1)
                    print(f"node {nid:02d}: temp = {t if t is not None else 'NO-REPLY'} °C")
                time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            bus.shutdown()
        except Exception:
            pass

if __name__ == "__main__":
    main()
