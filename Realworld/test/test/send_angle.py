#!/usr/bin/env python3
import argparse
import struct
import time
import can

# CAN IDs: TX = 0x140 + node_id, RX = 0x180 + node_id  :contentReference[oaicite:1]{index=1}
def open_bus(channel="can0"):
    return can.interface.Bus(channel=channel, interface="socketcan")

def send_and_wait(bus, tx_id, rx_id, data0, payload7=b"\x00"*7, timeout=0.3):
    frame = bytes([data0]) + payload7
    msg = can.Message(arbitration_id=tx_id, data=frame, is_extended_id=False)
    bus.send(msg)
    t_end = time.time() + timeout
    while time.time() < t_end:
        m = bus.recv(timeout=0.05)
        if not m:
            continue
        # 정상 응답(0x180+ID) 또는 일부 장치의 에코(0x140+ID) 허용
        if m.data[:1] == bytes([data0]) and m.arbitration_id in (rx_id, tx_id):
            # 동일 프레임 에코는 스킵
            if m.arbitration_id == tx_id and m.data == frame:
                continue
            return m
    return None

def cmd_run(bus, node_id):
    tx, rx = 0x140 + node_id, 0x180 + node_id
    send_and_wait(bus, tx, rx, 0x88)  # RUN(Enable)  :contentReference[oaicite:2]{index=2}

# ---------- A3: Multi-turn position (angle only, 0.01 deg/LSB) ----------
def send_A3_multi(bus, node_id, angle_deg):
    """
    angle_deg -> angleControl(int32, 0.01deg/LSB). 0xA3  :contentReference[oaicite:3]{index=3}
    """
    angle_cnt = int(round(angle_deg * 100.0))              # 0.01deg/LSB
    payload = b"\x00\x00\x00" + struct.pack("<i", angle_cnt)
    tx, rx = 0x140 + node_id, 0x180 + node_id
    return send_and_wait(bus, tx, rx, 0xA3, payload)

# ---------- A4: Multi-turn position + max speed ----------
def send_A4_multi(bus, node_id, angle_deg, max_speed_dps):
    """
    0xA4: DATA[2..3]=maxSpeed(uint16, 1 dps/LSB), DATA[4..7]=angle(int32, 0.01deg/LSB)  :contentReference[oaicite:4]{index=4}
    """
    max_speed = max(0, min(0xFFFF, int(round(max_speed_dps))))
    angle_cnt = int(round(angle_deg * 100.0))
    payload = b"\x00" + struct.pack("<H", max_speed) + struct.pack("<i", angle_cnt)
    tx, rx = 0x140 + node_id, 0x180 + node_id
    return send_and_wait(bus, tx, rx, 0xA4, payload)

# ---------- A6: Single-turn position + direction + max speed ----------
def send_A6_single(bus, node_id, angle_deg, direction, max_speed_dps):
    """
    0xA6: DATA[1]=dir(0x00 CW, 0x01 CCW), [2..3]=maxSpeed(uint16, 1 dps/LSB),
          [4..7]=angle(uint32, 0.01deg/LSB; 0~36000*ratio-1)  :contentReference[oaicite:5]{index=5}
    """
    if direction not in ("cw", "ccw"):
        raise ValueError("direction must be 'cw' or 'ccw'")
    dir_byte = b"\x00" if direction == "cw" else b"\x01"
    max_speed = max(0, min(0xFFFF, int(round(max_speed_dps))))
    # 단일회전은 부호 없는 카운트
    angle_cnt_u32 = max(0, int(round(angle_deg * 100.0)))  # 0.01deg/LSB
    payload = dir_byte + struct.pack("<H", max_speed) + struct.pack("<I", angle_cnt_u32)
    tx, rx = 0x140 + node_id, 0x180 + node_id
    return send_and_wait(bus, tx, rx, 0xA6, payload)

def main():
    ap = argparse.ArgumentParser(description="Quick angle sender for A3/A4/A6")
    ap.add_argument("--channel", default="can0")
    ap.add_argument("--bringup", action="store_true", help="(선택) ip link로 can up 시도")
    ap.add_argument("--bitrate", type=int, default=1000000)
    ap.add_argument("--node", type=int, required=True, help="Motor ID (1..32)")
    ap.add_argument("--mode", choices=["A3","A4","A6"], required=True)
    ap.add_argument("--angle", type=float, required=True, help="target angle (deg)")
    ap.add_argument("--speed", type=float, default=120.0, help="max speed dps (A4/A6용)")
    ap.add_argument("--dir", choices=["cw","ccw"], help="A6 단일회전 방향")
    ap.add_argument("--run", action="store_true", help="RUN(0x88) 먼저 보내기")
    args = ap.parse_args()

    if args.bringup:
        import subprocess
        try:
            subprocess.run(["sudo", "ip", "link", "set", args.channel, "down"], check=False)
            subprocess.run(["sudo", "ip", "link", "set", args.channel, "up", "type", "can", "bitrate", str(args.bitrate)], check=True)
        except Exception as e:
            print(f"[WARN] bringup 실패: {e}")

    bus = open_bus(args.channel)

    if args.run:
        cmd_run(bus, args.node)

    if args.mode == "A3":
        rep = send_A3_multi(bus, args.node, args.angle)
        print(f"A3 node={args.node} angle={args.angle}deg -> {'OK' if rep else 'NO-REPLY'}")

    elif args.mode == "A4":
        rep = send_A4_multi(bus, args.node, args.angle, args.speed)
        print(f"A4 node={args.node} angle={args.angle}deg speed<={args.speed}dps -> {'OK' if rep else 'NO-REPLY'}")

    elif args.mode == "A6":
        if not args.dir:
            raise SystemExit("A6는 --dir (cw/ccw) 지정이 필요합니다.")
        rep = send_A6_single(bus, args.node, args.angle, args.dir, args.speed)
        print(f"A6 node={args.node} {args.dir} angle={args.angle}deg speed<={args.speed}dps -> {'OK' if rep else 'NO-REPLY'}")

    bus.shutdown()

if __name__ == "__main__":
    main()
