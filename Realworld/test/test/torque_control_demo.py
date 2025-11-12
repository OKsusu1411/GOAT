#!/usr/bin/env python3
# torque_control_demo.py
# 사용법:
#   python3 torque_control_demo.py --node 1 --series MF --amps 2.5
#   python3 torque_control_demo.py --node 2 --series MG --amps -5.0

import argparse
import struct
import time
import can

# 시리즈별 스케일 (A/LSB) - 문서의 최대치 기준으로 산출
SCALE_A_PER_LSB = {
    "MF": 16.5 / 2048.0,   # ≈ 0.00806 A/LSB
    "MG": 33.0 / 2048.0,   # ≈ 0.01611 A/LSB
    # MH도 문서상 MF/MG와 동일한 0xA1을 쓰므로 필요시 값 지정
}

def open_bus(channel="can0"):
    return can.interface.Bus(channel=channel, bustype="socketcan")

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
    send_and_wait(bus, tx_id, rx_id, 0x88)  # 모터 실행

def read_state2(bus, node_id):
    tx_id = 0x140 + node_id
    rx_id = 0x180 + node_id
    rep = send_and_wait(bus, tx_id, rx_id, 0x9C)
    return rep

def torque_command(bus, node_id, series, amps):
    if series not in SCALE_A_PER_LSB:
        raise ValueError("series는 MF 또는 MG 중 하나여야 합니다.")
    scale = SCALE_A_PER_LSB[series]
    # 원하는 전류[A] -> iqControl(LSB)
    iq = int(round(amps / scale))
    # 안전 클램프
    if iq > 2048: iq = 2048
    if iq < -2048: iq = -2048

    tx_id = 0x140 + node_id
    rx_id = 0x180 + node_id
    # DATA[4..5]에 int16 little-endian
    iq_le = struct.pack("<h", iq)
    payload = b"\x00\x00\x00" + iq_le + b"\x00\x00"
    rep = send_and_wait(bus, tx_id, rx_id, 0xA1, payload)

    return iq, rep

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="can0")
    ap.add_argument("--node", type=int, required=True)
    ap.add_argument("--series", choices=["MF","MG"], required=True, help="모터 시리즈")
    ap.add_argument("--amps", type=float, required=True, help="목표 토크전류[A]")
    args = ap.parse_args()

    bus = open_bus(args.channel)

    # 1) 실행 상태로 전환(권장)
    run_enable(bus, args.node)

    # 2) 토크 명령 전송
    iq_cmd, rep = torque_command(bus, args.node, args.series, args.amps)
    print(f"sent 0xA1: node={args.node}, target={args.amps:.3f} A, iq={iq_cmd} LSB")

    # 3) 상태2로 피드백 확인 (iq 또는 power, 속도, 엔코더)
    rep2 = read_state2(bus, args.node)
    if rep2:
        d = rep2.data
        # data: [0]=0x9C, [1]=temp, [2..3]=iq or power, [4..5]=speed(dps), [6..7]=encoder
        temp = struct.unpack("<b", d[1:2])[0]
        raw_iq_or_power = struct.unpack("<h", d[2:4])[0]
        speed_dps = struct.unpack("<h", d[4:6])[0]
        enc = struct.unpack("<H", d[6:8])[0]

        if args.series in ("MF","MG"):
            # iq 해상도: MF=(33/4096 A)/LSB, MG=(66/4096 A)/LSB
            iq_res = (33.0/4096.0) if args.series=="MF" else (66.0/4096.0)
            iq_amp = raw_iq_or_power * iq_res
            print(f"reply 0x9C: temp={temp} °C, iq={raw_iq_or_power} LSB ({iq_amp:.3f} A), speed={speed_dps} dps, enc={enc}")
        else:
            print(f"reply 0x9C: temp={temp} °C, power={raw_iq_or_power}, speed={speed_dps} dps, enc={enc}")
    else:
        print("no reply to 0x9C (state2)")

    bus.shutdown()

if __name__ == "__main__":
    main()
