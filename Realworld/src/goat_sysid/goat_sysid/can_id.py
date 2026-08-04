# gear_check.py — 컨트롤러 노드 끄고 실행, 토크 0 (손으로 움직임)
import struct, time, yaml
from goat_control.utils.motor import CanInterface, MotorDriver, MotorParams

cfg = yaml.safe_load(open("goat_config.yaml", encoding="utf-8"))
CH, NODE_ID = "can0", 1          # 대상 모터로 교체

ci = CanInterface(channel=CH, interface="socketcan"); ci.open()
drv = MotorDriver(ci, MotorParams(node_id=NODE_ID))

def read_angle_and_encoder():
    m92 = ci.txrx(drv.can_ids.tx_id, drv.can_ids.rx_id, 0x92, b"\x00"*7, timeout=0.2)
    m9c = ci.txrx(drv.can_ids.tx_id, drv.can_ids.rx_id, 0x9C, b"\x00"*7, timeout=0.2)
    if m92 is None or m9c is None:
        return None
    raw7 = m92.data[1:8]
    sign = b"\x00" if raw7[-1] < 0x80 else b"\xff"
    angle_raw = int.from_bytes(raw7 + sign, "little", signed=True)
    enc = struct.unpack("<H", m9c.data[6:8])[0]
    return angle_raw, enc

print("시작 자세로 두고 Enter")
input()
a0 = read_angle_and_encoder()
print("start:", a0)

print("천천히 약 90도 회전시킨 뒤 Enter (중간에 되돌리지 마세요)")
input()
a1 = read_angle_and_encoder()
print("end:  ", a1)

d_raw = a1[0] - a0[0]
print(f"\nangle_raw 변화: {d_raw}")
print(f"  0.01 deg/LSB 가정 -> {d_raw*0.01:.2f} deg")
print(f"  0.001 deg/LSB 가정 -> {d_raw*0.001:.2f} deg")
print(f"encoder: {a0[1]} -> {a1[1]}  (delta={a1[1]-a0[1]})")
ci.close()