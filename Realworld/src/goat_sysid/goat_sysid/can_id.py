# read_params.py
import struct
from goat_control.utils.motor import CanInterface, MotorDriver, MotorParams

BUSES = ["can0", "can1"]
NODE_IDS = [[1,2,3,4], [1,2,3,4]]   # 실제 YAML 값으로 교체
PARAMS = {30: "inputTorqueLimit(16b)", 32: "inputSpeedLimit(32b)", 36: "inputCurrentRamp(32b)"}

for bus_i, ch in enumerate(BUSES):
    ci = CanInterface(channel=ch, interface="socketcan")
    ci.open()
    for nid in NODE_IDS[bus_i]:
        drv = MotorDriver(ci, MotorParams(node_id=nid))
        for pid, name in PARAMS.items():
            payload = bytes([pid]) + b"\x00" * 6      # DATA[1]=paramID, 나머지 0
            msg = ci.txrx(drv.can_ids.tx_id, drv.can_ids.rx_id,
                          0xC0, payload, timeout=0.2)
            if msg is None:
                print(f"{ch} id{nid} {name}: NO REPLY")
                continue
            d = msg.data
            v16 = struct.unpack("<h", d[3:5])[0]
            v32 = struct.unpack("<i", d[3:7])[0]
            print(f"{ch} id{nid} {name}: raw={d.hex()} v16={v16} v32={v32}")
    ci.close()