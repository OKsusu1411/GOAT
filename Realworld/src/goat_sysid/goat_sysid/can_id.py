# bench_send.py — 로봇 불필요, vcan0로도 됨
# sudo modprobe vcan && sudo ip link add dev vcan0 type vcan && sudo ip link set up vcan0
import can, time

bus = can.Bus(interface="socketcan", channel="vcan0")
msgs = [can.Message(arbitration_id=0x141+i, data=bytes(8), is_extended_id=False)
        for i in range(4)]

# A: 지금 방식 — 매번 객체 생성
t = time.perf_counter()
for _ in range(1000):
    for i in range(4):
        bus.send(can.Message(arbitration_id=0x141+i, data=bytes(8), is_extended_id=False))
print("A 생성+전송:", (time.perf_counter()-t)*1e3/1000, "ms / 4프레임")

# B: 객체 재사용
t = time.perf_counter()
for _ in range(1000):
    for m in msgs:
        bus.send(m)
print("B 재사용:  ", (time.perf_counter()-t)*1e3/1000, "ms / 4프레임")

bus.shutdown()