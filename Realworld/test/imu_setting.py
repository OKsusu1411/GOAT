import serial
import threading
import time
import math

PORT = "/dev/ttyUSB0"   # 또는 "COM3"
BAUD = 115200

ser = serial.Serial(port=PORT, baudrate=BAUD, timeout=1)

lock = threading.Lock()
sdata = []  # 최신 패킷(리스트)

def split_packet(vec):
    """14개 벡터를 의미 있는 dict로 변환"""
    if len(vec) != 14:
        return None
    w, x, y, z = vec[0:4]
    gx, gy, gz = vec[4:7]
    ax, ay, az = vec[7:10]
    mx, my, mz = vec[10:13]
    t_ms = vec[13]
    return {
        "quat": {"w": w, "x": x, "y": y, "z": z},
        "gyro": {"x": gx, "y": gy, "z": gz},   # 보통 deg/s(설정 확인)
        "acc":  {"x": ax, "y": ay, "z": az},   # m/s^2 또는 g(설정 확인)
        "mag":  {"x": mx, "y": my, "z": mz},   # 보통 µT
        "time_ms": t_ms
    }

def data_parser():
    global sdata
    while True:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue
            # 기대 포맷: "*<v1>,<v2>,...,<v14>"
            if not line.startswith('*'):
                continue
            parts = line[1:].split(',')
            # 숫자 변환
            try:
                vec = list(map(float, parts))
            except ValueError:
                continue
            with lock:
                sdata = vec
        except Exception:
            # 필요하면 로그 출력
            continue

th = threading.Thread(target=data_parser, daemon=True)
th.start()
time.sleep(0.3)

try:
    while True:
        with lock:
            vec = sdata[:]  # 스냅샷
        pkt = split_packet(vec)
        if pkt:
            # 보기 좋게 한 줄 요약 출력
            q = pkt["quat"]; g = pkt["gyro"]; a = pkt["acc"]; m = pkt["mag"]; t = pkt["time_ms"]
            print(f"[{t:.0f} ms] "
                  f"quat=({q['w']:.4f},{q['x']:.4f},{q['y']:.4f},{q['z']:.4f}) "
                  f"gyro=({g['x']:.3f},{g['y']:.3f},{g['z']:.3f}) "
                  f"acc=({a['x']:.3f},{a['y']:.3f},{a['z']:.3f}) "
                  f"mag=({m['x']:.1f},{m['y']:.1f},{m['z']:.1f})")
        time.sleep(0.01)

except KeyboardInterrupt:
    pass
finally:
    try:
        ser.close()
    except Exception:
        pass
