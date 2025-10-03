# ebimu_live_visualizer.py
# pip install pyserial matplotlib

import serial
import threading
import time
from collections import deque
import math
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (3D 활성화용)

# -------------------- 설정 --------------------
PORT = "/dev/ttyUSB0"   # Windows면 "COM3"
BAUD = 115200
TIMEOUT = 1.0

WINDOW_SEC = 10.0
UPDATE_HZ = 50
# --------------------------------------------

# 공유 데이터
lock = threading.Lock()
latest_vec = None  # 길이 14: quat(4)+gyro(3)+acc(3)+mag(3)+time(1)

# 그래프용 ring buffer
N = int(WINDOW_SEC * UPDATE_HZ)
ts = deque(maxlen=N)
gyro_x = deque(maxlen=N); gyro_y = deque(maxlen=N); gyro_z = deque(maxlen=N)
acc_x  = deque(maxlen=N); acc_y  = deque(maxlen=N); acc_z  = deque(maxlen=N)
mag_x  = deque(maxlen=N); mag_y  = deque(maxlen=N); mag_z  = deque(maxlen=N)

def parse_line(line: str):
    # 기대 포맷: "*v1,v2,...,v14"
    if not line.startswith('*'):
        return None
    try:
        parts = [float(x) for x in line[1:].split(',')]
    except ValueError:
        return None
    if len(parts) != 14:
        return None
    return parts

def reader_thread():
    global latest_vec
    try:
        ser = serial.Serial(PORT, BAUD, timeout=TIMEOUT)
    except Exception as e:
        print(f"[ERROR] Serial open failed: {e}")
        sys.exit(1)

    while True:
        try:
            raw = ser.readline().decode('utf-8', errors='ignore').strip()
            vec = parse_line(raw)
            if vec is None:
                continue
            with lock:
                latest_vec = vec
        except Exception:
            continue

def quat_to_R(w, x, y, z):
    n = math.sqrt(w*w + x*x + y*y + z*z)
    if n == 0:
        return [[1,0,0],[0,1,0],[0,0,1]]
    w, x, y, z = w/n, x/n, y/n, z/n
    R = [
        [1-2*(y*y+z*z),   2*(x*y - z*w),   2*(x*z + y*w)],
        [  2*(x*y + z*w), 1-2*(x*x+z*z),   2*(y*z - x*w)],
        [  2*(x*z - y*w),   2*(y*z + x*w), 1-2*(x*x+y*y)]
    ]
    return R

def update_buffers(vec):
    # vec: 14
    w,x,y,z = vec[0:4]
    gx,gy,gz = vec[4:7]   # 보통 deg/s
    ax,ay,az = vec[7:10]  # m/s^2 또는 g
    mx,my,mz = vec[10:13] # µT
    t_ms     = vec[13]

    ts.append(t_ms/1000.0)

    gyro_x.append(gx); gyro_y.append(gy); gyro_z.append(gz)
    acc_x.append(ax);  acc_y.append(ay);  acc_z.append(az)
    mag_x.append(mx);  mag_y.append(my);  mag_z.append(mz)

    return (w,x,y,z)

# -------------- Matplotlib Figure 구성 --------------
plt.figure(figsize=(12,6))

# 좌: 시계열 그래프 2개(gyro/acc)
ax1 = plt.subplot(2,2,1)
ax2 = plt.subplot(2,2,3)

# 우: 3D 자세
ax3 = plt.subplot(1,2,2, projection='3d')

# Gyro plot
ax1.set_title("Gyro (x,y,z)")
ax1.set_xlabel("time [s]")
ax1.set_ylabel("deg/s (or rad/s)")
l_gx, = ax1.plot([], [], label='gx')
l_gy, = ax1.plot([], [], label='gy')
l_gz, = ax1.plot([], [], label='gz')
ax1.legend(loc='upper right')

# Acc plot
ax2.set_title("Accel (x,y,z)")
ax2.set_xlabel("time [s]")
ax2.set_ylabel("m/s^2 (or g)")
l_ax, = ax2.plot([], [], label='ax')
l_ay, = ax2.plot([], [], label='ay')
l_az, = ax2.plot([], [], label='az')
ax2.legend(loc='upper right')

# 3D 축 범위/라벨
ax3.set_title("3D Attitude (Body Axes)")
ax3.set_xlim(-1.2, 1.2); ax3.set_ylim(-1.2, 1.2); ax3.set_zlim(-1.2, 1.2)
ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z')

# Body 축을 Line3D로 준비 (원점→축 끝점)
# 각 라인을 업데이트할 때 set_data_3d 로 좌표만 갱신
bx_line, = ax3.plot([0,1], [0,0], [0,0])  # body X
by_line, = ax3.plot([0,0], [0,1], [0,0])  # body Y
bz_line, = ax3.plot([0,0], [0,0], [0,1])  # body Z

def animate(_):
    with lock:
        vec = latest_vec[:] if latest_vec is not None else None

    if vec is None:
        return l_gx, l_gy, l_gz, l_ax, l_ay, l_az, bx_line, by_line, bz_line

    # 버퍼 갱신 + 쿼터니언
    w,x,y,z = update_buffers(vec)

    # Gyro/Accel 라인 데이터
    tt = list(ts)
    l_gx.set_data(tt, list(gyro_x))
    l_gy.set_data(tt, list(gyro_y))
    l_gz.set_data(tt, list(gyro_z))
    l_ax.set_data(tt, list(acc_x))
    l_ay.set_data(tt, list(acc_y))
    l_az.set_data(tt, list(acc_z))

    # x축 범위
    if tt:
        ax1.set_xlim(max(tt[0], tt[-1]-WINDOW_SEC), tt[-1])
        ax2.set_xlim(max(tt[0], tt[-1]-WINDOW_SEC), tt[-1])

    # y축 자동 스케일
    for ax, xs, ys, zs in [
        (ax1, gyro_x, gyro_y, gyro_z),
        (ax2,  acc_x,  acc_y,  acc_z),
    ]:
        allv = list(xs)+list(ys)+list(zs)
        if allv:
            vmin = min(allv); vmax = max(allv)
            if vmin == vmax:
                vmin -= 1; vmax += 1
            ax.set_ylim(vmin, vmax)

    # 3D 자세 업데이트(쿼터니언 → 회전행렬 → body 축 끝점)
    R = quat_to_R(w,x,y,z)
    bx = (R[0][0], R[1][0], R[2][0])  # body X
    by = (R[0][1], R[1][1], R[2][1])  # body Y
    bz = (R[0][2], R[1][2], R[2][2])  # body Z

    # 각 라인의 끝점 좌표를 갱신(원점(0,0,0) → (bx/by/bz))
    bx_line.set_data([0, bx[0]], [0, bx[1]])
    bx_line.set_3d_properties([0, bx[2]])

    by_line.set_data([0, by[0]], [0, by[1]])
    by_line.set_3d_properties([0, by[2]])

    bz_line.set_data([0, bz[0]], [0, bz[1]])
    bz_line.set_3d_properties([0, bz[2]])

    return l_gx, l_gy, l_gz, l_ax, l_ay, l_az, bx_line, by_line, bz_line

# 리더 스레드 시작
t = threading.Thread(target=reader_thread, daemon=True)
t.start()

ani = FuncAnimation(plt.gcf(), animate, interval=1000//UPDATE_HZ, blit=False)
plt.tight_layout()
plt.show()
