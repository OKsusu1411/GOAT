import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# ---------------------------------------------------------
# [설정 1] 파일 이름
file_name = 'right_thigh_step_440_final.csv'

# [설정 2] 분석할 관절 인덱스 선택 (0 ~ 5)
# 0: Left Hip, 1: Right Hip, 2: Left Thigh, 3: Right Thigh, 4: Left Knee, 5: Right Knee
target_joint_index = 3
# [설정 3] 저장할 그래프 파일의 DPI (해상도) 설정
save_dpi = 800
# ---------------------------------------------------------

# 1. 관절 이름 매핑 및 선택
joint_names = {
    0: "Left_Hip",
    1: "Right_Hip",
    2: "Left_Thigh",
    3: "Right_Thigh",
    4: "Left_Knee",
    5: "Right_Knee"
}

# 선택한 인덱스에 해당하는 이름 가져오기
joint_name_str = joint_names.get(target_joint_index, f"Joint_{target_joint_index}")

# 2. 파일 로드 (경로 에러 방지 포함)
if not os.path.exists(file_name):
    print(f"❌ 오류: '{file_name}' 파일을 찾을 수 없습니다.")
    print(f"📂 현재 경로: {os.getcwd()}")
    sys.exit()

df = pd.read_csv(file_name)

# 3. 데이터 처리
df['time'] = df['time_sec'] + df['time_nanosec'] * 1e-9
df['time'] = df['time'] - df['time'].iloc[0]

measured_pos_deg = np.degrees(df[f'pos_{target_joint_index}'])
target_pos = df[f'target_value_{target_joint_index}']
torque_cmd = df[f'torque_command_{target_joint_index}']

# ---------------------------------------------------------
# 4. 그래프 그리기 및 저장
# ---------------------------------------------------------

# --- 그래프 1: 각도 (Angle) ---
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(df['time'], measured_pos_deg, label=f'Measured: {joint_name_str.replace("_", " ")}', linewidth=2)
ax1.plot(df['time'], target_pos, label=f'Target: {joint_name_str.replace("_", " ")}', linestyle='--', color='red', alpha=0.7)
ax1.set_title(f'[{joint_name_str.replace("_", " ")}] Angle Response', fontsize=16, fontweight='bold')
ax1.set_xlabel('Time (seconds)', fontsize=12)
ax1.set_ylabel('Angle (Degrees)', fontsize=12)
ax1.legend(loc='upper right')
ax1.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

# 파일 이름 설정 및 저장
angle_file_name = f'Graph_{joint_name_str}_Angle_Response.png'
fig1.savefig(angle_file_name, dpi=save_dpi)
print(f"💾 Angle 그래프 저장 완료: {angle_file_name}")


# --- 그래프 2: 토크 (Torque) ---
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(df['time'], torque_cmd, label=f'Torque Cmd: {joint_name_str.replace("_", " ")}', color='orange', linewidth=2)
ax2.set_title(f'[{joint_name_str.replace("_", " ")}] Torque Command', fontsize=16, fontweight='bold')
ax2.set_xlabel('Time (seconds)', fontsize=12)
ax2.set_ylabel('Torque (Nm)', fontsize=12)
ax2.legend(loc='upper right')
ax2.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

# 파일 이름 설정 및 저장
torque_file_name = f'Graph_{joint_name_str}_Torque_Command.png'
fig2.savefig(torque_file_name, dpi=save_dpi)
print(f"💾 Torque 그래프 저장 완료: {torque_file_name}")


# 최종적으로 그래프를 화면에 표시 (저장이 완료된 후)
plt.show()