# GOAT Realworld 코드 분석

## 1. 패키지 구조

```
Realworld/src/
├── goat_control/          # 메인 제어 패키지
│   ├── goat_control/
│   │   ├── core/
│   │   │   ├── control/   # 제어기 (pd, pi, safety, pipeline)
│   │   │   ├── estimation/# 상태추정 (state_manager, calibration, filters, imu)
│   │   │   ├── comm/      # 통신 (can, motor_driver, motor_state_collector)
│   │   │   └── model/     # 로봇 모델 (goat_model, model_builder)
│   │   ├── nodes/         # ROS2 노드 9개
│   │   └── util_develop/  # 미완성 개발용 파일 3개 (미통합)
│   ├── config/goat_config.yaml
│   └── launch/goat_control_system.launch.py
├── goat_description/      # URDF, 메쉬, imu_tf_publisher
├── goat_sysid/            # 시스템 식별 (마찰, breakaway torque)
└── motor_interfaces/      # 커스텀 ROS2 메시지 정의
```

## 2. 데이터 흐름

```
CAN 모터 + 시리얼 IMU (하드웨어)
        ↓
motor_io_node  ──→  joint_states (JointState)
state_estimation_node ──→ /goat/imu_data (BaseStates)
        ↓
control_node (200 Hz)
  ├─ RobotState 합성 (StateManager → SI 단위)
  ├─ 액션 디코딩 (delta pos + wheel speed)
  ├─ PD/PI/NSC 제어 계산
  ├─ 안전 제한 (TorqueSafetyLimiter, JointSafetyLimiter)
  └─ goat/torque_commands, goat/observations 발행
        ↓
agent_node (PyTorch 정책 추론)
  └─ goat/actions → control_node (피드백 루프)
        ↓
motor_io_node ──→ CAN 토크 명령 전송
```

## 3. ROS2 토픽

| 토픽 | 타입 | 방향 |
|------|------|------|
| `joint_states` | sensor_msgs/JointState | motor_io → control |
| `/goat/imu_data` | motor_interfaces/BaseStates | state_est → control |
| `goat/actions` | std_msgs/Float32MultiArray | agent → control |
| `goat/torque_commands` | std_msgs/Float32MultiArray | control → motor_io |
| `goat/observations` | std_msgs/Float32MultiArray | control → agent |
| TF `odom→base_link` | tf2 | imu_tf_publisher 발행 |

## 4. 핵심 알고리즘

### 액션 해석
- 입력: 8개 액션 배열 (빈 배열 → NSC 모드)
- 조인트 0-5: delta position [rad] (자연 자세 기준 오프셋)
- 휠 6-7: 목표 속도 [rad/s]
- `desired_pos = natural_pos + delta_action`

### PD 제어 (조인트 0-5)
```
tau = Kp*(target_pos - pos) + Kd*(0 - vel)
```

### PI 제어 with Anti-windup (휠 6-7)
```
error = desired_speed - measured_speed
integrator += error * dt  (포화 시 조건부 freeze)
output = clip(Kp*error + Ki*integrator, ±limit)
```

### NSC (Natural Standing Configuration)
- Pinocchio 기반 전신 동역학 + 접촉 구속 야코비안
- 외부 루프: 휠 위치 제어 → 내부 루프: 자세 제어
- 다리: 기준 궤적 블렌딩(1000포인트) + PD 오차 피드백
- 구속: `Jc @ qdd = -Jcdot @ v` (휠 슬립 방지)

## 5. 주요 파라미터 (goat_config.yaml)

| 파라미터 | 값 |
|---------|-----|
| 제어 주기 | 200 Hz |
| 자연 자세 | [0, 0, 0.738, -0.738, 1.463, -1.463, 0, 0] rad |
| 토크 상수 (조인트) | 0.2616 Nm/A |
| 토크 상수 (휠) | 0.2478 Nm/A |
| 기어비 (무릎) | 0.5, 나머지 1.0 |
| Hip Kp/Kd | 0.3 / 0.015 |
| Thigh Kp/Kd | 0.27 / 0.01 |
| Knee Kp/Kd | 0.23 / 0.04 |
| Wheel Kp/Ki | 0.17 / 0.15 |
| 안전 LPF α | 0.951 |
| max_torque | 0.0 (비활성화됨!) |
| 비상정지 임계값 | 0.6 rad/s |

## 6. 모터 통신 프로토콜 (MG 시리즈, CAN)

- 상태 읽기: 0x9A/0x9C/0x92/0x94
- 토크 제어: 0xA1 (16-bit current in mA)
- 속도/위치 모드: 0xA2~0xA6
- 각도 스케일: 0.001°/LSB, 속도: 0.01°s/LSB

## 7. 문제점 및 리팩토링 대상

### 심각 (동작 안전성)
- `max_torque_per_joint` 전부 0.0 → 토크 클리핑 비활성화 상태
- 비상정지 임계값(0.6 rad/s)이 임의적이고 문서화 없음
- IMU ↔ 조인트 데이터 간 타임스탬프 동기화 없음

### 코드 품질
- `control_pipeline.py` 614줄 단일 파일에 NSC 포함 → 분리 필요
- `util_develop/` 3개 파일 미통합 상태로 존재 (controller.py, nominal_controller.py, policy_controller.py)
- `agent_node` 액션 스케일링 하드코딩 (joint×3.5, wheel×6.0)
- 주석 처리된 노드 4개 (calibration, log_viewer, nsc_tester, policy_keyboard_tester) - 미사용
- `GEMINI.md` 삭제됨(git 추적) - 정리 필요

### 구조 개선
- NSC 제어기 별도 모듈 분리 (현재 control_pipeline에 혼재)
- 하드코딩된 상수들: 휠 반지름(72.75mm), NSC 1000포인트, 200Hz
- CSV 로거 종료 시 미정리 (파일 깨질 수 있음)
- 모터 에러 플래그/상태 능동 모니터링 없음

### 미구현 기능
- `calibration_node`, `log_viewer_node`, `nsc_tester_node`, `policy_keyboard_tester` → setup.py에 등록되어 있으나 실질적으로 없음
- 개별 모터 헬스 워치독 없음
- 가변 강성/임피던스 제어 없음

## 8. 노드별 역할 요약

| 노드 | 주기 | 역할 |
|------|------|------|
| `motor_io_node` | ~200Hz | CAN 단독 소유자, 모터 읽기/쓰기 |
| `state_estimation_node` | 시리얼 스레드 | IMU 데이터 발행 |
| `control_node` | 200Hz | 메인 제어 루프 |
| `agent_node` | 정책 주기 | PyTorch 정책 추론, 키보드 모드 전환 |
| `imu_tf_publisher` | IMU 주기 | odom→base_link TF 발행 |

## 9. 리팩토링 우선순위 제안

1. **NSC 분리**: `control_pipeline.py`에서 NSC 로직을 `nsc_controller.py`로 분리
2. **상수 중앙화**: 하드코딩 상수들을 YAML 또는 `goat_model.py`로 이동
3. **util_develop 정리**: 통합하거나 삭제 결정
4. **안전 파라미터 활성화**: `max_torque` 설정 및 비상정지 기준 문서화
5. **미사용 노드 정리**: 빈 엔트리포인트 제거 또는 구현
6. **타임스탬프 동기화**: ApproximateTimeSynchronizer 재도입
