# Controller 리팩토링 Plan

## 목표
- `ControllerNode`: ROS2 노드. 센서 수신 → 제어기 선택 → 토크 발행
- `NominalController`: NSC 로직 (Pinocchio 동역학) 독립 클래스
- `PolicyController`: PD + PI 로직 독립 클래스
- `SafetyLimiter`: 토크 발행의 최종 게이트 (단일 클래스) — 조인트 각도 위반 OR 속도 비상정지 → 발행 차단(zero), 정상 → LPF+클리핑 후 발행

---

## 1. 최종 파일 구조

```
util_develop/
├── controller_node.py       # ROS2 ControllerNode (메인 노드)
|── base_controller.py       # Abstract Class (제어기 기본 슈퍼클래스)
├── nominal_controller.py    # NominalController (NSC 로직)
├── policy_controller.py     # PolicyController (PD + PI 로직)
└── safety_limiter.py        # SafetyLimiter (단일 통합 안전 게이트)
```

기존 `core/control/` 파일들(pd_controller, pi_controller, safety_limiter 등)은
그대로 유지하고 새 클래스들이 **내부적으로 재사용**한다.

---

## 2. ~~공통 인터페이스 (Abstract Base)~~ - 완료

```python
# 두 제어기 모두 이 인터페이스를 따른다
class BaseController(ABC):
    @abstractmethod
    def compute(
        self,
        joint_msg: JointState,
        imu_msg: BaseStates,
        dt_sec: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """
        Args:
            joint_msg: calibration offset이 적용된 JointState (pos/vel/effort)
            imu_msg:   최신 IMU 메시지 (quaternion, gyro, accel)
            dt_sec:    이전 compute() 이후 경과 시간 [s]
        Returns:
            raw_torque: (num_joints,) [Nm], SafetyLimiter 적용 전
        """

    @abstractmethod
    def reset(self) -> None:
        """내부 상태 초기화. 모드 전환 시 ControllerNode가 호출."""
```

---

## 3. ~~PolicyController 설계~~ - 완료

---

## 4. ~~NominalController 설계~~ - 완료

---

## 5. ~~SafetyLimiter 설계 (통합)~~ 완료

---

## 6. ControllerNode 설계

**역할**: ROS2 노드. 센서 수신 → 키보드 모드 전환 → 제어기 실행 → 안전 검사 → 토크 발행 오케스트레이터

### 6-1. 현재 구현 상태 (controller_node.py)

**✅ 구현 완료:**
```
__init__():
  ├── Launch 파라미터 (control_rate_hz, yaml_path, urdf_path, checkpoint_path, action_timeout_sec, debug_print_period_sec)
  ├── YAML 로드 → self.cfg dict
  ├── cfg["nsc_urdf_path"] = urdf_path (런타임 주입)
  ├── cfg["policy_checkpoint_path"] = checkpoint_path (런타임 주입, checkpoint_path가 있을 때만)
  ├── 제어기 생성: NominalController(cfg), PolicyController(cfg), SafetyLimiter(cfg)
  ├── Subscriber: joint_states(JointState) + /imu(BaseStates) → ApproximateTimeSynchronizer
  ├── Publisher: /torque(Float32MultiArray)
  ├── LatestBuffers (joint_state_msg, imu_msg)
  ├── 키보드 모드 전환: threading + termios (_keyboard_listener_loop)
  │   ├── 'p' → publish_mode = 'policy'
  │   ├── 'n' → publish_mode = 'nominal'
  │   └── 'q' / Ctrl+C → rclpy.shutdown()
  ├── publish_mode = None (idle 상태로 시작, 키보드 입력 전까지 토크 미발행)
  ├── _prev_mode = None (모드 전환 감지용)
  ├── num_joints, last_control_time (타이밍 변수)
  ├── sync_callback() → joint_callback + imu_callback
  ├── joint_callback() → buffers 저장 (calibration 미사용 — 이미 적용된 데이터 수신)
  ├── imu_callback() → buffers 저장
  ├── reset() → safety_limiter + 양쪽 controller reset
  ├── _switch_mode() → 모드 전환 시 이전 제어기 reset + SafetyLimiter LPF reset
  ├── control_timer (create_timer) → _control_loop 등록 (control_rate_hz 기반)
  ├── _control_loop() → dt 계산 → 센서 유효성 → 모드 전환 → compute() → SafetyLimiter → 발행
  │   ├── publish_mode == None → skip (idle 대기)
  │   └── is_blocked → control_timer.cancel() (latching kill switch)
  ├── _publish_torque_command() → Float32MultiArray 발행
  └── main() → rclpy.init/spin/shutdown + termios 복원
```

### 6-2. 초기화 — ✅ 구현 완료

```python
def __init__(self):
    # ... (기존 구현 유지) ...

    # CalibrationManager: 사용 안 함
    # → calibrated offset이 이미 적용된 JointState가 넘어온다고 가정

    # Action subscriber: 사용 안 함
    # → PolicyController가 compute() 내부에서 inference까지 모두 수행

    # 모드 전환 (None = idle, 키보드 입력 전까지 토크 미발행)
    self.publish_mode = None
    self._prev_mode = None

    # 제어 루프 타이밍
    self.num_joints = len(self.cfg["joint_names"])
    self.last_control_time = self.get_clock().now()

    # Control loop timer
    control_period_sec = 1.0 / max(self.control_rate_hz, 1.0)
    self.control_timer = self.create_timer(control_period_sec, self._control_loop)
```

### 6-3. 모드 전환 로직 — ✅ 구현 완료

키보드 입력(`_keyboard_listener_loop`)은 `self.publish_mode`만 변경한다.
실제 전환 처리는 `_control_loop` 진입 시 `_switch_mode()`를 호출하여 수행한다.

```python
def _switch_mode(self, new_mode: str) -> None:
    """Handle mode transition: reset previous controller + safety limiter LPF."""
    if new_mode == self._prev_mode:
        return
    if self._prev_mode == 'policy':
        self.policy_controller.reset()
    elif self._prev_mode == 'nominal':
        self.nominal_controller.reset()
    self.safety_limiter.reset()
    self.get_logger().info(f"Controller switched: {self._prev_mode} -> {new_mode}")
    self._prev_mode = new_mode
```

> **주의**: `safety_limiter.reset()`은 LPF 상태 + kill switch를 모두 리셋한다.
> kill switch가 발동된 상태에서 모드 전환으로 리셋되는 것이 의도적인지 확인 필요.
> 만약 kill switch는 모드 전환으로도 리셋 불가해야 한다면, reset() 분리 필요.

### 6-4. 제어 루프 (`_control_loop`) 상세 설계

```python
def _control_loop(self):
    # ---- 1. dt 계산 ----
    now_time = self.get_clock().now()
    dt_sec = (now_time - self.last_control_time).nanoseconds * 1e-9
    if dt_sec <= 0.0:
        dt_sec = 1e-3
    self.last_control_time = now_time

    # ---- 2. 센서 데이터 유효성 검사 ----
    joint_msg = self.buffers.joint_state_msg
    imu_msg = self.buffers.imu_msg
    if joint_msg is None:
        return  # 아직 joint_state를 수신하지 못함 → skip

    # ---- 3. 모드 전환 감지 ----
    self._switch_mode(self.publish_mode)

    # ---- 4. 활성 제어기 실행 ----
    # PolicyController가 내부에서 policy inference + PD/PI를 모두 수행.
    # action subscription / timeout / decode 불필요.
    if self.publish_mode == 'policy':
        raw_torque, _, _ = self.policy_controller.compute(joint_msg, imu_msg, dt_sec)

    elif self.publish_mode == 'nominal':
        raw_torque, _, _ = self.nominal_controller.compute(joint_msg, imu_msg, dt_sec)

    else:
        # 알 수 없는 모드 → zero torque
        raw_torque = np.zeros(self.num_joints)

    # ---- 5. SafetyLimiter 적용 ----
    joint_pos = np.asarray(joint_msg.position, dtype=float).flatten()
    joint_vel = np.asarray(joint_msg.velocity, dtype=float).flatten()
    safe_torque, is_blocked = self.safety_limiter.apply(raw_torque, joint_pos, joint_vel)

    # ---- 6. 블록 처리 ----
    if is_blocked:
        self.get_logger().error("SafetyLimiter BLOCKED! Publishing zero torque.")
        self.control_timer.cancel()  # 영구 정지 (latching kill switch)
        safe_torque = np.zeros(self.num_joints)

    # ---- 7. 토크 발행 ----
    self._publish_torque_command(safe_torque)

```

### ~~6-5. Action 수신 및 디코딩~~ — 삭제

> PolicyController가 `compute()` 내부에서 sensor → policy inference → PD/PI를 일체 수행.
> action subscription, timeout, decode 로직 모두 불필요.
> 모드 전환은 **키보드만** 담당.

### ~~6-6. Observation 발행~~ - 삭제
> 발행할 필요 없음.

### 6-7. 토크 발행 + main()

```python
def _publish_torque_command(self, torque: np.ndarray):
    msg = Float32MultiArray()
    msg.data = torque.astype(np.float32).tolist()
    self.torque_command_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, node.settings)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
```

### 6-8. 구독/발행 토픽

| 방향 | 토픽 | 타입 | 용도 |
|------|------|------|------|
| Sub | `/joint_states` | JointState | 관절 위치/속도 (calibrated offset 이미 적용됨) |
| Sub | `/imu` | BaseStates | IMU 데이터 (TODO: 토픽명/메시지 구성 확정) |
| Pub | `/torque` | Float32MultiArray | 최종 안전 토크 명령 |

> `goat/actions` subscription 삭제 — PolicyController가 내부에서 inference 수행
> `goat/observations` 발행은 후순위 (BaseStates 필드명 확정 후)

### 6-9. 구현 우선순위

1. ~~**import 정리** + docstring 갱신 + `rclpy` import 추가~~ ✅
2. ~~**checkpoint_path 파라미터 declare**~~ ✅
3. ~~**publish_mode / _prev_mode 초기화**, `_nsc_active` 제거~~ ✅
4. ~~**타이밍 변수 초기화** (last_control_time, num_joints)~~ ✅
5. ~~**_switch_mode** (모드 전환 시 reset 처리)~~ ✅
6. ~~**_control_loop** (핵심 제어 루프)~~ ✅
7. ~~**_publish_torque_command + main()**~~ ✅
8. ~~**control_timer 생성**~~ ✅

---

## 7. Config 처리 방식

기존 `GoatModel` + `model_builder.py` factory 패턴을 사용하지 않는다.
**YAML dict를 각 클래스 생성자에 직접 전달**하며, 각 클래스가 필요한 키만 파싱한다.

```python
# controller_node.py __init__ 내부
with open(yaml_path) as f:
    cfg = yaml.safe_load(f)

nominal_controller = NominalController(cfg, urdf_path)
policy_controller  = PolicyController(cfg)
safety_limiter     = SafetyLimiter(cfg)
```

각 클래스의 `__init__`에서 cfg dict를 파싱하고 필요한 numpy 배열/스칼라로 변환한다.
존재하지 않는 키에 대해서는 명확한 `KeyError` 또는 기본값을 명시한다.

---

## 8. 현재 코드 → 신규 코드 매핑

| 현재 위치 | 신규 위치 |
|-----------|-----------|
| `control_pipeline.compute_control()` PD+PI | `PolicyController.compute()` |
| `control_pipeline.compute_natural_torque()` | `NominalController.compute()` |
| `control_pipeline.apply_calibrated_offset()` | `ControllerNode.joint_callback()` 내부 |
| `JointSafetyLimiter` | `SafetyLimiter.apply()` — 위반 시 zero 차단으로 대체 |
| `TorqueSafetyLimiter` | `SafetyLimiter.apply()` 내부 통합 |
| `control_node.py` estop 인라인 | `SafetyLimiter.apply()` 내부 통합 |
| `control_node._control_loop()` | `ControllerNode._control_loop()` |
| NSC aux 함수들 (compute_contact_jacobian 등) | `NominalController` private 메서드 |
| `RobotState` 데이터 타입 | **폐기** — `JointState` + `BaseStates` 직접 사용 |
| `GoatModel` + `model_builder.py` factory | **미사용** — `cfg: dict` 직접 전달로 대체 |

---

## 9. 구현 순서

1. ~~**`safety_limiter.py`**~~ ✅ 완료 — cfg dict, latching kill switch, numpy array apply()
2. ~~**`base_controller.py`**~~ ✅ 완료 — compute(JointState, BaseStates, dt_sec) 시그니처
3. ~~**`goat_config.yaml`**~~ ✅ 완료 — `joint_vel_estop_threshold` 추가, `output_limit_per_joint` 제거, `joint_pos_soft_limit_coeff` 추가
4. ~~**`policy_controller.py`**~~ ✅ 1차 구현 완료 — PD(legs) + PI(wheels) + conditional anti-windup (wheel_tau_limit 기반)
5. ~~**`nominal_controller.py`**~~ ✅ 1차 migration 완료 — cfg dict 파싱, Pinocchio 동역학, compute(JointState, BaseStates, dt_sec) 시그니처
6. ~~**`controller_node.py`**~~ ✅ 완료 — D-1~D-8 전체 구현 (제어 루프, 모드 전환, 안전 게이트, main entrypoint)

---

### 남은 작업 — 코드 리뷰 결과 (2차)

#### Step A: `policy_controller.py` 버그 수정 ✅ 완료

- ~~A-1: `self.get_logger()` → `print()` 교체~~
- ~~A-2: `base_state.z` → `base_state.gyro.z` 수정~~
- ~~A-3: dict 순회 `.items()` 수정~~

#### Step B: `nominal_controller.py` — 수정 불필요

- B-1: `wheel_nv_id` 하드코딩 → 로봇 바뀔 일 없으므로 유지
- B-2: BaseStates 필드명 → 따로 수정 예정, 현재 건드리지 않음

#### Step C: `base_controller.py` 반환 타입 확정 ✅ 완료

- ~~`compute()` → `tuple[np.ndarray, np.ndarray, np.ndarray | None]` 3-tuple 반환으로 확정~~

#### Step D: `controller_node.py` 완성 (핵심 작업)

**전제**: CalibrationManager 사용 안 함 (calibrated offset이 이미 적용된 JointState 수신).
PolicyController가 내부에서 policy inference까지 모두 수행하므로 action subscription 불필요.

아래 순서대로 진행:

| # | 작업 | 상세 |
|---|------|------|
| D-1 | ~~**import 정리**~~ | ✅ 미사용 import 제거, `rclpy` import 추가, docstring 갱신 |
| D-2 | ~~**checkpoint_path 파라미터 선언**~~ | ✅ `self.declare_parameter("checkpoint_path", "")` 추가 |
| D-3 | ~~**`publish_mode` 초기화 + `_prev_mode`**~~ | ✅ `publish_mode = None` (idle), `_prev_mode = None`. `_nsc_active` 제거 |
| D-4 | ~~**타이밍 변수 초기화**~~ | ✅ `num_joints`, `last_control_time` 추가 |
| D-5 | ~~**`_switch_mode()`**~~ | ✅ plan 6-3 기반 구현 완료 |
| D-6 | ~~**`_control_loop()`**~~ | ✅ dt 계산 → 센서 유효성 → 모드 전환 → compute() → SafetyLimiter → 발행. idle skip + kill switch cancel |
| D-7 | ~~**`_publish_torque_command()` + `main()`**~~ | ✅ Float32MultiArray 발행 + rclpy entrypoint + termios 복원 |
| D-8 | ~~**control_timer 생성**~~ | ✅ `create_timer(1/control_rate_hz, _control_loop)` |
| D-9 | observation 발행 | 생략 (필요 없음)
| D-10| CSV 로깅 | 생략 (필요 없음. MATLAB에서 원격으로 진행할 예정)


#### 설계 결정 (확정)

| # | 항목 | 결정 | 비고 |
|---|------|------|------|
| F-1 | ~~PI 제어기 유지 여부~~ | ✅ **유지** | 우선 유지, 추후 필요 시 재검토 |
| F-2 | ~~margin 처리 방식~~ | ✅ **coefficient 방식** | 현재 구현 그대로 사용 |
| F-3 | ~~kill switch reset 정책~~ | ✅ **프로세스 재시작만 리셋 가능** | control_timer.cancel() + SafetyLimiter latching 이중 안전장치 |
| F-4 | BaseStates 필드명 | ⏳ **별도 수정 예정** | vel vs acc 확인 필요. 현재 plan 범위 밖 |
| F-5 | ~~NominalController lazy init~~ | ✅ **trajectory만 lazy init** | 시작 joint각이 필요하므로 trajectory만 첫 compute() 시 초기화. 나머지(Pinocchio 모델 등)는 즉시 로드 유지 |

---

## 10. 주요 설계 결정 사항

| 결정 | 이유 |
|------|------|
| RobotState 폐기, JointState+BaseStates 직접 사용 | 변환 레이어 제거, 노드 콜백에서 받은 메시지를 그대로 전달 |
| CalibrationManager 미사용 | calibrated offset이 이미 적용된 JointState가 넘어온다고 가정 |
| cfg dict를 생성자에 직접 전달 | GoatModel/factory 패턴 불필요, 각 클래스가 필요한 키만 파싱 |
| SafetyLimiter가 Boolean 게이트 반환 | 발행 차단 결정 권한을 노드에 위임, 테스트 용이 |
| JointSafetyLimiter + TorqueSafetyLimiter 통합 | 두 조건 모두 "zero 토크" 결과 → 단일 게이트로 충분 |
| BaseController.compute()가 imu_msg 항상 수신 | 인터페이스 통일 (PolicyController도 imu 사용) |
| NominalController: Pinocchio 모델은 즉시 로드, trajectory만 lazy init | 시작 joint각이 필요하므로 trajectory는 첫 compute() 시 초기화 |
| 모드 전환 시 SafetyLimiter.reset() 병행 호출 | LPF 상태 리셋으로 토크 점프 방지 |
| kill switch 발동 시 control_timer.cancel() | SafetyLimiter latching + 루프 영구 정지 이중 안전장치. 노드 재시작만 복구 가능 |
| AbstractBaseController 도입 | 향후 3번째 제어기 추가 시 인터페이스 보장 |
| compute() 반환값 3-tuple | `(tau_cmd, target_pos, wheel_ref_or_None)` — 로깅/디버그용 참조 정보 포함 |
| PolicyController 내부에서 policy inference 수행 | controller가 sensor→inference→PD/PI 전체 처리. action subscription 불필요, 모드 전환은 키보드만 담당 |
| PolicyController PI 제어기 유지 | 우선 유지, 추후 필요 시 재검토 |
| SafetyLimiter margin: coefficient 방식 | `joint_pos_limit * coeff`로 soft limit 계산 |
| D-9 observation 발행 생략 | 필요 없음 |
| D-10 CSV 로깅 생략 | MATLAB에서 원격으로 진행할 예정 |
