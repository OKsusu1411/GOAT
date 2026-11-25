# Motor CAN C++ Migration 검토

## Conclusion

**Python 파일을 클래스별로 그대로 번역하기보다, `SocketCAN 전송 계층 + 순수 protocol 함수 + MotorIO 상태·변환 계층`의 C++ 라이브러리를 권장한다.** 현재처럼 제어기와 같은 프로세스에서 호출하며, 첫 이식에서는 `0x9C 상태 읽기 → 제어 계산 → 0xA1 전송` 순서를 보존한다.

- 기본안은 **제어 스레드 한 곳이 모든 CAN 소켓과 모터 상태를 소유**하고, 여러 버스의 응답을 `poll/ppoll`로 함께 기다리는 방식이다. 별도 CAN RX 스레드, 모터별 thread pool, 공유 cache/event는 만들지 않는다.
- `MotorIO` 내부에서는 ROS 메시지 대신 사전 할당한 수치 상태를 사용한다. ROS 메시지 구성·발행은 상위 노드가 담당한다.
- `0xA1` 응답만 사용하는 통신량 최적화는 첫 이식과 분리한다. 읽기 요청을 없애면 피드백 시점과 지연이 달라진다.
- 현재 YAML의 topology가 일관되지 않아, 실기 검증 전 실제 배선과 모터 매핑을 확인해야 한다. 파일만으로 단일/이중 버스를 확정하지 않는다.

이번 작업은 분석과 설계 제안이다. 구현 파일·설정·빌드는 변경하지 않았다. 아래 Code는 승인 검토용 인터페이스와 변경 범위이며 완성된 구현 패치가 아니다.

## Logic

### 1. 분석 기준과 가정

- 기준은 현재 체크아웃한 소스이다. 기존 문서·docstring과 구현이 다르면 구현을 따른다.
- Linux SocketCAN, Classical CAN 8-byte 데이터 프레임을 대상으로 한다. 물리 버스 수와 joint 수는 초기화 시 설정으로 결정한다.
- C++ motor 라이브러리를 먼저 만들고, 향후 C++ controller가 직접 소유하는 흐름을 가정한다. 기존 Python controller를 즉시 C++ 라이브러리에 연결하는 작업까지 포함하지 않는다.
- 성능 개선은 구조적 예상이다. 현재 timeout/jitter의 실측 자료나 모터 공식 프로토콜 문서는 제공되지 않았으므로 속도 향상 수치와 하드웨어 해석은 확정하지 않는다.

### 2. 현재 Python 계층

아래 경로는 `src/goat_control/goat_control/` 기준이다.

| 계층 | 코드 근거 | 실제 책임 |
|---|---|---|
| 제어 루프 | `nodes/controller_node.py:87–92, 161–173, 336–441, 514–532` | MotorIO 소유, 별도 제어 스레드, read → compute/safety → write, 관측용 ROS publish |
| MotorIO | `nodes/motor_io.py:22–74, 88–113` | topology 검사, 버스/driver/manager 생성, 초기화, JointState 변환, 토크 입력 API |
| MotorManager | `utils/motor/motor_manager.py:44–134, 140–197, 203–211, 301–456` | 단위·기어·offset 변환, 위치 anchor/unwrap, 모터 상태, read/write 일괄 실행 |
| MotorDriver | `utils/motor/motor_driver.py:27–74, 97–163` | node ID별 CAN ID, 명령 payload 생성, 응답 event/cache 접근 |
| CanInterface | `utils/motor/can.py:21–49, 55–75, 124–195, 238–285` | python-can 버스 소유, 초기 동기 txrx, 런타임 RX 스레드, cache/event |

```text
ControllerNode의 hardware-control thread
  └─ MotorIO
      ├─ MotorManager: 변환 + encoder 상태 + scan
      ├─ MotorDriver × N: CAN ID와 명령별 포장
      └─ CanInterface × B: SocketCAN bus + RX thread + cache/event

초기화 시 MotorManager의 N-worker pool도 사용
런타임 read/write는 해당 pool에 작업을 제출하지 않음
```

현재 `/joint_states`, `/commands`는 제어기와 MotorIO 사이의 통신 경로가 아니라 외부 관측용 발행이다. C++로 옮기면서 별도 motor ROS node를 추가할 이유는 없다.

### 3. 초기화와 runtime 데이터 흐름

**초기화**

1. joint 수, bus index 범위, `(bus, node_id)` 중복을 검사하고 버스를 연다.
2. 모터별 작업을 pool에 제출한다. 같은 버스의 `txrx()`는 bus mutex로 직렬화된다. 서로 다른 버스는 병렬 진행할 수 있다.
3. 각 모터는 `0x9A → 0x94 → 0x92 → 0x9C → PI gain` 순서로 읽는다. 다리 PI는 `0xC0`/subcommand `0x0C`, 바퀴 PI는 `0x30`이다.
4. 절대각과 encoder를 anchor로 저장한다. 관절 한계와 offset으로 계산한 중심 부근에 boot 각도를 fold하고, 이후 encoder 변화량을 누적한다.
5. 초기 JointState를 만든 다음 버스별 background RX를 시작한다.

같은 버스 mutex가 **모터 초기화 전체가 아닌 개별 txrx**를 감싸므로 다른 모터 요청이 `0x92`와 `0x9C` 사이에 끼어들 수 있다. 두 값은 동일 시각 측정이 아니다. 초기 anchor는 정지 상태에서 검증하는 것이 적절하다.

**정상 제어 tick**

```text
read_joint_state()
  모든 모터의 0x9C event clear + request 전송
  → 모든 전송이 끝난 뒤 공통 deadline 설정
  → RX thread가 (arbitration_id, command) cache와 event 갱신
  → 제어 스레드가 각 모터 event 대기
  → decode, encoder unwrap, joint 단위 변환
  → JointState 생성

controller compute → safety

write_motor(tau)
  → joint torque clip
  → I = tau / (gear × Kt)
  → current LSB 반올림·제한
  → 모터별 0xA1 전송 후 반환
```

`write_torques()`는 `motor_manager.py:437–456`에서 끝난다. docstring과 달리 응답 대기, decode, 상태 반환을 수행하지 않는다. `0xA1` 응답은 RX cache에 들어가지만 제어 상태 갱신에는 사용하지 않는다. 따라서 현재는 tick당 모터별 `0x9C request/reply + 0xA1 request/reply` 통신이 예상된다. 실제 응답 개수·지연은 CAN capture로 확인해야 한다.

### 4. 유지해야 할 프로토콜·수치 계약

| 항목 | 현재 구현 | C++ 이식 조건 |
|---|---|---|
| ID | TX=`0x140 + node_id`, RX=`0x180 + node_id`; TX ID 응답도 허용 | bus와 ID를 함께 매핑하고 두 응답 ID를 하나의 motor slot으로 정규화 |
| `0xA1` 요청 | current signed int16, little-endian, DATA[4:6] | 다른 byte는 0, current limit와 joint torque limit 모두 보존 |
| `0x9C` 응답 | DATA[1] int8 온도, [2:4] int16 전류, [4:6] int16 속도, [6:8] uint16 encoder | signedness, byte order, DLC 검증을 명시 |
| 절대각 | `0x94`: DATA[4:8] uint32, `0x92`: DATA[1:8] signed 56-bit | sign extension을 명시적으로 구현; 일반 int64 load로 대체하지 않음 |
| 단위 | angle=0.001 deg/LSB, speed=0.1 deg/s/LSB, current=0.01611328125 A/LSB | 설정을 읽고 변환; 모터 firmware에 대한 독립 검증은 필요 |
| joint 변환 | `q=deg_to_rad(theta_motor/gear)-offset`, `dq=deg_to_rad(speed_motor/gear)`, `tau=I×gear×Kt` | 기어비 방향과 offset 적용 시점을 보존 |
| mapping | 상태 포장은 `joint_indices + wheel_indices`, 명령은 입력 순서대로 driver 접근 | 설정에서 joint→motor 관계를 하나로 확정하고 읽기·쓰기 양쪽에 적용 |
| 위치 누적 | 65536 counts/rev, 반주기 초과 delta를 wrap으로 판단 | 유효한 새 0x9C에만 적분; 누락 후 반회전 이상 이동은 encoder만으로 복원 불가 |

Python의 `round()`는 동률에서 짝수 방향, C++ `std::round()`는 동률에서 0 반대 방향이다. current LSB와 angle 양자화에서 단순 치환하면 byte/값이 달라질 수 있다. Python의 양수 modulus와 C++의 음수 나머지 차이도 boot fold 구현에서 확인한다. 처음에는 기존 0.001-degree 양자화까지 보존하고, double 각도 직접 유지에 따른 변경은 별도 비교한다.

### 5. 구현에서 확인된 문제와 이식 판단

| 관찰 | 근거 | 설계에 반영할 내용 |
|---|---|---|
| topology 불일치 | `src/goat_control/config/goat_config.yaml:16–30`: N=6, bus index 길이=4, channels=[can0]이나 index 1 포함 | 현재 파일로 MotorIO 초기화 검사 실패. 실제 배선 확인 후 수정안을 별도 승인받음 |
| 기본 read timeout 50 ms | `controller_node.py:91`, `motor_io.py:31,99`; manager 자체 기본값은 3 ms | 외부에서 50 ms를 넘기므로 실제 호출 기본값은 50 ms. 설정된 deadline과 관절별 측정 시간을 노출 |
| 문서의 200 Hz timer와 실제가 다름 | launch `:46–49` 기본 100 Hz; controller `:161–173,514–532` 전용 스레드 | 100 Hz=10 ms, 200 Hz=5 ms budget을 구분. 문서만 보고 200 Hz를 기준값으로 고정하지 않음 |
| runtime TX 결과가 상위에 전달되지 않음 | `can.py:187–194` 로그만 남김; write 응답 처리 없음 | socket enqueue 성공/실패를 즉시 반환. enqueue 성공을 모터 적용 확인으로 표현하지 않음 |
| runtime 응답 길이 검사 부족 | `can.py:164–170` nonempty만 검사; manager는 8-byte를 가정 | decoder 입구에서 프레임 크기, 표준 ID, RTR/ERR, 명령을 검증 |
| 오래된 cache를 새 응답처럼 볼 가능성 | event clear와 cache freshness는 별개; `motor_driver.py:121–134`는 RX ID를 무조건 우선 | 공유 latest cache 제거. 현재 요청 수집 상태에 bus/ID/cmd로 dispatch하고 중복과 예상 밖 응답을 구분 |
| 오류 상태가 런타임에 갱신되지 않음 | `poll_state1()`은 있지만 controller에서 주기 호출 없음; `poll_error_flags_once()` 구현 없음 | 1 Hz polling이 존재한다고 가정하지 않음. 복원한다면 별도 기능 변경으로 제안 |
| 실패 시 위치는 이전 값, 속도·전류는 NaN | `motor_manager.py:403–408`, controller `:374–379` | C++ snapshot에 motor별 valid/status를 포함하고 상위가 명시적으로 fault latch하도록 연결 |
| 0도와 미수신을 혼용할 수 있음 | `motor_manager.py:313–318`에서 `raw_multi != 0`일 때만 multi 사용 | C++는 수신 유효성과 0도를 분리. 정상 0도 및 None fallback case를 테스트 |
| 초기화 실패 시 자원 정리 경로가 약함 | MotorIO 생성 도중 open/read 실패 가능; pool shutdown 명시 없음 | RAII로 부분 생성 실패에도 열린 socket 정리. 초기화 완료 전 nonzero 출력 금지 |

50 ms 대기는 현재 launch 기본 주기의 5배이다. 그러나 실제 응답 분포를 보지 않고 timeout을 임의로 3 ms로 줄이는 것도 적절하지 않다. `T_read + T_compute + T_write + T_other < T_cycle`로 예산을 먼저 배분한다.

### 6. 권장 C++ 구성과 실행 모델

```text
goat_control_cpp/
├── include/goat_control_cpp/motor/
│   ├── can.h            # 기존 stub 확장: SocketCAN fd RAII, nonblocking TX/RX
│   ├── motor_types.h    # typed config, raw state, joint sample, status
│   ├── motor_protocol.h # frame encode/decode 순수 함수
│   └── motor_io.h       # 기존 stub 교체: 초기화, batch read/write, 위치 변환
├── src/motor/
│   ├── can.cpp
│   ├── motor_protocol.cpp
│   └── motor_io.cpp
└── test/motor/
    ├── test_motor_protocol.cpp
    └── test_motor_state.cpp
```

처음부터 Python의 MotorDriver와 MotorManager를 별도 클래스로 재현하지 않는다. 작은 protocol 함수와 MotorIO 내부의 per-motor 상태로 충분하다. 여러 모터 프로토콜이나 독립 사용자가 실제로 필요해질 때 분리할 수 있다.

**권장 runtime: 단일 소유자 + 여러 소켓 동시 대기**

1. 제어 스레드만 소켓을 읽고 쓴다. 버스별 socket은 한 개씩이며 모터별 socket을 만들지 않는다.
2. tick 시작에 이미 queued된 `0xA1` 등 이전 응답을 bounded drain하고 분류한다. 무제한 drain은 금지한다. backlog가 상한을 넘으면 정상 fresh scan으로 간주하지 않는다.
3. read 시작 시각부터의 절대 deadline을 설정하고, 사전 생성한 `0x9C` frame을 모든 모터에 보낸다. 버스 순서를 고정한 schedule은 초기화 때 만든다.
4. `poll/ppoll`로 전체 bus fd를 함께 기다린다. 한 bus나 motor부터 blocking `recv()`하여 다른 bus를 방치하지 않는다. 한 버스는 프레임 단위로 직렬 통신하지만, 서로 다른 물리 버스는 독립적으로 동작한다.
5. 준비된 fd에서 수신하고 bus/ID/cmd를 확인해 pending motor를 채운다. 모터당 1개만 인정하고 전체 수신 완료 또는 공통 deadline에 반환한다.
6. `0x9C`의 유효한 encoder만 위치에 반영한다. `0xA1`의 뒤늦은 encoder를 섞으면 적분 순서가 뒤집힐 수 있으므로 상태 입력으로 사용하지 않는다.
7. 제어 계산 후 모든 `0xA1`을 deadline 안에 nonblocking 전송한다. `EAGAIN/ENOBUFS` 등을 숨기지 않고 전송 실패를 반환한다. 첫 단계에서는 A1 응답을 기다리지 않는다.

송신을 한 스레드에서 순차 syscall로 제출해도 여러 물리 CAN controller는 독립 송신한다. 모터별 worker가 물리적인 단일 CAN 버스의 병렬성을 늘려주지는 않는다.

**대안과 tradeoff:** Python과 같은 bus별 RX thread도 가능하다. 제어 연산 중 지속 수신이 꼭 필요하면 유리하지만 cache 동기화, 깨우기, shutdown 복잡도가 생긴다. 현재 요청/응답 기반 6축 제어에는 먼저 단일 소유자 안을 측정하고, queue drop이나 수신 지연이 확인될 때 도입하는 편이 간단하다. 이 선택은 실시간 성능 보장이 아니다.

### 7. 최적화의 우선순위와 한계

1. **할당·변환 제거:** motor/config/state/frame/pollfd buffer는 생성 시 크기를 확정한다. tick에서 vector resize, map 삽입, string 생성, JointState/list/NumPy 왕복을 하지 않는다. joint 이름도 초기화 때만 처리한다.
2. **동기화 제거:** 소유자를 하나로 하여 per-frame mutex/event와 RX→control thread handoff를 없앤다. 측정 후 필요할 때만 별도 RX 구조를 추가한다.
3. **변환 상수 사전 계산:** `1/(gear×Kt)`, `deg_to_rad/gear`, encoder scale, CAN IDs, bus schedule을 준비한다. 처음에는 양자화와 경계 반올림 결과가 달라지지 않는지 비교한다.
4. **deadline 통일:** 모든 시간 제한은 monotonic/steady clock을 사용한다. TX 준비·전송 시간을 포함한 read 절대 deadline을 사용하고, 모터마다 timeout을 누적하지 않는다. 이는 기존의 “전송 후부터 timeout”과 다른 계약이므로 변경으로 명시한다.
5. **관측 분리:** hot path는 시간·count만 기록하고 문자열 로그/ROS 메시지화는 상위에서 수행한다. 현재 Apply 로그는 host write 호출 간격이지 실제 모터 토크 적용 시각이 아니다.
6. **추후 실험:** A1 feedback 중심으로 바꾸면 9C 요청을 줄일 수 있지만 이전 tick 토크의 응답으로 다음 tick 상태를 구성하게 된다. 별도 latency·제어 안정성 비교가 필요하다. CPU affinity, SCHED_FIFO, busy-wait는 이번 기본 설계에 포함하지 않는다.

사용자 공간 cache에 sequence를 붙이거나 TX 시각 이후 수신했는지 검사하는 것만으로 **현재 명령에 대한 응답임을 증명할 수는 없다**. 이 프로토콜에는 요청 transaction ID가 없기 때문이다. timeout 후 도착한 이전 9C 응답은 다음 요청과 구별되지 않을 수 있다. timeout은 invalid/fault로 처리하고 같은 명령을 즉시 정상 재시도하여 상태를 되살리지 않는다. 자동 복구가 필요하면 firmware 응답 지연 상한과 재동기화 절차를 별도로 정해야 한다.

SocketCAN에서는 버스별 표준 ID 필터, `CAN_RAW_RECV_OWN_MSGS=0`, 필요한 error-frame 구독을 사용한다. `0x140+id`의 실제 모터 응답과 자기 송신 echo를 구분해야 한다. 이 옵션들의 의미는 [Linux Kernel SocketCAN 문서](https://docs.kernel.org/networking/can.html#raw-protocol-sockets-with-can-filters-sock-raw)를 따른다. 자체 socket의 echo 억제는 다른 프로세스가 같은 ID로 송신하는 문제까지 해결하지 않으므로 실기 제어에서는 명령 송신 주체도 하나로 제한한다.

### 8. Python/C++ 경계

`cpp_migration.md:5–6`의 full hot-path in-process 목표와 맞춰 motor 라이브러리를 만든다. 이것만 빌드한다고 현재 Python controller가 C++ MotorIO를 호출하게 되지는 않는다.

- 이번 slice: C++ library와 자동 테스트, 별도 검증 프로그램에서 동작 확인. 기존 Python controller와 sysid는 유지한다.
- 이후 C++ controller 연결: 같은 프로세스에서 직접 호출하며 topic hop을 추가하지 않는다.
- Python controller에 먼저 붙여야 한다면 binding이 별도 필요하다. `pybind11` 등의 의존성, GIL 해제 구간과 데이터 복사 계약을 추가 검토해야 한다. 현재 단계에서 자동으로 추가하지 않는다.

`src/goat_sysid`의 `friction_id.py`, `actuator_target_test.py`는 Python MotorIO를, `can_id.py`, `current_input_test.py` 등은 Python driver를 직접 사용한다. C++ motor 추가를 이유로 해당 Python API를 제거하지 않는다.

## Code

### 1. 승인 대상 변경 지점

기존 줄 번호는 현재 checkout 기준이다. 신규 파일에는 기존 줄 번호가 없으므로 아래 선언부를 제안 위치로 표시한다. 전체 implementation patch는 설계 검토 후 구체화한다.

| 경로 및 현재 위치 | 제안 변경과 이유 |
|---|---|
| `src/goat_control_cpp/include/goat_control_cpp/motor/can.h:1–19` | stub을 noncopyable fd RAII transport로 교체. 프로토콜·thread/cache는 포함하지 않음 |
| `src/goat_control_cpp/include/goat_control_cpp/motor/motor_types.h` (신규) | config와 raw/joint/status 타입. ROS dependency와 미수신=0 혼용 제거 |
| `src/goat_control_cpp/include/goat_control_cpp/motor/motor_protocol.h` (신규) | 아래와 같은 명령 encoder/decoder 선언. Python byte packing을 독립 테스트 가능하게 함 |
| `src/goat_control_cpp/include/goat_control_cpp/motor/motor_io.h:1–22` | 현재 `int&` scalar torque API를 joint 배열 입력으로 교체. config, 초기화, batch deadline API 도입 |
| `src/goat_control_cpp/src/motor/{can,motor_protocol,motor_io}.cpp` (신규) | 위 transport, protocol, scan/anchor/변환 구현 |
| `src/goat_control_cpp/CMakeLists.txt:69` 앞, `:76` | motor library 추가, export 설치 target에 포함 |
| `src/goat_control_cpp/CMakeLists.txt:94–99`, `package.xml:16` 부근 | 실제 단위 테스트용 ament_cmake_gtest 추가 |
| `src/goat_control_cpp/test/motor/test_{motor_protocol,motor_state}.cpp` (신규) | frame/수치 동등성, unwrap와 invalid 상태 검증 |

첫 motor core에는 `sensor_msgs`나 YAML parser가 필요하지 않다. 향후 controller가 YAML을 typed config로 바꾸고 JointState를 작성한다. 현 stub의 `sensor_msgs` include도 그때 core에서 제거하는 범위에 해당한다.

### 2. `motor_io.h` 공개 API 제안

아래 선언은 `motor_types.h`의 `MotorConfig`, `JointSample`, `MotorFault`를 사용하는 인터페이스 발췌다. 내부 storage 구현은 생략했다. `JointSample`은 q/dq/tau, 수신 시각, valid/fault를 가진다. `MotorConfig`는 각 joint의 bus/node, gear/Kt/limits/offset과 초기화 protocol 종류 및 angle/current/encoder scale을 포함한다.

```cpp
#pragma once

#include <chrono>
#include <vector>

#include "goat_control_cpp/motor/motor_types.h"

namespace goat_control_cpp {

// Own CAN buses and motor state on the calling control thread.
class MotorIO {
 public:
  // Validate and own an immutable copy of the motor configuration.
  explicit MotorIO(const MotorConfig& config);

  // Release all opened CAN sockets without sending motor commands.
  ~MotorIO();

  // Prevent duplicate ownership of transport and encoder state.
  MotorIO(const MotorIO&) = delete;
  // Prevent assignment of transport and encoder ownership.
  MotorIO& operator=(const MotorIO&) = delete;

  // Read startup state and establish absolute-angle encoder anchors.
  void initialize();

  // Collect one 0x9C scan within an absolute monotonic deadline.
  MotorFault read_joint_state(
      std::chrono::steady_clock::time_point deadline, // Whole-scan deadline.
      std::vector<JointSample>& output); // Caller-preallocated joint samples.

  // Queue clipped joint torques as 0xA1 frames without awaiting replies.
  MotorFault write_motor(
      const std::vector<double>& torque_nm, // Joint-ordered torque commands.
      std::chrono::steady_clock::time_point deadline); // Whole-write deadline.

  // Close transports after the caller has stopped the control loop.
  void close() noexcept;
};

}  // namespace goat_control_cpp
```

입력/output 길이는 초기화 때 확정한 joint 수와 일치해야 하며, tick 안에서 output을 확장하지 않는다. `MotorFault`는 최소한 timeout, TX 실패, bus 이상, invalid command를 나타내고 output sample에는 motor별 valid를 남긴다. nonfinite torque는 정수 변환 전에 거부하고 상위 fault 처리로 전달한다. `initialize()` 실패는 시작 단계의 오류로, runtime 통신 실패는 명시적인 status로 처리한다.

종료 시 제로 토크는 상위에서 제어 스레드를 멈춘 뒤 명시적으로 보내고 결과를 처리한다. 소멸자의 암묵적 통신에 의존하지 않는다. BUS-OFF에서는 제로 명령도 전달할 수 없으므로, host 전송 성공을 모터 정지 확인으로 간주하지 않는다. 모터 측 watchdog 유무는 확인되지 않았다.

### 3. `motor_protocol.h` 함수 형태

```cpp
// Encode a validated, quantized current command into an 8-byte 0xA1 frame.
can_frame make_torque_frame(
    canid_t tx_id, // Precomputed standard request identifier.
    std::int16_t current_lsb); // Signed current after configured clipping.

// Decode a validated standard 0x9C response without changing persistent state.
bool decode_state2(
    const can_frame& frame, // Incoming Classical CAN frame.
    RawMotorState& output) noexcept; // Decoded values in protocol units.
```

초기화에 필요한 9A/94/92/PI encoder/decoder도 같은 파일에 둔다. bit packing과 물리 단위 변환을 분리하고, 포인터 cast를 통한 unaligned/signed 읽기를 피한다. `motor_types.h`, `<linux/can.h>`, `<cstdint>`를 include한다.

### 4. CMake/package 추가안

기존 IMU/serial target은 유지한다. motor core는 ROS 비의존이므로 이 slice에서는 추가 `ament_target_dependencies`가 필요하지 않다.

```cmake
# Build the ROS-independent motor transport and state library.
add_library(motor_io
  src/motor/can.cpp
  src/motor/motor_protocol.cpp
  src/motor/motor_io.cpp
)
target_include_directories(motor_io PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
```

`CMakeLists.txt:76`의 기존 행을 다음 형태로 바꾼다. 기존 export block을 재사용한다.

```cmake
  TARGETS serial imu_io motor_io
```

기존 `if(BUILD_TESTING)` 안에 추가할 테스트 등록:

```cmake
  # Register protocol byte-compatibility tests.
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_motor_protocol test/motor/test_motor_protocol.cpp)
  target_link_libraries(test_motor_protocol motor_io)

  # Register angle integration and invalid-sample tests.
  ament_add_gtest(test_motor_state test/motor/test_motor_state.cpp)
  target_link_libraries(test_motor_state motor_io)
```

`package.xml:16` 부근에 추가할 의존성:

```xml
  <test_depend>ament_cmake_gtest</test_depend>
```

## 검증·완료 조건

1. **Protocol 동등성:** Python의 순수 packing/변환 부분에서 golden vector를 만들고 양·음수 current, clip 경계, 반올림 동률, signed 온도/속도, 56-bit 음수 각도, 각 PI 형식을 비교한다. Python MotorIO 생성이나 hardware 실행은 필요 없다.
2. **상태 동등성:** encoder 정·역방향 wrap, 0도, single-turn fallback, offset/gear, mapping 순서 변경, boot fold 음수 각도를 확인한다. 누락·중복 응답에는 다시 적분하지 않는다.
3. **모의 통신:** vcan에서 여러 bus, 두 응답 ID, 역순·지연·누락·잘못된 길이·이전 A1 응답 혼재를 주입한다. read가 공통 deadline에 종료되고 TX 실패가 상위에 전달되는지 확인한다. vcan은 물리 bitrate, arbitration 지연, 실기 BUS-OFF를 재현하지 않는다.
4. **초기화·종료:** 모든 설정 검사, 도중 open 실패, anchor 취득 실패, 반복 close에서 자원 누수를 확인한다. joint 수/topology는 실배선과 대조한 설정으로 검증한다.
5. **실기 성능:** 같은 장치·설정·주파수에서 Python과 C++를 비교한다. read TX/대기, compute, write enqueue, tick jitter의 p50/p95/p99/max, deadline miss, 응답 누락, queue drop, CPU 사용률을 기록한다. 100 Hz/200 Hz를 구분하고 목표 cycle 예산 안에 들어오는지 평가한다.
6. **build:** 구현 후 `colcon build --packages-up-to goat_control_cpp --cmake-args -DCMAKE_BUILD_TYPE=Release`, `colcon test --packages-select goat_control_cpp`, `colcon test-result --verbose`를 실행한다. 성능 측정 build에 수치 안전 판정을 훼손하는 `-ffast-math`는 추가하지 않는다.

이번에 수행한 검증은 소스·참조 위치·변경 범위 확인이다. C++ 구현, compile, vcan, 실기 CAN 접근 및 성능 측정은 수행하지 않았다.
