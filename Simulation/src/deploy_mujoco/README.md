# deploy_mujoco

GOAT의 MuJoCo closed-loop 시뮬레이션.
[`unitree_rl_gym/deploy/deploy_mujoco`](https://github.com/unitreerobotics/unitree_rl_gym/tree/main/deploy/deploy_mujoco) 방식.

한 프로세스, 한 루프, ROS 없음:

```
observe(qpos/qvel/sensordata) -> policy(ONNX) -> PD+P torque -> safety -> mj_step x N
```

executor도 timer도 topic도 `/clock`도 없다. 제어 tick의 `dt` 는 항상 정확히 `control_dt` 이고,
한 tick이 정확히 `n_substeps` 번의 물리 스텝을 진행한다.

## 설치

```bash
pip install mujoco onnxruntime numpy pyyaml
```

## 실행

```bash
python3 deploy_mujoco.py                                       # 지그 위 (기본)
python3 deploy_mujoco.py --model configs/xml/goat_floating.xml # 평지 free base
```

정책은 **기본으로 돌아간다** (`--stopped` 로 정지 상태 시작).

키는 **viewer 창**에서 받는다. 터미널이 아니다 — 터미널에 타이핑하면 아무 일도 안 일어난다.
MuJoCo viewer가 space와 방향키를 자체 UI에 쓰고 있어 삼켜질 수 있으므로 모든 동작에 문자 별칭이 있다.

| 키 | 동작 |
|---|---|
| `space` / `P` | 정책 시작 / 정지 |
| `R` / `X` | 리셋 (`home` 키프레임, 실행 상태는 유지) |
| `Q` | 종료 (`esc` 는 viewer 자체가 처리) |
| `↑` `↓` / `W` `S` | `v_x` +/- |
| `←` `→` / `A` `D` | `w_z` -/+ |
| `0` / `Z` | 커맨드 0으로 |

정지 중에는 토크가 0이고 **safety 검사도 건너뛴다**. 안 그러면 다리가 중력으로 주저앉다가
위치 한계를 넘어 estop이 latch되고, 이후 space를 눌러도 토크가 계속 0이 된다.

인자: `--model`, `--config`, `--keyframe`, `--duration`, `--stopped`, `--headless`, `--realtime`/`--no-realtime`.
`--realtime` 은 viewer 있으면 기본 on, `--headless` 면 기본 off.
주기와 체크포인트는 CLI가 아니라 `configs/goat.yaml` 이 정한다.

> viewer로 띄운 실행은 종료 시 WSLg/glfw teardown에서 segfault(exit 139)가 난다.
> 작업은 전부 끝난 뒤에 나는 것이라 결과에는 영향이 없지만, 종료 코드를 보는 스크립트라면
> `--headless` 로 돌려라 (headless는 정상 0으로 끝난다).

## 주기

학습 환경과 맞춘다. `configs/goat.yaml`:

```yaml
timestep:   0.005   # 물리 200 Hz. MJCF의 <option timestep>을 덮어쓴다 (yaml이 유일한 원천)
control_dt: 0.005   # 제어 200 Hz -> n_substeps = 1 (매 물리 스텝마다 컨트롤러)
policy:
  decimation: 2     # 정책 100 Hz
```

즉 **물리 200 Hz / 제어 200 Hz / 정책 100 Hz**. 학습이 physics 200 Hz에 매 스텝 컨트롤러,
decimation 2였으므로 그대로 재현된다. `control_dt` 가 `timestep` 의 정수배가 아니면 시작 시 거부한다.

물리 dt를 학습보다 잘게 쪼개면(예: 1 kHz + n_substeps 5) 적분은 정확해지지만 정책이 학습 때와
다른 물리를 보게 된다. 정책 거동 재현이 목적이면 학습 dt를 그대로 쓰는 쪽이 맞다.

## 모델

| 파일 | 용도 | free base |
|---|---|---|
| `configs/xml/goat_floating.xml` | 평지 위 free base. 로봇 본체 정의 + `home` 키프레임 | ✓ |
| `configs/xml/goat_on_stand.xml` | 위를 include + 지그. 키프레임은 include된 것을 씀 | ✓ |
| `configs/xml/goat_fixed.xml` | base 고정 지그 실험용 | ✗ |

`home` 키프레임은 `goat_floating.xml` **한 곳에만** 정의한다 (MuJoCo가 동일 이름 key 중복을 거부).
base z=0.465 는 thigh ±0.9943 / knee ±1.884 자세에서 바퀴(r=0.07275)가 지면 2 mm 위에 오는 높이다.

액추에이터는 전부 순수 `<motor>`(토크) 여야 한다. 정책이 토크를 내보내고 PD를 직접 하므로
`<position>`/`<velocity>` 액추에이터는 그 숫자를 각도/속도로 재해석해 버린다. 시작 시 검사해서 막는다.
