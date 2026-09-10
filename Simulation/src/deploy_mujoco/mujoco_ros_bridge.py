from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

from motor_interfaces.msg import ImuState
from sensor_msgs.msg import JointState

import mujoco
import numpy as np

logger = logging.getLogger(__name__)

#: 센서 이름 규칙. 관절 이름 뒤에 붙는다.
POS_SUFFIX = "_pos"
VEL_SUFFIX = "_vel"
TRQ_SUFFIX = "_torque"

IMU_QUAT = "imu_quat"
IMU_GYRO = "imu_gyro"
IMU_VEL = "imu_vel"


@dataclass(frozen=True)
class BridgeConfig:
    """ MuJoCo initial setting configuration """
    model_path: str
    joint_names: Sequence[str]
    physics_substeps: int = 5
    control_dt: float = 0.005
    keyframe: Optional[str] = "home"


@dataclass
class SimState:
    """ MuJoCo Sensor data """
    qpos: np.ndarray      # (J,) 관절 위치
    qvel: np.ndarray      # (J,) 관절 속도
    torque: np.ndarray    # (J,) jointactuatorfrc
    quat: np.ndarray      # (4,) MuJoCo 순서 w,x,y,z
    gyro: np.ndarray      # (3,)


class MujocoRosBridge:
    def __init__(self, cfg: BridgeConfig) -> None:
        self.cfg = cfg
        self.joint_names = list(cfg.joint_names)
        self.njoint = len(self.joint_names)

        self.model = mujoco.MjModel.from_xml_path(cfg.model_path)
        self.data = mujoco.MjData(self.model)

        # physics_dt는 XML에만 적는다 (CLAUDE.md "하지 말 것"). 여기서는 읽기만.
        self.physics_dt = float(self.model.opt.timestep)
        self.physics_substeps = int(cfg.physics_substeps)
        self.control_dt = self.physics_dt * self.physics_substeps
        if abs(self.control_dt - cfg.control_dt) > 1e-12:
            suggested = cfg.control_dt / self.physics_dt
            raise RuntimeError(
                f"physics_dt({self.physics_dt}) x physics_substeps({self.physics_substeps}) "
                f"= {self.control_dt} != control_dt({cfg.control_dt}). "
                f"XML timestep을 바꿨다면 physics_substeps를 {suggested:g}로 맞춰라 "
                f"(정수여야 한다)."
            )

        self._joint_ids = self._resolve_joints()
        self._actuator_ids = self._resolve_actuators()
        self._sensor_adr = self._resolve_sensors()
        self._fallback_qpos_adr, self._fallback_qvel_adr = self._resolve_fallback_adr()

        self._ctrl = np.zeros(self.model.nu, dtype=np.float64)
        self._state = SimState(
            qpos=np.zeros(self.njoint), qvel=np.zeros(self.njoint),
            torque=np.zeros(self.njoint), quat=np.array([1.0, 0.0, 0.0, 0.0]),
            gyro=np.zeros(3), linvel=np.zeros(3),
        )
        self.step_count = 0
        #: 지연선(delay)이 채워지기까지 필요한 제어 tick 수. 0이면 워밍업 불필요.
        self.warmup_ticks = self._required_warmup_ticks()
        self.reset()

    # ------------------------------------------------------------------ #
    # 이름 해석 (CLAUDE.md 규칙 6: 선언 순서 가정 금지, 실패 시 즉시 예외)
    # ------------------------------------------------------------------ #
    def _resolve_joints(self) -> np.ndarray:
        ids = []
        for name in self.joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"모델에 관절 '{name}'이 없다")
            if self.model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
                raise RuntimeError(f"관절 '{name}'은 hinge가 아니다 (1-DoF만 지원)")
            ids.append(jid)
        return np.asarray(ids, dtype=int)

    def _resolve_actuators(self) -> np.ndarray:
        """관절 id -> 그 관절을 구동하는 액추에이터 id.

        액추에이터 '이름'이 아니라 transmission target(`actuator_trnid`)으로 찾는다.
        이름이 우연히 같기를 기대하지 않고, 토크가 실제로 어느 관절에 들어가는지로
        해석한다. 잘못된 관절에 토크가 들어가면 물리적으로 그럴듯해 보여 발견이 늦다.
        """
        ids = []
        for name, jid in zip(self.joint_names, self._joint_ids):
            found = [
                aid for aid in range(self.model.nu)
                if self.model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT
                and self.model.actuator_trnid[aid, 0] == jid
            ]
            if len(found) != 1:
                raise RuntimeError(
                    f"관절 '{name}'을 구동하는 액추에이터가 {len(found)}개다 (정확히 1개여야 한다)"
                )
            ids.append(found[0])
        return np.asarray(ids, dtype=int)

    def _resolve_sensors(self) -> dict:
        """센서 이름 -> sensordata 주소. 없으면 None (fallback + WARN)."""
        adr = {}
        for key, dim, required in (
            (IMU_QUAT, 4, True), (IMU_GYRO, 3, True), (IMU_VEL, 3, False),
        ):
            adr[key] = self._sensor_slice(key, dim, required)

        for suffix, dim in ((POS_SUFFIX, 1), (VEL_SUFFIX, 1), (TRQ_SUFFIX, 1)):
            for name in self.joint_names:
                adr[name + suffix] = self._sensor_slice(name + suffix, dim, False)
        return adr

    def _sensor_slice(self, name: str, dim: int, required: bool) -> Optional[slice]:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0:
            if required:
                raise RuntimeError(f"필수 센서 '{name}'이 모델에 없다")
            logger.warning(
                "센서 '%s'이 없다. fallback 경로로 읽는다 (raw qpos/qvel 또는 0) — "
                "그 값에는 XML의 delay/noise 설정이 적용되지 않는다.", name,
            )
            return None
        if int(self.model.sensor_dim[sid]) != dim:
            raise RuntimeError(f"센서 '{name}' 차원이 {self.model.sensor_dim[sid]}, 기대값 {dim}")
        start = int(self.model.sensor_adr[sid])
        return slice(start, start + dim)

    def _resolve_fallback_adr(self) -> tuple[np.ndarray, np.ndarray]:
        qpos = np.asarray([self.model.jnt_qposadr[j] for j in self._joint_ids], dtype=int)
        qvel = np.asarray([self.model.jnt_dofadr[j] for j in self._joint_ids], dtype=int)
        return qpos, qvel

    # ------------------------------------------------------------------ #
    # 물리
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        if self.cfg.keyframe:
            kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, self.cfg.keyframe)
            if kid < 0:
                raise RuntimeError(f"keyframe '{self.cfg.keyframe}'이 모델에 없다")
            mujoco.mj_resetDataKeyframe(self.model, self.data, kid)
        self.data.time = 0.0
        self._ctrl[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.step_count = 0
        mujoco.mj_forward(self.model, self.data)  # 첫 발행 전에 sensordata를 채운다

    def _required_warmup_ticks(self) -> int:
        """센서/액추에이터 지연선을 채우는 데 필요한 제어 tick 수.

        `nsample`/`delay`가 붙은 센서는 버퍼가 차기 전까지 **0을 뱉는다.** 리셋 직후
        상태를 그대로 발행하면 컨트롤러의 첫 관측이 "모든 관절이 0"이 되어, 실기에서는
        일어나지 않는 거대한 초기 오차를 정책에 먹인다.
        """
        delays = [float(np.max(self.model.sensor_delay)) if self.model.nsensor else 0.0,
                  float(np.max(self.model.actuator_delay)) if self.model.nu else 0.0]
        max_delay = max(delays)
        if max_delay <= 0.0:
            return 0
        return int(np.ceil(max_delay / self.control_dt))

    def warmup(self) -> None:
        """지연선이 찰 때까지 토크 0으로 돈다. 결정론적(고정 스텝 수)이다.

        `data.time`은 되돌리지 않는다 (지연 버퍼가 시각으로 조회된다). 대신 노드가
        워밍업 종료 시점의 `sim_time`/`step_count`를 원점으로 잡는다.
        """
        self.write_torque(np.zeros(self.njoint))
        for _ in range(self.warmup_ticks):
            self.step()

    def write_torque(self, tau: np.ndarray) -> None:
        """관절 순서 토크를 `data.ctrl`에 쓴다.

        `qfrc_applied`를 쓰지 않는다 (CLAUDE.md 규칙 5). 액추에이터의
        `nsample`/`delay` 쓰기 지연을 우회해버리기 때문이다.
        클리핑도 하지 않는다 — `ctrlrange`/`forcerange`가 MJCF에 이미 있고,
        파이썬에서 또 자르면 한계가 두 곳에 적히게 된다.
        """
        if tau.shape != (self.njoint,):
            raise ValueError(f"tau shape {tau.shape}, 기대값 ({self.njoint},)")
        self._ctrl[self._actuator_ids] = tau
        self.data.ctrl[:] = self._ctrl

    def step(self) -> None:
        for _ in range(self.physics_substeps):
            mujoco.mj_step(self.model, self.data)
        self.step_count += self.physics_substeps

    @property
    def sim_time(self) -> float:
        return float(self.data.time)

    # ------------------------------------------------------------------ #
    # 상태 읽기 — sensordata만 (CLAUDE.md 규칙 4)
    # ------------------------------------------------------------------ #
    def read_state(self) -> SimState:
        sd = self.data.sensordata
        st = self._state

        for i, name in enumerate(self.joint_names):
            sl = self._sensor_adr[name + POS_SUFFIX]
            st.qpos[i] = sd[sl][0] if sl else self.data.qpos[self._fallback_qpos_adr[i]]
            sl = self._sensor_adr[name + VEL_SUFFIX]
            st.qvel[i] = sd[sl][0] if sl else self.data.qvel[self._fallback_qvel_adr[i]]
            sl = self._sensor_adr[name + TRQ_SUFFIX]
            st.torque[i] = sd[sl][0] if sl else 0.0

        st.quat[:] = sd[self._sensor_adr[IMU_QUAT]]
        st.gyro[:] = sd[self._sensor_adr[IMU_GYRO]]
        sl = self._sensor_adr[IMU_VEL]
        st.linvel[:] = sd[sl] if sl else 0.0
        return st


# ---------------------------------------------------------------------- #
# msg 해독 — 여기서만 ROS 메시지 타입을 안다 (지연 import)
# ---------------------------------------------------------------------- #
def seq_to_stamp(seq: int, control_dt: float):
    """seq -> `builtin_interfaces/Time`. **정수 seq에서만 만든다.**

    `data.time`(float 누산)에서 stamp를 만들면 `TimeSynchronizer`의 정확 일치가
    깨진다. `sim_time == seq x control_dt`는 이렇게 해야 정의상 성립한다.
    """
    from builtin_interfaces.msg import Time

    ns = int(round(seq * control_dt * 1e9))
    return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


def stamp_to_seq(stamp, control_dt: float) -> Optional[int]:
    """stamp -> seq. 제어 격자에 안 맞으면 None (호출자가 WARN 후 폐기)."""
    control_ns = int(round(control_dt * 1e9))
    ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    if ns % control_ns != 0:
        return None
    return ns // control_ns


def state_to_msgs(state: SimState, joint_names: Sequence[str], stamp,
                  frame_id: str = "imu_link"):
    """`SimState` -> (`sensor_msgs/JointState`, `motor_interfaces/ImuState`).

    두 메시지의 stamp는 **반드시 동일**해야 한다. 컨트롤러가
    `TimeSynchronizer([joint, imu])`로 정확 일치 stamp를 요구한다.
    """

    js = JointState()
    js.header.stamp = stamp
    js.name = list(joint_names)
    js.position = [float(v) for v in state.qpos]
    js.velocity = [float(v) for v in state.qvel]
    js.effort = [float(v) for v in state.torque]

    imu = ImuState()
    imu.header.stamp = stamp
    imu.header.frame_id = frame_id
    imu.quat.w, imu.quat.x, imu.quat.y, imu.quat.z = (float(v) for v in state.quat)
    imu.gyro.x, imu.gyro.y, imu.gyro.z = (float(v) for v in state.gyro)
    imu.vel.x, imu.vel.y, imu.vel.z = (float(v) for v in state.linvel)
    # mag / time_ms는 정책 관측에 쓰이지 않는다. 0으로 둔다.
    return js, imu


def msg_to_torque(msg, joint_names: Sequence[str], out: np.ndarray) -> Optional[str]:
    """`sensor_msgs/JointState` 명령 -> 관절 순서 토크 벡터 (`out`에 in-place).

    `effort`만 읽는다. `position`/`velocity`(q_ref/v_ref)는 무시한다 —
    시뮬 안에 PD를 넣지 않는다 (CLAUDE.md).

    반환값: 폐기 사유 문자열, 정상이면 None. (`out`은 폐기 시 손대지 않는다)
    """
    effort = msg.effort
    names = msg.name
    if not names:
        return "명령에 name이 없다 (관절 순서 가정 금지)"
    if len(effort) != len(names):
        return f"name {len(names)}개 vs effort {len(effort)}개"

    index = {n: i for i, n in enumerate(joint_names)}
    staged = np.zeros(len(joint_names), dtype=np.float64)
    seen = 0
    for i, name in enumerate(names):
        j = index.get(name)
        if j is None:
            return f"모르는 관절 '{name}'"
        staged[j] = float(effort[i])
        seen += 1
    if seen != len(joint_names):
        return f"관절 {len(joint_names)}개 중 {seen}개만 왔다"

    out[:] = staged
    return None
