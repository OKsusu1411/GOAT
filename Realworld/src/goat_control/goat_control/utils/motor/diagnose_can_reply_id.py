#!/usr/bin/env python3
"""CAN 응답 주소 진단 (읽기 전용 — 모터 안 움직임).

목적
----
research.md의 "1순위 가설"을 판별한다:
  모터가 답장을 **약속한 주소(rx_id = 0x180+id)** 로 보내는가,
  아니면 **보내는 주소(tx_id = 0x140+id)로 echo** 하는가?

await_state2()는 rx_id Event만 기다리므로, 만약 모터가 tx_id로 답하면
매 틱 timeout(50ms)을 통째로 버린다. 이 스크립트로 실제 응답 주소를 눈으로 확인한다.

무엇을 보내나
------------
각 모터에 **0x9C(state2 읽기)** 명령만 보낸다. 0x9C는 순수 '읽기'라
토크/위치를 바꾸지 않는다 → 로봇이 움직이지 않으므로 아무 때나 안전하게 실행 가능.

준비
----
  1) CAN 인터페이스를 먼저 올려둘 것:  ./setup_can_dual.sh
  2) python-can 필요 (ROS 스택이 쓰는 것과 동일):  pip3 show python-can

실행
----
  python3 diagnose_can_reply_id.py

판독법
------
  "<- arb=0x18X ... [RX-id OK]"        → 정상 주소. 가설1 아님 (다른 원인 조사 필요)
  "<- arb=0x14X ... [TX-id ECHO <--]"  → tx_id echo. **가설1 확정** → Fix 1 적용
  "NO REPLY"                            → 모터 off / 배선 / bitrate 불일치 의심
"""
from __future__ import annotations

import time

try:
    import can
except ImportError:
    raise SystemExit(
        "python-can 가 없습니다. 설치:  pip3 install python-can\n"
        "(Jetson의 ROS 환경에는 이미 깔려 있을 것 — 그 환경에서 실행하세요.)"
    )

# 'can' 모듈이 진짜 python-can 인지, 버전이 무엇인지 먼저 보여준다.
# (로컬에 python-can 이 아닌 다른 'can' 패키지가 잡히면 여기서 바로 드러난다.)
_CAN_VERSION = getattr(can, "__version__", "unknown")
_CAN_FILE = getattr(can, "__file__", "unknown")


def make_bus(channel: str):
    """python-can 3.x / 4.x 양쪽에서 동작하도록 Bus 를 연다.

      - 4.x: can.Bus(interface="socketcan", ...)
      - 3.x: can.interface.Bus(bustype="socketcan", ...)
    어느 버전이 깔려 있든 되는 조합을 순서대로 시도한다.
    """
    import importlib

    # 1) Bus 팩토리 찾기 (4.x: can.Bus, 3.x: can.interface.Bus)
    bus_factory = getattr(can, "Bus", None)
    if bus_factory is None:
        try:
            interface_mod = importlib.import_module("can.interface")
            bus_factory = getattr(interface_mod, "Bus", None)
        except Exception:  # noqa: BLE001
            bus_factory = None
    if bus_factory is None:
        raise RuntimeError(
            "이 'can' 모듈에는 Bus 가 없습니다 (python-can 이 아니거나 손상). "
            f"version={_CAN_VERSION}, file={_CAN_FILE}\n"
            "        → 'pip3 install python-can' 또는 ROS 환경에서 실행하세요."
        )

    # 2) 채널 지정 키워드도 버전마다 다름: interface=(4.x) / bustype=(3.x)
    last_err = None
    for kw in ("interface", "bustype"):
        try:
            return bus_factory(**{kw: "socketcan"}, channel=channel,
                               receive_own_messages=False)
        except TypeError as exc:  # 키워드 불일치 → 다음 조합 시도
            last_err = exc
            continue
    raise RuntimeError(f"Bus 생성 실패 (interface/bustype 모두 거부): {last_err}")


# goat_config.yaml 토폴로지와 동일.
#   motor_node_ids  = [1,2,3,4,5,6,7,8]
#   motor_bus_index = [0,1,0,1,0,1,0,1]  → can0={1,3,5,7}, can1={2,4,6,8}
BUSES: dict[str, list[int]] = {
    "can0": [1, 3, 5, 7],
    "can1": [2, 4, 6, 8],
}

READ_STATE2_CMD = 0x9C       # 순수 읽기 명령 (모터 안 움직임)
REPLY_WAIT_SEC = 0.2         # 모터당 응답 대기 창
TX_BASE = 0x140             # 보내는 주소 = 0x140 + node_id
RX_BASE = 0x180             # 약속된 답장 주소 = 0x180 + node_id


def drain(bus: "can.BusABC") -> None:
    """큐에 남은 오래된 프레임을 비운다 (이전 모터 응답 섞임 방지)."""
    while bus.recv(timeout=0.0) is not None:
        pass


def probe_bus(channel: str, node_ids: list[int]) -> None:
    print(f"\n===== {channel} =====")
    try:
        bus = make_bus(channel)
    except Exception as exc:  # noqa: BLE001
        print(f"  [ERR] {channel} 열기 실패: {exc}")
        print(f"        → './setup_can_dual.sh' 로 인터페이스를 먼저 올렸는지 확인.")
        return

    rx_hits = tx_hits = no_reply = 0
    try:
        for node_id in node_ids:
            tx_id = TX_BASE + node_id
            rx_id = RX_BASE + node_id
            command_frame = bytes([READ_STATE2_CMD]) + b"\x00" * 7

            drain(bus)
            bus.send(can.Message(arbitration_id=tx_id, data=command_frame,
                                 is_extended_id=False))

            replies: list["can.Message"] = []
            deadline = time.time() + REPLY_WAIT_SEC
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                msg = bus.recv(timeout=remaining)
                if msg is None:
                    break
                # 혹시 자기 송신 echo가 섞이면(동일 id + 동일 데이터) 무시.
                if msg.arbitration_id == tx_id and bytes(msg.data) == command_frame:
                    continue
                replies.append(msg)

            print(f"  node {node_id}: 보낸주소=0x{tx_id:03X}  기대답장=0x{rx_id:03X}")
            if not replies:
                no_reply += 1
                print("    <- NO REPLY  (모터 off / 배선 / bitrate 불일치 의심)")
                continue

            for m in replies:
                arb = m.arbitration_id
                cmd = m.data[0] if m.data else -1
                if arb == rx_id:
                    rx_hits += 1
                    tag = "RX-id OK (정상 주소)"
                elif arb == tx_id:
                    tx_hits += 1
                    tag = "TX-id ECHO  <-- 가설1 확정!"
                else:
                    tag = "기타(예상밖 주소)"
                print(f"    <- arb=0x{arb:03X} cmd=0x{cmd:02X} [{tag}] "
                      f"data={bytes(m.data).hex()}")
    finally:
        bus.shutdown()

    print(f"  --- {channel} 요약: RX-id {rx_hits} / TX-id {tx_hits} / 무응답 {no_reply} ---")


def main() -> None:
    print("CAN 응답 주소 진단 (읽기 전용 0x9C — 모터 안 움직임)")
    print(f"  python-can version={_CAN_VERSION}  file={_CAN_FILE}")
    for channel, node_ids in BUSES.items():
        probe_bus(channel, node_ids)
    print(
        "\n[판정]\n"
        "  TX-id ECHO 가 찍히면 → research.md 1순위 확정 → Fix 1(이벤트 공유) 적용.\n"
        "  전부 RX-id OK 면     → 1순위 아님. 리더 스레드/이벤트 set 경로를 다시 조사.\n"
        "  전부 NO REPLY 면     → 통신 자체 문제(전원/배선/종단저항/bitrate) 우선 해결."
    )


if __name__ == "__main__":
    main()
