#!/usr/bin/env python3
import argparse
import time
import can

def open_bus(channel="can0"):
    return can.interface.Bus(channel=channel, interface="socketcan")

def txrx(bus, tx_id, rx_id, data0, payload7=b"\x00"*7, timeout=0.2, accept_tx=True, verbose=False):
    """요청을 보내고 응답을 기다림.
    - 정상 응답: rx_id(0x180+ID)
    - 일부 장치: tx_id(0x140+ID)로 응답 → accept_tx=True면 허용
    - loopback 에코(보낸 것과 데이터 동일)는 무시
    """
    frame = bytes([data0]) + payload7
    bus.send(can.Message(arbitration_id=tx_id, data=frame, is_extended_id=False))
    t_end = time.time() + timeout
    while time.time() < t_end:
        m = bus.recv(timeout=0.05)
        if not m or len(m.data) != 8 or m.data[0] != data0:
            continue
        if m.arbitration_id == rx_id:
            if verbose:
                print(f"[RX] 0x{m.arbitration_id:X} <- 0x{data0:02X} {m.data.hex(' ').upper()}")
            return m
        if accept_tx and m.arbitration_id == tx_id and m.data != frame:
            if verbose:
                print(f"[RX-TXID] 0x{m.arbitration_id:X} <- 0x{data0:02X} {m.data.hex(' ').upper()}")
            return m
    return None

def read_single_turn_deg(bus, node_id, **rxopts):
    """0x94: DATA[4..7] = uint32 (0.01°/LSB)"""
    tx, rx = 0x140 + node_id, 0x180 + node_id
    rep = txrx(bus, tx, rx, 0x94, **rxopts)
    if not rep:
        return None, None
    val_cdeg = int.from_bytes(rep.data[4:8], byteorder='little', signed=False)
    return val_cdeg / 100.0, rep.arbitration_id  # deg, src_id

def read_multi_turn_deg(bus, node_id, **rxopts):
    """0x92: DATA[1..7] = int64의 하위 7바이트 (0.01°/LSB) → 부호 확장"""
    tx, rx = 0x140 + node_id, 0x180 + node_id
    rep = txrx(bus, tx, rx, 0x92, **rxopts)
    if not rep:
        return None, None
    raw7 = rep.data[1:8]  # 7 bytes
    sign = b'\x00' if raw7[-1] < 0x80 else b'\xff'
    val = int.from_bytes(raw7 + sign, byteorder='little', signed=True)
    return val / 100.0, rep.arbitration_id  # deg, src_id

def main():
    ap = argparse.ArgumentParser(description="Read motor angle via CAN (0x94 single-turn, 0x92 multi-turn)")
    ap.add_argument("--channel", default="can0", help="SocketCAN channel")
    ap.add_argument("--node", type=int, required=True, help="Motor ID (1..32)")
    ap.add_argument("--mode", choices=["single","multi","both"], default="both")
    ap.add_argument("--loop", action="store_true", help="Continuously read")
    ap.add_argument("--rate", type=float, default=10.0, help="Loop rate (Hz) if --loop")
    ap.add_argument("--timeout", type=float, default=0.2, help="RX timeout seconds")
    ap.add_argument("--accept-tx", action="store_true", default=True,
                    help="Accept replies that come on TX ID (0x140+ID) as well (default: on)")
    ap.add_argument("--no-accept-tx", action="store_false", dest="accept_tx",
                    help="Do NOT accept replies on TX ID")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    bus = open_bus(args.channel)

    period = 1.0/args.rate if args.loop else None
    rxopts = dict(timeout=args.timeout, accept_tx=args.accept_tx, verbose=args.verbose)

    try:
        while True:
            s_deg = m_deg = None
            s_id = m_id = None

            if args.mode in ("single","both"):
                s_deg, s_id = read_single_turn_deg(bus, args.node, **rxopts)
            if args.mode in ("multi","both"):
                m_deg, m_id = read_multi_turn_deg(bus, args.node, **rxopts)

            if args.mode == "single":
                if s_deg is None:
                    print("single-turn: NO-REPLY")
                else:
                    print(f"single-turn: {s_deg:.2f} deg  (srcID=0x{s_id:X})")
            elif args.mode == "multi":
                if m_deg is None:
                    print("multi-turn:  NO-REPLY")
                else:
                    print(f"multi-turn:  {m_deg:.2f} deg  (srcID=0x{m_id:X})")
            else:
                s_txt = "NO-REPLY" if s_deg is None else f"{s_deg:.2f} deg (srcID=0x{s_id:X})"
                m_txt = "NO-REPLY" if m_deg is None else f"{m_deg:.2f} deg (srcID=0x{m_id:X})"
                print(f"single={s_txt} | multi={m_txt}")

            if not args.loop:
                break
            time.sleep(period)
    finally:
        bus.shutdown()

if __name__ == "__main__":
    main()
