#!/usr/bin/env python3
import argparse
import time
import can

def open_bus(channel="can0"):
    return can.interface.Bus(channel=channel, interface="socketcan")

def txrx(bus, tx_id, rx_id, data0, payload7=b"\x00"*7, timeout=0.2,
         tx_only=True, verbose=False):
    """요청을 보내고 응답을 기다림.
       - tx_only=True: TX-ID(0x140+ID)에서 '내용이 다른' 프레임만 응답으로 인정 (에코는 무시)
       - tx_only=False: 위 + RX-ID(0x180+ID)도 정상 응답으로 인정
    """
    frame = bytes([data0]) + payload7
    bus.send(can.Message(arbitration_id=tx_id, data=frame, is_extended_id=False))
    t_end = time.time() + timeout
    while time.time() < t_end:
        m = bus.recv(timeout=0.05)
        if not m or len(m.data) != 8 or m.data[0] != data0:
            continue

        # RX-ID 허용 여부
        if not tx_only and m.arbitration_id == rx_id:
            if verbose:
                print(f"[RX] 0x{m.arbitration_id:X} <- 0x{data0:02X} {m.data.hex(' ').upper()}")
            return m

        # TX-ID에서 오되, 내가 보낸 프레임(에코)와 내용이 달라야 응답으로 인정
        if m.arbitration_id == tx_id and m.data != frame:
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
    val = int.from_bytes(rep.data[4:8], byteorder='little', signed=False)
    return val / 100.0, rep.arbitration_id  # deg, src_id

def read_multi_turn_deg(bus, node_id, **rxopts):
    """0x92: DATA[1..7] = int64 하위 7바이트 (0.01°/LSB) → 부호 확장"""
    tx, rx = 0x140 + node_id, 0x180 + node_id
    rep = txrx(bus, tx, rx, 0x92, **rxopts)
    if not rep:
        return None, None
    raw7 = rep.data[1:8]
    sign = b'\x00' if raw7[-1] < 0x80 else b'\xff'
    val = int.from_bytes(raw7 + sign, byteorder='little', signed=True)
    return val / 100.0, rep.arbitration_id  # deg, src_id

def main():
    ap = argparse.ArgumentParser(description="Read BOTH single-turn(0x94) & multi-turn(0x92) angles")
    ap.add_argument("--channel", default="can0", help="SocketCAN channel")
    ap.add_argument("--node", type=int, required=True, help="Motor ID (1..32)")
    ap.add_argument("--loop", action="store_true", help="Continuously read")
    ap.add_argument("--rate", type=float, default=10.0, help="Loop rate (Hz) if --loop")
    ap.add_argument("--timeout", type=float, default=0.25, help="RX timeout seconds")
    # 기본은 TX-ID만 허용
    ap.add_argument("--accept-rx", action="store_true", help="Also accept replies on RX-ID (0x180+ID)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    bus = open_bus(args.channel)
    period = 1.0/args.rate if args.loop else None

    rxopts = dict(timeout=args.timeout, tx_only=(not args.accept_rx), verbose=args.verbose)

    try:
        while True:
            s_deg, s_id = read_single_turn_deg(bus, args.node, **rxopts)
            m_deg, m_id = read_multi_turn_deg(bus, args.node, **rxopts)

            s_txt = "NO-REPLY" if s_deg is None else f"{s_deg/10:.2f} deg (srcID=0x{s_id:X})"
            m_txt = "NO-REPLY" if m_deg is None else f"{m_deg/10:.2f} deg (srcID=0x{m_id:X})"
            print(f"single={s_txt} | multi={m_txt}")

            if not args.loop:
                break
            time.sleep(period)
    finally:
        bus.shutdown()

if __name__ == "__main__":
    main()
