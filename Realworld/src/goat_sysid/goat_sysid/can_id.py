from dataclasses import dataclass, asdict
from typing import Optional
from goat_control.utils.motor import CanInterface, MotorDriver
import struct, time, yaml
import argparse


@dataclass
class MotorTelemetry:
    """One full telemetry snapshot from a single motor."""
 
    node_id: int
 
    # --- torque / current ---------------------------------------------------
    iq_raw: Optional[int] = None            # 0x9C DATA[2:4], int16
    iq_amp: Optional[float] = None

    # --- motion -------------------------------------------------------------
    speed_raw: Optional[int] = None         # 0x9C DATA[4:6], int16
    speed_dps: Optional[float] = None
    encoder: Optional[int] = None           # 0x9C DATA[6:8], uint16 (wraps)
    encoder_raw: Optional[int] = None       # 0x90 DATA[4:6]
    multi_turn_raw: Optional[int] = None    # 0x92 DATA[1:8], int64 (7 bytes)
    multi_turn_deg: Optional[float] = None
    single_turn_raw: Optional[int] = None   # 0x94 DATA[4:8], uint32
    single_turn_deg: Optional[float] = None
 
    # --- power --------------------------------------------------------------
    bus_current_raw: Optional[int] = None   # 0x9A DATA[4:6], int16
    bus_current_a: Optional[float] = None
 
    # --- which commands actually answered -----------------------------------
    replied: tuple = ()
 
    def as_dict(self) -> dict:
        return asdict(self)

# ---------------------------------------------------------------------------
# Per-command parsers. Each takes the 8-byte payload and mutates the snapshot.
# ---------------------------------------------------------------------------
 
def _parse_state1(data: bytes, tel: MotorTelemetry, cfg: dict) -> None:
    """0x9A -- temperature, bus voltage, bus current, state, error flags."""
    # tel.temperature_c = struct.unpack("<b", data[1:2])[0]
    # tel.bus_voltage_raw = struct.unpack("<h", data[2:4])[0]
    tel.bus_current_raw = struct.unpack("<h", data[4:6])[0]
 
 
def _parse_state2(data: bytes, tel: MotorTelemetry, cfg: dict) -> None:
    """0x9C -- temperature, torque current iq, speed, encoder position."""
    # tel.temperature_c = struct.unpack("<b", data[1:2])[0]
    tel.iq_raw = struct.unpack("<h", data[2:4])[0]
    tel.speed_raw = struct.unpack("<h", data[4:6])[0]
    tel.encoder = struct.unpack("<H", data[6:8])[0]
 
    tel.iq_amp = tel.iq_raw * cfg["motor_current_amp_per_lsb"]
    tel.speed_dps = tel.speed_raw * cfg["speed_deg_per_sec_per_lsb"]
 
 
def _parse_multi_turn(data: bytes, tel: MotorTelemetry, cfg: dict) -> None:
    """0x92 -- accumulated multi-turn angle. int64 packed into 7 bytes."""
    raw7 = data[1:8]
    sign_byte = b"\x00" if raw7[-1] < 0x80 else b"\xff"
    tel.multi_turn_raw = int.from_bytes(raw7 + sign_byte, "little", signed=True)
    tel.multi_turn_deg = tel.multi_turn_raw * cfg["angle_deg_per_lsb"]
 
 
def _parse_single_turn(data: bytes, tel: MotorTelemetry, cfg: dict) -> None:
    """0x94 -- single-turn angle, wraps to 0 at the encoder zero point."""
    tel.single_turn_raw = struct.unpack("<I", data[4:8])[0]
    tel.single_turn_deg = tel.single_turn_raw * cfg["angle_deg_per_lsb"]
 
 
def _parse_encoder(data: bytes, tel: MotorTelemetry, cfg: dict) -> None:
    """0x90 -- encoder value, raw value, and the stored zero offset."""
    tel.encoder_raw = struct.unpack("<H", data[4:6])[0]
    # tel.encoder_zeroed = struct.unpack("<H", data[2:4])[0]
    # tel.encoder_offset = struct.unpack("<H", data[6:8])[0]

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


_COMMANDS = (
    (0x9A, _parse_state1, "state1"),
    (0x90, _parse_encoder, "encoder"),
    (0x94, _parse_single_turn, "single_turn"),
    (0x92, _parse_multi_turn, "multi_turn"),
    (0x9C, _parse_state2, "state2"),)

E7 = b"\x00" * 7

 
def read_all(can_interface,
             motor_driver,
             cfg,
             timeout: float = 0.2,
             skip: tuple = ()) -> MotorTelemetry:
    """Poll every telemetry command on one motor and return a full snapshot.
 
    Args:
        can_interface: an open CanInterface (reader thread must NOT be running)
        motor_driver:  MotorDriver bound to that interface
        timeout:       per-command response wait, seconds
        skip:          command bytes to skip, e.g. (0x9D, 0x90) for a fast pass
 
    Fields whose command did not reply stay None. `tel.replied` lists the
    command names that did answer, so a silent motor is distinguishable from
    a zero reading.
    """
    tel = MotorTelemetry(node_id=motor_driver.node_id)
    replied = []
 
    for cmd_byte, parser, name in _COMMANDS:
        if cmd_byte in skip:
            continue
 
        msg = can_interface.txrx(
                                tx_id=motor_driver.can_ids.tx_id,
                                rx_id=motor_driver.can_ids.rx_id,
                                cmd_byte=cmd_byte,
                                payload7=E7,
                                timeout=timeout,
                                accept_rx_id=True,
                                accept_tx_echo_diff=True,
                            )
        if msg is None:
            continue
        if len(msg.data) != 8:
            continue
 
        parser(bytes(msg.data), tel, cfg)
        replied.append(name)
 
    tel.replied = tuple(replied)
    return tel


def format_telemetry(tel: MotorTelemetry) -> str:
    """One-block human-readable dump, raw and scaled side by side."""
    if not tel.replied:
        return f"id{tel.node_id}: NO REPLY to any command"
 
    def line(label, raw, scaled, unit):
        raw_s = "----" if raw is None else f"{raw:>10d}"
        sc_s = "----" if scaled is None else f"{scaled:>12.4f}"
        return f"  {label:<18s} raw={raw_s}  ->{sc_s} {unit}"
 
    out = [f"id{tel.node_id}"]
    out.append(line("iq", tel.iq_raw, tel.iq_amp, "A"))
    out.append(line("speed", tel.speed_raw, tel.speed_dps, "dps"))
    out.append(line("multi_turn", tel.multi_turn_raw, tel.multi_turn_deg, "deg"))
    out.append(line("single_turn", tel.single_turn_raw, tel.single_turn_deg, "deg"))
    out.append(line("bus_current", tel.bus_current_raw, tel.bus_current_a, "A"))
    # out.append(line("bus_voltage", tel.bus_voltage_raw, tel.bus_voltage_v, "V"))
    out.append(f"  {'encoder':<18s} pos={tel.encoder} ")
    return "\n".join(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", default="can0", help="CAN channel")
    parser.add_argument("--node-id", type=int, default=1, help="Node ID")
    args = parser.parse_args()
    cfg = yaml.safe_load(open("src/goat_control/config/goat_config.yaml", encoding="utf-8"))

    can = CanInterface(channel=args.channel, interface="socketcan"); can.open()
    driver = MotorDriver(can, node_id=args.node_id)
    
    t0 = time.perf_counter()
    n = 0
    try:
        while True:
            tel = read_all(can, driver, cfg)
            t = time.perf_counter() - t0
            print(f"[t={t:7.3f}] {format_telemetry(tel)}\n", flush=True)
            n += 1
    except KeyboardInterrupt:
        pass
    finally:
        can.close()