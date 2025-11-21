#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from motor_interfaces.msg import MotorStates

def fmt_list(xs, fmt='{:.2f}', max_len=None):
    if xs is None:
        return '[]'
    arr = list(xs)
    if max_len is not None:
        arr = arr[:max_len]
    def _fmt(x):
        try:
            return fmt.format(x)
        except Exception:
            return str(x)
    return '[' + ', '.join(_fmt(x) for x in arr) + (' ...]' if max_len and len(xs) > max_len else ']')

class MotorStatesEcho(Node):
    def __init__(self):
        super().__init__('motor_states_echo')
        # 표시할 모터 개수(앞에서부터), None이면 전부 출력
        self.max_motors = self.declare_parameter('max_motors', 8).value
        # 단일/멀티턴의 단위가 0.01도라면 degree로 환산해서 같이 보여줄지 여부
        self.show_deg = self.declare_parameter('show_degrees', True).value

        self.sub = self.create_subscription(MotorStates, 'motor_states', self.cb, 10)
        self.get_logger().info('Subscribed to /motor_states')

    def cb(self, msg: MotorStates):
        n = self.max_motors if self.max_motors and self.max_motors > 0 else None

        # 기본 값들
        temps = msg.temperature_c
        iq = msg.iq_amp
        spd = msg.speed_dps
        enc = msg.encoder_raw
        st  = msg.single_turn_raw
        mt  = msg.multi_turn_raw
        err = msg.error_flags
        mst = msg.motor_state

        stamp = msg.header.stamp
        ts = f'{stamp.sec}.{str(stamp.nanosec).zfill(9)}'

        # 단일/멀티턴 각도(0.001° 단위 가정) -> ° 환산 표시 (선택)
        st_deg = [x * 0.001 for x in st] if self.show_deg else None
        mt_deg = [x * 0.001 for x in mt] if self.show_deg else None

        lines = []
        lines.append(f'\n=== /motor_states @ {ts} ===')
        lines.append(f'  temperature_c : {fmt_list(temps, "{:.1f}", n)}  (°C)')
        lines.append(f'  iq_amp        : {fmt_list(iq, "{:.3f}", n)}  (A)')
        lines.append(f'  speed_dps     : {fmt_list(spd, "{:.1f}", n)}  (dps)')
        lines.append(f'  encoder_raw   : {fmt_list(enc, "{}", n)}')

        if self.show_deg:
            lines.append(f'  single_turn   : {fmt_list(st, "{}", n)}  (raw)')
            lines.append(f'                  {fmt_list(st_deg, "{:.2f}", n)} (deg)')
            lines.append(f'  multi_turn    : {fmt_list(mt, "{}", n)}  (raw)')
            lines.append(f'                  {fmt_list(mt_deg, "{:.2f}", n)} (deg)')
        else:
            lines.append(f'  single_turn   : {fmt_list(st, "{}", n)}  (raw)')
            lines.append(f'  multi_turn    : {fmt_list(mt, "{}", n)}  (raw)')

        lines.append(f'  error_flags   : {fmt_list(err, "{}", n)}')
        lines.append(f'  motor_state   : {fmt_list(mst, "{}", n)}')

        self.get_logger().info('\n'.join(lines))

def main(args=None):
    rclpy.init(args=args)
    node = MotorStatesEcho()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
