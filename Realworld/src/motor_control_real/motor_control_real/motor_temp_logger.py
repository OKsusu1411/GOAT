#!/usr/bin/env python3
'''
python3 motor_temp_logger.py --ros-args -p include_indices:="[0,3,7]"
# 또는 문자열로
python3 motor_temp_logger.py --ros-args -p include_indices:="0,3,7"
'''

import os
import csv
from datetime import datetime
from typing import Iterable, List, Optional
import rclpy
from rclpy.node import Node
from motor_interfaces.msg import MotorStates
# 맨 위 import 추가
from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor

def _parse_indices(val) -> Optional[List[int]]:
    """ROS2 파라미터 값이 리스트든 문자열이든 정수 인덱스 리스트로 파싱."""
    if val is None:
        return None
    # rclpy가 이미 list로 주는 경우
    if isinstance(val, (list, tuple)):
        out = []
        for x in val:
            try:
                i = int(x)
                if i < 0:  # 음수 제외
                    continue
                if i not in out:
                    out.append(i)
            except Exception:
                continue
        return out if out else None
    # "0,2,5" 같은 문자열
    if isinstance(val, str):
        parts = [p.strip() for p in val.split(',') if p.strip() != '']
        out = []
        for p in parts:
            try:
                i = int(p)
                if i < 0:
                    continue
                if i not in out:
                    out.append(i)
            except Exception:
                continue
        return out if out else None
    return None

class MotorTempLogger(Node):
    def __init__(self):
        super().__init__('motor_temp_logger')

        # ====== 파라미터 ======
        self.output_path = self.declare_parameter('output_path', 'motor_temp_log.csv').value
        self.max_motors  = self.declare_parameter('max_motors', 8).value  # include_indices가 있으면 무시
        self.append_mode = self.declare_parameter('append', True).value
        self.flush_every = self.declare_parameter('flush_every', 1).value
        param_desc = ParameterDescriptor(dynamic_typing=True)
        self.declare_parameter('include_indices', '', param_desc)
        raw_inc = self.get_parameter('include_indices').value  # '' | '0,3,7' | [0,3,7]
        self.include_indices = _parse_indices(raw_inc)

        # ====== 파일 준비 ======
        mode = 'a' if self.append_mode else 'w'
        need_header = True
        if os.path.exists(self.output_path) and self.append_mode and os.stat(self.output_path).st_size > 0:
            need_header = False

        self.fh = open(self.output_path, mode, newline='')
        self.writer = csv.writer(self.fh)
        self.rows_written = 0
        self.header_written = not need_header  # 기존 파일에 헤더 있다고 가정
        self.header_indices: Optional[List[int]] = None  # 헤더에 사용한 인덱스(추적용)

        # ====== 구독 ======
        self.sub = self.create_subscription(MotorStates, 'motor_states', self.cb, 50)
        if self.include_indices:
            self.get_logger().info(
                f"Subscribed to /motor_states -> logging indices {self.include_indices} to {self.output_path}"
            )
        else:
            self.get_logger().info(
                f"Subscribed to /motor_states -> logging first {self.max_motors} motors to {self.output_path}"
            )

    def _select_indices(self, total: int) -> List[int]:
        """실제 사용할 인덱스 결정 (include_indices 있으면 그걸, 없으면 0..N-1 슬라이스)."""
        if self.include_indices:
            # 범위 밖 인덱스 필터링
            valid = [i for i in self.include_indices if 0 <= i < total]
            if len(valid) < len(self.include_indices):
                self.get_logger().warn(
                    f"Some indices out of range (total={total}). Using {valid}."
                )
            return valid
        # include_indices 없으면 max_motors 기준
        if self.max_motors and self.max_motors > 0:
            return list(range(min(self.max_motors, total)))
        return list(range(total))

    def _write_header(self, indices: List[int]):
        cols = ['ros_stamp', 'ros_sec', 'ros_nanosec', 'wall_time_iso', 'n_motors']
        # temp_<원본인덱스>_C 로 컬럼 이름 표기
        cols += [f'temp_{i}_C' for i in indices]
        self.writer.writerow(cols)
        self.fh.flush()
        self.header_written = True
        self.header_indices = indices

    def cb(self, msg: MotorStates):
        temps_full = list(msg.temperature_c) if msg.temperature_c is not None else []
        total = len(temps_full)
        indices = self._select_indices(total)

        # 첫 메시지에서 헤더 작성 (또는 append가 아닐 때)
        if not self.header_written:
            self._write_header(indices)

        # append로 열었는데 기존 파일 헤더와 현재 인덱스 구성이 다를 수 있음 → 경고
        if self.header_written and self.header_indices is not None and indices != self.header_indices:
            # 일관성을 위해 경고 1회 출력(너무 시끄럽지 않게)
            if self.rows_written == 0:
                self.get_logger().warn(
                    f"Header indices {self.header_indices} != current indices {indices}. "
                    f"열려 있는 CSV 헤더와 다르면 컬럼 정렬이 어긋날 수 있어요. "
                    f"필요하면 append:=false 로 새 파일을 만드세요."
                )

        # ROS/벽시계 타임스탬프
        sec = msg.header.stamp.sec
        nsec = msg.header.stamp.nanosec
        ros_stamp_str = f"{sec}.{str(nsec).zfill(9)}"
        wall_iso = datetime.now().astimezone().isoformat(timespec='microseconds')

        # 선택한 인덱스 순서로 값 추출
        temps_sel = [temps_full[i] for i in indices]
        row = [ros_stamp_str, sec, nsec, wall_iso, len(temps_sel)] + temps_sel

        self.writer.writerow(row)
        self.rows_written += 1
        if (self.rows_written % max(1, int(self.flush_every))) == 0:
            self.fh.flush()

        if (self.rows_written % 50) == 0:
            self.get_logger().info(f"Logged {self.rows_written} rows (indices={indices})")

    def destroy_node(self):
        try:
            if hasattr(self, 'fh') and self.fh:
                self.fh.flush()
                self.fh.close()
        finally:
            super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = MotorTempLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
