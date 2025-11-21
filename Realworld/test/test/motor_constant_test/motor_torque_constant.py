import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
#python3 motor_constant_sampler.py --index 3 --arm 0.252 --settle 2.00 --out motor_constant_measurements.csv

import argparse
import threading
import time
import csv
from datetime import datetime
from pathlib import Path

G = 9.81  # m/s^2, 중력가속도

def frange(start, stop, step):
    # 부동소수 오차 없이 포함형 범위 생성
    n = int(round((stop - start) / step)) + 1
    return [round(start + i * step, 6) for i in range(n)]

class TorqueTestPublisher(Node):
    def __init__(self,
                 topic: str,
                 index: int,
                 arm_m: float,
                 i_min: float,
                 i_max: float,
                 i_step: float,
                 reps: int,
                 settle_s: float,
                 outfile: Path):
        super().__init__('motor_constant_sampler')

        # 퍼블리셔
        self.publisher = self.create_publisher(Float32MultiArray, topic, 10)

        # 측정/출력 설정
        self.target_index = index
        self.arm_m = arm_m
        self.i_min = i_min
        self.i_max = i_max
        self.i_step = i_step
        self.reps = reps
        self.settle_s = settle_s
        self.outfile = outfile

        # 명령 벡터(8채널 가정)
        self.command = [0.0] * 8

        # 10 Hz 타이머로 지속 퍼블리시
        self.timer = self.create_timer(0.1, self.timer_callback)

        # CSV 준비 (헤더 없으면 생성)
        self._prepare_csv()

        # 측정 루프를 별도 스레드에서 시작
        self.worker = threading.Thread(target=self.measure_loop, daemon=True)
        self.worker.start()

    # --- 퍼블리시 타이머 ---
    def timer_callback(self):
        msg = Float32MultiArray()
        msg.data = self.command
        self.publisher.publish(msg)

    # --- CSV 헤더 준비 ---
    def _prepare_csv(self):
        if not self.outfile.parent.exists():
            self.outfile.parent.mkdir(parents=True, exist_ok=True)
        if not self.outfile.exists():
            with self.outfile.open('w', newline='') as f:
                w = csv.writer(f)
                w.writerow([
                    'timestamp',
                    'rep', 'index',
                    'current_A',
                    'mass_input_unit',
                    'mass_value',
                    'mass_kg',
                    'arm_m',
                    'torque_Nm'
                ])

    # --- 한 스텝에서 특정 인덱스만 전류 설정 ---
    def set_current_only_at_index(self, amps: float):
        # 다른 채널은 0 유지, 목표 인덱스만 전류 인가
        for i in range(len(self.command)):
            self.command[i] = 0.0
        self.command[self.target_index] = amps

    # --- 사용자 입력 파서: g 또는 kg 모두 허용 ---
    @staticmethod
    def ask_mass_kg():
        while True:
            raw = input("  ▶ 무게를 입력하세요 (예: 500g 또는 0.5kg) / 종료:q : ").strip().lower()
            if raw in ('q', 'quit', 'exit'):
                return None
            try:
                # 단위 파싱
                if raw.endswith('kg'):
                    val = float(raw[:-2])
                    return val
                elif raw.endswith('g'):
                    val = float(raw[:-1]) / 1000.0
                    return val
                else:
                    # 단위 없으면 g로 간주
                    val = float(raw)
                    # 10 이상이면 보통 g로 해석, 10 미만이면 kg일 가능성 높음 -> g로 가정 안내
                    if val >= 10.0:
                        return val / 1000.0
                    else:
                        # 10 미만은 kg로 입력했다고 간주
                        return val
            except Exception:
                print("  ※ 입력 형식 오류. 예: 500g, 0.5kg, 500")

    # --- 측정 루프 본체 ---
    def measure_loop(self):
        print("\n[측정 시작]\n"
              f"- 제어 토픽: '{self.publisher.topic_name}'\n"
              f"- 측정 채널 index: {self.target_index}\n"
              f"- 지레팔 길이(arm): {self.arm_m:.3f} m\n"
              f"- 전류 스윕: {self.i_max} → {self.i_min} A (step {self.i_step})\n"
              f"- 반복 측정: {self.reps}회\n"
              f"- 스텝 안정화 시간: {self.settle_s:.2f} s\n"
              f"- 저장 파일: {self.outfile}\n")

        # 0 → 음의 방향으로 내려가는 스윕 배열 구성 (요구사항: 0 ~ -7.0)
        # i_max는 보통 0.0, i_min은 음수(예:-7.0), i_step은 음수(예:-0.2)
        if not (self.i_step < 0 < (self.i_max - self.i_min)):
            print("※ 전류 스윕 파라미터를 확인하세요. (예: --i-max 0 --i-min -7.0 --i-step -0.2)")
            return

        sweep = [0.0] + frange(self.i_step, self.i_min, self.i_step)  # 0 다음에 -0.2, -0.4, ..., -7.0

        try:
            with self.outfile.open('a', newline='') as f:
                w = csv.writer(f)

                for rep in range(1, self.reps + 1):
                    print(f"\n==== 반복 {rep}/{self.reps} 시작 ====")

                    for amps in sweep:
                        print(f"\n[전류 설정] index {self.target_index} ← {amps:.3f} A")
                        self.set_current_only_at_index(amps)
                        time.sleep(self.settle_s)  # 안정화 대기

                        m_kg = self.ask_mass_kg()
                        if m_kg is None:
                            print("\n측정을 사용자에 의해 중단했습니다.")
                            self._safe_stop()
                            return

                        torque = m_kg * G * self.arm_m  # τ = m g r

                        # 저장
                        ts = datetime.now().isoformat(timespec='seconds')
                        mass_unit = 'kg'  # 입력을 kg로 통일 저장
                        w.writerow([ts, rep, self.target_index, amps, mass_unit, m_kg, m_kg, self.arm_m, torque])
                        f.flush()

                        print(f"  → 입력 무게: {m_kg:.4f} kg  |  토크: {torque:.4f} N·m  (CSV 저장)")

                    # 한 라운드 끝나면 0 A로 되돌려 다음 반복 준비
                    print("\n  → 다음 반복을 위해 전류 0 A로 복귀")
                    self.set_current_only_at_index(0.0)
                    time.sleep(self.settle_s)

            print("\n[측정 완료] 모든 반복이 끝났습니다. 전류 0 A로 정지합니다.")
            self._safe_stop()

        except KeyboardInterrupt:
            print("\n사용자 중단(Ctrl+C). 전류 0 A로 정지합니다.")
            self._safe_stop()

    def _safe_stop(self):
        self.set_current_only_at_index(0.0)
        # 잠깐 퍼블리시 유지
        time.sleep(0.3)

def main():
    parser = argparse.ArgumentParser(description="전류 스윕 기반 모터 상수 측정 도우미")
    parser.add_argument('--topic', type=str, default='torque_commands',
                        help="퍼블리시할 토픽 이름 (기본: torque_commands)")
    parser.add_argument('--index', type=int, default=3,
                        help="측정할 채널 인덱스 (0~7)")
    parser.add_argument('--arm', type=float, default=0.252,
                        help="지레팔 길이(레버암) [m] (기본: 0.252)")
    parser.add_argument('--i-max', type=float, default=0.0,
                        help="스윕 시작 전류 [A] (기본: 0.0)")
    parser.add_argument('--i-min', type=float, default=-7.0,
                        help="스윕 마지막 전류 [A] (기본: -7.0)")
    parser.add_argument('--i-step', type=float, default=-0.2,
                        help="스텝 크기 [A] (음수, 기본: -0.2)")
    parser.add_argument('--reps', type=int, default=5,
                        help="반복 횟수 (기본: 5)")
    parser.add_argument('--settle', type=float, default=1.0,
                        help="각 스텝 안정화 대기시간 [s] (기본: 1.0)")
    parser.add_argument('--out', type=str, default='motor_constant_measurements.csv',
                        help="CSV 저장 경로 (기본: ./motor_constant_measurements.csv)")
    args = parser.parse_args()

    rclpy.init()

    node = TorqueTestPublisher(
        topic=args.topic,
        index=args.index,
        arm_m=args.arm,
        i_min=args.i_min,
        i_max=args.i_max,
        i_step=args.i_step,
        reps=args.reps,
        settle_s=args.settle,
        outfile=Path(args.out)
    )

    try:
        rclpy.spin(node)  # 퍼블리시 루프
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
