#!/usr/bin/env python3
'''
python3 plot_motor_temp.py --csv ../motor_temp_log.csv --time wall0 --out temp_wall0.png
'''
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List

def find_temp_columns(cols: List[str]) -> List[str]:
    return [c for c in cols if c.startswith('temp_') and c.endswith('_C')]

def build_time_wall(df: pd.DataFrame):
    if 'wall_time_iso' not in df.columns:
        return None
    t = pd.to_datetime(df['wall_time_iso'], errors='coerce')
    return None if t.isna().all() else t

def build_time_wall_rel(df: pd.DataFrame):
    """wall_time_iso 기준 상대초 (첫 샘플을 t=0으로)."""
    t = build_time_wall(df)
    if t is None:
        return None
    t0 = t.iloc[0]
    # pandas Series -> numpy (float seconds)
    return (t - t0).dt.total_seconds().to_numpy()

def build_time_ros(df: pd.DataFrame):
    if ('ros_sec' in df.columns) and ('ros_nanosec' in df.columns):
        sec = df['ros_sec'].astype('float64').to_numpy()
        nsec = df['ros_nanosec'].astype('float64').to_numpy()
        t = sec + nsec * 1e-9
    elif 'ros_stamp' in df.columns:
        def parse_stamp(s):
            try:
                return float(str(s))
            except Exception:
                try:
                    s = str(s)
                    if '.' in s:
                        a, b = s.split('.', 1)
                        return float(a) + float(f"0.{b}")
                except Exception:
                    return np.nan
                return np.nan
        t = np.array([parse_stamp(s) for s in df['ros_stamp']], dtype='float64')
    else:
        return None
    if np.all(np.isnan(t)):
        return None
    t0 = t[~np.isnan(t)][0]
    return t - t0

def main():
    ap = argparse.ArgumentParser(description="Plot motor temperatures over time from CSV")
    ap.add_argument('--csv', required=True, help='motor_temp_log.csv 경로')
    ap.add_argument('--time', choices=['wall', 'wall0', 'ros'], default='wall',
                    help='x축 시간: wall(절대시간), wall0(wall 기준 t=0), ros(상대초)')
    ap.add_argument('--every', type=int, default=1, help='다운샘플 간격')
    ap.add_argument('--smooth', type=int, default=1, help='이동평균 윈도우 크기')
    ap.add_argument('--out', default=None, help='저장 파일 경로(.png 등)')
    args = ap.parse_args()

    path = Path(args.csv)
    if not path.exists():
        raise FileNotFoundError(f'CSV 파일을 찾을 수 없습니다: {path}')

    df = pd.read_csv(path)
    temp_cols = find_temp_columns(df.columns)
    if not temp_cols:
        raise ValueError('CSV에 temp_*_C 컬럼이 없습니다.')

    # 시간축 선택
    if args.time == 'wall':
        x = build_time_wall(df)
        if x is None:
            print('[WARN] wall_time_iso가 없어 ros 상대시간으로 대체합니다.')
            x = build_time_ros(df)
            if x is None:
                raise ValueError('시간축 생성 실패(wall/ros 모두 불가).')
        x_label = 'Time (wall clock)'
    elif args.time == 'wall0':
        x = build_time_wall_rel(df)
        if x is None:
            print('[WARN] wall_time_iso가 없어 ros 상대시간으로 대체합니다.')
            x = build_time_ros(df)
            if x is None:
                raise ValueError('시간축 생성 실패(wall0/ros 모두 불가).')
        x_label = 'Time [s] (wall clock, t0 = first message)'
    else:  # ros
        x = build_time_ros(df)
        if x is None:
            print('[WARN] ros 시간이 없어 wall 절대시간으로 대체합니다.')
            x = build_time_wall(df)
            if x is None:
                raise ValueError('시간축 생성 실패(ros/wall 모두 불가).')
        x_label = 'Time [s] (ROS relative)'

    # 다운샘플
    if args.every > 1:
        df = df.iloc[::args.every, :].reset_index(drop=True)
        if hasattr(x, 'iloc'):  # pandas Series/DatetimeIndex
            x = x.iloc[::args.every]
        else:                   # numpy array
            x = x[::args.every]

    # 스무딩
    if args.smooth > 1:
        for c in temp_cols:
            df[c] = pd.Series(df[c]).rolling(args.smooth, center=True, min_periods=1).mean()

    # 플롯
    plt.figure(figsize=(10, 5))
    for c in temp_cols:
        plt.plot(x, df[c], label=c)

    plt.title('Motor Temperatures Over Time')
    plt.ylabel('Temperature [°C]')
    plt.xlabel(x_label)
    if args.time == 'wall':
        try:
            plt.gcf().autofmt_xdate()
        except Exception:
            pass
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=150)
        print(f'Saved figure to {args.out}')
    else:
        plt.show()

if __name__ == '__main__':
    main()
