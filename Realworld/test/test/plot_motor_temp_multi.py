#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def find_temp_columns(cols: List[str]) -> List[str]:
    return [c for c in cols if c.startswith('temp_') and c.endswith('_C')]

def parse_only(val: Optional[str]) -> Optional[List[str]]:
    """--only '0,3,7' → ['temp_0_C','temp_3_C','temp_7_C']"""
    if not val:
        return None
    idx = []
    for p in (s.strip() for s in val.split(',')):
        if p == '':
            continue
        try:
            i = int(p)
            if i >= 0:
                idx.append(f'temp_{i}_C')
        except Exception:
            pass
    return idx or None

def build_time_wall(df: pd.DataFrame):
    if 'wall_time_iso' not in df.columns:
        return None
    t = pd.to_datetime(df['wall_time_iso'], errors='coerce')
    return None if t.isna().all() else t

def build_time_wall_rel(df: pd.DataFrame):
    t = build_time_wall(df)
    if t is None:
        return None
    t0 = t.iloc[0]
    return (t - t0).dt.total_seconds().to_numpy()  # float seconds

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

def make_time(df: pd.DataFrame, mode: str):
    if mode == 'wall':
        x = build_time_wall(df)
        if x is None:
            x = build_time_ros(df)
            if x is None:
                raise ValueError('시간축 생성 실패(wall/ros 모두 불가).')
        xlabel = 'Time (wall clock)'
        return x, xlabel
    if mode == 'wall0':
        x = build_time_wall_rel(df)
        if x is None:
            x = build_time_ros(df)
            if x is None:
                raise ValueError('시간축 생성 실패(wall0/ros 모두 불가).')
        xlabel = 'Time [s] (wall clock, t0=first sample)'
        return x, xlabel
    # ros
    x = build_time_ros(df)
    if x is None:
        x = build_time_wall(df)
        if x is None:
            raise ValueError('시간축 생성 실패(ros/wall 모두 불가).')
    xlabel = 'Time [s] (ROS relative)'
    return x, xlabel

def main():
    ap = argparse.ArgumentParser(description='Overlay motor temperature CSVs with different colors')
    ap.add_argument('--csv', nargs='+', required=True, help='비교할 CSV 경로들 (2개 이상 가능)')
    ap.add_argument('--labels', nargs='*', default=None, help='범례 라벨(파일 수와 같거나 생략 시 파일명 사용)')
    ap.add_argument('--time', choices=['wall', 'wall0', 'ros'], default='wall0', help='x축 시간 모드')
    ap.add_argument('--every', type=int, default=1, help='다운샘플 간격(예: 10이면 10개 중 1개만)')
    ap.add_argument('--smooth', type=int, default=1, help='이동평균 윈도우 크기(포인트)')
    ap.add_argument('--only', default=None, help='특정 모터 인덱스만 플롯(예: "0,3,7")')
    ap.add_argument('--out', default=None, help='저장 파일 경로(.png 등); 없으면 화면 표시')
    args = ap.parse_args()

    paths = [Path(p) for p in args.csv]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f'CSV 파일을 찾을 수 없음: {p}')

    # 라벨 준비
    if args.labels and len(args.labels) != len(paths):
        raise ValueError('--labels 개수는 --csv 개수와 같아야 합니다.')
    labels = args.labels or [p.stem for p in paths]

    # 색상/선스타일(파일마다 구분)
    # 사용자 요청: 파일마다 색 다르게
    default_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4'])
    line_styles = ['-', '--', '-.', ':']

    # 그리기
    plt.figure(figsize=(11, 5))
    xlabel = None

    only_cols = parse_only(args.only)  # ['temp_0_C', ...] or None

    for fi, (p, lab) in enumerate(zip(paths, labels)):
        df = pd.read_csv(p)

        # 시간축
        x, xlabel_now = make_time(df, args.time)
        xlabel = xlabel or xlabel_now  # 처음 값 채택

        # 다운샘플
        if args.every > 1:
            df = df.iloc[::args.every, :].reset_index(drop=True)
            if hasattr(x, 'iloc'):
                x = x.iloc[::args.every]
            else:
                x = x[::args.every]

        # 온도 컬럼 결정
        temp_cols = find_temp_columns(df.columns)
        if only_cols:
            temp_cols = [c for c in temp_cols if c in only_cols]
        if not temp_cols:
            raise ValueError(f'{p} 에 temp_*_C 컬럼이 없습니다(또는 --only 필터 후 없음).')

        # 스무딩
        if args.smooth > 1:
            for c in temp_cols:
                df[c] = pd.Series(df[c]).rolling(args.smooth, center=True, min_periods=1).mean()

        # 파일별 스타일 고정(같은 파일 내 여러 모터는 같은 색, 다른 선스타일)
        color = default_colors[fi % len(default_colors)]
        for ci, c in enumerate(temp_cols):
            ls = line_styles[ci % len(line_styles)]
            plt.plot(x, df[c], linestyle=ls, color=color, label=f'{lab}:{c}', alpha=0.9)

    plt.title('Motor Temperatures (overlay)')
    plt.xlabel(xlabel)
    plt.ylabel('Temperature [°C]')
    if args.time == 'wall':
        try:
            plt.gcf().autofmt_xdate()
        except Exception:
            pass
    plt.grid(True)
    plt.legend(ncol=2, fontsize='small')
    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=160)
        print(f'Saved figure to {args.out}')
    else:
        plt.show()

if __name__ == '__main__':
    main()
