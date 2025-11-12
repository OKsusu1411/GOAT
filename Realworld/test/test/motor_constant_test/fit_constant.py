#!/usr/bin/env python3
"""
fit_motor_constant.py
---------------------
CSV(전류-토크 측정치)를 읽어
 - 절편 포함 OLS (τ = Kτ * I + b)
 - 원점 통과 OLS (τ = Kτ0 * I)
두 가지로 피팅하고, 결과 요약 CSV/PNG를 저장합니다.

추가 옵션
--------
--plot-abs-current : 그래프에서 전류를 |I|로 표시하고, 회귀선도 |I| 기준으로 그림
                      (단, CSV로 저장되는 수치 결과는 원본 I 기준)

Usage (예시)
-----------
python3 fit_motor_constant.py --csv motor_constant_measurements.csv --index 3 --per-rep false --exclude-zero --min-abs-current 0.2 --plot-abs-current

여러 CSV를 함께 분석할 수도 있습니다:
python3 fit_motor_constant.py --csv run1.csv --csv run2.csv --index 3
"""

import argparse
from pathlib import Path
from datetime import datetime


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def r2_centered(y_true, y_pred):
    """표준 R^2 (절편 포함 모델에서 사용)"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def r2_uncentered(y_true, y_pred):
    """원점 통과 모델용 R^2 (uncentered) = 1 - SSE / sum(y^2)"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot_unc = np.sum(y_true ** 2)
    return 1 - ss_res / ss_tot_unc if ss_tot_unc > 0 else np.nan


def fit_with_intercept(x, y):
    """τ = a * I + b"""
    a, b = np.polyfit(x, y, 1)
    yhat = a * x + b
    return a, b, r2_centered(y, yhat)


def fit_through_origin(x, y):
    """τ = a0 * I (원점 통과 최소제곱)"""
    denom = np.dot(x, x)
    a0 = float(np.dot(x, y) / denom) if denom > 0 else np.nan
    yhat0 = a0 * x
    return a0, r2_uncentered(y, yhat0)


def load_and_filter(csv_paths, index=None, xcol='current_A', ycol='torque_Nm',
                    exclude_zero=False, min_abs_current=0.0):
    """CSV들을 로드해 필터링한 단일 DataFrame 반환"""
    dfs = []
    for p in csv_paths:
        df = pd.read_csv(p)
        missing = {c for c in [xcol, ycol] if c not in df.columns}
        if missing:
            raise ValueError(f"{p}에 필요한 컬럼이 없습니다: {missing}")
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    if index is not None and 'index' in data.columns:
        data = data[data['index'] == index].copy()
    if exclude_zero:
        data = data[np.abs(data[xcol]) > 1e-12].copy()
    if min_abs_current and min_abs_current > 0:
        data = data[np.abs(data[xcol]) >= float(min_abs_current)].copy()
    data = data.dropna(subset=[xcol, ycol])
    data = data.sort_values(by=[xcol, 'rep'] if 'rep' in data.columns else [xcol])
    return data


def summarize_group(df, group_key=None, xcol='current_A', ycol='torque_Nm'):
    """단일 그룹(또는 전체)에 대해 두 회귀를 수행하고 dict 결과 반환"""
    x = np.abs(df[xcol].to_numpy(dtype=float))
    y = df[ycol].to_numpy(dtype=float)
    res = {'group': group_key if group_key is not None else 'ALL',
           'n_samples': len(x)}
    if len(x) < 2:
        res.update({
            'K_tau_slope_Nm_per_A': np.nan,
            'intercept_Nm': np.nan,
            'R2': np.nan,
            'K_tau0_originfit_Nm_per_A': np.nan,
            'R2_uncentered': np.nan
        })
        return res

    a, b, R2 = fit_with_intercept(x, y)
    a0, R2u = fit_through_origin(x, y)

    res.update({
        'K_tau_slope_Nm_per_A': a,
        'intercept_Nm': b,
        'R2': R2,
        'K_tau0_originfit_Nm_per_A': a0,
        'R2_uncentered': R2u,
        'x_min_A': float(np.min(x)),
        'x_max_A': float(np.max(x)),
        'y_min_Nm': float(np.min(y)),
        'y_max_Nm': float(np.max(y)),
    })
    return res


def plot_all(df, results_all, xcol='current_A', ycol='torque_Nm',
             out_png=Path('fit_plot.png'), plot_abs=False):
    """
    산점도 + 두 회귀선 저장.
    plot_abs=True 이면 x축을 |I|로 변환하고, 회귀선도 |I| 기준 재추정하여 그림.
        """
    x = np.abs(df[xcol].to_numpy(dtype=float))
    y = df[ycol].to_numpy(dtype=float)

    if plot_abs:
        x_plot = np.abs(x)
        # |I| 기준으로 회귀 재계산 (그래프 일관성)
        a_p, b_p, R2_p = fit_with_intercept(x_plot, y)
        a0_p, R2u_p = fit_through_origin(x_plot, y)

        title = 'Motor Constant Fit (|I| view)'
        eq1_label = f'fit: τ={a_p:.4f}|I|+{b_p:.4f} (R²={R2_p:.3f})'
        eq0_label = f'origin fit: τ={a0_p:.4f}|I| (R²ᵤ={R2u_p:.3f})'
    else:
        x_plot = x
        a_p = results_all['K_tau_slope_Nm_per_A']
        b_p = results_all['intercept_Nm']
        a0_p = results_all['K_tau0_originfit_Nm_per_A']
        title = 'Motor Constant Fit'
        eq1_label = f'fit: τ={a_p:.4f}I+{b_p:.4f} (R²={results_all["R2"]:.3f})'
        eq0_label = f'origin fit: τ={a0_p:.4f}I (R²ᵤ={results_all["R2_uncentered"]:.3f})'

    # 산점도
    plt.figure()
    plt.scatter(x_plot, y, s=5, label='data')

    # 절편 포함 직선
    if np.isfinite(a_p) and np.isfinite(b_p):
        xs = np.linspace(np.min(x_plot), np.max(x_plot), 200)
        ys = a_p * xs + b_p
        plt.plot(xs, ys, linewidth=2, label=eq1_label)

    # 원점 통과 직선
    if np.isfinite(a0_p):
        xs = np.linspace(np.min(x_plot), np.max(x_plot), 200)
        ys0 = a0_p * xs
        plt.plot(xs, ys0, linestyle='--', linewidth=2, label=eq0_label)

    plt.xlabel('|Current| [A]' if plot_abs else 'Current [A]')
    plt.ylabel('Torque [N·m]')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='전류-토크 CSV 피팅 스크립트')
    parser.add_argument('--csv', type=str, action='append', required=True,
                        help='입력 CSV 경로(여러 개 지정 가능, --csv 를 반복)')
    parser.add_argument('--index', type=int, default=None,
                        help='특정 채널 index로 필터 (CSV에 index 컬럼이 있을 때만 적용)')
    parser.add_argument('--per-rep', type=str, default='false', choices=['true', 'false'],
                        help='rep 별 개별 피팅 여부 (기본: false)')
    parser.add_argument('--exclude-zero', action='store_true',
                        help='전류=0 데이터를 제외')
    parser.add_argument('--min-abs-current', type=float, default=0.0,
                        help='|I| < threshold 데이터를 제외 (기본: 0.0)')
    parser.add_argument('--xcol', type=str, default='current_A',
                        help='전류 컬럼명 (기본: current_A)')
    parser.add_argument('--ycol', type=str, default='torque_Nm',
                        help='토크 컬럼명 (기본: torque_Nm)')
    parser.add_argument('--out-prefix', dest='out_prefix', type=str, default='fit',
                        help='출력 파일 prefix (기본: fit)')
    parser.add_argument('--plot-abs-current', action='store_true',
                        help='그래프에서 전류를 |I|로 표시하고 회귀선도 |I| 기준으로 그림')

    args = parser.parse_args()

    csv_paths = [Path(p) for p in args.csv]
    for p in csv_paths:
        if not p.exists():
            raise FileNotFoundError(f'CSV 파일을 찾을 수 없습니다: {p}')

    df = load_and_filter(csv_paths,
                         index=args.index,
                         xcol=args.xcol,
                         ycol=args.ycol,
                         exclude_zero=args.exclude_zero,
                         min_abs_current=args.min_abs_current)

    if df.empty:
        print('필터링 후 데이터가 비었습니다. 옵션을 확인하세요.')
        return

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    suffix = '_absI' if args.plot_abs_current else ''
    out_csv = Path(f'{args.out_prefix}_summary_{ts}.csv')
    out_png = Path(f'{args.out_prefix}_plot{suffix}_{ts}.png')

    # 전체 데이터 피팅 (CSV/요약은 원본 I 기준으로 저장)
    all_res = summarize_group(df, group_key='ALL',
                              xcol=args.xcol, ycol=args.ycol)
    results = [all_res]

    # rep별 피팅
    if args.per_rep.lower() == 'true' and 'rep' in df.columns:
        for rep, g in df.groupby('rep'):
            results.append(summarize_group(g, group_key=f'rep={rep}',
                                           xcol=args.xcol, ycol=args.ycol))

    # 결과 저장
    res_df = pd.DataFrame(results)
    res_df.to_csv(out_csv, index=False)

    # 플롯 저장(그래프 옵션 반영)
    plot_all(df, all_res,
             xcol=args.xcol, ycol=args.ycol,
             out_png=out_png, plot_abs=args.plot_abs_current)

    # 콘솔 요약
    print('\n[피팅 결과 요약]')
    print(res_df.to_string(index=False))
    print(f'\n- 결과 요약 CSV: {out_csv}')
    print(f'- 플롯 이미지  : {out_png}\n')


if __name__ == '__main__':
    main()
