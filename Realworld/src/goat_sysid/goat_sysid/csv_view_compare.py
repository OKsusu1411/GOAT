#!/usr/bin/env python3
"""
사용 예시

[A: CSV 보기(단발/실시간)]
python3 csv_view_compare.py --mode A \
  --csv joint_4_sine.csv \
  --joints 4 \
  --window_sec 0 \
  --unit deg \
  --save_png

실시간 follow (CSV가 계속 append되는 중)
python3 csv_view_compare.py --mode A \
  --csv joint_0_sine.csv \
  --joints 6 \
  --follow \
  --window_sec 10 \
  --refresh_hz 20 \
  --unit deg \
  --fill_missing hold

[B: CSV 두 파일 비교(시작점 정렬 + RMSE)]
python3 csv_view_compare.py --mode B \
  --csv_a /home/heachanlee/GOAT/GOAT/Realworld/logs/run1.csv \
  --csv_b /home/heachanlee/GOAT/GOAT/Realworld/logs/run2.csv \
  --joints 6 \
  --use q_meas \
  --start_method delta \
  --start_thresh 0.01 \
  --zero_mode subtract_initial \
  --dt 0.002 \
  --save_png_compare
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Common CSV helpers
# -----------------------------
def load_csv_safe(csv_path: Path) -> pd.DataFrame:
    """
    CSV가 실행 중에 계속 append되는 경우 마지막 줄이 깨져있을 수 있어,
    최대한 안전하게 읽는다.
    """
    try:
        return pd.read_csv(csv_path)
    except Exception:
        return pd.read_csv(csv_path, engine="python", on_bad_lines="skip")


def parse_joints(joint_str: str) -> List[int]:
    return [int(x) for x in joint_str.split(",") if x.strip() != ""]


def cols_for_joint(prefix: str, j: int) -> str:
    # prefix in {"q_ref","q_meas","dq_meas","tau_cmd"}
    if prefix in ("q_ref", "q_meas"):
        return f"{prefix}_{j}_rad"
    if prefix == "dq_meas":
        return f"{prefix}_{j}_rad_s"
    if prefix == "tau_cmd":
        return f"{prefix}_{j}_nm"
    raise ValueError(prefix)


def require_columns(df: pd.DataFrame, joints: List[int]) -> None:
    need = ["t_sec"]
    for j in joints:
        need += [
            cols_for_joint("q_ref", j),
            cols_for_joint("q_meas", j),
            cols_for_joint("dq_meas", j),
            cols_for_joint("tau_cmd", j),
        ]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError(
            "CSV columns missing:\n  "
            + "\n  ".join(missing)
            + "\n\nCSV format must contain: t_sec, q_ref_i_rad, q_meas_i_rad, dq_meas_i_rad_s, tau_cmd_i_nm"
        )


def fill_missing_series(s: pd.Series, mode: str) -> pd.Series:
    """
    mode:
      - none: 그대로 (NaN이면 그래프가 끊길 수 있음)
      - hold: 이전값 유지 (ffill), 맨 처음 NaN이면 0으로
      - zero: NaN을 0으로
    """
    s = s.replace([np.inf, -np.inf], np.nan)
    if mode == "hold":
        s = s.ffill()
        s = s.fillna(0.0)
    elif mode == "zero":
        s = s.fillna(0.0)
    return s


# -----------------------------
# A) Live view (follow CSV) + save plot
# -----------------------------
def run_view_mode(
    csv_path: Path,
    joints: List[int],
    window_sec: float,
    follow: bool,
    refresh_hz: float,
    save_png: bool,
    out_dir: Optional[Path],
    basename: Optional[str],
    unit: str,          # "rad" or "deg"
    fill_missing: str,  # "none" | "hold" | "zero"
) -> None:
    if out_dir is None:
        out_dir = csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    if not basename:
        basename = csv_path.stem

    to_deg = (unit == "deg")
    scale_q = (180.0 / np.pi) if to_deg else 1.0
    scale_dq = (180.0 / np.pi) if to_deg else 1.0

    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    ax_q = fig.add_subplot(3, 1, 1)
    ax_dq = fig.add_subplot(3, 1, 2, sharex=ax_q)
    ax_tau = fig.add_subplot(3, 1, 3, sharex=ax_q)

    lines: Dict[int, Tuple] = {}
    for j in joints:
        (l_q,) = ax_q.plot([], [], label=f"q_meas[{j}]")
        (l_qref,) = ax_q.plot([], [], linestyle="--", label=f"q_ref[{j}]")
        (l_dq,) = ax_dq.plot([], [], label=f"dq[{j}]")
        (l_tau,) = ax_tau.plot([], [], label=f"tau[{j}]")
        lines[j] = (l_q, l_qref, l_dq, l_tau)

    ax_q.set_ylabel("q (deg)" if to_deg else "q (rad)")
    ax_dq.set_ylabel("dq (deg/s)" if to_deg else "dq (rad/s)")
    ax_tau.set_ylabel("tau (Nm)")
    ax_tau.set_xlabel("t (s)")

    ax_q.legend(loc="upper right", ncol=2, fontsize=8)
    ax_dq.legend(loc="upper right", ncol=2, fontsize=8)
    ax_tau.legend(loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout()
    plt.show()

    last_rows = 0
    period = 1.0 / max(refresh_hz, 0.5)

    print("\n[A 모드] 종료: 그래프 창에서 Ctrl+C 또는 터미널에서 Ctrl+C\n")

    try:
        while True:
            if not csv_path.exists():
                print(f"[WAIT] CSV not found: {csv_path}")
                time.sleep(0.5)
                continue

            df = load_csv_safe(csv_path)
            if df.empty or "t_sec" not in df.columns:
                time.sleep(period)
                continue

            require_columns(df, joints)

            # 정렬 + 결측/inf 처리(hold/zero/none)
            df = df.sort_values("t_sec")
            for j in joints:
                for pfx in ["q_meas", "q_ref", "dq_meas", "tau_cmd"]:
                    col = cols_for_joint(pfx, j)
                    if col in df.columns:
                        df[col] = fill_missing_series(df[col], fill_missing)

            # windowing
            t = df["t_sec"].to_numpy(dtype=float)
            tmax = float(t[-1])
            if window_sec > 0:
                tmin = max(0.0, tmax - window_sec)
                mask = t >= tmin
                dfw = df.loc[mask]
                t = dfw["t_sec"].to_numpy(dtype=float)
            else:
                dfw = df

            # update lines
            for j in joints:
                q = dfw[cols_for_joint("q_meas", j)].to_numpy(dtype=float) * scale_q
                qref = dfw[cols_for_joint("q_ref", j)].to_numpy(dtype=float) * scale_q
                dq = dfw[cols_for_joint("dq_meas", j)].to_numpy(dtype=float) * scale_dq
                tau = dfw[cols_for_joint("tau_cmd", j)].to_numpy(dtype=float)

                lq, lqref, ldq, ltau = lines[j]
                lq.set_data(t, q)
                lqref.set_data(t, qref)
                ldq.set_data(t, dq)
                ltau.set_data(t, tau)

            ax_q.relim(); ax_q.autoscale_view()
            ax_dq.relim(); ax_dq.autoscale_view()
            ax_tau.relim(); ax_tau.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()

            if not follow:
                break

            if len(df) == last_rows:
                time.sleep(period)
            else:
                last_rows = len(df)
                time.sleep(period)

    except KeyboardInterrupt:
        pass
    finally:
        if save_png:
            all_png = out_dir / f"{basename}_view_all.png"
            fig.savefig(all_png, dpi=150)
            print(f"[SAVE] {all_png}")

        plt.ioff()
        plt.show(block=False)


# -----------------------------
# B) Compare two CSVs with start alignment + RMSE
# -----------------------------
def detect_start_time(
    t: np.ndarray,
    q: np.ndarray,
    method: str = "delta",
    thresh: float = 0.01,
) -> float:
    if len(t) < 5:
        return float(t[0]) if len(t) else 0.0

    if method == "dq":
        dq = np.gradient(q, t)
        idx = int(np.argmax(np.abs(dq) >= thresh))
        if np.abs(dq[idx]) < thresh:
            return float(t[0])
        return float(t[idx])

    q0 = q[0]
    d = np.abs(q - q0)
    idx = int(np.argmax(d >= thresh))
    if d[idx] < thresh:
        return float(t[0])
    return float(t[idx])


def interp_on_common_grid(
    t1: np.ndarray,
    y1: np.ndarray,
    t2: np.ndarray,
    y2: np.ndarray,
    dt: float,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    t_min = max(float(t1[0]), float(t2[0]))
    t_max = min(float(t1[-1]), float(t2[-1]))
    if t_max <= t_min:
        return None
    grid = np.arange(t_min, t_max, dt)
    y1i = np.interp(grid, t1, y1)
    y2i = np.interp(grid, t2, y2)
    return grid, y1i, y2i


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def run_compare_mode(
    csv_a: Path,
    csv_b: Path,
    joints: List[int],
    use: str,            # "q_meas" or "q_ref"
    start_method: str,   # "delta" or "dq"
    start_thresh: float, # rad or rad/s
    zero_mode: str,      # "raw" or "subtract_initial"
    dt: float,
    save_png: bool,
    out_dir: Optional[Path],
    unit: str,           # "rad" or "deg" (출력/그래프 표시 단위)
    fill_missing: str,   # 결측 처리(비교에도 동일 적용)
) -> None:
    if out_dir is None:
        out_dir = csv_a.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df_a = load_csv_safe(csv_a)
    df_b = load_csv_safe(csv_b)

    require_columns(df_a, joints)
    require_columns(df_b, joints)

    # 정렬 + 결측 처리
    df_a = df_a.sort_values("t_sec")
    df_b = df_b.sort_values("t_sec")
    for j in joints:
        col = cols_for_joint("q_meas" if use == "q_meas" else "q_ref", j)
        df_a[col] = fill_missing_series(df_a[col], fill_missing)
        df_b[col] = fill_missing_series(df_b[col], fill_missing)

    prefix = "q_meas" if use == "q_meas" else "q_ref"

    to_deg = (unit == "deg")
    scale = (180.0 / np.pi) if to_deg else 1.0
    y_label = "deg" if to_deg else "rad"

    rows_out = []
    rmse_vals = []

    for j in joints:
        tA = df_a["t_sec"].to_numpy(dtype=float)
        tB = df_b["t_sec"].to_numpy(dtype=float)
        yA = df_a[cols_for_joint(prefix, j)].to_numpy(dtype=float)
        yB = df_b[cols_for_joint(prefix, j)].to_numpy(dtype=float)

        t0A = detect_start_time(tA, yA, method=start_method, thresh=start_thresh)
        t0B = detect_start_time(tB, yB, method=start_method, thresh=start_thresh)

        tA2 = tA - t0A
        tB2 = tB - t0B

        if zero_mode == "subtract_initial":
            yA0 = float(np.interp(0.0, tA2, yA))
            yB0 = float(np.interp(0.0, tB2, yB))
            yA2 = yA - yA0
            yB2 = yB - yB0
        else:
            yA2, yB2 = yA, yB

        packed = interp_on_common_grid(tA2, yA2, tB2, yB2, dt=dt)
        if packed is None:
            rows_out.append([j, t0A, t0B, ""])
            continue

        grid, yAi, yBi = packed
        e = rmse(yAi, yBi)
        rmse_vals.append(e)
        rows_out.append([j, t0A, t0B, e])

        if save_png:
            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(tA2, yA2 * scale, label=f"A {prefix}[{j}]")
            ax.plot(tB2, yB2 * scale, label=f"B {prefix}[{j}]")
            ax.set_xlabel("t aligned (s)")
            ax.set_ylabel(f"{y_label}" + (" (zeroed)" if zero_mode == "subtract_initial" else ""))
            ax.set_title(f"Joint {j} start-aligned | RMSE={e*scale:.6f} {y_label}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / f"compare_{csv_a.stem}_vs_{csv_b.stem}_joint{j}_{prefix}.png", dpi=150)
            plt.close(fig)

    print("\n=== [B 모드] Compare Summary ===")
    for (j, t0A, t0B, e) in rows_out:
        if e == "":
            print(f"joint {j}: startA={t0A:.4f}s startB={t0B:.4f}s RMSE=NA (no overlap)")
        else:
            print(f"joint {j}: startA={t0A:.4f}s startB={t0B:.4f}s RMSE={float(e)*scale:.6f} {y_label}")

    if rmse_vals:
        print(f"Overall mean RMSE: {float(np.mean(rmse_vals))*scale:.6f} {y_label}")

    summary_path = out_dir / f"compare_summary_{csv_a.stem}_vs_{csv_b.stem}.csv"
    with summary_path.open("w") as f:
        f.write("joint,startA_sec,startB_sec,rmse_in_rad\n")
        for (j, t0A, t0B, e) in rows_out:
            f.write(f"{j},{t0A:.6f},{t0B:.6f},{'' if e=='' else float(e):.9f}\n")
    print(f"[SAVE] {summary_path}")


# -----------------------------
# Interactive launcher (A/B)
# -----------------------------
def interactive_menu() -> str:
    print("\n무엇을 할까?")
    print("  A) CSV 실시간/단발 그래프 보기 (+PNG 저장)")
    print("  B) CSV 두 파일 비교 (시작점 정렬 + RMSE + 비교 그래프/요약 저장)")
    while True:
        sel = input("선택 (A/B): ").strip().lower()
        if sel in ("a", "b"):
            return sel
        print("A 또는 B만 입력해줘.")


def main():
    ap = argparse.ArgumentParser(description="CSV viewer & comparator (A/B interactive)")
    ap.add_argument("--mode", choices=["A", "B", "a", "b"], default="")

    # 공통 표시 옵션
    ap.add_argument("--unit", choices=["rad", "deg"], default="deg",
                    help="표시 단위 (deg 추천)")
    ap.add_argument("--fill_missing", choices=["none", "hold", "zero"], default="hold",
                    help="중간 NaN/결측 처리: hold=이전값 유지, zero=0으로 채움, none=그대로")

    # A args
    ap.add_argument("--csv", default="")
    ap.add_argument("--joints", default="6")
    ap.add_argument("--window_sec", type=float, default=10.0)
    ap.add_argument("--follow", action="store_true", help="CSV가 실행 중에 계속 늘어날 때 tail-follow로 갱신")
    ap.add_argument("--refresh_hz", type=float, default=20.0)
    ap.add_argument("--save_png", action="store_true")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--basename", default="")

    # B args
    ap.add_argument("--csv_a", default="")
    ap.add_argument("--csv_b", default="")
    ap.add_argument("--use", choices=["q_meas", "q_ref"], default="q_meas")
    ap.add_argument("--start_method", choices=["delta", "dq"], default="delta")
    ap.add_argument("--start_thresh", type=float, default=0.01, help="delta: rad, dq: rad/s")
    ap.add_argument("--zero_mode", choices=["raw", "subtract_initial"], default="subtract_initial")
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--save_png_compare", action="store_true")
    args = ap.parse_args()

    mode = args.mode.strip().lower() if args.mode else ""
    if mode not in ("a", "b"):
        mode = interactive_menu()

    out_dir = Path(args.out_dir) if args.out_dir else None
    joints = parse_joints(args.joints)

    if mode == "a":
        if not args.csv:
            args.csv = input("CSV 파일 경로: ").strip()
        csv_path = Path(args.csv)

        run_view_mode(
            csv_path=csv_path,
            joints=joints,
            window_sec=args.window_sec,
            follow=args.follow,
            refresh_hz=args.refresh_hz,
            save_png=args.save_png,
            out_dir=out_dir,
            basename=args.basename if args.basename else None,
            unit=args.unit,
            fill_missing=args.fill_missing,
        )
        return

    # mode == "b"
    if not args.csv_a:
        args.csv_a = input("CSV A 경로: ").strip()
    if not args.csv_b:
        args.csv_b = input("CSV B 경로: ").strip()

    csv_a = Path(args.csv_a)
    csv_b = Path(args.csv_b)

    run_compare_mode(
        csv_a=csv_a,
        csv_b=csv_b,
        joints=joints,
        use=args.use,
        start_method=args.start_method,
        start_thresh=args.start_thresh,
        zero_mode=args.zero_mode,
        dt=args.dt,  # <- 여기 오타 수정됨
        save_png=args.save_png_compare,
        out_dir=out_dir,
        unit=args.unit,
        fill_missing=args.fill_missing,
    )


if __name__ == "__main__":
    main()
