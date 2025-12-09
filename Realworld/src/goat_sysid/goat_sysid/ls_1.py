#!/usr/bin/env python3
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

def load_log(csv_path):
    """
    CSV 포맷:
    t, q_ref_deg, q_meas_deg, dq_meas_deg_per_s, tau_cmd
    """
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    t = data[:, 0]
    q_ref = data[:, 1]
    q_meas = data[:, 2]
    dq_meas_deg = data[:, 3]   # deg/s
    tau = data[:, 4]           # torque_commands 단위
    return t, q_ref, q_meas, dq_meas_deg, tau

def fit_ls(dq_deg_s, tau, v_min_deg_s):
    """
    tau = b * dq + tau_c * sign(dq) 를 LS로 피팅.
    """
    mask = np.abs(dq_deg_s) >= v_min_deg_s
    dq_used = dq_deg_s[mask]
    tau_used = tau[mask]

    if dq_used.size < 10:
        raise RuntimeError(f"유효 샘플이 너무 적음: {dq_used.size}개 (v_min={v_min_deg_s} deg/s)")

    dq = dq_used
    sign_dq = np.sign(dq)
    Phi = np.column_stack([dq, sign_dq])

    theta, residuals, rank, s = np.linalg.lstsq(Phi, tau_used, rcond=None)
    b_hat, tau_c_hat = theta[0], theta[1]
    return b_hat, tau_c_hat, dq_used, tau_used

def main():
    parser = argparse.ArgumentParser(description="Friction LS + plot")
    parser.add_argument("--csv", type=str, required=True,
                        help="SysID CSV 로그 경로")
    parser.add_argument("--vmin", type=float, default=5.0,
                        help="속도 threshold (deg/s), |dq| < vmin 제거")
    parser.add_argument("--save", type=str, default="friction_fit.png",
                        help="저장할 그림 파일 이름(.png 등)")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없음: {args.csv}")

    print(f"[PLOT] Load log from: {args.csv}")
    t, q_ref, q_meas, dq_meas_deg, tau = load_log(args.csv)

    print(f"[PLOT] 전체 샘플 수: {len(t)}")
    b_hat, tau_c_hat, dq_used, tau_used = fit_ls(dq_meas_deg, tau, args.vmin)

    print("\n===== LS 결과 (deg/s 기준) =====")
    print(f"b_hat   ≈ {b_hat:.6f}  [토크 / (deg/s)]")
    print(f"tau_c   ≈ {tau_c_hat:.6f}  [토크]")
    b_hat_rad = b_hat * (180.0 / np.pi)
    print(f"b_hat_rad ≈ {b_hat_rad:.6f}  [토크 / (rad/s)]")
    print(f"[PLOT] 사용된 샘플 수: {dq_used.size}")

    # ---- 플롯: tau vs dq (산점도 + 피팅 직선) ----
    v_min = np.min(dq_used)
    v_max = np.max(dq_used)
    v_line = np.linspace(v_min, v_max, 400)
    tau_line = b_hat * v_line + tau_c_hat * np.sign(v_line)

    plt.figure()
    plt.scatter(dq_used, tau_used, s=4, alpha=0.3, label="data")
    plt.plot(v_line, tau_line, linewidth=2, label="LS fit")
    plt.xlabel("joint speed dq (deg/s)")
    plt.ylabel("torque command (same unit)")
    plt.title("Friction LS fit: tau ≈ b*dq + tau_c*sign(dq)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.save, dpi=200)
    print(f"[PLOT] 그림 저장: {args.save}")

    # 로컬에서 GUI 가능하면 주석 해제해서 바로 보기
    # plt.show()

if __name__ == "__main__":
    main()
