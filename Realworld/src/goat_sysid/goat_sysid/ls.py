#!/usr/bin/env python3
import numpy as np
import argparse
import os

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

def fit_friction_ls(dq_deg_s, tau, v_min_deg_s=5.0):
    """
    LS로 마찰 모델 tau = b * dq + tau_c * sign(dq) 피팅.

    dq_deg_s : 속도 (deg/s)
    tau      : 토크 (same unit as torque_commands)
    v_min_deg_s : 너무 느린 구간(|dq|<v_min)은 제거 (정지/정적마찰 영역 제외)
    """
    # 1) 속도 threshold로 필터링
    mask = np.abs(dq_deg_s) >= v_min_deg_s
    dq_used = dq_deg_s[mask]
    tau_used = tau[mask]

    if dq_used.size < 10:
        raise RuntimeError(f"유효 샘플이 너무 적음: {dq_used.size}개 (v_min={v_min_deg_s} deg/s)")

    # 2) (선택) rad/s로 변환하고 싶으면 여기서
    #    단위까지 신경 쓸 거면 이 라인 활성화:
    # dq = np.deg2rad(dq_used)   # rad/s
    # 지금은 그냥 deg/s 단위 그대로 두고 b 단위도 "토크 / (deg/s)"로 해석
    dq = dq_used

    # 3) 설계 행렬 Φ 구성: [dq, sign(dq)]
    sign_dq = np.sign(dq)
    Phi = np.column_stack([dq, sign_dq])

    # 4) LS 해 계산
    theta, residuals, rank, s = np.linalg.lstsq(Phi, tau_used, rcond=None)
    b_hat = theta[0]
    tau_c_hat = theta[1]

    return b_hat, tau_c_hat, dq_used, tau_used

def main():
    parser = argparse.ArgumentParser(description="Friction LS fitting from SysID CSV log")
    parser.add_argument("--csv", type=str, required=True,
                        help="SysID에서 저장한 CSV 로그 경로")
    parser.add_argument("--vmin", type=float, default=5.0,
                        help="속도 threshold (deg/s). |dq| < vmin 샘플 제거")
    args = parser.parse_args()

    csv_path = args.csv
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없음: {csv_path}")

    print(f"[LS] Load log from: {csv_path}")
    t, q_ref, q_meas, dq_meas_deg, tau = load_log(csv_path)

    print(f"[LS] 전체 샘플 수: {len(t)}")
    print(f"[LS] v_min = {args.vmin} deg/s 기준으로 필터링 후 피팅")

    b_hat, tau_c_hat, dq_used, tau_used = fit_friction_ls(
        dq_meas_deg, tau, v_min_deg_s=args.vmin
    )

    print("\n===== LS 결과 (deg/s 기준) =====")
    print(f"b_hat   (점성 마찰 계수)  ≈ {b_hat:.6f}  [토크 단위 / (deg/s)]")
    print(f"tau_c_hat (쿨롱 마찰 계수) ≈ {tau_c_hat:.6f}  [토크 단위]")

    # (선택) URDF에 rad/s 기준으로 넣고 싶으면 여기서 환산 값도 같이 출력
    b_hat_rad = b_hat * (180.0 / np.pi)   # tau = b_deg * dq_deg = b_rad * dq_rad
    print("\n(참고) rad/s 기준으로 환산하면:")
    print(f"b_hat_rad ≈ {b_hat_rad:.6f}  [토크 단위 / (rad/s)]")
    print(f"tau_c_hat 은 토크 단위 그대로 사용 가능")

    # (선택) 간단한 통계
    print("\n[LS] 사용된 샘플 수:", dq_used.size)
    print(f"[LS] |dq| 평균 ≈ {np.mean(np.abs(dq_used)):.3f} deg/s,"
          f"  최대 ≈ {np.max(np.abs(dq_used)):.3f} deg/s")

if __name__ == "__main__":
    main()
