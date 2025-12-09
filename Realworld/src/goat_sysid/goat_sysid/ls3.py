#!/usr/bin/env python3
"""
fit_joint_sysid_plot.py

DynamicFrictionSysID로 뽑은 CSV 로그를 이용해서

  motor_only 모드:
    - 모터 관성 J_motor는 알고 있다고 가정
    - tau* = tau - J_motor * qdd
    - tau* = a * qd + b * sign(qd)  를 LS로 피팅
    - 그래프: dq vs tau* (data + fit)

  full_joint 모드:
    - tau = J * qdd + a * qd + b * sign(qd)  를 LS로 피팅
    - 그래프: dq vs tau (data + fit)
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def load_log(csv_path):
    """
    CSV 포맷 (DynamicFrictionSysID 기준):

    t, q_ref_deg, q_meas_deg, dq_meas_deg_per_s, tau_cmd
    """
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    t = data[:, 0]
    q_ref = data[:, 1]
    q_meas_deg = data[:, 2]
    dq_meas_deg = data[:, 3]
    tau = data[:, 4]
    return t, q_ref, q_meas_deg, dq_meas_deg, tau


def compute_vel_accel_rad(t, dq_deg, smooth_window=1):
    """
    dq_deg (deg/s) 시퀀스로부터
      - (선택) 이동평균으로 스무딩
      - qdd_deg (deg/s^2) 수치 미분
      - rad/s, rad/s^2 로 변환

    반환:
        dq_rad, qdd_rad
    """
    dq_deg = np.asarray(dq_deg)

    if smooth_window > 1:
        kernel = np.ones(smooth_window, dtype=float) / smooth_window
        dq_deg_filt = np.convolve(dq_deg, kernel, mode="same")
    else:
        dq_deg_filt = dq_deg

    # t 기준 gradient 사용 (t가 균일하지 않을 수도 있으니까)
    qdd_deg = np.gradient(dq_deg_filt, t)

    dq_rad = np.deg2rad(dq_deg_filt)
    qdd_rad = np.deg2rad(qdd_deg)
    return dq_rad, qdd_rad


def fit_motor_only(t, dq_deg, tau, J_motor, vmin_deg_s=5.0, smooth_window=1):
    """
    모터만 분리된 상태 (관성 J_motor는 알고 있다고 가정).
    tau* = tau - J_motor * qdd  를 만들고,
    tau* = a * qd + b * sign(qd) 를 LS로 피팅.

    반환:
        a_hat, b_hat, dq_used_rad, tau_used_star
    """
    dq_rad, qdd_rad = compute_vel_accel_rad(t, dq_deg, smooth_window=smooth_window)

    # 관성 토크 제거
    tau_star = tau - J_motor * qdd_rad

    # 저속 구간 제거 (정지마찰/노이즈 억제용)
    mask = np.abs(dq_deg) >= vmin_deg_s
    dq_used_rad = dq_rad[mask]
    tau_used_star = tau_star[mask]

    if dq_used_rad.size < 10:
        raise RuntimeError(f"[motor_only] 유효 샘플이 너무 적습니다: {dq_used_rad.size}개")

    Phi = np.column_stack([dq_used_rad, np.sign(dq_used_rad)])
    theta, residuals, rank, s = np.linalg.lstsq(Phi, tau_used_star, rcond=None)
    a_hat, b_hat = theta[0], theta[1]

    return a_hat, b_hat, dq_used_rad, tau_used_star


def fit_full_joint(t, dq_deg, tau, vmin_deg_s=5.0, smooth_window=1):
    """
    조인트 전체(모터+벨트+링크 등)에 대해
    tau = J * qdd + a * qd + b * sign(qd) 를 LS로 피팅.

    반환:
        J_hat, a_hat, b_hat, dq_used_rad, qdd_used_rad, tau_used
    """
    dq_rad, qdd_rad = compute_vel_accel_rad(t, dq_deg, smooth_window=smooth_window)

    mask = np.abs(dq_deg) >= vmin_deg_s
    dq_used_rad = dq_rad[mask]
    qdd_used_rad = qdd_rad[mask]
    tau_used = tau[mask]

    if dq_used_rad.size < 10:
        raise RuntimeError(f"[full_joint] 유효 샘플이 너무 적습니다: {dq_used_rad.size}개")

    Phi = np.column_stack([qdd_used_rad, dq_used_rad, np.sign(dq_used_rad)])
    theta, residuals, rank, s = np.linalg.lstsq(Phi, tau_used, rcond=None)
    J_hat, a_hat, b_hat = theta[0], theta[1], theta[2]

    return J_hat, a_hat, b_hat, dq_used_rad, qdd_used_rad, tau_used


def plot_motor_only(dq_used_rad, tau_used_star, a_hat, b_hat, save_fig=None, title_extra=""):
    """
    motor_only 모드용 플롯:
      x축: dq (deg/s)
      y축: tau* (tau - J_motor*qdd)
      data + LS fit
    """
    dq_used_deg = np.rad2deg(dq_used_rad)
    # 정렬해서 예쁘게 선 그리기
    idx = np.argsort(dq_used_deg)
    dq_line_deg = dq_used_deg[idx]
    dq_line_rad = np.deg2rad(dq_line_deg)
    tau_fit = a_hat * dq_line_rad + b_hat * np.sign(dq_line_rad)

    plt.figure()
    plt.scatter(dq_used_deg, tau_used_star, s=4, alpha=0.3, label="data (tau*)")
    plt.plot(dq_line_deg, tau_fit, linewidth=2, label="LS fit")
    plt.xlabel("joint speed dq (deg/s)")
    plt.ylabel("tau* = tau - J_motor * qdd (same torque unit)")
    plt.title("motor_only: tau* ≈ a*qd + b*sign(qd)" + title_extra)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_fig is not None:
        plt.savefig(save_fig, dpi=200)
        print(f"[motor_only] 그래프 저장: {save_fig}")

    # plt.show()  # 필요하면 직접 열어보기


def plot_full_joint(dq_used_rad, qdd_used_rad, tau_used, J_hat, a_hat, b_hat,
                    save_fig=None, title_extra=""):
    """
    full_joint 모드용 플롯:
      x축: dq (deg/s)
      y축: tau
      data + LS fit(각 샘플에서 예측된 tau_hat)
    """
    dq_used_deg = np.rad2deg(dq_used_rad)
    tau_hat = J_hat * qdd_used_rad + a_hat * dq_used_rad + b_hat * np.sign(dq_used_rad)

    # 정렬해서 선형 느낌으로 보기 좋게
    idx = np.argsort(dq_used_deg)
    dq_line_deg = dq_used_deg[idx]
    tau_line = tau_hat[idx]

    plt.figure()
    plt.scatter(dq_used_deg, tau_used, s=4, alpha=0.3, label="data (tau)")
    plt.plot(dq_line_deg, tau_line, linewidth=2, label="LS fit")
    plt.xlabel("joint speed dq (deg/s)")
    plt.ylabel("torque command (same unit)")
    plt.title("full_joint: tau ≈ J*qdd + a*qd + b*sign(qd)" + title_extra)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_fig is not None:
        plt.savefig(save_fig, dpi=200)
        print(f"[full_joint] 그래프 저장: {save_fig}")

    plt.show()  # 필요하면 직접 열어보기

def plot_full_joint_3d(dq_used_rad, qdd_used_rad, tau_used,
                       J_hat, a_hat, b_hat,
                       save_fig=None, title_extra=""):
    """
    full_joint 모드 3D 플롯:
      x축: dq (deg/s)
      y축: qdd (rad/s^2)
      z축: tau

    - 점: 실제 데이터
    - 면: LS로 피팅한 tau_hat = J*qdd + a*dq + b*sign(dq)
    """
    dq_deg = np.rad2deg(dq_used_rad)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 1) 데이터 산점도
    ax.scatter(dq_deg, qdd_used_rad, tau_used, s=4, alpha=0.3)

    # 2) 피팅 평면 생성
    dq_min, dq_max = dq_deg.min(), dq_deg.max()
    qdd_min, qdd_max = qdd_used_rad.min(), qdd_used_rad.max()

    dq_grid_deg, qdd_grid = np.meshgrid(
        np.linspace(dq_min, dq_max, 30),
        np.linspace(qdd_min, qdd_max, 30)
    )
    dq_grid_rad = np.deg2rad(dq_grid_deg)

    tau_grid = J_hat * qdd_grid + a_hat * dq_grid_rad + b_hat * np.sign(dq_grid_rad)

    ax.plot_surface(dq_grid_deg, qdd_grid, tau_grid, alpha=0.4)

    ax.set_xlabel("dq (deg/s)")
    ax.set_ylabel("qdd (rad/s^2)")
    ax.set_zlabel("tau (same unit)")
    ax.set_title("full_joint 3D: tau ≈ J*qdd + a*dq + b*sign(dq)" + title_extra)

    if save_fig is not None:
        plt.tight_layout()
        plt.savefig(save_fig, dpi=200)
        print(f"[full_joint] 3D 그래프 저장: {save_fig}")

    plt.show()  # GUI 있으면 이걸로 바로 보기


def main():
    parser = argparse.ArgumentParser(description="Joint SysID (J, a, b) + plot from DynamicFrictionSysID CSV")
    parser.add_argument("--csv", type=str, required=True,
                        help="DynamicFrictionSysID 로그 CSV 경로")
    parser.add_argument("--mode", type=str, choices=["motor_only", "full_joint"],
                        default="motor_only",
                        help="식별 모드 선택: motor_only / full_joint")
    parser.add_argument("--vmin", type=float, default=5.0,
                        help="속도 threshold (deg/s). |dq| < vmin 샘플은 제거")
    parser.add_argument("--smooth", type=int, default=1,
                        help="속도 스무딩용 moving average window size (샘플 수). 1이면 스무딩 안 함.")
    parser.add_argument("--J_motor", type=float, default=None,
                        help="motor_only 모드에서 사용할 모터 관성 J_motor [토크단위/(rad/s^2)]")
    parser.add_argument("--save_fig", type=str, default=None,
                        help="그래프를 저장할 파일 경로(.png 등). 지정 안 하면 저장 안 함.")
    parser.add_argument("--save_fig3d", type=str, default=None,
                        help="full_joint 3D 그래프를 저장할 파일 경로(.png 등). 지정 안 하면 3D 저장 안 함.")


    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {args.csv}")

    print(f"[SYSID] CSV 로드: {args.csv}")
    t, q_ref, q_meas_deg, dq_meas_deg, tau = load_log(args.csv)
    print(f"[SYSID] 전체 샘플 수: {len(t)}")
    print(f"[SYSID] mode={args.mode}, vmin={args.vmin} deg/s, smooth_window={args.smooth}")

    if args.mode == "motor_only":
        if args.J_motor is None:
            raise ValueError("--mode motor_only 를 쓰려면 --J_motor 를 반드시 지정해야 합니다.")

        print(f"[motor_only] J_motor = {args.J_motor:.6e} [토크단위/(rad/s^2)]")
        a_hat, b_hat, dq_used_rad, tau_used_star = fit_motor_only(
            t, dq_meas_deg, tau,
            J_motor=args.J_motor,
            vmin_deg_s=args.vmin,
            smooth_window=args.smooth
        )

        print("\n===== motor_only 결과 (rad/s 기준) =====")
        print(f"a_hat (점성 마찰 계수) ≈ {a_hat:.6e} [토크단위 / (rad/s)]")
        print(f"b_hat (쿨롱 마찰 계수) ≈ {b_hat:.6e} [토크단위]")
        print(f"[motor_only] 사용된 샘플 수: {dq_used_rad.size}")
        print(f"[motor_only] |dq| 평균 ≈ {np.mean(np.abs(dq_used_rad)):.3f} rad/s,"
              f" 최대 ≈ {np.max(np.abs(dq_used_rad)):.3f} rad/s")

        print("\n[URDF hint] 이 모터만 사용할 경우:")
        print(f"  <dynamics damping=\"{a_hat:.6e}\" friction=\"{b_hat:.6e}\" />")

        if args.save_fig is not None:
            title_extra = f"\n(vmin={args.vmin} deg/s, smooth={args.smooth})"
            plot_motor_only(dq_used_rad, tau_used_star, a_hat, b_hat,
                            save_fig=args.save_fig, title_extra=title_extra)

    elif args.mode == "full_joint":
        J_hat, a_hat, b_hat, dq_used_rad, qdd_used_rad, tau_used = fit_full_joint(
            t, dq_meas_deg, tau,
            vmin_deg_s=args.vmin,
            smooth_window=args.smooth
        )

        print("\n===== full_joint 결과 (rad/s 기준) =====")
        print(f"J_hat (등가 관성)      ≈ {J_hat:.6e} [토크단위 / (rad/s^2)]")
        print(f"a_hat (점성 마찰 계수) ≈ {a_hat:.6e} [토크단위 / (rad/s)]")
        print(f"b_hat (쿨롱 마찰 계수) ≈ {b_hat:.6e} [토크단위]")
        print(f"[full_joint] 사용된 샘플 수: {dq_used_rad.size}")
        print(f"[full_joint] |dq| 평균 ≈ {np.mean(np.abs(dq_used_rad)):.3f} rad/s,"
              f" 최대 ≈ {np.max(np.abs(dq_used_rad)):.3f} rad/s")

        print("\n[URDF hint] 조인트 전체 모델에 바로 넣고 싶으면 (마찰만):")
        print("  <dynamics damping=\"{:.6e}\" friction=\"{:.6e}\" />".format(a_hat, b_hat))
        print("  (J_hat은 CAD 관성과 비교/검증용으로 활용)")

        if args.save_fig is not None:
            title_extra = f"\n(vmin={args.vmin} deg/s, smooth={args.smooth})"
            plot_full_joint(dq_used_rad, qdd_used_rad, tau_used,
                            J_hat, a_hat, b_hat,
                            save_fig=args.save_fig,
                            title_extra=title_extra)
        if args.save_fig3d is not None:
            title_extra = f"\n(vmin={args.vmin} deg/s, smooth={args.smooth})"
            plot_full_joint(dq_used_rad, qdd_used_rad, tau_used,
                            J_hat, a_hat, b_hat,
                            save_fig=args.save_fig,
                            title_extra=title_extra)

        if args.save_fig3d is not None:
            title_extra = f"\n(vmin={args.vmin} deg/s, smooth={args.smooth})"
            plot_full_joint_3d(dq_used_rad, qdd_used_rad, tau_used,
                               J_hat, a_hat, b_hat,
                               save_fig=args.save_fig3d,
                               title_extra=title_extra)

    else:
        raise ValueError(f"알 수 없는 mode: {args.mode}")


if __name__ == "__main__":
    main()
