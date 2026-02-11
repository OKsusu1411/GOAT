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

'''
python3 /home/heachanlee/GOAT/GOAT/Realworld/src/goat_sysid/goat_sysid/ls3.py \
  --csv /home/heachanlee/GOAT/GOAT/Realworld/src/goat_sysid/goat_sysid/sysid_joint4_20260208_200748.csv \
  --mode full_joint \
  --vmin 2 \
  --smooth 21 \
  --loss l2 \
  --save_fig full_joint_2d.png \
  --save_fig3d full_joint_3d.png \
  --save_fig_fric full_joint_fric_only.png


'''

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from mpl_toolkits.mplot3d import Axes3D  # 파일 맨 위에 없으면 추가


def load_log(csv_path):
    """
    현재 CSV 포맷 (너가 올린 파일 기준):
      t_sec, q_ref_rad, q_meas_rad, dq_meas_rad_s, tau_cmd_nm

    내부 로직이 deg/s 기반(vmin, plot 등)이라
    여기서 rad -> deg 로 변환해서 반환한다.
    """
    data = np.genfromtxt(csv_path, delimiter=",", names=True)

    required = ["t_sec", "q_ref_rad", "q_meas_rad", "dq_meas_rad_s", "tau_cmd_nm"]
    for k in required:
        if k not in data.dtype.names:
            raise ValueError(f"CSV 컬럼 '{k}' 가 없습니다. 실제 컬럼: {data.dtype.names}")

    t = data["t_sec"]
    q_ref_deg = np.rad2deg(data["q_ref_rad"])
    q_meas_deg = np.rad2deg(data["q_meas_rad"])
    dq_deg = np.rad2deg(data["dq_meas_rad_s"])     # ✅ rad/s -> deg/s
    tau = data["tau_cmd_nm"]                       # ✅ N*m 그대로

    return t, q_ref_deg, q_meas_deg, dq_deg, tau



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


def fit_motor_only(t, dq_deg, tau, J_motor,
                   vmin_deg_s=5.0, smooth_window=1,
                   loss="l2"):
    """
    모터만 분리된 상태 (관성 J_motor는 알고 있다고 가정).
    tau* = tau - J_motor * qdd  를 만들고,
    tau* = a * qd + b * sign(qd) 를 피팅.

    loss:
      - "l2"      : 일반 least squares (np.linalg.lstsq)
      - "l1_irls" : IRLS 근사
      - "l1_cvx"  : cvxpy로 정확한 L1
      - "ransac"  : RANSAC 기반 robust 회귀
    """
    # deg/s → rad/s, rad/s^2
    dq_rad, qdd_rad = compute_vel_accel_rad(t, dq_deg, smooth_window=smooth_window)

    # 관성 토크 제거
    tau_star = tau - J_motor * qdd_rad

    # 너무 느린 구간 제거
    mask = np.abs(dq_deg) >= vmin_deg_s
    dq_used_rad = dq_rad[mask]
    tau_used_star = tau_star[mask]

    if dq_used_rad.size < 10:
        raise RuntimeError(f"[motor_only] 유효 샘플이 너무 적습니다: {dq_used_rad.size}개")

    # Φ = [ dq, sign(dq) ]
    Phi = np.column_stack([dq_used_rad, np.sign(dq_used_rad)])

    # ---- 손실함수별 회귀 ----
    if loss == "l2":
        theta, _, _, _ = np.linalg.lstsq(Phi, tau_used_star, rcond=None)
    elif loss == "l1_cvx":
        theta = solve_l1_cvxpy(Phi, tau_used_star, verbose=False)
    elif loss == "l1_irls":
        theta = irls_l1(Phi, tau_used_star, max_iter=30, eps=1e-6, verbose=False)
    elif loss == "ransac":
        theta = ransac_linear(
            Phi, tau_used_star,
            n_iter=500,
            min_inlier_ratio=0.3,
            random_state=0,
            verbose=True,
        )
    else:
        raise ValueError(f"Unknown loss: {loss}")

    a_hat, b_hat = theta[0], theta[1]
    return a_hat, b_hat, dq_used_rad, tau_used_star




def plot_full_joint_fric_only(dq_used_rad, qdd_used_rad, tau_used,
                              J_hat, a_hat, b_hat,
                              save_fig=None, title_extra=""):
    """
    full_joint 모드에서 '관성항 제거 후 마찰만' 2D 플롯:

      tau_fric_only = tau - J_hat * qdd

      x축: dq (deg/s)
      y축: tau_fric_only

    - 점: 관성항 제거된 실제 데이터 (tau - J*qdd)
    - 선: 마찰 모델 tau_hat_fric = a*dq + b*sign(dq)
    """
    dq_deg = np.rad2deg(dq_used_rad)
    tau_fric_only = tau_used - J_hat * qdd_used_rad

    # 선 그리기용으로 dq 정렬
    idx = np.argsort(dq_deg)
    dq_line_deg = dq_deg[idx]
    dq_line_rad = np.deg2rad(dq_line_deg)
    tau_fric_fit = a_hat * dq_line_rad + b_hat * np.sign(dq_line_rad)

    fig, ax = plt.subplots(figsize=(10, 8))  # 🔹 figure 크게

    ax.scatter(dq_deg, tau_fric_only, s=1, alpha=0.3, label="data (tau - J*qdd)")
    ax.plot(dq_line_deg, tau_fric_fit, linewidth=2,
            label="friction model a*dq + b*sign(dq)")

    ax.set_xlabel("joint speed dq (deg/s)", fontsize=12)
    ax.set_ylabel("tau_fric_only (same torque unit)", fontsize=12)
    ax.set_title("friction only: tau - J*qdd vs dq" + title_extra, fontsize=14)
    ax.grid(True)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=10)

    fig.tight_layout()

    if save_fig is not None:
        fig.savefig(save_fig, dpi=800)
        print(f"[full_joint] 관성 제거 마찰 그래프 저장: {save_fig}")

    # plt.show()


def fit_full_joint(t, dq_deg, tau,
                   vmin_deg_s=5.0, smooth_window=1,
                   loss="l2"):
    """
    조인트 전체(모터+벨트+링크 등)에 대해
    tau = J * qdd + a * qd + b * sign(qd) 를 피팅.

    loss:
      - "l2"      : 일반 least squares
      - "l1_cvx"  : cvxpy L1
      - "l1_irls" : IRLS 근사
      - "ransac"  : RANSAC 기반 robust 회귀
    """
    # deg/s → rad/s, rad/s^2
    dq_rad, qdd_rad = compute_vel_accel_rad(t, dq_deg, smooth_window=smooth_window)

    # 너무 느린 구간 제거
    mask = np.abs(dq_deg) >= vmin_deg_s
    dq_used_rad = dq_rad[mask]
    qdd_used_rad = qdd_rad[mask]
    tau_used = tau[mask]

    if dq_used_rad.size < 10:
        raise RuntimeError(f"[full_joint] 유효 샘플이 너무 적습니다: {dq_used_rad.size}개")

    # Φ = [ qdd, dq, sign(dq) ]
    Phi = np.column_stack([qdd_used_rad, dq_used_rad, np.sign(dq_used_rad)])

    # ---- 손실함수별 회귀 ----
    if loss == "l2":
        theta, _, _, _ = np.linalg.lstsq(Phi, tau_used, rcond=None)
    elif loss == "l1_cvx":
        theta = solve_l1_cvxpy(Phi, tau_used, verbose=False)
    elif loss == "l1_irls":
        theta = irls_l1(Phi, tau_used, max_iter=30, eps=1e-6, verbose=False)
    elif loss == "ransac":
        theta = ransac_linear(
            Phi, tau_used,
            n_iter=500,
            min_inlier_ratio=0.3,
            random_state=0,
            verbose=True,
        )
    else:
        raise ValueError(f"Unknown loss: {loss}")

    J_hat, a_hat, b_hat = theta[0], theta[1], theta[2]
    return J_hat, a_hat, b_hat, dq_used_rad, qdd_used_rad, tau_used



def ransac_linear(Phi, y,
                  n_iter=500,
                  sample_size=None,
                  residual_threshold=None,
                  min_inlier_ratio=0.5,
                  random_state=None,
                  verbose=False):
    """
    간단한 RANSAC 선형 회귀:
        tau ≈ Phi @ theta  꼴일 때,
        outlier에 robust하게 theta를 추정.

    파라미터
    --------
    Phi : (N, p) ndarray
        설계행렬 (예: [dq, sign(dq)], [qdd, dq, sign(dq)] 등)
    y : (N,) ndarray
        타깃 벡터 (예: tau, tau_star)
    n_iter : int
        RANSAC 반복 횟수
    sample_size : int or None
        한 번에 뽑을 최소 샘플 수. None이면 p(파라미터 수)로 설정.
    residual_threshold : float or None
        inlier 판정 기준 |residual| <= threshold.
        None이면 초기 LS로 추정한 residual의 median 기반으로 자동 설정.
    min_inlier_ratio : float
        best 모델로 채택하기 위한 최소 inlier 비율 (0~1).
    random_state : int or None
        난수 시드.
    verbose : bool
        True면 inlier 통계 로그 출력.

    반환
    ----
    theta_hat : (p,) ndarray
        RANSAC + inlier 재피팅으로 얻은 파라미터.
    """
    Phi = np.asarray(Phi, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    N, p = Phi.shape

    if sample_size is None:
        sample_size = p  # 파라미터 수만큼만 뽑아서도 LS 가능

    rng = np.random.default_rng(random_state)

    # residual_threshold 자동 설정
    if residual_threshold is None:
        # 전체 LS 한 번 돌려서 scale 추정
        theta_ls, *_ = np.linalg.lstsq(Phi, y, rcond=None)
        resid_ls = y - Phi @ theta_ls
        mad = np.median(np.abs(resid_ls)) + 1e-12  # 0 방지
        residual_threshold = 2.5 * mad  # 적당히 타이트한 기준

    best_theta = None
    best_inliers = 0

    for it in range(n_iter):
        idx_sample = rng.choice(N, size=sample_size, replace=False)
        Phi_s = Phi[idx_sample]
        y_s = y[idx_sample]

        # 최소 샘플로 임시 모델 추정
        theta_tmp, *_ = np.linalg.lstsq(Phi_s, y_s, rcond=None)

        # 전체 residual
        resid = y - Phi @ theta_tmp
        inlier_mask = np.abs(resid) <= residual_threshold
        n_inliers = int(inlier_mask.sum())

        if n_inliers > best_inliers and n_inliers >= min_inlier_ratio * N:
            best_inliers = n_inliers
            # inlier만 모아서 한 번 더 LS (refit)
            Phi_in = Phi[inlier_mask]
            y_in = y[inlier_mask]
            best_theta, *_ = np.linalg.lstsq(Phi_in, y_in, rcond=None)

    if best_theta is None:
        # RANSAC 실패 시 fallback: 전체 LS
        if verbose:
            print("[RANSAC] 유효한 모델을 찾지 못해 일반 LS로 대체합니다.")
        best_theta, *_ = np.linalg.lstsq(Phi, y, rcond=None)
        best_inliers = N

    if verbose:
        print(f"[RANSAC] best_inliers = {best_inliers}/{N}, "
              f"threshold={residual_threshold:.4g}")

    return best_theta


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
    dq_line_deg = dq_used_deg[idx]*0.001
    dq_line_rad = np.deg2rad(dq_line_deg)
    tau_fit = a_hat * dq_line_rad + b_hat * np.sign(dq_line_rad)

    # 🔹 figure 사이즈 키움
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(dq_used_deg, tau_used_star, s=4, alpha=0.3, label="data (tau*)")
    ax.plot(dq_line_deg, tau_fit, linewidth=2, label="fit")

    ax.set_xlabel("joint speed dq (deg/s)", fontsize=12)
    ax.set_ylabel("tau* = tau - J_motor * qdd (same torque unit)", fontsize=12)
    ax.set_title("motor_only: tau* ≈ a*qd + b*sign(qd)" + title_extra, fontsize=14)
    ax.grid(True)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=10)

    fig.tight_layout()

    if save_fig is not None:
        fig.savefig(save_fig, dpi=800)   # 🔹 dpi는 네가 쓰던 800 유지
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

    fig, ax = plt.subplots(figsize=(10, 8))  # 🔹 figure 크게

    ax.scatter(dq_used_deg, tau_used, s=4, alpha=0.3, label="data (tau)")
    ax.plot(dq_line_deg, tau_line, linewidth=2, label="fit")

    ax.set_xlabel("joint speed dq (deg/s)", fontsize=12)
    ax.set_ylabel("torque command (same unit)", fontsize=12)
    ax.set_title("full_joint: tau ≈ J*qdd + a*qd + b*sign(qd)" + title_extra, fontsize=14)
    ax.grid(True)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=10)

    fig.tight_layout()

    if save_fig is not None:
        fig.savefig(save_fig, dpi=800)
        print(f"[full_joint] 그래프 저장: {save_fig}")

    # plt.show()


def solve_l1_cvxpy(Phi, y, verbose=False):
    """
    min ||Phi @ theta - y||_1  을 cvxpy로 직접 푸는 함수.

    Phi : (N, p) numpy array
    y   : (N,)  numpy array

    반환:
        theta_hat : (p,) numpy array
    """
    Phi = np.asarray(Phi, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    N, p = Phi.shape
    theta = cp.Variable(p)

    residual = Phi @ theta - y
    objective = cp.Minimize(cp.norm1(residual))

    prob = cp.Problem(objective)

    # solver는 기본값으로 두거나, ECOS/SCS 명시해도 됨
    prob.solve(verbose=verbose)   # solver=cp.ECOS 같은 것도 가능

    if theta.value is None:
        raise RuntimeError("[cvxpy] 최적해를 찾지 못했습니다 (theta.value is None).")

    theta_hat = np.array(theta.value).reshape(-1)
    return theta_hat


def plot_full_joint_3d(dq_used_rad, qdd_used_rad, tau_used,
                       J_hat, a_hat, b_hat,
                       save_fig=None, title_extra=""):
    """
    full_joint 모드 3D 플롯:
      x축: dq (deg/s)
      y축: qdd (rad/s^2)
      z축: tau

    - 점: 실제 데이터
    - 면: 피팅 tau_hat = J*qdd + a*dq + b*sign(dq)
    """
    dq_deg = np.rad2deg(dq_used_rad)

    fig = plt.figure(figsize=(10, 8))  # 🔹 figure 크게
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

    ax.set_xlabel("dq (deg/s)", fontsize=12)
    ax.set_ylabel("qdd (rad/s^2)", fontsize=12)
    ax.set_zlabel("tau (same unit)", fontsize=12)
    ax.set_title("full_joint 3D: tau ≈ J*qdd + a*dq + b*sign(dq)" + title_extra, fontsize=14)

    fig.tight_layout()

    if save_fig is not None:
        fig.savefig(save_fig, dpi=800)
        print(f"[full_joint] 3D 그래프 저장: {save_fig}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Joint SysID (J, a, b) + plot from DynamicFrictionSysID CSV"
    )
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
                        help="2D 그래프를 저장할 파일 경로(.png 등). 지정 안 하면 저장 안 함.")
    parser.add_argument("--save_fig3d", type=str, default=None,
                        help="full_joint 3D 그래프를 저장할 파일 경로(.png 등). 지정 안 하면 3D 저장 안 함.")
    parser.add_argument("--save_fig_fric", type=str, default=None,
                        help="full_joint에서 관성항 제거한 마찰 그래프를 저장할 경로(.png).")
    parser.add_argument("--loss", type=str,
                        choices=["l2", "l1_cvx", "l1_irls", "ransac"],
                        default="l2",
                        help="회귀 손실: l2(기본), l1_cvx(cvxpy L1), l1_irls(IRLS), ransac")

    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {args.csv}")

    print(f"[SYSID] CSV 로드: {args.csv}")
    t, q_ref, q_meas_deg, dq_meas_deg, tau = load_log(args.csv)
    print(f"[SYSID] 전체 샘플 수: {len(t)}")
    print(f"[SYSID] mode={args.mode}, vmin={args.vmin} deg/s, smooth_window={args.smooth}, loss={args.loss}")

    # ---------------- motor_only 모드 ----------------
    if args.mode == "motor_only":
        if args.J_motor is None:
            raise ValueError("--mode motor_only 를 쓰려면 --J_motor 를 반드시 지정해야 합니다.")

        print(f"[motor_only] J_motor = {args.J_motor:.6e} [토크단위/(rad/s^2)]")

        a_hat, b_hat, dq_used_rad, tau_used_star = fit_motor_only(
            t, dq_meas_deg, tau,
            J_motor=args.J_motor,
            vmin_deg_s=args.vmin,
            smooth_window=args.smooth,
            loss=args.loss,          # 🔹 여기서 loss 전달
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
            title_extra = f"\n(vmin={args.vmin} deg/s, smooth={args.smooth}, loss={args.loss})"
            plot_motor_only(
                dq_used_rad, tau_used_star,
                a_hat, b_hat,
                save_fig=args.save_fig,
                title_extra=title_extra,
            )

    # ---------------- full_joint 모드 ----------------
    elif args.mode == "full_joint":
        J_hat, a_hat, b_hat, dq_used_rad, qdd_used_rad, tau_used = fit_full_joint(
            t, dq_meas_deg, tau,
            vmin_deg_s=args.vmin,
            smooth_window=args.smooth,
            loss=args.loss,          # 🔹 여기서도 loss 전달
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

        # 2D dq–tau 전체 모델
        if args.save_fig is not None:
            title_extra = f"\n(vmin={args.vmin} deg/s, smooth={args.smooth*0.01}, loss={args.loss})"
            plot_full_joint(
                dq_used_rad, qdd_used_rad, tau_used,
                J_hat, a_hat, b_hat,
                save_fig=args.save_fig,
                title_extra=title_extra,
            )

        # 3D dq–qdd–tau
        if args.save_fig3d is not None:
            title_extra = f"\n(vmin={args.vmin} deg/s, smooth={args.smooth*0.01}, loss={args.loss})"
            plot_full_joint_3d(
                dq_used_rad, qdd_used_rad, tau_used,
                J_hat, a_hat, b_hat,
                save_fig=args.save_fig3d,
                title_extra=title_extra,
            )

        # 관성항 제거한 마찰만 (tau - J*qdd vs dq)
        if args.save_fig_fric is not None:
            title_extra = f"\n(vmin={args.vmin} deg/s, smooth={args.smooth*0.01}, loss={args.loss})"
            plot_full_joint_fric_only(
                dq_used_rad, qdd_used_rad, tau_used,
                J_hat, a_hat, b_hat,
                save_fig=args.save_fig_fric,
                title_extra=title_extra,
            )

    else:
        raise ValueError(f"알 수 없는 mode: {args.mode}")


if __name__ == "__main__":
    main()
