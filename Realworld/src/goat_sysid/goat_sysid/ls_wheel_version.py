#!/usr/bin/env python3
"""
ls_wheel_version.py

Wheel SysID CSV (format):
  t_sec, wheel_index, omega_ref_rad_s, omega_meas_rad_s, tau_cmd_nm

Modes
-----
1) motor_only:
   - Assume known motor inertia J_motor
   - tau* = tau - J_motor * omega_dot
   - tau* = a * omega + b * sign(omega)
   - Plot: omega (deg/s) vs tau* (data + fit)

2) full_wheel:
   - Fit tau = J * omega_dot + a * omega + b * sign(omega)
   - Plot: omega (deg/s) vs tau (data + fit)
   - Optional plots:
       * friction-only: omega vs (tau - J*omega_dot)
       * 3D: omega (deg/s), omega_dot (rad/s^2), tau
"""

'''
python3 /home/heachanlee/GOAT/GOAT/Realworld/src/goat_sysid/goat_sysid/ls_wheel_version.py \
  --csv /home/heachanlee/GOAT/GOAT/Realworld/src/goat_sysid/goat_sysid/wheel_sysid_w6_20260208_213837.csv \
  --mode full_wheel \
  --vmin_deg_s 0.3 \
  --smooth 21 \
  --loss l2 \
  --save_fig wheel_full_2d.png \
  --save_fig3d wheel_full_3d.png \
  --save_fig_fric wheel_full_fric_only.png

'''


import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

try:
    import cvxpy as cp
    _HAS_CVXPY = True
except Exception:
    _HAS_CVXPY = False


# -----------------------
# IO
# -----------------------
def load_wheel_log(csv_path: str, wheel_index: int | None = None):
    """
    Read wheel sysid CSV.

    Required columns:
      - t_sec
      - wheel_index
      - omega_ref_rad_s
      - omega_meas_rad_s
      - tau_cmd_nm
    """
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")

    required = ["t_sec", "wheel_index", "omega_ref_rad_s", "omega_meas_rad_s", "tau_cmd_nm"]
    for k in required:
        if k not in data.dtype.names:
            raise ValueError(f"CSV 컬럼 '{k}'가 없습니다. 실제 컬럼: {data.dtype.names}")

    t = np.asarray(data["t_sec"], dtype=float)
    w_idx = np.asarray(data["wheel_index"], dtype=float)
    omega_ref = np.asarray(data["omega_ref_rad_s"], dtype=float)
    omega_meas = np.asarray(data["omega_meas_rad_s"], dtype=float)
    tau = np.asarray(data["tau_cmd_nm"], dtype=float)

    # wheel_index 필터
    uniq = np.unique(w_idx.astype(int))
    if wheel_index is None:
        # 여러 개가 있으면 첫 번째를 자동 선택
        wheel_index = int(uniq[0])
        if len(uniq) > 1:
            print(f"[load] wheel_index가 여러 개 있습니다: {uniq}. "
                  f"자동으로 {wheel_index}를 사용합니다. (--wheel_index로 지정 가능)")
    else:
        wheel_index = int(wheel_index)

    mask = (w_idx.astype(int) == wheel_index)
    t = t[mask]
    omega_ref = omega_ref[mask]
    omega_meas = omega_meas[mask]
    tau = tau[mask]

    if t.size == 0:
        raise RuntimeError(f"[load] wheel_index={wheel_index}에 해당하는 데이터가 0개입니다. "
                           f"CSV에 있는 wheel_index: {uniq}")

    return t, wheel_index, omega_ref, omega_meas, tau


# -----------------------
# Signal processing
# -----------------------
def compute_omega_and_accel(t, omega_rad_s, smooth_window: int = 1):
    """
    omega_rad_s -> (optional moving average) -> omega_dot via np.gradient(t)

    Return:
      omega_filt (rad/s)
      omega_dot  (rad/s^2)
    """
    omega = np.asarray(omega_rad_s, dtype=float)

    if smooth_window > 1:
        kernel = np.ones(smooth_window, dtype=float) / smooth_window
        omega_f = np.convolve(omega, kernel, mode="same")
    else:
        omega_f = omega

    omega_dot = np.gradient(omega_f, t)
    return omega_f, omega_dot


# -----------------------
# Robust solvers
# -----------------------
def solve_l1_cvxpy(Phi, y, verbose=False):
    if not _HAS_CVXPY:
        raise RuntimeError("cvxpy가 설치되어 있지 않습니다. `pip install cvxpy ecos scs` 후 --loss l1_cvx 사용하세요.")

    Phi = np.asarray(Phi, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    n, p = Phi.shape
    theta = cp.Variable(p)
    obj = cp.Minimize(cp.norm1(Phi @ theta - y))
    prob = cp.Problem(obj)
    prob.solve(verbose=verbose)

    if theta.value is None:
        raise RuntimeError("[cvxpy] 최적해를 찾지 못했습니다(theta.value is None).")

    return np.array(theta.value).reshape(-1)


def irls_l1(Phi, y, max_iter=30, eps=1e-6, verbose=False):
    """
    L1 근사 IRLS:
      반복적으로 w_i = 1/(|r_i|+eps) 로 두고 가중 least squares 수행
    """
    Phi = np.asarray(Phi, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    theta, *_ = np.linalg.lstsq(Phi, y, rcond=None)

    for it in range(max_iter):
        r = y - Phi @ theta
        w = 1.0 / (np.abs(r) + eps)  # 큰 residual에 작은 가중
        sqrtw = np.sqrt(w)

        Phi_w = Phi * sqrtw[:, None]
        y_w = y * sqrtw

        theta_new, *_ = np.linalg.lstsq(Phi_w, y_w, rcond=None)

        if np.linalg.norm(theta_new - theta) < 1e-10:
            theta = theta_new
            break

        theta = theta_new

        if verbose:
            print(f"[IRLS] iter={it+1}, |dtheta|={np.linalg.norm(theta_new-theta):.3e}")

    return theta


def ransac_linear(Phi, y,
                  n_iter=500,
                  sample_size=None,
                  residual_threshold=None,
                  min_inlier_ratio=0.5,
                  random_state=0,
                  verbose=False):
    """
    Simple RANSAC for linear regression y ~= Phi @ theta
    """
    Phi = np.asarray(Phi, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    N, p = Phi.shape

    if sample_size is None:
        sample_size = p

    rng = np.random.default_rng(random_state)

    if residual_threshold is None:
        theta_ls, *_ = np.linalg.lstsq(Phi, y, rcond=None)
        resid_ls = y - Phi @ theta_ls
        mad = np.median(np.abs(resid_ls)) + 1e-12
        residual_threshold = 2.5 * mad

    best_theta = None
    best_inliers = 0

    for _ in range(n_iter):
        idx = rng.choice(N, size=sample_size, replace=False)
        Phi_s = Phi[idx]
        y_s = y[idx]

        theta_tmp, *_ = np.linalg.lstsq(Phi_s, y_s, rcond=None)

        resid = y - Phi @ theta_tmp
        inlier_mask = np.abs(resid) <= residual_threshold
        n_in = int(inlier_mask.sum())

        if n_in > best_inliers and n_in >= int(min_inlier_ratio * N):
            best_inliers = n_in
            Phi_in = Phi[inlier_mask]
            y_in = y[inlier_mask]
            best_theta, *_ = np.linalg.lstsq(Phi_in, y_in, rcond=None)

    if best_theta is None:
        if verbose:
            print("[RANSAC] 유효 모델 실패 -> 전체 LS로 대체")
        best_theta, *_ = np.linalg.lstsq(Phi, y, rcond=None)
        best_inliers = N

    if verbose:
        print(f"[RANSAC] best_inliers={best_inliers}/{N}, threshold={residual_threshold:.4g}")

    return best_theta


def solve_by_loss(Phi, y, loss: str):
    if loss == "l2":
        theta, *_ = np.linalg.lstsq(Phi, y, rcond=None)
        return theta
    if loss == "l1_cvx":
        return solve_l1_cvxpy(Phi, y, verbose=False)
    if loss == "l1_irls":
        return irls_l1(Phi, y, max_iter=30, eps=1e-6, verbose=False)
    if loss == "ransac":
        return ransac_linear(Phi, y, n_iter=500, min_inlier_ratio=0.3, random_state=0, verbose=True)
    raise ValueError(f"Unknown loss: {loss}")


# -----------------------
# Fitters
# -----------------------
def fit_motor_only(t, omega_meas_rad_s, tau, J_motor,
                   vmin_rad_s=0.1, smooth_window=1, loss="l2"):
    omega_f, omega_dot = compute_omega_and_accel(t, omega_meas_rad_s, smooth_window=smooth_window)
    tau_star = tau - J_motor * omega_dot

    mask = np.abs(omega_f) >= vmin_rad_s
    omega_used = omega_f[mask]
    tau_used = tau_star[mask]

    if omega_used.size < 10:
        raise RuntimeError(f"[motor_only] 유효 샘플이 너무 적습니다: {omega_used.size}개 (vmin={vmin_rad_s:.4g} rad/s)")

    Phi = np.column_stack([omega_used, np.sign(omega_used)])
    theta = solve_by_loss(Phi, tau_used, loss)
    a_hat, b_hat = theta[0], theta[1]
    return a_hat, b_hat, omega_used, omega_dot[mask], tau_used


def fit_full_wheel(t, omega_meas_rad_s, tau,
                  vmin_rad_s=0.1, smooth_window=1, loss="l2"):
    omega_f, omega_dot = compute_omega_and_accel(t, omega_meas_rad_s, smooth_window=smooth_window)

    mask = np.abs(omega_f) >= vmin_rad_s
    omega_used = omega_f[mask]
    omega_dot_used = omega_dot[mask]
    tau_used = tau[mask]

    if omega_used.size < 10:
        raise RuntimeError(f"[full_wheel] 유효 샘플이 너무 적습니다: {omega_used.size}개 (vmin={vmin_rad_s:.4g} rad/s)")

    Phi = np.column_stack([omega_dot_used, omega_used, np.sign(omega_used)])
    theta = solve_by_loss(Phi, tau_used, loss)
    J_hat, a_hat, b_hat = theta[0], theta[1], theta[2]
    return J_hat, a_hat, b_hat, omega_used, omega_dot_used, tau_used


# -----------------------
# Plots
# -----------------------
def plot_2d(omega_used_rad_s, y_data, y_fit, title, save_fig=None):
    omega_deg_s = np.rad2deg(omega_used_rad_s)

    idx = np.argsort(omega_deg_s)
    x_line = omega_deg_s[idx]
    y_line = y_fit[idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(omega_deg_s, y_data, s=4, alpha=0.3, label="data")
    ax.plot(x_line, y_line, linewidth=2, label="fit")

    ax.set_xlabel("wheel speed omega (deg/s)", fontsize=12)
    ax.set_ylabel("torque (N*m)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True)
    ax.legend(fontsize=10)
    fig.tight_layout()

    if save_fig is not None:
        fig.savefig(save_fig, dpi=800)
        print(f"[save] {save_fig}")


def plot_full_wheel_3d(omega_used_rad_s, omega_dot_used, tau_used,
                       J_hat, a_hat, b_hat, save_fig3d=None, title_extra=""):
    omega_deg_s = np.rad2deg(omega_used_rad_s)
    tau_hat = J_hat * omega_dot_used + a_hat * omega_used_rad_s + b_hat * np.sign(omega_used_rad_s)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(omega_deg_s, omega_dot_used, tau_used, s=4, alpha=0.3)

    # surface
    o_min, o_max = omega_deg_s.min(), omega_deg_s.max()
    od_min, od_max = omega_dot_used.min(), omega_dot_used.max()

    og, odg = np.meshgrid(np.linspace(o_min, o_max, 30),
                          np.linspace(od_min, od_max, 30))
    og_rad = np.deg2rad(og)
    tg = J_hat * odg + a_hat * og_rad + b_hat * np.sign(og_rad)

    ax.plot_surface(og, odg, tg, alpha=0.4)

    ax.set_xlabel("omega (deg/s)")
    ax.set_ylabel("omega_dot (rad/s^2)")
    ax.set_zlabel("tau (N*m)")
    ax.set_title("full_wheel 3D: tau ≈ J*omega_dot + a*omega + b*sign(omega)" + title_extra)

    fig.tight_layout()
    if save_fig3d is not None:
        fig.savefig(save_fig3d, dpi=800)
        print(f"[save] {save_fig3d}")


# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Wheel SysID (J, a, b) from wheel CSV + plots")

    parser.add_argument("--csv", type=str, required=True, help="wheel sysid CSV path")
    parser.add_argument("--wheel_index", type=int, default=None,
                        help="CSV에 wheel_index가 여러개면 선택. 없으면 자동 선택")

    parser.add_argument("--mode", type=str, choices=["motor_only", "full_wheel"],
                        default="full_wheel", help="fit mode")

    # vmin: deg/s 기반 입력을 기본으로 제공(너가 이전에 쓰던 스타일)
    parser.add_argument("--vmin_deg_s", type=float, default=2.0,
                        help="|omega| < vmin 샘플 제거 (deg/s). 기본 2.0")
    parser.add_argument("--vmin_rad_s", type=float, default=None,
                        help="|omega| < vmin 샘플 제거 (rad/s). 지정하면 vmin_deg_s 무시")

    parser.add_argument("--smooth", type=int, default=21,
                        help="moving average window (samples). 1이면 스무딩 없음")

    parser.add_argument("--J_motor", type=float, default=None,
                        help="motor_only 모드에서 사용할 J_motor [N*m / (rad/s^2)]")

    parser.add_argument("--loss", type=str, choices=["l2", "l1_cvx", "l1_irls", "ransac"],
                        default="l2", help="regression loss")

    parser.add_argument("--save_fig", type=str, default=None, help="2D plot output (.png)")
    parser.add_argument("--save_fig3d", type=str, default=None, help="3D plot output (.png)")
    parser.add_argument("--save_fig_fric", type=str, default=None,
                        help="full_wheel에서 (tau - J*omega_dot) vs omega 저장")

    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    t, widx, omega_ref, omega_meas, tau = load_wheel_log(args.csv, wheel_index=args.wheel_index)
    print(f"[SYSID] CSV 로드: {args.csv}")
    print(f"[SYSID] wheel_index={widx}, samples={len(t)}")
    print(f"[SYSID] mode={args.mode}, smooth={args.smooth}, loss={args.loss}")

    if args.vmin_rad_s is not None:
        vmin_rad_s = float(args.vmin_rad_s)
        vmin_deg_s = np.rad2deg(vmin_rad_s)
    else:
        vmin_deg_s = float(args.vmin_deg_s)
        vmin_rad_s = np.deg2rad(vmin_deg_s)

    print(f"[SYSID] vmin = {vmin_deg_s:.3g} deg/s ({vmin_rad_s:.3g} rad/s)")

    if args.mode == "motor_only":
        if args.J_motor is None:
            raise ValueError("--mode motor_only 사용 시 --J_motor 가 필요합니다.")
        a_hat, b_hat, omega_used, omega_dot_used, tau_star_used = fit_motor_only(
            t, omega_meas, tau,
            J_motor=args.J_motor,
            vmin_rad_s=vmin_rad_s,
            smooth_window=args.smooth,
            loss=args.loss
        )
        print("\n===== motor_only 결과 =====")
        print(f"a_hat (viscous) ≈ {a_hat:.6e} [N*m / (rad/s)]")
        print(f"b_hat (coulomb) ≈ {b_hat:.6e} [N*m]")
        print(f"used samples = {omega_used.size}")

        # plot
        if args.save_fig:
            # y_fit
            tau_fit = a_hat * omega_used + b_hat * np.sign(omega_used)
            plot_2d(
                omega_used, tau_star_used, tau_fit,
                title="motor_only: tau* ≈ a*omega + b*sign(omega)",
                save_fig=args.save_fig
            )

    elif args.mode == "full_wheel":
        J_hat, a_hat, b_hat, omega_used, omega_dot_used, tau_used = fit_full_wheel(
            t, omega_meas, tau,
            vmin_rad_s=vmin_rad_s,
            smooth_window=args.smooth,
            loss=args.loss
        )
        print("\n===== full_wheel 결과 =====")
        print(f"J_hat (equiv inertia) ≈ {J_hat:.6e} [N*m / (rad/s^2)]")
        print(f"a_hat (viscous)       ≈ {a_hat:.6e} [N*m / (rad/s)]")
        print(f"b_hat (coulomb)       ≈ {b_hat:.6e} [N*m]")
        print(f"used samples = {omega_used.size}")

        # 2D: omega vs tau
        if args.save_fig:
            tau_hat = J_hat * omega_dot_used + a_hat * omega_used + b_hat * np.sign(omega_used)
            plot_2d(
                omega_used, tau_used, tau_hat,
                title="full_wheel: tau ≈ J*omega_dot + a*omega + b*sign(omega)",
                save_fig=args.save_fig
            )

        # 3D
        if args.save_fig3d:
            plot_full_wheel_3d(
                omega_used, omega_dot_used, tau_used,
                J_hat, a_hat, b_hat,
                save_fig3d=args.save_fig3d
            )

        # friction-only: tau - J*omega_dot vs omega
        if args.save_fig_fric:
            tau_fric = tau_used - J_hat * omega_dot_used
            tau_fric_fit = a_hat * omega_used + b_hat * np.sign(omega_used)
            plot_2d(
                omega_used, tau_fric, tau_fric_fit,
                title="friction only: (tau - J*omega_dot) vs omega",
                save_fig=args.save_fig_fric
            )

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
