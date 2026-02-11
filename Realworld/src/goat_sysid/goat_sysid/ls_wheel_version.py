#!/usr/bin/env python3
"""
ls_wheel_version.py

Wheel SysID CSV format:
  t_sec, wheel_index, omega_ref_rad_s, omega_meas_rad_s, tau_cmd_nm

Modes
-----
- motor_only:
    tau* = tau - J_motor * omega_dot
    tau* = a * omega + b * sign(omega)
- full_wheel:
    tau  = J * omega_dot + a * omega + b * sign(omega)

Display
-------
- 기본: 화면에 그래프 창 띄움 (plt.show)
- 저장도 원하면 --save_fig / --save_fig3d / --save_fig_fric 사용
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

try:
    import cvxpy as cp
    _HAS_CVXPY = True
except Exception:
    _HAS_CVXPY = False


def load_wheel_log(csv_path: str, wheel_index: int | None = None):
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")

    required = ["t_sec", "wheel_index", "omega_ref_rad_s", "omega_meas_rad_s", "tau_cmd_nm"]
    for k in required:
        if k not in data.dtype.names:
            raise ValueError(f"CSV 컬럼 '{k}'가 없습니다. 실제 컬럼: {data.dtype.names}")

    t = np.asarray(data["t_sec"], dtype=float)
    w_idx = np.asarray(data["wheel_index"], dtype=float).astype(int)
    omega_ref = np.asarray(data["omega_ref_rad_s"], dtype=float)
    omega_meas = np.asarray(data["omega_meas_rad_s"], dtype=float)
    tau = np.asarray(data["tau_cmd_nm"], dtype=float)

    uniq = np.unique(w_idx)
    if wheel_index is None:
        wheel_index = int(uniq[0])
        if len(uniq) > 1:
            print(f"[load] wheel_index가 여러 개: {uniq}. 자동으로 {wheel_index} 사용 (--wheel_index로 지정 가능)")
    else:
        wheel_index = int(wheel_index)

    mask = (w_idx == wheel_index)
    if mask.sum() == 0:
        raise RuntimeError(f"[load] wheel_index={wheel_index} 데이터가 0개. CSV wheel_index: {uniq}")

    return t[mask], wheel_index, omega_ref[mask], omega_meas[mask], tau[mask]


def compute_omega_and_accel(t, omega_rad_s, smooth_window: int = 1):
    omega = np.asarray(omega_rad_s, dtype=float)

    if smooth_window > 1:
        kernel = np.ones(smooth_window, dtype=float) / smooth_window
        omega_f = np.convolve(omega, kernel, mode="same")
    else:
        omega_f = omega

    omega_dot = np.gradient(omega_f, t)
    return omega_f, omega_dot


def solve_l1_cvxpy(Phi, y, verbose=False):
    if not _HAS_CVXPY:
        raise RuntimeError("cvxpy 미설치. `pip install cvxpy ecos scs` 후 --loss l1_cvx 사용")

    Phi = np.asarray(Phi, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    p = Phi.shape[1]
    theta = cp.Variable(p)
    prob = cp.Problem(cp.Minimize(cp.norm1(Phi @ theta - y)))
    prob.solve(verbose=verbose)

    if theta.value is None:
        raise RuntimeError("[cvxpy] 최적해 실패(theta.value None)")
    return np.array(theta.value).reshape(-1)


def irls_l1(Phi, y, max_iter=30, eps=1e-6):
    Phi = np.asarray(Phi, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    theta, *_ = np.linalg.lstsq(Phi, y, rcond=None)

    for _ in range(max_iter):
        r = y - Phi @ theta
        w = 1.0 / (np.abs(r) + eps)
        sw = np.sqrt(w)
        Phi_w = Phi * sw[:, None]
        y_w = y * sw
        theta_new, *_ = np.linalg.lstsq(Phi_w, y_w, rcond=None)

        if np.linalg.norm(theta_new - theta) < 1e-10:
            theta = theta_new
            break
        theta = theta_new

    return theta


def ransac_linear(Phi, y, n_iter=500, sample_size=None, residual_threshold=None,
                  min_inlier_ratio=0.3, random_state=0, verbose=True):
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
        theta_tmp, *_ = np.linalg.lstsq(Phi[idx], y[idx], rcond=None)

        resid = y - Phi @ theta_tmp
        inlier_mask = np.abs(resid) <= residual_threshold
        n_in = int(inlier_mask.sum())

        if n_in > best_inliers and n_in >= int(min_inlier_ratio * N):
            best_inliers = n_in
            best_theta, *_ = np.linalg.lstsq(Phi[inlier_mask], y[inlier_mask], rcond=None)

    if best_theta is None:
        if verbose:
            print("[RANSAC] 실패 -> 전체 LS로 대체")
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
        return solve_l1_cvxpy(Phi, y)
    if loss == "l1_irls":
        return irls_l1(Phi, y)
    if loss == "ransac":
        return ransac_linear(Phi, y)
    raise ValueError(f"Unknown loss: {loss}")


def fit_motor_only(t, omega_meas_rad_s, tau, J_motor, vmin_rad_s, smooth_window, loss):
    omega_f, omega_dot = compute_omega_and_accel(t, omega_meas_rad_s, smooth_window)
    tau_star = tau - J_motor * omega_dot

    mask = np.abs(omega_f) >= vmin_rad_s
    omega_u = omega_f[mask]
    omega_dot_u = omega_dot[mask]
    tau_u = tau_star[mask]

    if omega_u.size < 10:
        raise RuntimeError(f"[motor_only] 유효 샘플 10개 미만: {omega_u.size} (vmin={vmin_rad_s:.4g} rad/s)")

    Phi = np.column_stack([omega_u, np.sign(omega_u)])
    a_hat, b_hat = solve_by_loss(Phi, tau_u, loss)
    return a_hat, b_hat, omega_u, omega_dot_u, tau_u


def fit_full_wheel(t, omega_meas_rad_s, tau, vmin_rad_s, smooth_window, loss):
    omega_f, omega_dot = compute_omega_and_accel(t, omega_meas_rad_s, smooth_window)

    mask = np.abs(omega_f) >= vmin_rad_s
    omega_u = omega_f[mask]
    omega_dot_u = omega_dot[mask]
    tau_u = tau[mask]

    if omega_u.size < 10:
        raise RuntimeError(f"[full_wheel] 유효 샘플 10개 미만: {omega_u.size} (vmin={vmin_rad_s:.4g} rad/s)")

    Phi = np.column_stack([omega_dot_u, omega_u, np.sign(omega_u)])
    J_hat, a_hat, b_hat = solve_by_loss(Phi, tau_u, loss)
    return J_hat, a_hat, b_hat, omega_u, omega_dot_u, tau_u


def plot_2d(omega_u, y_data, y_fit, title, save_fig=None):
    omega_deg = np.rad2deg(omega_u)
    idx = np.argsort(omega_deg)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(omega_deg, y_data, s=4, alpha=0.3, label="data")
    ax.plot(omega_deg[idx], y_fit[idx], linewidth=2, label="fit")
    ax.set_xlabel("omega (deg/s)")
    ax.set_ylabel("tau (N*m)")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    if save_fig:
        fig.savefig(save_fig, dpi=300)
        print(f"[save] {save_fig}")

    return fig


def plot_3d(omega_u, omega_dot_u, tau_u, J_hat, a_hat, b_hat, save_fig3d=None):
    omega_deg = np.rad2deg(omega_u)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(omega_deg, omega_dot_u, tau_u, s=4, alpha=0.3)

    o_min, o_max = omega_deg.min(), omega_deg.max()
    od_min, od_max = omega_dot_u.min(), omega_dot_u.max()

    og, odg = np.meshgrid(np.linspace(o_min, o_max, 30),
                          np.linspace(od_min, od_max, 30))
    og_rad = np.deg2rad(og)
    tg = J_hat * odg + a_hat * og_rad + b_hat * np.sign(og_rad)

    ax.plot_surface(og, odg, tg, alpha=0.35)
    ax.set_xlabel("omega (deg/s)")
    ax.set_ylabel("omega_dot (rad/s^2)")
    ax.set_zlabel("tau (N*m)")
    ax.set_title("3D: tau ≈ J*omega_dot + a*omega + b*sign(omega)")
    fig.tight_layout()

    if save_fig3d:
        fig.savefig(save_fig3d, dpi=300)
        print(f"[save] {save_fig3d}")

    return fig


def main():
    parser = argparse.ArgumentParser(description="Wheel SysID + interactive plot viewer")

    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--wheel_index", type=int, default=None)

    parser.add_argument("--mode", choices=["motor_only", "full_wheel"], default="full_wheel")

    # vmin: deg/s 입력을 기본으로 제공
    parser.add_argument("--vmin_deg_s", type=float, default=2.0)
    parser.add_argument("--vmin_rad_s", type=float, default=None,
                        help="이걸 주면 vmin_deg_s 무시")

    parser.add_argument("--smooth", type=int, default=21)
    parser.add_argument("--J_motor", type=float, default=None)

    parser.add_argument("--loss", choices=["l2", "l1_cvx", "l1_irls", "ransac"], default="l2")

    # 저장 옵션 (원하면만)
    parser.add_argument("--save_fig", type=str, default=None)
    parser.add_argument("--save_fig3d", type=str, default=None)
    parser.add_argument("--save_fig_fric", type=str, default=None)

    # ✅ 화면 표시 옵션
    parser.add_argument("--show", action="store_true",
                        help="그래프 창을 띄워서 보기 (기본 권장)")
    parser.add_argument("--no_show", action="store_true",
                        help="그래프 창을 띄우지 않음(저장만 할 때)")

    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(args.csv)

    t, widx, omega_ref, omega_meas, tau = load_wheel_log(args.csv, args.wheel_index)
    print(f"[SYSID] CSV={args.csv}")
    print(f"[SYSID] wheel_index={widx}, samples={len(t)}")
    print(f"[SYSID] mode={args.mode}, smooth={args.smooth}, loss={args.loss}")

    if args.vmin_rad_s is not None:
        vmin_rad_s = float(args.vmin_rad_s)
    else:
        vmin_rad_s = np.deg2rad(float(args.vmin_deg_s))
    print(f"[SYSID] vmin={np.rad2deg(vmin_rad_s):.3g} deg/s ({vmin_rad_s:.3g} rad/s)")

    figs = []

    if args.mode == "motor_only":
        if args.J_motor is None:
            raise ValueError("--mode motor_only이면 --J_motor 필수")
        a_hat, b_hat, omega_u, omega_dot_u, tau_star_u = fit_motor_only(
            t, omega_meas, tau,
            J_motor=float(args.J_motor),
            vmin_rad_s=vmin_rad_s,
            smooth_window=args.smooth,
            loss=args.loss
        )
        print("\n===== motor_only =====")
        print(f"a_hat ≈ {a_hat:.6e} [N*m/(rad/s)]")
        print(f"b_hat ≈ {b_hat:.6e} [N*m]")
        print(f"used samples = {omega_u.size}")

        tau_fit = a_hat * omega_u + b_hat * np.sign(omega_u)
        figs.append(plot_2d(
            omega_u, tau_star_u, tau_fit,
            "motor_only: tau* ≈ a*omega + b*sign(omega)",
            save_fig=args.save_fig
        ))

    else:  # full_wheel
        J_hat, a_hat, b_hat, omega_u, omega_dot_u, tau_u = fit_full_wheel(
            t, omega_meas, tau,
            vmin_rad_s=vmin_rad_s,
            smooth_window=args.smooth,
            loss=args.loss
        )
        print("\n===== full_wheel =====")
        print(f"J_hat ≈ {J_hat:.6e} [N*m/(rad/s^2)]")
        print(f"a_hat ≈ {a_hat:.6e} [N*m/(rad/s)]")
        print(f"b_hat ≈ {b_hat:.6e} [N*m]")
        print(f"used samples = {omega_u.size}")

        tau_hat = J_hat * omega_dot_u + a_hat * omega_u + b_hat * np.sign(omega_u)
        figs.append(plot_2d(
            omega_u, tau_u, tau_hat,
            "full_wheel: tau ≈ J*omega_dot + a*omega + b*sign(omega)",
            save_fig=args.save_fig
        ))

        if args.save_fig3d is not None or args.show:
            figs.append(plot_3d(
                omega_u, omega_dot_u, tau_u,
                J_hat, a_hat, b_hat,
                save_fig3d=args.save_fig3d
            ))

        if args.save_fig_fric is not None or args.show:
            tau_fric = tau_u - J_hat * omega_dot_u
            tau_fric_fit = a_hat * omega_u + b_hat * np.sign(omega_u)
            figs.append(plot_2d(
                omega_u, tau_fric, tau_fric_fit,
                "friction only: (tau - J*omega_dot) vs omega",
                save_fig=args.save_fig_fric
            ))

    # ✅ 화면 표시
    if (args.show and not args.no_show) or (not args.no_show and not (args.save_fig or args.save_fig3d or args.save_fig_fric)):
        plt.show()


if __name__ == "__main__":
    main()
