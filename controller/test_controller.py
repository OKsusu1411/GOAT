import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on inverse dynamics control for an articulation.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--test_mode", type=str, default="withstand", choices=["withstand", "tracking", "plotting"])
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, Articulation, AssetBaseCfg
from isaaclab.utils import configclass
from lib.env.GOAT_base_env_cfg import GOATBaseEnvCfg


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Design the scene for inverse dynamics control."""
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    robot = GOATBaseEnvCfg.GOAT_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot",
                                             init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "hip_L_Joint": 0.0,
                "hip_R_Joint": 0.0,
                "thigh_L_Joint": 0.0,
                "thigh_R_Joint": 0.0,
                "knee_L_Joint": 0.0,
                "knee_R_Joint": 0.0,
                "wheel_L_Joint": 0.0,
                "wheel_R_Joint": 0.0,
            },
        ),
    )

    # Actuator's PD gain
    # set to 0 when using external low-level torque controller
    robot.actuators["leg"].stiffness = 0.0
    robot.actuators["leg"].damping = 0.0
    robot.actuators["wheel"].stiffness = 0.0
    robot.actuators["wheel"].damping = 0.0


class InverseDynamicsController:
    """
    PD 제어 기반의 역역학 토크 컨트롤러입니다.
    
    이 컨트롤러는 로봇의 동역학 모델을 사용하여 목표 조인트 가속도를 달성하기 위한
    피드포워드 토크를 계산합니다.
    
    τ = M(q) * (q̈_des + Kp*e + Kd*ė) + C(q, q̇) + G(q)
    """
    def __init__(self, kp: float, kd: float, num_envs: int, num_dof: int, device: str):
        """
        컨트롤러를 초기화합니다.

        Args:
            kp (float): 비례 이득 (Proportional gain).
            kd (float): 미분 이득 (Derivative gain).
            num_envs (int): 시뮬레이션 환경의 수.
            num_dof (int): 제어할 관절의 수 (degrees of freedom).
            device (str): 연산에 사용할 디바이스 (e.g., "cuda:0" or "cpu").
        """
        self.kp = torch.full((num_envs, num_dof), kp, device=device)
        self.kd = torch.full((num_envs, num_dof), kd, device=device)
        self.num_envs = num_envs
        self.num_dof = num_dof
        self.device = device

    def compute(
        self,
        dof_pos: torch.Tensor,
        dof_vel: torch.Tensor,
        dof_pos_des: torch.Tensor,
        dof_vel_des: torch.Tensor,
        dof_acc_des: torch.Tensor,
        mass_matrix: torch.Tensor,
        coriolis_term: torch.Tensor,
        gravity_term: torch.Tensor,
    ) -> torch.Tensor:
        """
        역역학에 기반한 제어 토크를 계산합니다.

        Args:
            dof_pos (torch.Tensor): 현재 조인트 각도 [rad].
            dof_vel (torch.Tensor): 현재 조인트 각속도 [rad/s].
            dof_pos_des (torch.Tensor): 목표 조인트 각도 [rad].
            dof_vel_des (torch.Tensor): 목표 조인트 각속도 [rad/s].
            dof_acc_des (torch.Tensor): 목표 피드포워드 조인트 각가속도 [rad/s^2].
            mass_matrix (torch.Tensor): 일반화된 질량 행렬 (M).
            coriolis_term (torch.Tensor): 코리올리 및 원심력 벡터 (C).
            gravity_term (torch.Tensor): 중력 보상 토크 벡터 (G).

        Returns:
            torch.Tensor: 계산된 조인트 토크.
        """
        # 1. 위치 및 속도 오차 계산
        pos_error = dof_pos_des - dof_pos
        vel_error = dof_vel_des - dof_vel

        # 2. PD 제어 법칙을 이용한 목표 가속도 계산
        # q̈_des = q̈_target + Kp * e + Kd * ė
        desired_acc = dof_acc_des + self.kp * pos_error + self.kd * vel_error

        # 3. 역역학 방정식 계산
        # τ = M(q) * q̈_des + C(q, q̇) + G(q)
        # unsqueeze와 squeeze는 행렬-벡터 곱셈을 위한 차원 맞추기입니다.
        torque = (
            torch.bmm(mass_matrix, desired_acc.unsqueeze(-1)).squeeze(-1)
            + coriolis_term
            + gravity_term
        )
        
        return torque


def run_simulator(sim: sim_utils.SimulationContext, env_cfg: GOATBaseEnvCfg):
    # define scene
    robot = Articulation(env_cfg.GOAT_cfg)
    env_cfg.scene.articulations["robot"] = robot
    scene = InteractiveScene(cfg=env_cfg.scene, sim=sim)
    sim_dt = sim.get_physics_dt()

    # Per one leg
    n_j_per_leg = 3
    base_dof = 6  # for floating base
    num_total_joints = robot.num_dof

    # Define joint indices for each leg
    # Assuming interleaved joint order: [hip_L, hip_R, thigh_L, thigh_R, knee_L, knee_R, ...]
    left_leg_indices = torch.tensor([0, 2, 4], device=sim.device, dtype=torch.long)
    right_leg_indices = torch.tensor([1, 3, 5], device=sim.device, dtype=torch.long)

    # Corresponding indices in the dynamics tensors (with floating base offset)
    left_leg_dyn_indices = left_leg_indices + base_dof
    right_leg_dyn_indices = right_leg_indices + base_dof

    # --- PD 역역학 컨트롤러 초기화 ---
    # Create separate controllers for each leg for independent control
    left_leg_controller = InverseDynamicsController(
        kp=100.0, kd=20.0, num_envs=scene.num_envs, num_dof=n_j_per_leg, device=scene.device
    )
    right_leg_controller = InverseDynamicsController(
        kp=100.0, kd=20.0, num_envs=scene.num_envs, num_dof=n_j_per_leg, device=scene.device
    )

    # ---------- 환경 준비 ----------
    sim_len = 2.0  # [s] 실험 길이
    joint_limits = robot.data.joint_pos_limits

    # ---------- 초기값 및 목표값 설정 ----------
    zero_joint_efforts = torch.zeros(scene.num_envs, num_total_joints, device=sim.device)
    q_init = robot.data.default_joint_pos.clone()
    q_target = q_init[:, : (n_j_per_leg * 2)].clone()  # 초기 자세를 목표 자세로 설정
    q_dot_target = torch.zeros_like(q_target)
    q_ddot_target = torch.zeros_like(q_target)

    robot.update(sim_dt)

    # --- 제어 로직 루프 --------------------------
    if args_cli.test_mode == "withstand":
        count = 0
        while simulation_app.is_running():
            if count % 500 == 0:
                # reset joint state to default
                print("[INFO] Reset state ...")
                default_joint_pos = robot.data.default_joint_pos.clone()
                default_joint_vel = robot.data.default_joint_vel.clone()
                # 강제로 조인트 워프 -> 초기 Configuration 재 설정
                robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
                # 내부 버퍼에 토크 저장 -> 텔레포트가 아닌 실제 제어 입력을 위한 명령 신호
                robot.set_joint_effort_target(zero_joint_efforts)
                # 버퍼에 쓰인 전체 제어명령 전부 실행
                robot.write_data_to_sim()
                robot.reset()
                robot.update(sim_dt)
            else:
                # --------- 역역학 제어 로직 ------------
                # 1) 현재 로봇 상태 읽기
                joint_pos = robot.data.joint_pos
                joint_vel = robot.data.joint_vel

                # 2) 물리 엔진으로부터 동역학 파라미터 가져오기
                mass_matrix_full = robot.root_physx_view.get_generalized_mass_matrices()
                coriolis_full = robot.root_physx_view.get_coriolis_and_centrifugal_forces()
                gravity_full = robot.root_physx_view.get_gravity_compensation_forces()

                # --- Left Leg Control ---
                joint_pos_left = torch.index_select(joint_pos, 1, left_leg_indices)
                joint_vel_left = torch.index_select(joint_vel, 1, left_leg_indices)
                q_target_left = torch.index_select(q_target, 1, left_leg_indices)
                q_dot_target_left = torch.index_select(q_dot_target, 1, left_leg_indices)
                q_ddot_target_left = torch.index_select(q_ddot_target, 1, left_leg_indices)

                mass_matrix_left = mass_matrix_full.index_select(1, left_leg_dyn_indices).index_select(
                    2, left_leg_dyn_indices
                )
                coriolis_left = coriolis_full.index_select(1, left_leg_dyn_indices)
                gravity_left = gravity_full.index_select(1, left_leg_dyn_indices)

                tau_left = left_leg_controller.compute(
                    dof_pos=joint_pos_left,
                    dof_vel=joint_vel_left,
                    dof_pos_des=q_target_left,
                    dof_vel_des=q_dot_target_left,
                    dof_acc_des=q_ddot_target_left,
                    mass_matrix=mass_matrix_left,
                    coriolis_term=coriolis_left,
                    gravity_term=gravity_left,
                )

                # --- Right Leg Control ---
                joint_pos_right = torch.index_select(joint_pos, 1, right_leg_indices)
                joint_vel_right = torch.index_select(joint_vel, 1, right_leg_indices)
                q_target_right = torch.index_select(q_target, 1, right_leg_indices)
                q_dot_target_right = torch.index_select(q_dot_target, 1, right_leg_indices)
                q_ddot_target_right = torch.index_select(q_ddot_target, 1, right_leg_indices)

                mass_matrix_right = mass_matrix_full.index_select(1, right_leg_dyn_indices).index_select(
                    2, right_leg_dyn_indices
                )
                coriolis_right = coriolis_full.index_select(1, right_leg_dyn_indices)
                gravity_right = gravity_full.index_select(1, right_leg_dyn_indices)

                tau_right = right_leg_controller.compute(
                    dof_pos=joint_pos_right,
                    dof_vel=joint_vel_right,
                    dof_pos_des=q_target_right,
                    dof_vel_des=q_dot_target_right,
                    dof_acc_des=q_ddot_target_right,
                    mass_matrix=mass_matrix_right,
                    coriolis_term=coriolis_right,
                    gravity_term=gravity_right,
                )

                # 4) 계산된 토크와 그리퍼 명령을 시뮬레이션에 적용
                tau = torch.zeros(scene.num_envs, num_total_joints, device=sim.device)
                tau.scatter_(1, left_leg_indices.repeat(scene.num_envs, 1), tau_left)
                tau.scatter_(1, right_leg_indices.repeat(scene.num_envs, 1), tau_right)
                robot.set_joint_effort_target(tau)
                robot.write_data_to_sim()

            # 물리 시뮬레이션 스텝
            sim.step()
            robot.update(sim_dt)
            scene.update(sim_dt)
            count += 1

    elif args_cli.test_mode == "plotting":
        log_t = []
        log_q = []
        n_leg_j = n_j_per_leg * 2

        # 새로운 목표 자세 설정
        q_target_plot = q_target.clone()
        q_target_plot[:, 2] += 0.5  # thigh_L_Joint
        q_target_plot[:, 4] += 1.0  # knee_L_Joint
        q_target_plot[:, 5] += 1.5  # knee_R_Joint

        t = 0.0
        # 초기 상태 리셋
        print("[INFO] Reset state for plotting...")
        robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
        robot.set_joint_effort_target(zero_joint_efforts)
        robot.write_data_to_sim()
        robot.reset()
        robot.update(sim_dt)

        # 0초 기록
        log_t.append(0.0)
        log_q.append(robot.data.joint_pos[:, :n_leg_j].clone())

        while t <= sim_len:
            # --------- 역역학 제어 로직 ------------
            joint_pos = robot.data.joint_pos
            joint_vel = robot.data.joint_vel
            mass_matrix_full = robot.root_physx_view.get_generalized_mass_matrices()
            coriolis_full = robot.root_physx_view.get_coriolis_and_centrifugal_forces()
            gravity_full = robot.root_physx_view.get_gravity_compensation_forces()

            # Left Leg
            joint_pos_left = torch.index_select(joint_pos, 1, left_leg_indices)
            joint_vel_left = torch.index_select(joint_vel, 1, left_leg_indices)
            q_target_left = torch.index_select(q_target_plot, 1, left_leg_indices)
            q_dot_target_left = torch.index_select(q_dot_target, 1, left_leg_indices)
            q_ddot_target_left = torch.index_select(q_ddot_target, 1, left_leg_indices)
            mass_matrix_left = mass_matrix_full.index_select(1, left_leg_dyn_indices).index_select(
                2, left_leg_dyn_indices
            )
            coriolis_left = coriolis_full.index_select(1, left_leg_dyn_indices)
            gravity_left = gravity_full.index_select(1, left_leg_dyn_indices)
            tau_left = left_leg_controller.compute(
                dof_pos=joint_pos_left,
                dof_vel=joint_vel_left,
                dof_pos_des=q_target_left,
                dof_vel_des=q_dot_target_left,
                dof_acc_des=q_ddot_target_left,
                mass_matrix=mass_matrix_left,
                coriolis_term=coriolis_left,
                gravity_term=gravity_left,
            )

            # Right Leg
            joint_pos_right = torch.index_select(joint_pos, 1, right_leg_indices)
            joint_vel_right = torch.index_select(joint_vel, 1, right_leg_indices)
            q_target_right = torch.index_select(q_target_plot, 1, right_leg_indices)
            q_dot_target_right = torch.index_select(q_dot_target, 1, right_leg_indices)
            q_ddot_target_right = torch.index_select(q_ddot_target, 1, right_leg_indices)
            mass_matrix_right = mass_matrix_full.index_select(1, right_leg_dyn_indices).index_select(
                2, right_leg_dyn_indices
            )
            coriolis_right = coriolis_full.index_select(1, right_leg_dyn_indices)
            gravity_right = gravity_full.index_select(1, right_leg_dyn_indices)
            tau_right = right_leg_controller.compute(
                dof_pos=joint_pos_right,
                dof_vel=joint_vel_right,
                dof_pos_des=q_target_right,
                dof_vel_des=q_dot_target_right,
                dof_acc_des=q_ddot_target_right,
                mass_matrix=mass_matrix_right,
                coriolis_term=coriolis_right,
                gravity_term=gravity_right,
            )

            tau = torch.zeros(scene.num_envs, num_total_joints, device=sim.device)
            tau.scatter_(1, left_leg_indices.repeat(scene.num_envs, 1), tau_left)
            tau.scatter_(1, right_leg_indices.repeat(scene.num_envs, 1), tau_right)
            robot.set_joint_effort_target(tau)
            robot.write_data_to_sim()

            # 시뮬레이션 스텝 및 로그
            sim.step()
            robot.update(sim_dt)
            scene.update(sim_dt)
            t += sim_dt
            log_t.append(t)
            log_q.append(robot.data.joint_pos[:, :n_leg_j].clone())

        # --- 결과 플롯 ---
        import matplotlib.pyplot as plt
        import numpy as np
        import math

        log_t_np = np.asarray(log_t)
        log_q_np = torch.stack(log_q, dim=0).cpu().numpy().squeeze(1)

        n_cols = 3
        n_rows = math.ceil(n_leg_j / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True)
        axes = axes.flatten()

        for i in range(n_leg_j):
            ax = axes[i]
            ax.plot(log_t_np, log_q_np[:, i], label="actual")
            ax.axhline(joint_limits[0, i, 0].cpu(), ls="--", label="lower_limit", color="r")
            ax.axhline(joint_limits[0, i, 1].cpu(), ls="--", label="upper_limit", color="g")
            ax.axhline(q_target_plot[0, i].cpu(), ls="--", label="target", color="k")
            ax.set_title(f"Joint {i}")
            ax.set_ylabel("angle [rad]")
            if i // n_cols == n_rows - 1:
                ax.set_xlabel("time [s]")
            if i == 0:
                ax.legend(loc="best")
        fig.tight_layout()
        plt.show()

    elif args_cli.test_mode == "tracking":
        t = 0.0
        # Target joint position
        q_target_track = robot.data.default_joint_pos.clone()[:, : (n_j_per_leg * 2)]
        q_target_track[:, 3] += 0.3  # thigh_R_Joint

        while simulation_app.is_running():
            if t > sim_len:
                print("[INFO] Reset state for tracking...")
                t = 0.0
                robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
                robot.set_joint_effort_target(zero_joint_efforts)
                robot.write_data_to_sim()
                robot.reset()
                robot.update(sim_dt)

            # --------- 역역학 제어 로직 ------------
            joint_pos = robot.data.joint_pos
            joint_vel = robot.data.joint_vel
            mass_matrix_full = robot.root_physx_view.get_generalized_mass_matrices()
            coriolis_full = robot.root_physx_view.get_coriolis_and_centrifugal_forces()
            gravity_full = robot.root_physx_view.get_gravity_compensation_forces()

            # Left Leg
            joint_pos_left = torch.index_select(joint_pos, 1, left_leg_indices)
            joint_vel_left = torch.index_select(joint_vel, 1, left_leg_indices)
            q_target_left = torch.index_select(q_target_track, 1, left_leg_indices)
            q_dot_target_left = torch.index_select(q_dot_target, 1, left_leg_indices)
            q_ddot_target_left = torch.index_select(q_ddot_target, 1, left_leg_indices)
            mass_matrix_left = mass_matrix_full.index_select(1, left_leg_dyn_indices).index_select(
                2, left_leg_dyn_indices
            )
            coriolis_left = coriolis_full.index_select(1, left_leg_dyn_indices)
            gravity_left = gravity_full.index_select(1, left_leg_dyn_indices)
            tau_left = left_leg_controller.compute(
                dof_pos=joint_pos_left,
                dof_vel=joint_vel_left,
                dof_pos_des=q_target_left,
                dof_vel_des=q_dot_target_left,
                dof_acc_des=q_ddot_target_left,
                mass_matrix=mass_matrix_left,
                coriolis_term=coriolis_left,
                gravity_term=gravity_left,
            )

            # Right Leg
            joint_pos_right = torch.index_select(joint_pos, 1, right_leg_indices)
            joint_vel_right = torch.index_select(joint_vel, 1, right_leg_indices)
            q_target_right = torch.index_select(q_target_track, 1, right_leg_indices)
            q_dot_target_right = torch.index_select(q_dot_target, 1, right_leg_indices)
            q_ddot_target_right = torch.index_select(q_ddot_target, 1, right_leg_indices)
            mass_matrix_right = mass_matrix_full.index_select(1, right_leg_dyn_indices).index_select(
                2, right_leg_dyn_indices
            )
            coriolis_right = coriolis_full.index_select(1, right_leg_dyn_indices)
            gravity_right = gravity_full.index_select(1, right_leg_dyn_indices)
            tau_right = right_leg_controller.compute(
                dof_pos=joint_pos_right,
                dof_vel=joint_vel_right,
                dof_pos_des=q_target_right,
                dof_vel_des=q_dot_target_right,
                dof_acc_des=q_ddot_target_right,
                mass_matrix=mass_matrix_right,
                coriolis_term=coriolis_right,
                gravity_term=gravity_right,
            )

            tau = torch.zeros(scene.num_envs, num_total_joints, device=sim.device)
            tau.scatter_(1, left_leg_indices.repeat(scene.num_envs, 1), tau_left)
            tau.scatter_(1, right_leg_indices.repeat(scene.num_envs, 1), tau_right)
            robot.set_joint_effort_target(tau)
            robot.write_data_to_sim()

            # 시뮬레이션 스텝
            sim.step()
            robot.update(sim_dt)
            scene.update(sim_dt)
            t += sim_dt


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 4.0], [0.0, 0.0, 0.0])
    
    scene_cfg = RobotSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    env_cfg = GOATBaseEnvCfg(scene=scene_cfg)
    
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, env_cfg)

if __name__ == "__main__":
    main()
    simulation_app.close()
