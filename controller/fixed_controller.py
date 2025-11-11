import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on inverse dynamics control for an articulation.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--mode", type=str, default="default", choices=["default", "plotting"])
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, Articulation, AssetBaseCfg
from isaaclab.utils import configclass
from lib.env.GOAT_base_env_cfg import GOATBaseEnvCfg, GOAT_Cfg


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Design the scene for inverse dynamics control."""
    # Ground
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # Robot
    robot = GOAT_Cfg.replace(
            spawn=GOAT_Cfg.spawn.replace(
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True, solver_position_iteration_count=4,
                    solver_velocity_iteration_count=0, fix_root_link=True       # Fixed_base link
                )
            ),

            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 1.5),
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
    def __init__(self, kp, kd, num_envs: int, num_dof: int, device: str):
        """
        컨트롤러를 초기화합니다.

        Args:
            kp (float or torch.Tensor): 비례 이득 (Proportional gain). float 값 또는 (1, num_dof) 크기의 텐서.
            kd (float or torch.Tensor): 미분 이득 (Derivative gain). float 값 또는 (1, num_dof) 크기의 텐서.
            num_envs (int): 시뮬레이션 환경의 수.
            num_dof (int): 제어할 관절의 수 (degrees of freedom).
            device (str): 연산에 사용할 디바이스 (e.g., "cuda:0" or "cpu").
        """
        # kp gain
        if isinstance(kp, float):
            self.kp = torch.full((num_envs, num_dof), kp, device=device)
        elif isinstance(kp, torch.Tensor):
            if kp.shape != (1, num_dof):
                raise ValueError(f"kp tensor must have shape (1, {num_dof}), but got {kp.shape}")
            self.kp = kp.to(device).expand(num_envs, -1) # Expand as num_envs
        else:
            raise TypeError("kp must be a float or a torch.Tensor of shape (1, num_dof)")

        # kd gain
        if isinstance(kd, float):
            self.kd = torch.full((num_envs, num_dof), kd, device=device)
        elif isinstance(kd, torch.Tensor):
            if kd.shape != (1, num_dof):
                raise ValueError(f"kd tensor must have shape (1, {num_dof}), but got {kd.shape}")
            self.kd = kd.to(device).expand(num_envs, -1) # Expand as num_envs
        else:
            raise TypeError("kd must be a float or a torch.Tensor of shape (1, num_dof)")

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


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # define scene
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()

    # Robot dof
    leg_dof = 3     # hip, thigh, knee joints
    num_total_joints = 8
    n_leg_j = leg_dof * 2       
    
    # Define joint indices for each leg
    # Isaac sim's Joint order: ['hip_L_Joint', 'hip_R_Joint', 'thigh_L_Joint', 'thigh_R_Joint', 'knee_L_Joint', 'knee_R_Joint', 'wheel_L_Joint', 'wheel_R_Joint']
    left_leg_indices = torch.tensor([0, 2, 4], device=sim.device, dtype=torch.long)
    right_leg_indices = torch.tensor([1, 3, 5], device=sim.device, dtype=torch.long)

    # --- Initialize PD Inverse Dynamics Controller ---
    # Create separate controllers for each leg for independent control
    leg_controller = InverseDynamicsController(
        kp=10.0, kd=5.0, num_envs=scene.num_envs, num_dof=leg_dof, device=scene.device
    )

    # ---------- 환경 준비 ----------
    sim_len = 4.0  # [s] 실험 길이
    joint_limits = robot.data.joint_pos_limits

    # ---------- 초기값 및 목표값 설정 ----------
    zero_joint_efforts = torch.zeros(scene.num_envs, num_total_joints, device=sim.device)
    q_init = robot.data.default_joint_pos[:, :n_leg_j].clone()
    q_target = q_init[:, :n_leg_j] + 0.5  # Left target joint position
    q_target[:, 1:6:2] = q_init[:, 1:6:2] - 0.5  # Right target joint position
    q_dot_target = torch.zeros_like(q_target)
    q_ddot_target = torch.zeros_like(q_target)

    robot.update(sim_dt)

    # --- 제어 로직 루프 --------------------------
    if args_cli.mode == "default":
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
                coriolis_full = robot.root_physx_view.get_coriolis_and_centrifugal_compensation_forces()
                gravity_full = robot.root_physx_view.get_gravity_compensation_forces()

                # --- Left Leg Control ---
                joint_pos_left = torch.index_select(joint_pos, 1, left_leg_indices)
                joint_vel_left = torch.index_select(joint_vel, 1, left_leg_indices)
                q_target_left = torch.index_select(q_target, 1, left_leg_indices)
                q_dot_target_left = torch.index_select(q_dot_target, 1, left_leg_indices)
                q_ddot_target_left = torch.index_select(q_ddot_target, 1, left_leg_indices)

                mass_matrix_left = mass_matrix_full.index_select(1, left_leg_indices).index_select(
                    2, left_leg_indices
                )
                coriolis_left = coriolis_full.index_select(1, left_leg_indices)
                gravity_left = gravity_full.index_select(1, left_leg_indices)

                tau_left = leg_controller.compute(
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

                mass_matrix_right = mass_matrix_full.index_select(1, right_leg_indices).index_select(
                    2, right_leg_indices
                )
                coriolis_right = coriolis_full.index_select(1, right_leg_indices)
                gravity_right = gravity_full.index_select(1, right_leg_indices)

                tau_right = leg_controller.compute(
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
                print("Torque:", tau)

            
            # 물리 시뮬레이션 스텝
            sim.step()
            robot.update(sim_dt)
            scene.update(sim_dt)
            count += 1

    elif args_cli.mode == "plotting":
        log_t = []
        log_q = []

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
            coriolis_full = robot.root_physx_view.get_coriolis_and_centrifugal_compensation_forces()
            gravity_full = robot.root_physx_view.get_gravity_compensation_forces()

            # Left Leg
            joint_pos_left = torch.index_select(joint_pos, 1, left_leg_indices)
            joint_vel_left = torch.index_select(joint_vel, 1, left_leg_indices)
            q_target_left = torch.index_select(q_target, 1, left_leg_indices)
            q_dot_target_left = torch.index_select(q_dot_target, 1, left_leg_indices)
            q_ddot_target_left = torch.index_select(q_ddot_target, 1, left_leg_indices)

            mass_matrix_left = mass_matrix_full.index_select(1, left_leg_indices).index_select(
                2, left_leg_indices
            )
            coriolis_left = coriolis_full.index_select(1, left_leg_indices)
            gravity_left = gravity_full.index_select(1, left_leg_indices)
            tau_left = leg_controller.compute(
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
            q_target_right = torch.index_select(q_target, 1, right_leg_indices)
            q_dot_target_right = torch.index_select(q_dot_target, 1, right_leg_indices)
            q_ddot_target_right = torch.index_select(q_ddot_target, 1, right_leg_indices)
            mass_matrix_right = mass_matrix_full.index_select(1, right_leg_indices).index_select(
                2, right_leg_indices
            )
            coriolis_right = coriolis_full.index_select(1, right_leg_indices)
            gravity_right = gravity_full.index_select(1, right_leg_indices)
            tau_right = leg_controller.compute(
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
        fig, axies = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True)
        axies = axies.flatten()

        joint_name = ["L_hip", "R_hip", "L_thigh", "R_thigh", "L_knee", "R_knee"]
        for i in range(n_leg_j):
            ax = axies[i]
            ax.plot(log_t_np, log_q_np[:, i], label="actual")
            ax.axhline(joint_limits[0, i, 0].cpu(), ls="--", label="lower_limit", color="r")
            ax.axhline(joint_limits[0, i, 1].cpu(), ls="--", label="upper_limit", color="g")
            ax.axhline(q_target[0, i].cpu(), ls="--", label="target", color="k")
            ax.set_title(joint_name[i])
            ax.set_ylabel("angle [rad]")
            if i // n_cols == n_rows - 1:
                ax.set_xlabel("time [s]")
            if i == 0:
                ax.legend(loc="best")
        fig.tight_layout()
        plt.show()

def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cpu")    # TODO: change to cuda after debugging
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 4.0], [0.0, 0.0, 0.0])
    
    scene_cfg = RobotSceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)

if __name__ == "__main__":
    main()
    simulation_app.close()
