import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulation.")
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

from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.markers.config import FRAME_MARKER_CFG, CUBOID_MARKER_CFG
from isaaclab_assets import FRANKA_PANDA_CFG, FRANKA_PANDA_HIGH_PD_CFG
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.controllers.joint_impedance import JointImpedanceController, JointImpedanceControllerCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Design the scene Implicit Actuators on the robot."""
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
    # table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 1.05))
    )
    # robot
    # robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot",
    #                                          init_state=ArticulationCfg.InitialStateCfg(
    #         pos=(0.0, 0.0, 1.05),
    #         joint_pos={
    #         "panda_joint1": 0.0,
    #         "panda_joint2": -0.569,
    #         "panda_joint3": 0.0,
    #         "panda_joint4": -2.810,
    #         "panda_joint5": 0.0,
    #         "panda_joint6": 3.037,
    #         "panda_joint7": 0.741,
    #         "panda_finger_joint.*": 0.04,
    #     },
    #     ),
    #     )
    
    robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot",
                                             init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.05),
            joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            "panda_finger_joint.*": 0.04,
        },
        ),
        )
    # Impedance Controller를 사용하는 경우, 액추에이터 PD제어 모델 사용 X (중복 토크 계산)
    # 액추에이터에 Impedance Controller가 붙음으로써 최하단 제어기의 역할을 하게 되는 개념.
    robot.actuators["panda_shoulder"].stiffness = 0.0
    robot.actuators["panda_shoulder"].damping = 0.0
    robot.actuators["panda_forearm"].stiffness = 0.0
    robot.actuators["panda_forearm"].damping = 0.0


def run_simulator(sim : sim_utils.SimulationContext, scene : InteractiveScene):
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    robot_entity_cfg.resolve(scene)
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    
    # 7개 조인트만 동작
    joint_ids = robot_entity_cfg.joint_ids
    n_j = len(joint_ids)
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    left_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/left_finger"))
    left_finger_link_idx = robot.find_bodies("panda_leftfinger")[0][0]
    right_finger_link_idx = robot.find_bodies("panda_rightfinger")[0][0]
    left_finger_joint_idx = robot.find_joints("panda_finger_joint1")[0][0]
    right_finger_joint_idx = robot.find_joints("panda_finger_joint2")[0][0]

    # Relative Impedance Controller
    joint_imp_cfg = JointImpedanceControllerCfg(command_type="p_rel", 
                                                impedance_mode="variable",
                                                stiffness=300.0,
                                                damping_ratio=0.1,
                                                inertial_compensation=True, 
                                                gravity_compensation=True,
                                                stiffness_limits=(0, 1000))
    
    joint_imp_controller = JointImpedanceController(cfg=joint_imp_cfg, 
                                                    num_robots=scene.num_envs,
                                                    dof_pos_limits=robot.data.joint_pos_limits[:, :n_j],
                                                    device=scene.device)

    # ---------- 환경 준비 ----------
    n_j          = 7           # Franka: 9 (팔7+그리퍼2) 또는 7
    sim_len      = 2.0                        # [s] 실험 길이
    kp_table = torch.tensor([100, 300, 300, 300, 80, 80, 80], device=scene.device)
    zeta         = 0.3                       # Damping ratio(=가상 댐퍼 비율)
    joint_limits = robot.data.joint_pos_limits
    gripper_commands = (0.02 * torch.sin(torch.linspace(0, 2*torch.pi, 100)) + 0.02).reshape(-1, 1).to(sim.device)

    # ---------- 슬라이스 정의 ----------
    pos_slice       = slice(0, n_j)
    stiff_slice     = slice(n_j, 2*n_j)
    damp_slice      = slice(2*n_j, 3*n_j)

    # ---------- 명령 버퍼 ----------
    commands = torch.zeros(scene.num_envs,
                        joint_imp_controller.num_actions,
                        device=scene.device)
    
    # ---------- 초기값 설정 ----------
    zero_joint_efforts = torch.zeros(scene.num_envs, n_j, device=sim.device)
    q_init   = robot.data.default_joint_pos.clone()
    q_dot_init = robot.data.default_joint_vel.clone()
    q_target = q_init[:, :n_j].clone()
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
                robot.set_joint_effort_target(zero_joint_efforts, joint_ids=joint_ids)
                # 버퍼에 쓰인 전체 제어명령 전부 실행
                robot.write_data_to_sim()
                robot.reset()
                robot.update(sim_dt)
            else:
                # --------- 임피던스 제어 로직 ------------
                # 1) 상대 offset 계산
                joint_pos = robot.data.joint_pos[:, :n_j]
                q_des_rel =  q_target - joint_pos
                # print(f"target joint : {q_target}")
                # print(f"current joint : {joint_pos}")

                # 2) commands 채우기
                commands.zero_()
                commands[:, pos_slice]   = q_des_rel
                commands[:, stiff_slice] = kp_table      
                commands[:, damp_slice]  = zeta

                # 3) set_command → compute 순서 : Pure Torque 신호 생성
                joint_imp_controller.set_command(commands)
                tau = joint_imp_controller.compute(
                    dof_pos      = robot.data.joint_pos[:, :n_j],
                    dof_vel      = robot.data.joint_vel[:, :n_j],
                    mass_matrix  = robot.root_physx_view.get_generalized_mass_matrices()[:, :n_j, :n_j],
                    gravity= robot.root_physx_view.get_gravity_compensation_forces()[:, :n_j]
                )

                ####################################################################################
                # 4) 실제 Torque 신호 내부 버퍼에 저장 -> 액추에이터 모델로부터 최종 토크 계산 위함
                #### 만약, Direct로 Effort를 줄 생각이면, 액추에이터 쪽 PD제어 로직을 꺼야한다.
                ####################################################################################
                k = count % gripper_commands.shape[0]
                robot.set_joint_effort_target(tau, joint_ids=joint_ids)
                robot.set_joint_position_target(gripper_commands[k].reshape(-1, 1).repeat(1, 2), joint_ids=[left_finger_joint_idx, right_finger_joint_idx])
                print(f"gripper command : {gripper_commands[k]}")

                # 타겟 조인트 포지션 버퍼에 저장 -> 실제 액추에이터 PD제어기로 인해 토크 계산을 위함
                # robot.set_joint_position_target(default_joint_pos[:, :n_j], joint_ids=robot_entity_cfg.joint_ids)
                # 버퍼에 세팅된 제어 명령 전부 실행
                robot.write_data_to_sim()

            # 물리 시뮬레이션 스텝
            sim.step()
            robot.update(sim_dt)
            scene.update(sim_dt)
            count += 1

            # 시각화
            left_finger_pos_w = robot.data.body_state_w[:, left_finger_link_idx, :7]
            right_finger_pos_w = robot.data.body_state_w[:, right_finger_link_idx, :7]
            left_marker.visualize(left_finger_pos_w[:, :3], left_finger_pos_w[:, 3:7])
            width = torch.norm(left_finger_pos_w[:, :3] - right_finger_pos_w[:, :3])
            # print(f"gripper width : {width}")


    elif args_cli.test_mode == "plotting":
        first = True
        log_t = []
        log_q = []

        q_target[:, 2] += 0.5
        q_target[:, 3] += 0.3
        q_target[:, 4] += 1.0
        q_target[:, 6] += 1.56

        i = 1
        t = 0.0
        while t <= sim_len:
            if first:
                # reset joint state to default
                first = False
                print("[INFO] Reset state ...")
                default_joint_pos = robot.data.default_joint_pos.clone()
                default_joint_vel = robot.data.default_joint_vel.clone()
                # 강제로 조인트 워프 -> 초기 Configuration 재 설정
                robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
                # 내부 버퍼에 토크 저장 -> 텔레포트가 아닌 실제 제어 입력을 위한 명령 신호
                robot.set_joint_effort_target(zero_joint_efforts, joint_ids=joint_ids)
                # 버퍼에 쓰인 전체 제어명령 전부 실행
                robot.write_data_to_sim()
                robot.reset()
                robot.update(sim_dt)
                # 0초 기록
                log_t.append(0.0)
                log_q.append(robot.data.joint_pos[:, :n_j].clone())

            # --------- 임피던스 제어 로직 ------------
            # 1) 상대 offset 계산
            joint_pos = robot.data.joint_pos[:, :n_j]
            q_des_rel =  q_target - joint_pos
            print(f"target joint : {q_target}")
            print(f"current joint : {joint_pos}")

            # 2) commands 채우기
            commands.zero_()
            commands[:, pos_slice]   = q_des_rel
            commands[:, stiff_slice] = kp_table      
            commands[:, damp_slice]  = zeta

            # 3) set_command → compute 순서 : Pure Torque 신호 생성
            joint_imp_controller.set_command(commands)
            tau = joint_imp_controller.compute(
                dof_pos      = robot.data.joint_pos[:, :n_j],
                dof_vel      = robot.data.joint_vel[:, :n_j],
                mass_matrix  = robot.root_physx_view.get_generalized_mass_matrices()[:, :n_j, :n_j],
                gravity= robot.root_physx_view.get_gravity_compensation_forces()[:, :n_j]
            )

            ####################################################################################
            # 4) 실제 Torque 신호 내부 버퍼에 저장 -> 액추에이터 모델로부터 최종 토크 계산 위함
            #### 만약, Direct로 Effort를 줄 생각이면, 액추에이터 쪽 PD제어 로직을 꺼야한다.
            ####################################################################################
            robot.set_joint_effort_target(tau, joint_ids=joint_ids)

            # 5) 타겟 조인트 포지션 버퍼에 저장 -> 실제 액추에이터 PD제어기로 인해 토크 계산을 위함
            # 버퍼에 세팅된 제어 명령 전부 실행
            robot.write_data_to_sim()

            # 6) 물리 시뮬레이션 스텝
            sim.step()
            robot.update(sim_dt)
            scene.update(sim_dt)
            t += sim_dt

            # 7) 로그 저장
            log_t.append(t)
            log_q.append(robot.data.joint_pos[:, :n_j].clone())
            


        # --- 결과 플롯 -------------------------------------------------
        import matplotlib.pyplot as plt
        import numpy as np
        import math
        log_t_np = np.asarray(log_t)
        log_q_np = torch.stack(log_q, dim=0).cpu().numpy().squeeze(1)

        n_cols = 3                                    # 한 줄에 최대 3장씩
        n_rows = math.ceil(n_j / n_cols)              # 필요한 줄 수
        fig, axes = plt.subplots(n_rows, n_cols,
                                figsize=(4*n_cols, 3*n_rows),
                                sharex=True)         # 시간축 공유
        axes = axes.flatten()                         # 1-D로 편리하게

        for i in range(n_j):
            ax = axes[i]
            ax.plot(log_t_np, log_q_np[:, i], label="actual")
            ax.axhline(joint_limits[0, i, 0].cpu(), ls="--", label="lower_limit", color='r')
            ax.axhline(joint_limits[0, i, 1].cpu(), ls="--", label="upper_limit", color='g')
            ax.axhline(q_target[0, i].cpu(),         ls="--", label="target", color="k")
            ax.set_title(f"Joint {i}")
            ax.set_ylabel("angle [rad]")
            if i // n_cols == n_rows - 1:         
                ax.set_xlabel("time [s]")
            # 범례는 첫 번째 서브플롯에만
            if i == 0:
                ax.legend(loc="best")
        fig.tight_layout()
        plt.show()
    
    elif args_cli.test_mode == "tracking":
        q_low  = joint_limits[0, :n_j, 0]
        q_high = joint_limits[0, :n_j, 1]
        t = 0.0
        q_target = robot.data.default_joint_pos.clone()[:, :n_j]
        q_target[:, 3] += 0.3 
        while simulation_app.is_running():
            if t > sim_len:
                # reset joint state to default
                print("[INFO] Reset state ...")
                t = 0.0
                default_joint_pos = robot.data.default_joint_pos.clone()
                default_joint_vel = robot.data.default_joint_vel.clone()
                # 강제로 조인트 워프 -> 초기 Configuration 재 설정
                robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
                # 내부 버퍼에 토크 저장 -> 텔레포트가 아닌 실제 제어 입력을 위한 명령 신호
                robot.set_joint_effort_target(zero_joint_efforts, joint_ids=joint_ids)
                # 버퍼에 쓰인 전체 제어명령 전부 실행
                robot.write_data_to_sim()
                robot.reset()
                robot.update(sim_dt)
                # Target Joint Angle 생성
                # q_target = torch.rand_like(q_target).mul(q_high - q_low).add(q_low)

            # --------- 임피던스 제어 로직 ------------
            # 1) 상대 offset 계산
            joint_pos = robot.data.joint_pos[:, :n_j]
            q_des_rel =  q_target - joint_pos
            print(f"target joint : {q_target}")
            print(f"current joint : {joint_pos}")

            # 2) commands 채우기
            commands.zero_()
            commands[:, pos_slice]   = q_des_rel
            commands[:, stiff_slice] = kp_table      
            commands[:, damp_slice]  = zeta

            # 3) set_command → compute 순서 : Pure Torque 신호 생성
            joint_imp_controller.set_command(commands)
            tau = joint_imp_controller.compute(
                dof_pos      = robot.data.joint_pos[:, :n_j],
                dof_vel      = robot.data.joint_vel[:, :n_j],
                mass_matrix  = robot.root_physx_view.get_generalized_mass_matrices()[:, :n_j, :n_j],
                gravity= robot.root_physx_view.get_gravity_compensation_forces()[:, :n_j]
            )

            ####################################################################################
            # 4) 실제 Torque 신호 내부 버퍼에 저장 -> 액추에이터 모델로부터 최종 토크 계산 위함
            #### 만약, Direct로 Effort를 줄 생각이면, 액추에이터 쪽 PD제어 로직을 꺼야한다.
            ####################################################################################
            robot.set_joint_effort_target(tau, joint_ids=joint_ids)

            # 5) 타겟 조인트 포지션 버퍼에 저장 -> 실제 액추에이터 PD제어기로 인해 토크 계산을 위함
            # 버퍼에 세팅된 제어 명령 전부 실행
            robot.write_data_to_sim()

            # 6) 물리 시뮬레이션 스텝
            sim.step()
            robot.update(sim_dt)
            scene.update(sim_dt)
            t += sim_dt


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 4.0], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = RobotSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()