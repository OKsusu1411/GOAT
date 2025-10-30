import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on inverse dynamics control for an articulation.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--test_mode", type=str, default="withstand", choices=["withstand", "plotting"])
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
from isaaclab.utils.math import compute_pose_error
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
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
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True, solver_position_iteration_count=4,
                    solver_velocity_iteration_count=0, fix_root_link=True       # Floating_base link
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

class IK_PD_Controller(DifferentialIKController):
    """
    
    τ = Kp*e + Kd*ė
    """
    def __init__(self, diff_ik_cfg: DifferentialIKControllerCfg, kp, kd, num_envs: int, num_dof: int, device: str):
        """
        컨트롤러를 초기화합니다.

        Args:
            kp (float or torch.Tensor): 비례 이득 (Proportional gain). float 값 또는 (1, num_dof) 크기의 텐서.
            kd (float or torch.Tensor): 미분 이득 (Derivative gain). float 값 또는 (1, num_dof) 크기의 텐서.
            num_envs (int): 시뮬레이션 환경의 수.
            num_dof (int): control dof per leg.
            device (str): 연산에 사용할 디바이스 (e.g., "cuda:0" or "cpu").
        """
        self.diff_ik_cfg = diff_ik_cfg
        self.device = device
        self.num_envs = num_envs
        self.num_dof = num_dof

        # Initialize Differential IK Controller
        super().__init__(diff_ik_cfg, num_envs=num_envs, device=device)

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
        
    # Override compute method
    def compute(
        self, ee_pos: torch.Tensor, ee_quat: torch.Tensor, jacobian: torch.Tensor, joint_pos: torch.Tensor
    ) -> torch.Tensor:
        """Computes the target joint positions that will yield the desired end effector pose.

        Args:
            ee_pos: The current end-effector position in shape (N, 3).
            ee_quat: The current end-effector orientation in shape (N, 4).
            jacobian: The geometric jacobian matrix in shape (N, 6, num_joints).
            joint_pos: The current joint positions in shape (N, num_joints).

        Returns:
            The target joint positions commands in shape (N, num_joints).
        """
        # compute the delta in joint-space
        if "position" in self.cfg.command_type:
            position_error = self.ee_pos_des - ee_pos
            jacobian_pos = jacobian[:, 0:3]
            joint_vel = super()._compute_delta_joint_pos(delta_pose=position_error, jacobian=jacobian_pos)
        else:
            position_error, axis_angle_error = compute_pose_error(
                ee_pos, ee_quat, self.ee_pos_des, self.ee_quat_des, rot_error_type="axis_angle"
            )
            pose_error = torch.cat((position_error, axis_angle_error), dim=1)
            print("Pose Error:", pose_error)
            joint_vel = super()._compute_delta_joint_pos(delta_pose=pose_error, jacobian=jacobian)
            # print("joint_vel:", joint_vel)
            self.joint_vel = joint_vel
        # return the desired joint positions, joint velocity
        return joint_pos + joint_vel

    def compute_torque(
        self,
        link_pose: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        foot_cmd: torch.Tensor,
        jacobian: torch.Tensor
    ) -> torch.Tensor:
        """

        Args:
            link_pose (torch.Tensor): Current all link pose [xyz + quaternion].
            joint_pos (torch.Tensor): Current joint position [rad].
            joint_vel (torch.Tensor): Current joint velocity [rad/s].
            foot_cmd (torch.Tensor): Reference joint pose [num_env, 2(L, R), 7].
            jacobian (torch.Tensor): Joint Jacobian matrix.

        Returns:
            torch.Tensor: Joint torque.
        """
        # Robot dof
        leg_dof = self.num_dof    # hip, thigh, knee joints
        base_dof = 6    # for floating base (linear + angular)
        num_total_joints = 8
        n_leg_j = leg_dof * 2

        # Define joint indices for each leg
        # Isaac sim's Joint order: ['hip_L_Joint', 'hip_R_Joint', 'thigh_L_Joint', 'thigh_R_Joint', 'knee_L_Joint', 'knee_R_Joint', 'wheel_L_Joint', 'wheel_R_Joint']
        left_leg_indices = torch.tensor([0, 2, 4], device=self.device, dtype=torch.long)
        right_leg_indices = torch.tensor([1, 3, 5], device=self.device, dtype=torch.long)

        # Corresponding indices in the dynamics tensors (with floating base offset)
        left_leg_dyn_indices = left_leg_indices + base_dof
        right_leg_dyn_indices = right_leg_indices + base_dof
        
        # --- Left Leg ---
        joint_pos_left = torch.index_select(joint_pos, 1, left_leg_indices)
        joint_vel_left = torch.index_select(joint_vel, 1, left_leg_indices)

        foot_cmd_left = foot_cmd[:, 0, :]

        foot_pos_left = link_pose[:, 7, :3]
        foot_quat_left = link_pose[:, 7, 3:]
        jacobian_left = jacobian[:, 6, :, left_leg_indices]

        # --- Right Leg ---
        joint_pos_right = torch.index_select(joint_pos, 1, right_leg_indices)
        joint_vel_right = torch.index_select(joint_vel, 1, right_leg_indices)

        foot_cmd_right = foot_cmd[:, 1, :]

        foot_pos_right = link_pose[:, 8, :3]
        foot_quat_right = link_pose[:, 8, 3:]
        jacobian_right = jacobian[:, 7, :, right_leg_indices]

        # Left foot IK + PD control
        super().set_command(command = foot_cmd_left, ee_pos=foot_pos_left, ee_quat=foot_quat_left)
        joint_pos_left_cmd = self.compute(ee_pos=foot_pos_left, ee_quat=foot_quat_left, jacobian=jacobian_left, joint_pos=joint_pos_left)
        joint_pos_left_error = joint_pos_left_cmd - joint_pos_left
        # print("Left Joint Pos Error:", joint_pos_left_error)
        # joint_vel_left_error = self.joint_vel - joint_vel_left
        joint_vel_left_error = - joint_vel_left
        # print("Left Joint Vel Error:", joint_vel_left_error)
        torque_left = self.kp * joint_pos_left_error + self.kd * joint_vel_left_error

        # Right foot IK + PD control
        super().set_command(command = foot_cmd_right, ee_pos=foot_pos_right, ee_quat=foot_quat_right)
        joint_pos_right_cmd = self.compute(ee_pos=foot_pos_right, ee_quat=foot_quat_right, jacobian=jacobian_right, joint_pos=joint_pos_right)
        joint_pos_right_error = joint_pos_right_cmd - joint_pos_right
        # joint_vel_right_error = self.joint_vel - joint_vel_right
        joint_vel_right_error = - joint_vel_right
        torque_right = self.kp * joint_pos_right_error + self.kd * joint_vel_right_error
        
        # Combine torque inputs
        # torque = torch.zeros(self.num_envs, num_total_joints, device=self.device)
        # torque.scatter_(1, left_leg_indices.repeat(self.num_envs, 1), torque_left)
        # torque.scatter_(1, right_leg_indices.repeat(self.num_envs, 1), torque_right)

        angle = torch.zeros(self.num_envs, num_total_joints, device=self.device)
        angle.scatter_(1, left_leg_indices.repeat(self.num_envs, 1), joint_pos_left_cmd)
        angle.scatter_(1, right_leg_indices.repeat(self.num_envs, 1), joint_pos_right_cmd)  
        
        # TODO : Wheel controller 만들기
        return angle

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # define scene
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    
    leg_dof = 3    # hip, thigh, knee joints
    base_dof = 6    # for floating base (linear + angular)
    num_total_joints = 8
    n_leg_j = leg_dof * 2
    # --- Initialize PD Inverse Dynamics Controller ---
    # Create separate controllers for each leg for independent control
    leg_controller = IK_PD_Controller(diff_ik_cfg= diff_ik_cfg, 
                                      kp=7.0,
                                      kd=4.0,
                                      num_envs=scene.num_envs,
                                      num_dof=leg_dof,
                                      device=scene.device)

    # ---------- 환경 준비 ----------
    sim_len = 3.0  # [s] 실험 길이
    joint_limits = robot.data.joint_pos_limits

    # ---------- 초기값 및 목표값 설정 ----------
    zero_joint_efforts = torch.zeros(scene.num_envs, num_total_joints, device=sim.device)
    q_init = robot.data.default_joint_pos[:, :n_leg_j].clone()
    q_target = torch.tensor([[-1.1063e-01,  3.0141e-01,  8.4402e-01,  9.7444e-01,  1.7835e-01, 1.3436e-01,  2.4591e-02],   # Left foot position (x, y, z, quat)
                             [-1.1063e-01, -3.0141e-01,  8.4402e-01,  9.7444e-01, -1.7835e-01, 1.3437e-01, -2.4593e-02]])  # Right foot position (x, y, z, quat)

    robot.update(sim_dt)

    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    left_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/left_foot_marker"))
    right_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/right_foot_marker"))

    left_marker.visualize(q_target[0, :3].unsqueeze(0), q_target[0, 3:].unsqueeze(0))
    right_marker.visualize(q_target[1, :3].unsqueeze(0), q_target[1, 3:].unsqueeze(0))

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
                link_pose = robot.data.body_link_pose_w

                jacobian = robot.root_physx_view.get_jacobians()
                
                q_target_left = q_target[0, :]
                q_target_left = q_target_left.expand(scene.num_envs, -1)

                q_target_right = q_target[1, :]
                q_target_right = q_target_right.expand(scene.num_envs, -1)

                foot_cmd = torch.cat((q_target_left.unsqueeze(1), q_target_right.unsqueeze(1)), dim=1)

                torque = leg_controller.compute_torque(link_pose=link_pose,
                                                       joint_pos=joint_pos,
                                                       joint_vel=joint_vel,
                                                       foot_cmd=foot_cmd,
                                                       jacobian=jacobian)

                # print("Computed Torque:", torque)
                # 4) 계산된 토크와 그리퍼 명령을 시뮬레이션에 적용
                # robot.set_joint_effort_target(torque)
                robot.write_joint_state_to_sim(torque, default_joint_vel)
                robot.write_data_to_sim()

            # 물리 시뮬레이션 스텝
            sim.step()
            robot.update(sim_dt)
            scene.update(sim_dt)
            count += 1

    elif args_cli.test_mode == "plotting":
        log_t = []
        log_q = []
        n_leg_j = leg_dof * 2

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
            coriolis_full = robot.root_physx_view.get_coriolis_and_centrifugal_compensation_forces()
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
            q_target_right = torch.index_select(q_target_plot, 1, right_leg_indices)
            q_dot_target_right = torch.index_select(q_dot_target, 1, right_leg_indices)
            q_ddot_target_right = torch.index_select(q_ddot_target, 1, right_leg_indices)
            mass_matrix_right = mass_matrix_full.index_select(1, right_leg_dyn_indices).index_select(
                2, right_leg_dyn_indices
            )
            coriolis_right = coriolis_full.index_select(1, right_leg_dyn_indices)
            gravity_right = gravity_full.index_select(1, right_leg_dyn_indices)
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
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True)
        axes = axes.flatten()

        joint_name = ["L_hip", "R_hip", "L_thigh", "R_thigh", "L_knee", "R_knee"]
        for i in range(n_leg_j):
            ax = axes[i]
            ax.plot(log_t_np, log_q_np[:, i], label="actual")
            ax.axhline(joint_limits[0, i, 0].cpu(), ls="--", label="lower_limit", color="r")
            ax.axhline(joint_limits[0, i, 1].cpu(), ls="--", label="upper_limit", color="g")
            ax.axhline(q_target_plot[0, i].cpu(), ls="--", label="target", color="k")
            ax.set_title(joint_name(i))
            ax.set_ylabel("angle [rad]")
            if i // n_cols == n_rows - 1:
                ax.set_xlabel("time [s]")
            if i == 0:
                ax.legend(loc="best")
        fig.tight_layout()
        plt.show()


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
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
