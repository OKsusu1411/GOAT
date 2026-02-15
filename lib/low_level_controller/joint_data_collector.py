import argparse
import time
import math
import numpy as np
import pandas as pd
import torch
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="PD torque control data collector.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import compute_pose_error
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG, CUBOID_MARKER_CFG
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from lib.env.GOAT_base_env_cfg import GOAT_Cfg
from lib.RRT.RRT_wrapper import RRTWrapper
from lib.utils import Env

# HIP_COL_FRI = 0
# HIP_VIS_FRI = 0
# KNEE_COL_FRI = 0
# KNEE_VIS_FRI = 0

@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Design the scene for low-level torque control."""
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
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),     # zero-G
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False, solver_position_iteration_count=4,
                    solver_velocity_iteration_count=0, fix_root_link=True               # Fixed_base link
              
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

class PD_Controller():
    """
    
    τ = Kp*e + Kd*ė
    """
    def __init__(self, kp, kd, num_envs: int, num_dof: int, device: str, dt: float):
        """
        Controller initialization

        Args:
            kp (float or torch.Tensor): Proportional gain. float value or (1, num_dof) size tensor.
            kd (float or torch.Tensor): Derivative gain. float value or (1, num_dof) size tensor.
            num_envs (int): Number of pararell training environments.
            num_dof (int): controllable dof per leg.
            device (str): "cuda:0" or "cpu".
            dt (float): Simulation time-step.
        """
        self.device = device
        self.num_envs = num_envs
        self.num_dof = num_dof
        self.dt = dt
        self.old_torque = torch.zeros(self.num_envs, num_dof * 2 + 2, device=self.device)

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

        # Friction coefficients
        # Order: [hip, thigh, knee]
        # Assuming knee friction is 0 as it's not defined.
        coulomb_coeffs = torch.tensor([HIP_COL_FRI, HIP_COL_FRI, 0.0], device=device).unsqueeze(0)
        viscous_coeffs = torch.tensor([HIP_VIS_FRI, HIP_VIS_FRI, 0.0], device=device).unsqueeze(0)
        self.coulomb_coeffs = coulomb_coeffs.expand(num_envs, -1)
        self.viscous_coeffs = viscous_coeffs.expand(num_envs, -1)

    def compute_torque(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        joint_pos_cmd: torch.Tensor,
        joint_limits: torch.Tensor,
        torque_limits: torch.Tensor,
        coriolis_full: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute joint input torque using PD controller.

        Args:
            joint_pos (torch.Tensor): Current joint position [rad].
            joint_vel (torch.Tensor): Current joint velocity [rad/s].
            joint_pos_cmd (torch.Tensor): Reference joint pos(angle) [num_env, 2(L, R), num_joints].
            joint_limits (torch.Tensor): Joint position limits [num_env, num_joints, 2].
            torque_limits (torch.Tensor): Joint torque limits [num_env, num_joints].

        Returns:
            torch.Tensor: Joint torque.
        """
        # Robot dof
        leg_dof = self.num_dof                  # hip, thigh, knee joints
        num_total_joints = leg_dof * 2 + 2      # 6(revolute) + 2(wheel)

        # Define joint indices for each leg
        # Isaac sim's Joint order: ['hip_L_Joint', 'hip_R_Joint', 'thigh_L_Joint', 'thigh_R_Joint', 'knee_L_Joint', 'knee_R_Joint', 'wheel_L_Joint', 'wheel_R_Joint']
        left_leg_indices = torch.tensor([0, 2, 4], device=self.device, dtype=torch.long)
        right_leg_indices = torch.tensor([1, 3, 5], device=self.device, dtype=torch.long)

        coriolis_left = coriolis_full.index_select(1, left_leg_indices)
        coriolis_right = coriolis_full.index_select(1, right_leg_indices)
        
        # --- Left Leg slicing ---
        joint_pos_left = torch.index_select(joint_pos, 1, left_leg_indices)
        joint_vel_left = torch.index_select(joint_vel, 1, left_leg_indices)
        joint_pos_cmd_left = torch.index_select(joint_pos_cmd, 1, left_leg_indices)
        joint_limits_left = torch.index_select(joint_limits, 1, left_leg_indices)

        # --- Right Leg slicing ---
        joint_pos_right = torch.index_select(joint_pos, 1, right_leg_indices)
        joint_vel_right = torch.index_select(joint_vel, 1, right_leg_indices)
        joint_pos_cmd_right = torch.index_select(joint_pos_cmd, 1, right_leg_indices)
        joint_limits_right = torch.index_select(joint_limits, 1, right_leg_indices)

        # Left foot PD control
        # joint_pos_cmd_left = torch.clamp(joint_pos_cmd_left, joint_limits_left[:, :, 0], joint_limits_left[:, :, 1])            # Clipping joint position command
        joint_pos_left_error = joint_pos_cmd_left - joint_pos_left
        joint_vel_left_error = - joint_vel_left                                                                                 # reference joint velocity = 0
        
        pd_torque_left = self.kp * joint_pos_left_error + self.kd * joint_vel_left_error
        # pd_torque_left[:, -1] /= 2
        
        # Friction compensation (Note: to cancel friction, this term should typically be added, not subtracted)
        friction_comp_left = self.coulomb_coeffs * torch.sign(joint_vel_left) + self.viscous_coeffs * joint_vel_left
        
        # Clip friction compensation to prevent it from overpowering the PD torque and reversing the command's sign
        # clipped_friction_comp_left = torch.clamp(friction_comp_left, -torch.abs(pd_torque_left), torch.abs(pd_torque_left))
        torque_left = pd_torque_left - friction_comp_left + coriolis_left

        # Right foot PD control
        # joint_pos_cmd_right = torch.clamp(joint_pos_cmd_right, joint_limits_right[:, :, 0], joint_limits_right[:, :, 1])        # Clipping joint position command
        joint_pos_right_error = joint_pos_cmd_right - joint_pos_right
        joint_vel_right_error = - joint_vel_right                                                                               # reference joint velocity = 0

        pd_torque_right = self.kp * joint_pos_right_error + self.kd * joint_vel_right_error
        # pd_torque_right[:, -1] /= 2

        # Friction compensation (Note: to cancel friction, this term should typically be added, not subtracted)
        friction_comp_right = self.coulomb_coeffs * torch.sign(joint_vel_right) + self.viscous_coeffs * joint_vel_right

        # Clip friction compensation to prevent it from overpowering the PD torque and reversing the command's sign
        # clipped_friction_comp_right = torch.clamp(friction_comp_right, -torch.abs(pd_torque_right), torch.abs(pd_torque_right))
        torque_right = pd_torque_right - friction_comp_right
        # print(torque_left)
        # Combine torque inputs
        torque = torch.zeros(self.num_envs, num_total_joints, device=self.device)
        pd_torque = torch.zeros(self.num_envs, num_total_joints, device=self.device)
        
        torque.scatter_(1, left_leg_indices.repeat(self.num_envs, 1), torque_left)
        torque.scatter_(1, right_leg_indices.repeat(self.num_envs, 1), torque_right)

        pd_torque.scatter_(1, left_leg_indices.repeat(self.num_envs, 1), pd_torque_left)
        pd_torque.scatter_(1, right_leg_indices.repeat(self.num_envs, 1), pd_torque_right)
        
        # # LPF for torque
        # torque = 0.951 * self.old_torque + (1 - 0.951) * torque
        # self.old_torque = torque.clone()

        # Clip torque based on torque_limits
        torque = torch.clamp(torque, -torque_limits, torque_limits)
        
        # TODO : Wheel controller
        return torque, pd_torque

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene): 
    # define scene
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    
    leg_dof = 3                         # hip, thigh, knee joints
    n_leg_j = leg_dof * 2
    num_total_joints = n_leg_j + 2

    # # --- Initialize PD torque Controller ---
    # leg_controller = PD_Controller(kp=torch.tensor([[0.33, 0.27, 1.4]]),
    #                                kd=torch.tensor([[0.01, 0.01, 0.001]]),
    #                                num_envs=scene.num_envs,
    #                                num_dof=leg_dof,
    #                                device=scene.device,
    # #                                dt=sim_dt)
    #     leg_controller = PD_Controller(kp=torch.tensor([[0.330, 0.248, 1.27]]),
    #                                    kd=torch.tensor([[0.01, 0.001, 0.001]]),
    # --- Initialize PD torque Controller ---
    
    leg_controller = PD_Controller(kp=torch.tensor([[0.330, 0.00, 4.37]]),
                                   kd=torch.tensor([[0.01, 0.00, 0.001]]),
                                   num_envs=scene.num_envs,
                                   num_dof=leg_dof,
                                   device=scene.device,
                                   dt=sim_dt)

    leg_controller = PD_Controller(kp=torch.tensor([[0.330, 0.270, 0.350]]),
                                   kd=torch.tensor([[0.015, 0.010, 0.018]]),
                                   num_envs=scene.num_envs,
                                   num_dof=leg_dof,
                                   device=scene.device,
                                   dt=sim_dt)
                               
    # ---------- Environment Initialization ----------
    joint_limits = robot.data.joint_pos_limits
    torque_limits = robot.data.joint_effort_limits
    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = robot.data.default_joint_vel.clone()

    # ---------- Refrence input setting ----------
    zero_joint_efforts = torch.zeros(scene.num_envs, num_total_joints, device=sim.device)
    robot.update(sim_dt)

    # -------------- Control loop --------------
    log_t = []
    log_q = []
    log_angle = []
    log_torque = []
    log_ref = []  # [추가] time-varying reference logging

    # reset state
    print("[INFO] Reset state for plotting...")

    # ---------- User Configuration for Sine Input ----------
    TARGET_JOINT_IDX = 1         # [Index] 제어할 관절 인덱스 (0: hip_L, 1: hip_R, 2: thigh_L, 3: thigh_R, 4: knee_L, 5: knee_R, 6~7 wheel)
    SINE_AMP_DEG = 20.0          # [deg] Sine amplitude (peak)
    SINE_FREQ_HZ = 0.2           # [Hz] Sine frequency
    SIM_DURATION = 30.0          # [sec] Total simulation time
    
    sim_len = SIM_DURATION
    print(f"[INFO] Simulation Length set to {sim_len}s (sine input).")

    # Set Initial Base Angle (Initial Offset)
    base_joint_pos = default_joint_pos.clone()
    # base_joint_pos[:, 0] -= torch.pi/180*80.0
    # base_joint_pos[:, 1] +=  torch.pi/180*50.0
    # base_joint_pos[:, 2] +=  torch.pi/180*20.0
    # base_joint_pos[:, 3] -= torch.pi/180*80.0
    # base_joint_pos[:, 4] += torch.pi/180*20.0
    # base_joint_pos[:, 5] -= torch.pi/180*20.0
    
    reference_angle = base_joint_pos.clone()

    robot.write_joint_state_to_sim(reference_angle, default_joint_vel)
    target_link_pose = robot.data.body_link_pose_w

    # Visualize target foot position
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    
    left_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/left_foot_marker"))
    right_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/right_foot_marker"))

    left_marker.visualize(target_link_pose[0, 7, :3].unsqueeze(0), target_link_pose[0, 7, 3:].unsqueeze(0))
    right_marker.visualize(target_link_pose[0, 8, :3].unsqueeze(0), target_link_pose[0, 8, 3:].unsqueeze(0))

    # Joint reset to initial pose
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
    # Joint torque reset to 0
    robot.set_joint_effort_target(zero_joint_efforts)
    robot.write_data_to_sim()
    robot.reset()
    robot.update(sim_dt)

    # Logging initial state
    t = 0.0
    log_t.append(0.0)
    log_q.append(robot.data.joint_pos[:, :n_leg_j].clone())
    log_angle.append(torch.zeros(scene.num_envs, n_leg_j, device=scene.device))
    log_torque.append(torch.zeros(scene.num_envs, n_leg_j, device=scene.device))
    log_ref.append(reference_angle[:, :n_leg_j].clone())

    # [수정됨] CSV 저장 포맷을 "csv_view_compare.py"가 읽는 형식으로 변경
    # columns:
    # t_sec,
    # q_ref_i_rad, q_meas_i_rad, dq_meas_i_rad_s, tau_cmd_i_nm  (i=0..7)
    csv_rows = []

    while t <= sim_len:
        # --------- Sine Input Calculation ------------ #
        # added_angle_rad(t) = A * sin(2*pi*f*t)
        added_angle_rad = (SINE_AMP_DEG * math.pi / 180.0) * math.sin(2.0 * math.pi * SINE_FREQ_HZ * t)

        reference_angle = base_joint_pos.clone()

        # [수정됨] 사용자가 지정한 하나의 관절(TARGET_JOINT_IDX)에만 사인파 입력을 적용
        reference_angle[:, TARGET_JOINT_IDX] += added_angle_rad
        
        # --------- Control ------------ #
        # State awareness
        joint_pos = robot.data.joint_pos
        joint_vel = robot.data.joint_vel

        # coriolis
        coriolis_full = robot.root_physx_view.get_coriolis_and_centrifugal_compensation_forces()
        # print(robot.data.projected_gravity_b)

        # Compute torque
        torque, pd_torque = leg_controller.compute_torque(joint_pos=joint_pos,
                                                          joint_vel=joint_vel,
                                                          joint_pos_cmd=reference_angle,
                                                          joint_limits=joint_limits,
                                                          torque_limits=torque_limits,
                                                          coriolis_full=coriolis_full)

        robot.set_joint_effort_target(torque)
        robot.write_data_to_sim()

        # Simulation step
        sim.step()
        robot.update(sim_dt)
        scene.update(sim_dt)
        t += sim_dt

        # -------- 기존 플롯 로깅(그대로 유지 + ref 추가) --------
        log_t.append(t)
        log_q.append(robot.data.joint_pos[:, :n_leg_j].clone())
        log_torque.append((torch.round(pd_torque[:, :n_leg_j] * 100) / 100).clone())
        log_ref.append(reference_angle[:, :n_leg_j].clone())
        
        # --- CSV Logging Logic (viewer/compare 호환 포맷) ---
        # env 0 기준으로 저장
        env_idx = 0
        q_meas = joint_pos[env_idx].detach().cpu().numpy()          # rad
        dq_meas = joint_vel[env_idx].detach().cpu().numpy()         # rad/s
        q_ref = reference_angle[env_idx].detach().cpu().numpy()     # rad
        tau_cmd = torque[env_idx].detach().cpu().numpy()            # Nm (실제로 시뮬에 넣은 토크)

        # row = [t_sec, (q_ref0,q_meas0,dq0,tau0), ..., (q_ref7,q_meas7,dq7,tau7)]
        row = [float(t)]
        for i in range(8):
            row += [
                float(q_ref[i]),
                float(q_meas[i]),
                float(dq_meas[i]),
                float(tau_cmd[i]),
            ]
        csv_rows.append(row)

    # --- Save CSV (viewer/compare 호환 포맷) ---
    print("[INFO] Saving CSV data (viewer/compare format)...")

    header = ["t_sec"]
    for i in range(8):
        header += [
            f"q_ref_{i}_rad",
            f"q_meas_{i}_rad",
            f"dq_meas_{i}_rad_s",
            f"tau_cmd_{i}_nm",
        ]

    df = pd.DataFrame(csv_rows, columns=header)
    csv_filename = "simulation_data_viewer_format_sine.csv"
    df.to_csv(csv_filename, index=False)
    print(f"[INFO] Data saved to {csv_filename}")


    # --- 결과 플롯 ---
    import matplotlib.pyplot as plt
    
    log_t_np = np.asarray(log_t)
    log_q_np = torch.stack(log_q, dim=0).cpu().numpy().squeeze(1)
    log_torque_np = torch.stack(log_torque, dim=0).cpu().numpy().squeeze(1)
    log_ref_np = torch.stack(log_ref, dim=0).cpu().numpy().squeeze(1)  # [추가]

    n_cols = 3
    n_rows = math.ceil(n_leg_j / n_cols)
    joint_name = ["L_hip", "R_hip", "L_thigh", "R_thigh", "L_knee", "R_knee"]

    # Joint Angle Plot (Degree) (actual vs ref)
    fig, axies = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True)
    axies = axies.flatten()

    # rad → deg 변환
    log_q_deg = np.degrees(log_q_np)
    log_ref_deg = np.degrees(log_ref_np)
    joint_limits_deg = np.degrees(joint_limits[0].cpu().numpy())

    for i in range(n_leg_j):
        ax = axies[i]
        ax.plot(log_t_np, log_q_deg[:, i], label="actual")
        ax.plot(log_t_np, log_ref_deg[:, i], ls="--", label="ref")  # [추가] 사인파 ref 궤적 (deg)
        ax.axhline(joint_limits_deg[i, 0], ls="--", label="lower_limit", color="r")
        ax.axhline(joint_limits_deg[i, 1], ls="--", label="upper_limit", color="g")

        ax.set_title(f"Joint Angle: {joint_name[i]}")
        ax.set_ylabel("angle [deg]")
        if i // n_cols == n_rows - 1:
            ax.set_xlabel("time [s]")
        if i == 0:
            ax.legend(loc="best")
    fig.tight_layout()

    # Joint Torque Plot
    fig_torque, axies_torque = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True)
    axies_torque = axies_torque.flatten()

    for i in range(n_leg_j):
        ax = axies_torque[i]
        ax.plot(log_t_np, log_torque_np[:, i], label="torque")
        ax.set_title(f"Joint Torque: {joint_name[i]}")
        ax.set_ylabel("Torque [Nm]")
        if i // n_cols == n_rows - 1:
            ax.set_xlabel("time [s]")
        if i == 0:
            ax.legend(loc="best")
    fig_torque.tight_layout()

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
