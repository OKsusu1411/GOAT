import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="IK + PD torque control for an articulation.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--mode", type=str, default="plotting", choices=["default", "plotting"])
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
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import compute_pose_error
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG, CUBOID_MARKER_CFG
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from lib.env.GOAT_base_env_cfg import GOAT_Cfg
from lib.RRT.RRT_wrapper import RRTWrapper
from lib.utils import Env

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
                    solver_velocity_iteration_count=0, fix_root_link=True             # Fixed_base link
              
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
    def __init__(self, diff_ik_cfg: DifferentialIKControllerCfg, kp, kd, num_envs: int, num_dof: int, device: str, dt: float):
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
        self.diff_ik_cfg = diff_ik_cfg
        self.device = device
        self.num_envs = num_envs
        self.num_dof = num_dof
        self.dt = dt
        self.old_torque = torch.zeros(self.num_envs, num_dof * 2 + 2, device=self.device)

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
            joint_delta_pos = super()._compute_delta_joint_pos(delta_pose=position_error, jacobian=jacobian_pos)
        else:
            position_error, axis_angle_error = compute_pose_error(
                ee_pos, ee_quat, self.ee_pos_des, self.ee_quat_des, rot_error_type="axis_angle"
            )
            pose_error = torch.cat((position_error, axis_angle_error), dim=1)
            joint_delta_pos = super()._compute_delta_joint_pos(delta_pose=pose_error, jacobian=jacobian)

        return joint_pos + 0.7 * joint_delta_pos

    def compute_torque(
        self,
        link_pose: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        foot_cmd: torch.Tensor,
        joint_limits: torch.Tensor,
        torque_limits: torch.Tensor,
        jacobian: torch.Tensor
    ) -> torch.Tensor:
        """

        Args:
            link_pose (torch.Tensor): Current all link pose [xyz + quaternion].
            joint_pos (torch.Tensor): Current joint position [rad].
            joint_vel (torch.Tensor): Current joint velocity [rad/s].
            foot_cmd (torch.Tensor): Reference joint pose [num_env, 2(L, R), 7].
            joint_limits (torch.Tensor): Joint position limits [num_env, num_joints, 2].
            torque_limits (torch.Tensor): Joint torque limits [num_env, num_joints].
            jacobian (torch.Tensor): Joint Jacobian matrix.

        Returns:
            torch.Tensor: Joint torque.
        """
        # Robot dof
        leg_dof = self.num_dof                  # hip, thigh, knee joints
        base_dof = 6                            # for floating base (linear + angular)
        num_total_joints = leg_dof * 2 + 2      # 6(revolute) + 2(wheel)

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
        joint_limits_left = torch.index_select(joint_limits, 1, left_leg_indices)
        foot_pos_left = link_pose[:, 7, :3]
        foot_quat_left = link_pose[:, 7, 3:]
        jacobian_left = jacobian[:, 6, :, left_leg_indices]

        # --- Right Leg ---
        joint_pos_right = torch.index_select(joint_pos, 1, right_leg_indices)
        joint_vel_right = torch.index_select(joint_vel, 1, right_leg_indices)
        foot_cmd_right = foot_cmd[:, 1, :]
        joint_limits_right = torch.index_select(joint_limits, 1, right_leg_indices)
        foot_pos_right = link_pose[:, 8, :3]
        foot_quat_right = link_pose[:, 8, 3:]
        jacobian_right = jacobian[:, 7, :, right_leg_indices]

        # Left foot IK + PD control
        super().set_command(command = foot_cmd_left, ee_pos=foot_pos_left, ee_quat=foot_quat_left)
        joint_pos_left_cmd = self.compute(ee_pos=foot_pos_left, ee_quat=foot_quat_left, jacobian=jacobian_left, joint_pos=joint_pos_left)
        joint_pos_left_cmd = torch.clamp(joint_pos_left_cmd, joint_limits_left[:, :, 0], joint_limits_left[:, :, 1])            # Clipping joint position command
        joint_pos_left_error = joint_pos_left_cmd - joint_pos_left
        joint_vel_left_error = - joint_vel_left                                             # reference joint velocity = 0
        torque_left = self.kp * joint_pos_left_error + self.kd * joint_vel_left_error
        # print("Left Torque:", torque_left)
        # print("Vel error:", joint_vel_left_error)

        # Right foot IK + PD control
        super().set_command(command = foot_cmd_right, ee_pos=foot_pos_right, ee_quat=foot_quat_right)
        joint_pos_right_cmd = self.compute(ee_pos=foot_pos_right, ee_quat=foot_quat_right, jacobian=jacobian_right, joint_pos=joint_pos_right)
        joint_pos_right_cmd = torch.clamp(joint_pos_right_cmd, joint_limits_right[:, :, 0], joint_limits_right[:, :, 1])        # Clipping joint position command
        joint_pos_right_error = joint_pos_right_cmd - joint_pos_right
        joint_vel_right_error = - joint_vel_right                                           # reference joint velocity = 0
        torque_right = self.kp * joint_pos_right_error + self.kd * joint_vel_right_error
        
        # Combine torque inputs
        torque = torch.zeros(self.num_envs, num_total_joints, device=self.device)
        torque.scatter_(1, left_leg_indices.repeat(self.num_envs, 1), torque_left)
        torque.scatter_(1, right_leg_indices.repeat(self.num_envs, 1), torque_right)
        
        # LPF for torque
        torque = 0.7 * self.old_torque + 0.3 * torque
        self.old_torque = torque.clone()

        # Clip torque based on torque_limits
        torque = torch.clamp(torque, -torque_limits, torque_limits)
        
        # Combine target joint angles (for debugging)
        angle = torch.zeros(self.num_envs, num_total_joints, device=self.device)
        angle.scatter_(1, left_leg_indices.repeat(self.num_envs, 1), joint_pos_left_cmd)
        angle.scatter_(1, right_leg_indices.repeat(self.num_envs, 1), joint_pos_right_cmd) 
        
        # TODO : Wheel controller 만들기
        return torque, angle

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene): 
    # define scene
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls")
    
    leg_dof = 3                         # hip, thigh, knee joints
    n_leg_j = leg_dof * 2
    num_total_joints = n_leg_j + 2

    # --- Initialize IK + PD torque Controller ---
    # Create separate controllers for each leg for independent control
    leg_controller = IK_PD_Controller(diff_ik_cfg=diff_ik_cfg, 
                                      kp=torch.tensor([[400.0, 400.0, 300.0]]),                       # TODO: Gain tuning required
                                      kd=torch.tensor([[20.0, 20.0, 15.0]]),                       # TODO: Gain tuning required
                                      num_envs=scene.num_envs,
                                      num_dof=leg_dof,
                                      device=scene.device,
                                      dt=sim_dt)

    # ---------- Environment Initialization ----------
    sim_len = 25.0  # [s] simulation length
    joint_limits = robot.data.joint_pos_limits
    torque_limits = robot.data.joint_effort_limits

    # ---------- Refrence input setting ----------
    zero_joint_efforts = torch.zeros(scene.num_envs, num_total_joints, device=sim.device)
    robot.update(sim_dt)

    # -------------- Control loop --------------
    if args_cli.mode == "default":
        count = 0
        while simulation_app.is_running():
            if count % 600 == 0:
                # reset joint state to default
                scene.reset()
                print("[INFO] Reset state ...")
                default_joint_pos = robot.data.default_joint_pos.clone()
                default_joint_vel = robot.data.default_joint_vel.clone()

                # Random joint angle within limits
                lower_limits = joint_limits[:, :, 0]
                upper_limits = joint_limits[:, :, 1]
                random_angle = lower_limits + torch.rand_like(lower_limits) * (upper_limits - lower_limits)
                robot.write_joint_state_to_sim(random_angle, default_joint_vel)
                target_link_pose = robot.data.body_link_pose_w
                q_target = target_link_pose[:, 7:, :]

                # Visualize target foot position
                frame_marker_cfg = FRAME_MARKER_CFG.copy()
                frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
                left_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/left_foot_marker"))
                right_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/right_foot_marker"))
                left_marker.visualize(q_target[0, 0, :3].unsqueeze(0), q_target[0, 0, 3:].unsqueeze(0))
                right_marker.visualize(q_target[0, 1, :3].unsqueeze(0), q_target[0, 1, 3:].unsqueeze(0))

                # Joint reset to initial pose
                robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
                # Joint torque reset to 0
                robot.set_joint_effort_target(zero_joint_efforts)
                robot.write_data_to_sim()
                robot.reset()
                robot.update(sim_dt)

            else:
                # --------- Control ------------
                # state awareness
                joint_pos = robot.data.joint_pos
                joint_vel = robot.data.joint_vel
                link_pose = robot.data.body_link_pose_w

                jacobian = robot.root_physx_view.get_jacobians()
                
                q_target_left = q_target[:, 0, :]
                q_target_right = q_target[:, 1, :]

                foot_cmd = torch.cat((q_target_left.unsqueeze(1), q_target_right.unsqueeze(1)), dim=1)

                # Compute torque
                torque, angle = leg_controller.compute_torque(link_pose=link_pose,
                                                              joint_pos=joint_pos,
                                                              joint_vel=joint_vel,
                                                              foot_cmd=foot_cmd,
                                                              joint_limits=joint_limits,
                                                              torque_limits=torque_limits,
                                                              jacobian=jacobian)
                
                robot.set_joint_effort_target(torque)
                # robot.write_joint_state_to_sim(angle, default_joint_vel)
                robot.write_data_to_sim()

            # Simulation step
            sim.step()
            robot.update(sim_dt)
            scene.update(sim_dt)
            count += 1

    elif args_cli.mode == "plotting":
        log_t = []
        log_q = []
        log_angle = []
        log_torque = []
        link_pose = robot.data.body_link_pose_w

        # reset state
        print("[INFO] Reset state for plotting...")
        default_joint_pos = robot.data.default_joint_pos.clone()
        default_joint_vel = robot.data.default_joint_vel.clone()

        # Random joint angle within limits
        lower_limits = joint_limits[:, :, 0]
        upper_limits = joint_limits[:, :, 1]
        random_angle = lower_limits + torch.rand_like(lower_limits) * (upper_limits - lower_limits)
        robot.write_joint_state_to_sim(random_angle, default_joint_vel)
        target_link_pose = robot.data.body_link_pose_w
        q_target = target_link_pose[:, 7:, :3]

        # Visualize target foot position
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        
        left_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/left_foot_marker"))
        right_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/right_foot_marker"))

        left_marker.visualize(target_link_pose[0, 7, :3].unsqueeze(0), target_link_pose[0, 7, 3:].unsqueeze(0))
        right_marker.visualize(target_link_pose[0, 8, :3].unsqueeze(0), target_link_pose[0, 8, 3:].unsqueeze(0))

        left_motion_planner = RRTWrapper(start=link_pose[0, 7].squeeze_(0), goal=target_link_pose[0, 7, :].squeeze_(0), env=Env.Map3D(5, 5, 5), max_dist=0.1, num_traj_points=50)
        left_optimal_trajectory = left_motion_planner.plan()

        right_motion_planner = RRTWrapper(start=link_pose[0, 8].squeeze_(0), goal=target_link_pose[0, 8, :].squeeze_(0), env=Env.Map3D(5, 5, 5), max_dist=0.1, num_traj_points=50)
        right_optimal_trajectory = right_motion_planner.plan()

        q_target[:, 0, :] = left_optimal_trajectory[0, :3]
        q_target[:, 1, :] = right_optimal_trajectory[0, :3]
        
        # Visualize target trajectory
        point_marker_cfg = CUBOID_MARKER_CFG.copy()
        point_marker_cfg.markers["cuboid"].size = (0.01, 0.01, 0.01)

        left_traj_marker = VisualizationMarkers(point_marker_cfg.replace(prim_path="/Visuals/left_traj"))
        right_traj_marker = VisualizationMarkers(point_marker_cfg.replace(prim_path="/Visuals/right_traj"))

        left_traj_marker.visualize(left_optimal_trajectory[:, :3])
        right_traj_marker.visualize(right_optimal_trajectory[:, :3])

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

        i = 0   # trajectory index
        while t <= sim_len:
            # --------- Control ------------
            # state awareness
            joint_pos = robot.data.joint_pos
            joint_vel = robot.data.joint_vel
            link_pose = robot.data.body_link_pose_w

            jacobian = robot.root_physx_view.get_jacobians()

            q_target_left = q_target[:, 0, :]
            q_target_right = q_target[:, 1, :]
            pos_error = torch.norm(link_pose[0, 7:, :3] - q_target[0, :, :3], dim=1)

            if torch.all(pos_error < 0.05):
                i = i + 1
                if i == left_optimal_trajectory.shape[0] - 1:
                    i = i - 1
                    print("[INFO] Reached target position.")

            # Update target foot position along the optimal trajectory
            q_target[:, 0, :] = left_optimal_trajectory[i, :3]
            q_target[:, 1, :] = right_optimal_trajectory[i, :3]
            foot_cmd = torch.cat((q_target_left.unsqueeze(1), q_target_right.unsqueeze(1)), dim=1)

            # Compute torque
            torque, angle = leg_controller.compute_torque(link_pose=link_pose,
                                                          joint_pos=joint_pos,
                                                          joint_vel=joint_vel,
                                                          foot_cmd=foot_cmd,
                                                          joint_limits=joint_limits,
                                                          torque_limits=torque_limits,
                                                          jacobian=jacobian)

            robot.set_joint_effort_target(torque)
            # robot.write_joint_state_to_sim(angle, default_joint_vel)
            robot.write_data_to_sim()

            # Simulation step
            sim.step()
            robot.update(sim_dt)
            scene.update(sim_dt)
            t += sim_dt
            log_t.append(t)
            log_q.append(robot.data.joint_pos[:, :n_leg_j].clone())
            log_angle.append(angle[:, :n_leg_j].clone())
            log_torque.append(torque[:, :n_leg_j].clone())

        # --- 결과 플롯 ---
        import matplotlib.pyplot as plt
        import numpy as np
        import math

        log_t_np = np.asarray(log_t)
        log_q_np = torch.stack(log_q, dim=0).cpu().numpy().squeeze(1)
        log_angle = torch.stack(log_angle, dim=0).cpu().numpy().squeeze(1)
        log_torque_np = torch.stack(log_torque, dim=0).cpu().numpy().squeeze(1)

        n_cols = 3
        n_rows = math.ceil(n_leg_j / n_cols)
        joint_name = ["L_hip", "R_hip", "L_thigh", "R_thigh", "L_knee", "R_knee"]

        # Joint Angle Plot
        fig, axies = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True)
        axies = axies.flatten()

        for i in range(n_leg_j):
            ax = axies[i]
            ax.plot(log_t_np, log_q_np[:, i], label="actual")
            ax.axhline(joint_limits[0, i, 0].cpu(), ls="--", label="lower_limit", color="r")
            ax.axhline(joint_limits[0, i, 1].cpu(), ls="--", label="upper_limit", color="g")
            ax.axhline(random_angle[0, i].cpu(), ls="--", label="optimal_angle", color="b")
            ax.plot(log_t_np, log_angle[:, i], ls="--", label="target", color="k")
            ax.set_title(f"Joint Angle: {joint_name[i]}")
            ax.set_ylabel("angle [rad]")
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
