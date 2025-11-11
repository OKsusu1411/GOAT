# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""This script demonstrates how to spawn Franka Emika robot & IK Test for Low & Middle-Level Controller Test.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/01_assets/run_articulation.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulation.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
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
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG, FRANKA_PANDA_CFG
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.controllers.joint_impedance import JointImpedanceController, JointImpedanceControllerCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms, quat_conjugate, quat_from_angle_axis, \
                                matrix_from_quat, quat_inv, quat_apply, quat_error_magnitude, combine_frame_transforms, quat_mul

from RRT.RRT_wrapper import RRTWrapper
from utils import Env



@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Design the scene Implicit Actuators on the robot."""
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.7405)),
    )
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    # Table
    table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0, 0.0], rot=[1.0, 0, 0, 0.0]),
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd",
                                   scale=(1.5, 2.0, 1.0)),
    )
    # robot
    robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot",
                                             init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
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
        )
        )
    # Impedance Controller를 사용하는 경우, 액추에이터 PD제어 모델 사용 X (중복 토크 계산)
    # 액추에이터에 Impedance Controller가 붙음으로써 최하단 제어기의 역할을 하게 되는 개념.
    robot.actuators["panda_shoulder"].stiffness = 0.0
    robot.actuators["panda_shoulder"].damping = 0.0
    robot.actuators["panda_forearm"].stiffness = 0.0
    robot.actuators["panda_forearm"].damping = 0.0
    
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.5, 0.2, 0.0), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )
    



def run_simulator(sim : sim_utils.SimulationContext, scene : InteractiveScene):
    # ============== Setup the scene =================
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    robot_entity_cfg.resolve(scene)

    robot = scene["robot"]
    object = scene["object"]

    point_marker_cfg = CUBOID_MARKER_CFG.copy()
    point_marker_cfg.markers["cuboid"].size = (0.01, 0.01, 0.01)
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    tcp_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/tcp_current"))
    hand_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/hand_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))
    traj_marker = VisualizationMarkers(point_marker_cfg.replace(prim_path="Visuals/ee_traj"))

    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=scene.device)

    num_active_joint = len(robot_entity_cfg.joint_ids)
    joint_imp_cfg = JointImpedanceControllerCfg(command_type="p_rel", 
                                                impedance_mode="variable",
                                                stiffness=400.0,
                                                damping_ratio=1.0,
                                                inertial_compensation=True, 
                                                gravity_compensation=True)
    joint_imp_controller = JointImpedanceController(cfg=joint_imp_cfg, 
                                                    num_robots=scene.num_envs,
                                                    dof_pos_limits=robot.data.joint_pos_limits[:, :num_active_joint],
                                                    device=scene.device)
    
    # =============== Link and Joint Indexs =================
    joint_ids = robot_entity_cfg.joint_ids
    hand_link_idx = robot.find_bodies("panda_hand")[0][0]
    left_finger_link_idx = robot.find_bodies("panda_leftfinger")[0][0]
    left_finger_joint_idx = robot.find_joints("panda_finger_joint1")[0][0]
    right_finger_link_idx = robot.find_bodies("panda_rightfinger")[0][0]
    right_finger_joint_idx = robot.find_joints("panda_finger_joint2")[0][0]
    finger_open_joint_pos = torch.full((scene.num_envs, 2), 0.04, device=scene.device)
    finger_ids = torch.tensor([left_finger_joint_idx, right_finger_joint_idx], device=scene.device)
    if robot.is_fixed_base:
        jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        jacobi_idx = robot_entity_cfg.body_ids[0]

    
    # ================ Initial Setting ====================
    # ---------- 환경 준비 ----------
    sim_dt = scene.physics_dt
    n_j          = 7           # Franka: 9 (팔7+그리퍼2) 또는 7
    kp_table = torch.tensor([100, 80, 80, 80, 80, 80, 80], device=scene.device)
    zeta         = 0.3                       # Damping ratio(=가상 댐퍼 비율)
    joint_limits = robot.data.joint_pos_limits
    offset = torch.tensor([0.0, 0.0, 0.107, 1.0, 0.0, 0.0, 0.0], device=scene.device).repeat([scene.num_envs, 1])


    # ---------- 명령 버퍼 ----------
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=scene.device)
    imp_commands = torch.zeros(scene.num_envs,
                               joint_imp_controller.num_actions,
                               device=scene.device)
    
    # ---------- 초기값 설정 ----------
    zero_joint_efforts = torch.zeros(scene.num_envs, n_j, device=sim.device)
    q_init   = robot.data.default_joint_pos.clone()

    # --------- 로봇 및 오브젝트 초기화 ---------
    joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_effort_target(zero_joint_efforts, joint_ids=joint_ids)
    robot.write_data_to_sim()

    object_root_state = object.data.default_root_state.clone()
    object_root_state[:, :3] += scene.env_origins
    object.write_root_pose_to_sim(object_root_state[:, :7])
    object.write_root_velocity_to_sim(object_root_state[:, 7:])

    object.reset()
    robot.reset()
    scene.reset()
    robot.update(sim_dt)

    # ============ Motion Planner Setting (RRT & IK) ===============
    # ------- TCP 6D Pose 정의 (월드 프레임, Root 프레임) -------
    root_start_w = robot.data.root_state_w[:, :7]
    hand_pos_w = robot.data.body_state_w[:, hand_link_idx, :7]
    tcp_start_pos_b = calculate_robot_tcp(hand_pos_w, root_start_w, offset)
    tcp_start_loc_w = root_start_w[:, :3] + quat_apply(root_start_w[:, 3:7], tcp_start_pos_b[:, :3])
    tcp_start_rot_w = quat_mul(root_start_w[:, 3:7], tcp_start_pos_b[:, 3:7])
    tcp_start_pos_w = torch.cat((tcp_start_loc_w, tcp_start_rot_w), dim=1)

    # ------- Target End-Effector Position 정의 --------
    object_pose_w = object.data.root_state_w[:, :7]
    object_loc_b, object_rot_b = subtract_frame_transforms(
         root_start_w[:, :3], root_start_w[:, 3:7], object_pose_w[:, :3], object_pose_w[:, 3:7])
    object_pose_b = torch.cat([object_loc_b, object_rot_b], dim=1)
    object_pose_b[:, 1] -= 0.6
    object_pose_b[:, 2] += 0.3
    
    # ------- Trajectory Planner : RRT -------
    motion_planner = RRTWrapper(start=tcp_start_pos_b.squeeze_(0), goal=object_pose_b.squeeze_(0), env=Env.Map3D(5, 5, 5), max_dist=0.1, num_traj_points=50)
    optimal_trajectory = motion_planner.plan()

    # ------- Motion Planner : Inverse Kinematics -------
    ik_commands[:, :3] = optimal_trajectory[0, :3]
    ik_commands[:, 3:] = optimal_trajectory[0, 3:7]
    diff_ik_controller.reset()
    diff_ik_controller.set_command(ik_commands, tcp_start_pos_b[:3], tcp_start_pos_b[3:7])



    # ========== Low-Level Controller (Joint Impadance Regulation) ===========
    # ------- Impedance Control Command 세팅 -------
    imp_commands[:, 7:14] = kp_table
    imp_commands[:, 14:] = zeta



    # ========== 제어 루프 ======================================
    i = 0
    while simulation_app.is_running():
        
        # --------- TCP의 6D Pose Error 세팅 ---------
        # Get the root & joint Pose in world frame
        root_pose_w = robot.data.root_state_w[:, :7]
        # Get the tcp pose in world frame
        hand_pos_w = robot.data.body_state_w[:, hand_link_idx, :7]
        hand_loc_b, hand_rot_b = subtract_frame_transforms(
            root_pose_w[:, :3], root_pose_w[:, 3:7], hand_pos_w[:, :3], hand_pos_w[:, 3:7])
        tcp_pose_b = calculate_robot_tcp(hand_pos_w, root_pose_w, offset)
        tcp_loc_w = root_pose_w[:, :3] + quat_apply(root_pose_w[:, 3:7], tcp_pose_b[:, :3])
        tcp_rot_w = quat_mul(root_pose_w[:, 3:7], tcp_pose_b[:, 3:7])
        tcp_pose_w = torch.cat((tcp_loc_w, tcp_rot_w), dim=1)
        # Compute the position error in the base frame
        tcp_pos_err_b = tcp_pose_b[:, :3] - ik_commands[:, :3]
        tcp_rot_err_b = quat_error_magnitude(tcp_pose_b[:, 3:7], ik_commands[:, 3:])
        # Visualization
        # hand_marker.visualize(hand_pos_w[:, :3], hand_pos_w[:, 3:7])
        tcp_marker.visualize(tcp_pose_w[:, :3], tcp_pose_w[:, 3:7])
        traj_marker.visualize(optimal_trajectory[:, :3] + robot.data.default_root_state[:, :3])
        goal_marker.visualize(object_pose_w[:, :3], object_pose_w[:, 3:7])

        # --------- Target Points 갱신 로직 ---------
        if torch.norm(tcp_pos_err_b) < 5e-2 and tcp_rot_err_b < 5e-2:
            i = (i+1) %  optimal_trajectory.shape[0]
            if i == 0:
                # Trajectory 끝에 도착하면, Reset the Scene
                print("[INFO]: Resetting robot state...")
                root_state = robot.data.default_root_state.clone()
                root_state[:, :3] += scene.env_origins
                joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                robot.set_joint_effort_target(zero_joint_efforts, joint_ids=joint_ids)

                object_root_state = object.data.default_root_state.clone()
                object_root_state[:, :3] += scene.env_origins
                object.write_root_pose_to_sim(object_root_state[:, :7])
                object.write_root_velocity_to_sim(object_root_state[:, 7:])
                robot.write_data_to_sim()

                object.reset()
                robot.reset()
                scene.reset()
                robot.update(sim_dt)
            # IK Command를 Next Point로 갱신
            ik_commands[:, :] = optimal_trajectory[i, :]
            diff_ik_controller.set_command(ik_commands, tcp_pose_b[:, :3], tcp_pose_b[:, 3:7])

        # --------- Controller 동작 (IK) -------------
        # Compute Jacobian
        jacobian = robot.root_physx_view.get_jacobians()[:, jacobi_idx, :, :num_active_joint]
        # Compute Jacobian for TCP Point
        jacobian_t = compute_frame_jacobian(robot, hand_rot_b, jacobian, offset)
        # Get the root & joint Pose in world frame
        joint_pos = robot.data.joint_pos[:, :num_active_joint]
        joint_vel = robot.data.joint_vel[:, :num_active_joint]
        # Compute the desired joint position using the IK and the end-effector pose from the base frame
        joint_pos_des = diff_ik_controller.compute(tcp_pose_b[:, :3], tcp_pose_b[:, 3:7], jacobian_t, joint_pos)

        # --------- Controller 동작 (Impedance) ---------
        # Set command for Impedance control
        imp_commands[:, :7] = joint_pos_des - joint_pos
        joint_imp_controller.set_command(command=imp_commands)
        # Target torques 계산 (중력, 관성, 코리올리 힘 보상)
        tau = joint_imp_controller.compute(
            dof_pos      = joint_pos,
            dof_vel      = joint_vel,
            mass_matrix  = robot.root_physx_view.get_generalized_mass_matrices()[:, :num_active_joint, :num_active_joint],
            gravity= robot.root_physx_view.get_gravity_compensation_forces()[:, :num_active_joint]
        )
        # Torque 신호 내부 버퍼에 저장
        robot.set_joint_effort_target(tau, joint_ids=robot_entity_cfg.joint_ids)
        # # 버퍼에 저장된 제어 신호 시뮬레이션에 모두 입력
        robot.write_data_to_sim()

        # 물리 엔진 스텝
        sim.step()
        scene.update(sim_dt)
        robot.update(sim_dt)
        

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=0.01)
    sim = SimulationContext(sim_cfg)
    sim_dt = sim.get_physics_dt()
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



def calculate_robot_tcp(hand_pos_w: torch.Tensor,
                        root_pos_w: torch.Tensor,
                        offset: torch.Tensor | None) -> torch.Tensor:
    
    hand_loc_b, hand_rot_b = subtract_frame_transforms(
        root_pos_w[:, :3], root_pos_w[:, 3:7], hand_pos_w[:, :3], hand_pos_w[:, 3:7])

    if offset is not None:
        tcp_loc_b, tcp_rot_b = combine_frame_transforms(
            hand_loc_b, hand_rot_b, offset[:, :3], offset[:, 3:7])
    else:
        tcp_loc_b = hand_loc_b; tcp_rot_b = hand_rot_b
    

    return torch.cat((tcp_loc_b, tcp_rot_b), dim=1)
    

def compute_skew_symmetric_matrix(vec: torch.Tensor) -> torch.Tensor:
    """Computes the skew-symmetric matrix of a vector.

    Args:
        vec: The input vector. Shape is (3,) or (N, 3).

    Returns:
        The skew-symmetric matrix. Shape is (1, 3, 3) or (N, 3, 3).

    Raises:
        ValueError: If input tensor is not of shape (..., 3).
    """
    # check input is correct
    if vec.shape[-1] != 3:
        raise ValueError(f"Expected input vector shape mismatch: {vec.shape} != (..., 3).")
    # unsqueeze the last dimension
    if vec.ndim == 1:
        vec = vec.unsqueeze(0)
    # create a skew-symmetric matrix
    skew_sym_mat = torch.zeros(vec.shape[0], 3, 3, device=vec.device, dtype=vec.dtype)
    skew_sym_mat[:, 0, 1] = -vec[:, 2]
    skew_sym_mat[:, 0, 2] = vec[:, 1]
    skew_sym_mat[:, 1, 2] = -vec[:, 0]
    skew_sym_mat[:, 1, 0] = vec[:, 2]
    skew_sym_mat[:, 2, 0] = -vec[:, 1]
    skew_sym_mat[:, 2, 1] = vec[:, 0]

    return skew_sym_mat

def compute_frame_jacobian(robot:Articulation, hand_rot_b: torch.Tensor, jacobian_w: torch.Tensor, offset:torch.Tensor) -> torch.Tensor:
    """Computes the geometric Jacobian of the target frame in the root frame.

    This function accounts for the target frame offset and applies the necessary transformations to obtain
    the right Jacobian from the parent body Jacobian.
    """
    # ========= 데이터 세팅 =========
    jacobian_b = jacobian_w
    root_quat = robot.data.root_quat_w
    root_rot_matrix = matrix_from_quat(quat_inv(root_quat))

    # ====== Hand Link의 Root Frame에서의 Jacobian 계산 ======
    jacobian_b[:, :3, :] = torch.bmm(root_rot_matrix, jacobian_b[:, :3, :])
    jacobian_b[:, 3:, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 3:, :])

    # ====== TCP의 Offset을 고려한 Frame Jacobian 보정 ======
    # ====== v_b = v_a + w * r_{ba} Kinematics 관계 반영 ======
    offset_b = quat_apply(hand_rot_b, offset[:, :3])
    s_offset = compute_skew_symmetric_matrix(offset_b[:, :3])
    jacobian_b[:, :3, :] += torch.bmm(-s_offset, jacobian_b[:, 3:, :])
    jacobian_b[:, 3:, :] = torch.bmm(matrix_from_quat(offset[:, 3:7]), jacobian_b[:, 3:, :])

    return jacobian_b



if __name__ == "__main__":
    main()
    simulation_app.close()

