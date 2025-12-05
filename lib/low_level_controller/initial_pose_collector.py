import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Get initial poses for recovery task")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.utils.math import normalize, quat_from_angle_axis
from isaaclab.utils import configclass
from lib.env.GOAT_base_env_cfg import GOAT_Cfg
from Simulation.Tasks.GOAT_PD_stand.env.GOAT_PD_stand_env_cfg import GOATPDStandEnvCfg

@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Design the scene for drop robot."""
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
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),     # yes Gravity
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True, solver_position_iteration_count=4,
                    solver_velocity_iteration_count=0, fix_root_link=False               # Floating_base link
                )
            ),

            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.65),
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

def _get_curriculum_quaternions(
    cfg,
    level: int,
    num_envs: int,
    device
) -> torch.Tensor:
    """
    Random quaternion for base link pose

    Args:
        cfg: Direct RL Env cfg
        level (int): Domain randomization level
        num_envs (int): number of parallel environments
        device (str): device for pytorch

    Returns:
        torch.Tensor: Quaternion (N, 4) - (w, x, y, z) form
    """

    level_scale = level / (cfg.total_DR_curriculum_level - 1)
    current_angle_limit = torch.pi * level_scale
    random_angles = torch.rand(num_envs, device=device) * current_angle_limit

    random_axes = torch.randn(num_envs, 3, device=device)
    random_axes = normalize(random_axes)

    quaternions = quat_from_angle_axis(random_angles, random_axes)

    return quaternions

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene): 
    cfg = GOATPDStandEnvCfg()

    # define scene
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    episode_length = 140
    
    count = 0
    curriculum_level = 0
    total_episode = 2
    curriculum_episode = int(total_episode / cfg.total_DR_curriculum_level)

    # ---------- Environment Initialization ----------
    joint_limits = robot.data.joint_pos_limits
    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = robot.data.default_joint_vel.clone()
    root_state = robot.data.default_root_state.clone().expand(scene.num_envs, -1)
    zero_joint_efforts = torch.zeros(scene.num_envs, robot.num_joints, device=sim.device)

    collected_data = []

    print(f"[INFO] Starting collection: {total_episode} iterations x {scene.num_envs} envs")

    # -------------- Control loop --------------
    while simulation_app.is_running():
        
        robot.set_joint_effort_target(target=zero_joint_efforts)
        robot.write_data_to_sim()

        # Simulation step
        sim.step()
        robot.update(sim_dt)
        scene.update(sim_dt)

        count += 1
        print(f"progress: {count}")

        if count % curriculum_episode == 0:
            curriculum_level += 1
            if curriculum_level >= cfg.total_DR_curriculum_level:
                curriculum_level = cfg.total_DR_curriculum_level - 1

        # ================== Pose initialize sequence ================== #
        lower_limits = joint_limits[:, :, 0]
        upper_limits = joint_limits[:, :, 1]
        random_angle = lower_limits + torch.rand(scene.num_envs, robot.num_joints, device=sim.device) * (upper_limits - lower_limits)

        root_state[:, 3:7] = _get_curriculum_quaternions(cfg=cfg, level=curriculum_level,num_envs=scene.num_envs, device=sim.device)
        
        robot.set_joint_effort_target(zero_joint_efforts)
        robot.write_root_state_to_sim(root_state)
        robot.write_joint_state_to_sim(random_angle, default_joint_vel)
        
        robot.write_data_to_sim()
        robot.update(sim_dt)

        # ================== Drop sequence ================== #
        for i in range(episode_length):
            robot.set_joint_effort_target(zero_joint_efforts)
            robot.write_data_to_sim()
            sim.step()
        
        final_root_pose = robot.data.root_link_pose_w   # (N, 7) [pos, quat]
        final_joint_pos = robot.data.joint_pos          # (N, num_dof)
        
        # 텐서 결합
        data_to_save = torch.cat([final_root_pose, final_joint_pos], dim=-1)
        
        # CPU로 이동 및 numpy 변환
        data_np = data_to_save.cpu().numpy()
        collected_data.append(data_np)
        
        if count >= total_episode:
            break
        
    # ================== Save sequence ================== #
    print("[INFO] Saving all data to CSV...")
    all_data_np = np.concatenate(collected_data, axis=0) # (total_episode * num_envs, dims)
    
    header = "root_x,root_y,root_z,root_q_w,root_q_x,root_q_y,root_q_z,"
    for i in range(robot.num_joints):
        header += f",joint_{i}"
    
    np.savetxt("initial_pose_data.csv", all_data_np, delimiter=",", header=header, comments="")
    print("[INFO] Done!")

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
