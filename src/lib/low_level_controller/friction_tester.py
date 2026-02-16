import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="PD torque control data collector.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to spawn.")
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
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),     # zero-G
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

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene): 
    # define scene
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    
    leg_dof = 3                         # hip, thigh, knee joints
    n_leg_j = leg_dof * 2
    num_total_joints = n_leg_j + 2

    # ---------- Environment Initialization ----------
    sim_len = 10.0  # [s] simulation length
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

    # reset state
    print("[INFO] Reset state for plotting...")

    # Joint reset to initial pose
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
    # Joint torque reset to 0
    robot.set_joint_effort_target(zero_joint_efforts)
    robot.write_data_to_sim()
    robot.reset()
    robot.update(sim_dt)

    # Joint reset to initial pose
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
    # Joint torque reset to 0
    robot.set_joint_effort_target(zero_joint_efforts)
    robot.write_data_to_sim()
    robot.reset()
    robot.update(sim_dt) # Populate initial data

    # --- Static Friction Test ---
    # Apply a range of torques to the test joint across all environments
    torque_to_apply = zero_joint_efforts.clone()
    test_joint = 2  # Corresponds to 'hip_R_Joint'
    test_torques = torch.linspace(0, 0.3, steps=scene.num_envs, device=scene.device)
    torque_to_apply[:, test_joint] = test_torques
    
    robot.set_joint_effort_target(torque_to_apply)
    robot.write_data_to_sim()
    
    # Step the simulation to see the effect of the torque
    sim.step()
    
    # Update the robot state to read the results (including acceleration)
    robot.update(sim_dt)
    
    # Now, read the joint acceleration from the robot's data buffer
    joint_acc = robot.data.joint_acc
    
    # --- Plot Torque vs. Acceleration ---
    # Move data to CPU for plotting
    applied_torques_np = test_torques.cpu().numpy()
    measured_accel_np = joint_acc[:, test_joint].cpu().numpy()

    # Create the plot
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(applied_torques_np, measured_accel_np, s=10, alpha=0.7)
    
    # Add a horizontal line at y=0 for reference
    ax.axhline(0, color='red', linestyle='--', linewidth=0.8)
    
    # Labels and Title
    ax.set_title(f"Torque vs. Acceleration for Joint {test_joint}")
    ax.set_xlabel("Applied Torque (Nm)")
    ax.set_ylabel("Resulting Joint Acceleration (rad/s^2)")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Show plot
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
