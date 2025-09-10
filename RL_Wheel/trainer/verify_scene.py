import argparse
import torch

from isaaclab.app import AppLauncher
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext

# Add argparse arguments
parser = argparse.ArgumentParser(description="Verify GOAT scene in Isaac Sim")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import the user's abstract class and its config
from env.GOAT_base_env_cfg import GOATBaseEnvCfg

def run_simulator(sim: sim_utils.SimulationContext, scene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["cartpole"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    # Apply random action
    # -- generate random joint efforts
    efforts = torch.randn_like(robot.data.joint_pos) * 5.0
    # -- apply action to the robot
    robot.set_joint_effort_target(efforts)
    # -- write data to sim
    scene.write_data_to_sim()
    # Perform step
    sim.step()
    # Increment counter
    count += 1
    # Update buffers
    scene.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_cfg = CartpoleSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
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