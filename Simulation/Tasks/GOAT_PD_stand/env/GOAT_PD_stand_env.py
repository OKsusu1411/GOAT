import torch
import os
import numpy as np

from __future__ import annotations
from isaaclab.utils.math import normalize, quat_from_angle_axis
from isaaclab.terrains import TerrainImporterCfg
from .GOAT_PD_stand_env_cfg import GOATPDStandEnvCfg
from lib.env.GOAT_base_env import GOATBaseEnv
from lib.low_level_controller.pd_controller import PD_Controller, PI_Controller
from lib.utils.Running_mean_std import RunningMeanStd

csv_path = "initial_pose_data.csv"              # Path to csv file

class GOATPDStandEnv(GOATBaseEnv):
    cfg: GOATPDStandEnvCfg

    def __init__(self, cfg: GOATPDStandEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.cfg = cfg
        self._robot = self.scene["robot"]
        self._contact_sensor = self.scene["contact_sensor"]
        
        # Curriculum level initialization
        self.DR_curriculum_level = 0
        self.task_curriculum_level = 0
        self.total_task_curriculum_level = cfg.total_task_curriculum_level
        self.total_DR_curriculum_level = cfg.total_DR_curriculum_level - 1

        # Noise curriculum (linear schedular)
        self.base_acceleration_noise_per = torch.linspace(start=0, end=cfg.max_base_acceleration_noise_per, steps=cfg.total_DR_curriculum_level)
        self.base_angular_vel_noise_per = torch.linspace(start=0, end=cfg.max_base_angular_vel_noise_per, steps=cfg.total_DR_curriculum_level)
        self.gravity_vector_noise_per = torch.linspace(start=0, end=cfg.max_gravity_vector_noise_per, steps=cfg.total_DR_curriculum_level)
        self.base_quaternion_noise_per = torch.linspace(start=0, end=cfg.max_base_quaternion_noise_per, steps=cfg.total_DR_curriculum_level)
        self.joint_pos_noise_per = torch.linspace(start=0, end=cfg.max_joint_pos_noise_per, steps=cfg.total_DR_curriculum_level)
        self.joint_vel_noise_per = torch.linspace(start=0, end=cfg.max_joint_vel_noise_per, steps=cfg.total_DR_curriculum_level)
        self.terrain_friction_random_per = torch.linspace(start=0, end=cfg.max_terrain_friction_random_per, steps=cfg.total_DR_curriculum_level)
        self.terrain_restitution_random_per = torch.linspace(start=0, end=cfg.max_terrain_restitution_random_per, steps=cfg.total_DR_curriculum_level)

        # Space initialization
        self.observation = torch.zeros((self.num_envs, self.cfg.observation_space), dtype=torch.float32, device=self.device)
        self.privileged_info = torch.zeros((self.num_envs, self.cfg.state_space - self.cfg.observation_space), dtype=torch.float32, device=self.device)
        self.state = torch.zeros((self.num_envs, self.cfg.state_space), dtype=torch.float32, device=self.device)

        # Running mean std initialization (for normalization)
        self.observation_normalizer = RunningMeanStd(shape=self.cfg.observation_space, device=self.device)
        self.state_normalizer = RunningMeanStd(shape=self.cfg.state_space, device=self.device)

        # Torque controller initialization
        self.zero_joint_efforts = torch.zeros(self.num_envs, cfg.num_total_joints, device=self.device)
        self.leg_controller = PD_Controller(kp=self.cfg.joint_kp,
                                            kd=self.cfg.joint_kd,
                                            num_envs=self.num_envs,
                                            num_dof=self.cfg.leg_dof,
                                            num_leg=self.cfg.num_leg,
                                            device=self.device,
                                            dt=self.cfg.sim_dt)
        self.wheel_controller = PI_Controller(kp=self.cfg.wheel_kp,
                                              ki=self.cfg.wheel_ki,
                                              num_envs=self.num_envs,
                                              num_dof=self.cfg.num_leg,         # One wheel per legs
                                              num_leg=self.cfg.num_leg,
                                              device=self.device,
                                              dt=self.cfg.sim_dt)
        
        # TODO: HW limit이 제대로 안불러와지면 그냥 하드 코딩 ㄱㄱ
        # HW limits
        self.joint_pos_limits = self._robot.data.joint_pos_limits
        self.joint_vel_limits = self._robot.data.joint_vel_limits
        self.torque_limits = self._robot.data.joint_effort_limits

        if os.path.exists(csv_path):
            print(f"[INFO] Loading initial poses from {csv_path}...")
            
            data_np = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
            
            full_data = torch.tensor(data_np, dtype=torch.float32, device=self.device)
            
            # Data Slicing
            # [Column 0] Curriculum Level
            self.initial_pose_curriculum_level = full_data[:, 0]
            
            # [Column 1:4] Root Position (x, y, z)
            self.init_root_pos = full_data[:, 1:4]
            
            # [Column 4:8] Root Orientation (w, x, y, z)
            self.init_root_quat = full_data[:, 4:8]
            
            # [Column 8:] Joint Positions (joint_0 ~ joint_7)
            self.init_joint_pos = full_data[:, 8:]
            
            # Data length
            self.num_init_samples = full_data.shape[0]

    def _reset_idx(self, env_ids: torch.Tensor):
        
        if self.total_task_curriculum_level[self.task_curriculum_level] == "balancing":
            # Domain randomization (initial pose)
            root_state = self._robot.data.default_root_state[env_ids].clone()
            root_state[:, 2] = 0.7 + torch.rand(len(env_ids), device=self.device) * 0.1
        
        elif self.total_task_curriculum_level[self.task_curriculum_level] == "recovery":
            # Domain randomization (initial pose)
            root_state = self._robot.data.default_root_state[env_ids].clone()
            root_state[:, 2] = 0.35 + torch.rand(len(env_ids), device=self.device) * 0.1
            root_state[:, 3:7] = self._get_curriculum_quaternions(len(env_ids), self.device)

            limits = self.joint_pos_limits[env_ids]
            joint_pos = limits[:, 0] + torch.rand_like(limits[:, 0]) * (limits[:, 1] - limits[:, 0]) * 0.5
            joint_vel = torch.randn_like(joint_pos) * 0.1

            # Slicing indices based on curriculum level
            curriculum_ids = int(self.num_init_samples/self.total_DR_curriculum_level)
            start_ids = curriculum_ids*self.DR_curriculum_level
            end_ids = curriculum_ids*(self.DR_curriculum_level+1)

            # Random sampling
            random_initial_pos_ids = torch.randint(start_ids, end_ids, (len(env_ids),), device=self.device)
            
            # Extract initial pose data 
            sampled_root_pos = self.init_root_pos[random_initial_pos_ids].clone()
            sampled_root_quat = self.init_root_quat[random_initial_pos_ids].clone()
            sampled_joint_pos = self.init_joint_pos[random_initial_pos_ids].clone()
            
            # Zero Velocity
            joint_vel = torch.zeros_like(sampled_joint_pos)
            root_vel = torch.zeros(len(env_ids), 6, device=self.device)
            
            # Root State
            root_state = torch.cat([sampled_root_pos, sampled_root_quat, root_vel], dim=-1)

            # Publish to sim
            self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
            self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
            self._robot.write_joint_state_to_sim(sampled_joint_pos, joint_vel, env_ids)

        # elif self.total_task_curriculum_level[self.task_curriculum_level] == "random":
            
        # Domain randomization (terrain friction)
        material_property = self._robot.root_physx_view.get_material_properties()
        friction_noise = self.terrain_friction_random_per(self.DR_curriculum_level)
        restitution_noise = self.terrain_restitution_random_per(self.DR_curriculum_level)

        material_property[env_ids, :, 0] = self._add_gaussian_noise(self.cfg.default_terrain_static_friction, friction_noise).unsqueeze(1)
        material_property[env_ids, :, 1] = self._add_gaussian_noise(self.cfg.default_terrain_dynamic_friction, friction_noise).unsqueeze(1)
        material_property[env_ids, :, 2] = self._add_gaussian_noise(self.cfg.default_terrain_restitution, restitution_noise).unsqueeze(1)

        # Publish to sim
        self._robot.root_physx_view.set_material_properties(material_property, env_ids)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.write_root_state_to_sim()
        
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        Preprocessor that helps applying policy's action to simulation

        Args:
            actions (torch.Tensor): Joint pos command (angle), wheel's velocity for each legs in shape (num_envs, 2, 4)
        """
            
        # Refine command
        self.actions = actions.clone()
        self.joint_pos_cmd = self.actions[:, :, :3]
        self.wheel_cmd_vel = self.actions[:, :, 3:]
                
    def _apply_action(self):                    # Since it's inside the decimation loop, the low-level controller has to be located here
        # Current state
        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel

        # Domain randomization (sensor noise)set_material_properties
        joint_pos_noise = self.joint_pos_noise_per(self.DR_curriculum_level)
        joint_vel_noise = self.joint_vel_noise_per(self.DR_curriculum_level)
        self.joint_pos_noissy = self._add_gaussian_noise(joint_pos, joint_pos_noise)
        self.joint_vel_noissy = self._add_gaussian_noise(joint_vel, joint_vel_noise)

        self.joint_torque_cmd = self.leg_controller.compute_torque(joint_pos=self.joint_pos_noissy,
                                                                   joint_vel=self.joint_vel_noissy,
                                                                   joint_pos_cmd=self.joint_pos_cmd,
                                                                   joint_pos_limits=self.joint_pos_limits,
                                                                   torque_limits=self.torque_limits)
        
        self.wheel_torque_cmd = self.wheel_controller.compute_torque(joint_vel=self.joint_vel_noissy,
                                                                     joint_vel_cmd=self.wheel_cmd_vel,
                                                                     joint_vel_limits=self.joint_vel_limits,
                                                                     torque_limits=self.torque_limits)
        
        # Combine torque commands
        self.torque_cmd = torch.cat((self.joint_torque_cmd, self.wheel_torque_cmd), dim=1)
        
        # Load to sim buffer
        self._robot.set_joint_effort_target(self.torque_cmd)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """
        Get sensor data with curriculum Gaussian noise

        Returns:
            dict[str, torch.Tensor]: Observation space, State(Privilege) space
        """
        # Observation data
        self.base_acceleration = self._robot.root_physx_view.get_link_accelerations()[:, 0, 3:]
        self.base_angular_vel = self._robot.root_physx_view.get_link_velocities()[:, 0, :3]
        self.gravity_vector = self._robot.data.projected_gravity_b          # Unit vector
        self.base_quaternion = self._robot.root_physx_view.get_root_transforms()[:, 3:]
        self.joint_pos = self._robot.data.joint_pos
        self.joint_vel = self._robot.data.joint_vel
        self.previous_action = self.actions.clone()

        # State(privileged) data
        self.base_vel = self._robot.root_physx_view.get_link_velocities()[:, 0, :3]
        self.base_height = self._robot.root_physx_view.get_root_transforms()[:, 2]
        self.contact_force = self._contact_sensor.data.net_forces_w.view(self.num_envs, -1)
        material_property = self._robot.root_physx_view.get_material_properties()
        self.friction_coefficient = torch.Tensor([material_property[:, 0, 0], material_property[:, 0, 1]], device=self.device)

        # Domain randomization (sensor noise)
        self.base_acceleration_noissy = self._add_gaussian_noise(self.base_acceleration, self.base_acceleration_noise_per(self.DR_curriculum_level))
        self.base_angular_vel_noissy = self._add_gaussian_noise(self.base_angular_vel, self.base_angular_vel_noise_per(self.DR_curriculum_level))
        self.gravity_vector_noissy = self._add_gaussian_noise(self.gravity_vector, self.gravity_vector_noise_per(self.DR_curriculum_level))
        self.base_quaternion_noissy = self._add_gaussian_noise(self.base_quaternion, self.base_quaternion_noise_per(self.DR_curriculum_level))

        self.observation = torch.cat((self.base_acceleration_noissy,
                                      self.base_angular_vel_noissy,
                                      self.gravity_vector_noissy,
                                      self.base_quaternion_noissy,
                                      self.joint_pos_noissy,
                                      self.joint_vel_noissy,
                                      self.previous_action),
                                      dim=1)
        
        self.privileged_info = torch.cat((self.base_vel,
                                          self.base_height,
                                          self.contact_force,
                                          self.friction_coefficient),
                                          dim=1)
        
        self.state = torch.cat((self.observation, self.privileged_info), dim=1)

        # Normalize observation, state space
        self.normalized_observation = self.observation_normalizer.normalize(self.observation)
        self.normalized_state = self.state_normalizer.normalize(self.state)

        return {"policy": self.normalized_observation, "value": self.normalized_state}
    
    def _get_rewards(self) -> torch.Tensor:
        # Scheduler
        self.success_rate       # TODO: success rate랑 threshold 정의해야됨

        # Domain randomization curriculum
        if self.success_rate > self.threshold:
            self.DR_curriculum_level += 1
            if self.DR_curriculum_level > self.total_DR_curriculum_level:
                self.task_curriculum_level += 1         # I'm on the next level
                self.DR_curriculum_level = 0

        elif self.success_rate < self.threshold:
            self.DR_curriculum_level -= 1
            if self.DR_curriculum_level < 0:
                self.task_curriculum_level -= 1         # Downgrade
                self.DR_curriculum_level = 0

        # Task curriculum
        if self.task_curriculum_level > len(self.total_task_curriculum_level) - 1:      # Maximum level
            self.task_curriculum_level -= 1

        elif self.task_curriculum_level < 0:                                            # Lowest level
            self.task_curriculum_level = 0
        
        if self.total_task_curriculum_level[self.task_curriculum_level] == "balancing":
            height_error = self.base_height - self.cfg.target_height
        elif self.total_task_curriculum_level[self.task_curriculum_level] == "recovery":

        # elif self.total_task_curriculum_level[self.task_curriculum_level] == "random":

        return total_reward
    
    def _get_dones(self): 
        terminated = False          # No terminate condition
        truncated = self.episode_length_buf >= self.cfg.max_episode_length - 1
        return terminated, truncated
    
    ## ==================== Auxilliary functions ==================== ##
    def _add_gaussian_noise(self, data: torch.Tensor, noise_percentage: int) -> torch.Tensor:
        """
        Add (noise_percentage)% noise to all components of data
        """
        noise_ratio = noise_percentage / 100.0
        # Standard normal distribution 
        noise = torch.randn_like(data, device=self.device)
        noisy_data = data * (1 + noise_ratio * noise)
        
        return noisy_data

    def _get_curriculum_quaternions(
        self,
        num_envs: int
    ) -> torch.Tensor:
        """
        Random quaternion for base link pose

        Args:
            num_envs (int): number of parallel environments

        Returns:
            torch.Tensor: Quaternion (N, 4) - (w, x, y, z) form
        """
        
        level_scale = self.DR_curriculum_level / (self.total_DR_curriculum_level - 1)
        current_angle_limit = torch.pi * level_scale
        random_angles = torch.rand(num_envs, device=self.device) * current_angle_limit

        random_axes = torch.randn(num_envs, 3, device=self.device)
        random_axes = normalize(random_axes)

        quaternions = quat_from_angle_axis(random_angles, random_axes)

        return quaternions