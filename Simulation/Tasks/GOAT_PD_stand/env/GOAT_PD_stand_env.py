from __future__ import annotations

import torch
import os
import numpy as np

from isaaclab.utils.math import normalize, quat_from_angle_axis
from isaaclab.terrains import TerrainImporter 
from isaaclab.sensors import ContactSensor
from isaacsim.core.utils import bounds
from isaacsim.core.utils import prims
from .GOAT_PD_stand_env_cfg import GOATPDStandEnvCfg
from lib.env.GOAT_base_env import GOATBaseEnv
from lib.low_level_controller.joint_controller import PD_Controller, PI_Controller
from lib.utils.Running_mean_std import RunningMeanStd

csv_path = "initial_pose_data.csv"              # Path to csv file

class GOATPDStandEnv(GOATBaseEnv):
    cfg: GOATPDStandEnvCfg

    def __init__(self, cfg: GOATPDStandEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.cfg = cfg
        self._contact_sensor =  self.scene.sensors["contact_sensor"]
        
        # Curriculum level initialization
        self.env_DR_curriculum_level = torch.zeros((self.num_envs, 1), dtype=torch.int, device=self.device)         # DR level of each parallel environments
        self.DR_curriculum_level = 0
        self.task_curriculum_level = 0
        self.total_task_curriculum_level = cfg.total_task_curriculum_level
        self.total_DR_curriculum_level = cfg.total_DR_curriculum_level - 1
        self.rollout = 0
        self.success_rate_buffer = torch.zeros(self.num_envs, cfg.success_rate_buffer_len, dtype=torch.float, device=self.device)
        self.buffer_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.env_success_rate = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        
        self.previous_actioin = torch.zeros([self.num_envs, self.cfg.action_space], device=self.device)
        self.env_indices = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self.global_success_rate = 0.0

        # Noise curriculum (linear schedular)
        self.base_acceleration_noise_per = torch.linspace(start=0, end=cfg.max_base_acceleration_noise_per, steps=cfg.total_DR_curriculum_level, device=self.device)
        self.base_angular_vel_noise_per = torch.linspace(start=0, end=cfg.max_base_angular_vel_noise_per, steps=cfg.total_DR_curriculum_level, device=self.device)
        self.gravity_vector_noise_per = torch.linspace(start=0, end=cfg.max_gravity_vector_noise_per, steps=cfg.total_DR_curriculum_level, device=self.device)
        self.base_quaternion_noise_per = torch.linspace(start=0, end=cfg.max_base_quaternion_noise_per, steps=cfg.total_DR_curriculum_level, device=self.device)
        self.joint_pos_noise_per = torch.linspace(start=0, end=cfg.max_joint_pos_noise_per, steps=cfg.total_DR_curriculum_level, device=self.device)
        self.joint_vel_noise_per = torch.linspace(start=0, end=cfg.max_joint_vel_noise_per, steps=cfg.total_DR_curriculum_level, device=self.device)
        self.terrain_friction_random_per = torch.linspace(start=0, end=cfg.max_terrain_friction_random_per, steps=cfg.total_DR_curriculum_level, device=self.device)
        self.terrain_restitution_random_per = torch.linspace(start=0, end=cfg.max_terrain_restitution_random_per, steps=cfg.total_DR_curriculum_level, device=self.device)

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
                                              num_dof=1,         # One wheel per legs
                                              num_leg=self.cfg.num_leg,
                                              device=self.device,
                                              dt=self.cfg.sim_dt)
        
        # HW limits
        self.joint_pos_limits = self._robot.data.joint_pos_limits
        self.joint_vel_limits = self._robot.data.joint_vel_limits
        self.joint_input_limits = self.cfg.joint_input_limits.unsqueeze(0).expand(self.num_envs, -1, -1).to(device=self.device)
        self.torque_limits = self.cfg.torque_limits.unsqueeze(0).expand(self.num_envs, -1).to(device=self.device)              # Isaac sim cannot bring torque limits from urdf

        if os.path.exists(csv_path):
            print(f"[INFO] Loading initial poses from {csv_path}...")
            
            data_np = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
            
            full_data = torch.tensor(data_np, dtype=torch.float32, device=self.device)
            
            # Data Slicing
            # Curriculum Level
            self.initial_pose_curriculum_level = full_data[:, 0]
            
            # Root Position (x, y, z)
            self.init_root_pos = full_data[:, 1:4]
            
            # Root Orientation (w, x, y, z)
            self.init_root_quat = full_data[:, 4:8]
            
            # Joint Positions (joint_0 ~ joint_7)
            self.init_joint_pos = full_data[:, 8:]
            
            # Data length
            self.num_init_samples = full_data.shape[0]
    
    def _setup_scene(self):
        super()._setup_scene()

        self.terrain = TerrainImporter(self.cfg.terrain_importer_cfg)
        self.cfg.dome_light_cfg.spawn.func(self.cfg.dome_light_cfg.prim_path,
                                           self.cfg.dome_light_cfg.spawn)
        
        # Compute collision box info
        robot_prim_path = "/World/envs/env_0/Robot"
        robot_bbox_cache = bounds.create_bbox_cache()
        robot_aabb = bounds.compute_aabb(bbox_cache=robot_bbox_cache,
                                         prim_path=robot_prim_path,
                                         include_children=True)
        self.robot_collision_min_z = - robot_aabb[2]

        # Spawn contact sensor
        contact_sensor = ContactSensor(cfg=self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = contact_sensor

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)

        # Update success rate for each environment
        episode_scores = torch.mean(self.success_rate_buffer[env_ids], dim=1)
        self.env_success_rate[env_ids] = episode_scores

        # Reset success rate buffer
        self.success_rate_buffer[env_ids] = 0.0
        self.buffer_ids[env_ids] = 0

        # Reset previous action observation
        self.actions[env_ids] = torch.zeros_like(self.actions[env_ids], device=self.device)

        if self.total_task_curriculum_level[self.task_curriculum_level] == "balancing":
            # Domain randomization (initial pose)
            # Base link state
            root_state = self._robot.data.default_root_state[env_ids].clone()
            root_state[:, 2] += self.robot_collision_min_z

            # Joint state
            limits = self.joint_pos_limits[env_ids]
            joint_pos = torch.zeros_like(limits[:, :, 0]) * (limits[:, :, 1] - limits[:, :, 0])
            joint_vel = torch.zeros_like(joint_pos)

        elif self.total_task_curriculum_level[self.task_curriculum_level] == "recovery":
            # Domain randomization (initial pose)
            # Extract initial pose data from csv file
            target_indices = torch.nonzero(self.initial_pose_curriculum_level == self.DR_curriculum_level, as_tuple=True)[0]

            if len(target_indices) == 0:
                target_indices = torch.arange(self.num_init_samples, device=self.device)

            random_idx = torch.randint(0, len(target_indices), (len(env_ids),), device=self.device)
            random_ids = target_indices[random_idx]

            # Base link state
            root_pos = self.init_root_pos[random_ids].clone()
            root_quat = self.init_root_quat[random_ids].clone()
            root_vel = torch.zeros(len(env_ids), 6, device=self.device)
            root_state = torch.cat([root_pos, root_quat, root_vel], dim=-1)

            # Joint state
            joint_pos = self.init_joint_pos[random_ids, :].clone()
            joint_vel = torch.zeros_like(joint_pos)

        # Change to global position
        root_state[:,:3] += self.scene.env_origins[env_ids]

        # DR_curriculum update for each environment
        self.env_DR_curriculum_level[env_ids] = self.DR_curriculum_level
        
        # Domain randomization (terrain friction)
        material_property = self._robot.root_physx_view.get_material_properties()
        friction_noise = self.terrain_friction_random_per[self.DR_curriculum_level]
        restitution_noise = self.terrain_restitution_random_per[self.DR_curriculum_level]

        material_property[env_ids, :, 0] = self._add_gaussian_noise(self.cfg.default_terrain_static_friction, friction_noise)
        material_property[env_ids, :, 1] = self._add_gaussian_noise(self.cfg.default_terrain_dynamic_friction, friction_noise)
        material_property[env_ids, :, 2] = self._add_gaussian_noise(self.cfg.default_terrain_restitution, restitution_noise)

        # Publish to sim
        # self._robot.root_physx_view.set_material_properties(material_property, env_ids)
        self._robot.write_joint_state_to_sim(position=joint_pos,
                                             velocity=joint_vel,
                                             env_ids=env_ids)
        self._robot.write_root_state_to_sim(root_state=root_state,
                                            env_ids=env_ids)
        
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        Preprocessor that helps applying policy's action to simulation

        Args:
            actions (torch.Tensor): Joint pos command (angle), wheel's velocity for each legs in shape (num_envs, 2, 4)
        """
        
        # Refine command
        self.actions = actions.clone()
        self.joint_pos_delta_cmd = self.actions[:, :-2] * self.cfg.joint_action_weight
        self.wheel_vel_cmd = self.actions[:, -2:] * self.cfg.wheel_action_weight
        
    def _apply_action(self):                    # Since it's inside the decimation loop, the low-level controller has to be located here
        # Current state
        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel
 
        # Domain randomization (sensor noise)set_material_properties
        joint_pos_noise = self.joint_pos_noise_per[self.env_DR_curriculum_level]
        joint_vel_noise = self.joint_vel_noise_per[self.env_DR_curriculum_level]
        self.joint_pos_noissy = self._add_gaussian_noise(joint_pos, joint_pos_noise)
        self.joint_vel_noissy = self._add_gaussian_noise(joint_vel, joint_vel_noise)

        # Made joint command
        self.joint_pos_cmd = self.joint_pos_noissy[:, :-2] + self.joint_pos_delta_cmd

        self.joint_torque_cmd = self.leg_controller.compute_torque(joint_pos=self.joint_pos_noissy,
                                                                   joint_vel=self.joint_vel_noissy,
                                                                   joint_pos_cmd=self.joint_pos_cmd,
                                                                   joint_pos_limits=None,
                                                                   torque_limits=self.torque_limits)
        
        self.wheel_torque_cmd = self.wheel_controller.compute_torque(joint_vel=self.joint_vel_noissy,
                                                                     joint_vel_cmd=self.wheel_vel_cmd,
                                                                     joint_vel_limits=self.joint_vel_limits,
                                                                     torque_limits=self.torque_limits)
        
        # Combine torque commands
        self.torque_cmd = torch.cat((self.joint_torque_cmd, self.wheel_torque_cmd), dim=1)
        
        # print(self.joint_pos_cmd[0,:])
        # print(self.torque_cmd[0,:])
        
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
        self.gravity_vector = self._robot.data.projected_gravity_b                                      # Unit vector
        self.base_quaternion = self._robot.root_physx_view.get_root_transforms()[:, 3:]
        self.joint_pos = self._robot.data.joint_pos
        self.joint_vel = self._robot.data.joint_vel
        self.previous_action = self.actions.clone()
        self.flat_previous_action = self.previous_action.view(self.num_envs, -1)

        # State(privileged) data
        self.base_vel = self._robot.root_physx_view.get_link_velocities()[:, 0, :3]
        self.base_height = self._robot.root_physx_view.get_root_transforms()[:, 2].unsqueeze(1)
        self.contact_force = self._contact_sensor.data.net_forces_w.view(self.num_envs, -1)
        material_property = self._robot.root_physx_view.get_material_properties()                   # device is "cpu" not "cuda" 
        self.friction_coefficient = torch.stack([material_property[:, 0, 0], material_property[:, 0, 1]], dim=-1).to(self.device)
        
        # Domain randomization (sensor noise)
        self.base_acceleration_noissy = self._add_gaussian_noise(self.base_acceleration, self.base_acceleration_noise_per[self.env_DR_curriculum_level])
        self.base_angular_vel_noissy = self._add_gaussian_noise(self.base_angular_vel, self.base_angular_vel_noise_per[self.env_DR_curriculum_level])
        self.gravity_vector_noissy = self._add_gaussian_noise(self.gravity_vector, self.gravity_vector_noise_per[self.env_DR_curriculum_level])
        self.base_quaternion_noissy = self._add_gaussian_noise(self.base_quaternion, self.base_quaternion_noise_per[self.env_DR_curriculum_level])
        self.joint_pos_noissy = self._add_gaussian_noise(self.joint_pos, self.joint_pos_noise_per[self.env_DR_curriculum_level])
        self.joint_vel_noissy = self._add_gaussian_noise(self.joint_vel, self.joint_vel_noise_per[self.env_DR_curriculum_level])

        self.observation = torch.cat((self.base_acceleration_noissy,
                                      self.base_angular_vel_noissy,
                                      self.gravity_vector_noissy,
                                      self.base_quaternion_noissy,
                                      self.joint_pos_noissy,
                                      self.joint_vel_noissy,
                                    #   self.flat_previous_action),
                                     ),
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
        # ======================= Scheduler ======================= #
        current_time = self.episode_length_buf.float()
        # Target gravity in base frame (Upright state = [0, 0, -1])
        target_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        
        # Upright_rate (1.0: upright properly, -1.0: upside down)
        upright_rate = torch.sum(self.gravity_vector * target_gravity, dim=1)       # Dot product
        
        # boolean for success measure
        is_upright = upright_rate > (self.cfg.upright_threshold * torch.pi / 180)
        is_height_reached = torch.abs(self.base_height - self.cfg.target_height) < self.cfg.height_threshold
        
        # Velocity criteria (Only strict for balancing)
        lin_vel_norm = torch.norm(self.base_vel, dim=1)                             # L2 norm 
        ang_vel_norm = torch.norm(self.base_angular_vel, dim=1)
        is_stable = (lin_vel_norm < 0.5) & (ang_vel_norm < 1.0)

        is_upright = is_upright.view(-1)
        is_height_reached = is_height_reached.view(-1)
        is_stable = is_stable.view(-1)
        current_task_name = self.total_task_curriculum_level[self.task_curriculum_level]

        # Task-specific success definition
        if current_task_name == "balancing":
            # Balancing: must be upright, at height, and stable
            success_measure = is_upright & is_height_reached & is_stable
        elif current_task_name == "recovery":
            # Recovery: Just need to get up (velocity constraint is relaxed)
            success_measure = is_upright & is_height_reached

        # Update success rate buffer
        step_success_float = success_measure.float()
        self.success_rate_buffer[self.env_indices, self.buffer_ids] = step_success_float      # Stack success_rate
        self.buffer_ids = (self.buffer_ids + 1) % self.cfg.success_rate_buffer_len            # Update index
        
        # Compute global success rate
        num_successful_envs = torch.sum(self.env_success_rate > 0.8)
        self.global_success_rate = num_successful_envs / self.num_envs

        # Level adjustment by curriculum
        # if self.global_success_rate > self.cfg.curriculum_level_up_threshold:
        #     self.DR_curriculum_level += 1
        #     print("Level up!!")
        #     if self.DR_curriculum_level >= self.total_DR_curriculum_level:
        #         self.task_curriculum_level += 1         # I'm on the next level
        #     self.global_success_rate = 0

        # elif self.global_success_rate < self.cfg.curriculum_level_down_threshold:
        #     self.DR_curriculum_level -= 1
        #     print("Level Down!!")
        #     if self.DR_curriculum_level < 0:
        #         self.task_curriculum_level -= 1         # Downgrade
        #     self.global_success_rate = 0

        # Clipping
        self.task_curriculum_level = max(0, min(self.task_curriculum_level, len(self.total_task_curriculum_level) - 1))
        self.DR_curriculum_level = max(0, min(self.DR_curriculum_level, self.total_DR_curriculum_level - 1))
        # print(f"DR: {self.DR_curriculum_level},     Task: {self.cfg.total_task_curriculum_level[self.task_curriculum_level]}")
        
        # ======================= Reward ======================= #
        # Orientation Reward (Projected Gravity Alignment) [Highest Priority]
        orient_error = torch.norm(self.gravity_vector - target_gravity, dim=1)
        r_orient = torch.exp(-torch.square(orient_error) / 0.3)                                       # Raidial Basis FUnction (RBF)

        # Base Height Reward
        height_error = torch.norm(self.base_height - self.cfg.target_height, dim=1)
        r_height = torch.exp(-torch.square(height_error) / 0.3)
        
        # vel_penalty_scale = torch.clamp(upright_rate, 0.0, 1.0)                                     # Clamp the rate
        # vel_penalty_scale = torch.pow(vel_penalty_scale, 4)                                         # Make it sharper (only active when really it's upright)

        # r_vel_lin = -torch.sum(torch.abs(self.base_vel), dim=1) * vel_penalty_scale                 # Penalty
        # r_vel_ang = -torch.sum(torch.abs(self.base_angular_vel), dim=1) * vel_penalty_scale         # Penalty
        # r_vel_joint = -torch.sum(torch.abs(self.joint_vel[:, :-2]), dim=1) * vel_penalty_scale      # Penalty

        vel_lin_error = torch.norm(-self.base_vel, dim=1)
        r_vel_lin = torch.exp(-torch.square(vel_lin_error) / 0.5)

        vel_ang_error = torch.norm(-self.base_angular_vel, dim=1)
        r_vel_ang = torch.exp(-torch.square(vel_ang_error) / 0.5)

        vel_joint_error = torch.norm(-self.joint_vel, dim=1)
        r_vel_joint = torch.exp(-torch.square(vel_joint_error) / 1)

        # Energy / Action Smoothness
        r_effort = -torch.sum(torch.abs(self.torque_cmd), dim=1)                                 # Penalty
        
        r_terminated = - self.reset_terminated.float()

        r_alive = self.cfg.r_alive_weight * current_time/1000

        # Total Reward Summation
        total_reward = (
            self.cfg.r_orient_weight * r_orient +
            self.cfg.r_height_weight * r_height +
            self.cfg.r_vel_lin_weight * r_vel_lin +
            self.cfg.r_vel_ang_weight * r_vel_ang +
            self.cfg.r_vel_joint_weight * r_vel_joint +
            self.cfg.r_effort_weight * r_effort +
            self.cfg.r_terminated_weight * r_terminated +
            self.cfg.r_alive_weight * r_alive
        )

        return total_reward
    
    def _get_dones(self): 
        tilt_threshold_rad = torch.tensor(self.cfg.base_tilt_reset_condition, device=self.device) * torch.pi / 180.0
        cos_threshold = torch.cos(tilt_threshold_rad)

        target_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device)
        base_tilt = torch.sum(self.gravity_vector * target_gravity, dim=1)

        terminated = (self.base_height < self.cfg.height_reset_condition) | (base_tilt < cos_threshold).unsqueeze(-1)
        terminated = terminated.squeeze(-1)


        truncated = self.episode_length_buf >= (self.cfg.max_episode_length - 1)

        return terminated, truncated
    
    ## ==================== Auxilliary functions ==================== ##
    def _add_gaussian_noise(self, data: torch.Tensor | float, noise_percentage: torch.Tensor) -> torch.Tensor:
        """
        Add (noise_percentage)% noise to all components of data
        """
        noise_ratio = noise_percentage / 100.0
        # Standard normal distribution
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32, device=self.device)

        noise = torch.randn_like(data, device=self.device)
        noisy_data = data * (1 + noise_ratio * noise)
        
        return noisy_data

    def _get_curriculum_quaternions(self, num_envs: int) -> torch.Tensor:
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
    
    def get_rollout(self, rollout: int):
        """
        Get rollout number of agent
        
        Args:
            rollout (int): rollout number 
        """
        self.rollout = rollout