import torch
import isaaclab.sim as sim_utils

from __future__ import annotations
from isaaclab.utils.math import random_orientation
from isaaclab.terrains import TerrainImporterCfg
from .GOAT_PD_stand_env_cfg import GOATPDStandEnvCfg
from lib.env.GOAT_base_env import GOATBaseEnv
from lib.low_level_controller.pd_controller import PD_Controller


class GOATPDStandEnv(GOATBaseEnv):
    cfg: GOATPDStandEnvCfg

    def __init__(self, cfg: GOATPDStandEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.cfg = cfg
        self._robot = self.scene["robot"]
        
        # Curriculum level initialization
        self.curriculum_level = 0
        self.max_curriculum_level = cfg.max_curriculum_level - 1

        # Noise curriculum (linear schedular)
        self.base_acceleration_noise_per = torch.linspace(start=0, end=cfg.max_base_acceleration_noise_per, steps=self.max_curriculum_level)
        self.base_angular_vel_noise_per = torch.linspace(start=0, end=cfg.max_base_angular_vel_noise_per, steps=self.max_curriculum_level)
        self.gravity_vector_noise_per = torch.linspace(start=0, end=cfg.max_gravity_vector_noise_per, steps=self.max_curriculum_level)
        self.base_quaternion_noise_per = torch.linspace(start=0, end=cfg.max_base_quaternion_noise_per, steps=self.max_curriculum_level)
        self.joint_pos_noise_per = torch.linspace(start=0, end=cfg.max_joint_pos_noise_per, steps=self.max_curriculum_level)
        self.joint_vel_noise_per = torch.linspace(start=0, end=cfg.max_joint_vel_noise_per, steps=self.max_curriculum_level)
        self.terrain_friction_random_per = torch.linspace(start=0, end=cfg.max_terrain_friction_random_per, steps=self.max_curriculum_level)

        # Space initialization
        self.observation = torch.zeros((self.num_envs, self.cfg.observation_space), dtype=torch.float32, device=self.device)
        self.privileged_info = torch.zeros((self.num_envs, self.cfg.state_space - self.cfg.observation_space), dtype=torch.float32, device=self.device)
        self.state = torch.zeros((self.num_envs, self.cfg.state_space), dtype=torch.float32, device=self.device)

        # Torque controller initialization
        self.zero_joint_efforts = torch.zeros(self.num_envs, cfg.num_total_joints, device=self.device)
        self.leg_controller = PD_Controller(kp=self.cfg.kp,
                                            kd=self.cfg.kd,
                                            num_envs=self.num_envs,
                                            num_dof=self.cfg.leg_dof,
                                            device=self.device,
                                            dt=self.cfg.sim_dt)

        # HW limits
        self.joint_limits = self._robot.data.joint_pos_limits
        self.torque_limits = self._robot.data.joint_effort_limits

    def _reset_idx(self, env_ids: torch.Tensor):
        # TODO: curriculum scheduler만들기
        
        # Domain randomization (initial pose)
        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, 2] = 0.35 + torch.rand(len(env_ids), device=self.device) * 0.1
        root_state[:, 3:7] = random_orientation(len(env_ids), self.device)

        limits = self.joint_limits[env_ids]
        joint_pos = limits[:, 0] + torch.rand_like(limits[:, 0]) * (limits[:, 1] - limits[:, 0]) * 0.5
        joint_vel = torch.randn_like(joint_pos) * 0.1

        # Domain randomization (terrain friction)
        self.cfg.terrain = self.cfg.terrain.replace(
            physics_material=self.cfg.terrain.physics_material.replace(
                sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.7,
                    dynamic_friction=0.5,
                    restitution=0.0
            ),
            debug_vis=False)
        )

        # Publish to simulator
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

        # Domain randomization (sensor noise)
        joint_pos_noise = self.joint_pos_noise_per(self.curriculum_level)
        joint_vel_noise = self.joint_vel_noise_per(self.curriculum_level)
        self.joint_pos_noissy = self._add_gaussian_noise(joint_pos, joint_pos_noise)
        self.joint_vel_noissy = self._add_gaussian_noise(joint_vel, joint_vel_noise)

        self.torque_cmd = self.leg_controller.compute_torque(joint_pos=self.joint_pos_noissy,
                                                            joint_vel=self.joint_vel_noissy,
                                                            joint_pos_cmd=self.joint_pos_cmd,
                                                            joint_limits=self.joint_limits,
                                                            torque_limits=self.torque_limits)
        self._robot.set_joint_effort_target(self.torque_cmd)
        
        # TODO: wheel controller

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """
        Get sensor data with curriculum Gaussian noise

        Returns:
            dict[str, torch.Tensor]: Observation space, State(Privilege) space
        """
        # Observation data
        self.base_acceleration = self._robot.root_physx_view.get_link_accelerations()[:, 0, 3:]
        self.base_angular_vel = self._robot.root_physx_view.get_link_velocities()[:, 0, :3]
        self.gravity_vector = self._robot.data.projected_gravivity_b
        self.base_quaternion = self._robot.root_physx_view.get_root_transforms()[:, 3:]
        self.joint_pos = self._robot.data.joint_pos
        self.joint_vel = self._robot.data.joint_vel
        self.previous_action = self.actions.clone()

        # State(privileged) data
        self.base_vel = self._robot.root_physx_view.get_link_velocities()[:, 0, :3]
        self.base_height = self._robot.root_physx_view.get_root_transforms()[:, 2]
        self.contact_force      # TODO: add F/T sensor
        self.friction_coefficient = torch.Tensor([self.cfg.terrain.physics_material.static_friction, self.cfg.terrain.physics_material.dynamic_friction], device=self.device).repeat(self.num_envs, 1)

        # Domain randomization (sensor noise)
        self.base_acceleration_noissy = self._add_gaussian_noise(self.base_acceleration, self.base_acceleration_noise_per(self.curriculum_level))
        self.base_angular_vel_noissy = self._add_gaussian_noise(self.base_angular_vel, self.base_angular_vel_noise_per(self.curriculum_level))
        self.gravity_vector_noissy = self._add_gaussian_noise(self.gravity_vector, self.gravity_vector_noise_per(self.curriculum_level))
        self.base_quaternion_noissy = self._add_gaussian_noise(self.base_quaternion, self.base_quaternion_noise_per(self.curriculum_level))

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
        return {"policy": self.observation, "value": self.state}
    
    def _get_rewards(self) -> torch.Tensor:
        
        return torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
    
    def _get_dones(self): 
        terminated = False          # Continous task
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

