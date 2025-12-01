import torch
import isaaclab.sim as sim_utils

from __future__ import annotations
from isaaclab.utils.math import normalize, quat_from_angle_axis
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
        self.DR_curriculum_level = 0
        self.task_curriculum_level = cfg.total_task_curriculum_level[0]
        self.total_DR_curriculum_level = cfg.total_DR_curriculum_level - 1

        # Noise curriculum (linear schedular)
        self.base_acceleration_noise_per = torch.linspace(start=0, end=cfg.max_base_acceleration_noise_per, steps=self.total_DR_curriculum_level)
        self.base_angular_vel_noise_per = torch.linspace(start=0, end=cfg.max_base_angular_vel_noise_per, steps=self.total_DR_curriculum_level)
        self.gravity_vector_noise_per = torch.linspace(start=0, end=cfg.max_gravity_vector_noise_per, steps=self.total_DR_curriculum_level)
        self.base_quaternion_noise_per = torch.linspace(start=0, end=cfg.max_base_quaternion_noise_per, steps=self.total_DR_curriculum_level)
        self.joint_pos_noise_per = torch.linspace(start=0, end=cfg.max_joint_pos_noise_per, steps=self.total_DR_curriculum_level)
        self.joint_vel_noise_per = torch.linspace(start=0, end=cfg.max_joint_vel_noise_per, steps=self.total_DR_curriculum_level)
        self.terrain_friction_random_per = torch.linspace(start=0, end=cfg.max_terrain_friction_random_per, steps=self.total_DR_curriculum_level)

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
        
        if self.task_curriculum_level == "balancing":
            # Domain randomization (initial pose)
            root_state = self._robot.data.default_root_state[env_ids].clone()
            root_state[:, 2] = 0.35 + torch.rand(len(env_ids), device=self.device) * 0.1
            root_state[:, 3:7] = random_orientation(len(env_ids), self.device)

            limits = self.joint_limits[env_ids]
            joint_pos = limits[:, 0] + torch.rand_like(limits[:, 0]) * (limits[:, 1] - limits[:, 0]) * 0.5
            joint_vel = torch.randn_like(joint_pos) * 0.1
        
        elif self.task_curriculum_level == "recovery":
            # Domain randomization (initial pose)
            root_state = self._robot.data.default_root_state[env_ids].clone()
            root_state[:, 2] = 0.7 + torch.rand(len(env_ids), device=self.device) * 0.1
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
        joint_pos_noise = self.joint_pos_noise_per(self.DR_curriculum_level)
        joint_vel_noise = self.joint_vel_noise_per(self.DR_curriculum_level)
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
        material_property = self._robot.root_physx_view.get_material_properties()
        self.friction_coefficient = torch.Tensor([material_property[:,0, 0], material_property[:, 0, 1]], device=self.device)

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

    def get_curriculum_quaternions(
        self,
        num_envs: int,
        current_level: int,
        total_levels: int,
        max_angle_rad: float = torch.pi,
        device: str = "cuda"
    ) -> torch.Tensor:
        """
        커리큘럼 레벨에 따라 선형적으로 증가하는 회전 각도를 가진 랜덤 쿼터니언을 반환합니다.

        Args:
            num_envs (int): 생성할 환경(쿼터니언)의 개수
            current_level (int): 현재 커리큘럼 레벨 (0부터 total_levels-1 까지)
            total_levels (int): 전체 레벨 수 (기본값: 5)
            max_angle_rad (float): 마지막 레벨에서의 최대 회전 각도 (라디안, 기본값: pi)
            device (str): 텐서가 생성될 디바이스

        Returns:
            torch.Tensor: 생성된 쿼터니언 (N, 4) - (w, x, y, z) 형식
        """
        # Level clipping
        if current_level >= total_levels:
            current_level = total_levels - 1
        
        level_scale = current_level / (total_levels - 1)
        current_angle_limit = max_angle_rad * level_scale
        random_angles = torch.rand(num_envs, device=self.device) * current_angle_limit

        # 4. 회전 축(Axis) 랜덤 샘플링
        # 3차원 공간의 랜덤 벡터 생성 후 정규화
        random_axes = torch.randn(num_envs, 3, device=device)
        random_axes = normalize(random_axes)

        # 5. Axis-Angle을 쿼터니언으로 변환
        # 제공된 math.py의 quat_from_angle_axis 함수 사용
        quaternions = quat_from_angle_axis(random_angles, random_axes)

        return quaternions

    def randomize_and_log_friction(
        self,
        env: GOATBaseEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        make_consistent: bool = False,
    ):
        # 1. 기존 Isaac Lab의 랜덤화 함수 호출 (실제 물리 적용)
        # 이 함수는 내부적으로 재질을 생성하고 할당합니다.
        # 이 클래스 인스턴스를 함수 내부에서 접근하기 어렵다면, 
        # mdp.randomize_rigid_body_material 로직을 직접 구현하여 값을 가로채야 합니다.
        
        # (간소화를 위해 직접 구현 로직의 핵심만 가져와 값을 저장하는 방식 예시)
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
            
        # 2. 랜덤 값 직접 샘플링 (저장을 위해)
        num_samples = len(env_ids)
        
        # Static Friction 샘플링
        s_range = torch.tensor(static_friction_range, device=env.device)
        static_fric = torch.rand(num_samples, device=env.device) * (s_range[1] - s_range[0]) + s_range[0]
        
        # Dynamic Friction 샘플링
        d_range = torch.tensor(dynamic_friction_range, device=env.device)
        dynamic_fric = torch.rand(num_samples, device=env.device) * (d_range[1] - d_range[0]) + d_range[0]
        
        if make_consistent:
            dynamic_fric = torch.min(static_fric, dynamic_fric)
            
        # 3. Env 변수에 저장 (이것이 핵심)
        # 환경 클래스에 미리 self.friction_coeffs = torch.zeros(...) 를 선언해두세요.
        if not hasattr(env, "friction_coeffs"):
            env.friction_coeffs = torch.zeros(env.num_envs, 2, device=env.device)
            
        env.friction_coeffs[env_ids, 0] = static_fric
        env.friction_coeffs[env_ids, 1] = dynamic_fric
        
        # 4. 실제 시뮬레이션에 적용
        # asset 가져오기
        asset = env.scene[asset_cfg.name]
        
        # PhysX View를 통해 재질 속성 설정
        # (주의: 기존 material 구조를 유지하려면 get_material_properties로 읽은 뒤 수정해서 set 해야 함)
        current_materials = asset.root_physx_view.get_material_properties()
        
        # 특정 환경들의 모든 Shape에 대해 마찰력 덮어쓰기
        # shape: (num_envs, num_shapes, 3)
        # env_ids에 해당하는 행의 0번(static), 1번(dynamic) 컬럼 업데이트
        # 여기서는 단순화를 위해 모든 shape에 동일 마찰력 적용 가정
        for i, env_id in enumerate(env_ids):
            current_materials[env_id, :, 0] = static_fric[i]
            current_materials[env_id, :, 1] = dynamic_fric[i]
            
        asset.root_physx_view.set_material_properties(current_materials, env_ids)