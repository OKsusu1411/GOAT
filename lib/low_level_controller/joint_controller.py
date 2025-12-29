import torch

class PD_Controller():
    """
    τ = Kp*e + Kd*ė
    """
    def __init__(self, kp, kd, num_envs: int, num_dof: int, num_leg: int, device: str, dt: float):
        """
        PD Controller initialization for position control

        Args:
            kp (float or torch.Tensor): Proportional gain. float value or (1, num_dof) size tensor.
            kd (float or torch.Tensor): Derivative gain. float value or (1, num_dof) size tensor.
            num_envs (int): Number of pararell training environments.
            num_dof (int): controllable dof per leg.
            num_leg (int): Number of legs
            device (str): "cuda:0" or "cpu".
            dt (float): Simulation time-step.
        """
        self.device = device
        self.num_envs = num_envs
        self.num_dof = num_dof
        self.num_leg = num_leg
        self.dt = dt
        self.old_torque = torch.zeros(self.num_envs, num_dof * num_leg, device=self.device)

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

    def compute_torque(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        joint_pos_cmd: torch.Tensor,
        joint_pos_limits: torch.Tensor,
        torque_limits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute joint input torque using PD controller.

        Args:
            joint_pos (torch.Tensor): Current joint position [rad].
            joint_vel (torch.Tensor): Current joint velocity [rad/s].
            joint_pos_cmd (torch.Tensor): Reference joint pos(angle) [num_env, num_leg, num_joint].
            joint_pos_limits (torch.Tensor): Joint position limits [num_env, num_joint, 2].
            torque_limits (torch.Tensor): Joint torque limits [num_env, num_joint].

        Returns:
            torch.Tensor: Joint torque.
        """
        # Robot dof
        leg_dof = self.num_dof                          # hip, thigh, knee joints
        num_joint = leg_dof * self.num_leg

        # Define joint indices for each leg
        # Isaac sim's Joint order: ['hip_L_Joint', 'hip_R_Joint', 'thigh_L_Joint', 'thigh_R_Joint', 'knee_L_Joint', 'knee_R_Joint', 'wheel_L_Joint', 'wheel_R_Joint']
        left_leg_indices = torch.tensor([0, 2, 4], device=self.device, dtype=torch.long)
        right_leg_indices = torch.tensor([1, 3, 5], device=self.device, dtype=torch.long)
        torque_limits = torque_limits[:, :num_joint]    # Extract joint torque limits
        joint_pos_limits = joint_pos_limits[:, :num_joint]

        # --- Left Leg slicing ---
        joint_pos_left = torch.index_select(joint_pos, 1, left_leg_indices)
        joint_vel_left = torch.index_select(joint_vel, 1, left_leg_indices)
        joint_pos_cmd_left = torch.index_select(joint_pos_cmd, 1, left_leg_indices)
        joint_limits_left = torch.index_select(joint_pos_limits, 1, left_leg_indices)

        # --- Right Leg slicing ---
        joint_pos_right = torch.index_select(joint_pos, 1, right_leg_indices)
        joint_vel_right = torch.index_select(joint_vel, 1, right_leg_indices)
        joint_pos_cmd_right = torch.index_select(joint_pos_cmd, 1, right_leg_indices)
        joint_limits_right = torch.index_select(joint_pos_limits, 1, right_leg_indices)

        # Left foot PD control
        joint_pos_cmd_left = torch.clamp(joint_pos_cmd_left, joint_limits_left[:, :, 0], joint_limits_left[:, :, 1])            # Clipping joint position command
        joint_pos_left_error = joint_pos_cmd_left - joint_pos_left
        joint_vel_left_error = - joint_vel_left                                                                                 # reference joint velocity = 0
        torque_left = self.kp * joint_pos_left_error + self.kd * joint_vel_left_error

        # Right foot PD control
        joint_pos_cmd_right = torch.clamp(joint_pos_cmd_right, joint_limits_right[:, :, 0], joint_limits_right[:, :, 1])        # Clipping joint position command
        joint_pos_right_error = joint_pos_cmd_right - joint_pos_right
        joint_vel_right_error = - joint_vel_right                                                                               # reference joint velocity = 0
        torque_right = self.kp * joint_pos_right_error + self.kd * joint_vel_right_error
        
        # Combine torque inputs
        torque = torch.zeros(self.num_envs, num_joint, device=self.device)
        torque.scatter_(1, left_leg_indices.repeat(self.num_envs, 1), torque_left)
        torque.scatter_(1, right_leg_indices.repeat(self.num_envs, 1), torque_right)
        
        # LPF for torque
        torque = 0.951 * self.old_torque + (1 - 0.951) * torque
        self.old_torque = torque.clone()

        # Clip torque based on torque_limits
        torque = torch.clamp(torque, -torque_limits, torque_limits)
        
        return torque
    
class PI_Controller():
    def __init__(self, kp, ki, num_envs: int, num_dof: int, num_leg: int, device: str, dt: float):
        """
        PI Controller initialization  for velocity control

        Args:
            kp (float or torch.Tensor): Proportional gain. float value or (1, num_dof) size tensor.
            ki (float or torch.Tensor): Intergral gain. float value or (1, num_dof) size tensor.
            num_envs (int): Number of pararell training environments.
            num_dof (int): controllable dof per leg.
            num_leg (int): Number of legs
            device (str): "cuda:0" or "cpu".
            dt (float): Simulation time-step.
        """
        self.device = device
        self.num_envs = num_envs
        self.num_dof = num_dof
        self.num_leg = num_leg
        self.dt = dt
        self.old_torque = torch.zeros(self.num_envs, num_dof * num_leg, device=self.device)

        # kp gain
        if isinstance(kp, float):
            self.kp = torch.full((num_envs, num_dof), kp, device=device)
        elif isinstance(kp, torch.Tensor):
            if kp.shape != (1, num_dof):
                raise ValueError(f"kp tensor must have shape (1, {num_dof}), but got {kp.shape}")
            self.kp = kp.to(device).expand(num_envs, -1)        # Expand as num_envs
        else:
            raise TypeError("kp must be a float or a torch.Tensor of shape (1, num_dof)")

        # ki gain
        if isinstance(ki, float):
            self.ki = torch.full((num_envs, num_dof), ki, device=device)
        elif isinstance(ki, torch.Tensor):
            if ki.shape != (1, num_dof):
                raise ValueError(f"ki tensor must have shape (1, {num_dof}), but got {ki.shape}")
            self.ki = ki.to(device).expand(num_envs, -1)        # Expand as num_envs
        else:
            raise TypeError("ki must be a float or a torch.Tensor of shape (1, num_dof)")
    
    def compute_torque(
        self,
        joint_vel: torch.Tensor,
        joint_vel_cmd: torch.Tensor,
        joint_vel_limits: torch.Tensor,
        torque_limits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute joint input torque using PI controller.

        Args:
            joint_vel (torch.Tensor): Current joint velocity [rad/s].
            joint_vel_cmd (torch.Tensor): Reference joint velocity(rad/s) [num_env, num_leg, num_joint].
            joint_limits (torch.Tensor): Joint position limits [num_env, num_joint, 2].
            torque_limits (torch.Tensor): Joint torque limits [num_env, num_joint].

        Returns:
            torch.Tensor: Joint torque.
        """
        # Robot dof
        leg_dof = self.num_dof                          # wheel joint
        num_joint = leg_dof * self.num_leg

        # Define joint indices for each leg
        # Isaac sim's Joint order: ['hip_L_Joint', 'hip_R_Joint', 'thigh_L_Joint', 'thigh_R_Joint', 'knee_L_Joint', 'knee_R_Joint', 'wheel_L_Joint', 'wheel_R_Joint']
        left_leg_indices = torch.tensor([6], device=self.device, dtype=torch.long)
        right_leg_indices = torch.tensor([7], device=self.device, dtype=torch.long)
        torque_limits = torque_limits[:, 6:]            # Extract wheel torque limits

        # --- Left Leg slicing ---
        joint_vel_left = torch.index_select(joint_vel, 1, left_leg_indices)
        joint_vel_cmd_left = joint_vel_cmd[:, 0].unsqueeze(-1)
        joint_vel_limits_left = torch.index_select(joint_vel_limits, 1, left_leg_indices)

        # --- Right Leg slicing ---
        joint_vel_right = torch.index_select(joint_vel, 1, right_leg_indices)
        joint_vel_cmd_right = joint_vel_cmd[:, 1].unsqueeze(-1)
        joint_vel_limits_right = torch.index_select(joint_vel_limits, 1, right_leg_indices)

        # Left foot PI control
        joint_vel_cmd_left = torch.clamp(joint_vel_cmd_left, - joint_vel_limits_left, joint_vel_limits_left)            # Clipping joint position command
        joint_vel_left_error = joint_vel_cmd_left - joint_vel_left
        torque_left = self.kp * joint_vel_left_error + self.ki * joint_vel_left_error * self.dt                                         # PI feedback

        # Right foot PI control
        joint_vel_cmd_right = torch.clamp(joint_vel_cmd_right, - joint_vel_limits_right, joint_vel_limits_right)        # Clipping joint position command
        joint_vel_right_error = joint_vel_cmd_right - joint_vel_right
        torque_right = self.kp * joint_vel_right_error + self.ki * joint_vel_right_error * self.dt                                      # PI feedback
        
        # Combine torque inputs
        torque = torch.zeros(self.num_envs, num_joint, device=self.device)
        torque = torch.cat((torque_left, torque_right), dim=1)
        
        # LPF for torque
        torque = 0.951 * self.old_torque + (1 - 0.951) * torque
        self.old_torque = torque.clone()

        # Clip torque based on torque_limits
        torque = torch.clamp(torque, -torque_limits, torque_limits)
        
        return torque