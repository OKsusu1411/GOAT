import torch
import numpy as np
from typing import List, Tuple
from .RRT_star import RRTStar
from lib.utils import Env
from scipy.interpolate import splprep, splev


class RRTWrapper:
    """
    Wrapper for RRT* motion planning.

    """
    def __init__(self, start: torch.Tensor, goal: torch.Tensor, env: Env.Map3D, max_dist: float = 0.05, num_traj_points: int = 50):
        self.env = env
        self.start = start
        self.goal = goal
        self.start_trans_tuple = tuple(self.start[:3].tolist())
        self.goal_trans_tuple = tuple(self.goal[:3].tolist())
        self.start_rot = self.start[3:7]
        self.goal_rot = self.goal[3:7]   
        self.planner = RRTStar(start=self.start_trans_tuple, goal=self.goal_trans_tuple, env=env, max_dist=max_dist)
        self.num_traj_points = num_traj_points
    
    def plan(self) -> torch.Tensor:
        """
        Plan a path from start to goal using RRT*.

        Parameters:
            start (torch.Tensor): Start point coordinates.
            goal (torch.Tensor): Goal point coordinates.

        Returns:
            torch.Tensor: Planned path as a tensor of shape (N, 3).
        """
        _, path_trans, _ = self.planner.plan()
        path_trans = torch.flip(torch.as_tensor(path_trans, dtype=self.start.dtype, device=self.start.device), dims=(0,))
        path_trans = self.smoothing(path_trans, num_points=self.num_traj_points)

        path_rot = self.slerp(self.start_rot, self.goal_rot, path_trans.shape[0])

        path = torch.cat((
            path_trans,
            path_rot.squeeze_(1)
        ), dim=-1)

        return path


    def slerp(self, q0: torch.Tensor, q1: torch.Tensor, num_points: int) -> torch.Tensor:
        """
        Spherical linear interpolation between two points.

        Parameters:
            start (tuple): Start point coordinates.
            end (tuple): End point coordinates.
            num_points (int): Number of points to interpolate.

        Returns:
            List[Tuple[float, float, float]]: Interpolated points.
        """
        t = torch.linspace(0, 1, num_points, device=q0.device)

        dot_product = torch.sum(q0 * q1, dim=-1)
        q1 = torch.where(dot_product < 0, -q1, q1)
        dot_product = dot_product.abs()

        theta_0 = torch.acos(dot_product)
        sin_0 = torch.sin(theta_0)

        factor0 = torch.sin((1 - t) * theta_0) / sin_0
        factor1 = torch.sin(t * theta_0) / sin_0

        return (factor0.unsqueeze_(-1) * q0 + factor1.unsqueeze_(-1) * q1)


    def smoothing(self, path: torch.Tensor, num_points: int = 50, degree: int = 3) -> torch.Tensor:
        """
        Smooth the path using a simple averaging method.

        Parameters:
            path (torch.Tensor): Path to smooth.

        Returns:
            torch.Tensor: Smoothed path.
        """
        assert path.dim() == 2 and path.shape[1] == 3, "Path must be a 2D tensor with shape (N, 3)."

        xyz = path.detach().cpu().numpy()

        tck, _  = splprep(xyz.T, s=0, k=degree)

        u = torch.linspace(0, 1, num_points, device=path.device).cpu().numpy()
        xyz_new = np.vstack(splev(u, tck)).T


        return torch.as_tensor(xyz_new, dtype=path.dtype, device=path.device)
        


    def run(self):
        self.planner.run()
  