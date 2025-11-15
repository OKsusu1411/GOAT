"""
@file: planner.py
@breif: Abstract class for planner
@author: Winter
@update: 2023.1.17
"""
import math
from abc import abstractmethod, ABC
from lib.utils.Node import Node
from lib.utils.Env import Map3D

class Planner(ABC):
    def __init__(self, start: tuple, goal: tuple, env: Map3D) -> None:
        # plannig start and goal
        self.start = Node(start, start, 0, 0)
        self.goal = Node(goal, goal, 0, 0)
        # environment
        self.env = env

    def dist(self, node1: Node, node2: Node) -> float:
        dx, dy, dz = node2.x - node1.x, node2.y - node1.y, node2.z - node1.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    def unit_vec(self, node1: Node, node2: Node, std=1e-5) -> tuple[float, float, float]:
        dx, dy, dz = node2.x - node1.x, node2.y - node1.y, node2.z - node1.z
        norm = math.sqrt(dx*dx + dy*dy + dz*dz)
        norm += std
        return dx/norm, dy/norm, dz/norm
    

    @abstractmethod
    def plan(self):
        '''
        Interface for planning.
        '''
        pass

    @abstractmethod
    def run(self):
        '''
        Interface for running both plannig and animation.
        '''
        pass