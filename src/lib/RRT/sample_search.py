"""
@file: graph_search.py
@breif: Base class for planner based on graph searching
@author: Winter
@update: 2023.1.17
"""
import numpy as np
from itertools import combinations
import math
from .planner import Planner
from lib.utils.Node import Node
from lib.utils.Env import Map3D

class SampleSearcher(Planner):
    """
    Base class for planner based on sample searching.

    Parameters:
        start (tuple): start point coordinate
        goal (tuple): goal point coordinate
        env (Map): environment
    """
    def __init__(self, start: tuple, goal: tuple, env: Map3D, delta: float=0.5) -> None:
        super().__init__(start, goal, env)
        # inflation bias
        self.delta = delta

    def isCollision(self, node1: Node, node2: Node) -> bool:
        """
        Judge collision when moving from node1 to node2.

        Parameters:
            node1 (Node): node 1
            node2 (Node): node 2

        Returns:
            collision (bool): True if collision exists else False
        """
        if self.isInsideObs(node1) or self.isInsideObs(node2):
            return True

        for rect in self.env.obs_rect:
            if self.isInterRect(node1, node2, rect):
                return True

        for circle in self.env.obs_circ:
            if self.isInterCircle(node1, node2, circle):
                return True

        return False

    def isInsideObs(self, node: Node) -> bool:
        """
        Judge whether a node inside tht obstacles or not.

        Parameters:
            node (Node): node

        Returns:
            inside (bool): True if inside the obstacles else False
        """
        x, y, z = node.current

        for (ox, oy, oz, r) in self.env.obs_circ:
            if math.hypot(x - ox, y - oy, z - oz) <= r + self.delta:
                return True

        for (ox, oy, oz, w, d, h) in self.env.obs_rect:
            if 0 <= x - (ox - self.delta) <= w + 2 * self.delta \
                and 0 <= y - (oy - self.delta) <= d + 2 * self.delta \
                and 0 <= z - (oz - self.delta) <= h + 2 * self.delta:
                return True

        for (ox, oy, oz, w, d, h) in self.env.boundary:
            if 0 <= x - (ox - self.delta) <= w + 2 * self.delta \
                and 0 <= y - (oy - self.delta) <= d + 2 * self.delta \
                and 0 <= z - (oz - self.delta) <= h + 2 * self.delta:
                return True

        return False

    def isInterRect(self, node1: Node, node2: Node, rect: list) -> bool:
        # obstacle and it's vertex
        ox, oy, oz, w, d, h = rect
        
        xmin, xmax = ox - self.delta, ox + w + self.delta
        ymin, ymax = oy - self.delta, oy + d + self.delta
        zmin, zmax = oz - self.delta, oz + h + self.delta
        
        x0, y0, z0 = node1.current
        x1, y1, z1 = node2.current
        dx, dy, dz = x1-x0, y1-y0, z1-z0

        t_enter, t_exit = 0.0, 1.0
        
        for p0, dp, pmin, pmax in ( (x0, dx, xmin, xmax),
                                    (y0, dy, ymin, ymax),
                                    (z0, dz, zmin, zmax),):
            # 평행
            if abs(dp) < 1e-12:               # 선분이 축과 평행
                if p0 < pmin or p0 > pmax:    # 평행 + 박스 밖 → 교차 X
                    return False
                continue                      # 평행 + 박스 안 → 다음 축 검사

            # 축과 만나는 두 파라미터 t
            t1 = (pmin - p0) / dp
            t2 = (pmax - p0) / dp
            if t1 > t2:                       # 정렬
                t1, t2 = t2, t1

            # 세 축(t1, t2) 중 가장 ‘늦게 들어온’ t와 가장 ‘빨리 나간’ t 계산
            t_enter = max(t_enter, t1)
            t_exit  = min(t_exit,  t2)

            if t_enter > t_exit:              # 상자를 완전히 통과하지 못함
                return False

        # t_exit >= 0, t_enter <= 1 조건은 위 과정에서 자동 보장
        return True

    def isInterCircle(self, node1: Node, node2: Node, circle: list) -> bool:
        # obstacle
        ox, oy, r = circle

        # origin
        x, y = node1.current

        # direction
        dx = node2.x - node1.x
        dy = node2.y - node1.y
        d  = [dx, dy]
        d2 = np.dot(d, d)

        if d2 == 0:
            return False

        # projection
        t = np.dot([ox - x, oy - y], d) / d2
        if 0 <= t <= 1:
            shot = Node((x + t * dx, y + t * dy), None, None, None)
            center = Node((ox, oy), None, None, None)
            if self.dist(shot, center) <= r + self.delta:
                return True

        return False