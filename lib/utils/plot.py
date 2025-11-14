"""
Plot tools 3‑D
Adapted from the original 2‑D implementation by Huiming Zhou.
This version visualises a 3‑D RRT / RRT* search on a ``Map3D`` environment.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import itertools
import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3‑D projection)
from matplotlib.patches import FancyArrowPatch
import numpy as np

from .Env import Env, Map3D
from .Node import Node


class Plot3D:
    """Utility class for 3‑D visualisation of a planning episode."""

    #: how often to call ``plt.pause`` when streaming *expand* set
    STREAM_STEP = 10

    def __init__(self, start: Tuple[float, float, float],
                 goal: Tuple[float, float, float],
                 env: Env):
        self.start: Node = Node(start, start, 0, 0)
        self.goal: Node = Node(goal, goal, 0, 0)
        self.env: Env = env

        self.fig = plt.figure("planning‑3d")
        # --- the only 3‑D axes ---
        self.ax: Axes3D = self.fig.add_subplot(111, projection="3d")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

    # ------------------------------------------------------------------
    #  Public entry point – animation wrapper
    # ------------------------------------------------------------------
    def animation(self,
                  path: List[Tuple[float, float, float]] | None,
                  title: str,
                  cost: float | None = None,
                  expand: List[Node] | None = None,
                  cost_curve: Sequence[float] | None = None) -> None:
        if cost is not None:
            title += f"  |  cost: {cost:.2f}"
        self.plot_environment(title)

        if expand:
            self.plot_expanded(expand)
        if path:
            self.plot_path(path)
        if cost_curve:
            self._plot_cost_curve(cost_curve, title)

        plt.show()

    # ------------------------------------------------------------------
    #  Environment & obstacles
    # ------------------------------------------------------------------
    def plot_environment(self, title: str) -> None:
        """Draw start, goal and all static obstacles."""
        # start / goal markers
        self.ax.scatter(*self.start.current, marker="s", c="#ff0000", s=40, depthshade=False)
        self.ax.scatter(*self.goal.current,  marker="s", c="#1155cc", s=40, depthshade=False)

        if isinstance(self.env, Map3D):
            # boundary & rectangular obstacles are both cuboids → same helper
            for cuboid in itertools.chain(self.env.boundary, self.env.obs_rect):
                self._draw_cuboid(*cuboid, color="k", alpha=0.7)

            # circular obstacles are treated as vertical cylinders -> approximate by 16‑gon walls
            for (ox, oy, oz, r, h) in self.env.obs_circ:
                self._draw_cylinder(ox, oy, oz, r, h, color="gray", alpha=0.25)

        self.ax.set_title(title)
        self.ax.set_box_aspect([1, 1, 1])  # equal aspect ratio

    # ------------------------------------------------------------------
    #  Expand set visualisation
    # ------------------------------------------------------------------
    def plot_expanded(self, expanded: List[Node]) -> None:
        """Stream the tree expansion in 3‑D."""
        counter = 0
        for nd in expanded:
            if nd.parent is None:
                continue
            parent: Node = nd.parent
            self.ax.plot3D([parent[0], nd.x], [parent[1], nd.y], [parent[2], nd.z],
                           c="#cccccc", linewidth=0.6)
            counter += 1
            if counter % self.STREAM_STEP == 0:
                plt.pause(0.001)
        plt.pause(0.01)

    # ------------------------------------------------------------------
    #  Path
    # ------------------------------------------------------------------
    def plot_path(self, path: Sequence[Tuple[float, float, float]], *,
                  color: str = "#13ae00", linestyle: str = "-", linewidth: float = 2.0) -> None:
        xs, ys, zs = zip(*path)
        self.ax.plot3D(xs, ys, zs, linestyle=linestyle, linewidth=linewidth, c=color)
        # emphasise start/goal again to draw over path
        self.ax.scatter(*self.start.current, marker="s", c="#ff0000", s=40, depthshade=False)
        self.ax.scatter(*self.goal.current,  marker="s", c="#1155cc", s=40, depthshade=False)

    # ------------------------------------------------------------------
    #  Aux cost curve window
    # ------------------------------------------------------------------
    @staticmethod
    def _plot_cost_curve(cost_vals: Sequence[float], title: str) -> None:
        plt.figure("cost curve")
        plt.plot(cost_vals, c="b")
        plt.xlabel("epoch")
        plt.ylabel("cost")
        plt.title(title)
        plt.grid(True)

    # ------------------------------------------------------------------
    #  Basic shapes helpers
    # ------------------------------------------------------------------
    def _draw_cuboid(self,
                     ox: float, oy: float, oz: float,
                     w: float, d: float, h: float,
                     *, color: str = "k", alpha: float = 0.1) -> None:
        """Wireframe of a cuboid given lower‑near‑left corner and size."""
        # eight vertices
        x_faces = [ox, ox + w]
        y_faces = [oy, oy + d]
        z_faces = [oz, oz + h]
        verts = list(itertools.product(x_faces, y_faces, z_faces))

        # 12 edges as pairs of vertex indices
        idx_pairs = [(0, 1), (0, 2), (0, 4),
                     (3, 1), (3, 2), (3, 7),
                     (5, 1), (5, 4), (5, 7),
                     (6, 2), (6, 4), (6, 7)]
        for i, j in idx_pairs:
            sx, sy, sz = verts[i]
            ex, ey, ez = verts[j]
            self.ax.plot3D([sx, ex], [sy, ey], [sz, ez], c=color, alpha=alpha)

    def _draw_cylinder(self, cx: float, cy: float, z0: float,
                       r: float, h: float,
                       *, color: str = "gray", alpha: float = 0.25, sides: int = 16) -> None:
        """Approximate a vertical cylinder by wall polygons."""
        theta = np.linspace(0, 2 * math.pi, sides)
        xs = cx + r * np.cos(theta)
        ys = cy + r * np.sin(theta)
        zs_low = np.full_like(xs, z0)
        zs_high = np.full_like(xs, z0 + h)
        for i in range(sides - 1):
            self.ax.plot3D([xs[i], xs[i + 1]], [ys[i], ys[i + 1]], [zs_low[i], zs_low[i + 1]], c=color, alpha=alpha)
            self.ax.plot3D([xs[i], xs[i + 1]], [ys[i], ys[i + 1]], [zs_high[i], zs_high[i + 1]], c=color, alpha=alpha)
            self.ax.plot3D([xs[i], xs[i]], [ys[i], ys[i]], [zs_low[i], zs_high[i]], c=color, alpha=alpha)

    # ------------------------------------------------------------------
    #  Misc API parity helpers (no‑ops or 2‑D specific stubs)
    # ------------------------------------------------------------------
    def clean(self):
        self.ax.cla()
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

    def update(self):
        self.fig.canvas.draw_idle()

    def connect(self, event_name: str, func):
        self.fig.canvas.mpl_connect(event_name, func)
