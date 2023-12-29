from enum import Enum
from pathlib import Path
from typing import Union

import numpy as np
import open3d as o3d

from rpal.algorithms.active_area_search import ActiveAreaSearch
from rpal.algorithms.bayesian_optimization import BayesianOptimization
from rpal.algorithms.gp import SquaredExpKernel
from rpal.algorithms.grid import SurfaceGridMap
from rpal.algorithms.gui import HeatmapAnimation


class GridVisualizer:
    def __init__(self, grid):
        self.buffer = []
        self.grid = grid

    def visualize(self, save=False, save_path="gaussian_process.mp4"):
        ani.visualize()

    def save(self, save_path="gaussian_process.mp4"):
        ani.save_animation(save_path)

    def add(self, optimal_state):
        self.buffer.append((optimal_state, grid.copy()))


class RandomSearch:
    def __init__(self, pcd):
        self.grid = SurfaceGridMap(pcd, grid_size=0.001)
        self.history = []
        self.X_visited = []
        self.next_state = None

    def update_outcome(self, prev_value):
        assert self.next_state is not None
        self.grid.grid[self.next_state] = prev_value

    def next(self):
        self.next_state = self.grid.sample_uniform(X_visited=np.array(self.X_visited))
        pt, norm = self.grid.idx_to_pt(tuple(self.next_state))
        self.history.append((self.next_state, self.grid.grid.copy()))
        self.X_visited.append(self.next_state)
        return (pt, norm)


class ActiveSearchAlgos:
    AAS = 1
    BO = 2


class ActiveSearch:
    def __init__(
        self,
        algo: ActiveSearchAlgos,
        pcd: o3d.geometry.PointCloud,
        scale: float,
        **kwargs
    ):
        self.grid = SurfaceGridMap(pcd, grid_size=0.001)
        self.kernel = SquaredExpKernel(scale=scale)
        self.prev_pt = None
        self.prev_value = None
        self.algo = None
        self.history = []

        if algo is ActiveSearchAlgos.AAS:
            qt_dim = max(self.grid.shape)
            qt_dim += 10
            qt_dim = (qt_dim // 10) * 10
            self.group_quadtree = QuadTree(qt_dim, qt_dim, group_dim, group_dim)
            self.algo = ActiveAreaSearch(
                self.grid, self.group_quadtree, kernel, **kwargs
            )
        elif algo is ActiveSearchAlgos.BO:
            self.algo = BayesianOptimization(grid, kernel)
        else:
            raise RuntimeError("Invalid algo!")

    def update_outcome(self, prev_pt, prev_value):
        self.prev_pt = prev_pt
        self.prev_value = prev_value

    def next(self):
        if self.prev_pt is None or self.prev_value is None:
            prev_idx = self.grid.sample_states_uniform()
        else:
            prev_idx = self.grid.pt_to_idx(self.prev_pt)
        optim_idx = self.algo.get_optimal_state(prev_idx, self.prev_value)
        self.history.append((optim_idx, self.algo.grid_mean.copy()))
        pt, norm = self.grid.idx_to_pt(optim_idx)
        return (pt, norm)


if __name__ == "__main__":
    from rpal.utils.constants import *
    from rpal.utils.pcd_utils import scan2mesh, mesh2roi

    pcd = o3d.io.read_point_cloud(str(SURFACE_SCAN_PATH))
    surface_mesh = scan2mesh(pcd)
    roi_pcd = mesh2roi(surface_mesh, bbox_pts=BBOX_ROI)
    planner = RandomSearch(roi_pcd)
    print(planner.next())  # TODO: fix this
    planner_as = ActiveSearch(ActiveSearchAlgos.BO, roi_pcd, 0.1)
    print(planner_as.next())
