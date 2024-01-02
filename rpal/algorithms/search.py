from enum import Enum
from pathlib import Path
from typing import Union, Tuple

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
    def __init__(self, pcd, grid_size=0.005):
        self.grid = SurfaceGridMap(pcd, grid_size=grid_size)
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
        grid_size=0.005,
        **kwargs
    ):
        self.grid = SurfaceGridMap(pcd, grid_size=grid_size)
        self.kernel = SquaredExpKernel(scale=scale)
        self.prev_idx = None
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
            self.algo = BayesianOptimization(self.grid, self.kernel)
        else:
            raise RuntimeError("Invalid algo!")

    def next(self):
        if self.prev_idx is None:
            optim_idx = self.grid.sample_uniform()
        else:
            optim_idx = self.algo.get_optimal_state(self.prev_idx, self.prev_value)
        self.prev_idx = optim_idx
        self.history.append((optim_idx, self.algo.grid_mean.copy()))
        pt, norm = self.grid.idx_to_pt(optim_idx)
        return (pt, norm)

    def update_outcome(self, val: float):
        print("val", val)
        self.prev_value = val


if __name__ == "__main__":
    from rpal.utils.constants import *
    from rpal.utils.pcd_utils import scan2mesh, mesh2roi, visualize_pcds
    from rpal.utils.useful_poses import O_T_CAM, O_xaxis
    from rpal.utils.transform_utils import quat2mat
    from scipy.spatial.transform import Rotation

    np.random.seed(100)
    pcd = o3d.io.read_point_cloud(str(SURFACE_SCAN_PATH))
    surface_mesh = scan2mesh(pcd)
    roi_pcd = mesh2roi(surface_mesh)
    planner = RandomSearch(roi_pcd, grid_size=0.001)
    planner.grid.visualize(show_tf=True)
    # planner = ActiveSearch(ActiveSearchAlgos.BO, roi_pcd, 2)
    palp_Ts = []
    surf_norms = []
    for i in range(100):
        palp = planner.next()
        planner.update_outcome(1.0)

        T = np.eye(4)
        O_R_E = quat2mat(GT_SCAN_POSE[-4:])
        result = Rotation.align_vectors(
            np.array([palp[1], np.array([0, -1, 0])]),
            np.array([[0, 0, -1], np.array([0, 1, 0])]),
            weights=np.array([10, 0.1]),
        )
        print(result[1])
        T[:3, :3] = result[0].as_matrix()
        T[:3, 3] = palp[0]
        palp_Ts.append(T.copy())
        surf_norms.append(palp)

    ani = HeatmapAnimation(planner.history, planner.history[0][1])
    ani.visualize()

    roi_pcd.paint_uniform_color([1, 0, 0])

    gt_scan = o3d.io.read_point_cloud(str(GT_PATH))
    visualize_pcds(
        [planner.grid._grid_pcd, gt_scan],
        meshes=[surface_mesh],
        tfs=palp_Ts,
        surf_norms=surf_norms,
    )
