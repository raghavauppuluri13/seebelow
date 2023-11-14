import numpy as np
from typing import Union
from enum import Enum
import open3d as o3d
from pathlib import Path
from rpal.algorithms.grid import SurfaceGridMap
from rpal.algorithms.gp import SquaredExpKernel
from rpal.algorithms.active_area_search import ActiveAreaSearch
from rpal.algorithms.bayesian_optimization import BayesianOptimization


class RandomSearch:
    def __init__(self, pcd):
        self.pcd = pcd
        self.grid = SurfaceGridMap(pcd, grid_size=0.001)
        self.pcd_len = len(np.asarray(pcd.points))

    def next(self):
        i = np.random.randint(0, self.pcd_len - 1)
        pt = self.pcd.points[i]
        norm = self.pcd.normals[i]
        return (pt, norm)


class ActiveSearchAlgos:
    AAS = 1
    BO = 2


class ActiveSearch:
    def __init__(self, pcd: o3d.geometry.PointCloud, algo: ActiveSearchAlgos, **kwargs):
        self.pcd = pcd
        self.pcd_len = len(np.asarray(pcd.points))
        self.grid = SurfaceGridMap(pcd, grid_size=0.001)
        self.kernel = SquaredExpKernel(scale=kwargs["scale"])
        self.prev_pt = None
        self.prev_value = None

        if algo is ActiveSearchAlgos.AAS:
            qt_dim = max(self.grid.shape)
            qt_dim += 10
            qt_dim = (qt_dim // 10) * 10
            self.group_quadtree = QuadTree(qt_dim, qt_dim, group_dim, group_dim)
            self.algo = ActiveAreaSearch(
                self.grid, self.group_quadtree, kernel, **kwargs
            )
        elif algo is ActiveSearchAlgos.BO:
            bo = BayesianOptimization(grid, kernel)
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
        pt, norm = self.grid.idx_to_pt(optim_idx)
        return (pt, norm)


if __name__ == "__main__":
    pts = np.random.uniform(0, 1, size=(100, 3))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.estimate_normals()
    planner = Random(pcd)
    planner_as = ActiveSearch(pcd, ActiveSearchAlgos.BO)
    print(planner.next())
