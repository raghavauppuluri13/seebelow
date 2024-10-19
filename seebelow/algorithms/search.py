from pathlib import Path
import numpy as np
import open3d as o3d

from rpal.algorithms.active_area_search import ActiveAreaSearch
from rpal.algorithms.bayesian_optimization import BayesianOptimization
from rpal.algorithms.gp import SquaredExpKernel
from rpal.algorithms.grid import SurfaceGridMap
from rpal.algorithms.gui import HeatmapAnimation
import rpal.utils.constants as rpal_const


class SearchHistory:
    def __init__(self):
        self._history = []

    def add(self, next_state: tuple, grid: np.ndarray):
        self._history.append(
            np.array(
                (np.array(next_state), grid),
                dtype=rpal_const.HISTORY_DTYPE(grid.shape),
            )
        )

    def save(self, folder: Path):
        np.save(str(folder / "search_history.npy"), self.history)

    @property
    def history(self):
        return np.array(self._history)


class Search:
    def __init__(self):
        pass

    def update_outcome(self, prev_value):
        raise NotImplementedError

    def next(self):
        raise NotImplementedError


class RandomSearch(Search):
    def __init__(self, surface_grid_map):
        self.grid = surface_grid_map
        self.next_state = None

    def update_outcome(self, val: float):
        assert self.next_state is not None
        self.grid.update(self.next_state, val)

    def next(self):
        self.next_state = self.grid.sample_uniform(from_unvisited=True)
        pt, norm = self.grid.idx_to_pt(tuple(self.next_state))
        return (pt, norm)

    @property
    def grid_estimate(self):
        return self.next_state, self.grid.grid


class ActiveSearchAlgos:
    AAS = 1
    BO = 2


class ActiveSearch(Search):
    def __init__(
        self,
        algo: ActiveSearchAlgos,
        surface_grid_map: SurfaceGridMap,
        kernel_scale: float,
        **kwargs
    ):
        self.grid = surface_grid_map
        self.kernel = SquaredExpKernel(scale=kernel_scale)
        self.next_state = None
        self.algo = None

        if algo is ActiveSearchAlgos.AAS:
            raise NotImplementedError
            qt_dim = max(self.grid.shape)
            qt_dim += 10
            qt_dim = (qt_dim // 10) * 10
            self.group_quadtree = QuadTree(qt_dim, qt_dim, group_dim, group_dim)
            self.algo = ActiveAreaSearch(
                self.grid, self.group_quadtree, self.kernel, **kwargs
            )
        elif algo is ActiveSearchAlgos.BO:
            self.algo = BayesianOptimization(self.grid, self.kernel)
        else:
            raise RuntimeError("Invalid algo!")

    def next(self):
        if self.next_state is None:
            self.next_state = self.grid.sample_uniform(from_unvisited=True)
        else:
            self.next_state = self.algo.get_optimal_state()
        pt, norm = self.grid.idx_to_pt(self.next_state)
        return (pt, norm)

    def update_outcome(self, val: float):
        self.grid.update(self.next_state, val)

    @property
    def grid_estimate(self):
        """
        Returns
        -------
        tuple:
            The index of the last sampled state
        np.ndarray:
            The estimated grid.
        """
        print("grid mean_max", self.algo.grid_mean.max())
        return self.next_state, self.algo.grid_mean.copy()


class ActiveSearchWithRandomInit(Search):
    def __init__(
        self,
        algo: ActiveSearchAlgos,
        surface_grid_map: SurfaceGridMap,
        kernel_scale: float,
        random_sample_count: int = 10,
        **kwargs
    ):
        self.grid = surface_grid_map
        self.active_search = ActiveSearch(
            algo, surface_grid_map, kernel_scale, **kwargs
        )
        self.random_search = RandomSearch(surface_grid_map)
        self.random_sample_count = random_sample_count
        self.palp_count = 0

    def next(self):
        self.palp_count += 1
        if self.palp_count < self.random_sample_count:
            return self.random_search.next()
        else:
            return self.active_search.next()

    def update_outcome(self, prev_val: float):
        if self.palp_count < self.random_sample_count:
            self.random_search.update_outcome(prev_val)
        else:
            self.active_search.update_outcome(prev_val)

    @property
    def grid_estimate(self):
        if self.palp_count < self.random_sample_count:
            return self.random_search.grid_estimate
        else:
            return self.active_search.grid_estimate


if __name__ == "__main__":
    from rpal.utils.pcd_utils import scan2mesh, mesh2roi, visualize_pcds
    from rpal.utils.transform_utils import quat2mat
    from scipy.spatial.transform import Rotation

    np.random.seed(100)
    pcd = o3d.io.read_point_cloud(str(rpal_const.SURFACE_SCAN_PATH))
    surface_mesh = scan2mesh(pcd)
    roi_pcd = mesh2roi(surface_mesh, bbox_pts=rpal_const.BBOX_ROI)

    search_history = SearchHistory()
    surface_grid_map = SurfaceGridMap(roi_pcd, grid_size=0.001)
    # planner = RandomSearch(roi_pcd, surface_grid_map)
    # planner.grid.visualize(show_tf=True)
    # planner = ActiveSearch(ActiveSearchAlgos.BO, roi_pcd, kernel_scale=2.0)
    planner = ActiveSearchWithRandomInit(
        ActiveSearchAlgos.BO, surface_grid_map, kernel_scale=2.0
    )
    palp_Ts = []
    surf_norms = []
    for i in range(100):
        palp = planner.next()
        search_history.add(*planner.grid_estimate)
        planner.update_outcome(1.0)

        T = np.eye(4)
        O_R_E = quat2mat(rpal_const.GT_SCAN_POSE[-4:])
        result = Rotation.align_vectors(
            np.array([palp[1], np.array([0, -1, 0])]),
            np.array([[0, 0, -1], np.array([0, 1, 0])]),
            weights=np.array([10, 0.1]),
        )
        T[:3, :3] = result[0].as_matrix()
        T[:3, 3] = palp[0]
        palp_Ts.append(T.copy())
        surf_norms.append(palp)

    ani = HeatmapAnimation(np.array(search_history.history))
    ani.visualize()

    roi_pcd.paint_uniform_color([1, 0, 0])

    gt_scan = o3d.io.read_point_cloud(str(rpal_const.GT_PATH))
    visualize_pcds(
        [planner.grid._grid_pcd, gt_scan],
        meshes=[surface_mesh],
        tfs=palp_Ts,
        surf_norms=surf_norms,
    )
