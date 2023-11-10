import open3d as o3d
from pathlib import Path
import numpy as np
from collections import defaultdict
from rpal.algorithms.cluster_store import SurfaceGridMap
from rpal.algorithms.gp import SquaredExpKernel
from rpal.algorithms.active_area_search import ActiveAreaSearch
from rpal.algorithms.quadtree import QuadTree

if __name__ == "__main__":
    surface_pcd_path = Path(
        "/home/raghava/projects/robot-palpation/rpal/scripts/surface.ply"
    )
    phantom_pcd = o3d.io.read_point_cloud(str(surface_pcd_path.absolute()))
    phantom_v = np.asarray(phantom_pcd.points)
    gridmap = SurfaceGridMap(phantom_pcd)
    # gridmap.visualize()

    kernel = SquaredExpKernel(scale=0.5)
    qt_dim = max(gridmap.shape)
    qt_dim += 10
    qt_dim = (qt_dim // 10) * 10
    group_dim = 5
    print(qt_dim)
    group_quadtree = QuadTree(qt_dim, qt_dim, group_dim, group_dim)
    group_quadtree.area

    aas = ActiveAreaSearch(
        gridmap, group_quadtree, kernel, noise_var=0.01, threshold=5, confidence=0.7
    )

    samples = np.array([[5, 5, 8], [0, 8, 2], [8, 0, 4], [8, 8, 1]])

    for sample in samples:
        next_state = aas.get_optimal_state(sample[:2], sample[-1], normalized=True)
