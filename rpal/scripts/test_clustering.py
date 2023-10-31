import open3d as o3d
from pathlib import Path
import numpy as np
from collections import defaultdict
from rpal.algorithms.cluster_store import GridStore

if __name__ == "__main__":
    surface_pcd_path = Path("./surface.ply")
    phantom_pcd = o3d.io.read_point_cloud(str(surface_pcd_path.absolute()))
    phantom_v = np.asarray(phantom_pcd.points)

    store = GridStore(phantom_pcd)
    # store = ClusterStore(phantom_pcd)

    # store.visualize()
