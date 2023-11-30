import open3d as o3d
from pathlib import Path
import numpy as np
from collections import defaultdict
from rpal.utils.pcd_utils import surface_mesh_to_pcd
from rpal.utils.constants import RPAL_PKG_PATH

from rpal.algorithms.search import RandomSearch, ActiveSearch, ActiveSearchAlgos

if __name__ == "__main__":
    surface_pcd_cropped = surface_mesh_to_pcd(str(RPAL_PKG_PATH / "meshes" / "phantom_mesh.ply"))
    search = RandomSearch(surface_pcd_cropped)
    search.grid.visualize(show_tf=False)
