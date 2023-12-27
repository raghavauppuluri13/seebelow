from collections import defaultdict
from pathlib import Path

import numpy as np
import open3d as o3d

from rpal.algorithms.search import (ActiveSearch, ActiveSearchAlgos,
                                    RandomSearch)
from rpal.utils.constants import RPAL_PKG_PATH
from rpal.utils.pcd_utils import surface_mesh_to_pcd

if __name__ == "__main__":
    surface_pcd_cropped = surface_mesh_to_pcd(str(RPAL_PKG_PATH / "meshes" / "phantom_mesh.ply"))
    search = RandomSearch(surface_pcd_cropped)
    search.grid.visualize(show_tf=False)
