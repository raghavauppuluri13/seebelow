from collections import defaultdict
from pathlib import Path

import numpy as np
import open3d as o3d

from rpal.algorithms.search import ActiveSearch, ActiveSearchAlgos, RandomSearch
from rpal.utils.constants import *
from rpal.utils.pcd_utils import mesh2roi, scan2mesh

if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud(str(SURFACE_SCAN_PATH))
    surface_mesh = scan2mesh(pcd)
    roi_pcd = mesh2roi(surface_mesh)
    search = RandomSearch(roi_pcd)
    search.grid.visualize(show_tf=False)
