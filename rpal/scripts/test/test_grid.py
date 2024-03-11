from collections import defaultdict
from pathlib import Path
import argparse

import numpy as np
import open3d as o3d

from rpal.algorithms.search import ActiveSearch, ActiveSearchAlgos, RandomSearch
from rpal.utils.constants import *
from rpal.utils.pcd_utils import mesh2roi, scan2mesh

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--debug", action="store_true")
    argparser.add_argument("--show_tf", action="store_true")
    args = argparser.parse_args()

    pcd = o3d.io.read_point_cloud(str(SURFACE_SCAN_PATH))
    surface_mesh = scan2mesh(pcd)
    bbox_pts = None
    if args.debug:
        bbox_pts = BBOX_ROI
    roi_pcd = mesh2roi(surface_mesh, bbox_pts=bbox_pts)
    search = RandomSearch(roi_pcd)
    search.grid.visualize(show_tf=args.show_tf)
