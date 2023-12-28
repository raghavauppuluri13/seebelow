from rpal.utils.pcd_utils import (
    crop_pcd,
    pick_surface_bbox,
    preprocess_raw_phantom_scan,
)
import matplotlib.pyplot as plt
from rpal.utils.constants import *

import open3d as o3d
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--pcd", type=str, required=True)

args = parser.parse_args()

pcd = o3d.io.read_point_cloud(args.pcd)
pcd = preprocess_raw_phantom_scan(pcd)

o3d.visualization.draw_geometries([pcd])
