from rpal.utils.pcd_utils import (
    crop_pcd,
    pick_surface_bbox,
)
import matplotlib.pyplot as plt
import rpal.utils.constants as rpal_const

import open3d as o3d
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--pcd", type=str, required=True)

args = parser.parse_args()

pcd = o3d.io.read_point_cloud(str(rpal_const.RPAL_MESH_PATH / args.pcd))

o3d.visualization.draw_geometries([pcd])
