from seebelow.utils.pcd_utils import (
    crop_pcd,
    pick_surface_bbox,
)
import matplotlib.pyplot as plt
import seebelow.utils.constants as seebelow_const

import open3d as o3d
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--pcd", type=str, required=True)

args = parser.parse_args()

pcd = o3d.io.read_point_cloud(str(seebelow_const.SEEBELOW_MESH_PATH / args.pcd))

o3d.visualization.draw_geometries([pcd])
