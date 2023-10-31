import open3d as o3d
from copy import deepcopy
import argparse
import os
import math
import numpy as np
from scipy.spatial.transform import Rotation

import pinocchio as pin


from interpolator import Interpolator, InterpType
from utils import visualize_pcds, pose2mat

quat_gt = np.array([0, -0.7071068, 0, 0.7071068])
pos_gt = np.array([0.56616064, 0.12552764, 0.0545865 - 0.009845])
pos_quat = np.hstack([pos_gt, quat_gt])
print(pos_quat)
T = pose2mat(pos_quat)


gt_mesh = o3d.io.read_triangle_mesh("./tumor_big.stl")

verts = np.asarray(gt_mesh.vertices)

verts /= 1000

gt_pcd = o3d.geometry.PointCloud()
gt_pcd.points = o3d.utility.Vector3dVector(verts)

gt_pcd = gt_pcd.transform(T)
gt_pcd.paint_uniform_color([1, 0, 0])


parser = argparse.ArgumentParser(description="Time-series Heatmap Generator")
parser.add_argument(
    "--dataset_path",
    type=str,
    default="./data/dataset_08-12-2023_05-02-59",
    help="Folder containing time-series data",
)
args = parser.parse_args()

# Data Load
dataset_path = args.dataset_path
print(f"{dataset_path}/reconstruction.ply")

pcd = o3d.io.read_point_cloud(f"{dataset_path}/reconstruction.ply")

bounding_box = pcd.get_axis_aligned_bounding_box()
lookat_point = bounding_box.get_center()

pcd.estimate_normals()
pcd.orient_normals_consistent_tangent_plane(10)
recon_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, linear_fit=True, depth=5
)[0]
recon_mesh = recon_mesh.filter_smooth_simple(number_of_iterations=5)
recon_mesh.compute_vertex_normals()

pcd.paint_uniform_color([0, 0, 0])

mesh = o3d.io.read_triangle_mesh(f"./out_mesh.ply")

pts = np.asarray(pcd.points)


# Calculate the mean point (center) of the point cloud
lookat_point = pts.mean(axis=0)


def rotate_view(vis):
    ctr = vis.get_view_control()
    ctr.set_lookat(lookat_point)
    ctr.rotate(2.0, 2.0)  # Adjust rotation speed by changing these values
    return False


vis = o3d.visualization.VisualizerWithKeyCallback()
vis.register_animation_callback(rotate_view)
vis.create_window()
vis.add_geometry(pcd)
vis.add_geometry(mesh)
vis.add_geometry(gt_pcd)
vis.run()
vis.destroy_window()
