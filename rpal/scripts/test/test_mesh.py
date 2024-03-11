import argparse
import math
import os
from copy import deepcopy

import numpy as np
import open3d as o3d
import pinocchio as pin
from scipy.spatial.transform import Rotation

from interpolator import Interpolator, InterpType
from rpal.utils.math_utils import unit
from rpal.utils.pcd_utils import visualize_pcds
from rpal.utils.transform_utils import pose2mat

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
data_file_path = os.path.join(dataset_path, "timeseries.txt")
data = np.loadtxt(data_file_path)

mesh = o3d.io.read_triangle_mesh("./out_mesh.ply")
mesh = mesh.subdivide_midpoint(number_of_iterations=2)
mesh = mesh.compute_triangle_normals()
mesh = mesh.compute_vertex_normals()

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
"""
print("")
print("1) Please pick 4 point as the corners your bounding box [shift + left click].")
print("   Press [shift + right click] to undo point picking")
print("2) After picking points, press 'Q' to close the window")
vis = o3d.visualization.VisualizerWithEditing()
vis.create_window()
vis.add_geometry(pcd)
vis.add_geometry(mesh)
vis.run()  # user picks points
vis.destroy_window()
points_idx = vis.get_picked_points()
pcd_npy = np.asarray(pcd.points)
bbox_pts = np.zeros((8, 3))
pts = pcd_npy[points_idx]

pts[:, -1] += 0.5
bbox_pts[:4] = pts
pts[:, -1] -= 1
bbox_pts[4:8] = pts

bbox = o3d.geometry.OrientedBoundingBox.create_from_points(
    o3d.utility.Vector3dVector(bbox_pts)
)
"""
pcd.estimate_normals()
pcd.normalize_normals()
pcd.orient_normals_consistent_tangent_plane(k=100)
# pcd = pcd.crop(bbox)

N = 10  # Number of samples
A = len(np.asarray(pcd.points))
random_sample = np.random.randint(0, A, N)

voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.003)
o3d.visualization.draw_geometries([voxel_grid])

N = 10  # Number of vectors
dtype_f = np.float32
dtype_i = np.int32

# Combine start points and endpoints for positions array

Ts = []
FT_p1s = []
FT_p2s = []
Fs = []

recon_pts = []

for t in range(len(data)):
    pos_quat = data[t][:7]
    T = pose2mat(pos_quat)
    O_p = T @ np.array([0, 0, 0, 1])
    check = voxel_grid.check_if_included(o3d.utility.Vector3dVector([O_p[:3]]))
    F = data[t][-3:]
    if np.linalg.norm(F) > 8:
        recon_pts.append(pos_quat[:3])
    F_i = np.ones(4)
    F[2] = 0
    F = T[:3, :3] @ F

    if np.linalg.norm(F) > 4:
        F_unit = unit(F)
        O_p_f2 = O_p[:3] + F_unit
        FT_p1s.append(O_p[:3])
        FT_p2s.append(O_p_f2)
    if np.alltrue(check):
        Ts.append(T)
        vox_i = voxel_grid.get_voxel(O_p[:3])
recon_pcd = o3d.geometry.PointCloud()
recon_npy = np.array(recon_pts)
print("RECON_SHAPE", recon_npy.shape)
recon_pcd.points = o3d.utility.Vector3dVector(recon_npy)
N = len(FT_p1s)
FT_pos = np.vstack((np.array(FT_p1s), np.array(FT_p2s)))

# Create line indices
indices = np.hstack(
    (np.arange(0, N)[:, np.newaxis], np.arange(N, 2 * N)[:, np.newaxis])
)

# Create an empty LineSet
lineset = o3d.geometry.LineSet()

# Set positions and indices
lineset.points = o3d.utility.Vector3dVector(FT_pos)
lineset.lines = o3d.utility.Vector2iVector(indices.astype(dtype_i))

# Set colors (optional)
colors = np.random.rand(len(indices), 3).astype(dtype_f)
lineset.colors = o3d.utility.Vector3dVector(colors)

goals = []
vn = np.asarray(mesh.vertex_normals)
v = np.asarray(mesh.vertices)

interp = Interpolator(interp_type=InterpType.SE3)

ABOVE_HEIGHT = 0.03
PALPATE_DEPTH = 0.02
O_z_axis = np.array([[0, 0, 1]])

O_p_PH_0 = np.array([0.59473506, 0.06602833, 0.0616792])
O_p_PH_1 = np.array([0.53661529, 0.18703561, 0.0616792])
O_p_PH_2 = np.array([0.52858962, 0.18280019, 0.0616792])

O_p_PH_0 = np.array([0.59473506, 0.06602833, 0.060168])
O_T_PH_0 = np.eye(4)
O_T_PH_0[:3, 3] = O_p_PH_0
O_T_PH_1 = np.eye(4)
O_T_PH_1[:3, 3] = O_p_PH_1
O_T_PH_2 = np.eye(4)
O_T_PH_2[:3, 3] = O_p_PH_2

O_t_overhead = np.array([0.56746543, 0.12762998, 0.09405758])
O_R_overhead = pin.rpy.rpyToMatrix(np.array([3.13075706, -0.03085785, -0.27018787]))

O_x_axis = O_R_overhead @ np.array([1, 0, 0])


for i in random_sample:
    vn_p = pcd.normals[i]
    v_p = pcd.points[i]

    R = Rotation.align_vectors(
        np.array([vn_p, np.array([1, 0, 0])]),
        np.array([[0, 0, -1], O_x_axis]),
        weights=np.array([1, 0.5]),
    )[0].as_matrix()

    z = R @ np.array([0, 0, 1])

    print(np.dot(vn_p, z))

    O_T_P = np.eye(4)
    O_T_P[:3, :3] = R
    O_T_P[:3, 3] = v_p + ABOVE_HEIGHT * vn_p
    # O_T_P[:3, 3] = v_p - PALPATE_DEPTH * vn_p
    O_T_P[:3, 3] = v_p
    goals.append(deepcopy(O_T_P))

O_T_overhead = pin.SE3.Identity()
O_T_overhead.translation = O_t_overhead
O_T_overhead.rotation = O_R_overhead

pts = []

# pts.append(O_T_PH_0)
# pts.append(O_T_PH_1)
# pts.append(O_T_PH_2)
pts.append(O_T_overhead.homogeneous)
for goal in goals:
    end = pin.SE3(goal[:3, :3], goal[:3, 3])
    interp.init(O_T_overhead, end, steps=2)
    while not interp.done:
        pose = interp.next()
        # pts.append(pose.homogeneous)
# print(pts)
# pts = Ts + pts

visualize_pcds([mesh, recon_pcd], tfs=pts)
