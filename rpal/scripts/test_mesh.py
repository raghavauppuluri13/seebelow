import open3d as o3d
from copy import deepcopy
import argparse
import os
import math
import numpy as np
from scipy.spatial.transform import Rotation

import pinocchio as pin


from interpolator import Interpolator, InterpType
from utils import visualize_pcds

EPS = np.finfo(float).eps * 4.0


def unit_vector(data, axis=None, out=None):
    """
    Returns ndarray normalized by length, i.e. eucledian norm, along axis.

    E.g.:
        >>> v0 = numpy.random.random(3)
        >>> v1 = unit_vector(v0)
        >>> numpy.allclose(v1, v0 / numpy.linalg.norm(v0))
        True

        >>> v0 = numpy.random.rand(5, 4, 3)
        >>> v1 = unit_vector(v0, axis=-1)
        >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=2)), 2)
        >>> numpy.allclose(v1, v2)
        True

        >>> v1 = unit_vector(v0, axis=1)
        >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=1)), 1)
        >>> numpy.allclose(v1, v2)
        True

        >>> v1 = numpy.empty((5, 4, 3), dtype=numpy.float32)
        >>> unit_vector(v0, axis=1, out=v1)
        >>> numpy.allclose(v1, v2)
        True

        >>> list(unit_vector([]))
        []

        >>> list(unit_vector([1.0]))
        [1.0]

    Args:
        data (np.array): data to normalize
        axis (None or int): If specified, determines specific axis along data to normalize
        out (None or np.array): If specified, will store computation in this variable

    Returns:
        None or np.array: If @out is not specified, will return normalized vector. Otherwise, stores the output in @out
    """
    if out is None:
        data = np.array(data, dtype=np.float32, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


def pose2mat(pos_quat):
    """
    Converts pose to homogeneous matrix.

    Args:
        pose (2-tuple): a (pos, orn) tuple where pos is vec3 float cartesian, and
            orn is vec4 float quaternion.

    Returns:
        np.array: 4x4 homogeneous matrix
    """
    homo_pose_mat = np.zeros((4, 4), dtype=np.float32)
    homo_pose_mat[:3, :3] = quat2mat(pos_quat[3:7])
    homo_pose_mat[:3, 3] = np.array(pos_quat[:3], dtype=np.float32)
    homo_pose_mat[3, 3] = 1.0
    return homo_pose_mat


def quat2mat(quaternion):
    """
    Converts given quaternion to matrix.

    Args:
        quaternion (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: 3x3 rotation matrix
    """
    # awkward semantics for use with numba
    inds = np.array([3, 0, 1, 2])
    q = np.asarray(quaternion).copy().astype(np.float32)[inds]

    n = np.dot(q, q)
    if n < EPS:
        return np.identity(3)
    q *= math.sqrt(2.0 / n)
    q2 = np.outer(q, q)
    return np.array(
        [
            [1.0 - q2[2, 2] - q2[3, 3], q2[1, 2] - q2[3, 0], q2[1, 3] + q2[2, 0]],
            [q2[1, 2] + q2[3, 0], 1.0 - q2[1, 1] - q2[3, 3], q2[2, 3] - q2[1, 0]],
            [q2[1, 3] - q2[2, 0], q2[2, 3] + q2[1, 0], 1.0 - q2[1, 1] - q2[2, 2]],
        ]
    )


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
mesh = mesh.subdivide_midpoint(number_of_iterations=1)
voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=0.005)

mesh = mesh.compute_triangle_normals()
mesh = mesh.compute_vertex_normals()

o3d.visualization.draw_geometries([voxel_grid])

N = 10  # Number of vectors
dtype_f = np.float32
dtype_i = np.int32


# Combine start points and endpoints for positions array

l = []
Ts = []
FT_p1s = []
FT_p2s = []
Fs = []

for t in range(len(data)):
    pos_quat = data[t][:7]
    T = pose2mat(pos_quat)
    O_p = T @ np.array([0, 0, 0, 1])
    check = voxel_grid.check_if_included(o3d.utility.Vector3dVector([O_p[:3]]))
    F = data[t][-3:]
    F_i = np.ones(4)
    F[2] = 0
    F = T[:3, :3] @ F
    if np.linalg.norm(F) > 4:
        F_unit = unit_vector(F)
        O_p_f2 = O_p[:3] + F_unit
        FT_p1s.append(O_p[:3])
        FT_p2s.append(O_p_f2)
    if np.alltrue(check):
        Ts.append(T)
        vox_i = voxel_grid.get_voxel(O_p[:3])
        l.append(vox_i)
N = len(FT_p1s)
FT_pos = np.vstack((np.array(FT_p1s), np.array(FT_p2s)))
print(FT_pos)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
pcd.estimate_normals()

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

print(len(vn))
print(len(v))

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
test_pts = [50, 400, 610]
for i in test_pts:
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
    O_T_P[:3, 3] = v_p - PALPATE_DEPTH * vn_p
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
    interp.init(O_T_overhead, end, steps=10)
    while not interp.done:
        pose = interp.next()
        pts.append(pose.homogeneous)

pts = Ts + pts

visualize_pcds([mesh, lineset, pcd], tfs=pts)
