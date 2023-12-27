import argparse
import datetime
import os
import time

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from rpal.utils.math_utils import (get_rot_mat_from_basis,
                                   three_pts_to_rot_mat, unit)
from rpal.utils.pcd_utils import crop_pcd, get_centered_bbox, visualize_pcds

EVAL_scale = 1
EVAL_R = np.array(
    [
        [-0.80397832, -0.57559227, 0.14937338],
        [0.59348632, -0.76090159, 0.2623028],
        [-0.03732103, 0.29953682, 0.95335452],
    ]
)
EVAL_t = np.array([-0.06189939, -0.01550329, -0.19499999])
EVAL_BBOX_PARAMS = [0.14, -0.01, 0.05, -0.05, 0.05, -0.05]


O_p_PH_0 = np.array([0.59473506, 0.06602833, 0.0616792])
O_p_PH_1 = np.array([0.53661529, 0.18703561, 0.0616792])
O_p_PH_2 = np.array([0.52858962, 0.18280019, 0.0616792])

O_R_PH = three_pts_to_rot_mat(O_p_PH_0, O_p_PH_1, O_p_PH_2, neg_x=False)
O_T_PH = np.eye(4)
O_T_PH[:3, :3] = O_R_PH
O_T_PH[:3, 3] = O_p_PH_0


def get_calibration(pcd):
    print("")
    print(
        "1) Please pick 4 points using [shift + left click].\n \
        your 1st point should be the origin point \n \
        and your 2nd point should be along the x-axis \n \
        and your 3rd point should be another point on xy plane (not on x axis) \n"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1,  # specify the size of coordinate frame
    )
    vis.add_geometry(frame)
    vis.run()  # user picks points
    vis.destroy_window()
    points_idx = vis.get_picked_points()
    pcd_npy = np.asarray(pcd.points)

    frame_pts = pcd_npy[points_idx[:3]]

    # compute basis vectors
    xaxis = unit(frame_pts[1] - frame_pts[0])
    v_another = unit(frame_pts[2] - frame_pts[0])
    zaxis = unit(np.cross(xaxis, v_another))
    yaxis = unit(np.cross(zaxis, xaxis))

    R = get_rot_mat_from_basis(xaxis, yaxis, zaxis)
    t = -frame_pts[0]

    # compute length of vine robot

    return R, t, 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-pcd_path",
        help="path to point cloud ply file",
        type=str,
        default="./out.ply",
    )
    parser.add_argument("-eval", help="view constants", action="store_true")

    args = vars(parser.parse_args())
    pcd = o3d.io.read_point_cloud(args["pcd_path"])

    if args["eval"]:
        scale = EVAL_scale
        R = EVAL_R
        t = EVAL_t
    else:
        R, t, scale = get_calibration(pcd)

    pcd = crop_pcd(pcd, R, t, scale, EVAL_BBOX_PARAMS, visualize=False)
    pcd.compute_convex_hull()

    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(10)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, linear_fit=True, depth=5
    )[0]
    mesh = mesh.filter_smooth_simple(number_of_iterations=5)
    mesh.compute_vertex_normals()
    mesh.remove_degenerate_triangles()

    # visualize_pcds([pcd])
    vn = np.asarray(mesh.vertex_normals)
    v = np.asarray(mesh.vertices)
    i = 30
    i = 340
    vn_p = vn[i]
    v_p = v[i]
    print(vn_p)
    print(v_p)

    R = Rotation.align_vectors(np.array([vn_p]), np.array([[0, 0, 1]]))[0].as_matrix()

    O_T_P = np.eye(4)
    O_T_P[:3, :3] = R
    O_T_P[:3, 3] = v_p

    visualize_pcds([mesh], tfs=[np.linalg.inv(O_T_PH), O_T_P])

    MESH_BBOX_PARAMS = [0.13, 0.01, 0.05, -0.05, 0.1, -0.1]
    mesh = crop_pcd(mesh, np.eye(3), np.zeros(3), 1, MESH_BBOX_PARAMS, visualize=False)

    mesh = mesh.transform(O_T_PH)

    o3d.io.write_triangle_mesh("./out_mesh.ply", mesh)

    print("T = np.", repr(O_T_PH))

    print("R = np.", repr(R))
    print("t = np.", repr(t))
    print("BBOX_PARAMS = np.", EVAL_BBOX_PARAMS)
