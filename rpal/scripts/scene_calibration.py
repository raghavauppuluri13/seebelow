import numpy as np
import os
import cv2
import time
import open3d as o3d
import argparse
import datetime

from utils import visualize_pcds, unit, get_centered_bbox, crop_pcd

EVAL_R = np.array(
    [
        [0.78598308, 0.53488944, -0.31003851],
        [-0.60047267, 0.54107235, -0.58878968],
        [-0.14718412, 0.64894838, 0.7464602],
    ]
)
EVAL_t = np.array([0.01657456, 0.06089821, -0.23600002])
EVAL_BBOX_PARAMS = [0.07, -0.01, 0.05, -0.05, 0.05, -0.05]
EVAL_scale = 1


def get_rot_mat_from_basis(b1, b2, b3):
    A = np.eye(3)
    A[:, 0] = b1
    A[:, 1] = b2
    A[:, 2] = b3
    return A.T


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

    pcd = crop_pcd(pcd, R, t, scale, EVAL_BBOX_PARAMS, visualize=True)
    pcd.compute_convex_hull()
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(10)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, linear_fit=True
    )[0]
    mesh.compute_vertex_normals()
    mesh.remove_degenerate_triangles()

    visualize_pcds([pcd])
    visualize_pcds([mesh])

    print("R = np.", repr(R))
    print("t = np.", repr(t))
    print("BBOX_PARAMS = np.", EVAL_BBOX_PARAMS)
