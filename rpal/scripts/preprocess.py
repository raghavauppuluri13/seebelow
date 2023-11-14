import open3d as o3d
import argparse
from rpal.utils.pcd_utils import crop_pcd, visualize_pcds
from rpal.utils.math_utils import unit
import os
import numpy as np
import cv2
from pathlib import Path

from multiprocessing.pool import ThreadPool

CROP_PARAMS = (
    np.array(
        [
            [0.78598308, 0.53488944, -0.31003851],
            [-0.60047267, 0.54107235, -0.58878968],
            [-0.14718412, 0.64894838, 0.7464602],
        ]
    ),
    np.array([0.01657456, 0.06089821, -0.23600002]),
    1,
    [0.11, -0.002, 0.05, -0.05, 0.05, -0.05],
)


def process(pcd):
    global CROP_PARAMS
    pcd = crop_pcd(pcd, *CROP_PARAMS)
    # Convert the point cloud to a NumPy array
    colors = np.asarray(pcd.colors)

    threshold = 0.1

    # Find the indices of the points that are not close to black
    non_black_indices = np.where(np.any(colors > threshold, axis=1))[0]

    # Filter the points and colors using the indices
    filtered_points = np.asarray(pcd.points)[non_black_indices]
    filtered_colors = colors[non_black_indices]

    # Create a new point cloud with the filtered points and colors
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
    pcd.compute_convex_hull()
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(10)

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, linear_fit=True
    )[0]

    mesh.compute_vertex_normals()
    mesh.remove_degenerate_triangles()

    return mesh


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ply Visualizer")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data/dataset_08-12-2023_05-02-59",
        help="the folder of ply files",
    )
    args = parser.parse_args()

    # Path to the folder containing the .ply files
    folder_path = Path(args.dataset_path)

    pcds_dir = folder_path / "raw_pcd"
    proc_pcd_dir = folder_path / "proc_pcd"
    if not os.path.exists(proc_pcd_dir):
        os.mkdir(proc_pcd_dir)

    # Get all the files with the .ply extension
    ply_files = [f for f in os.listdir(pcds_dir) if f.endswith(".ply")]

    # Sort the files numerically (assuming they are named as 0.ply, 1.ply, ...)
    ply_files.sort(key=lambda x: int(os.path.splitext(x)[0]))

    # Loop through the sorted .ply files

    def process_raw_pcd(pcd_metadata):
        idx, ply_file = pcd_metadata
        # Read the mesh from the .ply file
        print(idx)
        pcd_path = os.path.join(pcds_dir, ply_file)
        pcd = o3d.io.read_point_cloud(pcd_path)
        mesh = process(pcd)

        o3d.io.write_triangle_mesh(str((proc_pcd_dir / f"{idx}.ply").absolute()), mesh)

    with ThreadPool() as pool:
        pool.map(process_raw_pcd, enumerate(ply_files))
