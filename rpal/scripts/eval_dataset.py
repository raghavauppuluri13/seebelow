import open3d as o3d
import numpy as np
import argparse
from rpal.utils.constants import *
from rpal.utils.pcd_utils import scan2mesh, mesh2roi, visualize_pcds
from rpal.algorithms.gui import HeatmapAnimation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Time-series Heatmap Generator and dataset evaluation"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="dataset_01-02-2024_22-15-10",
        help="Folder name in data directory containing experiment data",
    )
    args = parser.parse_args()
    dataset_path = RPAL_DATA_PATH / args.dataset_path

    # Data Load
    history = np.load(dataset_path / "search_history.npy")
    print(history.shape)
    ani = HeatmapAnimation(history)
    ani.visualize()

    timeseries = np.load(dataset_path / "timeseries.npy")

    palpations_cnt = timeseries[:]["palp_id"].max()

    print("Stats: ")
    print(f"Palpations: {palpations_cnt}")

    reconstruction_pcd = o3d.io.read_point_cloud(
        str(dataset_path / "reconstruction.ply")
    )

    roi_pcd = o3d.io.read_point_cloud(str(dataset_path / "surface.ply"))
    gt_scan = o3d.io.read_point_cloud(str(GT_PATH))

    roi_pcd.paint_uniform_color([0, 1, 1])
    reconstruction_pcd.paint_uniform_color([0, 1, 0])

    visualize_pcds(
        [roi_pcd, reconstruction_pcd, gt_scan],
    )
