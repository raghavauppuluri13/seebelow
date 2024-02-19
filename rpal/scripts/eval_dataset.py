import open3d as o3d
import numpy as np
import argparse
from rpal.utils.constants import *
from rpal.utils.transform_utils import quat2mat
from rpal.utils.pcd_utils import (scan2mesh, mesh2roi, visualize_pcds, color_icp, color_filter,
                                  disk_pcd, clustering)
from rpal.algorithms.gui import HeatmapAnimation
from datetime import datetime
from rpal.utils.rerun_utils import pcd_to_rr
import rerun as rr
from rerun.datatypes import TranslationAndMat3x3
import os


def get_later_datasets(target_dataset, folder_path):
    target_date = datetime.strptime(target_dataset, "dataset_%m-%d-%Y_%H-%M-%S")
    later_datasets = []
    for file in os.listdir(folder_path):
        if file.startswith("dataset_"):
            file_date = datetime.strptime(file, "dataset_%m-%d-%Y_%H-%M-%S")
            if file_date > target_date and "timeseries.npy" in [
                    file for file in os.listdir(folder_path)
            ]:
                later_datasets.append(file)
    return later_datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Time-series Heatmap Generator and dataset evaluation")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="dataset_01-02-2024_22-15-10",
        help="Folder name in data directory containing experiment data",
    )
    parser.add_argument(
        "--combine-later-datasets",
        type=bool,
        default=False,
        help="Combine later datasets into one",
    )
    args = parser.parse_args()

    dataset_path = RPAL_DATA_PATH / args.dataset_path

    def aggregate(datasets):
        for later_dataset in later_datasets:
            dataset_path = RPAL_DATA_PATH / later_dataset
            timeseries = np.load(dataset_path / "timeseries.npy")
            reconstruction_pcd += o3d.io.read_point_cloud(str(dataset_path / "reconstruction.ply"))

    # Data Load
    history = np.load(dataset_path / "search_history.npy")
    timeseries = np.load(dataset_path / "timeseries.npy")

    #ani = HeatmapAnimation(history)
    #ani.visualize()

    later_datasets = get_later_datasets(args.dataset_path, RPAL_DATA_PATH)
    palpations_cnt = timeseries[:]["palp_id"].max()
    print("Stats: ")
    print(f"Palpations: {palpations_cnt}")
    roi_pcd = o3d.io.read_point_cloud(str(dataset_path / "surface.ply"))
    gt_scan = o3d.io.read_point_cloud(str(GT_PATH))
    gt_scan = color_filter(gt_scan, color_to_filter=[0.0, 0.0, 0.0], threshold=0.1)
    gt_tumors = clustering(gt_scan, eps=0.002, min_points=50)
    gt_tumors = sorted(gt_tumors, key=lambda pc: len(pc.points))
    gt_tumor = gt_tumors[1]
    # gt_tumor = gt_tumor.voxel_down_sample(voxel_size=0.001)

    # log data
    rr.init("rpal_eval", spawn=True)
    rr.log('pcds/ground_truth_tumor', pcd_to_rr("gt_tumor", np.asarray(gt_tumor.points)))
    collect_pts = timeseries[:]["collect_points_flag"]
    collect_pts_t, _ = np.where(collect_pts == 1)
    O_p_E = timeseries[collect_pts_t]["O_p_EE"]
    O_q_E = timeseries[collect_pts_t]["O_q_EE"]
    init_O_p_surf = False
    init_O_p_f = False
    for t in range(len(timeseries)):
        if palp_state == 
        Fxyz = timeseries[t]["Fxyz"]
        O_p_E = timeseries[t]["O_p_EE"]
        O_q_E = timeseries[t]["O_q_EE"]
        collect_pts_flag = timeseries[t]["collect_points_flag"]
        if collect_pts_flag and not init_O_p_surf:
            O_p_surf = O_p_E
            init_O_p_surf = True
        elif not collect_pts_flag and init_O_p_surf:
            init_O_p_surf = False

        if not using_force_control_flag:
            O_p_f = O_p_E
        stiffness = Fxyz[2] / np.linalg.norm(O_p_f - O_p_surf)
        T = np.eye(4)
        T[:3, :3] = quat2mat(O_q_E.flatten())
        T[:3, 3] = O_p_E.flatten()
        rr_tf = TranslationAndMat3x3(translation=T[:3, 3], mat3x3=T[:3, :3])
        rr.log("pcds/eef_pose", rr.Transform3D(transform=rr_tf))
        if collect_pts_flag:
            rr.log("stiffness", rr.Scalar(stiffness))
            rr.log("pcds/reconstruction", pcd_to_rr("reconstruction", O_p_E))
    '''
    #reconstruction_pcd = reconstruction_pcd.voxel_down_sample(voxel_size=0.001)
    recon_pth = RPAL_MESH_PATH / "combined_palpations_01-08-2024_21-16-56.ply"
    reconstruction_pcd = o3d.geometry.PointCloud()
    reconstruction_pcd = o3d.io.read_point_cloud(str(recon_pth))
    reconstruction_pcd.paint_uniform_color([0, 1, 0])

    # roi_pcd.paint_uniform_color([0, 1, 1])
    visualize_pcds([gt_tumor, reconstruction_pcd])


    #scan_pcd = o3d.io.read_point_cloud(str(SURFACE_SCAN_PATH))
    #surface_mesh = scan2mesh(scan_pcd)

    # save_path = datetime.now().strftime("combined_palpations_%m-%d-%Y_%H-%M-%S.ply")

    # o3d.io.write_point_cloud(str(save_path), reconstruction_pcd)

    # color_icp(reconstruction_pcd, gt_scan, vis=True)
    # visualize_pcds([roi_pcd], meshes=[surface_mesh], tf_size=0.005)
    '''
