import open3d as o3d
import numpy as np
import argparse
from rpal.utils.constants import *
from rpal.utils.transform_utils import quat2mat
from rpal.utils.pcd_utils import (scan2mesh, mesh2roi, visualize_pcds, color_icp, color_filter,
                                  disk_pcd, clustering)
from rpal.algorithms.gui import HeatmapAnimation
import matplotlib
from datetime import datetime
from rpal.utils.rerun_utils import pcd_to_rr
import rerun as rr
from rerun.datatypes import TranslationAndMat3x3
import os

cmap = matplotlib.colormaps["turbo_r"]
norm = matplotlib.colors.Normalize(
    vmin=0.0,
    vmax=1.0,
)


def get_later_datasets(target_dataset, folder_path):
    target_date = datetime.strptime(target_dataset, "dataset_%m-%d-%Y_%H-%M-%S")
    later_datasets = []
    for file in os.listdir(folder_path):
        if file.startswith("dataset_"):
            file_date = datetime.strptime(file, "dataset_%m-%d-%Y_%H-%M-%S")
            if file_date > target_date and "timeseries.npy" in [
                    f for f in os.listdir(folder_path / file)
            ]:
                dataset_path = RPAL_DATA_PATH / file
                try:
                    timeseries = np.load(dataset_path / "timeseries.npy")
                    #history = np.load(dataset_path / "search_history.npy")
                    #stiffness_exp = timeseries[0]["stiffness"]
                except Exception as e:
                    print(f"Error loading dataset {file}: {e}")
                    continue
                later_datasets.append(dataset_path)
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
        "--combine",
        type=bool,
        default=False,
        help="Combine later datasets into one",
    )
    args = parser.parse_args()

    gt_scan = o3d.io.read_point_cloud(str(GT_PATH))
    gt_scan = color_filter(gt_scan, color_to_filter=[0.0, 0.0, 0.0], threshold=0.1)
    gt_tumors = clustering(gt_scan, eps=0.002, min_points=50)
    gt_tumors = sorted(gt_tumors, key=lambda pc: len(pc.points))
    gt_tumor = gt_tumors[1]

    if args.combine:
        datasets = get_later_datasets(args.dataset_path, RPAL_DATA_PATH)
    else:
        datasets = [RPAL_DATA_PATH / args.dataset_path]
    palpations_cnt = 0
    tumor_without_CF = o3d.geometry.PointCloud()
    rr.init("rpal_eval", spawn=True)
    for i, dataset_path in enumerate(datasets):
        # history = np.load(dataset_path / "search_history.npy")
        #ani = HeatmapAnimation(history)
        #ani.visualize()
        timeseries = np.load(dataset_path / "timeseries.npy")
        palpations_cnt += timeseries[:]["palp_id"].max()
        roi_pcd = o3d.io.read_point_cloud(str(dataset_path / "surface.ply"))
        recon_pth = dataset_path / "reconstruction.ply"
        reconstruction_pcd = o3d.io.read_point_cloud(str(recon_pth))
        # reconstruction_pcd = reconstruction_pcd.voxel_down_sample(voxel_size=0.001)
        # gt_tumor = gt_tumor.voxel_down_sample(voxel_size=0.001)

        # log data
        rr.log('pcds/ground_truth_tumor', pcd_to_rr("gt_tumor", np.asarray(gt_tumor.points)))
        recon_np = np.asarray(reconstruction_pcd.points)
        rr.log(f'pcds/reconstructed_tumor/{dataset_path.name}', pcd_to_rr("recon_tumor", recon_np))
        collect_pts = timeseries[:]["collect_points_flag"]
        collect_pts_t, _ = np.where(collect_pts == 1)
        O_p_E = timeseries[collect_pts_t]["O_p_EE"]
        O_q_E = timeseries[collect_pts_t]["O_q_EE"]
        init_O_p_surf = False
        init_O_p_f = False

        Fxyz = timeseries[:]["Fxyz"]
        O_p_E = timeseries[:]["O_p_EE"]
        using_force_control_flag = timeseries[:]["using_force_control_flag"]
        palp_state = timeseries[:]["palp_state"]

        O_p_surf_t = np.where(np.diff(timeseries[:]["palp_id"], axis=0) == 1)[0]
        O_p_surf = np.einsum("ijk->ik", timeseries[O_p_surf_t]["O_p_EE"])
        O_p_f_t = np.where(np.diff(timeseries[:]["using_force_control_flag"], axis=0) == 1)[0]
        O_p_f = np.einsum("ijk->ik", timeseries[O_p_f_t]["O_p_EE"])
        if O_p_f.shape == O_p_surf.shape:
            stiffness_fz = Fxyz[O_p_f_t, 0, 2] / (np.linalg.norm(O_p_f - O_p_surf) + 1e-6)
            stiffness_fz /= 3000
            gradients = stiffness_fz
            assert np.max(gradients) <= 1.0 and np.min(
                gradients) >= 0.0, "Gradients should be between 0 and 1: {}, {}".format(
                    np.max(gradients), np.min(gradients))
            colors = cmap(norm(gradients))
            final_pcd = o3d.geometry.PointCloud()
            final_pcd.points = o3d.utility.Vector3dVector(O_p_f)
            final_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
            tumor_without_CF += final_pcd
            rr.log(f'pcds/positioning_end/{dataset_path.name}',
                   pcd_to_rr("positioning_end", O_p_f, colors))

        for t in range(len(timeseries)):
            Fxyz = timeseries[t]["Fxyz"]
            O_p_E = timeseries[t]["O_p_EE"]
            O_q_E = timeseries[t]["O_q_EE"]
            collect_pts_flag = timeseries[t]["collect_points_flag"]
            using_force_control_flag = timeseries[t]["using_force_control_flag"]
            palp_state = timeseries[t]["palp_state"]
            stiffness_fz = 0
            stiffness_exp = 0
            if palp_state == PalpateState.PALPATE:
                if not init_O_p_surf:
                    O_p_surf = O_p_E
                    init_O_p_surf = True
                if using_force_control_flag and not init_O_p_f:
                    O_p_f = O_p_E
                    stiffness_fz = Fxyz[0, 2] / np.linalg.norm(O_p_f - O_p_surf)
                    stiffness_fz /= PALP_CONST.stiffness_normalization
                    #stiffness_exp = timeseries[t]["stiffness"]
                    init_O_p_f = True
                T = np.eye(4)
                T[:3, :3] = quat2mat(O_q_E.flatten())
                T[:3, 3] = O_p_E.flatten()
                rr_tf = TranslationAndMat3x3(translation=T[:3, 3], mat3x3=T[:3, :3])
                rr.log("pcds/eef_pose", rr.Transform3D(transform=rr_tf))
                rr.log("force/x", rr.Scalar(Fxyz[0, 0]))
                rr.log("force/y", rr.Scalar(Fxyz[0, 1]))
                rr.log("force/z", rr.Scalar(Fxyz[0, 2]))
                if collect_pts_flag:
                    #rr.log("stiffness/experiment", rr.Scalar(stiffness_exp))
                    rr.log("stiffness/Fz", rr.Scalar(stiffness_fz))
            else:
                init_O_p_surf = False
                init_O_p_f = False
            #rr.log('search_grid', rr.Tensor(search_history[t]['grid'], dim_names=('X', 'Y')))
        '''
        recon_pth = RPAL_MESH_PATH / "combined_palpations_01-08-2024_21-16-56.ply"
        reconstruction_pcd = o3d.geometry.PointCloud()
        reconstruction_pcd = o3d.io.read_point_cloud(str(recon_pth))
        reconstruction_pcd.paint_uniform_color([0, 1, 0])
        # roi_pcd.paint_uniform_color([0, 1, 1])
        visualize_pcds([gt_tumor, reconstruction_pcd])
        #scan_pcd = o3d.io.read_point_cloud(str(SURFACE_SCAN_PATH))
        #surface_mesh = scan2mesh(scan_pcd)
        # o3d.io.write_point_cloud(str(save_path), reconstruction_pcd)
        # color_icp(reconstruction_pcd, gt_scan, vis=True)
        # visualize_pcds([roi_pcd], meshes=[surface_mesh], tf_size=0.005)
        '''

    save_path = datetime.now().strftime("final_tumor_%m-%d-%Y_%H-%M-%S")
    save_path_ply = save_path + ".ply"
    save_path_mesh = "mesh_" + save_path + ".ply"
    o3d.io.write_point_cloud(str(RPAL_MESH_PATH.parent / save_path_ply), tumor_without_CF)
    tumor_mesh = scan2mesh(tumor_without_CF)
    o3d.io.write_triangle_mesh(str(RPAL_MESH_PATH.parent / save_path_mesh), tumor_mesh)
    print("Stats: ")
    print(f"Palpations: {palpations_cnt}")
