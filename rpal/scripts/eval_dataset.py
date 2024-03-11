import open3d as o3d
import numpy as np
import argparse
import glob
from rpal.utils.constants import *
from rpal.utils.transform_utils import quat2mat
from rpal.utils.pcd_utils import (scan2mesh, mesh2roi, visualize_pcds, color_icp, color_filter,
                                  stl_to_pcd, color_entity, disk_pcd, clustering, inverse_crop)
from rpal.algorithms.gui import HeatmapAnimation
import copy
import matplotlib
from datetime import datetime
from rpal.utils.rerun_utils import pcd_to_rr
import rerun as rr
from rerun.datatypes import TranslationAndMat3x3
import os
import yaml

cmap = matplotlib.colormaps["rainbow"]
norm = matplotlib.colors.Normalize(
    vmin=0.0,
    vmax=1.0,
)
TUMOR_STL = {
    'hemisphere': "tumor_hemisphere.stl",
    "crescent": "tumor_crescent.stl",
}
TUMOR_ID = {"hemisphere": 1, "crescent": 2, "oval": 0}
TAU = 5e-3
ALGOS = ["bo", "random"]


def compute_f_score(mesh_gt, mesh_reconstructed, center=True):
    tau = TAU

    def center_mesh(mesh):
        m = copy.deepcopy(mesh)
        center = m.get_center()
        m.translate(-center)
        return m

    def preprocess(mesh):
        # Center the meshes to their mass center
        mesh = center_mesh(mesh)
        # Subsample meshes
        pcd = mesh.sample_points_uniformly(10000)
        oboxes = pcd.detect_planar_patches(normal_variance_threshold_deg=20,
                                           coplanarity_deg=85,
                                           outlier_ratio=0.50,
                                           min_plane_edge_length=0.005,
                                           min_num_points=0,
                                           search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        print("Detected {} patches".format(len(oboxes)))
        geometries = []
        for obox in oboxes:
            pcd = inverse_crop(obox, pcd)
        return pcd

    pcd_gt = preprocess(mesh_gt)
    pcd_reconstructed = preprocess(mesh_reconstructed)

    o3d.visualization.draw_geometries([pcd_gt, pcd_reconstructed])

    # Convert to numpy arrays for distance computation
    points_gt = np.asarray(pcd_gt.points)
    points_reconstructed = np.asarray(pcd_reconstructed.points)

    # Compute distances for precision
    dists_precision = o3d.geometry.KDTreeFlann(pcd_gt)
    precision_count = 0
    for point in points_reconstructed:
        _, _, dist = dists_precision.search_knn_vector_3d(point, 1)
        if np.sqrt(dist[0]) < tau:
            precision_count += 1
    precision = precision_count / len(points_reconstructed)

    # Compute distances for recall
    dists_recall = o3d.geometry.KDTreeFlann(pcd_reconstructed)
    recall_count = 0
    for point in points_gt:
        _, _, dist = dists_recall.search_knn_vector_3d(point, 1)
        if np.sqrt(dist[0]) < tau:
            recall_count += 1
    recall = recall_count / len(points_gt)

    # Compute F-score
    f_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f_score


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
        "--glob",
        type=bool,
        default=True,
        help="find datasets by glob",
    )
    parser.add_argument(
        "--combine",
        type=bool,
        default=False,
        help="Combine later datasets into one",
    )
    parser.add_argument(
        "--tumor",
        type=str,
        default='hemisphere',
        help="Tumor type to evaluate",
    )
    parser.add_argument(
        "--rerun",
        type=bool,
        default=False,
        help="Visualize with rerun",
    )
    args = parser.parse_args()

    gt_scan = o3d.io.read_point_cloud(str(GT_PATH))
    gt_scan = color_filter(gt_scan, color_to_filter=[0.0, 0.0, 0.0], threshold=0.1)
    gt_tumors = clustering(gt_scan, eps=0.002, min_points=50)
    gt_tumors = sorted(gt_tumors, key=lambda pc: len(pc.points))
    gt_tumor = gt_tumors[TUMOR_ID[args.tumor]]
    gt_tumor = color_entity(gt_tumor, color_map='rainbow')
    ground_truth_mesh_scan = color_entity(scan2mesh(gt_tumor))
    ground_truth_mesh_cad = color_entity(scan2mesh(
        stl_to_pcd(str(RPAL_MESH_PATH / TUMOR_STL[args.tumor])), False),
                                         dir_vec=np.array([1, 0, 0]))
    dataset_map = {}
    datasets = []
    if args.glob:
        for tumor in TUMOR_STL.keys():
            for algo in ALGOS:
                sets = glob.glob(str(RPAL_DATA_PATH / "*{}*{}*".format(tumor, algo)))
                dataset_map[(tumor, algo)] = np.arange(len(datasets), len(datasets) + len(sets))
                datasets += [RPAL_DATA_PATH / s for s in sets]

        # tumors (cresc, hemi) x search algos (bo, random) x CF and w/o CF x datasets
        f_score_results = np.zeros((len(datasets), 2))
        print("Found {} datasets".format(len(datasets)))
    elif args.combine:
        datasets = get_later_datasets(args.dataset_path, RPAL_DATA_PATH)
    else:
        datasets = [RPAL_DATA_PATH / args.dataset_path]
    rerun = False
    if rerun:
        print("Spawning rerun")
        rr.init("rpal_eval", spawn=True)
    for i, dataset_path in enumerate(datasets):
        print(f"Evaluating {dataset_path.name}")
        palpations_cnt = 0
        tumor_without_CF = o3d.geometry.PointCloud()
        save_path = RPAL_MESH_PATH.parent / 'eval' / f'eval_{dataset_path.name}'
        save_path.mkdir(parents=True, exist_ok=True)
        search_history = np.load(dataset_path / "search_history.npy")
        if rerun:
            ani = HeatmapAnimation(search_history)
            ani.visualize()
        timeseries = np.load(dataset_path / "timeseries.npy")
        palpations_cnt += timeseries[:]["palp_id"].max()
        roi_pcd = o3d.io.read_point_cloud(str(dataset_path / "roi.ply"))
        grid_pcd = o3d.io.read_point_cloud(str(dataset_path / "grid.ply"))
        recon_pth = dataset_path / "reconstruction.ply"
        reconstruction_pcd = o3d.io.read_point_cloud(str(recon_pth))
        reconstruction_pcd = color_entity(reconstruction_pcd, color_map='rainbow')
        # reconstruction_pcd = reconstruction_pcd.voxel_down_sample(voxel_size=0.001)
        # gt_tumor = gt_tumor.voxel_down_sample(voxel_size=0.001)

        # log data
        if rerun:
            rr.log(
                'pcds/ground_truth_tumor',
                pcd_to_rr("gt_tumor",
                          np.asarray(gt_tumor.points),
                          colors=np.asarray(gt_tumor.colors)))
            rr.log(f'pcds/grid/{dataset_path.name}', pcd_to_rr("grid", np.asarray(grid_pcd.points)))
            rr.log(f'pcds/roi/{dataset_path.name}', pcd_to_rr("roi", np.asarray(roi_pcd.points)))
            recon_np = np.asarray(reconstruction_pcd.points)
            rr.log(f'pcds/reconstructed_tumor/{dataset_path.name}',
                   pcd_to_rr("recon_tumor", recon_np, colors=np.asarray(reconstruction_pcd.colors)))
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
            if rerun:
                rr.log(f'pcds/positioning_end/{dataset_path.name}',
                       pcd_to_rr("positioning_end", O_p_f, colors))
        if rerun:
            for t in range(len(timeseries)):
                Fxyz = timeseries[t]["Fxyz"]
                O_p_E = timeseries[t]["O_p_EE"]
                O_q_E = timeseries[t]["O_q_EE"]
                collect_pts_flag = timeseries[t]["collect_points_flag"]
                using_force_control_flag = timeseries[t]["using_force_control_flag"]
                palp_state = timeseries[t]["palp_state"]
                palp_id = timeseries[t]["palp_id"]
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
                palp_id = min(len(search_history) - 1, palp_id)
                grid = search_history[palp_id]['grid']
                rr.log('search_grid', rr.Tensor(grid, dim_names=('batch', 'X', 'Y')))

        tumor_mesh = color_entity(scan2mesh(tumor_without_CF))
        tumor_mesh_with_CF = color_entity(scan2mesh(reconstruction_pcd))
        # compute f-scores
        f_score_sanity_check = compute_f_score(ground_truth_mesh_cad, ground_truth_mesh_scan)
        f_score = compute_f_score(ground_truth_mesh_scan, tumor_mesh, center=False)
        f_score_with_CF = compute_f_score(ground_truth_mesh_scan, tumor_mesh_with_CF)
        print(f"F-score sanity check: {f_score_sanity_check}")
        print(f"F-score with CF: {f_score_with_CF}")
        print(f"F-score without CF: {f_score}")
        yaml.dump({
            "f_score": f_score,
            "f_score_with_CF": f_score_with_CF,
            "tau": TAU
        }, open(save_path / "f_score.yaml", "w"))

        o3d.io.write_triangle_mesh(str(save_path / "mesh_without_CF.ply"), tumor_mesh)
        o3d.io.write_triangle_mesh(str(save_path / "mesh_with_CF.ply"), tumor_mesh_with_CF)
        o3d.io.write_triangle_mesh(str(save_path / "mesh_gt_scan.ply"), ground_truth_mesh_scan)
        o3d.io.write_triangle_mesh(str(save_path / "mesh_gt_cad.ply"), ground_truth_mesh_cad)

        if args.glob:
            f_score_results[i, 0] = f_score_with_CF
            f_score_results[i, 1] = f_score

        if rerun:
            for mesh in [
                    tumor_mesh, tumor_mesh_with_CF, ground_truth_mesh_scan, ground_truth_mesh_cad
            ]:
                mesh.compute_vertex_normals()
                o3d.visualization.draw_geometries([mesh])

    for tumor in TUMOR_STL.keys():
        for algo in ALGOS:
            fscores_ids = dataset_map[(tumor, algo)]
            print("{},{}: {}".format(tumor, algo, np.mean(f_score_results[fscores_ids], axis=0)))
