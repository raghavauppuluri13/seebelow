import open3d as o3d
from collections import defaultdict
import numpy as np
import argparse
import glob
from rpal.utils.constants import *
from rpal.utils.transform_utils import quat2mat
from rpal.utils.pcd_utils import (
    scan2mesh,
    mesh2roi,
    visualize_pcds,
    color_icp,
    color_filter,
    mesh2polyroi,
    pick_surface_bbox,
    stl_to_pcd,
    color_entity,
    disk_pcd,
    clustering,
    inverse_crop,
)
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
TUMOR_ID = {"hemisphere": 2, "crescent": 3}
TAU = 3e-3
ALGOS = ["bo", "random"]

# set to None if you want to reselect crop polygon geometry
EVAL_CROP = {
    "hemisphere": np.array(
        [
            [0.5325745344161987, 0.005577473901212215, 0.0685637779533863],
            [0.5281904935836792, 0.00452816765755415, 0.06843702122569084],
            [0.5247376561164856, 0.0020533567294478416, 0.0689111165702343],
            [0.5222628116607666, -0.0008339229971170425, 0.06852295622229576],
            [0.5202004909515381, -0.004693150520324707, 0.0678398534655571],
            [0.5193755626678467, -0.011558104306459427, 0.06682687997817993],
            [0.5202004909515381, -0.014857852831482887, 0.06484301388263702],
            [0.5222628116607666, -0.01803570706397295, 0.06506950035691261],
            [0.5251501202583313, -0.02053224854171276, 0.06536503881216049],
            [0.5300997495651245, -0.021869818679988384, 0.06668160483241081],
            [0.5375241637229919, -0.020632412284612656, 0.06790422648191452],
            [0.5421661734580994, -0.018570070154964924, 0.06432020664215088],
            [0.5457735061645508, -0.014857853297144175, 0.06309395655989647],
            [0.5483525991439819, -0.009083293378353119, 0.06289023160934448],
            [0.5471891164779663, -0.004133671522140503, 0.06520787253975868],
            [0.544123649597168, 0.004115698859095573, 0.06293772161006927],
            [0.5457735061645508, 0.00041433796286582947, 0.0641762875020504],
            [0.5388736128807068, 0.005765574052929878, 0.06536503881216049],
        ]
    ),
    "crescent": np.array(
        [
            [0.5196736454963684, -0.03556692600250244, 0.06781825795769691],
            [0.5226125717163086, -0.035873252898454666, 0.06809931993484497],
            [0.5247294902801514, -0.036262162029743195, 0.06716791912913322],
            [0.5247317850589752, -0.03315087594091892, 0.0677492506802082],
            [0.526707798242569, -0.027706125751137733, 0.06683722883462906],
            [0.528948962688446, -0.026928303763270378, 0.06764630973339081],
            [0.5253212749958038, -0.029650678858160973, 0.06675796210765839],
            [0.5331710875034332, -0.027030589058995247, 0.06686007231473923],
            [0.5356190204620361, -0.027317214757204056, 0.06664690375328064],
            [0.537752777338028, -0.028872858732938766, 0.06619352102279663],
            [0.5420056283473969, -0.031206322833895683, 0.06272066943347454],
            [0.5437861084938049, -0.03392869792878628, 0.062299979850649834],
            [0.5449528694152832, -0.037642963230609894, 0.06297096610069275],
            [0.545908659696579, -0.04092909023165703, 0.06370022892951965],
            [0.5448916256427765, -0.046762749552726746, 0.06335987895727158],
            [0.5437861084938049, -0.048707304522395134, 0.0641503594815731],
            [0.5411734580993652, -0.05065185949206352, 0.0658862367272377],
            [0.5381621718406677, -0.05142968147993088, 0.06730492785573006],
            [0.5358902513980865, -0.05220750346779823, 0.06780612096190453],
            [0.5297853350639343, -0.05220750346779823, 0.06804957613348961],
            [0.5328966379165649, -0.05220750346779823, 0.06825357675552368],
            [0.5236492156982422, -0.051818592473864555, 0.06583882123231888],
            [0.5204514861106873, -0.0510407704859972, 0.06408649682998657],
            [0.5188958644866943, -0.049561651423573494, 0.0635569617152214],
            [0.5173401832580566, -0.04716104455292225, 0.0638435035943985],
            [0.5159686207771301, -0.0440403763204813, 0.06390064209699631],
            [0.5155048072338104, -0.04092909023165703, 0.0637487918138504],
            [0.5179062485694885, -0.038206715136766434, 0.06608225405216217],
            [0.5189803242683411, -0.03665107488632202, 0.06678558513522148],
        ]
    ),
}


def compute_f_score(mesh_gt, mesh_reconstructed):
    tau = TAU

    def center_mesh(mesh):
        m = copy.deepcopy(mesh)
        center = m.get_center()
        m.translate(-center)
        return m

    def preprocess(mesh):
        # Center the meshes to their mass center
        # Subsample meshes
        mesh = center_mesh(mesh)
        pcd = mesh.sample_points_uniformly(3000)
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
    f_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return f_score


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
        default="hemisphere",
        help="Tumor type to evaluate",
    )
    parser.add_argument(
        "--rerun",
        type=bool,
        default=False,
        help="Visualize with rerun",
    )
    args = parser.parse_args()

    combined_tumor_recon = defaultdict(o3d.geometry.PointCloud)
    gt_scan = o3d.io.read_point_cloud(str(GT_PATH))
    gt_scan = color_filter(gt_scan, color_to_filter=[0.0, 0.0, 0.0], threshold=0.1)
    gt_tumors = clustering(gt_scan, eps=0.002, min_points=50)
    gt_tumors = sorted(gt_tumors, key=lambda pc: len(pc.points))

    print("Found {} tumour clusters".format(len(gt_tumors)))
    gt_tumors_scan_pcd = [
        color_entity(gt_tumor, color_map="rainbow") for gt_tumor in gt_tumors
    ]
    gt_tumors_scan_mesh = [
        color_entity(scan2mesh(gt_tumor)) for gt_tumor in gt_tumors_scan_pcd
    ]
    dataset_map = {}
    final_fscores_map = defaultdict(list)
    datasets = []
    if args.glob:
        for tumor in TUMOR_ID.keys():
            for algo in ALGOS:
                sets = glob.glob(str(RPAL_DATA_PATH / "*{}*{}*".format(tumor, algo)))
                dataset_map[(tumor, algo)] = np.arange(
                    len(datasets), len(datasets) + len(sets)
                )
                datasets += [RPAL_DATA_PATH / s for s in sets]

        f_scores = np.zeros((len(datasets), 2))
        print("Found {} datasets".format(len(datasets)))
    else:
        datasets = [RPAL_DATA_PATH / args.dataset_path]
    rerun = False
    if rerun:
        print("Spawning rerun")
        rr.init("rpal_eval", spawn=True)
    for i, dataset_path in enumerate(datasets):
        print(f"Evaluating {dataset_path.name}")
        palpations_cnt = 0

        tumor_pcd_without_CF = o3d.geometry.PointCloud()
        save_path = RPAL_MESH_PATH.parent / "eval" / f"eval_{dataset_path.name}"
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
        tumor_pcd_with_CF = o3d.io.read_point_cloud(str(recon_pth))
        tumor_pcd_with_CF = color_entity(tumor_pcd_with_CF)

        # ground truth init
        cfg = yaml.safe_load(open(str(dataset_path / "config.yml")))
        tumor_type = cfg["tumor_type"]
        algo = cfg["algo"]
        if args.combine:
            combined_tumor_recon[TUMOR_ID[tumor_type]] += tumor_pcd_with_CF
            continue
        print(f"Evaluating tumor {tumor_type}")
        ground_truth_mesh_scan = gt_tumors_scan_mesh[TUMOR_ID[tumor_type]]
        ground_truth_pcd_scan = gt_tumors[TUMOR_ID[tumor_type]]

        # log data
        if rerun:
            rr.log(
                "pcds/ground_truth_tumor",
                pcd_to_rr(
                    "gt_tumor",
                    np.asarray(gt_tumor.points),
                    colors=np.asarray(gt_tumor.colors),
                ),
            )
            rr.log(
                f"pcds/grid/{dataset_path.name}",
                pcd_to_rr("grid", np.asarray(grid_pcd.points)),
            )
            rr.log(
                f"pcds/roi/{dataset_path.name}",
                pcd_to_rr("roi", np.asarray(roi_pcd.points)),
            )
            recon_np = np.asarray(reconstruction_pcd.points)
            rr.log(
                f"pcds/reconstructed_tumor/{dataset_path.name}",
                pcd_to_rr(
                    "recon_tumor",
                    recon_np,
                    colors=np.asarray(reconstruction_pcd.colors),
                ),
            )
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
        O_p_f_t = np.where(
            np.diff(timeseries[:]["using_force_control_flag"], axis=0) == 1
        )[0]
        O_p_f = np.einsum("ijk->ik", timeseries[O_p_f_t]["O_p_EE"])
        if O_p_f.shape == O_p_surf.shape:
            stiffness_fz = Fxyz[O_p_f_t, 0, 2] / (
                np.linalg.norm(O_p_f - O_p_surf) + 1e-6
            )
            stiffness_fz /= 3000
            gradients = stiffness_fz
            assert (
                np.max(gradients) <= 1.0 and np.min(gradients) >= 0.0
            ), "Gradients should be between 0 and 1: {}, {}".format(
                np.max(gradients), np.min(gradients)
            )
            colors = cmap(norm(gradients))
            final_pcd = o3d.geometry.PointCloud()
            final_pcd.points = o3d.utility.Vector3dVector(O_p_f)
            final_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
            tumor_pcd_without_CF += final_pcd
            if rerun:
                rr.log(
                    f"pcds/positioning_end/{dataset_path.name}",
                    pcd_to_rr("positioning_end", O_p_f, colors),
                )
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
                        # stiffness_exp = timeseries[t]["stiffness"]
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
                        # rr.log("stiffness/experiment", rr.Scalar(stiffness_exp))
                        rr.log("stiffness/Fz", rr.Scalar(stiffness_fz))
                else:
                    init_O_p_surf = False
                    init_O_p_f = False
                palp_id = min(len(search_history) - 1, palp_id)
                grid = search_history[palp_id]["grid"]
                rr.log("search_grid", rr.Tensor(grid, dim_names=("batch", "X", "Y")))

        tumor_mesh_without_CF = mesh2polyroi(
            color_entity(scan2mesh(tumor_pcd_without_CF)),
            polybox_pts=EVAL_CROP[tumor_type],
            return_mesh=True,
        )
        tumor_mesh_with_CF = mesh2polyroi(
            color_entity(scan2mesh(tumor_pcd_with_CF)),
            polybox_pts=EVAL_CROP[tumor_type],
            return_mesh=True,
        )
        print(f"tumor_mesh_without_CF: {len(tumor_mesh_without_CF.vertices)}")
        print(f"tumor_mesh_with_CF: {len(tumor_mesh_with_CF.vertices)}")
        # compute f-scores
        f_score_without_CF = compute_f_score(
            ground_truth_mesh_scan, tumor_mesh_without_CF
        )
        f_score_with_CF = compute_f_score(ground_truth_mesh_scan, tumor_mesh_with_CF)
        # print(f"F-score sanity check: {f_score_sanity_check}")
        print(f"F-score with CF: {f_score_with_CF}")
        print(f"F-score without CF: {f_score_without_CF}")
        yaml.dump(
            {
                "f_score_with_CF": f_score_with_CF,
                "f_score_without_CF": f_score_without_CF,
                "tau": TAU,
            },
            open(save_path / "f_score.yaml", "w"),
        )

        o3d.io.write_triangle_mesh(
            str(save_path / "mesh_without_CF.ply"), tumor_mesh_without_CF
        )
        o3d.io.write_triangle_mesh(
            str(save_path / "mesh_with_CF.ply"), tumor_mesh_with_CF
        )
        o3d.io.write_triangle_mesh(
            str(save_path / "mesh_gt_scan.ply"), ground_truth_mesh_scan
        )

        if args.glob:
            f_scores[i, 0] = f_score_with_CF
            f_scores[i, 1] = f_score_without_CF
            # if (input("Do you want to eval this run? (y/n) ") == "y"):
            print(f"tumor {tumor_type}, algo {algo}")
            final_fscores_map[(tumor_type, algo)].append(i)
            print(
                "current # of samples for this category: ",
                len(final_fscores_map[(tumor_type, algo)]),
            )
        if rerun:
            for mesh in [tumor_mesh, tumor_mesh_with_CF, ground_truth_mesh_scan]:
                mesh.compute_vertex_normals()
                o3d.visualization.draw_geometries([mesh])
    if args.combine:
        for tumor, i in TUMOR_ID.items():
            combined_mesh = color_entity(scan2mesh(combined_tumor_recon[i]))
            combined_mesh_roi = mesh2polyroi(
                combined_mesh, polybox_pts=EVAL_CROP[tumor], return_mesh=True
            )
            ground_truth_mesh_scan = gt_tumors_scan_mesh[i]
            o3d.visualization.draw_geometries([combined_mesh_roi])
            fscore = compute_f_score(ground_truth_mesh_scan, combined_mesh_roi)
            print(f"Combined F-score for all experiments {tumor}: {fscore}")
    else:
        for tumor in TUMOR_ID.keys():
            for algo in ALGOS:
                fscores_ids = final_fscores_map[(tumor, algo)]
                print(
                    "MEAN {},{} #{}: {}".format(
                        tumor,
                        algo,
                        len(fscores_ids),
                        np.mean(f_scores[fscores_ids], axis=0),
                    )
                )
                print(
                    "MAX {},{} #{}: {}".format(
                        tumor,
                        algo,
                        len(fscores_ids),
                        np.max(f_scores[fscores_ids], axis=0),
                    )
                )
                print(
                    "MIN {},{} #{}: {}".format(
                        tumor,
                        algo,
                        len(fscores_ids),
                        np.min(f_scores[fscores_ids], axis=0),
                    )
                )
