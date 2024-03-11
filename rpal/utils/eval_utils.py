import open3d as o3d


def compute_f_score(mesh_gt, mesh_reconstructed, tau=5.0):
    # Convert meshes to point clouds
    pcd_gt = mesh_gt.sample_points_uniformly(number_of_points=10000)
    pcd_reconstructed = mesh_reconstructed.sample_points_uniformly(number_of_points=10000)

    # Compute distances from reconstructed to ground-truth and vice versa
    dist_reconstructed_to_gt = pcd_reconstructed.compute_point_cloud_distance(pcd_gt)
    dist_gt_to_reconstructed = pcd_gt.compute_point_cloud_distance(pcd_reconstructed)

    # Compute precision and recall
    precision = sum([1
                     for d in dist_reconstructed_to_gt if d < tau]) / len(dist_reconstructed_to_gt)
    recall = sum([1 for d in dist_gt_to_reconstructed if d < tau]) / len(dist_gt_to_reconstructed)

    # Compute F-score
    if precision + recall > 0:
        f_score = 2 * (precision * recall) / (precision + recall)
    else:
        f_score = 0

    return f_score
