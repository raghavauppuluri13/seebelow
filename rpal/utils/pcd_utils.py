import numpy as np
import open3d as o3d
from rpal.utils.constants import array2constant
from scipy.spatial.transform import Rotation


def disk_pcd(radius, num_points):
    # Generate random values for u
    u = np.random.uniform(0, 1, num_points)
    # Apply the inverse transform to get r
    r = np.sqrt(u)
    r = radius * r
    # Generate random values for theta
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    # Convert to Cartesian coordinates
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.zeros(num_points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack((x, y, z)).T)
    return pcd


def clustering(pcd, eps=0.02, min_points=10):
    # Cluster the point cloud using DBSCAN
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))

    # Get the maximum label to determine the number of clusters
    max_label = labels.max()

    # Create a list to store the point clouds associated with each cluster
    clustered_point_clouds = []

    # Iterate through each cluster and extract the associated points to form separate point clouds
    for label in range(max_label + 1):
        if label != -1:  # Exclude the points labeled as noise
            clustered_points = pcd.select_by_index(np.where(labels == label)[0])
            clustered_point_clouds.append(clustered_points)

    return clustered_point_clouds


def color_filter(pcd, color_to_filter=[0.0, 0.0, 0.0], threshold=0.1):
    # Calculate the Euclidean distance between each point's color and the color to filter
    distances = np.linalg.norm(np.asarray(pcd.colors) - color_to_filter, axis=1)
    indices = np.where(distances < threshold)[0]
    # Create a new point cloud with the filtered points
    filtered_pcd = pcd.select_by_index(indices)
    return filtered_pcd


def color_icp(
    source,
    target,
    voxel_radius=[0.001, 0.001, 0.001],
    max_iter=[50, 30, 14],
    vis=False,
):
    current_transformation = np.identity(4)
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        print([iter, radius, scale])

        print("3-1. Downsample with a voxel size %.2f" % radius)
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        print("3-2. Estimate normal.")
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        print("3-3. Applying colored point cloud registration")
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down,
            target_down,
            radius,
            current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                              relative_rmse=1e-6,
                                                              max_iteration=iter),
        )

    if vis:
        draw_registration_result_original_color(source, target, result_icp.transformation)
    return result_icp.transformation


def stl_to_pcd(stl_path, scale=0.001, transform=np.eye(4), color=[1, 0, 0]):
    gt_mesh = o3d.io.read_triangle_mesh(stl_path)
    verts = np.asarray(gt_mesh.vertices)
    verts *= scale
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(verts)
    gt_pcd = gt_pcd.transform(transform)
    gt_pcd.paint_uniform_color(color)
    return gt_pcd


def animate_point_cloud(pcd, other_geoms=[]):
    pts = np.asarray(pcd.points)
    lookat_point = pts.mean(axis=0)

    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.set_lookat(lookat_point)
        ctr.rotate(2.0, 2.0)  # Adjust rotation speed by changing these values
        return False

    # Calculate the mean point (center) of the point cloud
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_animation_callback(rotate_view)
    vis.create_window()
    vis.add_geometry(pcd)
    for geom in other_geoms:
        vis.add_geometry(geom)
    vis.run()
    vis.destroy_window()


def scan2mesh(pcd):
    from rpal.utils.constants import BBOX_PHANTOM, GT_SCAN_POSE

    bbox = pick_surface_bbox(pcd, bbox_pts=BBOX_PHANTOM)
    pcd = pcd.crop(bbox)
    pcd = pcd.voxel_down_sample(voxel_size=0.001)
    camera = list(GT_SCAN_POSE[:3])
    radius = 1 * 100
    _, pt_map = pcd.hidden_point_removal(camera, radius)
    pcd = pcd.select_by_index(pt_map)
    pcd.compute_convex_hull()
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(10)

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd)[0]
    mesh.compute_vertex_normals()
    mesh.remove_degenerate_triangles()
    return mesh


def mesh2roi(surface_mesh, bbox_pts=None):
    surface_mesh = surface_mesh.subdivide_midpoint(number_of_iterations=1)
    surface_mesh.compute_vertex_normals()
    surface_mesh.remove_degenerate_triangles()

    surface_pcd = o3d.geometry.PointCloud()
    surface_pcd.points = o3d.utility.Vector3dVector(np.asarray(surface_mesh.vertices))
    surface_pcd.colors = o3d.utility.Vector3dVector(np.asarray(surface_mesh.vertex_colors))

    bbox: o3d.geometry.OrientedBoundingBox = pick_surface_bbox(surface_pcd, bbox_pts=bbox_pts)

    print(array2constant("BBOX_ROI", np.asarray(bbox.get_box_points())))

    surface_pcd.estimate_normals()
    surface_pcd.normalize_normals()
    surface_pcd.orient_normals_consistent_tangent_plane(k=100)
    surface_pcd = surface_pcd.crop(bbox)
    return surface_pcd


def box_center_to_corner(box_center):
    """https://stackoverflow.com/questions/62938546/how-to-draw-bounding-boxes-and-update-them-real-time-in-python"""
    # To return
    corner_boxes = np.zeros((8, 3))

    translation = box[0:3]
    h, w, l = size[3], size[4], size[5]
    rotation = box[6]

    # Create a bounding box outline
    bounding_box = np.array([
        [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
        [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2],
    ])

    # Standard 3x3 rotation matrix around the Z axis
    rotation_matrix = np.array([
        [np.cos(rotation), -np.sin(rotation), 0.0],
        [np.sin(rotation), np.cos(rotation), 0.0],
        [0.0, 0.0, 1.0],
    ])

    # Repeat the [x, y, z] eight times
    eight_points = np.tile(translation, (8, 1))

    # Translate the rotated bounding box by the
    # original center position to obtain the final box
    corner_box = np.dot(rotation_matrix, bounding_box) + eight_points.transpose()

    corner_box = corner_box.transpose()

    # Our lines span from points 0 to 1, 1 to 2, 2 to 3, etc...
    lines = [
        [0, 1],
        [1, 2],
        [2, 3],
        [0, 3],
        [4, 5],
        [5, 6],
        [6, 7],
        [4, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]

    # Use the same color for all lines
    colors = [[1, 0, 0] for _ in range(len(lines))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corner_box)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def pick_surface_bbox(pcd, bbox_pts=None):
    if bbox_pts is None:
        print("")
        print("1) Please pick 4 point as the corners your bounding box [shift + left click].")
        print("   Press [shift + right click] to undo point picking")
        print("2) After picking points, press 'Q' to close the window")

        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()  # user picks points
        vis.destroy_window()
        points_idx = vis.get_picked_points()
        pcd_npy = np.asarray(pcd.points)
        bbox_pts = np.zeros((8, 3))
        pts = pcd_npy[points_idx]
        pts[:, -1] += 0.5
        bbox_pts[:4] = pts
        pts[:, -1] -= 1
        bbox_pts[4:8] = pts
    bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(bbox_pts))
    return bbox


def crop_pcd(pcd, R, t, scale, bbox_params, visualize=False):
    # pretranslate
    T = np.eye(4)
    T[:3, 3] = t
    pcd.transform(T)

    # rotate
    T = np.eye(4)
    T[:3, :3] = R
    pcd.transform(T)

    # bbox
    corners = get_centered_bbox(*bbox_params)
    corners *= scale
    aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(corners))

    # Crop the point cloud
    cropped_pcd = pcd.crop(aabb)
    if visualize:
        visualize_pcds([cropped_pcd], frames=list(corners))

    return cropped_pcd


def get_centered_bbox(x_pos, x_neg, y_pos, y_neg, z_pos, z_neg):
    x_pts = [[x_pos, 0, 0], [x_neg, 0, 0]]
    y_pts = [[0, y_pos, 0], [0, y_neg, 0]]
    z_pts = [[0, 0, z_pos], [0, 0, z_neg]]

    pts = []
    for x in x_pts:
        for y in y_pts:
            for z in z_pts:
                pt = np.array([x, y, z]).sum(axis=0)
                pts.append(pt)
    pts = np.array(pts)
    return pts


def visualize_pcds(pcds, meshes=[], frames=[], tfs=[], surf_norms=[], tf_size=0.1):
    pts = np.asarray(pcds[0].points)
    lookat_point = pts.mean(axis=0)

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1,  # specify the size of coordinate frame
    )
    pcds.append(frame)

    for frame in frames:
        pcds.append(
            o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=tf_size,
                origin=list(frame)  # specify the size of coordinate frame
            ))

    for tf in tfs:
        f = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=tf_size  # specify the size of coordinate frame
        )
        f.transform(tf)
        pcds.append(f)

    # Get the camera parameters of the visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().mesh_show_back_face = True

    for pcd in pcds:
        vis.add_geometry(pcd)

    for mesh in meshes:
        vis.add_geometry(mesh)

    for p, normal in surf_norms:
        arrow: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.01,
            cone_radius=0.015,
            cylinder_height=0.05,
            cone_height=0.01,
        )
        arrow.paint_uniform_color([0.5, 0.5, 0.5])
        T = np.eye(4)
        result = Rotation.align_vectors(
            np.array([normal]),
            np.array([[0, 0, 1]]),
        )
        T[:3, :3] = result[0].as_matrix()
        T[:3, 3] = p
        arrow.transform(T)
        vis.add_geometry(arrow)

    ctr = vis.get_view_control()
    lookat_pts = []
    for pcd in pcds:
        if isinstance(pcd, o3d.geometry.PointCloud):
            pts = np.asarray(pcd.points)
            lookat_pts.append(pts.mean(axis=0))
    lookat_point = np.array(lookat_pts).mean(axis=0)
    ctr.set_lookat(lookat_point)

    # Update the visualization window
    vis.run()
    vis.destroy_window()
