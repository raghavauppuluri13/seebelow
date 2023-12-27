import numpy as np
import open3d as o3d


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


def surface_mesh_to_pcd(mesh_path, bbox_pts=None):
    surface_mesh = o3d.io.read_triangle_mesh(mesh_path)
    surface_mesh = surface_mesh.subdivide_midpoint(number_of_iterations=2)
    surface_mesh.compute_vertex_normals()
    surface_mesh.remove_degenerate_triangles()

    surface_pcd = o3d.geometry.PointCloud()
    surface_pcd.points = o3d.utility.Vector3dVector(np.asarray(surface_mesh.vertices))
    surface_pcd.colors = o3d.utility.Vector3dVector(
        np.asarray(surface_mesh.vertex_colors)
    )

    bbox = pick_surface_bbox(surface_pcd, bbox_pts=bbox_pts)

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
    bounding_box = np.array(
        [
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2],
        ]
    )

    # Standard 3x3 rotation matrix around the Z axis
    rotation_matrix = np.array(
        [
            [np.cos(rotation), -np.sin(rotation), 0.0],
            [np.sin(rotation), np.cos(rotation), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

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
        print(
            "1) Please pick 4 point as the corners your bounding box [shift + left click]."
        )
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
    bbox = o3d.geometry.OrientedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(bbox_pts)
    )
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
        o3d.utility.Vector3dVector(corners)
    )

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


def visualize_pcds(pcds, frames=[], tfs=[]):
    pts = np.asarray(pcds[0].points)
    lookat_point = pts.mean(axis=0)

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1,  # specify the size of coordinate frame
    )
    pcds.append(frame)

    for frame in frames:
        pcds.append(
            o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.1, origin=list(frame)  # specify the size of coordinate frame
            )
        )

    for tf in tfs:
        f = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1  # specify the size of coordinate frame
        )
        f.transform(tf)
        pcds.append(f)

    # Get the camera parameters of the visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().mesh_show_back_face = True

    for pcd in pcds:
        vis.add_geometry(pcd)

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
