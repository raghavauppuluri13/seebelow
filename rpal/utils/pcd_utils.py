import open3d as o3d
import numpy as np


def surface_mesh_to_pcd(mesh_path):
    surface_mesh = o3d.io.read_triangle_mesh(mesh_path)
    surface_mesh = surface_mesh.subdivide_midpoint(number_of_iterations=2)
    surface_mesh.compute_vertex_normals()
    surface_mesh.remove_degenerate_triangles()

    surface_pcd = o3d.geometry.PointCloud()
    surface_pcd.points = o3d.utility.Vector3dVector(np.asarray(surface_mesh.vertices))

    bbox = pick_surface_bbox(surface_pcd)

    surface_pcd.estimate_normals()
    surface_pcd.normalize_normals()
    surface_pcd.orient_normals_consistent_tangent_plane(k=100)
    surface_pcd = surface_pcd.crop(bbox)
    return surface_pcd


def pick_surface_bbox(pcd):
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

    ctr = vis.get_view_control().convert_to_pinhole_camera_parameters()

    # Set the center of the viewport to the origin
    ctr.extrinsic = np.eye(4)
    vis.get_view_control().convert_from_pinhole_camera_parameters(ctr)

    # Update the visualization window
    vis.run()
    vis.destroy_window()
