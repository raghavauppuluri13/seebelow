import open3d as o3d

# Load point cloud
pcd = o3d.io.read_point_cloud("surface.ply")

# Detect planar patches
oboxes = pcd.detect_planar_patches(normal_variance_threshold_deg=60, coplanarity_deg=75, min_plane_edge_length=0.01)

# Print number of detected patches
print("Detected {} patches".format(len(oboxes)))

# Visualize planar patches
geometries = []
for obox in oboxes:
    mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(
        obox, scale=[1, 1, 0.0001]
    )
    geometries.append(mesh)
o3d.visualization.draw_geometries(geometries)
