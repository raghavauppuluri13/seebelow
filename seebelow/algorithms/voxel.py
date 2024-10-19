import numpy as np
import open3d as o3d

# Create an empty voxel grid
voxel_grid = o3d.geometry.VoxelGrid()

# Set the voxel size
voxel_grid.voxel_size = 0.1

# Set the voxels at coordinates (0, 0, 0), (0, 0, 1), and (1, 1, 1) to 1
voxels = np.array([[0, 0, 0], [0, 0, 1], [1, 1, 1]])
voxel_grid.voxels = o3d.utility.Vector3iVector(voxels)

# Visualize the voxel grid
o3d.visualization.draw_geometries([voxel_grid])
