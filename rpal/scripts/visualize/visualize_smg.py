import rpal.utils.constants as rpal_const
import numpy as np
from rpal.algorithms.grid import SurfaceGridMap
import open3d as o3d
from rpal.utils.pcd_utils import scan2mesh, mesh2roi, visualize_pcds

BBOX_ROI = np.array([[0.5019415445763421, -0.051618050819477325, 0.5774690032616653],
                     [0.5034256408649197, -0.051676614297235135, -0.44316242513642623],
                     [0.49413665553621955, 0.04050607695706571, 0.5774523681515827],
                     [0.5808408971658378, -0.04493356316916076, 0.5775833469573993],
                     [0.5745201044142928, 0.04713200112962447, -0.44306471655077473],
                     [0.5730360081257152, 0.04719056460738228, 0.5775667118473168],
                     [0.5823249934544154, -0.04499212664691857, -0.4430480814406921],
                     [0.49562075182479715, 0.0404475134793079, -0.4431790602465089]])

if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud(str(rpal_const.SURFACE_SCAN_PATH))
    surface_mesh = scan2mesh(pcd)
    o3d.visualization.draw_geometries([surface_mesh])
    roi_pcd = mesh2roi(surface_mesh, bbox_pts=BBOX_ROI)
    surface_grid_map = SurfaceGridMap(roi_pcd, grid_size=rpal_const.PALP_CONST.grid_size)
    surface_grid_map._grid_pcd.paint_uniform_color([0, 1, 1])

    visualize_pcds([surface_grid_map._grid_pcd], meshes=[surface_mesh])
