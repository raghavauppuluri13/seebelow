import rpal.utils.constants as rpal_const
import numpy as np
from rpal.algorithms.grid import SurfaceGridMap
import open3d as o3d
from rpal.utils.pcd_utils import scan2mesh, mesh2roi, visualize_pcds

if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud(str(rpal_const.SURFACE_SCAN_PATH))
    surface_mesh = scan2mesh(pcd)
    roi_mesh = mesh2roi(surface_mesh, bbox_pts=rpal_const.BBOX_DOCTOR_ROI, return_mesh=True)
    o3d.io.write_triangle_mesh('surface_mesh.ply', roi_mesh)
