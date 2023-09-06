import open3d as o3d
from devices import RealsenseCapture

rs = RealsenseCapture()

pcd = rs.read()

o3d.io.write_point_cloud("./out.ply", pcd)
