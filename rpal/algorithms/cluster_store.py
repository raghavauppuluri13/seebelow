import numpy as np
import open3d as o3d

from rpal.scripts.utils import visualize_pcds


class GridStore:
    DX = 0.002

    def __init__(self, pcd):
        self.pcd = pcd
        self.pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        self.verts = np.asarray(pcd.points)
        self.norms = np.asarray(pcd.normals)
        self.center = self.verts.mean(axis=0)

        pcd.estimate_normals()
        pcd.normalize_normals()
        pcd.orient_normals_consistent_tangent_plane(k=100)

        verts = np.asarray(pcd.points)
        norms = np.asarray(pcd.normals)

        vert_center = verts.mean(axis=0)
        dist_from_center = np.linalg.norm(verts - vert_center, axis=1)

        corner_pt_idxs = np.argpartition(dist_from_center, -4)[-4:]

        origin_idx = corner_pt_idxs[0]
        origin = verts[origin_idx]
        corners = verts[corner_pt_idxs]
        vecs = corners[1:] - origin
        vecs = vecs[np.argsort(np.linalg.norm(vecs, axis=1))]
        zaxis = norms[origin_idx]
        print(np.dot(np.cross(vecs[0], vecs[1]), zaxis))
        if np.dot(np.cross(vecs[0], vecs[1]), zaxis) > 0:
            xaxis = vecs[0]
        else:
            xaxis = vecs[1]
        yaxis = np.cross(zaxis, xaxis)
        xaxis = vecs[0]
        xaxis /= np.linalg.norm(xaxis)
        proj_z_x = np.dot(zaxis, xaxis) / np.linalg.norm(zaxis) * zaxis
        xaxis = xaxis - proj_z_x
        xaxis /= np.linalg.norm(xaxis)

        T = np.eye(4)
        T[:3, :3] = np.hstack(
            [xaxis[:, np.newaxis], yaxis[:, np.newaxis], zaxis[:, np.newaxis]]
        )
        T[:3, 3] = origin

        visualize_pcds([pcd], tfs=[T])
