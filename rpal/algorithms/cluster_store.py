import numpy as np
import open3d as o3d
from typing import Type, TypeVar

from rpal.scripts.utils import visualize_pcds, unit

"""
approximate surface using grid squares

1. build grid_center_pt to pointset map
2. build pointset to grid_center_pt map

normalize point
3. given new point, query kNN on grid_centers to get closest grid square using map
4. linearly interpolate within grid square to get normalized point

unnormalize point
1. linearly interpolate within grid square to get unnormalized point
"""

T = TypeVar("T", bound="Idx2D")


class Idx2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def iter(self, dx, dy) -> T:
        return Idx2D(self.x + dx, self.y + dy)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))


class GridStore:
    grid_size = 0.002
    nn = 5

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

        def project_to_plane(plane_normal, axis_to_project):
            axis_to_project /= np.linalg.norm(axis_to_project)
            proj_to_plane_from_axis = (
                np.dot(plane_normal, axis_to_project)
                / np.linalg.norm(plane_normal)
                * plane_normal
            )
            return unit(axis_to_project - proj_to_plane_from_axis)

        # ensure xaxis is orthogonal to zaxis
        if np.dot(np.cross(vecs[0], vecs[1]), zaxis) > 0:
            xaxis = vecs[1]
        else:
            xaxis = vecs[0]
        xaxis = vecs[0]
        xaxis = project_to_plane(zaxis, xaxis)

        yaxis = np.cross(zaxis, xaxis)

        self.T = np.eye(4)
        self.T[:3, :3] = np.hstack(
            [xaxis[:, np.newaxis], yaxis[:, np.newaxis], zaxis[:, np.newaxis]]
        )
        self.T[:3, 3] = origin
        visualize_pcds([self.pcd], tfs=[self.T])
        self.pcd.transform(self.pcd)

        bbox = self.pcd.get_axis_aligned_bounding_box()

        min_bound = bbox.get_min_bound()
        max_bound = bbox.get_max_bound()

        # (norm,center_pt) -> id
        # 2d grid[id]

        self.grid_list = []
        self.grid_inds = {}

        visualize_pcds([self.pcd])

        def build_grid(root_point, idx=Idx2D(0, 0)):
            if idx in self.grid_inds:
                return 0
            lower_bound = root_point - min_bound >= 0
            upper_bound = root_point - max_bound <= 0
            if np.alltrue(lower_bound) and np.alltrue(upper_bound):
                num_neighbors, inds, dists = self.pcd_tree.search_knn_vector_3d(
                    root_point, self.nn
                )
                grid_normal = np.mean(norms[inds])

                grid_sq = np.zeros(6)
                grid_sq[:3] = root_point
                grid_sq[3:] = grid_normal

                self.grid_list.append(grid_sq)
                self.grid_inds[idx] = len(self.grid_list) - 1

                xaxis_grid = project_to_plane(grid_normal, xaxis)
                yaxis_grid = np.cross(grid_normal, xaxis_grid)

                new_root_x = root_point + xaxis_grid * self.grid_size
                new_root_y = root_point + yaxis_grid * self.grid_size

                print("here")

                return build_grid(new_root_x, idx.iter(1, 0)) + build_grid(
                    new_root_y, idx.iter(0, 1)
                )

            else:
                return 0

        print(len(self.grid_list))

        build_grid(origin)
