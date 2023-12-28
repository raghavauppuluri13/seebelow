from functools import cached_property
from typing import Tuple, Type, TypeVar

import numpy as np
import open3d as o3d

from rpal.utils.math_utils import project_axis_to_plane, unit
from rpal.utils.pcd_utils import visualize_pcds


"""
approximate surface using grid squares

1. build grid_center_pt to pointset map
2. build pointset to grid_center_pt map

pt_to_idx point
3. given new point, query kNN on grid_centers to get closest grid square using map
4. linearly interpolate within grid square to get pt_to_idxd point

idx_to_pt
1. linearly interpolate within grid square to get unpt_to_idxd point
"""


class Grid:
    def __init__(self):
        self.grid = None

    @cached_property
    def vectorized_states(self):
        nx, ny = self.shape
        gx = np.arange(0, self.grid.shape[0])
        gy = np.arange(0, self.grid.shape[1])
        Xx, Xy = np.meshgrid(gx, gy)

        states = np.array([Xx.reshape(-1), Xy.reshape(-1)]).transpose()
        states = states[:, np.newaxis, :]
        return states

    def __getitem__(self, index):
        r, c = index
        return self.grid[r, c]

    def sample_states_uniform(self):
        vectorized_states = self.vectorized_states
        state = vectorized_states[np.random.randint(0, vectorized_states.shape[0])]
        return list(state.flatten())

    @property
    def shape(self):
        raise NotImplementedError()

    @property
    def grid_size(self):
        raise NotImplementedError()


class GridMap2D(Grid):
    def __init__(self, r, c, grid_size=0.001):
        self._r = r
        self._c = c
        self._grid_size = grid_size
        self.grid = np.zeros((self._r, self._c))

    @property
    def shape(self):
        return (self._r, self._c)

    @property
    def grid_size(self):
        return self._grid_size


class SurfaceGridMap(Grid):
    def __init__(self, pcd, grid_size=0.001, nn=10):
        self._grid_size = grid_size
        self._nn = nn

        self._pcd = pcd
        self._pcd.estimate_normals()
        self._pcd.normalize_normals()
        self._pcd.orient_normals_consistent_tangent_plane(k=100)

        verts = np.asarray(self._pcd.points)
        norms = np.asarray(self._pcd.normals)

        # generate grid origin
        vert_center = verts.mean(axis=0)
        dist_from_center = np.linalg.norm(verts - vert_center, axis=1)
        corner_pt_idxs = np.argpartition(dist_from_center, -4)[-4:]
        origin_idx = corner_pt_idxs[0]
        origin = verts[origin_idx]
        corners = verts[corner_pt_idxs]
        vecs = corners[1:] - origin

        # argsort by distance to ensure first two are xaxis and yaxis
        vecs = vecs[np.argsort(np.linalg.norm(vecs, axis=1))]
        zaxis_origin = norms[origin_idx]

        # ensure xaxis is orthogonal to zaxis
        if np.dot(np.cross(vecs[0], vecs[1]), zaxis_origin) > 0:
            xaxis_origin = vecs[0]
        else:
            xaxis_origin = vecs[1]
        xaxis_origin = project_axis_to_plane(zaxis_origin, xaxis_origin)
        yaxis_origin = np.cross(zaxis_origin, xaxis_origin)

        self._T = np.eye(4)
        self._T[:3, :3] = np.hstack(
            [
                xaxis_origin[:, np.newaxis],
                yaxis_origin[:, np.newaxis],
                zaxis_origin[:, np.newaxis],
            ]
        )
        self._T[:3, 3] = origin

        # use surface bounding box to stop grid expansion
        max_p = verts.max(axis=0) + 0.001
        min_p = verts.min(axis=0) - 0.001

        bbox_pts = []
        for x in [min_p[0], max_p[0]]:
            for y in [min_p[1], max_p[1]]:
                for z in [min_p[2], max_p[2]]:
                    bbox_pts.append(np.array([x, y, z]))
        bbox_pts = np.array(bbox_pts)
        self._bbox = o3d.geometry.OrientedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(bbox_pts)
        )
        self._bbox.color = [1, 0, 0]

        # cells are defined by their center point, normal vector, and xyz axes
        self._pcd_tree = o3d.geometry.KDTreeFlann(self._pcd)
        self._cells = []
        self.grid_idx2cell_idx = {}
        self.cell_idx2grid_idx = {}

        def build_grid(cell_center, grid_idx=(0, 0)):
            if grid_idx in self.grid_idx2cell_idx:
                return 0

            inds = self._bbox.get_point_indices_within_bounding_box(
                o3d.utility.Vector3dVector([cell_center])
            )

            # remove if not within bounding box
            if len(inds) > 0:
                num_neighbors, inds, dists = self._pcd_tree.search_knn_vector_3d(
                    cell_center, self._nn
                )

                # remove if a new cell is far away from any point in self._pcd
                if np.min(dists) > self._grid_size:
                    return 0

                # Offically a grid cell by now
                # grid_normal = np.mean(norms[inds], axis=0) -> NOTE: not used, creates unwanted artifacts
                grid_normal = norms[inds[np.argmin(dists)]]

                xaxis_cell = project_axis_to_plane(grid_normal, xaxis_origin)
                yaxis_cell = np.cross(grid_normal, xaxis_cell)

                new_cell_idx_center_dx = cell_center + xaxis_cell * self._grid_size
                new_cell_idx_center_dy = cell_center + yaxis_cell * self._grid_size

                self._cells.append((xaxis_cell, yaxis_cell, grid_normal, cell_center))

                cell_idx = len(self._cells) - 1
                self.grid_idx2cell_idx[grid_idx] = cell_idx
                self.cell_idx2grid_idx[cell_idx] = grid_idx

                new_cell_idx_in_x = (grid_idx[0] + 1, grid_idx[1] + 0)
                new_cell_idx_in_y = (grid_idx[0] + 0, grid_idx[1] + 1)

                return build_grid(
                    new_cell_idx_center_dx, new_cell_idx_in_x
                ) + build_grid(new_cell_idx_center_dy, new_cell_idx_in_y)
            else:
                return 0

        build_grid(origin)

        idxs = np.asarray(list(self.grid_idx2cell_idx.keys()))
        self._grid_shape = list(np.max(idxs, axis=0))
        self.grid = np.zeros(self._grid_shape)

        self._grid_arr = np.zeros((len(self.grid_idx2cell_idx), 3))
        # populate grid pcd
        for idx in self.grid_idx2cell_idx.values():
            self._grid_arr[idx] = self._cells[idx][-1]  # add root point
        self._grid_pcd = o3d.geometry.PointCloud()
        self._grid_pcd.points = o3d.utility.Vector3dVector(self._grid_arr)
        self._grid_pcd.estimate_normals()
        self._grid_pcd_tree = o3d.geometry.KDTreeFlann(self._grid_pcd)

    def pt_to_idx(self, pt):
        num_neighbors, inds, dists = self._grid_pcd_tree.search_knn_vector_3d(pt, 1)
        assert len(inds) == 1
        return self.cell_idx2grid_idx[inds[0]]

    def idx_to_pt(self, idx):
        assert isinstance(idx, tuple)
        cell_idx = self.grid_idx2cell_idx[idx]
        norm = self._grid_pcd.normals[cell_idx]
        pt = self._grid_pcd.points[cell_idx]
        return (pt, norm)

    def visualize(self, show_tf=False):
        tfs = []
        if show_tf:
            cluster_colors = np.random.rand(len(self._grid_pcd.points), 3)
            for grid, idx in self.grid_idx2cell_idx.items():
                xaxis, yaxis, zaxis, cell_center = self._cells[idx]
                # Create array of all possible combinations of 0 and 1 for x, y, and z
                corners = np.array(np.meshgrid([0, 1], [0, 1], [0, 1])).T.reshape(-1, 3)
                corners = corners.astype(np.float32)
                # Multiply by 2 and subtract 1 to get corners of unit cube
                corners = corners * 2 - 1
                corners_norm = np.zeros((8, 4))
                corners_norm[:, :3] = corners
                grid_T = np.eye(4)
                grid_T[:3, :3] = np.hstack(
                    [xaxis[:, np.newaxis], yaxis[:, np.newaxis], zaxis[:, np.newaxis]]
                )
                grid_T[:3, 3] = cell_center
                tfs.append(grid_T)

        # original pcd is cyan
        self._pcd.paint_uniform_color([0, 1, 1])
        # generated grid centers are in magenta
        self._grid_pcd.paint_uniform_color([1, 0, 1])
        visualize_pcds([self._grid_pcd, self._pcd, self._bbox], tfs=tfs)

    @property
    def shape(self):
        return tuple(self._grid_shape)

    @property
    def grid_size(self):
        return self._grid_size
