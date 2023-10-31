import open3d as o3d
import numpy as np
import yaml
import time
import math
import queue
import threading

EPS = np.finfo(float).eps * 4.0


def thread_read(buffer, capture, stop_event):
    while not stop_event.is_set():
        data = capture.read()
        buffer.put(data)


class Buffer:
    def __init__(self, max_size):
        self.queue = queue.LifoQueue()
        self.max_size = max_size

    def put(self, item):
        if self.queue.qsize() >= self.max_size:
            self.queue.get()
        self.queue.put(item)

    def get(self):
        return self.queue.get()


class DatasetWriter:
    def __init__(self, cfg_args, record_pcd=True, print_hz=True):
        import datetime
        from devices import RealsenseCapture
        from utils import Hz
        from pathlib import Path
        import os
        import threading

        self.record_pcd = record_pcd

        self.stop_event = threading.Event()

        self.hz = Hz(print_hz=print_hz)
        self.save_buffer = []
        self.i = 0

        data_dir = Path("./data")
        # Create dataset folders
        self.dataset_folder = data_dir / datetime.datetime.now().strftime(
            "dataset_%m-%d-%Y_%H-%M-%S"
        )
        self.raw_pcd_dir = self.dataset_folder / "raw_pcd"
        self.timeseries_file = self.dataset_folder / "timeseries.txt"
        self.reconstruction_file = self.dataset_folder / "reconstruction.ply"
        self.reconstruction_raw = self.dataset_folder / "reconstruction.txt"
        self.surface_pcd = self.dataset_folder / "surface.ply"
        os.mkdir(self.dataset_folder)

        if self.record_pcd:
            os.mkdir(self.raw_pcd_dir)
            self.pcd_thread.start()
            self.rs_cap = RealsenseCapture()
            self.pcd_buffer = Buffer(10)
            self.pcd_thread = threading.Thread(
                target=thread_read, args=(self.pcd_buffer, self.rs_cap, self.stop_event)
            )

        with open(str(self.dataset_folder / "config.yml"), "w") as outfile:
            yaml.dump(vars(cfg_args), outfile, default_flow_style=False)

    def save_subsurface_pcd(self, pts):
        subsurface_pcd = o3d.geometry.PointCloud()
        subsurface_pcd.points = o3d.utility.Vector3dVector(pts)
        o3d.io.write_point_cloud(
            str(self.reconstruction_file.absolute()), subsurface_pcd
        )

        np.savetxt(str(self.reconstruction_raw), pts, fmt="%1.8f")

    def save_cropped_surface_pcd(self, pcd):
        o3d.io.write_point_cloud(str(self.surface_pcd.absolute()), pcd)

    def save(self):
        self.stop_event.set()
        if self.record_pcd:
            self.pcd_thread.join()

        np.savetxt(str(self.timeseries_file), self.save_buffer, fmt="%1.8f")

        save = input("Save or not? (enter 0 or 1)")
        save = bool(int(save))

        if not save:
            import shutil

            shutil.rmtree(f"{str(self.dataset_folder)}")

    def update(self, d_eef_pos_quat, Fxyz):
        if self.record_pcd:
            pcd = self.pcd_buffer.get()
        if Fxyz is not None:
            self.hz.clock()
            x = np.zeros(10)  # (x,y,z,x,y,z,w,fx,fy,fz)
            x[:3] = d_eef_pos_quat[1].flatten()
            x[3:7] = d_eef_pos_quat[0].flatten()
            x[7:10] = Fxyz
            self.save_buffer.append(x)
            if self.record_pcd:
                o3d.io.write_point_cloud(
                    str((self.raw_pcd_dir / f"{self.i}.ply").absolute()), pcd
                )
            self.i += 1


class Hz:
    BUFFER_SIZE = 10
    PRINT_DELAY = 2

    def __init__(self, print_hz=False):
        self.last_t = None
        self.hz = None
        self.buffer = np.zeros(self.BUFFER_SIZE)
        self.b_i = 0

        self.last_print_t = None
        self.print_hz = print_hz

    def clock(self):
        if self.last_t is None:
            self.last_t = time.time()
            return
        dt = time.time() - self.last_t
        self.buffer[self.b_i] = 1 / dt
        self.b_i = (self.b_i + 1) % self.BUFFER_SIZE
        self.last_t = time.time()

        if self.last_print_t is None:
            self.last_print_t = time.time()
        elif self.print_hz and time.time() - self.last_print_t > self.PRINT_DELAY:
            print("HZ: ", self.get_hz())
            self.last_print_t = time.time()

    def get_hz(self):
        return self.buffer.mean()


def get_rot_mat_from_basis(b1, b2, b3):
    A = np.eye(3)
    A[:, 0] = b1
    A[:, 1] = b2
    A[:, 2] = b3
    return A.T


def rot_from_a_to_b(a: np.ndarray, b: np.ndarray):
    cross_1_2 = np.cross(a, b)
    skew_symm_cross_1_2 = np.array(
        [
            [0, -cross_1_2[2], cross_1_2[1]],
            [cross_1_2[2], 0, -cross_1_2[0]],
            [-cross_1_2[1], cross_1_2[0], 0],
        ]
    )
    cos = np.dot(a, b)
    R = (
        np.identity(3)
        + skew_symm_cross_1_2
        + np.dot(skew_symm_cross_1_2, skew_symm_cross_1_2) * 1 / (1 + cos + 1e-15)
    )
    return R


def three_pts_to_rot_mat(p1, p2, p3, neg_x=False):
    xaxis = unit(p2 - p1)
    if neg_x:
        xaxis *= -1
    v_another = unit(p3 - p1)
    zaxis = -unit(np.cross(xaxis, v_another))
    yaxis = unit(np.cross(zaxis, xaxis))
    return get_rot_mat_from_basis(xaxis, yaxis, zaxis)


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


def unit(v):
    return v / np.linalg.norm(v)


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


def unit_vector(data, axis=None, out=None):
    """
    Returns ndarray normalized by length, i.e. eucledian norm, along axis.

    E.g.:
        >>> v0 = numpy.random.random(3)
        >>> v1 = unit_vector(v0)
        >>> numpy.allclose(v1, v0 / numpy.linalg.norm(v0))
        True

        >>> v0 = numpy.random.rand(5, 4, 3)
        >>> v1 = unit_vector(v0, axis=-1)
        >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=2)), 2)
        >>> numpy.allclose(v1, v2)
        True

        >>> v1 = unit_vector(v0, axis=1)
        >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=1)), 1)
        >>> numpy.allclose(v1, v2)
        True

        >>> v1 = numpy.empty((5, 4, 3), dtype=numpy.float32)
        >>> unit_vector(v0, axis=1, out=v1)
        >>> numpy.allclose(v1, v2)
        True

        >>> list(unit_vector([]))
        []

        >>> list(unit_vector([1.0]))
        [1.0]

    Args:
        data (np.array): data to normalize
        axis (None or int): If specified, determines specific axis along data to normalize
        out (None or np.array): If specified, will store computation in this variable

    Returns:
        None or np.array: If @out is not specified, will return normalized vector. Otherwise, stores the output in @out
    """
    if out is None:
        data = np.array(data, dtype=np.float32, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


def quat2mat(quaternion):
    """
    Converts given quaternion to matrix.

    Args:
        quaternion (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: 3x3 rotation matrix
    """
    # awkward semantics for use with numba
    inds = np.array([3, 0, 1, 2])
    q = np.asarray(quaternion).copy().astype(np.float32)[inds]

    n = np.dot(q, q)
    if n < EPS:
        return np.identity(3)
    q *= math.sqrt(2.0 / n)
    q2 = np.outer(q, q)
    return np.array(
        [
            [1.0 - q2[2, 2] - q2[3, 3], q2[1, 2] - q2[3, 0], q2[1, 3] + q2[2, 0]],
            [q2[1, 2] + q2[3, 0], 1.0 - q2[1, 1] - q2[3, 3], q2[2, 3] - q2[1, 0]],
            [q2[1, 3] - q2[2, 0], q2[2, 3] + q2[1, 0], 1.0 - q2[1, 1] - q2[2, 2]],
        ]
    )


def pose2mat(pos_quat):
    """
    Converts pose to homogeneous matrix.

    Args:
        pose (2-tuple): a (pos, orn) tuple where pos is vec3 float cartesian, and
            orn is vec4 float quaternion.

    Returns:
        np.array: 4x4 homogeneous matrix
    """
    homo_pose_mat = np.zeros((4, 4), dtype=np.float32)
    homo_pose_mat[:3, :3] = quat2mat(pos_quat[3:7])
    homo_pose_mat[:3, 3] = np.array(pos_quat[:3], dtype=np.float32)
    homo_pose_mat[3, 3] = 1.0
    return homo_pose_mat
