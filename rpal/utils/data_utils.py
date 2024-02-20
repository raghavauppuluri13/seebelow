import queue
import time
import numpy as np
import open3d as o3d
import yaml
import datetime
import os
import shutil

import rpal.utils.constants as rpal_const
from rpal.utils.config_utils import dict_from_class


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
    def __init__(self, print_hz=True):
        self.hz = Hz(print_hz=print_hz)
        self.save_buffer = []
        self.i = 0

        # Create dataset folders
        self.dataset_folder = (
            rpal_const.RPAL_DATA_PATH
            / datetime.datetime.now().strftime("dataset_%m-%d-%Y_%H-%M-%S")
        )
        self.raw_pcd_dir = self.dataset_folder / "raw_pcd"
        self.timeseries_file = self.dataset_folder / "timeseries.npy"
        self.reconstruction_file = self.dataset_folder / "reconstruction.ply"
        self.reconstruction_raw = self.dataset_folder / "reconstruction.txt"
        self.surface_pcd = self.dataset_folder / "surface.ply"
        os.mkdir(self.dataset_folder)
        with open(str(self.dataset_folder / "config.yml"), "w") as outfile:
            yaml.dump(
                dict_from_class(rpal_const.PALP_CONST),
                outfile,
                default_flow_style=False,
            )

    def save_subsurface_pcd(self, pts):
        subsurface_pcd = o3d.geometry.PointCloud()
        subsurface_pcd.points = o3d.utility.Vector3dVector(pts)
        o3d.io.write_point_cloud(
            str(self.reconstruction_file.absolute()), subsurface_pcd
        )

        np.savetxt(str(self.reconstruction_raw), pts, fmt="%1.8f")

    def save_roi_pcd(self, pcd):
        o3d.io.write_point_cloud(str(self.surface_pcd.absolute()), pcd)

    def save(self):
        np.save(str(self.timeseries_file), np.array(self.save_buffer))
        save = input(f"Save or not to {str(self.dataset_folder)}? (enter 0 or 1)")
        save = bool(int(save))
        if not save:
            shutil.rmtree(f"{str(self.dataset_folder)}")

    def add_sample(self, sample):
        assert sample.dtype == rpal_const.PALP_DTYPE
        self.save_buffer.append(sample)


class CalibrationWriter:
    def __init__(self):
        self.images = []
        self.poses = []
        self.calibration_path = (
            rpal_const.RPAL_CFG_PATH
            / datetime.datetime.now().strftime("camera_calibration_%m-%d-%Y_%H-%M-%S")
        )
        self.poses_save_path = self.calibration_path / "final_ee_poses.txt"
        self.img_save_path = self.calibration_path / "imgs"
        self.calib_save_path = self.calibration_path / "config.yaml"

        with open(str(rpal_const.BASE_CALIB_FOLDER / "config.yaml"), "r") as file:
            self.calib_cfg = yaml.safe_load(file)

        self.calib_cfg["path_to_intrinsics"] = str(self.calibration_path)
        self.camera_name = "wrist_d415"

    def add(self, im, pos_quat):
        self.images.append(im)
        self.poses.append(pos_quat)

    def write(self):
        import cv2
        import yaml
        from tqdm import tqdm

        save = input("Save or not? (enter 0 or 1)")
        save = bool(int(save))
        if save:
            shutil.copytree(
                str(rpal_const.BASE_CALIB_FOLDER), str(self.calibration_path)
            )
            os.mkdir(str(self.img_save_path))

            with open(str(self.calib_save_path), "w") as outfile:
                yaml.dump(self.calib_cfg, outfile, default_flow_style=False)
            for i in tqdm(range(len(self.images))):
                cv2.imwrite(
                    str(self.img_save_path / f"_{self.camera_name}_image{i}.png"),
                    self.images[i],
                )
            self.poses = np.array(self.poses, dtype=np.float32)
            print(self.poses[:5])
            poses = np.insert(
                self.poses, 0, np.arange(1, self.poses.shape[0] + 1), axis=1
            )
            # Save the array to a text file
            np.savetxt(
                str(self.poses_save_path),
                poses,
                fmt=tuple(["%d"] + ["%.8f"] * (poses.shape[1] - 1)),
                delimiter=" ",
            )
            print("Saved!")
        else:
            print("Aborted!")
