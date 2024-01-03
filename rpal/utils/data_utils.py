import math
import queue
import threading
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml


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
    def __init__(self, cfg_args=None, print_hz=True):
        import datetime
        from rpal.utils.constants import RPAL_DATA_PATH
        from pathlib import Path
        import os

        self.hz = Hz(print_hz=print_hz)
        self.save_buffer = []
        self.i = 0

        # Create dataset folders
        self.dataset_folder = RPAL_DATA_PATH / datetime.datetime.now().strftime(
            "dataset_%m-%d-%Y_%H-%M-%S"
        )
        self.raw_pcd_dir = self.dataset_folder / "raw_pcd"
        self.timeseries_file = self.dataset_folder / "timeseries.txt"
        self.reconstruction_file = self.dataset_folder / "reconstruction.ply"
        self.reconstruction_raw = self.dataset_folder / "reconstruction.txt"
        self.surface_pcd = self.dataset_folder / "surface.ply"
        os.mkdir(self.dataset_folder)

        if cfg_args is not None:
            with open(str(self.dataset_folder / "config.yml"), "w") as outfile:
                yaml.dump(vars(cfg_args), outfile, default_flow_style=False)

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
        save = input("Save or not? (enter 0 or 1)")
        save = bool(int(save))
        if not save:
            import shutil

            shutil.rmtree(f"{str(self.dataset_folder)}")

    def add_sample(self, sample):
        from rpal.utils.constants import PALP_DTYPE

        assert sample.dtype == PALP_DTYPE
        self.save_buffer.append(sample)
