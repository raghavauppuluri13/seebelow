import argparse
import datetime
import time
import urllib
from multiprocessing import Process
from typing import NamedTuple

import cv2
import numpy as np
import open3d as o3d
import open3d.core as o3c
import pyrealsense2 as rs
import serial


class RealsenseCapture:
    def __init__(self):

        self.rs_cfg = o3d.t.io.RealSenseSensorConfig(
            {
                "serial": "",
                "color_format": "RS2_FORMAT_RGB8",
                "color_resolution": "0,480",
                "depth_format": "RS2_FORMAT_Z16",
                "depth_resolution": "0,480",
                "fps": "60",
                "visual_preset": "RS2_L500_VISUAL_PRESET_MAX_RANGE",
            }
        )

        self.rs = o3d.t.io.RealSenseSensor()
        self.rs.init_sensor(self.rs_cfg)
        self.rs.start_capture(True)  # true: start recording with capture
        self.intrinsics = o3c.Tensor(self.rs.get_metadata().intrinsics.intrinsic_matrix)

    def read(self, get_mask=None):
        im_rgbd = self.rs.capture_frame(True, True)  # wait for frames and align them

        color_tensor = im_rgbd.color.as_tensor()
        color_np = color_tensor.numpy()

        if get_mask is not None:
            mask = get_mask(color_np.copy())
            depth_tensor = im_rgbd.depth.as_tensor()
            depth_np = depth_tensor.numpy()
            depth_np[mask == 0] = 0

        pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(
            im_rgbd, self.intrinsics
        ).to_legacy()

        return color_np, pcd


class ForceSensor:
    def __init__(self, port="/dev/ttyACM0", baudrate="115200"):
        self.serial = serial.Serial(port=port)
        self.serial.baudrate = baudrate

        while True:
            while self.serial.inWaiting() == 1:
                pass
            bytes_data = self.serial.readline()
            print(str(bytes_data, encoding="utf-8"))
            if bytes_data.startswith(bytes("T:", "utf-8")):
                print("initialized")
                break
            else:
                self.serial.write(bytes("ACK", "utf-8"))

    def read(self):
        if self.serial.inWaiting() == 0:
            return
        else:
            while self.serial.inWaiting() == 2:
                continue
            self.data_stream = self.serial.readline()
        self.data_string = str(self.data_stream)
        self.elements = self.data_string.split(",")
        Fxyz = np.array(
            [
                (float)(self.elements[1]),
                (float)(self.elements[2]),
                (float)(self.elements[3]),
            ],
            dtype=np.double,
        )
        return Fxyz
