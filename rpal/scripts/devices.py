import cv2
import time
#import k4a
import open3d as o3d

import urllib
import argparse
import datetime

import pyrealsense2 as rs

from multiprocessing import Process
from util import crop_pcd, visualize_pcds, unit

from k4a._bindings.k4atypes import *

from typing import NamedTuple

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
'''
class KinectCapture:
    def __init__(self):
        self.device = k4a.Device.open()

        # Start Cameras
        self.device_config = DeviceConfiguration(
            color_format = EImageFormat.COLOR_BGRA32,
            color_resolution = EColorResolution.RES_720P,
            depth_mode = EDepthMode.NFOV_UNBINNED,
            camera_fps = EFramesPerSecond.FPS_30,
            synchronized_images_only = True,
            depth_delay_off_color_usec = 0,
            wired_sync_mode = EWiredSyncMode.STANDALONE,
            subordinate_delay_off_master_usec = 0,
            disable_streaming_indicator = False)

        self.device_config = k4a.DEVICE_CONFIG_BGRA32_1080P_WFOV_UNBINNED_FPS15
        self.device.start_cameras(self.device_config)

        # Get Calibration
        self.calibration = self.device.get_calibration(
            depth_mode=self.device_config.depth_mode,
            color_resolution=self.device_config.color_resolution)

        # Create Transformation
        self.transformation = k4a.Transformation(self.calibration)

    def release(self):
        self.device.stop_cameras()

    def read(self):
        capture = self.device.get_capture(-1)

        kinect_rgb = capture.color.data
        # Get Point Cloud
        point_cloud = self.transformation.depth_image_to_point_cloud(capture.depth, k4a.ECalibrationType.DEPTH)

        # Save Point Cloud To Ascii Format File. Interleave the [X, Y, Z] channels into [x0, y0, z0, x1, y1, z1, ...]
        height, width, channels = point_cloud.data.shape
        xyz_data = point_cloud.data.reshape(height * width, channels)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_data)
        return (pcd,kinect_rgb)
'''

class RealsenseCapture:

    def __init__(self):
        self.pipeline = rs.pipeline()
        self.pipeline.start()
        # Configure depth and color streams
        self.config = rs.config()

        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
        self.device = self.pipeline_profile.get_device()

        self.config.enable_stream(rs.stream.depth, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(self.config)

        # Get stream profile and camera intrinsics
        profile = self.pipeline.get_active_profile()
        depth_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.depth))
        depth_intrinsics = self.depth_profile.get_intrinsics()

        # Processing blocks
        self.pc = rs.pointcloud()
        self.decimate = rs.decimation_filter()
        self.decimate.set_option(rs.option.filter_magnitude, 1)
        self.colorizer = rs.colorizer()
    
    def read(self):

        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_frame = self.decimate.process(depth_frame)

        # Grab new intrinsics (may be changed by decimation)
        depth_intrinsics = rs.video_stream_profile(
            depth_frame.profile).get_intrinsics()
        w, h = depth_intrinsics.width, depth_intrinsics.height

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_colormap = np.asanyarray(
            colorizer.colorize(depth_frame).get_data())

        if state.color:
            mapped_frame, color_source = color_frame, color_image
        else:
            mapped_frame, color_source = depth_frame, depth_colormap

        points = pc.calculate(depth_frame)
        pc.map_to(mapped_frame)

        # Pointcloud data to arrays
        v, t = points.get_vertices(), points.get_texture_coordinates()
        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
        uv = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

        colors = np.hstack((uv, np.ones((100, 1))))  # add alpha channel

        pcd = o3d.PointCloud()

        pcd.points = o3d.utility.Vector3dVector(verts)  
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

        return pcd


