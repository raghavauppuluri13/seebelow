import argparse
import ctypes
import multiprocessing as mp
import os
import subprocess
import sys
from datetime import datetime
from io import StringIO
from multiprocessing import shared_memory
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pinocchio as pin
import yaml
from scipy.spatial.transform import Rotation
from tfvis.visualizer import RealtimeVisualizer
from tqdm import tqdm

from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.utils.input_utils import input2action
from deoxys.utils.io_devices import SpaceMouse
from deoxys.utils.log_utils import get_deoxys_example_logger
from deoxys.utils.transform_utils import mat2euler, mat2quat, quat2mat, quat2axisangle
from rpal.utils.constants import *
from rpal.utils.data_utils import Hz
from rpal.utils.devices import RealsenseCapture
from rpal.utils.keystroke_counter import KeyCode, KeystrokeCounter
from rpal.utils.pcd_utils import pick_surface_bbox, visualize_pcds
from rpal.utils.segmentation_utils import get_color_mask, get_hsv_threshold
from rpal.utils.time_utils import Ratekeeper
from rpal.utils.transform_utils import euler2mat
from rpal.utils.interpolator import Interpolator, InterpType


def deoxys_ctrl(shm_posearr_name, stop_event):
    existing_shm = shared_memory.SharedMemory(name=shm_posearr_name)
    O_T_EE_posquat_shm = np.ndarray(7, dtype=np.float32, buffer=existing_shm.buf)
    robot_interface = FrankaInterface(
        str(RPAL_CFG_PATH / PAN_PAN_FORCE_CFG), use_visualizer=False, control_freq=20
    )

    osc_absolute_ctrl_cfg = YamlConfig(
        str(RPAL_CFG_PATH / OSC_ABSOLUTE_CFG)
    ).as_easydict()
    interp = Interpolator(interp_type=InterpType.SE3)

    goals = []

    O_T_P = np.eye(4)
    O_T_P[:3, 3] = BBOX_PHANTOM.mean(axis=0)
    P_T_O = np.linalg.inv(O_T_P)
    O_T_E = np.eye(4)
    O_T_E[:3, :3] = quat2mat(GT_SCAN_POSE[3:7])
    O_T_E[:3, 3] = GT_SCAN_POSE[:3]

    goals.append(O_T_E)

    P_T_E = np.matmul(P_T_O, O_T_E)
    for ang in np.linspace(np.radians(5), np.radians(20), 3):
        Ry = np.eye(4)
        Ry[:3, :3] = euler2mat(np.array([0, ang, 0]))
        goals.append(O_T_P @ Ry @ P_T_E)

    while len(robot_interface._state_buffer) == 0:
        continue

    while len(goals) != 0 or not interp.done:
        q, p = robot_interface.last_eef_quat_and_pos
        O_T_EE_posquat[:3] = p.flatten()
        O_T_EE_posquat[3:7] = q.flatten()

        curr_eef_pose = robot_interface.last_eef_rot_and_pos
        curr_pose_se3 = pin.SE3.Identity()
        curr_pose_se3.rotation = curr_eef_pose[0]
        curr_pose_se3.translation = curr_eef_pose[1]

        if interp.done:
            pose_goal = goals.pop()
            goal_pose_se3 = pin.SE3.Identity()
            goal_pose_se3.rotation = pose_goal[:3, :3]
            goal_pose_se3.translation = pose_goal[:3, 3]
            interp.init(curr_pose_se3, goal_pose_se3, steps=int(STEP_FAST / 1))

        action = np.zeros(7)
        next_se3_pose = interp.next()
        xyz_quat = pin.SE3ToXYZQUAT(next_se3_pose)
        axis_angle = quat2axisangle(xyz_quat[3:7])

        action[:3] = xyz_quat[:3]
        action[3:6] = axis_angle

        robot_interface.control(
            controller_type=OSC_CTRL_TYPE,
            action=action,
            controller_cfg=osc_absolute_ctrl_cfg,
        )

    # stop
    stop_event.set()
    robot_interface.close()

    existing_shm.close()
    existing_shm.unlink()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--interface-cfg", type=str, default="pan-pan-force.yml")
    argparser.add_argument(
        "--calibration-cfg", type=str, default="camera_calibration_12-16-2023_14-48-12"
    )
    argparser.add_argument("--cam-name", type=str, default="wrist_d415")
    args = argparser.parse_args()

    O_T_EE_posquat = np.zeros(7, dtype=np.float32)
    shm = shared_memory.SharedMemory(create=True, size=O_T_EE_posquat.nbytes)
    O_T_EE_posquat = np.ndarray(
        O_T_EE_posquat.shape, dtype=O_T_EE_posquat.dtype, buffer=shm.buf
    )
    stop_event = mp.Event()
    ctrl_process = mp.Process(target=deoxys_ctrl, args=(shm.name, stop_event))
    ctrl_process.start()

    while np.all(O_T_EE_posquat == 0):
        continue

    # print(np_to_constant("GT_SCAN_POSE", O_T_EE_posquat))

    with open(
        str(RPAL_CFG_PATH / args.calibration_cfg / "extrinsics.yaml"), "r"
    ) as file:
        calib_cfg = yaml.safe_load(file)
        xyzxyzw = calib_cfg[args.cam_name]
        ee_pos = np.array(xyzxyzw[:3])
        ee_rot = quat2mat(xyzxyzw[-4:])
        E_T_C = np.eye(4)
        E_T_C[:3, :3] = ee_rot
        E_T_C[:3, 3] = ee_pos

    rs = RealsenseCapture()
    pcd = o3d.geometry.PointCloud()

    rk = Ratekeeper(1)

    rtv = RealtimeVisualizer()
    rtv.add_frame("BASE")
    rtv.set_frame_tf("BASE", np.eye(4))
    rtv.add_frame("EEF", "BASE")
    rtv.add_frame("EEF_45", "EEF")
    rot_45 = np.eye(4)
    rot_45[:3, :3] = euler2mat([0, 0, -np.pi / 4])
    rtv.set_frame_tf("EEF_45", rot_45)
    rtv.add_frame("CAM", "EEF")
    rtv.add_frame("TUMOR", "BASE")

    while not stop_event.is_set():
        # _, new_pcd = rs.read(get_mask=lambda x: get_color_mask(x, TUMOR_HSV_THRESHOLD))
        _, new_pcd = rs.read()

        ee_pos = np.array(O_T_EE_posquat[:3])
        ee_mat = quat2mat(O_T_EE_posquat[3:7])
        O_T_E = np.eye(4)
        O_T_E[:3, :3] = ee_mat
        O_T_E[:3, 3] = ee_pos
        O_T_C = O_T_E @ E_T_C
        new_pcd.transform(O_T_C)
        bbox = pick_surface_bbox(new_pcd, bbox_pts=BBOX_PHANTOM)
        new_pcd = new_pcd.crop(bbox)
        pcd += new_pcd
        tumor_pts = np.asarray(pcd.points)
        tumor_mean = tumor_pts.mean(axis=0)
        O_T_TUM = np.eye(4)
        O_T_TUM[:3, 3] = tumor_mean

        # tf visualizer
        rtv.set_frame_tf("EEF", O_T_E)
        rtv.set_frame_tf("CAM", E_T_C)
        rtv.set_frame_tf("TUMOR", O_T_TUM)

        rk.keep_time()

    ctrl_process.join()
    print("CTRL STOPPED!")

    now_str = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

    o3d.io.write_point_cloud(str(RPAL_MESH_PATH / f"tumors_gt_{now_str}.ply"), pcd)

    shm.close()
