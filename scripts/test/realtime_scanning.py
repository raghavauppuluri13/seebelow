import argparse
import ctypes
import multiprocessing as mp
import os
import subprocess
import sys
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
from deoxys.utils.transform_utils import mat2euler, mat2quat, quat2mat
from seebelow.utils.constants import *
from seebelow.utils.data_utils import Hz
from seebelow.utils.devices import RealsenseCapture
from seebelow.utils.time_utils import Ratekeeper
from seebelow.utils.transform_utils import euler2mat


def deoxys_ctrl(shm_posearr_name, stop_event):
    existing_shm = shared_memory.SharedMemory(name=shm_posearr_name)
    O_T_EE_shm = np.ndarray(7, dtype=np.float32, buffer=existing_shm.buf)
    print(args.interface_cfg)
    robot_interface = FrankaInterface(
        str(SEEBELOW_CFG_PATH / PAN_PAN_FORCE_CFG), use_visualizer=False, control_freq=20
    )

    osc_delta_ctrl_cfg = YamlConfig(str(SEEBELOW_CFG_PATH / OSC_DELTA_CFG)).as_easydict()
    device = SpaceMouse(vendor_id=SPACEM_VENDOR_ID, product_id=SPACEM_PRODUCT_ID)
    device.start_control()

    while len(robot_interface._state_buffer) == 0:
        continue

    while not stop_event.is_set():
        q, p = robot_interface.last_eef_quat_and_pos
        O_T_EE[:3] = p.flatten()
        O_T_EE[3:7] = q.flatten()

        action, grasp = input2action(
            device=device,
            controller_type=OSC_CTRL_TYPE,
        )

        robot_interface.control(
            controller_type=OSC_CTRL_TYPE,
            action=action,
            controller_cfg=osc_delta_ctrl_cfg,
        )

    robot_interface.control(
        controller_type=OSC_CTRL_TYPE,
        action=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [1.0],
        controller_cfg=osc_delta_ctrl_cfg,
        termination=True,
    )
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

    O_T_EE = np.zeros(7, dtype=np.float32)
    shm = shared_memory.SharedMemory(create=True, size=O_T_EE.nbytes)
    O_T_EE = np.ndarray(O_T_EE.shape, dtype=O_T_EE.dtype, buffer=shm.buf)
    stop_event = mp.Event()
    ctrl_process = mp.Process(target=deoxys_ctrl, args=(shm.name, stop_event))
    ctrl_process.start()

    rs = RealsenseCapture()
    with open(
        str(SEEBELOW_CFG_PATH / args.calibration_cfg / "extrinsics.yaml"), "r"
    ) as file:
        calib_cfg = yaml.safe_load(file)
        xyzxyzw = calib_cfg[args.cam_name]
        print(xyzxyzw)
        ee_pos = np.array(xyzxyzw[:3])
        ee_rot = quat2mat(xyzxyzw[-4:])
        E_T_C = np.eye(4)
        E_T_C[:3, :3] = ee_rot
        E_T_C[:3, 3] = ee_pos

    im, pcd = rs.read()

    rk = Ratekeeper(30)

    rtv = RealtimeVisualizer()
    rtv.add_frame("BASE")
    rtv.set_frame_tf("BASE", np.eye(4))
    rtv.add_frame("EEF", "BASE")
    rtv.add_frame("EEF_45", "EEF")
    rot_45 = np.eye(4)
    rot_45[:3, :3] = euler2mat([0, 0, -np.pi / 4])
    rtv.set_frame_tf("EEF_45", rot_45)
    rtv.add_frame("CAM", "EEF")
    rtv.add_frame("PHANTOM", "BASE")
    try:
        while 1:
            if np.all(O_T_EE == 0):
                print("Waiting for pose...")
                continue
            # im, new_pcd = rs.read()
            ee_pos = np.array(O_T_EE[:3])
            ee_rot = quat2mat(O_T_EE[3:7])
            O_T_E = np.eye(4)
            O_T_E[:3, :3] = ee_rot
            O_T_E[:3, 3] = ee_pos
            O_T_C = O_T_E @ E_T_C
            rtv.set_frame_tf("EEF", O_T_E)
            rtv.set_frame_tf("CAM", E_T_C)
            rk.keep_time()
    except KeyboardInterrupt:
        pass
    stop_event.set()
    ctrl_process.join()
    print("CTRL STOPPED!")

    shm.close()
    shm.unlink()
    vis.destroy_window()
