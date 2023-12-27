"""Generate camera calibration dataset for https://github.com/ToyotaResearchInstitute/handical"""
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
import pinocchio as pin
import yaml
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.utils.input_utils import input2action
from deoxys.utils.io_devices import SpaceMouse
from deoxys.utils.log_utils import get_deoxys_example_logger
from deoxys.utils.transform_utils import mat2euler, quat2mat
from rpal.utils.constants import *
from rpal.utils.data_utils import Hz
from rpal.utils.devices import RealsenseCapture
from rpal.utils.keystroke_counter import KeyCode, KeystrokeCounter
from rpal.utils.time_utils import Ratekeeper

INT_SIZE = ctypes.sizeof(ctypes.c_int)

PROPRIO_DIM = 7  # pos: x,y,z, rot: x,y,z,w


class CalibrationWriter:
    def __init__(self):
        import shutil
        import datetime

        self.images = []
        self.poses = []
        self.calibration_path = RPAL_CFG_PATH / datetime.datetime.now().strftime(
            "camera_calibration_%m-%d-%Y_%H-%M-%S"
        )
        self.poses_save_path = self.calibration_path / "final_ee_poses.txt"
        self.img_save_path = self.calibration_path / "imgs"
        self.calib_save_path = self.calibration_path / "config.yaml"

        with open(str(BASE_CALIB_FOLDER / "config.yaml"), "r") as file:
            self.calib_cfg = yaml.safe_load(file)

        self.calib_cfg["path_to_intrinsics"] = str(self.calibration_path)
        self.camera_name = "wrist_d415"

    def add(self, im, pos_euler):
        self.images.append(im)
        self.poses.append(pos_euler)

    def write(self):
        import shutil

        save = input("Save or not? (enter 0 or 1)")
        save = bool(int(save))
        if save:
            shutil.copytree(str(BASE_CALIB_FOLDER), str(self.calibration_path))
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


def deoxys_ctrl(shm_posearr_name, stop_event):
    existing_shm = shared_memory.SharedMemory(name=shm_posearr_name)
    O_T_EE_shm = np.ndarray(PROPRIO_DIM, dtype=np.float32, buffer=existing_shm.buf)
    robot_interface = FrankaInterface(
        str(RPAL_CFG_PATH / args.interface_cfg), use_visualizer=False, control_freq=20
    )

    osc_delta_ctrl_cfg = YamlConfig(str(RPAL_CFG_PATH / OSC_DELTA_CFG)).as_easydict()
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
    args = argparser.parse_args()

    O_T_EE = np.zeros(PROPRIO_DIM, dtype=np.float32)
    shm = shared_memory.SharedMemory(create=True, size=O_T_EE.nbytes)
    O_T_EE = np.ndarray(O_T_EE.shape, dtype=O_T_EE.dtype, buffer=shm.buf)
    stop_event = mp.Event()
    ctrl_process = mp.Process(target=deoxys_ctrl, args=(shm.name, stop_event))
    ctrl_process.start()

    rs = RealsenseCapture()
    calibration_writer = CalibrationWriter()
    with open(str(BASE_CALIB_FOLDER / "config.yaml"), "r") as file:
        calib_cfg = yaml.safe_load(file)
    cb_size = (
        calibration_writer.calib_cfg["board"]["nrows"],
        calibration_writer.calib_cfg["board"]["ncols"],
    )
    im, pcd = rs.read()

    rk = Ratekeeper(30)

    with KeystrokeCounter() as key_counter:
        try:
            while 1:
                if np.all(O_T_EE == 0):
                    print("Waiting for pose...")
                    continue
                im, new_pcd = rs.read()

                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char="r"):
                        ret, corners = cv2.findChessboardCorners(
                            im,
                            cb_size,
                            flags=cv2.CALIB_CB_ADAPTIVE_THRESH
                            + cv2.CALIB_CB_FAST_CHECK
                            + cv2.CALIB_CB_NORMALIZE_IMAGE,
                        )

                        if ret:
                            print(f"added #{len(calibration_writer.images)} data point")
                            print(f"O_T_EE: {O_T_EE}")
                            calibration_writer.add(im, O_T_EE.copy())
                        else:
                            print(f"nope!")
                rk.keep_time()
        except KeyboardInterrupt:
            pass
        stop_event.set()
        ctrl_process.join()
        print("CTRL STOPPED!")

        calibration_writer.write()

        shm.close()
        shm.unlink()
