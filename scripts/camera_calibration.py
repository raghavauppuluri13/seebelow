"""Generate camera calibration dataset for https://github.com/ToyotaResearchInstitute/handical"""
import argparse
import multiprocessing as mp
from multiprocessing import shared_memory

import cv2
import numpy as np
import yaml

from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.input_utils import input2action
from deoxys.utils.io_devices import SpaceMouse
import seebelow.utils.constants as seebelow_const
from seebelow.utils.data_utils import CalibrationWriter
from seebelow.utils.devices import RealsenseCapture
from seebelow.utils.keystroke_counter import KeyCode, KeystrokeCounter
from seebelow.utils.time_utils import Ratekeeper

PROPRIO_DIM = 7  # pos: x,y,z, rot: x,y,z,w


def deoxys_ctrl(shm_posearr_name, stop_event):
    existing_shm = shared_memory.SharedMemory(name=shm_posearr_name)
    O_T_EE = np.ndarray(PROPRIO_DIM, dtype=np.float32, buffer=existing_shm.buf)
    robot_interface = FrankaInterface(
        str(seebelow_const.PAN_PAN_FORCE_CFG),
        use_visualizer=False,
        control_freq=20,
    )

    osc_delta_ctrl_cfg = YamlConfig(str(seebelow_const.OSC_DELTA_CFG)).as_easydict()
    device = SpaceMouse(
        vendor_id=seebelow_const.SPACEM_VENDOR_ID, product_id=seebelow_const.SPACEM_PRODUCT_ID
    )
    device.start_control()

    while len(robot_interface._state_buffer) == 0:
        continue

    while not stop_event.is_set():
        q, p = robot_interface.last_eef_quat_and_pos
        O_T_EE[:3] = p.flatten()
        O_T_EE[3:7] = q.flatten()

        action, grasp = input2action(
            device=device,
            controller_type=seebelow_const.OSC_CTRL_TYPE,
        )

        robot_interface.control(
            controller_type=seebelow_const.OSC_CTRL_TYPE,
            action=action,
            controller_cfg=osc_delta_ctrl_cfg,
        )

    robot_interface.control(
        controller_type=seebelow_const.OSC_CTRL_TYPE,
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
    with open(str(seebelow_const.BASE_CALIB_FOLDER / "config.yaml"), "r") as file:
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
