import argparse
import os
import pickle
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.input_utils import input2action
from deoxys.utils.io_devices import SpaceMouse


from devices import RealsenseCapture, ForceSensor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", type=str, default="an-an-force.yml")
    parser.add_argument(
        "--controller-cfg", type=str, default="osc-position-controller.yml"
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    data_dir = Path("./data")
    # Create dataset folders
    dataset_folder = datetime.datetime.now().strftime(
        "{}/dataset_%m-%d-%Y_%H-%M-%S".format(data_dir.absolute())
    )
    os.mkdir(dataset_folder)

    device = SpaceMouse(vendor_id=9583, product_id=50741)
    device.start_control()

    robot_interface = FrankaInterface(config_root + f"/{args.interface_cfg}")

    # Initialize camera interfaces. The example uses two cameras. You
    # need to specify camera id in camera_node script from rpl_vision_utils

    # rs_cap = RealsenseCapture()

    controller_cfg = YamlConfig(config_root + f"/{args.controller_cfg}").as_easydict()
    controller_type = "OSC_POSITION"

    # initialize data
    data = {
        "action": [],
        "proprio_ee": [],
        "proprio_joints": [],
        "proprio_gripper_state": [],
    }

    i = 0
    start = False
    while i < 2000:
        i += 1
        start_time = time.time_ns()
        action, grasp = input2action(
            device=device,
            controller_type=controller_type,
        )
        if action is None:
            break
        robot_interface.control(
            controller_type=controller_type,
            action=action,
            controller_cfg=controller_cfg,
        )

        if len(robot_interface._state_buffer) == 0:
            continue

        last_state = robot_interface._state_buffer[-1]
        last_gripper_state = robot_interface._gripper_state_buffer[-1]
        if np.linalg.norm(action[:-1]) < 1e-3 and not start:
            continue

        start = True
        print(action)
        # Record ee pose,  joints

        last_state.O_T_EE

        data["action"].append(action)
        data["proprio_ee"].append(np.array(last_state.O_T_EE))
        data["proprio_joints"].append(np.array(last_state.q))

        end_time = time.time_ns()
        print(f"Time profile: {(end_time - start_time) / 10 ** 9}")

    np.savez(f"{folder}/action", data=np.array(data["action"]))
    np.savez(f"{folder}/proprio_ee", data=np.array(data["proprio_ee"]))
    np.savez(f"{folder}/proprio_joints", data=np.array(data["proprio_joints"]))

    robot_interface.close()

    save = input("Save or not? (enter 0 or 1)")
    save = bool(int(save))

    if not save:
        import shutil

        shutil.rmtree(f"{folder}")


if __name__ == "__main__":
    main()
