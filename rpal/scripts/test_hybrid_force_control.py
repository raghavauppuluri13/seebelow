import time

import argparse
from collections import deque

import numpy as np
from deoxys.franka_interface import FrankaInterface
from deoxys import config_root
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.utils.input_utils import input2action
from deoxys.utils import YamlConfig
from deoxys.utils.io_devices import SpaceMouse
from deoxys.utils.log_utils import get_deoxys_example_logger

from devices import ForceSensor

from utils import DatasetWriter, Hz, three_pts_to_rot_mat

from interpolator import Interpolator, InterpType

logger = get_deoxys_example_logger()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", type=str, default="./an-an-force.yml")
    parser.add_argument("--controller-type", type=str, default="OSC_POSE")
    parser.add_argument(
        "--controller-cfg", type=str, default="./osc-pose-controller.yml"
    )
    parser.add_argument("--vendor-id", type=int, default=9583)
    parser.add_argument("--product-id", type=int, default=50734)

    args = parser.parse_args()

    robot_interface = FrankaInterface(
        args.interface_cfg, use_visualizer=False, control_freq=100
    )

    controller_type = args.controller_type
    controller_type = "RPAL_HYBRID_POSITION_FORCE"
    # controller_cfg = get_default_controller_config(controller_type=controller_type)
    controller_cfg = YamlConfig(args.controller_cfg).as_easydict()

    robot_interface._state_buffer = []

    action = np.zeros(9)
    action[6] = 1

    try:
        while True:
            robot_interface.control(
                controller_type=controller_type,
                action=action,
                controller_cfg=controller_cfg,
            )
            # print(f"Time duration: {((end_time - start_time) / (10**9))}")
    except KeyboardInterrupt:
        pass

    # stop
    robot_interface.control(termination=True)

    robot_interface.close()
