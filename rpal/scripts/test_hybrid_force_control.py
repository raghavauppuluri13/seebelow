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
from rpal.utils.math_utils import rot_about_orthogonal_axes
from rpal.utils.control_utils import generate_joint_space_min_jerk


logger = get_deoxys_example_logger()

total_time = 2
CTRL_FREQ = 100
mag = 4
ANGLE = 35

start = np.zeros(2)
end = np.full(2, ANGLE)
force_osc_out = generate_joint_space_min_jerk(start, end, total_time / 2, 1 / CTRL_FREQ)
force_osc_in = generate_joint_space_min_jerk(end, start, total_time / 2, 1 / CTRL_FREQ)
force_osc = force_osc_out + force_osc_in

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
        args.interface_cfg, use_visualizer=False, control_freq=CTRL_FREQ
    )

    controller_type = args.controller_type
    controller_type = "RPAL_HYBRID_POSITION_FORCE"
    # controller_cfg = get_default_controller_config(controller_type=controller_type)
    controller_cfg = YamlConfig(args.controller_cfg).as_easydict()

    robot_interface._state_buffer = []

    action = np.zeros(9)
    z_unit = np.array([0, 0, -1])

    start_time = time.time()

    try:
        while True:
            if time.time() - start_time > total_time:
                start_time = time.time()

            force_idx = int((time.time() - start_time) / (1 / CTRL_FREQ))
            theta_phi = force_osc[force_idx]["position"]
            R = rot_about_orthogonal_axes(z_unit, theta_phi[0], theta_phi[1])
            wrench = R @ z_unit
            assert np.isclose(np.linalg.norm(wrench), 1)
            print("wrench: ", wrench)
            action[-3:] = mag * wrench
            robot_interface.control(
                controller_type=controller_type,
                action=action,
                controller_cfg=controller_cfg,
            )
            # print(f"Time duration: {((end_time - start_time) / (10**9))}")
    except KeyboardInterrupt:
        pass

    # stop
    action = np.zeros(9)
    robot_interface.control(termination=True)

    robot_interface.close()
