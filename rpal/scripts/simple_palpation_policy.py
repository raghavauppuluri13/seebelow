import time

import argparse

import numpy as np
from deoxys.franka_interface import FrankaInterface
from deoxys import config_root
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.utils.input_utils import input2action
from deoxys.utils import YamlConfig
from deoxys.utils.io_devices import SpaceMouse
from deoxys.utils.log_utils import get_deoxys_example_logger

from utils import DatasetWriter

logger = get_deoxys_example_logger()

NO_TUMOR_J = [
    0.30956509,
    0.36256234,
    -0.15206966,
    -2.15620242,
    0.09976851,
    2.64834064,
    0.74043235,
]

TUMOR_J = [
    0.29606024,
    0.37608184,
    -0.18526146,
    -2.11430951,
    0.08906509,
    2.58114769,
    0.6631098,
]
RESET_VERTICAL = [
    0.11368487,
    -0.22414043,
    -0.02266427,
    -2.46654432,
    -0.04040972,
    2.31614642,
    0.97029041,
]


def reset(reset_joint_positions, robot_interface):

    controller_cfg = "joint-position-controller.yml"
    controller_type = "JOINT_POSITION"

    action = reset_joint_positions + [-1.0]

    controller_cfg = YamlConfig(config_root + f"/{controller_cfg}").as_easydict()

    while True:
        if len(robot_interface._state_buffer) > 0:
            if (
                np.max(
                    np.abs(
                        np.array(robot_interface._state_buffer[-1].q)
                        - np.array(reset_joint_positions)
                    )
                )
                < 1e-3
            ):
                break
        robot_interface.control(
            controller_type=controller_type,
            action=action,
            controller_cfg=controller_cfg,
        )


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", type=str, default="an-an-force.yml")
    parser.add_argument("--controller-type", type=str, default="OSC_POSE")
    parser.add_argument("--tumor", help="palpating tumor or no?", action="store_true")
    parser.add_argument("--vendor-id", type=int, default=9583)
    parser.add_argument("--product-id", type=int, default=50734)

    args = parser.parse_args()

    robot_interface = FrankaInterface(args.interface_cfg, use_visualizer=False)

    reset(RESET_VERTICAL, robot_interface)

    dataset_writer = DatasetWriter()

    reset(TUMOR_J if args.tumor else NO_TUMOR_J, robot_interface)

    controller_type = args.controller_type
    controller_cfg = get_default_controller_config(controller_type=controller_type)
    # controller_cfg = YamlConfig("config/osc-pose-controller.yml").as_easydict()

    robot_interface._state_buffer = []

    INIT_EEF = None
    CURR_EEF = None
    FORCE = None

    try:
        for i in range(10000):
            start_time = time.time_ns()

            action = np.zeros(7)
            if CURR_EEF is not None:
                d = CURR_EEF[2, 3]
                if d >= -0.013:
                    action[2] = -0.05

            robot_interface.control(
                controller_type=controller_type,
                action=action,
                controller_cfg=controller_cfg,
            )
            end_time = time.time_ns()
            print(f"Time duration: {((end_time - start_time) / (10**9))}")

            if len(robot_interface._state_buffer) == 0:
                continue
            if INIT_EEF is None:
                INIT_EEF = robot_interface.last_eef_pose
            else:
                CURR_EEF = robot_interface.last_eef_pose - INIT_EEF
                dataset_writer.update(CURR_EEF[:3, 3])

    except KeyboardInterrupt:
        pass

    dataset_writer.save()

    reset(TUMOR_J if args.tumor else NO_TUMOR_J, robot_interface)

    robot_interface.control(
        controller_type=controller_type,
        action=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [1.0],
        controller_cfg=controller_cfg,
        termination=True,
    )

    robot_interface.close()

    # Check if there is any state frame missing
    for (state, next_state) in zip(
        robot_interface._state_buffer[:-1], robot_interface._state_buffer[1:]
    ):
        if (next_state.frame - state.frame) > 1:
            print(state.frame, next_state.frame)


if __name__ == "__main__":
    main()
