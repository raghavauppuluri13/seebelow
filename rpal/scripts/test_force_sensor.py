from devices import ForceSensor
import time

import argparse
import time

import numpy as np
from deoxys.franka_interface import FrankaInterface
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.utils.input_utils import input2action
from deoxys.utils.io_devices import SpaceMouse
from deoxys.utils.log_utils import get_deoxys_example_logger

logger = get_deoxys_example_logger()


def main():

    f = ForceSensor()

    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", type=str, default="an-an-force.yml")
    parser.add_argument("--controller-type", type=str, default="OSC_POSE")

    parser.add_argument("--vendor-id", type=int, default=9583)
    parser.add_argument("--product-id", type=int, default=50734)

    args = parser.parse_args()

    robot_interface = FrankaInterface(args.interface_cfg, use_visualizer=False)

    controller_type = args.controller_type
    controller_cfg = get_default_controller_config(controller_type=controller_type)
    # controller_cfg = YamlConfig("config/osc-pose-controller.yml").as_easydict()

    robot_interface._state_buffer = []

    INIT_EEF = None
    CURR_EEF = None
    FORCE = None

    save_buffer = np.zeros((10000, 4))

    try:
        for i in range(10000):
            Fxyz = f.read()
            if Fxyz is not None:
                FORCE = Fxyz
            start_time = time.time_ns()

            action = np.zeros(7)
            if CURR_EEF is not None:
                d = CURR_EEF[2, 3]
                if d >= -0.01:
                    action[2] = -0.05
                print(d)
                print(FORCE)
                save_buffer[i, 0] = d
                save_buffer[i, 1:] = FORCE

            robot_interface.control(
                controller_type=controller_type,
                action=action,
                controller_cfg=controller_cfg,
            )
            end_time = time.time_ns()
            logger.debug(f"Time duration: {((end_time - start_time) / (10**9))}")

            if len(robot_interface._state_buffer) == 0:
                continue
            if INIT_EEF is None:
                INIT_EEF = robot_interface.last_eef_pose
            else:
                CURR_EEF = robot_interface.last_eef_pose - INIT_EEF
    except KeyboardInterrupt:
        pass

    np.savetxt("data.txt", save_buffer, fmt="%1.4f")
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
