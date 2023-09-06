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
SAMPLE_RATE = 30  # hz
GRID_DIMS = np.array([0.01, 0.04])

np.random.seed(200)

O_p_PH_0 = np.array([0.63013406, 0.01578391, 0.06457141])
O_p_PH_1 = np.array([0.56891733, 0.1352639, 0.05884567])
O_p_PH_2 = np.array([0.58591638, 0.14173307, 0.06000158])

O_R_PH = three_pts_to_rot_mat(O_p_PH_0, O_p_PH_1, O_p_PH_2)
O_T_PH = np.eye(4)
O_T_PH[:3, :3] = O_R_PH
O_T_PH[:3, 3] = O_p_PH_0


class PalpateState:
    ABOVE = 0
    PALPATE = 1
    RETURN = 2
    INIT = -1

    def __init__(self):
        self.state = self.INIT

    def next(self):
        if self.state == self.INIT:
            self.state = self.ABOVE
        else:
            self.state = (self.state + 1) % 3


def uniform_sampling_1d(min_pose, max_pose):
    return np.random.uniform(min_pose, max_pose)


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

    force_cap = ForceSensor()

    robot_interface = FrankaInterface(
        args.interface_cfg, use_visualizer=False, control_freq=100
    )

    dataset_writer = DatasetWriter(args, record_pcd=False, print_hz=False)

    interp = Interpolator()

    controller_type = args.controller_type
    # controller_cfg = get_default_controller_config(controller_type=controller_type)
    controller_cfg = YamlConfig(args.controller_cfg).as_easydict()

    robot_interface._state_buffer = []

    init_eef_pose = None
    curr_eef_pose = None
    FORCE = None
    panda_force = None
    Frms = 0.0
    Fxyz = 0.0

    sample_time = time.perf_counter()

    unit_v = np.zeros(3)
    unit_v[2] = 1

    p1 = np.array([0.59738917, 0.10919975, 0.06216035])
    p2 = np.array([0.61790621, 0.05147912, 0.06164065])

    v = p2 - p1

    pts = deque([p1 + t * v for t in np.arange(0, 1, 0.05)])

    p_base_phantom_linear_lower = np.array([0.62841146, 0.05381222, 0.06318731])
    p_base_phantom_linear_upper = np.array([0.59710796, 0.11066157, 0.06230661])

    p_base_overhead = np.array([0.60534092, 0.07798562, 0.1423677])
    p_base_phantom = np.array([0.60534092, 0.07798562, 0.06623677])

    goals = deque([p_base_overhead])

    def palpate(pose, unit_normal=np.array([0, 0, 1])):
        ABOVE_HEIGHT = 0.005
        PALPATE_DEPTH = 0.02
        goals.appendleft(pose + ABOVE_HEIGHT * unit_normal)
        goals.appendleft(pose - PALPATE_DEPTH * unit_normal)
        goals.appendleft(pose + ABOVE_HEIGHT * unit_normal)

    palp_state = PalpateState()

    try:
        while len(robot_interface._state_buffer) == 0:
            continue
        curr_eef_pose = robot_interface.last_eef_quat_and_pos
        interp.init(curr_eef_pose[1].flatten(), goals.pop(), steps=200)
        while True:
            curr_eef_pose = robot_interface.last_eef_quat_and_pos
            Fxyz_temp = force_cap.read()
            if Fxyz_temp is not None:
                Fxyz = Fxyz_temp
                Frms = np.sqrt(np.sum(Fxyz**2))

            if Frms > 8.0:
                print("FORCE EXCEEDED!: ", Frms)
                print("CURRENT_STATE: ", palp_state.state)
            if palp_state.state == PalpateState.PALPATE and Frms > 8.0:
                palp_state.next()
                interp.init(curr_eef_pose[1].flatten(), goals.pop(), steps=200)

            if len(goals) > 0 and interp.done:
                palp_state.next()
                print(palp_state.state)

                steps = 200
                if palp_state.state == PalpateState.PALPATE:
                    steps = 2000
                interp.init(curr_eef_pose[1].flatten(), goals.pop(), steps=steps)
            elif len(goals) == 0 and interp.done:
                # next_pose = uniform_sampling_1d(
                #    p_base_phantom_linear_lower, p_base_phantom_linear_upper
                # )
                if len(pts) == 0:
                    break
                next_pose = pts.popleft()
                palpate(next_pose)

            if palp_state.state == PalpateState.PALPATE:
                # print("PALPATING!")
                dataset_writer.update(curr_eef_pose, Fxyz)

            action = np.zeros(7)
            action[:3] = interp.next()
            action[3] = np.pi

            robot_interface.control(
                controller_type=controller_type,
                action=action,
                controller_cfg=controller_cfg,
            )
            end_time = time.perf_counter()
            # print(f"Time duration: {((end_time - start_time) / (10**9))}")
    except KeyboardInterrupt:
        pass

    dataset_writer.save()

    # stop
    robot_interface.control(
        controller_type=controller_type,
        action=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [1.0],
        controller_cfg=controller_cfg,
        termination=True,
    )

    robot_interface.close()
