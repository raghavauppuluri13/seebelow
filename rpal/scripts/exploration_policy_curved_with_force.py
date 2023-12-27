import argparse
import time
from collections import deque
from copy import deepcopy
from pathlib import Path

import numpy as np
import open3d as o3d
import pinocchio as pin
from pinocchio.rpy import matrixToRpy, rpyToMatrix
from scipy.spatial.transform import Rotation

from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.utils.input_utils import input2action
from deoxys.utils.io_devices import SpaceMouse
from deoxys.utils.log_utils import get_deoxys_example_logger
from deoxys.utils.transform_utils import quat2axisangle
from rpal.algorithms.search import (ActiveSearch, ActiveSearchAlgos,
                                    RandomSearch)
from rpal.utils.constants import (RPAL_PKG_PATH,
                                  SIMPLE_TEST_BBOX_PHANTOM_HEMISPHERE)
from rpal.utils.control_utils import generate_joint_space_min_jerk
from rpal.utils.data_utils import DatasetWriter, Hz
from rpal.utils.devices import ForceSensor
from rpal.utils.interpolator import Interpolator, InterpType
from rpal.utils.math_utils import (rot_about_orthogonal_axes,
                                   three_pts_to_rot_mat, unit)
from rpal.utils.pcd_utils import surface_mesh_to_pcd
from rpal.utils.proc_utils import RingBuffer, RunningStats
from rpal.utils.useful_poses import O_T_CAM, O_xaxis


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


logger = get_deoxys_example_logger()
SAMPLE_RATE = 30  # hz
PHANTOM_MESH_PATH = str(RPAL_PKG_PATH / "meshes" / "phantom_mesh.ply")
RPAL_HYBRID_POSITION_FORCE = "RPAL_HYBRID_POSITION_FORCE"
np.random.seed(100)
F_norm_LIMIT = 5.0
PALP_WRENCH_MAG = 4  # N
BUFFER_SIZE = 100
FORCE_BUFFER_STABILITY_THRESHOLD = 0.05  # N
POS_BUFFER_STABILITY_THRESHOLD = 1e-4  # m
ABOVE_HEIGHT = 0.02
PALPATE_DEPTH = 0.035

sample_time = time.perf_counter()
goals = deque([O_T_CAM])
palp_state = PalpateState()
curr_pose_se3 = pin.SE3.Identity()
subsurface_pts = []
subsurface_pt = None
max_dist = -np.inf
using_force_control = False
wrench_goal = None
start_data_collection = False
force_buffer = RingBuffer(BUFFER_SIZE, FORCE_BUFFER_STABILITY_THRESHOLD)
pos_buffer = RingBuffer(BUFFER_SIZE, POS_BUFFER_STABILITY_THRESHOLD)
running_stats = RunningStats()
init_eef_pose = None
curr_eef_pose = None
prev_pt = None
prev_norm = None
FORCE = None
panda_force = None
STEP_FAST = 100
STEP_SLOW = 2000
F_norm = 0.0
Fxyz = 0.0
max_stiffness = -np.inf
palp_pt = None
CTRL_FREQ = 100
start_time = None

oscill_start_time = None
T_oscill = 2  # oscillation period (s)
CTRL_FREQ = 100
ANGLE = 10  # deg
start_angles = np.zeros(2)  # theta, phi
end_angles = np.full(2, ANGLE)
force_oscill_out = generate_joint_space_min_jerk(
    start_angles, end_angles, T_oscill / 2, 1 / CTRL_FREQ
)
force_oscill_in = generate_joint_space_min_jerk(
    end_angles, start_angles, T_oscill / 2, 1 / CTRL_FREQ
)
force_oscill_traj = force_oscill_out + force_oscill_in

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

    controller_type = args.controller_type
    controller_cfg = YamlConfig(args.controller_cfg).as_easydict()

    force_cap = ForceSensor()
    robot_interface = FrankaInterface(args.interface_cfg, use_visualizer=False)
    robot_interface._state_buffer = []
    dataset_writer = DatasetWriter(args, record_pcd=False, print_hz=False)
    interp = Interpolator(interp_type=InterpType.SE3)

    surface_pcd_cropped = surface_mesh_to_pcd(PHANTOM_MESH_PATH)
    search = RandomSearch(surface_pcd_cropped)
    # search.grid.visualize()
    # search = ActiveSearch(phantom_pcd, ActiveSearchAlgos.BO)

    def palpate(pos, O_surf_norm_unit=np.array([0, 0, 1])):
        O_zaxis = np.array([[0, 0, 1]])

        assert np.isclose(np.linalg.norm(O_surf_norm_unit), 1)

        # Uses Kabasch algo get rotation that aligns the eef tip -z_axis and
        # the normal vector, and secondly ensure that the y_axis is aligned
        R = Rotation.align_vectors(
            np.array([O_surf_norm_unit, np.array([1, 0, 0])]),
            np.array([[0, 0, -1], O_xaxis]),
            weights=np.array([1, 0.5]),
        )[0].as_matrix()

        palp_se3 = pin.SE3.Identity()
        palp_se3.translation = pos - PALPATE_DEPTH * O_surf_norm_unit
        palp_se3.rotation = R

        above_se3 = pin.SE3.Identity()
        above_se3.translation = pos + ABOVE_HEIGHT * O_surf_norm_unit
        above_se3.rotation = R

        goals.appendleft(above_se3)
        goals.appendleft(palp_se3)
        goals.appendleft(O_T_CAM)

    def state_transition():
        palp_state.next()
        print(palp_state.state)
        steps = STEP_FAST
        if palp_state.state == PalpateState.PALPATE:
            steps = STEP_SLOW
        pose_goal = goals.pop()
        interp.init(curr_pose_se3, pose_goal, steps=steps)

    try:
        while len(robot_interface._state_buffer) == 0:
            continue
        curr_eef_pose = robot_interface.last_eef_rot_and_pos
        curr_pose_se3.rotation = curr_eef_pose[0]
        curr_pose_se3.translation = curr_eef_pose[1]
        pose_goal = goals.pop()
        interp.init(curr_pose_se3, pose_goal, steps=STEP_FAST)
        while True:
            curr_eef_pose = robot_interface.last_eef_rot_and_pos
            curr_pose_se3.rotation = curr_eef_pose[0]
            curr_pose_se3.translation = curr_eef_pose[1]

            """
            print(
                "rot error: ",
                np.linalg.norm(interp._goal.rotation - curr_pose_se3.rotation),
            )
            print(
                "translation error: ",
                np.linalg.norm(interp._goal.translation - curr_pose_se3.translation),
            )
            """

            Fxyz_temp = force_cap.read()
            if Fxyz_temp is not None:
                Fxyz = Fxyz_temp
                F_norm = np.sqrt(np.sum(Fxyz**2))

            force_buffer.append(F_norm)
            if palp_pt is not None:
                pos_buffer.append(np.linalg.norm(palp_pt - curr_pose_se3.translation))

            if force_buffer.overflowed():
                running_stats.update(force_buffer.buffer)

            if using_force_control and pos_buffer.is_stable:
                print("FORCE STABLE!")
                start_data_collection = False
                using_force_control = False
                max_stiffness = -np.inf
                state_transition()

            if len(goals) > 0 and interp.done:
                state_transition()

            elif len(goals) == 0 and interp.done:
                palp_pt, surf_normal = search.next()
                wrench_unit = -surf_normal
                palpate(palp_pt, surf_normal)

            if palp_state.state == PalpateState.PALPATE:
                assert palp_pt is not None
                stiffness = F_norm / np.linalg.norm(palp_pt - curr_pose_se3.translation)
                max_stiffness = max(stiffness, max_stiffness)
                # search.update_outcome(palp_pt, max_stiffness)

                if F_norm > F_norm_LIMIT and not using_force_control:
                    using_force_control = True
                    start_data_collection = True
                    oscill_start_time = time.time()
                    print("START DATA COLLECTION!")
            if start_data_collection:
                subsurface_pt = robot_interface.last_eef_rot_and_pos[1]
                subsurface_pts.append(subsurface_pt)

            curr_eef_quat_pos = robot_interface.last_eef_quat_and_pos
            dataset_writer.update(curr_eef_quat_pos, Fxyz)

            if using_force_control:
                if time.time() - oscill_start_time > T_oscill:
                    oscill_start_time = time.time()
                idx = int((time.time() - oscill_start_time) / (1 / CTRL_FREQ))

                print("Using force control!")
                print("force buffer std", np.std(force_buffer.buffer))
                print("pos buffer std", np.std(pos_buffer.buffer))
                action = np.zeros(9)
                assert wrench_unit is not None
                theta_phi = force_oscill_traj[idx]["position"]
                R = rot_about_orthogonal_axes(wrench_unit, theta_phi[0], theta_phi[1])
                # wrench_unit = R @ wrench_unit
                assert np.isclose(np.linalg.norm(wrench_unit), 1)
                action[-3:] = PALP_WRENCH_MAG * wrench_unit
                print("wrench: ", action[-3:])
                robot_interface.control(
                    controller_type=RPAL_HYBRID_POSITION_FORCE,
                    action=action,
                    controller_cfg=controller_cfg,
                )
            else:
                action = np.zeros(7)
                next_se3_pose = interp.next()
                xyz_quat = pin.SE3ToXYZQUAT(next_se3_pose)
                axis_angle = quat2axisangle(xyz_quat[3:7])

                # print(se3_pose)
                action[:3] = xyz_quat[:3]
                action[3:6] = axis_angle
                # print(action)

                robot_interface.control(
                    controller_type=controller_type,
                    action=action,
                    controller_cfg=controller_cfg,
                )
            end_time = time.perf_counter()
            # print(f"Time duration: {((end_time - start_time) / (10**9))}")
    except KeyboardInterrupt:
        pass

    dataset_writer.save_subsurface_pcd(np.array(subsurface_pts).squeeze())
    dataset_writer.save()

    # stop
    robot_interface.control(
        termination=True,
    )

    robot_interface.close()
