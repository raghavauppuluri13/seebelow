import time

from copy import deepcopy

import argparse
from collections import deque
import open3d as o3d

import numpy as np
from deoxys.franka_interface import FrankaInterface
from deoxys import config_root
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.utils.input_utils import input2action
from deoxys.utils.transform_utils import quat2axisangle
from deoxys.utils import YamlConfig
from deoxys.utils.io_devices import SpaceMouse
from deoxys.utils.log_utils import get_deoxys_example_logger

from scipy.spatial.transform import Rotation

from devices import ForceSensor

from utils import DatasetWriter, Hz, three_pts_to_rot_mat

from pinocchio.rpy import matrixToRpy, rpyToMatrix

import pinocchio as pin

from interpolator import Interpolator, InterpType

logger = get_deoxys_example_logger()
SAMPLE_RATE = 30  # hz
RPAL_HYBRID_POSITION_FORCE = "RPAL_HYBRID_POSITION_FORCE"
GRID_DIMS = np.array([0.01, 0.04])

np.random.seed(100)

O_p_PH_0 = np.array([0.63013406, 0.01578391, 0.06457141])
O_p_PH_1 = np.array([0.56891733, 0.1352639, 0.05884567])
O_p_PH_2 = np.array([0.58591638, 0.14173307, 0.06000158])

O_R_PH = three_pts_to_rot_mat(O_p_PH_0, O_p_PH_1, O_p_PH_2, neg_x=True)
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

    robot_interface = FrankaInterface(args.interface_cfg, use_visualizer=False)

    dataset_writer = DatasetWriter(args, record_pcd=False, print_hz=False)

    interp = Interpolator(interp_type=InterpType.SE3)

    controller_type = args.controller_type
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

    O_t_overhead = np.array([0.56746543, 0.12762998, 0.10405758])
    O_rpy = np.array([3.13075706, -0.03085785, -0.27018787])
    O_R_overhead = pin.rpy.rpyToMatrix(O_rpy)
    O_T_overhead = pin.SE3.Identity()
    O_T_overhead.translation = O_t_overhead
    O_T_overhead.rotation = O_R_overhead

    O_v_x_axis = O_R_overhead @ np.array([1, 0, 0])

    phantom_mesh = o3d.io.read_triangle_mesh("./out_mesh.ply")
    phantom_mesh = phantom_mesh.subdivide_midpoint(number_of_iterations=2)
    phantom_mesh.compute_vertex_normals()
    phantom_mesh.remove_degenerate_triangles()

    phantom_pcd = o3d.geometry.PointCloud()
    phantom_pcd.points = o3d.utility.Vector3dVector(np.asarray(phantom_mesh.vertices))

    print("")
    print(
        "1) Please pick 4 point as the corners your bounding box [shift + left click]."
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(phantom_pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    points_idx = vis.get_picked_points()
    pcd_npy = np.asarray(phantom_pcd.points)
    bbox_pts = np.zeros((8, 3))
    pts = pcd_npy[points_idx]

    pts[:, -1] += 0.5
    bbox_pts[:4] = pts
    pts[:, -1] -= 1
    bbox_pts[4:8] = pts

    bbox = o3d.geometry.OrientedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(bbox_pts)
    )

    phantom_pcd.estimate_normals()
    phantom_pcd.normalize_normals()
    phantom_pcd.orient_normals_consistent_tangent_plane(k=100)
    phantom_pcd = phantom_pcd.crop(bbox)

    N = 100  # Number of samples
    A = len(np.asarray(phantom_pcd.points))
    random_sample = np.random.randint(0, A, N)

    phantom_vn = np.asarray(phantom_mesh.vertex_normals)
    phantom_v = np.asarray(phantom_mesh.vertices)

    pts = deque([])
    for i in random_sample:
        vn_p = phantom_pcd.normals[i]
        v_p = phantom_pcd.points[i]
        pts.append((v_p, vn_p))

    goals = deque([O_T_overhead])

    def palpate(pos, O_v_surf_norm_unit=np.array([0, 0, 1])):
        ABOVE_HEIGHT = 0.03
        PALPATE_DEPTH = 0.035
        O_z_axis = np.array([[0, 0, 1]])

        assert np.isclose(np.linalg.norm(O_v_surf_norm_unit), 1)

        # Uses Kabasch algo get rotation that aligns the eef tip z-axis and
        # the normal vector, ensure that the y-axis is aligned
        R = Rotation.align_vectors(
            np.array([O_v_surf_norm_unit, np.array([1, 0, 0])]),
            np.array([[0, 0, -1], O_v_x_axis]),
            weights=np.array([1, 0.5]),
        )[0].as_matrix()

        palp_se3 = pin.SE3.Identity()
        palp_se3.translation = pos - PALPATE_DEPTH * O_v_surf_norm_unit
        palp_se3.rotation = R

        above_se3 = pin.SE3.Identity()
        above_se3.translation = pos + ABOVE_HEIGHT * O_v_surf_norm_unit
        above_se3.rotation = R

        goals.appendleft(above_se3)
        goals.appendleft(palp_se3)
        goals.appendleft(O_T_overhead)

    palp_state = PalpateState()
    curr_pose_se3 = pin.SE3.Identity()
    subsurface_pts = []
    subsurface_pt = None
    max_dist = -np.inf
    using_force_control = False

    Frms_LIMIT = 8.0

    try:
        while len(robot_interface._state_buffer) == 0:
            continue
        curr_eef_pose = robot_interface.last_eef_rot_and_pos
        curr_pose_se3.rotation = curr_eef_pose[0]
        curr_pose_se3.translation = curr_eef_pose[1]
        pose_goal = goals.pop()
        interp.init(curr_pose_se3, pose_goal, steps=100)
        while True:
            curr_eef_pose = robot_interface.last_eef_rot_and_pos
            curr_pose_se3.rotation = curr_eef_pose[0]
            curr_pose_se3.translation = curr_eef_pose[1]

            print(
                "rot error: ",
                np.linalg.norm(interp._goal.rotation - curr_pose_se3.rotation),
            )
            print(
                "translation error: ",
                np.linalg.norm(interp._goal.translation - curr_pose_se3.translation),
            )

            Fxyz_temp = force_cap.read()
            if Fxyz_temp is not None:
                Fxyz = Fxyz_temp
                Frms = np.sqrt(np.sum(Fxyz**2))

            if palp_state.state == PalpateState.PALPATE and Frms > Frms_LIMIT:
                using_force_control = True

            if len(goals) > 0 and interp.done:
                palp_state.next()
                print(palp_state.state)

                steps = 100
                if palp_state.state == PalpateState.PALPATE:
                    steps = 2000
                pose_goal = goals.pop()
                interp.init(curr_pose_se3, pose_goal, steps=steps)
            elif len(goals) == 0 and interp.done:
                # next_pose = uniform_sampling_1d(
                #    p_base_phantom_linear_lower, p_base_phantom_linear_upper
                # )
                if len(pts) == 0:
                    break
                palp_pos, surf_normal = pts.popleft()
                palpate(palp_pos, surf_normal)

            if palp_state.state == PalpateState.PALPATE:
                dist = np.linalg.norm(
                    pose_goal.translation - robot_interface.last_eef_rot_and_pos[1]
                )
                if dist > max_dist:
                    max_dist = max(dist, max_dist)
                    subsurface_pt = robot_interface.last_eef_rot_and_pos[1]
                print("PALPATING!")
            else:
                if not np.allclose(max_dist, -np.inf):
                    subsurface_pts.append(subsurface_pt)
                    max_dist = -np.inf

            curr_eef_quat_pos = robot_interface.last_eef_quat_and_pos
            dataset_writer.update(curr_eef_quat_pos, Fxyz)

            if using_force_control:
                action = np.zeros(9)
                action[6] = 1
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
        controller_type=controller_type,
        action=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [1.0],
        controller_cfg=controller_cfg,
        termination=True,
    )

    robot_interface.close()
