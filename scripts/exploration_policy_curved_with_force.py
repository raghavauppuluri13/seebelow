import time
from collections import deque
import click
from pathlib import Path
import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np
import open3d as o3d
import pinocchio as pin
from scipy.spatial.transform import Rotation

from deoxys.franka_interface import FrankaInterface
from deoxys.utils.transform_utils import quat2axisangle, quat2mat
from deoxys.utils import YamlConfig
from seebelow.algorithms.grid import SurfaceGridMap
from seebelow.algorithms.search import (
    RandomSearch,
    SearchHistory,
    Search,
    ActiveSearchWithRandomInit,
    ActiveSearchAlgos,
)
from seebelow.utils.control_utils import generate_joint_space_min_jerk
from seebelow.utils.data_utils import DatasetWriter
from seebelow.utils.devices import ForceSensor
from seebelow.utils.interpolator import Interpolator, InterpType
from seebelow.utils.time_utils import Ratekeeper
from seebelow.utils.proc_utils import RingBuffer, RunningStats
import seebelow.utils.constants as seebelow_const
from seebelow.utils.constants import PALP_CONST
from seebelow.utils.constants import PalpateState
from seebelow.utils.pcd_utils import scan2mesh, mesh2roi
from tfvis.visualizer import RealtimeVisualizer


def main_ctrl(shm_buffer, stop_event: mp.Event, save_folder: Path, search: Search):
    existing_shm = shared_memory.SharedMemory(name=shm_buffer)
    data_buffer = np.ndarray(1, dtype=seebelow_const.PALP_DTYPE, buffer=existing_shm.buf)
    np.random.seed(PALP_CONST.seed)
    goals = deque([])
    force_buffer = RingBuffer(PALP_CONST.buffer_size)
    pos_buffer = RingBuffer(PALP_CONST.buffer_size)
    running_stats = RunningStats()

    palp_state = PalpateState()
    curr_pose_se3 = pin.SE3.Identity()
    using_force_control_flag = False
    collect_points_flag = False
    curr_eef_pose = None
    F_norm = 0.0
    Fxyz = np.zeros(3)
    palp_pt = None
    surf_normal = None
    palp_progress = 0.0  # [0, 1]
    palp_id = 0
    stiffness = 0.0
    search_history = SearchHistory()
    CF_start_time = None
    oscill_start_time = None
    start_angles = np.full(2, -PALP_CONST.angle_oscill)  # theta, phi
    end_angles = np.full(2, PALP_CONST.angle_oscill)
    max_cf_time = 0.1 if PALP_CONST.discrete_only else PALP_CONST.max_cf_time
    force_oscill_out = generate_joint_space_min_jerk(
        start_angles,
        end_angles,
        PALP_CONST.t_oscill / 2,
        1 / PALP_CONST.ctrl_freq,
    )
    force_oscill_in = generate_joint_space_min_jerk(
        end_angles,
        start_angles,
        PALP_CONST.t_oscill / 2,
        1 / PALP_CONST.ctrl_freq,
    )
    force_oscill_traj = force_oscill_out + force_oscill_in
    force_cap = ForceSensor()
    robot_interface = FrankaInterface(
        str(seebelow_const.PAN_PAN_FORCE_CFG),
        use_visualizer=False,
        control_freq=80,
    )
    force_ctrl_cfg = YamlConfig(str(seebelow_const.FORCE_CTRL_CFG)).as_easydict()
    osc_abs_ctrl_cfg = YamlConfig(str(seebelow_const.OSC_ABSOLUTE_CFG)).as_easydict()
    robot_interface._state_buffer = []
    interp = Interpolator(interp_type=InterpType.SE3)

    def palpate(pos, O_surf_norm_unit=np.array([0, 0, 1])):
        assert np.isclose(np.linalg.norm(O_surf_norm_unit), 1)

        # Uses Kabasch algo get rotation that aligns the eef tip -z_axis and
        # the normal vector, and secondly ensure that the y_axis is aligned
        R = Rotation.align_vectors(
            np.array([O_surf_norm_unit, np.array([0, -1, 0])]),
            np.array([[0, 0, -1], np.array([0, 1, 0])]),
            weights=np.array([10, 0.1]),
        )[0].as_matrix()

        palp_se3 = pin.SE3.Identity()
        palp_se3.translation = pos - PALP_CONST.palpate_depth * O_surf_norm_unit
        palp_se3.rotation = R

        above_se3 = pin.SE3.Identity()
        above_se3.translation = pos + PALP_CONST.above_height * O_surf_norm_unit
        above_se3.rotation = R

        reset_pose = pin.SE3.Identity()
        reset_pose.translation = seebelow_const.RESET_PALP_POSE[:3]
        reset_pose.rotation = quat2mat(seebelow_const.RESET_PALP_POSE[3:7])

        goals.appendleft(above_se3)
        goals.appendleft(palp_se3)
        goals.appendleft(reset_pose)

    def state_transition():
        palp_state.next()
        steps = seebelow_const.STEP_FAST
        if palp_state.state == PalpateState.PALPATE:
            steps = seebelow_const.STEP_SLOW
        pose_goal = goals.pop()
        interp.init(curr_pose_se3, pose_goal, steps=steps)

    while len(robot_interface._state_buffer) == 0:
        continue

    start_pose = pin.SE3.Identity()
    start_pose.translation = seebelow_const.GT_SCAN_POSE[:3]
    start_pose.rotation = quat2mat(seebelow_const.GT_SCAN_POSE[3:7])
    goals.appendleft(start_pose)

    curr_eef_pose = robot_interface.last_eef_rot_and_pos
    curr_pose_se3.rotation = curr_eef_pose[0]
    curr_pose_se3.translation = curr_eef_pose[1]
    pose_goal = goals.pop()
    interp.init(curr_pose_se3, pose_goal, steps=seebelow_const.STEP_FAST)
    try:
        while not stop_event.is_set():
            curr_eef_pose = robot_interface.last_eef_rot_and_pos
            curr_pose_se3.rotation = curr_eef_pose[0]
            curr_pose_se3.translation = curr_eef_pose[1]

            Fxyz_temp = force_cap.read()
            if Fxyz_temp is not None:
                Fxyz = Fxyz_temp
                F_norm = np.sqrt(np.sum(Fxyz**2))

            force_buffer.append(F_norm)
            if palp_pt is not None:
                pos_buffer.append(np.linalg.norm(palp_pt - curr_pose_se3.translation))

            if force_buffer.overflowed():
                running_stats.update(force_buffer.buffer)

            # terminate palpation and reset goals

            # done with palpation
            if (using_force_control_flag
                    and (palp_progress >= 1 or time.time() - CF_start_time > max_cf_time)
                    # and pos_buffer.std < PALP_CONST.pos_stable_thres
                ):
                print("palpation done")
                collect_points_flag = False
                using_force_control_flag = False
                state_transition()
                palp_id += 1
                if palp_id == PALP_CONST.max_palpations:
                    print("terminate")
                    palp_state.state = PalpateState.TERMINATE
                    goals.clear()
                    interp.init(curr_pose_se3, start_pose, steps=seebelow_const.STEP_FAST)

            # initiate palpate
            if len(goals) > 0 and interp.done:
                state_transition()
            elif len(goals) == 0 and interp.done:
                if palp_state.state == PalpateState.TERMINATE:
                    print("breaking")
                    break
                palp_pt, surf_normal = search.next()
                search_history.add(*search.grid_estimate)
                palpate(palp_pt, surf_normal)

            # start palpation
            stiffness = 0.0
            if palp_state.state == PalpateState.PALPATE:
                assert palp_pt is not None

                # update stiffness
                palp_disp = curr_pose_se3.translation - palp_pt

                # scalar projection of displacement along -surface normal normalized to max alotted displacement
                # between 0 and 1
                palp_progress = (np.dot(palp_disp, -surf_normal) / PALP_CONST.max_palp_disp)
                if (Fxyz[2] >= PALP_CONST.max_Fz
                        or palp_progress >= 1.0) and not using_force_control_flag:
                    CF_start_time = time.time()
                    print("CONTOUR FOLLOWING!")
                    stiffness = Fxyz[2] / (np.linalg.norm(curr_pose_se3.translation - palp_pt) +
                                           1e-6)
                    stiffness /= PALP_CONST.stiffness_normalization
                    using_force_control_flag = True
                    collect_points_flag = True
                    oscill_start_time = time.time()
                    print("STIFFNESS: ", stiffness)
                    search.update_outcome(stiffness)

            # control: force
            if using_force_control_flag:
                if time.time() - oscill_start_time > PALP_CONST.t_oscill:
                    oscill_start_time = time.time()
                idx = int((time.time() - oscill_start_time) / (1 / PALP_CONST.ctrl_freq))
                action = np.zeros(9)
                oscill_pos = force_oscill_traj[idx]["position"]
                action[0] = oscill_pos[0]
                action[1] = oscill_pos[1]
                action[2] = -0.005
                robot_interface.control(
                    controller_type=seebelow_const.FORCE_CTRL_TYPE,
                    action=action,
                    controller_cfg=force_ctrl_cfg,
                )

            # control: OSC
            else:
                action = np.zeros(7)
                next_se3_pose = interp.next()
                target_xyz_quat = pin.SE3ToXYZQUAT(next_se3_pose)
                axis_angle = quat2axisangle(target_xyz_quat[3:7])

                # print(se3_pose)
                action[:3] = target_xyz_quat[:3]
                action[3:6] = axis_angle
                # print(action)

                robot_interface.control(
                    controller_type=seebelow_const.OSC_CTRL_TYPE,
                    action=action,
                    controller_cfg=osc_abs_ctrl_cfg,
                )
            q, p = robot_interface.last_eef_quat_and_pos
            save_action = np.zeros(9)
            save_action[:len(action)] = action
            data_buffer[0] = (
                Fxyz,
                q.flatten(),  # quat
                p.flatten(),  # pos
                target_xyz_quat[3:7],
                target_xyz_quat[:3],
                save_action,
                palp_progress,
                palp_pt if palp_pt is not None else np.zeros(3),
                surf_normal if surf_normal is not None else np.zeros(3),
                palp_id,
                palp_state.state,
                stiffness,
                using_force_control_flag,
                collect_points_flag,
            )

    except KeyboardInterrupt:
        pass

    # stop
    robot_interface.close()
    search_history.save(save_folder)
    print("history saved")
    stop_event.set()
    existing_shm.close()


@click.command()
@click.option(
    "--tumor",
    "-t",
    type=str,
    help="tumor type [crescent,hemisphere]",
    default="hemisphere",
)
@click.option("--algo", "-a", type=str, help="algorithm [bo, random]", default="random")
@click.option("--select_bbox", "-b", type=bool, help="choose bounding box", default=False)
@click.option("--max_palpations", "-m", type=int, help="max palpations", default=60)
@click.option("--autosave", "-s", type=bool, help="autosave", default=False)
@click.option("--seed", "-e", type=int, help="seed", default=None)
@click.option("--debug", "-d", type=bool, help="runs visualizations", default=False)
@click.option("--discrete_only", "-s", type=bool, help="discrete probing only", default=False)
def main(tumor, algo, select_bbox, max_palpations, autosave, seed, debug, discrete_only):
    pcd = o3d.io.read_point_cloud(str(seebelow_const.SURFACE_SCAN_PATH))
    surface_mesh = scan2mesh(pcd)
    PALP_CONST.max_palpations = max_palpations
    PALP_CONST.algo = algo
    PALP_CONST.seed = np.random.randint(1000) if seed is None else seed
    PALP_CONST.tumor_type = tumor
    PALP_CONST.discrete_only = discrete_only

    if PALP_CONST.tumor_type == "hemisphere":
        seebelow_const.BBOX_DOCTOR_ROI = seebelow_const.ROI_HEMISPHERE
    elif PALP_CONST.tumor_type == "crescent":
        seebelow_const.BBOX_DOCTOR_ROI = seebelow_const.ROI_CRESCENT
    bbox_roi = seebelow_const.BBOX_DOCTOR_ROI
    if select_bbox:
        bbox_roi = None

    roi_pcd = mesh2roi(surface_mesh, bbox_pts=bbox_roi)
    print("here")
    surface_grid_map = SurfaceGridMap(roi_pcd, grid_size=seebelow_const.PALP_CONST.grid_size)
    if debug:
        surface_grid_map.visualize()

    if algo == "bo":
        search = ActiveSearchWithRandomInit(
            ActiveSearchAlgos.BO,
            surface_grid_map,
            kernel_scale=seebelow_const.PALP_CONST.kernel_scale,
            random_sample_count=seebelow_const.PALP_CONST.random_sample_count,
        )
    elif algo == "random":
        search = RandomSearch(surface_grid_map)
    # search.grid.visualize()
    dataset_writer = DatasetWriter(prefix=f"{tumor}_{algo}", print_hz=False)
    data_buffer = np.zeros(1, dtype=seebelow_const.PALP_DTYPE)
    shm = shared_memory.SharedMemory(create=True, size=data_buffer.nbytes)
    data_buffer = np.ndarray(data_buffer.shape, dtype=data_buffer.dtype, buffer=shm.buf)
    stop_event = mp.Event()
    ctrl_process = mp.Process(
        target=main_ctrl,
        args=(shm.name, stop_event, dataset_writer.dataset_folder, search),
    )
    ctrl_process.start()

    subsurface_pts = []

    rk = Ratekeeper(50, name="data_collect")

    rtv = RealtimeVisualizer()
    rtv.add_frame("BASE")
    rtv.set_frame_tf("BASE", np.eye(4))
    rtv.add_frame("EEF", "BASE")
    try:
        while not stop_event.is_set():
            if np.all(data_buffer["O_q_EE"] == 0):
                # print("Waiting for deoxys...")
                continue

            O_p_EE = data_buffer["O_p_EE"].flatten()
            O_p_EE_target = data_buffer["O_p_EE_target"].flatten()
            O_q_EE_target = data_buffer["O_q_EE_target"].flatten()
            O_q_EE = data_buffer["O_q_EE"].flatten()
            if data_buffer["collect_points_flag"]:
                subsurface_pts.append(O_p_EE)

            print(data_buffer)
            # print("Pos ERROR: ", np.linalg.norm(O_p_EE - O_p_EE_target))
            # print("Rot ERROR: ", np.linalg.norm(O_q_EE - O_q_EE_target))

            dataset_writer.add_sample(data_buffer.copy())
            ee_pos = np.array(O_p_EE)
            ee_rmat = quat2mat(O_q_EE)
            O_T_E = np.eye(4)
            O_T_E[:3, :3] = ee_rmat
            O_T_E[:3, 3] = ee_pos
            rtv.set_frame_tf("EEF", O_T_E)
            rk.keep_time()
    except KeyboardInterrupt:
        pass
    stop_event.set()
    print("CTRL STOPPED!")

    while ctrl_process.is_alive():
        continue
    ctrl_process.join()

    dataset_writer.save_subsurface_pcd(np.array(subsurface_pts).squeeze())
    dataset_writer.save_roi_pcd(roi_pcd)
    dataset_writer.save_grid_pcd(surface_grid_map.grid_pcd)
    dataset_writer.save(autosave)
    shm.close()
    shm.unlink()


if __name__ == "__main__":
    main()
