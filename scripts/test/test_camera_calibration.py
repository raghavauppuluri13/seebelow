import argparse
import numpy as np
import yaml
from tfvis.visualizer import PlaybackVisualizer

from deoxys.utils.transform_utils import quat2mat
from seebelow.utils.time_utils import Ratekeeper
from seebelow.utils.transform_utils import euler2mat
import seebelow.utils.constants as seebelow_const


def read_ee_poses(ee_poses_fn):
    ee_poses = []
    with open(ee_poses_fn, "r") as f:
        for line in f:
            ixyzxyzw = [float(a) for a in line.split()]
            xyzxyzw = np.array((ixyzxyzw[1:]))
            T = np.eye(4)
            T[:3, :3] = quat2mat(xyzxyzw[-4:])
            T[:3, 3] = xyzxyzw[:3]
            ee_poses.append(T)
    return ee_poses


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--interface-cfg", type=str, default="pan-pan-force.yml")
    argparser.add_argument(
        "--calibration-cfg", type=str, default="camera_calibration_12-16-2023_14-48-12"
    )
    argparser.add_argument("--cam-name", type=str, default="wrist_d415")
    args = argparser.parse_args()

    with open(
        str(seebelow_const.SEEBELOW_CFG_PATH / args.calibration_cfg / "extrinsics.yaml"), "r"
    ) as file:
        calib_cfg = yaml.safe_load(file)
        xyzwxyz = calib_cfg[args.cam_name]
        xyzwxyz = np.array(xyzwxyz)
        E_T_C = np.eye(4)
        E_T_C[:3, :3] = quat2mat(xyzwxyz[-4:][[1, 2, 3, 0]])
        E_T_C[:3, 3] = xyzwxyz[:3]

    ee_poses = read_ee_poses(
        str(seebelow_const.SEEBELOW_CFG_PATH / args.calibration_cfg / "final_ee_poses.txt")
    )

    pbv = PlaybackVisualizer()
    pbv.add_frame("BASE")
    pbv.set_frame_tf("BASE", np.eye(4))
    pbv.add_frame("EEF", "BASE")
    pbv.add_frame("EEF_45", "EEF")
    rot_45 = np.eye(4)
    rot_45[:3, :3] = euler2mat([0, 0, -np.pi / 4])
    pbv.set_frame_tf("EEF_45", rot_45)
    pbv.add_frame("CAM", "EEF")

    pbv.set_frame_tf("CAM", E_T_C)

    for pose in ee_poses:
        pbv.set_frame_tf("EEF", pose)
    print("Done")
    pbv.push()
    pbv.keep_alive()
