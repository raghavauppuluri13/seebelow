import open3d as o3d
import yaml
from rpal.utils.transform_utils import *
from rpal.utils.pcd_utils import visualize_pcds
from rpal.utils.constants import *
import argparse

if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--calibration_folder", type=str)
    args.add_argument("--cam_name", type=str, default='wrist_d415')
    args = args.parse_args()

    calib_folder = RPAL_CFG_PATH.parent / 'temp_config' / Path(args.calibration_folder)

    with open(calib_folder / 'extrinsics.yaml', 'r') as f:
        calib_cfg = yaml.safe_load(f)
        print(calib_cfg)

    xyzrpy = calib_cfg[args.cam_name]
    print(xyzrpy)
    #xyzrpy = [0,-0.077,-0.109,0,0,3.1415]


    ee_pos = np.array(xyzrpy[:3])
    ee_rot = euler2mat(xyzrpy[-3:])
    C_T_E = np.eye(4)
    C_T_E[:3, :3] = ee_rot
    C_T_E[:3, 3] = ee_pos

    with open(str(calib_folder / 'final_ee_poses.txt') , 'r') as f:
        traj = []
        for line in f:
            ixyzrpy = [float(x) for x in line.split()]

            ee_pos = np.array(ixyzrpy[1:4])
            ee_rot = euler2mat(ixyzrpy[-3:])
            O_T_E = np.eye(4)
            O_T_E[:3, :3] = ee_rot
            O_T_E[:3, 3] = ee_pos
            traj.append(O_T_E @ np.linalg.inv(C_T_E))
    pcd = o3d.geometry.PointCloud()
    visualize_pcds([pcd], tfs=traj)
