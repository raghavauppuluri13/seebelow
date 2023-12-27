import argparse

import numpy as np
import pandas as pd
import quaternion as quat


def read_transformations(csv_file):
    # Read the CSV file
    data = pd.read_csv(csv_file, skiprows=7, header=None)

    # Extract frame1 and frame2
    f1_data = data.iloc[:, 2:9]
    f2_data = data.iloc[:, 9:16]

    # Compute the average transformations
    f1_data = f1_data.mean()
    f2_data = f2_data.mean()

    return f1_data.to_numpy(), f2_data.to_numpy()


def compute_T_F1_F2(pose_W_F1, pose_W_F2):
    print(pose_W_F1[:4])
    q_W_F1 = np.quaternion(*pose_W_F1[:4])
    q_W_F2 = np.quaternion(*pose_W_F2[:4])
    T_pose_W_F1 = np.eye(4)
    T_pose_W_F2 = np.eye(4)
    T_pose_W_F1[:3, 3] = pose_W_F1[-3:]
    T_pose_W_F2[:3, 3] = pose_W_F2[-3:]
    T_pose_W_F1[:3, :3] = quat.as_rotation_matrix(q_W_F1)
    T_pose_W_F2[:3, :3] = quat.as_rotation_matrix(q_W_F2)

    T_pose_F2_F1 = np.matmul(np.linalg.inv(T_pose_W_F2), T_pose_W_F1)

    return T_pose_F2_F1[:3, 3], quat.from_rotation_matrix(T_pose_F2_F1[:3, :3])


def main():
    parser = argparse.ArgumentParser(
        description="Compute the transformation of the board frame expressed in the UR5 frame."
    )
    parser.add_argument(
        "csv_file",
        help="Path to the CSV file containing the calibration transformations.",
    )
    args = parser.parse_args()

    # Read the transformations from the CSV file
    pose_W_F1, pose_W_F2 = read_transformations(args.csv_file)

    # Compute the final transformation
    pos, quat = compute_T_F1_F2(pose_W_F1, pose_W_F2)

    print("Position (XYZ):", pos)
    print("Rotation (Quaternion):", quat)


if __name__ == "__main__":
    main()
