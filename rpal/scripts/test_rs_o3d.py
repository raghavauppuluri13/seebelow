import open3d as o3d

import numpy as np
from rpal.utils.devices import RealsenseCapture
import cv2
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--rgb", type=bool, default=False)
args = argparser.parse_args()

print(args)

rs = RealsenseCapture()

im, pcd = rs.read()
if not args.rgb:
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

while 1:  # Set the number of frames to display the image
    im, new_pcd = rs.read()
    if args.rgb:
        cv2.imshow("Image", np.asarray(im))
        cv2.waitKey(1)  # Wait for 1 millisecond
    else:
        pcd.points = new_pcd.points
        pcd.colors = new_pcd.colors
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
vis.destroy_window()
