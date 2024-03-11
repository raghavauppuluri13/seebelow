import argparse

import cv2
import numpy as np
import open3d as o3d

from rpal.utils.constants import *
from rpal.utils.devices import RealsenseCapture
from rpal.utils.segmentation_utils import get_color_mask, get_hsv_threshold

argparser = argparse.ArgumentParser()
argparser.add_argument("--rgb", type=bool, default=False)
argparser.add_argument("--tumors-only", type=bool, default=False)
args = argparser.parse_args()

rs = RealsenseCapture()

im, pcd = rs.read(get_mask=lambda x: get_color_mask(x, TUMOR_HSV_THRESHOLD))

if not args.rgb:
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

while 1:  # Set the number of frames to display the image
    if args.tumors_only:
        im, new_pcd = rs.read(get_mask=lambda x: get_color_mask(x, TUMOR_HSV_THRESHOLD))
    else:
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
