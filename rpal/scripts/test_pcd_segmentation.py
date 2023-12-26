import open3d as o3d
import numpy as np
from datetime import datetime
from rpal.utils.devices import RealsenseCapture
from rpal.utils.constants import *
from rpal.utils.segmentation_utils import get_hsv_threshold, get_color_mask
from rpal.utils.pcd_utils import visualize_pcds
import cv2
import argparse

rs = RealsenseCapture()

# HSV color thresholding
im, pcd = rs.read(get_mask=lambda x: get_color_mask(x, TUMOR_HSV_THRESHOLD))
# im, pcd = rs.read()

visualize_pcds([pcd])

now_str = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
o3d.io.write_point_cloud(str(RPAL_MESH_PATH / f"tumors_gt_{now_str}.ply"), pcd)
