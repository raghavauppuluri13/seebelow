import argparse
from datetime import datetime

import cv2
import numpy as np
import open3d as o3d

from seebelow.utils.constants import *
from seebelow.utils.devices import RealsenseCapture
from seebelow.utils.pcd_utils import pick_surface_bbox, visualize_pcds
from seebelow.utils.segmentation_utils import get_color_mask, get_hsv_threshold

rs = RealsenseCapture()

# HSV color thresholding

im, pcd = rs.read()
# im, pcd = rs.read(get_mask=lambda x: get_hsv_threshold(x))
# im, pcd = rs.read(get_mask=lambda x: get_color_mask(x, TUMOR_HSV_THRESHOLD))

visualize_pcds([pcd])

now_str = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
o3d.io.write_point_cloud(str(SEEBELOW_MESH_PATH / f"tumors_gt_{now_str}.ply"), pcd)
