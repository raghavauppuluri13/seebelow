"""
transform syntax: TO_T_FROM
O: franka base frame
CAM: overhead camera frame
"""

import pinocchio as pin
import numpy as np


O_t_CAM = np.array([0.56746543, 0.12762998, 0.10405758])
O_rpy_CAM = np.array([3.13075706, -0.03085785, -0.27018787])
O_R_CAM = pin.rpy.rpyToMatrix(O_rpy_CAM)

O_T_CAM = pin.SE3.Identity()
O_T_CAM.translation = O_t_CAM
O_T_CAM.rotation = O_R_CAM
O_xaxis = O_R_CAM @ np.array([1, 0, 0])
