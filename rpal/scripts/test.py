import numpy as np
import pinocchio as pin

from scipy.spatial.transform import Rotation

O_R_EE = np.array(
    [
        [0.663637, -0.74774567, 0.02150223],
        [-0.74609248, -0.66370329, -0.05332878],
        [0.05414747, 0.0193483, -0.99834548],
    ]
)

O_v = np.array([0, 0, 1])
print(O_v)
dR = Rotation.align_vectors(np.array([O_v]), np.array([[0, 0, -1]]))[0].as_matrix()

axis_angle = pin.rpy.matrixToRpy(dR)
print(axis_angle)
