import numpy as np
from pathlib import Path
import rpal

SPACEM_VENDOR_ID = 9583
SPACEM_PRODUCT_ID = 50741
RPAL_PKG_PATH = Path(rpal.__file__).parent.absolute()
RPAL_CFG_PATH = Path(rpal.__file__).parent.absolute() / "config"
RPAL_MESH_PATH = Path(rpal.__file__).parent.absolute() / "meshes"

OSC_CTRL_TYPE = "OSC_POSE"
OSC_DELTA_CFG = "osc-pose-controller-delta.yml"
OSC_ABSOLUTE_CFG = "osc-pose-controller.yml"
BASE_CALIB_FOLDER = RPAL_CFG_PATH / "base_camera_calib"

SIMPLE_TEST_BBOX_PHANTOM_HEMISPHERE = np.array(
    [
        [0.56058667, 0.11291592, 0.56610973],
        [0.57072537, 0.09391879, 0.56419049],
        [0.58044308, 0.12096349, 0.56508788],
        [0.59013209, 0.09949125, 0.5632396],
        [0.56058667, 0.11291592, -0.43389027],
        [0.57072537, 0.09391879, -0.43580951],
        [0.58044308, 0.12096349, -0.43491212],
        [0.59013209, 0.09949125, -0.4367604],
    ]
)
