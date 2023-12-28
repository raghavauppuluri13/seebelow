from pathlib import Path
import numpy as np
import rpal


def np_to_constant(name, arr):
    return f"{name} = np.array({arr.tolist()})"


# spacemouse vendor and product id
SPACEM_VENDOR_ID = 9583
SPACEM_PRODUCT_ID = 50741

# useful paths
RPAL_PKG_PATH = Path(rpal.__file__).parent.absolute()
RPAL_CFG_PATH = Path(rpal.__file__).parent.absolute() / "config"
RPAL_MESH_PATH = Path(rpal.__file__).parent.absolute() / "meshes"
RPAL_CKPT_PATH = Path(rpal.__file__).parent.absolute() / ".ckpts"

RPAL_HYBRID_POSITION_FORCE = "RPAL_HYBRID_POSITION_FORCE"
OSC_CTRL_TYPE = "OSC_POSE"
OSC_DELTA_CFG = "osc-pose-controller-delta.yml"
OSC_ABSOLUTE_CFG = "osc-pose-controller.yml"
JNT_POSITION_CFG = "joint-position-controller.yml"
BASE_CALIB_FOLDER = RPAL_CFG_PATH / "base_camera_calib"

PAN_PAN_FORCE_CFG = "pan-pan-force.yml"


# tumor color thresholding for ground truth collection
TUMOR_HSV_THRESHOLD = (np.array([80, 136, 3]), np.array([115, 255, 19]))

BBOX_PHANTOM = np.array(
    [
        [0.6142629185692412, -0.08319621761996245, -0.4822808014982284],
        [0.6144339613249947, -0.08269308024442612, 0.5220440469872084],
        [0.584357354416293, 0.1043882593073316, -0.482369682731668],
        [0.4547662923308071, -0.10862387090857632, -0.48224089972213735],
        [0.42503177093361244, 0.07946374339425405, 0.5219950675298598],
        [0.4248607281778589, 0.07896060601871772, -0.482329780955577],
        [0.45493733508656065, -0.10812073353304, 0.5220839487632994],
        [0.5845283971720465, 0.10489139668286793, 0.5219551657537688],
    ]
)

GT_SCAN_POSE = np.array(
    [
        0.5174325704574585,
        -0.0029695522971451283,
        0.25471308946609497,
        -0.929806649684906,
        0.36692020297050476,
        -0.025555845350027084,
        0.01326837856322527,
    ]
)


# interpolation
STEP_FAST = 100
STEP_SLOW = 2000
