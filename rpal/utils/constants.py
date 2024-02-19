from pathlib import Path
import numpy as np
import rpal


def array2constant(var_name: str, arr: np.ndarray) -> str:
    return f"{var_name.upper()} = np.array({arr.tolist()})"


# spacemouse vendor and product id
SPACEM_VENDOR_ID = 9583
SPACEM_PRODUCT_ID = 50741

# useful paths
RPAL_PKG_PATH = Path(rpal.__file__).parent.absolute()
RPAL_CFG_PATH = Path(rpal.__file__).parent.absolute() / "config"
RPAL_MESH_PATH = Path(rpal.__file__).parent.absolute() / "meshes"
RPAL_DATA_PATH = Path(rpal.__file__).parent.absolute() / "data"
RPAL_CKPT_PATH = Path(rpal.__file__).parent.absolute() / ".ckpts"

# camera calibration

CAMERA_CALIB_FOLDER = RPAL_CFG_PATH / "camera_calibration_12-16-2023_14-48-12"

# deoxys controllers
OSC_CTRL_TYPE = "OSC_POSE"
FORCE_CTRL_TYPE = "RPAL_HYBRID_POSITION_FORCE"
OSC_DELTA_CFG = RPAL_CFG_PATH / "osc-pose-controller-delta.yml"
OSC_ABSOLUTE_CFG = RPAL_CFG_PATH / "osc-pose-controller.yml"
JNT_POSITION_CFG = RPAL_CFG_PATH / "joint-position-controller.yml"
BASE_CALIB_FOLDER = RPAL_CFG_PATH / "base_camera_calib"
PAN_PAN_FORCE_CFG = RPAL_CFG_PATH / "pan-pan-force.yml"

# surface scan
SURFACE_SCAN_PATH = RPAL_PKG_PATH / "meshes" / "tumors_gt_01-05-2024_12-48-52.ply"
GT_PATH = RPAL_PKG_PATH / "meshes" / "tumors_gt_12-27-2023_19-22-31.ply"


# tumor color thresholding for ground truth collection
TUMOR_HSV_THRESHOLD = (np.array([80, 136, 3]), np.array([115, 255, 19]))
BBOX_PHANTOM = np.array(
    [
        [0.501941544576342, -0.05161805081947522, 0.5774690032616651],
        [0.5034256408649196, -0.05167661429723301, -0.443162425136426],
        [0.49413665553622205, 0.04050607695706589, 0.5774523681515825],
        [0.5808408971658352, -0.04493356316916096, 0.5775833469573991],
        [0.5745201044142929, 0.04713200112962236, -0.4430647165507745],
        [0.5730360081257153, 0.04719056460738015, 0.5775667118473166],
        [0.5823249934544128, -0.04499212664691875, -0.44304808144069185],
        [0.49562075182479964, 0.0404475134793081, -0.44317906024650866],
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

RESET_PALP_POSE = np.array(
    [
        0.5174325704574585,
        -0.0029695522971451283,
        0.12471308946609497,
        -0.929806649684906,
        0.36692020297050476,
        -0.025555845350027084,
        0.01326837856322527,
    ]
)


BBOX_ROI = np.array(
    [
        [0.506196825331111, -0.043235306355001196, 0.5804695750208919],
        [0.5066164297007841, -0.04348050754514507, -0.4311007035987048],
        [0.4928122028439124, 0.020206739021827125, 0.5804486448773705],
        [0.566811847214656, -0.030447105067965306, 0.580491618616227],
        [0.5538468290971307, 0.03274973911871914, -0.43109959014689136],
        [0.5534272247274575, 0.03299494030886301, 0.5804706884727056],
        [0.5672314515843292, -0.03069230625810918, -0.4310786600033698],
        [0.49323180721358556, 0.01996153783168325, -0.43112163374222634],
    ]
)


# interpolation
STEP_FAST = 250
STEP_SLOW = 2000

# search


def HISTORY_DTYPE(grid_size):
    return np.dtype(
        [
            ("sample_pt", np.dtype((np.int32, 2))),
            ("grid", np.dtype((np.float32, grid_size))),
        ]
    )


# palpation
PALP_DTYPE = np.dtype(
    [
        ("Fxyz", np.dtype((np.float32, (3)))),
        ("O_q_EE", np.dtype((np.float32, (4)))),
        ("O_p_EE", np.dtype((np.float32, (3)))),
        ("O_q_EE_target", np.dtype((np.float32, (4)))),
        ("O_p_EE_target", np.dtype((np.float32, (3)))),
        ("palp_id", np.int64),
        ("palp_state", np.int8),
        ("stiffness", np.float32),
        ("using_force_control_flag", np.int8),
        ("collect_points_flag", np.int8),
    ]
)


class PALP_CONST:
    max_wrench_norm_OSC = 4.0
    wrench_norm_FC = 5.5  # N
    buffer_size = 100
    dataset_buffer_size = 100
    force_stable_thres = 0.05  # N
    pos_stable_thres = 1e-4  # m
    above_height = 0.01
    palpate_depth = 0.035
    wrench_oscill_hz = 100
    t_oscill = 2  # oscillation period (s)
    angle_oscill = 10  # deg
    seed = 100
    ctrl_freq = 80
    grid_size = 0.0025  # m
    kernel_scale = 2  # normalized grid space units
    random_sample_count = 20  # normalized grid space units
    stiffness_normalization = 700


BBOX_DOCTOR_ROI = np.array(
    [
        [0.5408100548268969, -0.024274716849727936, 0.584794524289607],
        [0.5408328740458453, -0.024275135787011577, -0.4173593733857894],
        [0.5287603932331272, -0.023360972249816016, 0.584794249534733],
        [0.5415114869448344, -0.015024845872336573, 0.5847945363945514],
        [0.5294846445700131, -0.014111520209708295, -0.41735963603571885],
        [0.5294618253510647, -0.014111101272424653, 0.5847942616396774],
        [0.5415343061637828, -0.015025264809620215, -0.4173593612808449],
        [0.5287832124520756, -0.023361391187099657, -0.41735964814066334],
    ]
)
