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
FORCE_CTRL_CFG = RPAL_CFG_PATH / "hybrid-force-controller.yml"
BASE_CALIB_FOLDER = RPAL_CFG_PATH / "base_camera_calib"
PAN_PAN_FORCE_CFG = RPAL_CFG_PATH / "pan-pan-force.yml"

# surface scan
# SURFACE_SCAN_PATH = RPAL_PKG_PATH / "meshes" / "tumors_gt_02-22-2024_12-29-44.ply"
SURFACE_SCAN_PATH = RPAL_PKG_PATH / "meshes" / "tumors_gt_03-22-2024_18-53-24.ply"

# GT_PATH = RPAL_PKG_PATH / "meshes" / "tumors_gt_12-27-2023_19-22-31.ply"
# GT_PATH = RPAL_PKG_PATH / "meshes" / "tumors_gt_02-22-2024_10-43-10.ply"
GT_PATH = RPAL_PKG_PATH / "meshes" / "tumors_gt_03-22-2024_18-48-42.ply"

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
        # 7 if no force, 9 if using_force_control
        ("action", np.dtype((np.float32, (9)))),
        ("palp_progress", np.float32),
        ("palp_pt", np.dtype((np.float32, (3)))),
        ("surf_normal", np.dtype((np.float32, (3)))),
        ("palp_id", np.int64),
        ("palp_state", np.int8),
        ("stiffness", np.float32),
        ("using_force_control_flag", np.int8),
        ("collect_points_flag", np.int8),
    ]
)


class PalpateState:
    ABOVE = 0
    PALPATE = 1
    RETURN = 2
    INIT = -1
    TERMINATE = -2

    def __init__(self):
        self.state = self.INIT

    def next(self):
        if self.state == self.TERMINATE:
            return
        if self.state == self.INIT:
            self.state = self.ABOVE
        else:
            self.state = (self.state + 1) % 3


class PALP_CONST:
    max_Fz = 5.0
    buffer_size = 100
    dataset_buffer_size = 100
    force_stable_thres = 0.05  # N
    pos_stable_thres = 1e-4  # m
    above_height = 0.01
    max_palp_disp = 0.021
    palpate_depth = 0.035
    t_oscill = 5  # oscillation period (s)
    angle_oscill = 0.002  # m
    seed = 102
    ctrl_freq = 80
    grid_size = 0.0025  # m
    kernel_scale = 2  # normalized grid space units
    random_sample_count = 10  # normalized grid space units
    stiffness_normalization = 3000
    stiffness_tumor_filter = 0.41
    max_palpations = 60
    algo = "bo"
    tumor_type = "hemisphere"
    discrete_only = False
    max_cf_time = 7.0


ROI_HEMISPHERE = np.array(
    [
        [0.5494851054901664, 0.021066363604046116, 0.5827981199328817],
        [0.5495652615098048, 0.02104891589692932, -0.4204256386930405],
        [0.549627864241707, -0.01690763997263283, 0.582798791769309],
        [0.5190613538207638, 0.02095198908389557, 0.5827956911115523],
        [0.5192842685919428, -0.017039462199900175, -0.42042739567794263],
        [0.5192041125723044, -0.01702201449278338, 0.5827963629479797],
        [0.5191415098404022, 0.020934541376778772, -0.4204280675143699],
        [0.5497080202613454, -0.016925087679749626, -0.42042496685661324],
    ]
)

ROI_HEMISPHERE = np.array(
    [
        [0.5459514929198789, -0.011521716006896944, 0.5878734235675551],
        [0.5459939011919658, -0.011529641163773864, -0.41561700402616597],
        [0.5290622953176659, -0.012271549640846082, 0.5878727157390413],
        [0.5454821194948848, -0.0009495834188418244, 0.5878733202370974],
        [0.5286353301647587, -0.0017073422096678826, -0.41561781518513763],
        [0.5285929218926718, -0.0016994170527909625, 0.5878726124085836],
        [0.5455245277669717, -0.0009575085757187445, -0.4156171073566237],
        [0.5291047035897528, -0.012279474797723003, -0.4156177118546799],
    ]
)


ROI_CRESCENT = np.array(
    [
        [0.5298615459594098, -0.06406711885822355, 0.5833643911652737],
        [0.5299749699903449, -0.06420170241600358, -0.4256088427635702],
        [0.5182548491482448, -0.031148948269858225, 0.5833586955505059],
        [0.5563537974314252, -0.05472615199856168, 0.5833661233394669],
        [0.5448605246511953, -0.021942564967976374, -0.42561280620414493],
        [0.5447471006202602, -0.02180798141019635, 0.5833604277246991],
        [0.5564672214623603, -0.0548607355563417, -0.425607110589377],
        [0.5183682731791799, -0.03128353182763825, -0.4256145383783381],
    ]
)

ROI_CRESCENT_1 = np.array(
    [
        [0.5564672214623719, -0.05486073555633771, -0.4256071105893771],
        [0.5563537974314369, -0.05472615199855758, 0.583366123339467],
        [0.5448605246511921, -0.02194256496796696, -0.42561280620414504],
        [0.5299749699903481, -0.0642017024160131, -0.4256088427635703],
        [0.5182548491482332, -0.03114894826986222, 0.583358695550506],
        [0.5183682731791682, -0.031283531827642354, -0.42561453837833824],
        [0.529861545959413, -0.06406711885823296, 0.5833643911652738],
        [0.544747100620257, -0.021807981410186826, 0.5833604277246992],
    ]
)

ROI_CRESCENT = np.array(
    [
        [0.5717376726175577, 0.009660589865789344, 0.5843518104089543],
        [0.5719407937317497, 0.00969949476501444, -0.42870669711150183],
        [0.5498923198941191, 0.001629906786663993, 0.5843471219479565],
        [0.5644267865487526, 0.02954792530315525, 0.5843511082969463],
        [0.542784554939506, 0.021556147123254998, -0.4287120876845077],
        [0.542581433825314, 0.021517242224029898, 0.5843464198359484],
        [0.5646299076629445, 0.029586830202380347, -0.4287073992235099],
        [0.5500954410083111, 0.0016688116858890913, -0.4287113855724996],
    ]
)


BBOX_DOCTOR_ROI = ROI_CRESCENT
