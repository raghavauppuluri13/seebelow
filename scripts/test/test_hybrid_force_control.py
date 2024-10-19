import argparse
import time
import numpy as np

from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.log_utils import get_deoxys_example_logger
from seebelow.utils.control_utils import generate_joint_space_min_jerk
from seebelow.utils.constants import PALP_CONST
import seebelow.utils.constants as seebelow_const

logger = get_deoxys_example_logger()

total_time = 5
POS = 0.002

start = np.full(2, -POS)
end = np.full(2, POS)
force_osc_out = generate_joint_space_min_jerk(
    start, end, total_time / 2, 1 / PALP_CONST.ctrl_freq
)
force_osc_in = generate_joint_space_min_jerk(
    end, start, total_time / 2, 1 / PALP_CONST.ctrl_freq
)
force_osc = force_osc_out + force_osc_in

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    robot_interface = FrankaInterface(
        str(seebelow_const.PAN_PAN_FORCE_CFG),
        use_visualizer=False,
        control_freq=PALP_CONST.ctrl_freq,
    )
    force_ctrl_cfg = YamlConfig(str(seebelow_const.FORCE_CTRL_CFG)).as_easydict()
    robot_interface._state_buffer = []

    action = np.zeros(9)
    z_unit = np.array([0, 0, -1])

    start_time = time.time()

    try:
        while True:
            if time.time() - start_time > total_time:
                start_time = time.time()

            force_idx = int((time.time() - start_time) / (1 / PALP_CONST.ctrl_freq))
            pos_t = force_osc[force_idx]["position"]
            # action[-3:] = mag * wrench
            # action[-3:] = z_unit
            action[0] = pos_t[0]
            action[1] = pos_t[1]
            action[2] = -0.005
            print(action)
            robot_interface.control(
                controller_type=seebelow_const.FORCE_CTRL_TYPE,
                action=action,
                controller_cfg=force_ctrl_cfg,
            )
            # print(f"Time duration: {((end_time - start_time) / (10**9))}")
    except KeyboardInterrupt:
        pass

    # stop
    action = np.zeros(9)
    robot_interface.close()
