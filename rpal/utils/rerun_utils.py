import rerun as rr
import numpy as np
import hashlib
from rpal.utils.constants import HISTORY_DTYPE
import matplotlib


def encode_string_to_uint8(s):
    # Hash the string using SHA-256
    hash_bytes = hashlib.sha256(s.encode('utf-8')).digest()

    # Take the first 3 bytes of the hash for simplicity (could use other strategies)
    selected_bytes = hash_bytes[:3]

    # Convert each byte to an integer (uint8_t)
    uint8_values = [selected_bytes[i] for i in range(3)]

    return uint8_values


def pcd_to_rr(pcd_name, pcd_np, colors=None):
    if colors is None:
        color = encode_string_to_uint8(pcd_name)
        colors = np.tile(np.array(color), (len(pcd_np), 1))
    return rr.Points3D(pcd_np, colors=colors)


def search_grid_to_rr(search_grid):
    return rr.Tensor(search_grid, dim_names=('X', 'Y'))
