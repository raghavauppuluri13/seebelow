import rerun as rr
import numpy as np
import hashlib


def pcd_to_rr(pcd_name, pcd_np):
    class_id = int(hashlib.sha256(pcd_name.encode('utf-8')).hexdigest(), 16) % (1 << 16)
    uint8_value = np.uint8(class_id)
    return rr.Points3D(pcd_np,
                       colors=[uint8_value] * len(pcd_np),
                       class_ids=[class_id] * len(pcd_np))
