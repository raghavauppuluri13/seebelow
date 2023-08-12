import json
import open3d as o3d

c = {
    "serial": "",
    "color_format": "RS2_FORMAT_RGB8",
    "color_resolution": "0,540",
    "depth_format": "RS2_FORMAT_Z16",
    "depth_resolution": "0,480",
    "fps": "30",
    "visual_preset": "RS2_L500_VISUAL_PRESET_MAX_RANGE",
}


rs = o3d.t.io.RealSenseSensor()
rs.init_sensor(rs_cfg)
rs.start_capture(True)  # true: start recording with capture
intrinsics = rs.get_metadata().intrinsics

for fid in range(150):
    im_rgbd = rs.capture_frame(True, True)  # wait for frames and align them
    pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(im_rgbd, intrinsics)

rs.stop_capture()
