import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as sp
from matplotlib import cm, patches
from scipy.interpolate import griddata
from tqdm import tqdm

# Argument Parser
parser = argparse.ArgumentParser(description="Time-series Heatmap Generator")
parser.add_argument(
    "--dataset_path",
    type=str,
    default="./data/dataset_08-12-2023_05-02-59",
    help="Folder containing time-series data",
)
args = parser.parse_args()

# Data Load
dataset_path = args.dataset_path
data_file_path = os.path.join(dataset_path, "timeseries.txt")
data = np.loadtxt(data_file_path)

# Split Data
pos = data[:, :3]
force = data[:, -3:]

Frms = np.sqrt(np.sum(force**2, axis=1))

# Video Writer Setup
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# out = cv2.VideoWriter("heatmap_video.mp4", fourcc, 20.0, (640, 480))

# Gaussian Smoothing

# Circle Data
point1 = [0.61346058, 0.07027999, 0.05241557]  # magnet
radius1 = 0.01732 / 2
point2 = [0.60665408, 0.09511717, 0.05193599]  # 3d print
radius2 = 0.005

pos_x = pos[Frms > 5, 1]
pos_y = pos[Frms > 5, 0]
pos_z = pos[Frms > 5, 2]

print("pos_y", pos_y.std())
print("pos_x", pos_x.std())


x_min, x_max = np.min(pos_x), np.max(pos_x)
y_min, y_max = np.min(pos_y), np.max(pos_y)
dim_x = 30
dim_y = 30


# Frms = Frms[pos[:, 2] < 0.055]
pos_palp = pos[pos[:, 2] < 0.06]

plt.axis("equal")

x = np.linspace(x_min, x_max, dim_x)
y = np.linspace(y_min, y_max, dim_y)

X, Y = np.meshgrid(x, y)

# Interpolate (x,y,z) points [mat] over a normal (x,y) grid [X,Y]
#   Depending on your "error", you may be able to use other methods
Z = griddata((pos_x, pos_y), pos_z, (X, Y), method="nearest")
plt.pcolormesh(X, Y, Z)

# plt.scatter(pos_palp[:, 1], pos_palp[:, 0], marker="x")
# Add circles
circle1 = patches.Circle(
    (point1[1], point1[0]),
    radius1,
    fill=False,
    color="blue",
)
circle2 = patches.Circle(
    (point2[1], point2[0]),
    radius2,
    fill=False,
    color="green",
)

# plt.gca().add_patch(circle1)
# plt.gca().add_patch(circle2)

plt.title("Heatmap with smoothing")
plt.xlabel("Y (m)")
plt.ylabel("X (m)")
cbar = plt.colorbar()
cbar.set_label("Z (m)", rotation=270, labelpad=15)
plt.draw()

# Convert to OpenCV
fig = plt.gcf()
fig.canvas.draw()
img_arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
img_arr = img_arr.reshape(fig.canvas.get_width_height()[::-1] + (3,))
img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)

dataset_name = dataset_path.split("/")[-1]
cv2.imwrite(f"{dataset_path}/{dataset_name}_2d_heatmap.png", img_arr)
