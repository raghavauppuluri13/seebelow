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

t = np.arange(len(Frms))

plt.plot(t, pos[:, 2])

plt.title("1d heatmap")
plt.xlabel("t (steps)")
plt.ylabel("Z (m)")
plt.draw()

# Convert to OpenCV
fig = plt.gcf()
fig.canvas.draw()
img_arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
img_arr = img_arr.reshape(fig.canvas.get_width_height()[::-1] + (3,))
img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
dataset_name = dataset_path.split("/")[-1]
cv2.imwrite(f"{dataset_name}_heatmap.png", img_arr)
