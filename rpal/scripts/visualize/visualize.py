import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import OpenGL.GL as gl
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from OpenGL.GL import *
from PyQt5 import Qt
from PyQt5.QtCore import QCoreApplication, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QHBoxLayout, QLabel, QMainWindow,
                             QOpenGLWidget, QVBoxLayout, QWidget)
from vedo import *
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


class MainWindow(Qt.QMainWindow):
    def __init__(self, dataset_path, data_file_path, parent=None, pcd_viz=False):

        Qt.QMainWindow.__init__(self)
        self.frame = Qt.QFrame()
        self.layout = Qt.QHBoxLayout()
        self.plot_layout = Qt.QVBoxLayout()
        self.pcd_layout = Qt.QVBoxLayout()
        self.widget = QVTKRenderWindowInteractor(self.frame)

        self.dataset_path = dataset_path
        self.data_file_path = data_file_path
        self.data = np.loadtxt(data_file_path)
        self.plot_data_position = [[], [], []]
        self.plot_data_force = [[], [], []]
        self.current_index = 0

        self.pcd_dir = Path(self.dataset_path) / "proc_pcd"

        self.cam = dict(
            position=(0.149849, 0.0663566, -5.95956e-3),
            focal_point=(0.0567244, 9.27022e-3, 9.16724e-3),
            viewup=(-0.261789, 0.175583, -0.949019),
            distance=0.110272,
            clipping_range=(3.55267e-4, 0.355267),
        )

        self.cam = dict(
            position=(0.202200, 0.116974, -0.0639292),
            focal_point=(0.0579752, 0.0140666, 0.0183658),
            viewup=(-0.503663, 0.0151277, -0.863768),
            distance=0.195354,
            clipping_range=(0.0130828, 0.434731),
        )

        # Set up the GUI layout with two images on the left and four subplots on the right
        self.pcd_viz = pcd_viz

        if self.pcd_viz:
            self.curr = Mesh(str(self.pcd_dir / f"{self.current_index}.ply"))
            self.pcd_layout.addWidget(self.widget)
        self.plt = Plotter(interactive=True, axes=0, qt_widget=self.widget)

        # Matplotlib figure with four subplots
        self.fig, self.axs = plt.subplots(2, 1, figsize=(10, 10))
        self.canvas = FigureCanvas(self.fig)
        self.plot_layout.addWidget(self.canvas)

        # Timer to update the GUI
        self.timer = Qt.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(1)  # 50ms delay

        self.layout.addLayout(self.plot_layout)
        self.layout.addLayout(self.pcd_layout)

        self.frame.setLayout(self.layout)
        self.setCentralWidget(self.frame)

        self.axs[0].clear()
        self.axs[1].clear()
        if self.pcd_viz:
            self.plt.show(self.curr, __doc__, camera=self.cam)
        self.show()

    def update(self):
        if self.pcd_viz:
            self.next_mesh = Mesh(str(self.pcd_dir / f"{self.current_index}.ply"))
            self.plt += self.next_mesh
            self.plt -= self.curr
            self.curr = self.next_mesh

        # Update Matplotlib Plot
        N = 100
        data_point = self.data[self.current_index]
        data_slice = self.data[self.current_index : self.current_index + N]
        colors = ["r", "g", "b"]
        labels_position = ["X Position", "Y Position", "Z Position"]
        labels_force = ["Force X (N)", "Force Y (N)", "Force Z (N)"]

        # Update Position Subplot
        self.axs[0].clear()
        for i in range(2, 3):
            self.plot_data_position[i].extend(data_slice[:, i].tolist())
            self.axs[0].plot(
                self.plot_data_position[i], color=colors[i], label=labels_position[i]
            )
        self.axs[0].legend()
        self.axs[0].set_title("Position")
        self.axs[0].autoscale_view()

        # Update Force Subplot
        self.axs[1].clear()
        for i in range(3):
            self.plot_data_force[i].extend(data_slice[:, -3 + i].tolist())
            self.axs[1].plot(
                self.plot_data_force[i], color=colors[i], label=labels_force[i]
            )
        self.axs[1].legend()
        self.axs[1].autoscale_view()

        self.plt.render()
        self.canvas.draw_idle()

        # Increment index
        self.current_index += N
        if self.current_index >= self.data.shape[0]:
            self.current_index = 0
            self.timer.stop()
            return
            # QCoreApplication.instance().quit()

    def onClose(self):
        self.widget.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ply Visualizer")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data/dataset_08-12-2023_05-02-59",
        help="the folder of ply files",
    )
    parser.add_argument(
        "--file_type", type=str, default=".ply", help="the folder of ply files"
    )
    parser.add_argument(
        "--window_width",
        type=int,
        default="640",
        help="the width of visualization window",
    )
    parser.add_argument(
        "--pause_time",
        type=float,
        default="0.2",
        help="time to sleep for each frame, only effective in auto-play mode",
    )
    parser.add_argument(
        "--window_height",
        type=int,
        default="480",
        help="the height of visualization window",
    )
    parser.add_argument(
        "--camera_view",
        type=str,
        default="camera_params.json",
        help="The screen camera pose json file, if you don't have one record it with key P when you are in an open3d window",
    )
    args = parser.parse_args()

    dataset_path = args.dataset_path
    data_file_path = os.path.join(dataset_path, "timeseries.txt")

    app = Qt.QApplication(sys.argv)

    window = MainWindow(dataset_path, data_file_path)

    app.aboutToQuit.connect(window.onClose)
    app.exec_()
