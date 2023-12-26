import os
import sys

from setuptools import setup, find_packages

setup(
    name="rpal",
    version="0.0.1",
    packages=find_packages(),
    description="Robot Palpation Code",
    url="https://github.com/raghavauppuluri13/robot-palpation",
    author="Raghava Uppuluri",
    install_requires=[
        "open3d-cpu==0.17.0",
        "numpy",
        "pin==2.6.20",
        "opencv-python",
        "tqdm",
        "scipy",
        "pynput",
        "pyyaml",
        "serial",
        "pyrealsense2",
        "torch",
        # deoxys
        "setproctitle",
        "easydict",
        "termcolor",
        "zmq",
        "protobuf==3.20.0",
        "pybullet",
        "numba",
        "glfw",
        "hidapi",
    ],
)
