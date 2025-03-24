import os
import sys

from setuptools import setup, find_packages

setup(
    name="seebelow",
    version="0.0.1",
    packages=find_packages(),
    description="codebase for the seebelow paper",
    url="https://github.com/raghavauppuluri13/seebelow",
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
        "tfvis @ git+https://github.com/raghavauppuluri13/tfvis.git",
        # deoxys
        "setproctitle",
        "easydict",
        "termcolor",
        "zmq",
        "protobuf==3.13.0",
        "pybullet",
        "numba",
        "glfw",
        "hidapi",
    ],
)
