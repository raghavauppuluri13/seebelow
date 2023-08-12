"""Modify mesh vertex positions"""
from vedo import *
from pathlib import Path

pcd_dir = Path("./data/dataset_08-12-2023_05-02-59/proc_pcd/")

curr = Mesh(str(pcd_dir / f"{0}.ply"))

## Template python code to position this camera: ##
cam = dict(
    position=(0.149849, 0.0663566, -5.95956e-3),
    focal_point=(0.0567244, 9.27022e-3, 9.16724e-3),
    viewup=(-0.261789, 0.175583, -0.949019),
    distance=0.110272,
    clipping_range=(3.55267e-4, 0.355267),
)

plt = Plotter(interactive=True, axes=0)
plt.show(curr, __doc__, camera=cam)

for i in range(1, 100):
    next_mesh = Mesh(str(pcd_dir / f"{i}.ply"))
    plt += next_mesh
    plt -= curr
    curr = next_mesh
    plt.render()

plt.interactive().close()
