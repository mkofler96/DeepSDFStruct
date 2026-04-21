"""
PLY File Screenshot Generator
=============================

This utility script generates screenshots and animated GIFs from PLY mesh files.
It uses Vedo for 3D rendering with a fixed camera configuration to ensure
consistent viewpoints across multiple meshes.

Usage
-----
From command line::

    python create_screenshots_from_plyfiles.py /path/to/ply/files

The script will:
1. Find all .ply files in the specified folder
2. Generate a .png screenshot for each mesh
3. Create an animated GIF combining all screenshots

This is useful for:
- Visualizing reconstruction sequences
- Creating animations of optimization progress
- Generating consistent documentation images
- Batch processing of mesh files
"""

import vedo
import pathlib
from natsort import natsorted
import imageio.v3 as imageio

cam = dict(
    position=(3.51484, 3.27242, 4.06787),
    focal_point=(-3.98290e-3, -2.36815e-3, 1.02887e-3),
    viewup=(-0.308731, 0.853375, -0.420043),
    roll=2.37119,
    distance=6.29647,
    clipping_range=(3.00412, 10.4561),
)


def main(ply_file_folder):
    images = []
    plt = vedo.Plotter(interactive=False)
    for ply_filename in natsorted(pathlib.Path(ply_file_folder).rglob("*.ply")):
        print(f"Processing {ply_filename}")
        mesh = vedo.load(str(ply_filename))
        plt.show(mesh, interactive=False, camera=vedo.camera_from_dict(cam))
        vedo.screenshot(ply_filename.with_suffix(".png").as_posix())
        image = imageio.imread(str(ply_filename.with_suffix(".png")))
        images.append(image)
        plt.clear()

    plt.close()
    imageio.imwrite(ply_filename.parent / "reconstruction.gif", images, duration=300)


if __name__ == "__main__":
    # argument parser
    import argparse

    parser = argparse.ArgumentParser(description="Create screenshots from ply files")
    parser.add_argument("ply_file_folder", type=str, help="Folder containing ply files")
    args = parser.parse_args()
    main(args.ply_file_folder)
