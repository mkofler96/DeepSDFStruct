"""
Mesh-to-Analytical SDF Evaluation
=================================

This module provides functions for evaluating mesh quality by computing
SDF values at mesh vertices using ground truth analytical SDFs.

This is useful for:
- Quantifying reconstruction accuracy
- Measuring surface deviation
- Validating mesh generation algorithms
"""

import trimesh


def mesh_to_analytical(gt_sdf, gen_mesh):
    """
    Calculates the SDF for every vertex on the mesh
    """
    mesh = trimesh.Trimesh(gen_mesh)

    return gt_sdf(mesh.vertices)
