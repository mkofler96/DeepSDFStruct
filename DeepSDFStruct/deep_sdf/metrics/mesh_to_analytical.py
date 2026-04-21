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
from DeepSDFStruct.mesh import torchSurfMesh
from DeepSDFStruct.SDF import SDFBase


def mesh_to_analytical(gt_sdf: SDFBase, gen_mesh: torchSurfMesh) -> float:
    """
    Calculates the SDF for every vertex on the mesh
    """

    return gt_sdf.forward(gen_mesh.vertices).abs().mean().item()
