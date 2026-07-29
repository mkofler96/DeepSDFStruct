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
from scipy.spatial import cKDTree

from DeepSDFStruct.mesh import torchSurfMesh
from DeepSDFStruct.SDF import SDFBase


def mesh_to_analytical(gt_sdf: SDFBase, gen_mesh: torchSurfMesh) -> float:
    """
    Calculates the SDF for every vertex on the mesh
    """

    return gt_sdf.forward(gen_mesh.vertices).abs().mean().item()


def chamfer_distance(mesh_a, mesh_b, n_surface_samples: int = 10000) -> float:
    """
    Symmetric Chamfer distance between two meshes.

    Parameters
    ----------
    mesh_a : trimesh.Trimesh
        First mesh.
    mesh_b : trimesh.Trimesh
        Second mesh.
    n_surface_samples : int
        Number of points sampled on each mesh surface.

    Returns
    -------
    float
        Symmetric Chamfer distance.
    """

    points_a, _ = trimesh.sample.sample_surface(mesh_a, n_surface_samples)
    points_b, _ = trimesh.sample.sample_surface(mesh_b, n_surface_samples)

    tree_a = cKDTree(points_a)
    tree_b = cKDTree(points_b)

    dist_a_to_b, _ = tree_b.query(points_a)
    dist_b_to_a, _ = tree_a.query(points_b)

    dist_a_to_b = dist_a_to_b**2
    dist_b_to_a = dist_b_to_a**2

    cd = dist_a_to_b.mean() + dist_b_to_a.mean()
    return float(cd)
