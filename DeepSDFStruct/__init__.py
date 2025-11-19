"""
DeepSDFStruct - Differentiable Framework for 3D Microstructured Materials
===========================================================================

DeepSDFStruct is a comprehensive library for creating, manipulating, and optimizing
3D microstructured geometries using Signed Distance Functions (SDFs) and spline-based
lattices. The framework integrates classical geometric representations with deep
learning approaches, enabling differentiable design and optimization workflows.

Key Components
--------------

SDF Representations
    - ``DeepSDFStruct.SDF``: Abstract base class and core SDF utilities
    - ``DeepSDFStruct.sdf_primitives``: Geometric primitives (spheres, cylinders, etc.)
    - ``DeepSDFStruct.lattice_structure``: Periodic lattice microstructures

Mesh Operations
    - ``DeepSDFStruct.mesh``: Mesh generation, processing, and export
    - ``DeepSDFStruct.flexicubes``: Advanced dual contouring (3D)
    - ``DeepSDFStruct.flexisquares``: Dual contouring for 2D cross-sections

Deep Learning
    - ``DeepSDFStruct.deep_sdf``: Neural network models and training
    - ``DeepSDFStruct.sampling``: Data generation and sampling strategies

Optimization
    - ``DeepSDFStruct.optimization``: MMA and gradient-based optimization
    - ``DeepSDFStruct.parametrization``: Spatially-varying parameter functions

Utilities
    - ``DeepSDFStruct.torch_spline``: Differentiable B-spline operations
    - ``DeepSDFStruct.plotting``: Visualization tools
    - ``DeepSDFStruct.utils``: General utility functions

Examples
--------
Create a simple sphere and generate a mesh::

    from DeepSDFStruct.sdf_primitives import SphereSDF
    from DeepSDFStruct.mesh import create_3D_mesh
    
    sphere = SphereSDF(center=[0, 0, 0], radius=0.5)
    mesh = create_3D_mesh(sphere, resolution=64)

Create a lattice structure::

    from DeepSDFStruct.lattice_structure import LatticeSDFStruct
    from DeepSDFStruct.torch_spline import TorchSpline
    
    # Define deformation spline and unit cell
    # ... (see documentation for details)
    
    lattice = LatticeSDFStruct(
        tiling=[3, 3, 3],
        deformation_spline=deformation,
        microtile=unit_cell
    )

For comprehensive examples, see the example notebook in the repository.
"""

import DeepSDFStruct.utils

DeepSDFStruct.utils.configure_logging()

__version__ = "1.3.1"
__author__ = "Michael Kofler"
