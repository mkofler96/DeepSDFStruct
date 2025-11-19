DeepSDFStruct Documentation
===========================

**DeepSDFStruct** is a differentiable framework for generating and deforming 3D microstructured materials using Signed Distance Functions (SDFs) and spline-based lattices. The library provides tools for creating, manipulating, and optimizing complex 3D geometries with applications in additive manufacturing, materials science, and computational design.

Overview
--------

DeepSDFStruct combines classical geometric representations (SDFs, splines) with deep learning approaches to enable:

* **Flexible Geometry Definition**: Define complex 3D structures using primitive shapes, lattice patterns, or learned implicit representations
* **Differentiable Operations**: All operations are PyTorch-compatible, enabling gradient-based optimization
* **Mesh Generation**: Extract high-quality surface and volume meshes using FlexiCubes and FlexiSquares algorithms
* **Deep Learning Integration**: Train neural networks to represent and generate novel geometric structures
* **Structural Optimization**: Optimize designs for specific mechanical properties or manufacturing constraints

Key Features
------------

Signed Distance Functions (SDFs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The core abstraction is the ``SDFBase`` class, which represents geometries as implicit functions that return the signed distance to the nearest surface. The library provides:

* **Primitive Shapes**: Spheres, cylinders, boxes, tori, and other basic geometric primitives
* **Lattice Structures**: Periodic microstructures with customizable unit cells
* **Mesh-based SDFs**: Convert existing triangle meshes to SDF representations
* **Deep Learning SDFs**: Neural network-based implicit representations trained from data
* **Composition Operations**: Combine multiple SDFs using boolean operations (union, intersection, difference)

Spline-based Deformations
~~~~~~~~~~~~~~~~~~~~~~~~~~

DeepSDFStruct uses B-splines for smooth, parametric deformations:

* **TorchSpline**: PyTorch-compatible spline evaluation for differentiable operations
* **Parametrization**: Define spatially-varying material properties or geometry parameters
* **Deformation Mapping**: Apply smooth deformations to base geometries

Mesh Generation
~~~~~~~~~~~~~~~

Generate high-quality meshes from SDF representations:

* **FlexiCubes**: Advanced dual contouring for smooth, feature-preserving surface meshes
* **FlexiSquares**: 2D mesh extraction for cross-sections and planar geometries
* **Volume Meshing**: Generate tetrahedral meshes for finite element analysis
* **Mesh Processing**: Clean, repair, and optimize generated meshes

Deep Learning Components
~~~~~~~~~~~~~~~~~~~~~~~~

Train neural networks to learn geometric representations:

* **DeepSDF Architecture**: Multi-layer perceptron decoder for implicit geometry
* **Hierarchical Models**: Multi-scale representations for complex structures
* **Training Pipeline**: Complete workflow for dataset generation, training, and inference
* **Latent Space Optimization**: Optimize learned representations for specific objectives

Optimization and Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

Tools for structural design and optimization:

* **Gradient-based Optimization**: Leverage automatic differentiation for shape optimization
* **MMA Optimizer**: Method of Moving Asymptotes for constrained optimization
* **Finite Element Integration**: Interface with TorchFEM for structural analysis
* **Design of Experiments**: Generate parameter sweeps and design spaces

Core Modules
------------

* ``DeepSDFStruct.SDF``: Base classes for signed distance functions
* ``DeepSDFStruct.mesh``: Mesh generation and processing utilities  
* ``DeepSDFStruct.lattice_structure``: Periodic lattice structure generation
* ``DeepSDFStruct.sdf_primitives``: Primitive geometric shapes
* ``DeepSDFStruct.sampling``: Strategies for sampling points from SDFs
* ``DeepSDFStruct.optimization``: Optimization algorithms and utilities
* ``DeepSDFStruct.torch_spline``: PyTorch-compatible B-spline operations
* ``DeepSDFStruct.parametrization``: Parametrization functions for spatially-varying properties
* ``DeepSDFStruct.deep_sdf``: Deep learning models and training pipelines
* ``DeepSDFStruct.flexicubes``: FlexiCubes mesh extraction algorithm
* ``DeepSDFStruct.flexisquares``: FlexiSquares 2D mesh extraction
* ``DeepSDFStruct.splinepy_unitcells``: Predefined unit cell geometries

Installation
------------

Install directly from GitHub using pip::

    pip install git+https://github.com/mkofler96/DeepSDFStruct.git

Or add to your UV project::

    uv add git+https://github.com/mkofler96/DeepSDFStruct.git

Quick Start
-----------

Basic SDF operations::

    import torch
    from DeepSDFStruct.sdf_primitives import SphereSDF, CylinderSDF
    from DeepSDFStruct.mesh import create_3D_mesh

    # Create a sphere SDF
    sphere = SphereSDF(center=[0, 0, 0], radius=0.5)
    
    # Query SDF values
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    distances = sphere(points)
    
    # Generate mesh
    mesh = create_3D_mesh(sphere, resolution=64)

For more examples, see the `example notebook <https://github.com/mkofler96/DeepSDFStruct/blob/main/example.ipynb>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api_reference

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
