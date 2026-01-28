"""
FlexiCubes - Differentiable 3D Mesh Extraction
==============================================

This module implements FlexiCubes, a differentiable 3D variant of the
Dual Marching Cubes algorithm for extracting meshes from scalar fields.

FlexiCubes enables gradient-based optimization of mesh representations
by making the mesh extraction process fully differentiable. This is
essential for inverse problems and optimization tasks where the geometry
must be optimized to satisfy certain constraints or objectives.

The module uses precomputed lookup tables to handle all 256 possible
Marching Cubes configurations efficiently.
"""
