"""
Splinepy Unit Cell Library
==========================

This module provides a collection of spline-based unit cell geometries
for lattice structures. These are built using the splinepy library and
can be used with the LatticeSDFStruct class.

Available Unit Cells
-------------------

CrossLattice (cross_lattice.py)
    2D cross-shaped lattice with configurable strut thickness.
    
Chi3D (chi_3D.py)
    3D chi-shaped lattice structure, common in mechanical metamaterials.
    
Snappy3D (snappy_3d.py)
    3D snappy lattice with snap-through behavior.
    
DoubleLatticeExtruded (double_lattice_extruded.py)
    Extruded double lattice structure.

All unit cells inherit from splinepy's TileBase class and provide:
- Parametric thickness control
- Sensitivity computation for optimization
- 2D and 3D variants
- Integration with the DeepSDFStruct framework

These geometries are particularly useful for:
- Mechanical metamaterial design
- Additive manufacturing
- Topology optimization
- Multi-scale structural analysis
"""
