"""
FlexiSquares - Differentiable 2D Isocontour Extraction
======================================================

This module implements FlexiSquares, a differentiable 2D variant of the
Dual Marching Squares algorithm for extracting isocontours from scalar fields.

FlexiSquares enables gradient-based optimization of surface representations
by making the mesh extraction process differentiable. This is particularly
useful for inverse problems where the geometry must be optimized to meet
certain criteria.

The module uses precomputed lookup tables to handle all possible Marching
Squares configurations efficiently.
"""
