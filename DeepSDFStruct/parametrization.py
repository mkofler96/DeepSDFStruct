"""
Parametrization Functions for Spatially-Varying Properties
===========================================================

This module provides parametrization functions that map from spatial coordinates
to parameter values, enabling SDFs with spatially-varying properties. This is
particularly useful for lattice structures where properties like strut thickness
or material density vary across the structure.

Classes
-------
Constant
    Spatially-constant parameter that is independent of position.
    Useful as a baseline or for uniform structures.
    
SplineParametrization
    B-spline-based parametrization providing smooth spatial variation.
    Parameters vary smoothly according to a spline function.

Both classes inherit from torch.nn.Module, making their parameters
automatically discoverable for gradient-based optimization.

Examples
--------
Create a lattice with varying thickness::

    from DeepSDFStruct.parametrization import SplineParametrization
    from DeepSDFStruct.torch_spline import TorchSpline
    import splinepy
    
    # Create a B-spline for thickness variation
    spline = splinepy.BSpline(
        degrees=[2, 2, 2],
        control_points=...,
        knot_vectors=...
    )
    
    # Wrap in parametrization module
    param_func = SplineParametrization(spline)
    
    # Use in lattice structure
    lattice = LatticeSDFStruct(
        tiling=[3, 3, 3],
        microtile=unit_cell,
        parametrization=param_func
    )
"""

import torch
import torch.nn as nn
import splinepy as sp
from DeepSDFStruct.torch_spline import TorchSpline


class Constant(nn.Module):
    """Spatially-constant parameter value.

    This parametrization returns the same parameter value(s) for all query
    points, independent of position. Useful for uniform structures or as
    a simple baseline.

    The parameter is stored as a torch.nn.Parameter, making it automatically
    discoverable for optimization.

    Parameters
    ----------
    value : float, list, or torch.Tensor
        The constant parameter value(s). Can be scalar or vector.
    device : str or torch.device, optional
        Device for computation ('cpu' or 'cuda').
    dtype : torch.dtype, optional
        Data type for the parameter.

    Attributes
    ----------
    param : torch.nn.Parameter
        The learnable constant parameter.

    Examples
    --------
    >>> from DeepSDFStruct.parametrization import Constant
    >>> import torch
    >>>
    >>> # Scalar constant
    >>> const = Constant(0.5)
    >>> queries = torch.rand(100, 3)
    >>> params = const(queries)
    >>> print(params.shape)  # (100, 1)
    >>> print(params[0])  # tensor([0.5])
    >>>
    >>> # Vector constant
    >>> const_vec = Constant([0.5, 1.0, 1.5])
    >>> params = const_vec(queries)
    >>> print(params.shape)  # (100, 3)
    """

    def __init__(self, value, device=None, dtype=None):
        super().__init__()
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, device=device, dtype=dtype)
        self.param = nn.Parameter(value)

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        """Evaluate constant parameter at query points.

        Parameters
        ----------
        queries : torch.Tensor
            Query point coordinates of shape (N, d). The spatial
            coordinates are ignored.

        Returns
        -------
        torch.Tensor
            Parameter values of shape (N, m) where m is the
            parameter dimension. All rows are identical.
        """
        N = queries.shape[0]
        return self.param.expand(N, -1)

    def set_param(self, new_value: torch.Tensor):
        """Update the constant parameter value.

        Parameters
        ----------
        new_value : torch.Tensor
            New parameter value(s).
        """
        with torch.no_grad():
            self.param.copy_(new_value.to(self.param.device))


class SplineParametrization(nn.Module):
    """B-spline-based spatially-varying parametrization.

    Uses a B-spline, Bezier, or NURBS function to provide smoothly-varying
    parameters across space. The spline control points are learnable parameters
    that can be optimized.

    This is useful for creating gradual transitions in material properties,
    thickness variations, or other spatially-dependent design variables.

    Parameters
    ----------
    spline : splinepy.BSpline, splinepy.Bezier, or splinepy.NURBS
        The spline function defining parameter variation. The spline's
        output dimension determines the parameter dimension.
    device : str or torch.device, optional
        Device for computation ('cpu' or 'cuda').

    Attributes
    ----------
    torch_spline : TorchSpline
        PyTorch-compatible wrapper for the spline.

    Examples
    --------
    >>> from DeepSDFStruct.parametrization import SplineParametrization
    >>> import splinepy
    >>> import torch
    >>>
    >>> # Create a 3D B-spline for thickness variation
    >>> spline = splinepy.BSpline(
    ...     degrees=[2, 2, 2],
    ...     control_points=torch.rand(27, 1),  # 3x3x3 grid, 1D output
    ...     knot_vectors=[
    ...         [0, 0, 0, 1, 1, 1],
    ...         [0, 0, 0, 1, 1, 1],
    ...         [0, 0, 0, 1, 1, 1]
    ...     ]
    ... )
    >>>
    >>> # Create parametrization
    >>> param = SplineParametrization(spline)
    >>>
    >>> # Evaluate at query points
    >>> queries = torch.rand(100, 3)
    >>> thickness = param(queries)
    >>> print(thickness.shape)  # (100, 1)
    """

    def __init__(self, spline: sp.BSpline | sp.Bezier | sp.NURBS, device=None):
        super().__init__()
        self.torch_spline = TorchSpline(spline, device=device)

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        """Evaluate spline parametrization at query points.

        Parameters
        ----------
        queries : torch.Tensor
            Query point coordinates of shape (N, d) where d matches
            the spline's parametric dimension.

        Returns
        -------
        torch.Tensor
            Parameter values of shape (N, m) where m is the spline's
            output dimension (number of parameter values per point).
        """
        return self.torch_spline(queries)

    def set_param(self, new_value: torch.Tensor):
        """Update the spline control points.

        Parameters
        ----------
        new_value : torch.Tensor
            New control point values. Shape must match the spline's
            control point array.
        """
        with torch.no_grad():
            self.torch_spline.control_points.copy_(
                new_value.to(self.torch_spline.control_points.device)
            )
