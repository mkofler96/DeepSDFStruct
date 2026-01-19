"""
Lattice Structure Generation
=============================

This module provides tools for creating periodic lattice structures with
deformable geometries. Lattices are built by tiling a unit cell (microtile)
in a regular pattern and applying optional smooth deformations via B-splines.

The main class, LatticeSDFStruct, enables creation of complex microstructured
materials with spatially-varying properties, useful for applications in:
- Topology optimization
- Additive manufacturing
- Metamaterial design
- Lightweight structural components

Key Features
------------
- Periodic tiling of arbitrary unit cell geometries
- Smooth deformations via spline-based mapping
- Spatially-varying parametrization (e.g., varying strut thickness)
- Support for 2D and 3D lattices
- Integration with boundary conditions and capping
"""

import logging


import numpy as _np
import torch as _torch

from splinepy._base import SplinepyBase as _SplinepyBase
from splinepy import BSpline as _BSpline
from .SDF import SDFBase as _SDFBase
from .SDF import CapBorderDict
from DeepSDFStruct.torch_spline import TorchSpline
import DeepSDFStruct
import gustaf as gus

logger = logging.getLogger(DeepSDFStruct.__name__)


class LatticeSDFStruct(_SDFBase):
    """Helper class to facilitate the construction of periodic lattice SDF structures.

    This class creates periodic lattice structures by tiling a unit cell geometry
    (microtile) in a regular pattern and optionally deforming the result through
    a spline-based mapping. The microtile can be parametrized to have spatially-
    varying properties (e.g., thickness that varies across the structure).

    The lattice is defined in parametric space [0,1]^d and can be mapped to
    physical space through a deformation spline. Boundary conditions can be
    applied to cap the structure at domain boundaries.

    Parameters
    ----------
    tiling : list of int or int, optional
        Number of repetitions of the microtile in each parametric dimension.
        If an int, uses the same tiling in all dimensions.
    deformation_spline : TorchSpline, optional
        Spline function that maps from parametric to physical space,
        enabling smooth geometric deformations of the lattice.
    microtile : SDFBase, optional
        The unit cell geometry to be tiled. Should be defined in the
        unit cube [0,1]^d.
    parametrization : torch.nn.Module, optional
        Function that provides spatially-varying parameters for the microtile
        (e.g., varying strut thickness). Takes parametric coordinates and
        returns parameter values.
    cap_border_dict : CapBorderDict, optional
        Dictionary specifying whether material should be added or removed
        at domain faces (for capping the structure at boundaries).
    cap_outside_of_unitcube : bool, default True
        If True, caps geometry outside the unit cube.

    Attributes
    ----------
    tiling : list of int
        Number of tiles in each dimension.
    microtile : SDFBase
        The unit cell geometry.
    geometric_dim : int
        Geometric dimensionality (2 or 3).
    parametric_dimension : int
        Parametric dimensionality (equal to geometric_dim).

    Notes
    -----
    The microtile should ideally have periodic boundary conditions to ensure
    smooth connections between adjacent tiles. For parametrized microtiles,
    the microtile must provide:
    - `evaluation_points`: Points where parameters are evaluated
    - `para_dim`: Dimensionality of the parameter space
    - `_set_param`: Method to update parameters

    Examples
    --------
    >>> from DeepSDFStruct.lattice_structure import LatticeSDFStruct
    >>> from DeepSDFStruct.sdf_primitives import SphereSDF
    >>> from DeepSDFStruct.torch_spline import TorchSpline
    >>> import torch
    >>>
    >>> # Create a simple unit cell
    >>> unit_cell = SphereSDF(center=[0.5, 0.5, 0.5], radius=0.3)
    >>>
    >>> # Create lattice with 3x3x3 tiling
    >>> lattice = LatticeSDFStruct(
    ...     tiling=[3, 3, 3],
    ...     microtile=unit_cell
    ... )
    >>>
    >>> # Query lattice SDF
    >>> points = torch.rand(100, 3)
    >>> distances = lattice(points)
    """

    deformation_spline: TorchSpline

    def __init__(
        self,
        tiling: list[int] | int | None = None,
        deformation_spline: TorchSpline | None = None,
        microtile: _SDFBase | None = None,
        parametrization: _torch.nn.Module | None = None,
    ):
        """Helper class to facilitate the construction of microstructures.

        Parameters
        ----------
        deformation_function : spline
          Outer function that describes the contour of the microstructured
          geometry
        tiling : list of integers
          microtiles per parametric dimension
        microtile : SDFBase
          Representation of the building block defined in the unit cube
        parametrization_function : Callable (optional)
          Function to describe spline parameters
        """
        if not isinstance(parametrization, _torch.nn.Module):
            raise TypeError("Parametrization must be of type _Parametrization")
        super().__init__(
            deformation_spline=deformation_spline, parametrization=parametrization
        )
        self.tiling = tiling
        self.microtile = microtile
        self.geometric_dim = len(tiling)

    @property
    def parametric_dimension(self):
        return len(self.tiling)

    def _get_domain_bounds(self):
        match self.microtile.geometric_dim:
            case 2:
                return _torch.tensor([[0.0, 0.0], [1.0, 1.0]])
            case 3:
                return _torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    def _compute(self, samples: _torch.Tensor):
        """Function, that - if required - parametrizes the microtiles.

        In order to use said function, the Microtile needs to provide a couple
        of attributes:

         - evaluation_points - a list of points defined in the unit cube
           that will be evaluated in the parametrization function to provide
           the required set of data points
         - para_dim - dimensionality of the parametrization
           function and number of design variables for said microtile

        Parameters
        ----------
        None

        Returns
        -------
         : Callable
          Function that describes the local tile parameters
        """
        orig_device = samples.device
        orig_dtype = samples.dtype
        bounds = _torch.tensor(
            self.deformation_spline.spline.parametric_bounds,
            device=orig_device,
            dtype=orig_dtype,
        )
        if self.parametrization is not None:
            spline_domain_samples = _torch.clamp(samples, min=bounds[0], max=bounds[1])
            parameters = self.parametrization(spline_domain_samples)
            self.microtile._set_param(parameters)

        queries_transformed = _torch.zeros_like(samples)
        for i_dim, t in enumerate(self.tiling):
            queries_transformed[:, i_dim] = transform(samples[:, i_dim], t)
        sdf_values = self.microtile(queries_transformed)
        # self.plot_transformed_untransformed(queries, queries_transformed)
        return sdf_values

    def _sanity_check(self):
        """Check all members and consistency of user data.

        Parameters
        ----------
        updated_properties : bool
          Sets the updated_properties variable to value, which indicates,
          wheither the microstructure needs to be rebuilt

        Returns
        -------
        passes: bool
        """
        pass

    def plot_samples(self, samples, sdf_values):
        vp = gus.Vertices(vertices=samples)
        vp.vertex_data["distance"] = sdf_values
        vp.show_options["cmap"] = "coolwarm"
        gus.show(vp, axes=1)

    def plot_intermesh(self, verts: _torch.Tensor, faces: _torch.Tensor):
        gus_faces = gus.Faces(vertices=verts.cpu().detach(), faces=faces.cpu().detach())
        gus.show(gus_faces, axes=1)

    def plot_slice(self, *args, **kwargs):
        xmin = self.deformation_spline.control_points[:, 0].min().item()
        xmax = self.deformation_spline.control_points[:, 0].max().item()
        ymin = self.deformation_spline.control_points[:, 1].min().item()
        ymax = self.deformation_spline.control_points[:, 1].max().item()

        kwargs.setdefault("xlim", (xmin, xmax))
        kwargs.setdefault("ylim", (ymin, ymax))

        return super().plot_slice(*args, **kwargs)


def constantLatvec(value):
    return _BSpline([0, 0, 0], [[-1, 1], [-1, 1], [-1, 1]], [value])


def transform(x, t, bounds=[0, 1]):
    # transform x from [0,1] to [0,1]
    x_norm = (x - bounds[0]) / (bounds[1] - bounds[0])
    x_transformed = 2 * _torch.abs(t * x_norm / 2 - _torch.floor((t * x_norm + 1) / 2))
    return x_transformed


def check_tiling_input(tiling):
    if isinstance(tiling, list) or isinstance(tiling, tuple):
        if len(tiling) != 3:
            raise ValueError("Tiling must be a list of 3 integers")
        tiling = _np.array(tiling)
    elif isinstance(tiling, int):
        tiling = _np.array([tiling, tiling, tiling])
    else:
        raise ValueError("Tiling must be a list or an integer")
