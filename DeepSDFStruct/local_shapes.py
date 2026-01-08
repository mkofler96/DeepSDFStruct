import logging


import numpy as _np
import torch as _torch

from .SDF import SDFBase as _SDFBase
from .SDF import CapBorderDict
import DeepSDFStruct
from DeepSDFStruct.lattice_structure import check_tiling_input, transform

logger = logging.getLogger(DeepSDFStruct.__name__)


class LocalShapesSDF(_SDFBase):
    """Helper class to facilitatae the construction of Lattice SDF Structures."""

    def __init__(
        self,
        tiling: list[int] | int | None = None,
        unit_cell: _SDFBase | None = None,
        parametrization: _torch.nn.Module | None = None,
        bounds=_torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]),
    ):
        """Helper class to facilitatae the construction of microstructures.

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
        check_tiling_input(tiling)
        super().__init__(deformation_spline=None, parametrization=parametrization)
        self.tiling = (
            [tiling] * unit_cell.geometric_dim if isinstance(tiling, int) else tiling
        )
        self.unit_cell = unit_cell
        self.bounds = bounds

    @property
    def tiling(self):
        """Number of microtiles per parametric dimension.

        Parameters
        ----------
        None

        Returns
        -------
        tiling : list<int>
        """
        if hasattr(self, "_tiling"):
            return self._tiling
        else:
            return None

    @tiling.setter
    def tiling(self, tiling):
        """Setter for the tiling attribute, defining the number of microtiles
        per parametric dimension.

        Parameters
        ----------
        tiling : int / list<int>
          Number of tiles for each dimension respectively
        Returns
        -------
        None
        """
        if (
            not isinstance(tiling, list)
            and not isinstance(tiling, int)
            and not isinstance(tiling, tuple)
        ):
            raise ValueError(
                "Tiling mus be either list of integers of integer " "value"
            )
        self._tiling = tiling

    @property
    def unit_cell(self):
        """Microtile that is either a spline, a list of splines, or a class
        that provides a `create_tile` function."""
        if hasattr(self, "_microtile"):
            return self._microtile
        else:
            self._logi(
                "microtile is empty. "
                "Please checkout splinepy.microstructure.tiles.show() for "
                "predefined tile collections."
            )
            return None

    @unit_cell.setter
    def unit_cell(self, microtile):
        """Setter for microtile.

        Microtile must be either a spline, a list of splines, or a class that
        provides (at least) a `create_tile` function and a `dim` member.

        Parameters
        ----------
        microtile : spline / list<splines> / user-object
          arbitrary long list of splines that define the microtile

        Returns
        -------
        None
        """
        # place single tiles into a list to provide common interface
        if not isinstance(microtile, _SDFBase):
            raise TypeError(f"Microtile must be SDF, not {type(microtile)}")
        # Assign Microtile object to member variable
        self._microtile = microtile

    def _get_domain_bounds(self):
        return self.bounds

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
        if self.parametrization is not None:
            parameters = self.parametrization(samples)
            self.unit_cell._set_param(parameters)

        queries_transformed = _torch.zeros_like(samples)
        if self.unit_cell.geometric_dim == 2:
            tx, ty = self._tiling
            queries_transformed[:, 0] = transform(
                samples[:, 0], tx, bounds=self.bounds[:, 0]
            )
            queries_transformed[:, 1] = transform(
                samples[:, 1], ty, bounds=self.bounds[:, 1]
            )
        elif self.unit_cell.geometric_dim == 3:
            tx, ty, tz = self._tiling
            queries_transformed[:, 0] = transform(
                samples[:, 0], tx, bounds=self.bounds[:, 0]
            )
            queries_transformed[:, 1] = transform(
                samples[:, 1], ty, bounds=self.bounds[:, 1]
            )
            queries_transformed[:, 2] = transform(
                samples[:, 2], tz, bounds=self.bounds[:, 2]
            )

        sdf_values = self.unit_cell(queries_transformed)

        return sdf_values
