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
    """Helper class to facilitatae the construction of Lattice SDF Structures."""

    deformation_spline: _SplinepyBase

    def __init__(
        self,
        tiling: list[int] | int | None = None,
        deformation_spline: TorchSpline | None = None,
        microtile: _SDFBase | None = None,
        parametrization: _torch.nn.Module | None = None,
        cap_border_dict: CapBorderDict = None,
        cap_outside_of_unitcube: bool = True,
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
        super().__init__(
            deformation_spline=deformation_spline,
            parametrization=parametrization,
            cap_border_dict=cap_border_dict,
            cap_outside_of_unitcube=cap_outside_of_unitcube,
        )
        self.tiling = tiling
        self.microtile = microtile
        self.geometric_dim = len(tiling)

    @property
    def parametric_dimension(self):
        return len(self.tiling)

    def _get_domain_bounds(self):
        return _np.array([[-1, 1], [-1, 1], [-1, 1]])

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
        xmin = self._deformation_spline.control_points[:, 0].min().item()
        xmax = self._deformation_spline.control_points[:, 0].max().item()
        ymin = self._deformation_spline.control_points[:, 1].min().item()
        ymax = self._deformation_spline.control_points[:, 1].max().item()

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
