import logging


import numpy as _np
import torch as _torch

from splinepy._base import SplinepyBase as _SplinepyBase
from splinepy import BSpline as _BSpline
from .SDF import SDFBase as _SDFBase
from .SDF import CapBorderDict
from DeepSDFStruct.parametrization import _Parametrization
import gustaf as gus

logger = logging.getLogger(__name__)

device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")


class LatticeSDFStruct(_SDFBase):
    """Helper class to facilitatae the construction of Lattice SDF Structures."""

    deformation_spline: _SplinepyBase

    def __init__(
        self,
        tiling: list[int] | int = None,
        deformation_spline: _SplinepyBase = None,
        microtile: _SDFBase = None,
        parametrization: _Parametrization = None,
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
        if not isinstance(parametrization, _Parametrization):
            raise TypeError("Parametrization must be of type _Parametrization")
        super().__init__(
            deformation_spline=deformation_spline,
            parametrization=parametrization,
            cap_border_dict=cap_border_dict,
            cap_outside_of_unitcube=cap_outside_of_unitcube,
        )
        self.tiling = [tiling] * 3 if isinstance(tiling, int) else tiling
        self.microtile = microtile

    @property
    def deformation_spline(self):
        """Deformation function defining the outer geometry (contour) of the
        microstructure.

        Parameters
        ----------
        None

        Returns
        -------
        deformation_function : spline
        """
        if hasattr(self, "_deformation_spline"):
            return self._deformation_spline
        else:
            return None

    @deformation_spline.setter
    def deformation_spline(self, deformation_spline):
        """Deformation function setter defining the outer geometry of the
        microstructure. Must be spline type and as such inherit from
        splinepy.Spline.

        Parameters
        ----------
        deformation_function : spline

        Returns
        -------
        None
        """

        if not isinstance(deformation_spline, _SplinepyBase):
            raise ValueError(
                "Deformation spline must be splinepy-Spline." " e.g. splinepy.NURBS"
            )
        self._deformation_spline = deformation_spline
        self._sanity_check()

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
        # Is defaulted to False using function arguments
        self._sanity_check()
        logger.debug(f"Successfully set tiling to : {self.tiling}")

    @property
    def microtile(self):
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

    @microtile.setter
    def microtile(self, microtile):
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

        self._sanity_check()

    def _set_param(self, parameters):
        self.parameters = parameters

    def _get_domain_bounds(self):
        return _np.array([[-1, 1], [-1, 1], [-1, 1]])

    def _compute(self, samples: _torch.tensor):
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
        spline_domain_samples = _torch.clamp(samples, min=bounds[0], max=bounds[1])
        if self.parametrization is not None:
            parameters = self.parametrization(spline_domain_samples)
            self.microtile._set_param(parameters)

        queries_transformed = _torch.zeros_like(samples)
        tx, ty, tz = self._tiling
        queries_transformed[:, 0] = transform(samples[:, 0], tx)
        queries_transformed[:, 1] = transform(samples[:, 1], ty)
        queries_transformed[:, 2] = transform(samples[:, 2], tz)

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

    def plot_intermesh(self, verts: _torch.tensor, faces: _torch.tensor):
        gus_faces = gus.Faces(vertices=verts.cpu().detach(), faces=faces.cpu().detach())
        gus.show(gus_faces, axes=1)

    def plot_slice(self, *args, **kwargs):
        xmin = self._deformation_spline.control_points[:, 0].min()
        xmax = self._deformation_spline.control_points[:, 0].max()
        ymin = self._deformation_spline.control_points[:, 1].min()
        ymax = self._deformation_spline.control_points[:, 1].max()

        kwargs.setdefault("xlim", (xmin, xmax))
        kwargs.setdefault("ylim", (ymin, ymax))

        return super().plot_slice(*args, **kwargs)


def constantLatvec(value):
    return _BSpline([0, 0, 0], [[-1, 1], [-1, 1], [-1, 1]], [value])


def transform(x, t):
    # transform x from [0,1] to [-1,1]
    # x = (x + 1) / 2 # if this is enabled, transforms from [-1,1] to [1,1]
    return 4 * _torch.abs(t * x / 2 - _torch.floor((t * x + 1) / 2)) - 1


def check_tiling_input(tiling):
    if isinstance(tiling, list):
        if len(tiling) != 3:
            raise ValueError("Tiling must be a list of 3 integers")
        tiling = _np.array(tiling)
    elif isinstance(tiling, int):
        tiling = _np.array([tiling, tiling, tiling])
    else:
        raise ValueError("Tiling must be a list or an integer")


# def _prepare_flexicubes_querypoints(N):
#     """
#     takes the tiling and a resolution as input
#     output: DeepSDFStruct.flexicubes constructor, samples and cube indices
#             the points are located in the region [0,1] with a margin of 0.025
#             -> [-0.025, 1.025]
#     """
#     # check_tiling_input(tiling)

#     flexi_cubes_constructor = FlexiCubes(device=device)
#     samples, cube_idx = flexi_cubes_constructor.construct_voxel_grid(
#         resolution=tuple(N)
#     )

#     samples = samples * 1.1 + _torch.tensor([0.5, 0.5, 0.5], device=device)
#     tolerance = 1e-6
#     _torch._assert(
#         _torch.all(samples.ge(-0.05 - tolerance) & samples.le(1.05 + tolerance)),
#         "Samples are out of bounds",
#     )

#     return flexi_cubes_constructor, samples, cube_idx

#     samples = samples.to(device)
#     cube_idx = cube_idx.to(device)
#     # transform samples from [-0.5, 0.5] to [-1.05, 1.05]
#     N_tot = samples.shape[0]
#     N = N + 1

#     tx, ty, tz = tiling

#     samples_transformed = _torch.zeros(N_tot, 3)
#     samples_transformed[:, 0] = transform(samples[:, 0], tx)
#     samples_transformed[:, 1] = transform(samples[:, 1], ty)
#     samples_transformed[:, 2] = transform(samples[:, 2], tz)

#     samples_transformed.requires_grad = False

#     inside_domain = torch.where(
#         (samples[:, 0] >= -1)
#         & (samples[:, 0] <= 1)
#         & (samples[:, 1] >= -1)
#         & (samples[:, 1] <= 1)
#         & (samples[:, 2] >= -1)
#         & (samples[:, 2] <= 1)
#     )
#     return flexi_cubes_constructor, samples_transformed, samples
