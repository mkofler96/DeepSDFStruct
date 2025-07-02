import logging
from typing import TypedDict

import numpy as _np
import torch as _torch

from splinepy._base import SplinepyBase as _SplinepyBase
from splinepy import BSpline as _BSpline
from .SDF import SDFBase as _SDFBase
from .mesh import torchSurfMesh
from flexicubes.flexicubes import FlexiCubes
import gustaf as gus

logger = logging.getLogger(__name__)

device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")

# used to define the unit cube
location_lookup = {
    "x0": (0, 0),
    "x1": (0, 1),
    "y0": (1, 0),
    "y1": (1, 1),
    "z0": (2, 0),
    "z1": (2, 1),
}


class CapType(TypedDict):
    cap: int
    measure: float


class CapBorderDict(TypedDict):
    x0: CapType = {"cap": -1, "measure": 0}
    x1: CapType = {"cap": -1, "measure": 0}
    y0: CapType = {"cap": -1, "measure": 0}
    y1: CapType = {"cap": -1, "measure": 0}
    z0: CapType = {"cap": -1, "measure": 0}
    z1: CapType = {"cap": -1, "measure": 0}


class LatticeSDFStruct(_SDFBase):
    """Helper class to facilitatae the construction of Lattice SDF Structures."""

    def __init__(
        self,
        tiling: list[int] | int = None,
        deformation_spline: _SplinepyBase = None,
        microtile: _SDFBase = None,
        parametrization_spline: _SplinepyBase = None,
        cap_border_dict: CapBorderDict = None,
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
        if deformation_spline is not None:
            self._deformation_spline = deformation_spline

        if tiling is not None:
            self._tiling = tiling

        if microtile is not None:
            self._microtile = microtile

        if parametrization_spline is not None:
            self._parametrization_spline = parametrization_spline

        if cap_border_dict is not None:
            self._cap_border_dict = cap_border_dict
        else:
            self._cap_border_dict = {
                "x0": {"cap": 1, "measure": 0.02},
                "x1": {"cap": 1, "measure": 0.02},
                "y0": {"cap": 1, "measure": 0.02},
                "y1": {"cap": 1, "measure": 0.02},
                "z0": {"cap": 1, "measure": 0.02},
                "z1": {"cap": 1, "measure": 0.02},
            }

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
        if not isinstance(tiling, list) and not isinstance(tiling, int):
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

    @property
    def parametrization_spline(self):
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
        parametrization_function : Callable
          Function that describes the local tile parameters
        """
        if hasattr(self, "_parametrization_function"):
            return self._parametrization_spline
        else:
            return None

    @parametrization_spline.setter
    def parametrization_spline(self, parametrization_spline):
        if not isinstance(parametrization_spline, _SplinepyBase):
            raise ValueError(
                "Deformation spline must be splinepy-Spline." " e.g. splinepy.NURBS"
            )
        self._parametrization_spline = parametrization_spline
        self._sanity_check()

    def _set_param(self, parameters):
        pass

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
        spline_domain_samples = self._deformation_spline.proximities(
            samples.detach().cpu().numpy()
        )
        spline_domain_samples = _torch.tensor(
            spline_domain_samples, device=orig_device, dtype=orig_dtype
        )

        queries_transformed = _torch.zeros_like(spline_domain_samples)
        tx, ty, tz = self._tiling
        queries_transformed[:, 0] = transform(spline_domain_samples[:, 0], tx)
        queries_transformed[:, 1] = transform(spline_domain_samples[:, 1], ty)
        queries_transformed[:, 2] = transform(spline_domain_samples[:, 2], tz)
        if self._parametrization_spline is not None:
            parameters = self._parametrization_spline.evaluate(
                spline_domain_samples.detach().cpu().numpy()
            )
            parameters = _torch.tensor(parameters, device=orig_device, dtype=orig_dtype)
            self.microtile._set_param(parameters)
        sdf_values = self.microtile(queries_transformed)
        # self.plot_transformed_untransformed(queries, queries_transformed)
        for loc, cap_dict in self._cap_border_dict.items():
            cap, measure = cap_dict["cap"], cap_dict["measure"]
            dim, location = location_lookup[loc]
            if "0" in loc:
                multiplier = -1
            elif "1" in loc:
                multiplier = 1
            border_sdf = (
                spline_domain_samples[:, dim] - multiplier * (location - measure)
            ) * -multiplier
            # border_sdf = border_sdf.view(-1, 1)
            border_sdf = border_sdf.to(orig_device)
            sdf_values = sdf_values.to(orig_device)
            if cap == -1:
                # sdf_values = _torch.maximum(sdf_values, -border_sdf)

                sdf_values = _torch.maximum(sdf_values, -border_sdf)
            elif cap == 1:
                sdf_values = _torch.minimum(sdf_values, border_sdf)
            else:
                raise ValueError("Cap must be -1 or 1")

        # cap everything outside the unit cube
        # this is broken now, needs to be fixed
        # if cap dict is fully given, it does not make a difference
        for dim, measure, location in zip(
            [0, 0, 1, 1, 2, 2], [-1, 1, -1, 1, -1, 1], [0, 1, 0, 1, 0, 1]
        ):
            border_sdf = (spline_domain_samples[:, dim] - measure) * -measure
            # border_sdf = border_sdf.view(-1, 1)
            sdf_values = _torch.maximum(sdf_values, -border_sdf)

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

    def create_surface_mesh(self, N_base, differentiate=False):
        N = process_N_base_input(N_base, self.tiling)

        constructor, samples, cube_idx = _prepare_flexicubes_querypoints(N)

        sdf_values = self.evaluate_sdf(samples)
        verts, faces, _ = constructor(
            voxelgrid_vertices=samples,
            scalar_field=sdf_values,
            cube_idx=cube_idx,
            resolution=tuple(N),
            output_tetmesh=False,
        )

        # self.plot_samples(samples, sdf_values)
        # self.plot_intermesh(verts, faces)

        prev_device = verts.device
        ffd_vertices = self.deformation_spline.evaluate(verts.detach().cpu().numpy())
        ffd_vertices_torch = _torch.tensor(ffd_vertices, device=prev_device)
        if differentiate:
            raise NotImplementedError("Differentiable version not implemented yet.")
            # spline_jac = self.macro_spline.jacobian(vertices)
            # ffd_jac = np.matmul(spline_jac, jacobian)

        return torchSurfMesh(ffd_vertices_torch, faces)

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


def process_N_base_input(N, tiling):
    if isinstance(N, list):
        if len(N) != 3:
            raise ValueError("Number of grid points must be a list of 3 integers")
        N = _torch.tensor(N)
    elif isinstance(N, int):
        N = _torch.tensor([N, N, N])
    else:
        raise ValueError("Number of grid points must be a list or an integer")
    # add 1 on each side to slightly include the border
    N_mod = N * _torch.tensor(tiling) + 1
    return N_mod


def check_tiling_input(tiling):
    if isinstance(tiling, list):
        if len(tiling) != 3:
            raise ValueError("Tiling must be a list of 3 integers")
        tiling = _np.array(tiling)
    elif isinstance(tiling, int):
        tiling = _np.array([tiling, tiling, tiling])
    else:
        raise ValueError("Tiling must be a list or an integer")


def _prepare_flexicubes_querypoints(N):
    """
    takes the tiling and a resolution as input
    output: flexicubes constructor, samples and cube indices
            the points are located in the region [0,1] with a margin of 0.025
            -> [-0.025, 1.025]
    """
    # check_tiling_input(tiling)

    flexi_cubes_constructor = FlexiCubes(device=device)
    samples, cube_idx = flexi_cubes_constructor.construct_voxel_grid(
        resolution=tuple(N)
    )

    samples = samples * 1.1 + _torch.tensor([0.5, 0.5, 0.5], device=device)
    tolerance = 1e-6
    _torch._assert(
        _torch.all(samples.ge(-0.05 - tolerance) & samples.le(1.05 + tolerance)),
        "Samples are out of bounds",
    )
    return flexi_cubes_constructor, samples, cube_idx

    samples = samples.to(device)
    cube_idx = cube_idx.to(device)
    # transform samples from [-0.5, 0.5] to [-1.05, 1.05]
    N_tot = samples.shape[0]
    N = N + 1

    tx, ty, tz = tiling

    samples_transformed = _torch.zeros(N_tot, 3)
    samples_transformed[:, 0] = transform(samples[:, 0], tx)
    samples_transformed[:, 1] = transform(samples[:, 1], ty)
    samples_transformed[:, 2] = transform(samples[:, 2], tz)

    samples_transformed.requires_grad = False

    inside_domain = torch.where(
        (samples[:, 0] >= -1)
        & (samples[:, 0] <= 1)
        & (samples[:, 1] >= -1)
        & (samples[:, 1] <= 1)
        & (samples[:, 2] >= -1)
        & (samples[:, 2] <= 1)
    )
    return flexi_cubes_constructor, samples_transformed, samples
