"""
Signed Distance Function (SDF) Base Classes and Operations
==========================================================

This module provides the foundational classes and utilities for working with
Signed Distance Functions (SDFs) in DeepSDFStruct. SDFs are implicit geometric
representations that encode the distance from any point in space to the nearest
surface, with the sign indicating whether the point is inside (negative) or
outside (positive) the geometry.

Key Features
------------

SDFBase Abstract Class
    Base class for all SDF representations with support for:
    - Spline-based geometric deformations
    - Spatially-varying parametrization
    - Boundary conditions and capping
    - Boolean operations (union, intersection)
    - Differentiable operations for optimization

SDFfromMesh
    Convert triangular surface meshes to SDF representations using
    fast winding number algorithms for robust inside/outside testing.

SDFfromDeepSDF
    Neural network-based SDF using trained DeepSDF models for
    complex, learned geometric representations.

Union and Intersection
    Combine multiple SDFs using smooth boolean operations with
    configurable smoothing for differentiable geometry.

Utility Functions
    - Grid sampling for SDF evaluation
    - Gradient computation for normal vectors
    - Boundary condition application

The module enables flexible construction and manipulation of complex
3D geometries in a differentiable framework suitable for optimization,
simulation, and machine learning applications.

Examples
--------
Create and evaluate an SDF from a mesh::

    import trimesh
    from DeepSDFStruct.SDF import SDFfromMesh

    mesh = trimesh.load('model.stl')
    sdf = SDFfromMesh(mesh)

    # Query SDF values
    points = torch.rand(1000, 3)
    distances = sdf(points)

Combine SDFs with boolean operations::

    from DeepSDFStruct.sdf_primitives import SphereSDF
    from DeepSDFStruct.SDF import Union

    sphere1 = SphereSDF([0, 0, 0], radius=1.0)
    sphere2 = SphereSDF([1, 0, 0], radius=1.0)
    combined = Union([sphere1, sphere2], smoothing=0.1)
"""

from abc import ABC, abstractmethod
import torch
import numpy as np
import igl
import trimesh
import gustaf


from typing import TypedDict
from DeepSDFStruct.deep_sdf.models import DeepSDFModel
from DeepSDFStruct.parametrization import Constant
import DeepSDFStruct

import logging

import matplotlib.pyplot as plt
import matplotlib.tri as mtri

logger = logging.getLogger(DeepSDFStruct.__name__)


class CapType(TypedDict):
    cap: int
    measure: float


class CapBorderDict(TypedDict):
    """
    A dictionary type describing boundary conditions ("caps")
    for each axis direction (x, y, z).

    Each key (`x0`, `x1`, `y0`, `y1`, `z0`, `z1`) corresponds to
    one boundary face of a 3D domain, and maps to a dictionary
    with two fields:

    - `cap` (int): Type of cap applied (e.g., -1 = none, 1 = active).
    - `measure` (float): Numerical measure associated with the cap
      (e.g., thickness, scaling factor, tolerance).

    Example
    -------
    >>> caps: CapBorderDict = {
    ...     "x0": {"cap": 1, "measure": 0.02},
    ...     "x1": {"cap": 1, "measure": 0.02},
    ...     "y0": {"cap": 1, "measure": 0.02},
    ...     "y1": {"cap": 1, "measure": 0.02},
    ...     "z0": {"cap": 1, "measure": 0.02},
    ...     "z1": {"cap": 1, "measure": 0.02},
    ... }
    """

    x0: CapType = {"cap": -1, "measure": 0}
    x1: CapType = {"cap": -1, "measure": 0}
    y0: CapType = {"cap": -1, "measure": 0}
    y1: CapType = {"cap": -1, "measure": 0}
    z0: CapType = {"cap": -1, "measure": 0}
    z1: CapType = {"cap": -1, "measure": 0}


UNIT_CUBE_CAPS_2D: CapBorderDict = {
    "x0": {"cap": -1, "measure": 0},
    "x1": {"cap": -1, "measure": 0},
    "y0": {"cap": -1, "measure": 0},
    "y1": {"cap": -1, "measure": 0},
}
UNIT_CUBE_CAPS_3D: CapBorderDict = {
    "x0": {"cap": -1, "measure": 0},
    "x1": {"cap": -1, "measure": 0},
    "y0": {"cap": -1, "measure": 0},
    "y1": {"cap": -1, "measure": 0},
    "z0": {"cap": -1, "measure": 0},
    "z1": {"cap": -1, "measure": 0},
}


def get_equidistant_grid_sample(
    bounds: torch.Tensor | np.ndarray,
    grid_spacing: float,
    dtype=torch.float32,
    device="cpu",
) -> torch.Tensor:
    """
    Generates an equidistant 3D grid of points within the given bounding box.

    Parameters
    ----------
    bounds : torch.Tensor
        Tensor of shape (2,3), [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    grid_spacing : float
        Approximate spacing between points along each axis.

    Returns
    -------
    points : torch.Tensor
        Tensor of shape (N,3) containing all grid points.
    """
    if isinstance(bounds, np.ndarray):
        bounds = torch.tensor(bounds, dtype=dtype, device=device)
    assert bounds.shape == (2, 3), "Bounds should be of shape (2,3)"
    mins, maxs = bounds[0], bounds[1]

    # Compute number of points along each axis (ceil to include max)
    num_points = torch.ceil((maxs - mins) / grid_spacing).to(torch.int64) + 1

    # Generate linspace for each axis
    xs = torch.linspace(mins[0], maxs[0], num_points[0], dtype=dtype, device=device)
    ys = torch.linspace(mins[1], maxs[1], num_points[1], dtype=dtype, device=device)
    zs = torch.linspace(mins[2], maxs[2], num_points[2], dtype=dtype, device=device)

    # Generate full 3D grid
    grid = torch.meshgrid(xs, ys, zs, indexing="ij")
    points = torch.stack(grid, dim=-1).reshape(-1, 3)

    # Assertions to verify grid covers the bounds
    tol = 1e-6
    mins_generated = points.min(dim=0).values
    maxs_generated = points.max(dim=0).values
    assert torch.allclose(
        mins_generated, bounds[0], atol=tol
    ), f"Grid min {mins_generated} does not match bounds {bounds[0]}"
    assert torch.allclose(
        maxs_generated, bounds[1], atol=tol
    ), f"Grid max {maxs_generated} does not match bounds {bounds[1]}"

    return points


class SDFBase(torch.nn.Module, ABC):
    """Abstract base class for Signed Distance Functions with optional
    deformation and parametrization.

    This class provides the foundation for all SDF representations in
    DeepSDFStruct. SDFs represent geometry as an implicit function that
    returns the signed distance from any query point to the nearest
    surface. Negative values indicate points inside the geometry,
    positive values indicate points outside, and zero indicates points
    on the surface.

    The class supports:
    - Optional spline-based deformations for smooth transformations
    - Parametrization functions for spatially-varying properties
    - Composition operations (union, intersection) via overloading

    Parameters
    ----------
    parametrization : torch.nn.Module, optional
        A function that provides spatially-varying parameters for the
        SDF (e.g., varying thickness in a lattice).
    geometric_dim : int, default 3
        Geometric dimension of the SDF (2 or 3).

    Notes
    -----
    Subclasses must implement:
    - ``_compute(queries)``: Calculate SDF values for query points
    - ``_get_domain_bounds()``: Return the bounding box of geometry

    Examples
    --------
    >>> from DeepSDFStruct.sdf_primitives import SphereSDF
    >>> import torch
    >>>
    >>> # Create a sphere SDF
    >>> sphere = SphereSDF(center=[0, 0, 0], radius=1.0)
    >>>
    >>> # Query SDF values
    >>> points = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    >>> distances = sphere(points)
    >>> print(distances)  # [-1.0, 1.0] (inside, outside)
    """

    geometric_dim: int

    def __init__(self, parametrization: torch.nn.Module | None = None, geometric_dim=3):
        super().__init__()
        self.parametrization = parametrization
        self.geometric_dim = geometric_dim

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        """Evaluate the SDF at given query points.

        This method validates input, computes SDF values using the subclass
        implementation, and applies optional boundary conditions and capping.

        Parameters
        ----------
        queries : torch.Tensor
            Query points of shape (N, 2) for 2D or (N, 3) for 3D, where N is
            the number of points to evaluate.

        Returns
        -------
        torch.Tensor
            Signed distance values of shape (N, 1). Negative values indicate
            points inside the geometry, positive values outside.

        Raises
        ------
        ValueError
            If queries have invalid shape.
        RuntimeError
            If SDF computation returns invalid output.
        """
        self._validate_input(queries)
        sdf_values = self._compute(queries)
        if sdf_values is None:
            raise RuntimeError("Invalid SDF output")
        return sdf_values

    def _validate_input(self, queries: torch.Tensor):
        # Example check: 2D tensor with shape (N, 3) or (N, 2where N points, each point is a column vector
        if queries.ndim != 2 or (queries.shape[1] not in [2, 3]):
            raise ValueError(
                f"Expected input of shape (N, 3) or (N, 2), got {queries.shape}"
            )

    def get_device(self):
        """
        Return the device of the first parameter or buffer in this module.

        If the module has no parameters and no buffers, returns cpu.
        """
        # Check parameters first
        for p in self.parameters(recurse=True):
            return p.device

        # If no parameters, check buffers
        for b in self.buffers(recurse=True):
            return b.device

        # No tensors at all
        return "cpu"

    def get_dtype(self):
        """
        Return the dtype of the first parameter or buffer in this module.

        If the module has no parameters and no buffers, returns float32.
        """
        # Check parameters first
        for p in self.parameters(recurse=True):
            return p.dtype

        # If no parameters, check buffers
        for b in self.buffers(recurse=True):
            return b.dtype

        # No tensors at all
        return torch.float32

    @abstractmethod
    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        """Compute SDF values for query points.

        Subclasses must implement this method to define their specific
        geometry. The method should return signed distances without
        applying any boundary conditions or capping (those are handled
        by __call__).

        Parameters
        ----------
        queries : torch.Tensor
            Query points of shape (N, dim) where dim is 2 or 3.

        Returns
        -------
        torch.Tensor
            Signed distance values of shape (N, 1).
        """
        pass

    @abstractmethod
    def _get_domain_bounds(self) -> torch.Tensor:
        """Return the bounding box of the SDF's domain.

        Subclasses must implement this to specify the spatial extent
        of their geometry. This is used for mesh generation and sampling.

        Returns
        -------
        np.ndarray
            Array of shape (2, dim) where the first row contains minimum
            coordinates and the second row contains maximum coordinates.
        """
        pass

    def plot_slice(
        self,
        origin=(0, 0, 0),
        normal=(0, 0, 1),
        res=(100, 100),
        ax=None,
        clim=(-1, 1),
        cmap="seismic",
        show_zero_level=True,
        deformation_function=None,
    ):
        """Plot a 2D slice through an SDF as a contour plot.

        This function evaluates an SDF on a planar grid and visualizes the
        signed distance values using a color map. The zero level set (the
        actual surface) can be highlighted with a contour line.

        Parameters
        ----------
        fun : callable
            The SDF function to visualize. Should accept a torch.Tensor
            of shape (N, 3) and return distances of shape (N, 1).
        origin : tuple of float, default (0, 0, 0)
            A point on the slice plane.
        normal : tuple of float, default (0, 0, 1)
            Normal vector of the slice plane. Currently supports only
            axis-aligned planes: (1,0,0), (0,1,0), or (0,0,1).
        res : tuple of int, default (100, 100)
            Resolution of the slice grid (num_points_u, num_points_v).
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure.
        xlim : tuple of float, default (-1, 1)
            Range along the first plane axis.
        ylim : tuple of float, default (-1, 1)
            Range along the second plane axis.
        clim : tuple of float, default (-1, 1)
            Color map limits for distance values.
        cmap : str, default 'seismic'
            Matplotlib colormap name.
        show_zero_level : bool, default True
            If True, draws a black contour line at distance=0 (the surface).
        deformation_function : callable, optional
            Deformation mapping from parametric to physical space. If given,
            sample points are deformed before SDF evaluation and plotting.

        Returns
        -------
        fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
            Only returned if ax was None (i.e., a new figure was created).

        Examples
        --------
        >>> from DeepSDFStruct.sdf_primitives import SphereSDF
        >>> from DeepSDFStruct.plotting import plot_slice
        >>> import matplotlib.pyplot as plt
        >>>
        >>> # Create a sphere
        >>> sphere = SphereSDF(center=[0, 0, 0], radius=0.5)
        >>>
        >>> # Plot XY slice at z=0
        >>> fig, ax = plot_slice(
        ...     sphere,
        ...     origin=(0, 0, 0),
        ...     normal=(0, 0, 1),
        ...     res=(200, 200)
        ... )
        >>> plt.title("XY Slice of Sphere")
        >>> plt.show()

        Notes
        -----
        The 'seismic' colormap is well-suited for SDFs as it uses blue for
        negative (inside) and red for positive (outside), with white near zero.
        """
        plt_show = False
        if ax is None:
            fig, ax = plt.subplots()
            plt_show = True
        bounds = self._get_domain_bounds()
        xlim, ylim = project_bounds(origin, normal, bounds=bounds)
        points = generate_plane_points(origin, normal, res, xlim, ylim)

        sdf_device = self.get_device()
        points = torch.from_numpy(points).to(torch.float32).to(sdf_device)

        if deformation_function is not None:
            points_deformed = deformation_function.forward(points)
        else:
            points_deformed = points

        sdf_values = self._compute(points).reshape(-1).detach().cpu().numpy()
        points_np = points_deformed.detach().cpu().numpy()
        axis0, axis1 = _get_plane_plot_axes(normal)
        x_plot = points_np[:, axis0]
        y_plot = points_np[:, axis1]
        triangles = _build_structured_grid_triangles(res[0], res[1])
        triangulation = mtri.Triangulation(x_plot, y_plot, triangles=triangles)

        cbar = ax.tricontourf(triangulation, sdf_values, cmap=cmap, levels=10)
        if show_zero_level:
            ax.tricontour(
                triangulation, sdf_values, levels=[0], colors="black", linewidths=0.5
            )
        cbar.set_clim(clim[0], clim[1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect(1)
        if plt_show:
            plt.show()
            return fig, ax

    def __add__(self, other):
        return UnionSDF(self, other)

    def to2D(self, axes: list[int], offset=0.0):
        """
        Converts SDF to 2D

        :param axis: list of axes that will be used for the 2D
        """
        sdf2D = SDF2D(self, axes, offset=offset)
        sdf2D.parametrization = self.parametrization
        return sdf2D


class SDF2D(SDFBase):
    obj: SDFBase

    def __init__(self, obj: SDFBase, axes: list[int], offset=0.0):
        super().__init__()
        self.obj = obj
        assert (
            len(axes) == 2
        ), "List of axes must be of size 2 and needs to correspond to the 2D plane"
        self.axes = axes
        self.offset = offset
        self.geometric_dim = 2

    def _compute(self, queries):
        queries_3D = (
            torch.zeros(
                (queries.shape[0], 3), dtype=queries.dtype, device=queries.device
            )
            + self.offset
        )
        queries_3D[:, self.axes[0]] = queries[:, 0]
        queries_3D[:, self.axes[1]] = queries[:, 1]
        result = self.obj._compute(queries_3D)
        return result

    def _get_domain_bounds(self):
        bounds_3d = self.obj._get_domain_bounds()
        axes = torch.as_tensor(self.axes, device=bounds_3d.device)
        return torch.index_select(bounds_3d, dim=1, index=axes)

    def _set_param(self, parameter):
        return self.obj._set_param(parameter)


class SummedSDF(SDFBase):
    def __init__(self, obj1: SDFBase, obj2: SDFBase):
        raise NotImplementedError("SummedSDF has been replaced by UnionSDF")


class UnionSDF(SDFBase):
    def __init__(self, *objects: SDFBase):
        super().__init__()

        if len(objects) < 2:
            raise ValueError("UnionSDF requires at least two objects.")

        self.objects = list(objects)

        # Check geometric dimensions match
        geometric_dim = self.objects[0].geometric_dim
        for i, obj in enumerate(self.objects[1:], start=1):
            if obj.geometric_dim != geometric_dim:
                raise ValueError(
                    f"geometric dim mismatch between object 0 ({geometric_dim}) "
                    f"and object {i} ({obj.geometric_dim})"
                )

        self.geometric_dim = geometric_dim

    def _compute(self, queries):
        # Compute first object
        result = self.objects[0]._compute(queries)

        # Iteratively take minimum with the rest
        for obj in self.objects[1:]:
            result = torch.minimum(result, obj._compute(queries))

        return result

    def _get_domain_bounds(self):
        # Initialize with first object's bounds
        bounds = self.objects[0]._get_domain_bounds()
        lower = bounds[0]
        upper = bounds[1]

        # Expand bounds across all objects
        for obj in self.objects[1:]:
            print(f"current bounds of union lower: {lower} upper: {upper}")
            obj_bounds = obj._get_domain_bounds()
            lower = torch.minimum(lower, obj_bounds[0])
            upper = torch.maximum(upper, obj_bounds[1])

        return torch.stack([lower, upper], dim=0)

    def _set_param(self, parameter):
        return None


class DifferenceSDF(SDFBase):
    """
    Subtracts multiple objects from a base object.

    Computes:
        obj0 - (obj1 ∪ obj2 ∪ ...)

    i.e.
        max(d0, -min(d1, d2, ...))
    """

    def __init__(self, base_obj: SDFBase, *subtract_objs: SDFBase):
        super().__init__()

        if len(subtract_objs) == 0:
            raise ValueError("DifferenceSDF requires at least one object to subtract.")

        self.base_obj = base_obj
        self.subtract_objs = list(subtract_objs)

        geometric_dim = base_obj.geometric_dim

        for i, obj in enumerate(self.subtract_objs):
            if obj.geometric_dim != geometric_dim:
                raise ValueError(
                    f"Geometric dimension mismatch between base "
                    f"({geometric_dim}) and subtract object {i} "
                    f"({obj.geometric_dim})"
                )

        self.geometric_dim = geometric_dim

    def _compute(self, queries):
        d_base = self.base_obj._compute(queries)

        # Compute union of subtraction objects
        d_sub = self.subtract_objs[0]._compute(queries)
        for obj in self.subtract_objs[1:]:
            d_sub = torch.minimum(d_sub, obj._compute(queries))

        return torch.maximum(d_base, -d_sub)

    def _get_domain_bounds(self):
        # Difference cannot expand beyond base object
        return self.base_obj._get_domain_bounds()

    def _set_param(self, parameter):
        return None


class NegatedCallable(SDFBase):
    def __init__(self, obj: SDFBase):
        super().__init__()
        self.obj = obj

    def _compute(self, input_param):
        result = self.obj(input_param)
        return -result

    def _get_domain_bounds(self):
        # the domain bounds get smaller when we substract something
        return self.obj._get_domain_bounds()


class BoxSDF(SDFBase):
    def __init__(
        self, box_size: float = 1, center: torch.tensor = torch.tensor([0, 0, 0])
    ):
        super().__init__()
        self.box_size = box_size
        self.center = center

    def _compute(self, queries: torch.tensor) -> torch.tensor:
        output = (
            torch.linalg.norm(queries - self.center, axis=1, ord=torch.inf)
            - self.box_size
        )
        return output.reshape(-1, 1)


def union_torch(D, k=0):
    """
    D: np.array of shape (num_points, num_geometries)
    k: smoothness parameter
    """
    if k == 0:
        return torch.min(D, axis=1)[0].view(-1, 1)
    # Start with the first column as d1
    d1 = D[:, 0].copy()

    # Loop over remaining columns
    for i in range(1, D.shape[1]):
        d2 = D[:, i]
        h = torch.clip(0.5 + 0.5 * (d2 - d1) / k, 0, 1)
        d1 = d2 + (d1 - d2) * h - k * h * (1 - h)

    return d1.view(-1, 1)


def union_numpy(D, k=0):
    """
    D: np.array of shape (num_points, num_geometries)
    k: smoothness parameter
    """
    if k == 0:
        return np.min(D, axis=1)
    # Start with the first column as d1
    d1 = D[:, 0].copy()

    # Loop over remaining columns
    for i in range(1, D.shape[1]):
        d2 = D[:, i]
        h = np.clip(0.5 + 0.5 * (d2 - d1) / k, 0, 1)
        d1 = d2 + (d1 - d2) * h - k * h * (1 - h)

    return d1


class SDFfromMesh(SDFBase):
    """Create an SDF from a triangle mesh using closest-point queries.

    This class wraps a triangle mesh and computes signed distances by finding
    the closest point on the mesh surface to each query point. The sign is
    determined using winding number or ray casting to determine inside/outside.

    The mesh can be optionally normalized to fit within a unit cube centered
    at the origin, which is useful for consistent scaling across different
    geometries.

    Parameters
    ----------
    mesh : trimesh.Trimesh or gustaf.faces.Faces
        The input triangle mesh. If a gustaf Faces object is provided,
        it will be converted to a trimesh object.
    dtype : numpy dtype, default np.float32
        Data type for distance calculations.
    flip_sign : bool, default False
        If True, flips the sign of the computed distances (inside becomes
        outside and vice versa).
    scale : bool, default True
        If True, normalizes the mesh to fit within a unit cube [-1, 1]^3
        centered at the origin.
    threshold : float, default 1e-5
        Small threshold value for numerical stability in distance computations.

    Attributes
    ----------
    mesh : trimesh.Trimesh
        The (possibly normalized) triangle mesh.
    dtype : numpy dtype
        Data type used for calculations.
    flip_sign : bool
        Whether distances are sign-flipped.
    threshold : float
        Numerical threshold for stability.

    Notes
    -----
    This class uses libigl for efficient closest-point queries and embree
    for ray intersection tests to determine inside/outside status.

    Examples
    --------
    >>> import trimesh
    >>> from DeepSDFStruct.SDF import SDFfromMesh
    >>> import torch
    >>>
    >>> # Load or create a mesh
    >>> mesh = trimesh.creation.box(extents=[1, 1, 1])
    >>>
    >>> # Create SDF from mesh
    >>> sdf = SDFfromMesh(mesh, scale=True)
    >>>
    >>> # Query distances
    >>> points = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    >>> distances = sdf(points)
    """

    def __init__(
        self,
        mesh,
        dtype=np.float32,
        flip_sign=False,
        scale=True,
        threshold=1e-5,
        backend="igl",
    ):
        super().__init__()
        if type(mesh) is gustaf.faces.Faces:
            mesh = trimesh.Trimesh(mesh.vertices, mesh.faces)

        if scale:
            # scales from [0,1] to [-1,1]
            # https://www.brainvoyager.com/bv/doc/UsersGuide/CoordsAndTransforms/SpatialTransformationMatrices.html
            mesh, _, _ = normalize_mesh_to_unit_cube(mesh)
        self.mesh = mesh
        self.dtype = dtype
        self.flip_sign = flip_sign
        self.threshold = threshold
        self.backend = backend

    def _set_param(self, mesh):
        self.mesh = mesh

    def _get_domain_bounds(self):
        return self.mesh.bounds

    def _compute(self, queries: torch.Tensor | np.ndarray):
        is_tensor = isinstance(queries, torch.Tensor)

        if is_tensor:
            orig_device = queries.device
            orig_dtype = queries.dtype
            queries_np = queries.detach().cpu().numpy()
        else:
            queries_np = np.asarray(queries)
            orig_device = None  # No device for numpy input

        if self.backend == "trimesh":
            # Compute squared distance
            squared_distance, hit_index, hit_coordinates = (
                igl.point_mesh_squared_distance(
                    queries_np,
                    self.mesh.vertices,
                    np.array(self.mesh.faces, dtype=np.int32),
                )
            )
            distances = np.sqrt(squared_distance, dtype=self.dtype)

            # Determine sign (negative if inside)
            contains = trimesh.ray.ray_pyembree.RayMeshIntersector(
                self.mesh, scale_to_box=False
            ).contains_points(queries_np)

            distances[contains] *= -1.0
        elif self.backend == "igl":
            distances, _, _, _ = igl.signed_distance(
                queries_np,
                self.mesh.vertices,
                np.array(self.mesh.faces, dtype=np.int32),
            )
        # Apply threshold
        distances -= self.threshold

        result = distances.reshape(-1, 1)

        if is_tensor:
            return torch.tensor(result, device=orig_device, dtype=orig_dtype)
        else:
            return result


def normalize_mesh_to_unit_cube(mesh: trimesh.Trimesh, shrink_factor: float = 1.0):
    """
    Transform mesh coordinates uniformly to [-1, 1] in all axes.
    Keeps aspect ratio of original mesh.

    shrink_factor : float
        Uniform scaling factor applied after normalization.
        1.0  -> exactly fills [-1, 1]
        0.95 -> 5% smaller

    :return: The fitted mesh, the inverse scaling factor applied (can directly be used as input for the TorchScaling), and the translation vector used.
    """
    logger.debug(f"Scaling mesh from {mesh.bounds.flatten()}")
    # --- Compute bounding box ---
    bbox_min = mesh.bounds[0]  # [x_min, y_min, z_min]
    bbox_max = mesh.bounds[1]  # [x_max, y_max, z_max]

    # Center of the mesh
    center = (bbox_max + bbox_min) / 2.0

    # Largest extent
    base_scale = (
        np.max(bbox_max - bbox_min) / 2.0
    )  # divide by 2 because [-1,1] spans 2 units

    # --- Build transformation matrix ---
    matrix = np.eye(4)

    # Translate to origin
    matrix[:3, 3] = -center

    # Apply translation
    mesh.apply_transform(matrix)

    # --- Apply uniform scaling ---
    applied_scale = shrink_factor / base_scale
    scale_matrix = np.eye(4)
    scale_matrix[:3, :3] *= applied_scale
    mesh.apply_transform(scale_matrix)
    logger.debug(f"to {mesh.bounds.flatten()}")
    return mesh, 1.0 / applied_scale, center


class SDFfromLineMesh(SDFBase):
    line_mesh: gustaf.Edges

    def __init__(self, line_mesh: gustaf.Edges, thickness, smoothness=0):
        """
        takes a line mesh and the thickness of the lines as inputs and
        generates a SDF from it
        for now only supports lines in 2D
        """
        super().__init__()
        self.line_mesh = line_mesh
        self.t = thickness
        self.smoothness = smoothness
        self.geometric_dim = line_mesh.vertices.shape[1]

    def _get_domain_bounds(self):
        return self.line_mesh.bounds()

    def _set_param(self, parameters):
        self.t = parameters[0]
        self.smoothness = parameters[1]

    def _compute(self, queries: torch.Tensor | np.ndarray):
        is_tensor = isinstance(queries, torch.Tensor)
        if is_tensor:
            orig_device = queries.device
            orig_dtype = queries.dtype
            queries_np = queries.detach().cpu().numpy()
        else:
            queries_np = np.asarray(queries)
            orig_device = None  # No device for numpy input
        lines = self.line_mesh.vertices[self.line_mesh.edges]
        sdf = point_segment_distance(lines[:, 0], lines[:, 1], queries_np) - self.t / 2
        if is_tensor:
            sdf = torch.tensor(sdf, dtype=orig_dtype, device=orig_device)
            return union_torch(sdf, k=self.smoothness)
        else:
            return union_numpy(sdf, k=self.smoothness)


class SDFfromDeepSDF(SDFBase):
    def __init__(self, model: DeepSDFModel, max_batch=32**3):
        super().__init__()
        self.model = model
        self.latvec = None
        self.parametrization = None
        self.max_batch = max_batch
        self.set_latent_vec(model._trained_latent_vectors[0])
        self.geometric_dim = model._decoder.geom_dimension

    def set_latent_vec(self, latent_vec: torch.Tensor):
        """
        Set conditioning parameters for the model (e.g., latent code).
        """
        # if dimension is 1, set parametrization
        # if dimnsion is equal to the number of query points, directly set latvec
        if latent_vec.ndim == 1:
            self.parametrization = Constant(latent_vec, device=self.model.device)
        elif latent_vec.ndim == 2:
            assert (
                latent_vec.shape[1] == self.model._trained_latent_vectors[0].shape[0]
            ), "Learned latent shape and assigned latent shape mismatch"
            self.latvec = latent_vec  # (n_query_points, latent_dim)
        else:
            raise ValueError(
                f"Expected latent_vec to have 1 or 2 dimensions, got shape {latent_vec.shape}"
            )

    def _get_domain_bounds(self):
        return torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])

    def _set_param(self, parameters):
        return self.set_latent_vec(parameters)

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        # DeepSDF queries range from -1 to 1
        orig_device = queries.device
        queries = queries.to(self.get_device())
        n_queries = queries.shape[0]

        sdf_values = torch.zeros(n_queries, device=self.get_device())

        head = 0
        if self.latvec is None:
            latvec = self.parametrization(queries).to(self.get_device())
        else:
            latent_dim = self.model._trained_latent_vectors[0].shape[0]
            num_samples = queries.shape[0]
            if self.latvec.ndim == 1:
                if self.latvec.shape[0] != latent_dim:
                    raise ValueError(
                        f"Latent vector shape mismatch: {self.latvec.shape} does"
                        f"not align with latent dimension {latent_dim}."
                    )
                latvec = self.latvec.expand(-1, num_samples).T
            elif self.latvec.ndim == 2:
                if (self.latvec.shape[0] != num_samples) or (
                    self.latvec.shape[1] != latent_dim
                ):
                    raise ValueError(
                        f"Latent vector shape mismatch: {self.latvec.shape} does"
                        f" not align with {num_samples} queries."
                        f" Must be of shape ({num_samples}, {latent_dim})"
                    )
                latvec = self.latvec

        while head < n_queries:
            end = min(head + self.max_batch, n_queries)
            query_batch = queries[head:end]
            sdf_values[head:end] = (
                self.model._decode_sdf(latvec[head:end], query_batch).squeeze(1)
                # .detach()
            )
            head = end

        return sdf_values.to(orig_device).reshape(-1, 1)


def _cap_outside_of_unitcube(samples, sdf_values, max_dim=3):

    for dim in range(max_dim):
        border_sdf = samples[:, dim]
        sdf_values = torch.maximum(sdf_values, -border_sdf.reshape(-1, 1))
        border_sdf = 1 - samples[:, dim]
        sdf_values = torch.maximum(sdf_values, -border_sdf.reshape(-1, 1))
    return sdf_values


def point_segment_distance(P1, P2, query_points):
    """
    Calculates the minimum distance from one or more query points
    to one or more line segments defined by endpoints P1 and P2.

    Args:
        P1 (np.ndarray): Array of shape (M, 2) or (2,) representing
            first endpoints of segments.
        P2 (np.ndarray): Array of shape (M, 2) or (2,) representing
            second endpoints of segments.
        query_points (np.ndarray): Array of shape (N, 2) or (2,)
            representing query point(s).

    Returns:
        np.ndarray: Array of shape (N,) with the minimum distance from
            each query point to the closest segment.
    """
    P1 = np.atleast_2d(P1)  # (M, 2)
    P2 = np.atleast_2d(P2)  # (M, 2)
    Q = np.atleast_2d(query_points)  # (N, 2)

    # Handle degenerate case: one segment only
    # if P1.shape[0] == 1 and P2.shape[0] == 1:
    #     P1 = np.repeat(P1, Q.shape[0], axis=0)
    #     P2 = np.repeat(P2, Q.shape[0], axis=0)

    v = P2 - P1  # (M, 2) segment vectors
    w = Q[:, None, :] - P1[None, :, :]  # (N, M, 2): vector from P1 to each query point

    seg_len2 = np.sum(v**2, axis=1)  # (M,) squared lengths

    # Avoid division by zero (degenerate segments)
    seg_len2_safe = np.where(seg_len2 == 0, 1, seg_len2)

    # Projection factor t for each (Q, segment)
    t = np.einsum("nmd,md->nm", w, v) / seg_len2_safe  # (N, M)
    t = np.clip(t, 0, 1)  # clamp to segment

    # Closest point on segment
    projection = P1[None, :, :] + t[..., None] * v[None, :, :]  # (N, M, 2)

    # Distances
    distances = np.linalg.norm(Q[:, None, :] - projection, axis=2)  # (N, M)

    # For each query point, take the closest segment
    return distances


# used to define the unit cube
location_lookup = {
    "x0": (0, 0),
    "x1": (0, 1),
    "y0": (1, 0),
    "y1": (1, 1),
    "z0": (2, 0),
    "z1": (2, 1),
}


class TransformedSDF(SDFBase):
    """
    Generic SDF wrapper that applies a transformation to the input queries.
    Transformation can be rotation, translation, or scaling.
    """

    def __init__(
        self, sdf: SDFBase, rotationMatrix=None, translation=None, scaleFactor=None
    ):
        super().__init__()
        self.sdf = sdf
        if (rotationMatrix is None) or (rotationMatrix == [0, 0, 0]):
            rotationMatrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        if translation is None:
            translation = [0, 0, 0]
        if scaleFactor is None:
            scaleFactor = 1
        r = torch.as_tensor(rotationMatrix, dtype=torch.float32)
        s = torch.as_tensor([scaleFactor], dtype=torch.float32)
        t = torch.as_tensor(translation, dtype=torch.float32)
        self.translation = torch.nn.Parameter(t.reshape(1, 3))
        self.rotationMatrix = torch.nn.Parameter(r.reshape(3, 3))
        self.scale = torch.nn.Parameter(s)

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        xyz = queries

        # apply scale, for now, only uniform scale is allowd
        xyz = xyz / self.scale  # inverse scale

        # apply rotation
        xyz = xyz @ self.rotationMatrix.T  # rotate points

        # apply translation
        xyz = xyz - self.translation

        sdf_vals = self.sdf._compute(xyz)

        # rescale distances if scaled
        sdf_vals = sdf_vals * self.scale
        return sdf_vals

    def _get_domain_bounds(self) -> torch.Tensor:
        return self.sdf._get_domain_bounds()


class CappedBorderSDF(SDFBase):
    """
    Applies planar boundary caps to another SDF.
    """

    cap_border_dict: CapBorderDict

    def __init__(self, sdf: SDFBase, cap_border_dict=None, scale=(1, 1, 1)):
        super().__init__(geometric_dim=sdf.geometric_dim)
        self.sdf = sdf
        if cap_border_dict is None:
            match self.sdf.geometric_dim:
                case 2:
                    cap_border_dict = UNIT_CUBE_CAPS_2D
                case 3:
                    cap_border_dict = UNIT_CUBE_CAPS_3D
        self.cap_border_dict = cap_border_dict
        self.scale = scale

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        sdf_values = self.sdf(queries)

        bounds = self.sdf._get_domain_bounds().to(
            device=queries.device, dtype=queries.dtype
        )

        for loc, cap_dict in self.cap_border_dict.items():
            cap = cap_dict["cap"]
            measure = cap_dict["measure"]

            dim, side = location_lookup[loc]
            bound = bounds[side, dim]
            x = queries[:, dim].view(-1, 1)
            if side == 0 and cap == -1:
                border_sdf = x - (bound + measure)
            elif side == 0 and cap == 1:
                border_sdf = (bound + measure) - x
            elif side == 1 and cap == -1:
                border_sdf = (bound - measure) - x
            elif side == 1 and cap == 1:
                border_sdf = x - (bound - measure)
            else:
                raise RuntimeError(f"Side must be either 0 or 1, not {side}")

            # cap == 1 means add material
            if cap == 1:
                sdf_values = torch.minimum(sdf_values, -border_sdf * self.scale[dim])
                # sdf_values = torch.where(outside, border_sdf, sdf_values)

            # cap == -1 means remove material
            elif cap == -1:
                sdf_values = torch.maximum(sdf_values, -border_sdf * self.scale[dim])
                # sdf_values = torch.where(outside, border_sdf, sdf_values)

            else:
                raise ValueError("Cap must be -1 (remove) or 1 (add)")

        return sdf_values

    def _get_domain_bounds(self):
        return self.sdf._get_domain_bounds()


def generate_plane_points(origin, normal, res, xlim, ylim):
    """Generate evenly spaced points on a plane in 3D space.

    Creates a regular grid of points on a plane defined by a point and normal
    vector. The grid is axis-aligned in the plane's local coordinate system.

    Parameters
    ----------
    origin : array-like of shape (3,)
        A point on the plane (3D vector).
    normal : array-like of shape (3,)
        Normal vector of the plane (3D vector). Currently supports only
        axis-aligned normals: [1,0,0], [0,1,0], or [0,0,1].
    res : tuple of int
        Grid resolution (num_points_u, num_points_v).
    xlim : tuple of float
        Range along the first plane axis (umin, umax).
    ylim : tuple of float
        Range along the second plane axis (vmin, vmax).

    Returns
    -------
    points : np.ndarray of shape (num_points_u * num_points_v, 3)
        3D coordinates of grid points.
    u : np.ndarray of shape (num_points_u * num_points_v,)
        First plane coordinate for each point.
    v : np.ndarray of shape (num_points_u * num_points_v,)
        Second plane coordinate for each point.

    Raises
    ------
    NotImplementedError
        If normal is not axis-aligned.

    Examples
    --------
    >>> from DeepSDFStruct.plotting import generate_plane_points
    >>> import numpy as np
    >>>
    >>> # Generate points on XY plane at z=0.5
    >>> points, u, v = generate_plane_points(
    ...     origin=[0, 0, 0.5],
    ...     normal=[0, 0, 1],
    ...     res=(10, 10),
    ...     xlim=(-1, 1),
    ...     ylim=(-1, 1)
    ... )
    >>> print(points.shape)  # (100, 3)
    >>> print(np.allclose(points[:, 2], 0.5))  # True (all on z=0.5 plane)

    Notes
    -----
    The function determines two orthogonal axes (u and v) in the plane
    based on the normal vector. For axis-aligned planes:
    - Normal [0,0,1] (XY plane): u=[1,0,0], v=[0,1,0]
    - Normal [0,1,0] (XZ plane): u=[1,0,0], v=[0,0,1]
    - Normal [1,0,0] (YZ plane): u=[0,1,0], v=[0,0,1]
    """
    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal)
    origin = np.array(origin)
    # Find two orthogonal vectors to the normal that lie on the plane (u and v axes)
    if np.allclose(normal, [0, 0, 1]):  # Special case when the normal is along z-axis
        u = np.array([1, 0, 0])
        v = np.array([0, 1, 0])
    elif np.allclose(normal, [0, 1, 0]):  # Special case when the normal is along z-axis
        u = np.array([1, 0, 0])
        v = np.array([0, 0, 1])
    elif np.allclose(normal, [1, 0, 0]):  # Special case when the normal is along z-axis
        u = np.array([0, 1, 0])
        v = np.array([0, 0, 1])
    else:
        raise NotImplementedError(
            "Normal vector other than [1,0,0], [0,1,0] and [0,0,1] not supported yet."
        )

    u_coords = np.linspace(xlim[0], xlim[1], res[0])
    v_coords = np.linspace(ylim[0], ylim[1], res[1])
    u_mesh, v_mesh = np.meshgrid(u_coords, v_coords)
    u_flat = u_mesh.reshape(-1)
    v_flat = v_mesh.reshape(-1)
    points = origin + u_flat[:, None] * u + v_flat[:, None] * v

    return points


def _get_plane_plot_axes(normal):
    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal)
    if np.allclose(normal, [0, 0, 1]):
        return 0, 1
    if np.allclose(normal, [0, 1, 0]):
        return 0, 2
    if np.allclose(normal, [1, 0, 0]):
        return 1, 2
    raise NotImplementedError(
        "Normal vector other than [1,0,0], [0,1,0] and [0,0,1] not supported yet."
    )


def _build_structured_grid_triangles(nx, ny):
    triangles = []
    for j in range(ny - 1):
        row_start = j * nx
        next_row_start = (j + 1) * nx
        for i in range(nx - 1):
            p0 = row_start + i
            p1 = row_start + i + 1
            p2 = next_row_start + i
            p3 = next_row_start + i + 1
            triangles.append([p0, p1, p2])
            triangles.append([p1, p3, p2])
    return np.asarray(triangles, dtype=np.int32)


def project_bounds(origin, normal, bounds=None):
    """
    Project 3D AABB bounds onto a slice plane and return 2D limits.

    Parameters
    ----------
    origin : (3,)
        Point on plane
    normal : (3,)
        Plane normal
    bounds : (2, 3)
        AABB bounds [[xmin,ymin,zmin],[xmax,ymax,zmax]]
        If None, defaults to unit cube [0,1]^3

    Returns
    -------
    xlim, ylim : tuple
        Limits in plane coordinates
    """
    import numpy as np

    if bounds is None:
        bounds = np.array([[0, 0, 0], [1, 1, 1]])

    bmin, bmax = bounds

    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)

    origin = np.asarray(origin, dtype=float)

    # --- build orthonormal basis (u, v) ---
    if np.allclose(normal, [0, 0, 1]):
        u = np.array([1, 0, 0])
        v = np.array([0, 1, 0])
    elif np.allclose(normal, [0, 1, 0]):
        u = np.array([1, 0, 0])
        v = np.array([0, 0, 1])
    elif np.allclose(normal, [1, 0, 0]):
        u = np.array([0, 1, 0])
        v = np.array([0, 0, 1])
    else:
        raise NotImplementedError(
            "normals different to the main axis are not implemented yet."
        )

    # --- generate 8 corners of AABB ---
    corners = np.array(
        [
            [x, y, z]
            for x in [bmin[0], bmax[0]]
            for y in [bmin[1], bmax[1]]
            for z in [bmin[2], bmax[2]]
        ]
    )

    # --- project corners into plane coordinates ---
    rel = corners - origin  # important: relative to plane origin

    u_coords = rel @ u
    v_coords = rel @ v

    xlim = (u_coords.min(), u_coords.max())
    ylim = (v_coords.min(), v_coords.max())

    return xlim, ylim
