from abc import ABC, abstractmethod
import torch
import numpy as np
import igl
import trimesh
import gustaf


from typing import TypedDict
from DeepSDFStruct.deep_sdf.models import DeepSDFModel
from DeepSDFStruct.torch_spline import TorchSpline
from DeepSDFStruct.parametrization import Constant
import DeepSDFStruct

import logging

import matplotlib.pyplot as plt

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

    This class provides the foundation for all SDF representations in DeepSDFStruct.
    SDFs represent geometry as an implicit function that returns the signed distance
    from any query point to the nearest surface. Negative values indicate points
    inside the geometry, positive values indicate points outside, and zero indicates
    points on the surface.

    The class supports:
    - Optional spline-based deformations for smooth geometric transformations
    - Parametrization functions for spatially-varying properties
    - Border capping to constrain geometry within specified bounds
    - Composition operations (union, intersection) via operator overloading

    Parameters
    ----------
    deformation_spline : TorchSpline, optional
        A spline function that maps from parametric to physical space,
        enabling smooth deformations of the base geometry.
    parametrization : torch.nn.Module, optional
        A neural network or function that provides spatially-varying parameters
        for the SDF (e.g., varying thickness in a lattice).
    cap_border_dict : CapBorderDict, optional
        Dictionary specifying boundary conditions for each face of the domain.
        Keys are 'x0', 'x1', 'y0', 'y1', 'z0', 'z1' for the six faces.
    cap_outside_of_unitcube : bool, default False
        If True, caps the SDF values outside the unit cube to create
        a bounded geometry.
    geometric_dim : int, default 3
        Geometric dimension of the SDF (2 or 3).

    Notes
    -----
    Subclasses must implement:
    - ``_compute(queries)``: Calculate SDF values for query points
    - ``_get_domain_bounds()``: Return the bounding box of the geometry

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

    def __init__(
        self,
        deformation_spline: TorchSpline | None = None,
        parametrization: torch.nn.Module | None = None,
        geometric_dim=3,
    ):
        super().__init__()
        self.deformation_spline = deformation_spline
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

        If the module has no parameters and no buffers, returns None.
        """
        # Check parameters first
        for p in self.parameters(recurse=True):
            return p.device

        # If no parameters, check buffers
        for b in self.buffers(recurse=True):
            return b.device

        # No tensors at all
        return "cpu"

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

    def plot_slice(self, *args, **kwargs):
        return plot_slice(self, *args, **kwargs)

    def __add__(self, other):
        return UnionSDF(self, other)

    def to2D(self, axes: list[int], offset=0.0):
        """
        Converts SDF to 2D

        :param axis: list of axes that will be used for the 2D
        """
        sdf2D = SDF2D(self, axes, offset=offset)
        sdf2D.deformation_spline = self.deformation_spline
        sdf2D.parametrization = self.parametrization
        return sdf2D


class SDF2D(SDFBase):
    obj: SDFBase

    def __init__(self, obj: SDFBase, axes: list[int], offset=0.0):
        super().__init__()
        self.obj = obj
        self.deformation_spline = obj.deformation_spline
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
    def __init__(self, obj1: SDFBase, obj2: SDFBase):
        super().__init__()
        self.obj1 = obj1
        self.obj2 = obj2
        self.deformation_spline = obj1.deformation_spline
        if self.obj1.geometric_dim != self.obj2.geometric_dim:
            raise ValueError(
                f"geometric dim of object 1 ({self.obj1.geometric_dim}) differs from geometric dim of object 2 ({self.obj2.geometric_dim})"
            )
        self.geometric_dim = self.obj1.geometric_dim

    def _compute(self, queries):
        result1 = self.obj1._compute(queries)
        result2 = self.obj2._compute(queries)
        return -torch.maximum(-result1, -result2)

    def _get_domain_bounds(self):
        bounds1 = self.obj1._get_domain_bounds()
        bounds2 = self.obj2._get_domain_bounds()

        lower = torch.minimum(bounds1[0], bounds2[0])
        upper = torch.maximum(bounds1[1], bounds2[1])

        return torch.stack([lower, upper], dim=0)

    def _set_param(self, parameter):
        return None


class DifferenceSDF(SDFBase):
    """
    Subtracts objs2 from obj1
    """

    def __init__(self, obj1: SDFBase, obj2: SDFBase):
        super().__init__()
        self.obj1 = obj1
        self.obj2 = obj2
        self.deformation_spline = obj1.deformation_spline

    def _compute(self, queries):
        result1 = self.obj1._compute(queries)
        result2 = self.obj2._compute(queries)
        return torch.maximum(result1, -result2)

    def _get_domain_bounds(self):
        # the domain bounds get smaller when we substract something
        return self.obj1._get_domain_bounds()

    def _set_param(self, parameter):
        return None


class NegatedCallable(SDFBase):
    def __init__(self, obj):
        super().__init__()
        self.obj = obj
        self.deformation_spline = obj.deformation_spline

    def _compute(self, input_param):
        result = self.obj(input_param)
        return -result


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
        return torch.min(D, axis=1)[0]
    # Start with the first column as d1
    d1 = D[:, 0].copy()

    # Loop over remaining columns
    for i in range(1, D.shape[1]):
        d2 = D[:, i]
        h = torch.clip(0.5 + 0.5 * (d2 - d1) / k, 0, 1)
        d1 = d2 + (d1 - d2) * h - k * h * (1 - h)

    return d1


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
        self, mesh, dtype=np.float32, flip_sign=False, scale=True, threshold=1e-5
    ):
        super().__init__()
        if type(mesh) is gustaf.faces.Faces:
            mesh = trimesh.Trimesh(mesh.vertices, mesh.faces)

        if scale:
            # scales from [0,1] to [-1,1]
            # https://www.brainvoyager.com/bv/doc/UsersGuide/CoordsAndTransforms/SpatialTransformationMatrices.html
            mesh = normalize_mesh_to_unit_cube(mesh)
        self.mesh = mesh
        self.dtype = dtype
        self.flip_sign = flip_sign
        self.threshold = threshold

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

        # Compute squared distance
        squared_distance, hit_index, hit_coordinates = igl.point_mesh_squared_distance(
            queries_np, self.mesh.vertices, np.array(self.mesh.faces, dtype=np.int32)
        )

        distances = np.sqrt(squared_distance, dtype=self.dtype)

        # Determine sign (negative if inside)
        contains = trimesh.ray.ray_pyembree.RayMeshIntersector(
            self.mesh, scale_to_box=False
        ).contains_points(queries_np)

        distances[contains] *= -1.0

        # Apply threshold
        distances -= self.threshold

        result = distances.reshape(-1, 1)

        if is_tensor:
            return torch.tensor(result, device=orig_device, dtype=orig_dtype)
        else:
            return result


def normalize_mesh_to_unit_cube(mesh: trimesh.Trimesh):
    """
    Transform mesh coordinates uniformly to [-1, 1] in all axes.
    Keeps aspect ratio of original mesh.
    """
    logger.debug(f"Scaling mesh from {mesh.bounds.flatten()}")
    # --- Compute bounding box ---
    bbox_min = mesh.bounds[0]  # [x_min, y_min, z_min]
    bbox_max = mesh.bounds[1]  # [x_max, y_max, z_max]

    # Center of the mesh
    center = (bbox_max + bbox_min) / 2.0

    # Largest extent
    scale = (
        np.max(bbox_max - bbox_min) / 2.0
    )  # divide by 2 because [-1,1] spans 2 units

    # --- Build transformation matrix ---
    matrix = np.eye(4)

    # Translate to origin
    matrix[:3, 3] = -center

    # Apply translation
    mesh.apply_transform(matrix)

    # --- Apply uniform scaling ---
    scale_matrix = np.eye(4)
    scale_matrix[:3, :3] *= 1.0 / scale
    mesh.apply_transform(scale_matrix)
    logger.debug(f"to {mesh.bounds.flatten()}")
    return mesh


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
        queries = queries.to(self.model.device) * 2 - 1
        n_queries = queries.shape[0]

        sdf_values = torch.zeros(n_queries, device=self.model.device)

        head = 0
        if self.latvec is None:
            latvec = self.parametrization(queries).to(self.model.device)
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
        P1 (np.ndarray): Array of shape (M, 2) or (2,) representing first endpoints of segments.
        P2 (np.ndarray): Array of shape (M, 2) or (2,) representing second endpoints of segments.
        query_points (np.ndarray): Array of shape (N, 2) or (2,) representing query point(s).

    Returns:
        np.ndarray: Array of shape (N,) with the minimum distance from each query point
                    to the closest segment.
    """
    P1 = np.atleast_2d(P1)  # (M, 2)
    P2 = np.atleast_2d(P2)  # (M, 2)
    Q = np.atleast_2d(query_points)  # (N, 2)

    # Handle degenerate case: one segment only
    if P1.shape[0] == 1 and P2.shape[0] == 1:
        P1 = np.repeat(P1, Q.shape[0], axis=0)
        P2 = np.repeat(P2, Q.shape[0], axis=0)

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

    def __init__(self, sdf: SDFBase, rotation=None, translation=None, scale=None):
        super().__init__()
        self.sdf = sdf
        self.rotation = rotation
        self.translation = translation
        self.scale = scale

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        xyz = queries

        # apply scale
        if self.scale is not None:
            xyz = xyz / self.scale  # inverse scale

        # apply rotation
        if self.rotation is not None:
            xyz = xyz @ self.rotation.T  # rotate points

        # apply translation
        if self.translation is not None:
            xyz = xyz - self.translation

        sdf_vals = self.sdf._compute(xyz)

        # rescale distances if scaled
        if self.scale is not None:
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
        self.deformation_spline = sdf.deformation_spline
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
                outside = x < (bound + measure)
            elif side == 0 and cap == 1:
                border_sdf = (bound + measure) - x
                outside = x < (bound + measure)
            elif side == 1 and cap == -1:
                border_sdf = (bound - measure) - x
                outside = x > (bound - measure)
            elif side == 1 and cap == 1:
                border_sdf = x - (bound - measure)
                outside = x > (bound - measure)
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


def plot_slice(
    fun: SDFBase,
    origin=(0, 0, 0),
    normal=(0, 0, 1),
    res=(100, 100),
    ax=None,
    xlim=(-1, 1),
    ylim=(-1, 1),
    clim=(-1, 1),
    cmap="seismic",
    show_zero_level=True,
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

    points, u, v = generate_plane_points(origin, normal, res, xlim, ylim)

    sdf_device = fun.get_device()
    points = torch.from_numpy(points).to(torch.float32).to(sdf_device)
    sdf = fun(points).reshape((res[0], res[1]))
    X = u.reshape((res[0], res[1]))
    Y = v.reshape((res[0], res[1]))
    sdf = sdf.detach().cpu().numpy()

    # cbar = ax[0].scatter(X, Y, c=sdf, cmap="seismic")c
    cbar = ax.contourf(X, Y, sdf, cmap=cmap, levels=10)
    if show_zero_level:
        ax.contour(X, Y, sdf, levels=[0], colors="black", linewidths=0.5)
    cbar.set_clim(clim[0], clim[1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1)
    if plt_show:
        plt.show()
        return fig, ax


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

    # Create grid points in 2D space (u-v plane)
    u_coords = np.linspace(xlim[0], xlim[1], res[0])
    v_coords = np.linspace(ylim[0], ylim[1], res[1])

    points = []
    u_exp = []
    v_exp = []
    for u_val in u_coords:
        for v_val in v_coords:
            point = origin + u_val * u + v_val * v
            u_exp.append(u_val)
            v_exp.append(v_val)
            points.append(point)

    return np.array(points), np.array(u_exp), np.array(v_exp)
