from abc import ABC, abstractmethod
import torch
import numpy as np
import igl
import trimesh
import gustaf


from typing import TypedDict
from DeepSDFStruct.deep_sdf.models import DeepSDFModel
from DeepSDFStruct.plotting import plot_slice
from DeepSDFStruct.torch_spline import TorchSpline
from DeepSDFStruct.parametrization import Constant
import DeepSDFStruct

import logging

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


class SDFBase(ABC):
    """Abstract base class for Signed Distance Functions with optional
    deformation and parametrization.
    """

    def __init__(
        self,
        deformation_spline: TorchSpline | None = None,
        parametrization: torch.nn.Module | None = None,
        cap_border_dict: CapBorderDict = None,
        cap_outside_of_unitcube=False,
    ):
        self._deformation_spline = deformation_spline

        self._parametrization = parametrization
        if parametrization is not None:
            self.parameters = parametrization.parameters
        else:
            self.parameters = None

        self._cap_border_dict = cap_border_dict
        self.cap_outside_of_unitcube = cap_outside_of_unitcube

    @property
    def deformation_spline(self):
        return self._deformation_spline

    @deformation_spline.setter
    def deformation_spline(self, spline):
        self._deformation_spline = spline

    @property
    def parametrization(self):
        return self._parametrization

    @parametrization.setter
    def parametrization(self, p):
        self._parametrization = p

    @property
    def cap_border_dict(self):
        return self._cap_border_dict

    @cap_border_dict.setter
    def cap_border_dict(self, d):
        self._cap_border_dict = d

    def __call__(self, queries: torch.Tensor) -> torch.Tensor:
        self._validate_input(queries)
        sdf_values = self._compute(queries)
        if sdf_values is None:
            raise RuntimeError("Invalid SDF output")
        if self._cap_border_dict is not None:
            for loc, cap_dict in self._cap_border_dict.items():
                cap, measure = cap_dict["cap"], cap_dict["measure"]
                dim, location = location_lookup[loc]
                if "0" in loc:
                    multiplier = -1
                elif "1" in loc:
                    multiplier = 1
                border_sdf = (
                    queries[:, dim] - multiplier * (location - measure)
                ).reshape(-1, 1) * -multiplier
                # # border_sdf = border_sdf.view(-1, 1)
                # border_sdf = border_sdf.to(orig_device)
                # sdf_values = sdf_values.to(orig_device)
                if cap == -1:
                    # sdf_values = _torch.maximum(sdf_values, -border_sdf)

                    sdf_values = torch.maximum(sdf_values, -border_sdf)
                elif cap == 1:
                    sdf_values = torch.minimum(sdf_values, border_sdf)
                else:
                    raise ValueError("Cap must be -1 or 1")

            # cap everything outside of the unit cube
            # k and d are y = k*(x-dx) + dy
        if self.cap_outside_of_unitcube:
            sdf_values = _cap_outside_of_unitcube(queries, sdf_values)
        return sdf_values

    def _validate_input(self, queries: torch.Tensor):
        # Example check: 2D tensor with shape (N, 3) or (N, 2where N points, each point is a column vector
        if queries.ndim != 2 or (queries.shape[1] not in [2, 3]):
            raise ValueError(
                f"Expected input of shape (N, 3) or (N, 2), got {queries.shape}"
            )

    @abstractmethod
    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        """
        Subclasses implement this to compute SDF values.
        """
        pass

    @abstractmethod
    def _get_domain_bounds(self) -> np.array:
        """
        Subclasses implement this to know on which domain the SDF is defined
        """
        pass

    def plot_slice(self, *args, **kwargs):
        return plot_slice(self, *args, **kwargs)

    def __add__(self, other):
        return SummedSDF(self, other)


class SummedSDF(SDFBase):
    def __init__(self, obj1: SDFBase, obj2: SDFBase):
        super().__init__()
        self.obj1 = obj1
        self.obj2 = obj2

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


class NegatedCallable(SDFBase):
    def __init__(self, obj):
        super().__init__()
        self.obj = obj

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


def union(D, k=0):
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
    def __init__(
        self, mesh, dtype=np.float32, flip_sign=False, scale=True, threshold=1e-5
    ):
        """
        Computes signed distance for 3D meshes.

        Parameters
        -----------
        mesh: trimesh.Trimesh
        queries: (n, 3) np.ndarray
        dtype: type
        (Optional) Default is "np.float32". Any numpy compatible dtypes.

        Returns
        --------
        signed_distances: (n,) np.ndarray
        """
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
        return union(sdf, k=self.smoothness)


class SDFfromDeepSDF(SDFBase):
    def __init__(self, model: DeepSDFModel, max_batch=32**3):
        super().__init__()
        self.model = model
        self.latvec = None
        self.parametrization = None
        self.max_batch = max_batch

    def set_latent_vec(self, latent_vec: torch.Tensor):
        """
        Set conditioning parameters for the model (e.g., latent code).
        """
        # if dimension is 1, set parametrization
        # if dimnsion is equal to the number of query points, directly set latvec
        if latent_vec.ndim == 1:
            self.parametrization = Constant(latent_vec, device=self.model.device)
        elif latent_vec.ndim == 2:
            self.latvec = latent_vec  # (latent_dim, n_query_points)
        else:
            raise ValueError(
                f"Expected latent_vec to have 1 or 2 dimensions, got shape {latent_vec.shape}"
            )

    def _get_domain_bounds(self):
        return np.array([[-1, 1], [-1, 1], [-1, 1]])

    def _set_param(self, parameters):
        return self.set_latent_vec(parameters)

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        # DeepSDF queries range from -1 to 1
        orig_device = queries.device
        queries = queries.to(self.model.device) * 2 - 1
        n_queries = queries.shape[0]

        sdf_values = torch.zeros(n_queries, device=self.model.device)

        head = 0
        while head < n_queries:
            end = min(head + self.max_batch, n_queries)
            query_batch = queries[head:end]

            if self.latvec is None:
                latvec = self.parametrization(query_batch).to(self.model.device)
            else:
                latvec = self.latvec.to(self.model.device)[head:end]

            sdf_values[head:end] = (
                self.model._decode_sdf(latvec, query_batch).squeeze(1)
                # .detach()
            )

            head = end

        return sdf_values.to(orig_device).reshape(-1, 1)


def _cap_outside_of_unitcube(samples, sdf_values):
    dy = 0
    for dim, k, dx in zip(
        [0, 0, 1, 1, 2, 2], [1, -1, 1, -1, 1, -1], [0, 1, 0, 1, 0, 1]
    ):
        x = samples[:, dim]
        border_sdf = k * (x - dx) + dy
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
