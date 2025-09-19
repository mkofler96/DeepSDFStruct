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
from DeepSDFStruct.parametrization import _Parametrization, Constant

import logging

logger = logging.getLogger(__name__)


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


class SDFBase(ABC):
    """Abstract base class for Signed Distance Functions with optional
    deformation and parametrization.
    """

    def __init__(
        self,
        deformation_spline: TorchSpline = None,
        parametrization: _Parametrization = None,
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
    def parametrization(self, p: _Parametrization):
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
                ) * -multiplier
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

    def reconstruct_from_mesh(
        self,
        mesh: gustaf.Faces,
        num_iterations=1000,
        lr=5e-4,
        l2reg=False,
        device="cpu",
    ):
        parameters = (
            torch.ones_like(self.parameters).normal_(mean=0, std=0.1).to(device)
        )

        parameters.requires_grad = True

        optimizer = torch.optim.Adam([parameters], lr=lr)

        loss_num = 0

        queries_parameter_space = self.deformation_spline.spline.proximities(
            mesh.vertices
        )
        verts_min = mesh.vertices.min(axis=0)
        verts_max = mesh.vertices.max(axis=0)

        print("Min/Max in PHYSICAL space:\n")
        for name, mn, mx in zip(
            ["x", "y", "z"], verts_min.tolist(), verts_max.tolist()
        ):
            print(f"{name}: min={mn:.6f}, max={mx:.6f}")
        queries_ps_torch = torch.tensor(
            queries_parameter_space, device=device, dtype=parameters.dtype
        )
        queries_min = queries_ps_torch.min(dim=0).values
        queries_max = queries_ps_torch.max(dim=0).values

        print("\nMin/Max in QUERY space:\n")
        for name, mn, mx in zip(
            ["x", "y", "z"], queries_min.tolist(), queries_max.tolist()
        ):
            print(f"{name}: min={mn:.6f}, max={mx:.6f}")

        for e in range(num_iterations):
            optimizer.zero_grad()

            # SDF at vertices needs to be zero
            self.parametrization.set_param(parameters)
            loss = torch.norm(self.__call__(queries_ps_torch))
            if l2reg:
                loss += 1e-4 * torch.mean(parameters.pow(2))
            loss.backward()
            optimizer.step()

            if e % 50 == 0:
                loss_num = loss.detach().item()
                print(f"Epoch: {e:5g} | Loss: {loss_num:.5f}")

            loss_num = loss.cpu().data.numpy()

        print("Reconstructed parameters:")
        print(parameters)
        return parameters


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
            rescale = 2.0
            tform = [-1.0 for i in range(3)]
            matrix = np.eye(4)
            matrix[:3, :3] *= rescale
            mesh.apply_transform(matrix)
            matrix = np.eye(4)
            matrix[:3, 3] = tform
            mesh.apply_transform(matrix)
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
        queries = queries.to(self.model.device) * 2 - 1
        n_queries = queries.shape[0]

        if self.latvec is None:
            latvec = self.parametrization(queries)
        else:
            latvec = self.latvec.to(self.model.device)

        sdf_values = torch.zeros(n_queries, device=self.model.device)

        head = 0
        while head < n_queries:
            end = min(head + self.max_batch, n_queries)
            query_batch = queries[head:end]

            sdf_values[head:end] = (
                self.model._decode_sdf(latvec[head:end], query_batch).squeeze(1)
                # .detach()
            )

            head = end

        return sdf_values


def _cap_outside_of_unitcube(samples, sdf_values):
    dy = 0
    for dim, k, dx in zip(
        [0, 0, 1, 1, 2, 2], [1, -1, 1, -1, 1, -1], [0, 1, 0, 1, 0, 1]
    ):
        x = samples[:, dim]
        border_sdf = k * (x - dx) + dy
        sdf_values = torch.maximum(sdf_values, -border_sdf)
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
