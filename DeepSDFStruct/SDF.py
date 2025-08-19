from abc import ABC, abstractmethod
import torch
import numpy as np
import igl
import trimesh
import gustaf

from DeepSDFStruct.deep_sdf.models import DeepSDFModel
from DeepSDFStruct.plotting import plot_slice

import logging

logger = logging.getLogger(__name__)


class SDFBase(ABC):
    def __call__(self, queries: torch.Tensor) -> torch.Tensor:
        self._validate_input(queries)
        return self._compute(queries)

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
    def _set_param(self, parameters: torch.Tensor):
        pass

    @abstractmethod
    def _get_domain_bounds(self) -> np.array:
        """
        Subclasses implement this to know on which domain the SDF is defined
        """
        pass

    def plot_slice(self, *args, **kwargs):
        return plot_slice(self, *args, **kwargs)


class SummedSDF(SDFBase):
    def __init__(self, obj1, obj2):
        self.obj1 = obj1
        self.obj2 = obj2

    def __call__(self, input_param):
        result1 = self.obj1(input_param)
        result2 = self.obj2(input_param)
        return -torch.maximum(-result1, -result2)


class NegatedCallable(SDFBase):
    def __init__(self, obj):
        self.obj = obj

    def __call__(self, input_param):
        result = self.obj(input_param)
        return -result


class BoxSDF(SDFBase):
    def __init__(
        self, box_size: float = 1, center: torch.tensor = torch.tensor([0, 0, 0])
    ):
        self.box_size = box_size
        self.center = center

    def _compute(self, queries: torch.tensor) -> torch.tensor:
        output = (
            torch.linalg.norm(queries - self.center, axis=1, ord=torch.inf)
            - self.box_size
        )
        return output.reshape(-1, 1)


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

    def __init__(self, line_mesh: gustaf.Edges, thickness):
        """
        takes a line mesh and the thickness of the lines as inputs and
        generates a SDF from it
        for now only supports lines in 2D
        """
        self.line_mesh = line_mesh
        self.t = thickness

    def _get_domain_bounds(self):
        return self.line_mesh.bounds()

    def _set_param(self, parameters):
        self.t = parameters

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
        sdf = point_segment_distance(lines[:, 0], lines[:, 1], queries_np) - self.t
        if is_tensor:
            sdf = torch.tensor(sdf, dtype=orig_dtype, device=orig_device)
        return sdf


class SDFfromDeepSDF(SDFBase):
    def __init__(self, model: DeepSDFModel, max_batch=32**3):
        self.model = model
        self._latent_vec = None
        self.max_batch = max_batch

    def set_latent_vec(self, latent_vec: torch.Tensor):
        """
        Set conditioning parameters for the model (e.g., latent code).
        """
        self._latent_vec = latent_vec
        if latent_vec.ndim == 1:
            self._latent_vec = latent_vec.unsqueeze(1)  # (latent_dim, 1)
        elif latent_vec.ndim == 2:
            self._latent_vec = latent_vec  # (latent_dim, n_query_points)
        else:
            raise ValueError(
                f"Expected latent_vec to have 1 or 2 dimensions, got shape {latent_vec.shape}"
            )

    def _get_domain_bounds(self):
        return np.array([[-1, 1], [-1, 1], [-1, 1]])

    def _set_param(self, parameters):
        return self.set_latent_vec(parameters)

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        queries = queries.to(self.model.device)
        n_queries = queries.shape[0]

        latent_vec = self._latent_vec
        if latent_vec is None:
            latent_vec = self.model._trained_latent_vectors[0].unsqueeze(1)
            logger.info(
                f"No latent vector set. Using default latent vec shape: {latent_vec.shape}"
            )
        latent_vec = latent_vec.to(self.model.device)

        sdf_values = torch.zeros(n_queries, device=self.model.device)

        head = 0
        while head < n_queries:
            end = min(head + self.max_batch, n_queries)
            query_batch = queries[head:end]

            sdf_values[head:end] = (
                self.model._decode_sdf(latent_vec[head:end], query_batch)
                .squeeze(1)
                .detach()
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
    return distances.min(axis=1)
