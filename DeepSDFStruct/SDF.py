from abc import ABC, abstractmethod
import torch
import numpy as np
import igl
import trimesh
import gustaf

from deep_sdf.models import DeepSDFModel
from DeepSDFStruct.plotting import plot_slice

import logging

logger = logging.getLogger(__name__)


class SDFBase(ABC):
    def __call__(self, queries: torch.Tensor) -> torch.Tensor:
        self._validate_input(queries)
        return self._compute(queries)

    def _validate_input(self, queries: torch.Tensor):
        # Example check: 2D tensor with shape (N, 3) where N points, each point is a column vector
        if queries.ndim != 2 or queries.size(1) != 3:
            raise ValueError(f"Expected input of shape (N, 3), got {queries.shape}")

    @abstractmethod
    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        """
        Subclasses implement this to compute SDF values.
        """
        pass

    @abstractmethod
    def _set_param(self, parameters: torch.Tensor):
        pass

    def plot_slice(self, *args, **kwargs):
        plot_slice(self, *args, **kwargs)


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

    def _compute(self, queries: torch.Tensor | np.ndarray):
        is_tensor = isinstance(queries, torch.Tensor)

        if is_tensor:
            orig_device = queries.device
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
            return torch.tensor(result, device=orig_device)
        else:
            return result


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
                self.model._decode_sdf(latent_vec, query_batch).squeeze(1).detach()
            )

            head = end

        return sdf_values
