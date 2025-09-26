import os
import numpy as np
import json
import pathlib
import typing
import gustaf as gus
import trimesh
from DeepSDFStruct.SDF import SDFfromMesh, SDFBase

# from analysis.problems.homogenization import computeHomogenizedMaterialProperties
import splinepy
import torch
from collections import defaultdict
from tqdm import tqdm
import logging


logger = logging.getLogger(__name__)


class DataSetInfo(typing.TypedDict):
    dataset_name: str
    class_name: str


class SphereParameters(typing.TypedDict):
    cx: float
    cy: float
    cz: float
    r: float


class SampledSDF:
    samples: torch.tensor
    distances: torch.tensor

    def split_pos_neg(self):
        pos_mask = torch.where(self.distances >= 0.0)[0]
        neg_mask = torch.where(self.distances < 0.0)[0]
        pos = SampledSDF(
            samples=self.samples[pos_mask], distances=self.distances[pos_mask]
        )
        neg = SampledSDF(
            samples=self.samples[neg_mask], distances=self.distances[neg_mask]
        )
        return pos, neg

    def create_gus_plottable(self):
        vp = gus.Vertices(vertices=self.samples)
        vp.vertex_data["distance"] = self.distances
        return vp

    @property
    def stacked(self):
        return torch.hstack((self.samples, self.distances))

    def __init__(self, samples, distances):
        self.samples = samples
        self.distances = distances

    def __add__(self, other):
        return SampledSDF(
            samples=torch.vstack((self.samples, other.samples)),
            distances=torch.vstack((self.distances, other.distances)),
        )


def process_single_geometry(args):
    (
        class_name,
        instance_id,
        geometry,
        outdir,
        dataset_name,
        unify_multipatches,
        compute_mechanical_properties,
        n_faces,
        n_samples,
        sampling_strategy,
        show,
        get_sdf_from_geometry,
        sample_sdf,
    ) = args

    logger.info(f"processing {instance_id} in geometry list {class_name}")
    file_name = f"{instance_id}.npz"

    folder_name = pathlib.Path(outdir) / dataset_name / class_name
    fname = folder_name / file_name

    if not os.path.exists(folder_name):
        os.makedirs(folder_name, exist_ok=True)

    if os.path.isfile(fname) and not show:
        logger.warning(f"File {fname} already exists")
        return

    sdf = get_sdf_from_geometry(geometry, n_faces, unify_multipatches)
    pos, neg = sample_sdf(
        sdf, show=show, n_samples=n_samples, sampling_strategy=sampling_strategy
    )

    if compute_mechanical_properties:
        mesh_file_name = f"{instance_id}.mesh"
        raise NotImplementedError("Compute homogenized material not available yet.")
        mesh_file_path = folder_name / "homogenization" / instance_id / mesh_file_name
        # E = computeHomogenizedMaterialProperties(
        #     sdf, mesh_file_path=mesh_file_path, mirror=True
        # )
        np.savez(fname, neg=neg.stacked, pos=pos.stacked, E=E)
    else:
        np.savez(fname, neg=neg.stacked, pos=pos.stacked)


class SDFSampler:
    def __init__(self, outdir, splitdir, dataset_name, unify_multipatches=True) -> None:
        self.outdir = outdir
        self.splitdir = splitdir
        self.dataset_name = dataset_name
        self.unify_multipatches = unify_multipatches
        self.geometries = {}

    def add_class(self, geom_list: list, class_name: str) -> None:
        instances = {}
        for i, geom in enumerate(geom_list):
            instance_name = f"{class_name}_{i:05}"
            instances[instance_name] = geom
        self.geometries[class_name] = instances

    def get_SDF_list(self, n_faces=100) -> list[SDFBase]:
        sdf_list = []
        for class_name, instance_list in self.geometries.items():
            logger.info(f"processing geometry list {class_name}")
            for instance_id, geometry in tqdm(
                instance_list.items(), desc="Processing instances"
            ):
                sdf = self.get_sdf_from_geometry(
                    geometry, n_faces, self.unify_multipatches
                )
                sdf_list.append(sdf)
        return sdf_list

    def process_geometries(
        self,
        sampling_strategy="uniform",
        n_faces=100,
        n_samples: int = 1e5,
        unify_multipatches=True,
        compute_mechanical_properties=True,
        show=False,
    ):
        for class_name, instance_list in self.geometries.items():
            logger.info(f"processing geometry list {class_name}")
            for instance_id, geometry in tqdm(
                instance_list.items(), desc="Processing instances"
            ):

                file_name = f"{instance_id}.npz"

                folder_name = pathlib.Path(self.outdir) / self.dataset_name / class_name
                fname = folder_name / file_name
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)
                if os.path.isfile(fname) and show == False:
                    logger.warning(f"File {fname} already exists")
                    continue
                sdf = self.get_sdf_from_geometry(
                    geometry, n_faces, self.unify_multipatches
                )
                pos, neg = self.sample_sdf(
                    sdf,
                    show=show,
                    n_samples=n_samples,
                    sampling_strategy=sampling_strategy,
                )
                if compute_mechanical_properties:
                    mesh_file_name = f"{instance_id}.mesh"
                    mesh_file_path = (
                        folder_name / "homogenization" / instance_id / mesh_file_name
                    )
                    raise NotImplementedError(
                        "Compute homogenized material not available yet."
                    )
                    E = computeHomogenizedMaterialProperties(
                        sdf, mesh_file_path=mesh_file_path, mirror=True
                    )
                    np.savez(fname, neg=neg.stacked, pos=pos.stacked, E=E)
                else:
                    np.savez(fname, neg=neg.stacked, pos=pos.stacked)

    def sample_sdf(
        self,
        sdf,
        show=False,
        n_samples: int = 1e5,
        sampling_strategy="uniform",
        box_size=None,
        stds=[0.0025, 0.00025],
    ):

        sampled_sdf = random_sample_sdf(
            sdf, bounds=(-1, 1), n_samples=int(n_samples), type=sampling_strategy
        )

        pos, neg = sampled_sdf.split_pos_neg()

        if show:
            vp_pos = pos.create_gus_plottable()
            vp_neg = neg.create_gus_plottable()
            vp_pos.show_options["cmap"] = "coolwarm"
            vp_neg.show_options["cmap"] = "coolwarm"
            vp_pos.show_options["vmin"] = -0.1
            vp_pos.show_options["vmax"] = 0.1
            vp_neg.show_options["vmin"] = -0.1
            vp_neg.show_options["vmax"] = 0.1
            gus.show(vp_neg, vp_pos)
        return pos, neg

    def get_sdf_from_geometry(
        self,
        geometry,
        n_faces: int,
        unify_multipatches: bool = True,
        threshold: float = 1e-5,
    ) -> SDFBase:
        if isinstance(geometry, splinepy.Multipatch):
            if unify_multipatches:
                patch_meshs = []
                for patch in geometry.patches:
                    patch_faces = patch.extract.faces(n_faces)
                    patch_mesh = trimesh.Trimesh(
                        vertices=patch_faces.vertices, faces=patch_faces.faces
                    )
                    # add all patches as meshs to one boolean addition
                    patch_meshs.append(SDFfromMesh(patch_mesh))
                sdf_geom = patch_meshs[0]
                for pm in patch_meshs[1:]:
                    sdf_geom = sdf_geom + pm
            else:
                sdf_geom = SDFfromMesh(
                    geometry.extract.faces(n_faces), threshold=threshold
                )

        else:
            raise NotImplementedError(
                f"Geometry of type {type(geometry)} not supported yet."
            )

        return sdf_geom

    def write_json(self, json_fname):
        json_content = defaultdict(lambda: defaultdict(list))
        for class_name, instance_list in self.geometries.items():
            for instance_id, geometry in instance_list.items():
                file_name = f"{instance_id}"
                json_content[self.dataset_name][class_name].append(file_name)
        # json_content = {
        #     data_info["dataset_name"]: {data_info["class_name"]: split_files}
        # }
        json_fname = pathlib.Path(f"{self.splitdir}/{json_fname}")
        if not json_fname.parent.is_dir():
            os.makedirs(json_fname.parent)
        with open(json_fname, "w", encoding="utf-8") as f:
            json.dump(json_content, f, indent=4)


def move(t_mesh, new_center):
    t_mesh.vertices += new_center - t_mesh.bounding_box.centroid


def noisy_sample(t_mesh, std, count):
    return t_mesh.sample(int(count)) + torch.random.normal(
        scale=std, size=(int(count), 3)
    )


def random_points(count):
    """random points in a unit sphere centered at (0, 0, 0)"""
    points = torch.random.uniform(-1, 1, (int(count * 3), 3))
    points = points[torch.linalg.norm(points, axis=1) <= 1]
    if points.shape[0] < count:
        print("Too little random sampling points. Resampling.......")
        random_points(count=count, boundary="unit_sphere")
    elif points.shape[0] > count:
        return points[torch.random.choice(points.shape[0], count)]
    else:
        return points


def random_points_cube(count, box_size):
    """random points in a cube with size box_size centered at (0, 0, 0)"""
    points = torch.random.uniform(-box_size / 2, box_size / 2, (int(count), 3))
    return points


def random_sample_sdf(
    sdf, bounds, n_samples, type="uniform", device="cpu", dtype=torch.float32
):

    bounds = torch.tensor(bounds, dtype=dtype, device=device)
    if type == "plane":
        samples = torch.random.uniform(
            bounds[0], bounds[1], (n_samples, 2), device=device, dtype=dtype
        )
        samples = torch.hstack((samples, torch.zeros((n_samples, 1))))
    elif type == "spherical_gaussian":
        samples = torch.random.randn(n_samples, 3, device=device, dtype=dtype)
        samples /= torch.linalg.norm(samples, axis=1).reshape(-1, 1)
        # samples += torch.random.uniform(bounds[0], bounds[1], (n_samples, 3))
        samples = samples + torch.random.normal(0, 0.01, (n_samples, 3))
    elif type == "uniform":
        samples = (
            torch.rand((n_samples, 3), device=device, dtype=dtype)
            * (bounds[1] - bounds[0])
            + bounds[0]
        )
    distances = sdf(samples)
    return SampledSDF(samples=samples, distances=distances)


def sample_mesh_surface(
    sdf: SDFBase,
    mesh: gus.Faces,
    n_samples: int,
    stds: list[float],
    device="cpu",
    dtype=torch.float32,
) -> SampledSDF:
    """
    Sample noisy points around a mesh surface and evaluate them with a signed distance function (SDF).

    This function uses trimesh.sample to generate surface samples
    and perturbs them with Gaussian noise of varying standard deviations,
    and queries the SDF at those points.

    Args:
        sdf (SDFBase): A callable SDF object that takes 3D points and returns signed distances.
        mesh (gus.Faces): A mesh object containing the vertices.
        n_samples (int): Number of mesh vertices to sample
        stds (list[float]): Standard deviations for Gaussian noise added to sampled vertices.
            - Typical values: [0.05, 0.0015].
            - Larger values spread samples farther from the surface; smaller values keep them closer.
        device (str, optional): Torch device to place tensors on (e.g., "cpu" or "cuda").
        dtype (torch.dtype, optional): Data type for generated tensors (default: torch.float32).

    Returns:
        SampledSDF: An object containing:
            - samples (torch.Tensor): The perturbed sample points of shape (n_samples * len(stds), 3).
            - distances (torch.Tensor): The corresponding SDF values at those sample points.
    """
    samples = []

    trim = trimesh.Trimesh(mesh.vertices, mesh.faces)

    random_samples = torch.tensor(trim.sample(n_samples), dtype=dtype, device=device)

    for std in stds:
        noise = torch.randn((n_samples, 3), device=device, dtype=dtype) * std
        samples.append(random_samples + noise)

    queries = torch.vstack(samples)

    distances = sdf(queries)

    return SampledSDF(samples=queries, distances=distances)
