import os
import vtk
import numpy as np
import json
import pathlib
import typing
import gustaf as gus
import trimesh
from DeepSDFStruct.SDF import SDFfromMesh, SDFBase
import DeepSDFStruct

# from analysis.problems.homogenization import computeHomogenizedMaterialProperties
import splinepy
import torch
from collections import defaultdict
from tqdm import tqdm
import logging


logger = logging.getLogger(DeepSDFStruct.__name__)


class DataSetInfo(typing.TypedDict):
    dataset_name: str
    class_name: str


class SphereParameters(typing.TypedDict):
    cx: float
    cy: float
    cz: float
    r: float


class SampledSDF:
    samples: torch.Tensor
    distances: torch.Tensor

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

    np.savez(fname, neg=neg.stacked, pos=pos.stacked)


class SDFSampler:
    def __init__(
        self,
        outdir,
        splitdir,
        dataset_name,
        unify_multipatches=True,
        stds=[0.05, 0.025],
    ) -> None:
        self.outdir = outdir
        self.splitdir = splitdir
        self.dataset_name = dataset_name
        self.unify_multipatches = unify_multipatches
        self.geometries = {}
        self.stds = stds

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
        add_surface_samples=True,
        also_save_vtk=False,
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
                if os.path.isfile(fname):
                    logger.warning(f"File {fname} already exists")
                    continue
                sdf = self.get_sdf_from_geometry(
                    geometry, n_faces, self.unify_multipatches
                )
                sampled_sdf = random_sample_sdf(
                    sdf,
                    bounds=(-1, 1),
                    n_samples=int(n_samples),
                    type=sampling_strategy,
                )
                if add_surface_samples:
                    if not isinstance(geometry, trimesh.Trimesh):
                        logger.warning(
                            "Add surface samples was specified, but geometry"
                            f"is not given as a trimesh.Trimesh but as {type(geometry)}"
                        )
                    else:
                        surf_samples = sample_mesh_surface(
                            sdf,
                            sdf.mesh,
                            int(n_samples // 2),
                            self.stds,
                            device="cpu",
                            dtype=torch.float32,
                        )
                        sampled_sdf += surf_samples
                pos, neg = sampled_sdf.split_pos_neg()

                np.savez(fname, neg=neg.stacked, pos=pos.stacked)
                if also_save_vtk:
                    save_points_to_vtp(
                        fname.with_suffix(".vtp"), neg=neg.stacked, pos=pos.stacked
                    )

    def get_sdf_from_geometry(
        self,
        geometry,
        n_faces: int,
        unify_multipatches: bool = True,
        threshold: float = 1e-5,
    ) -> SDFfromMesh:
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
        elif isinstance(geometry, trimesh.Trimesh):
            sdf_geom = SDFfromMesh(geometry, threshold=threshold)

        else:
            raise NotImplementedError(
                f"Geometry of type {type(geometry)} not supported yet."
            )

        return sdf_geom

    def get_meshs_from_folder(self, foldername, mesh_type) -> list:
        """
        Reads all mesh files of a given type (extension) from a folder using meshio.

        Parameters
        ----------
        foldername : str
            Path to the folder containing the mesh files.
        mesh_type : str
            Mesh file extension (e.g., 'vtk', 'obj', 'stl', 'msh', 'xdmf').

        Returns
        -------
        list[trimesh.Trimesh]
            A list of trimesh.Trimesh objects loaded from the folder.
        """
        meshes = []

        # Normalize extension (remove dot if present)
        mesh_type = mesh_type.lstrip(".")

        # Iterate through all files in the folder

        for filename in tqdm(os.listdir(foldername), desc="Loading meshs"):
            if filename.lower().endswith("." + mesh_type.lower()):
                filepath = os.path.join(foldername, filename)
                try:
                    faces = gus.io.meshio.load(filepath)
                    trim = trimesh.Trimesh(faces.vertices, faces.elements)
                    meshes.append(trim)
                    logger.info(f"Loaded mesh: {filename}")
                except ValueError as e:
                    logger.warning(f"Could not read {filename}: {e}")

        if not meshes:
            print(f"No .{mesh_type} meshes found in {foldername}.")

        return meshes

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


def save_points_to_vtp(filename, neg, pos):
    """
    Save pos/neg SDF sample points as a VTU point cloud using vtkPolyData.
    Each point has an SDF scalar value.
    """
    # Combine points
    all_points = np.vstack((pos, neg))
    coords = all_points[:, :3]
    sdf_vals = all_points[:, 3]

    # --- Create vtkPoints ---
    vtk_points = vtk.vtkPoints()
    for pt in coords:
        vtk_points.InsertNextPoint(pt)

    # --- Create PolyData ---
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)

    # Add vertex cells (required for points in PolyData)
    verts = vtk.vtkCellArray()
    for i in range(len(coords)):
        verts.InsertNextCell(1)
        verts.InsertCellPoint(i)
    polydata.SetVerts(verts)

    # --- Add SDF scalar values ---
    vtk_array = vtk.vtkDoubleArray()
    vtk_array.SetName("SDF")
    vtk_array.SetNumberOfValues(len(sdf_vals))
    for i, val in enumerate(sdf_vals):
        vtk_array.SetValue(i, val)
    polydata.GetPointData().SetScalars(vtk_array)

    # --- Write to VTU ---
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Write()

    logger.debug(f"Saved {len(coords)} points with SDF to '{filename}'")


def augment_by_FFD(
    meshs: list[trimesh.Trimesh],
    n_control_points: int = 5,
    std_dev_fraction: float | None = 0.05,
    n_transformations: int = 10,
    save_meshs=False,
) -> list[trimesh.Trimesh]:
    """
    Takes list of meshs and augments the meshs by applying a freeform deformation
    """
    new_meshs = []

    for i_mesh, mesh in enumerate(tqdm(meshs, desc="Augmenting meshs")):
        bbox = mesh.bounds  # shape (2, 3)
        # Compute approximate spacing between control points along each axis
        spacing = (bbox[1] - bbox[0]) / (n_control_points - 1)
        # Use a fraction of spacing (e.g., 15%) as std_dev
        std_dev_local = std_dev_fraction * spacing

        for i_FFD in range(n_transformations):
            ffd = splinepy.FFD()
            ffd.mesh = gus.Faces(mesh.vertices, mesh.faces)
            ffd.spline.insert_knots(0, np.linspace(0, 1, n_control_points)[1:-1])
            ffd.spline.insert_knots(1, np.linspace(0, 1, n_control_points)[1:-1])
            ffd.spline.insert_knots(2, np.linspace(0, 1, n_control_points)[1:-1])
            ffd.spline.elevate_degrees([0, 1, 2])
            ffd.spline.control_points += np.random.normal(
                loc=0.0, scale=std_dev_local, size=ffd.spline.control_points.shape
            )
            new_meshs.append(trimesh.Trimesh(ffd.mesh.vertices, ffd.mesh.faces))
            if save_meshs:
                save_meshs = True

                # Make sure the directory exists
                os.makedirs("tmp", exist_ok=True)
                gus.io.meshio.export(f"tmp/mesh_{i_mesh}_{i_FFD}.obj", ffd.mesh)

    return new_meshs
