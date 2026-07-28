#!/usr/bin/env python3
"""
Primitive-Scene Training Data Generator
=======================================

Generates training scenes of simple geometric primitives (spheres, boxes,
cylinders) randomly positioned, oriented, and scaled inside a bounding box. The
ground-truth SDF is computed *analytically* by combining the primitive SDFs with
the minimum operation (``UnionSDF``).

For each scene the SDF is sampled in two complementary ways:
  - uniformly in the volume (``random_sample_sdf``), and
  - near the surface via Gaussian perturbations at several standard deviations
    (``sample_mesh_surface``).

The output is written in the layout consumed by ``SDFSamples`` in
``training_latent_field.py``::

    <data_source>/
    ├── SdfSamples/<dataset_name>/<class_name>/<instance>.npz   # pos/neg, [x,y,z,sdf]
    ├── SdfSamples/<dataset_name>/<vtp_subdir>/<instance>.vtp   # ParaView point clouds
    └── splits/<split_name>.json                                # {dataset:{class:[instance,...]}}

Run directly to generate a dataset using the editable ``CONFIG`` dict at the
bottom of this file::

    python -m DeepSDFStruct.deep_sdf.generate_primitive_dataset
"""

import json
import pathlib
import logging
import datetime
from importlib.metadata import version

import numpy as np
import torch
import trimesh

import DeepSDFStruct
from DeepSDFStruct.sdf_primitives import SphereSDF, BoxSDF, CylinderSDF
from DeepSDFStruct.SDF import TransformedSDF, SDFBase
from DeepSDFStruct.sampling import (
    SampledSDF,
    random_sample_sdf,
    sample_mesh_surface,
    save_points_to_vtp,
)

logger = logging.getLogger(DeepSDFStruct.__name__)


def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Uniformly distributed random 3x3 rotation matrix (Shoemake's method)."""
    u1, u2, u3 = rng.random(3)
    q = np.array(
        [
            np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2),  # x
            np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2),  # y
            np.sqrt(u1) * np.sin(2.0 * np.pi * u3),  # z
            np.sqrt(u1) * np.cos(2.0 * np.pi * u3),  # w
        ]
    )
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _place(
    canonical_sdf: SDFBase,
    canonical_mesh: trimesh.Trimesh,
    R: np.ndarray,
    center: np.ndarray,
) -> tuple[SDFBase, trimesh.Trimesh]:
    """Apply the same rigid transform (rotation R, translation center) to both
    an analytical SDF and its matching mesh so they coincide exactly.

    World SDF is ``s0(R^T (x - center))`` (rigid => distance preserving), which
    ``TransformedSDF`` reproduces with ``rotationMatrix=R^T`` and
    ``translation=R^T @ center``. The matching world vertices are
    ``center + R @ v_canonical``.
    """
    Rt = R.T
    sdf = TransformedSDF(
        canonical_sdf,
        rotationMatrix=Rt.tolist(),  # list-of-lists avoids TransformedSDF's `== [0,0,0]` check
        translation=(Rt @ center).tolist(),
        scaleFactor=1.0,
    )

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = center
    mesh = canonical_mesh.copy()
    mesh.apply_transform(T)
    return sdf, mesh


class _AnisoScaledSDF(SDFBase):
    """Wrap a *unit* canonical SDF with an anisotropic (per-axis) scaling.

    The geometry / zero-level set is exact, but because non-uniform scaling does
    not preserve Euclidean distance, the off-surface magnitude is an
    approximation. We multiply by ``min(scale)`` so the field stays a valid
    1-Lipschitz signed distance (it never overestimates the true distance and is
    exact along the least-scaled axis). Boxes do not use this wrapper because
    ``BoxSDF`` represents anisotropic extents exactly.
    """

    def __init__(self, sdf: SDFBase, scale_vec):
        super().__init__()
        self.sdf = sdf
        s = torch.as_tensor(scale_vec, dtype=torch.float32).reshape(1, 3)
        self.register_buffer("scale_vec", s)
        self.correction = float(s.min().item())

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        sv = self.scale_vec.to(device=queries.device, dtype=queries.dtype)
        return self.sdf._compute(queries / sv) * self.correction

    def _get_domain_bounds(self) -> torch.Tensor:
        return self.sdf._get_domain_bounds() * self.scale_vec.reshape(-1)


def _make_primitive(
    prim_type: str, rng: np.random.Generator
) -> tuple[SDFBase, trimesh.Trimesh, np.ndarray]:
    """Build a primitive SDF and a matching trimesh with an independent random
    per-axis scale (``scale_vec`` = semi-size along x/y/z).

    Boxes use exact per-axis ``BoxSDF`` extents. Spheres/cylinders start from a
    unit canonical shape and receive an anisotropic scale wrapper (ellipsoid /
    elliptical cylinder); the returned SDF is the object to feed into ``_place``
    (rotation + translation), and the mesh is the matching pre-scaled mesh.
    """
    s_lo, s_hi = _make_primitive.scale_range
    scale_vec = rng.uniform(s_lo, s_hi, size=3)  # independent x/y/z semi-sizes

    if prim_type == "sphere":
        unit_sdf = SphereSDF(center=[0.0, 0.0, 0.0], radius=1.0)
        sdf = _AnisoScaledSDF(unit_sdf, scale_vec)
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        mesh.apply_scale(scale_vec.tolist())  # -> ellipsoid with semi-axes scale_vec
    elif prim_type == "box":
        extents = (2.0 * scale_vec).tolist()  # exact anisotropic box
        sdf = BoxSDF(center=[0.0, 0.0, 0.0], extents=extents)
        mesh = trimesh.creation.box(extents=extents)
    elif prim_type == "cylinder":
        # unit cylinder: radius 1 (x/y), height 2 (half-height 1 along z)
        unit_sdf = CylinderSDF(
            point=[0.0, 0.0, 0.0], axis=[0.0, 0.0, 1.0], radius=1.0, height=2.0
        )
        sdf = _AnisoScaledSDF(unit_sdf, scale_vec)
        mesh = trimesh.creation.cylinder(radius=1.0, height=2.0, sections=32)
        mesh.apply_scale(scale_vec.tolist())  # elliptical cross-section + scaled height
    else:
        raise ValueError(f"Unknown primitive type: {prim_type}")

    return sdf, mesh, scale_vec


def _build_scene(
    primitive_types: list[str],
    n_primitives: int,
    bounds: np.ndarray,
    random_rotation: bool,
    rng: np.random.Generator,
) -> tuple[SDFBase, trimesh.Trimesh]:
    """Compose a scene SDF (union of placed primitives) and the concatenated
    surface mesh used for near-surface sampling."""
    bounds_lo, bounds_hi = bounds[0], bounds[1]
    sdfs: list[SDFBase] = []
    meshes: list[trimesh.Trimesh] = []

    for _ in range(n_primitives):
        prim_type = str(rng.choice(primitive_types))
        canonical_sdf, canonical_mesh, scale_vec = _make_primitive(prim_type, rng)

        # keep the primitive (roughly) inside the box; out-of-bounds samples are
        # rejected later regardless, this just avoids wasting samples.
        margin = float(np.max(scale_vec))
        lo = np.minimum(bounds_lo + margin, bounds_hi - margin)
        hi = np.maximum(bounds_lo + margin, bounds_hi - margin)
        center = rng.uniform(lo, hi)

        R = _random_rotation_matrix(rng) if random_rotation else np.eye(3)
        sdf, mesh = _place(canonical_sdf, canonical_mesh, R, center)
        sdfs.append(sdf)
        meshes.append(mesh)

    scene_sdf = sdfs[0]
    for s in sdfs[1:]:
        scene_sdf = scene_sdf + s  # UnionSDF via torch.minimum

    scene_mesh = trimesh.util.concatenate(meshes)
    return scene_sdf, scene_mesh


def _filter_to_bounds(sampled: SampledSDF, bounds: np.ndarray) -> SampledSDF:
    """Drop any sample whose coordinates fall outside ``bounds`` (e.g. when
    near-surface Gaussian perturbations push points beyond the box)."""
    lo = torch.tensor(bounds[0], dtype=sampled.samples.dtype)
    hi = torch.tensor(bounds[1], dtype=sampled.samples.dtype)
    inside = ((sampled.samples >= lo) & (sampled.samples <= hi)).all(dim=1)
    return SampledSDF(samples=sampled.samples[inside], distances=sampled.distances[inside])


def generate_primitive_dataset(cfg: dict) -> dict:
    """Generate a primitive-scene SDF dataset according to ``cfg``.

    Returns the dataset summary dict (also written to ``summary.json``).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    data_source = pathlib.Path(cfg["data_source"])
    dataset_name = cfg["dataset_name"]
    class_name = cfg["class_name"]
    bounds = np.asarray(cfg["bounds"], dtype=np.float64)

    # propagate the size range to the primitive factory
    _make_primitive.scale_range = tuple(cfg["scale_range"])

    sample_dir = data_source / "SdfSamples" / dataset_name / class_name
    vtp_dir = data_source / "SdfSamples" / dataset_name / cfg["vtp_subdir"]
    split_path = data_source / "splits" / cfg["split_name"]

    sample_dir.mkdir(parents=True, exist_ok=True)
    if cfg["save_vtp"]:
        vtp_dir.mkdir(parents=True, exist_ok=True)
    split_path.parent.mkdir(parents=True, exist_ok=True)

    instance_names: list[str] = []
    n_scenes = int(cfg["num_scenes"])
    for i in range(n_scenes):
        idx = int(cfg["instance_start_index"]) + i
        instance_name = f"{dataset_name}_{idx}"
        instance_names.append(instance_name)

        npz_path = sample_dir / f"{instance_name}.npz"
        if npz_path.is_file() and not cfg["overwrite"]:
            logger.info(f"[skip] {npz_path} (exists)")
            continue

        # deterministic, per-scene seeding (covers numpy + torch + trimesh.sample)
        scene_seed = int(cfg["seed"]) + idx
        rng = np.random.default_rng(scene_seed)
        np.random.seed(scene_seed % (2**32 - 1))
        torch.manual_seed(scene_seed)

        with torch.no_grad():
            scene_sdf, scene_mesh = _build_scene(
                primitive_types=list(cfg["primitive_types"]),
                n_primitives=int(cfg["primitives_per_scene"]),
                bounds=bounds,
                random_rotation=bool(cfg["random_rotation"]),
                rng=rng,
            )

            uniform = random_sample_sdf(
                scene_sdf,
                bounds=bounds.tolist(),
                n_samples=int(cfg["n_uniform"]),
                sampling_strategy="uniform",
            )
            surface = sample_mesh_surface(
                scene_sdf,
                scene_mesh,
                int(cfg["n_surface_per_std"]),
                list(cfg["stds"]),
            )
            combined = uniform + surface
            # near-surface Gaussian perturbations can push points past the box;
            # reject anything outside the bounds so no sample lies outside.
            combined = _filter_to_bounds(combined, bounds)

        pos, neg = combined.split_pos_neg()
        np.savez(
            npz_path,
            neg=neg.stacked.detach().cpu().numpy(),
            pos=pos.stacked.detach().cpu().numpy(),
        )

        if cfg["save_vtp"]:
            save_points_to_vtp(vtp_dir / f"{instance_name}.vtp", combined.stacked)

        logger.info(
            f"[{i + 1}/{n_scenes}] {instance_name}: "
            f"{pos.samples.shape[0]} pos / {neg.samples.shape[0]} neg -> {npz_path}"
        )

    # split json: {dataset: {class: [instance, ...]}}
    split = {dataset_name: {class_name: instance_names}}
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=4)
    logger.info(f"Wrote split {split_path}")

    summary = {
        "dataset_name": dataset_name,
        "class_name": class_name,
        "num_scenes": n_scenes,
        "primitives_per_scene": int(cfg["primitives_per_scene"]),
        "primitive_types": list(cfg["primitive_types"]),
        "bounds": bounds.tolist(),
        "n_uniform": int(cfg["n_uniform"]),
        "n_surface_per_std": int(cfg["n_surface_per_std"]),
        "stds": list(cfg["stds"]),
        "scale_range": list(cfg["scale_range"]),
        "random_rotation": bool(cfg["random_rotation"]),
        "seed": int(cfg["seed"]),
        "date_created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sdf_struct_version": version("DeepSDFStruct"),
    }
    summary_path = data_source / "SdfSamples" / dataset_name / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Wrote summary {summary_path}")

    return summary


CONFIG = {
    "data_source": "/Users/lukas/projects/projectsPhd/TrainingData",  # DataSource root
    "dataset_name": "2026_06_25_primitive_shapes_4",
    "class_name": "shapes",
    "split_name": "primitives_train_4.json",  # written under <data_source>/splits/
    "num_scenes": 4,  # paper N=100
    "primitives_per_scene": 10,  # paper 10
    "primitive_types": ["sphere", "box", "cylinder"],
    "bounds": [[-1, -1, -1], [1, 1, 1]],  # Omega_box
    "n_uniform": 100_000,  # uniform samples / scene
    "n_surface_per_std": 500_000,  # x len(stds) => 1,000,000 near-surface
    "stds": [0.005, 0.0001],  # paper sigma1, sigma2
    "scale_range": [0.1, 0.5],  # characteristic half-size of primitives
    "random_rotation": True,
    "seed": 42,
    "save_vtp": True,  # ParaView point clouds
    "vtp_subdir": "paraview",
    "overwrite": True,
    "instance_start_index": 10000,
}


if __name__ == "__main__":
    generate_primitive_dataset(CONFIG)
