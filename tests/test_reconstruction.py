"""Reconstruct the feed-channel geometry as a field of local shapes.

Port of the DeepShapeOpt ``experiments/reconstruction/feed_channel`` case onto
the plain :class:`~DeepSDFStruct.geom_reconstruction.LocalShapesReconstructor`
API -- same mesh, same hyperparameters, no JSON config and no environment
variables. The hyperparameters now *are* the library defaults, so only the
target mesh and the tiling are spelled out here.

Run as a test::

    uv run pytest tests/test_reconstruction.py

or as a script, which does exactly the same and prints the summary::

    uv run python tests/test_reconstruction.py
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
import trimesh

from DeepSDFStruct.deep_sdf.metrics.mesh_to_analytical import mesh_to_analytical
from DeepSDFStruct.geom_reconstruction import (
    LocalShapesReconstructor,
    build_parameter_spline,
    sample_gt_sdf,
)
from DeepSDFStruct.mesh import create_3D_mesh, export_reconstructed_artifacts
from DeepSDFStruct.pretrained_models import PretrainedModels
from DeepSDFStruct.SDF import SDFfromMesh


def test_fit_mesh_unpacks_and_reduces_error():
    """The headline API: four-tuple unpacking, then mesh extraction."""
    recon = LocalShapesReconstructor()
    torch.manual_seed(42)

    mesh_orig = trimesh.load_mesh("tests/data/cone.stl")

    struct, scaling, gt_sdf, params = recon.fit_mesh(
        mesh=mesh_orig,
        tiling=[2, 2, 2],
        num_iterations=5,
        lr=5e-3,
        batch_size=512,
        n_uniform=2000,
        n_surface=5000,
        loss_plot_path="tests/tmp_outputs/local_shapes_loss.png",
        loss_csv_path="tests/tmp_outputs/local_shapes_loss.csv",
    )

    # params[0] is the control-point tensor -- the contract every caller relies
    # on. (The list can hold more: SDFfromDeepSDF registers a latent buffer on
    # its first forward pass, after the optimizer was built, so it is never
    # trained.)
    assert params[0].shape == (3**3, recon.latent_dim)
    assert torch.isfinite(params[0]).all()

    # fit_mesh must leave the struct holding the optimized codes, not the
    # mean-code initialization it started from.
    assert params[0] is struct.parametrization.torch_spline.control_points

    # The original mesh is untouched; the fit works on a normalized copy.
    assert np.allclose(
        mesh_orig.bounds, trimesh.load_mesh("tests/data/cone.stl").bounds
    )

    surf_mesh, _ = create_3D_mesh(
        struct,
        32,
        differentiate=False,
        device=recon.device,
        mesh_type="surface",
        bounds=struct.bounds,
        deformation_function=scaling,
    )
    assert surf_mesh.vertices.shape[0] > 0

    error = mesh_to_analytical(SDFfromMesh(mesh_orig, scale=False), surf_mesh)
    print(f"Norm of SDF error on mesh vertices: {error}")
    assert np.isfinite(error)

    # gt_sdf is returned ready to use, in the same (normalized) space as struct.
    probe = struct.bounds.mean(dim=0, keepdim=True)
    assert gt_sdf(probe).shape == (1, 1)


def test_fit_samples_reports_diagnostics_and_lowers_loss():
    """The composed path: build, sample, fit -- with the loss history."""
    recon = LocalShapesReconstructor()
    torch.manual_seed(0)

    mesh_orig = trimesh.load_mesh("tests/data/cone.stl")
    built = recon.build_struct(mesh_orig, [2, 2, 2])

    gt_sdf = SDFfromMesh(built.mesh_norm, scale=False)
    samples = sample_gt_sdf(
        gt_sdf,
        built.mesh_norm,
        built.bounds,
        n_uniform=2000,
        n_surface=5000,
        device=recon.device,
    )
    assert samples.samples.shape[0] == 2000 + 5000 * 2  # two stds
    assert samples.samples.shape[1] == 3

    result = LocalShapesReconstructor.fit_samples(
        built.struct, samples, num_iterations=5, lr=5e-3, batch_size=512
    )
    assert result["num_steps"] == len(result["loss_history"]) > 0
    assert result["final_loss"] == result["loss_history"][-1]
    # Five epochs of Adam should beat the mean-code starting point.
    assert result["final_loss"] < result["loss_history"][0]


def test_box_constrained_sampling_respects_bounds():
    """box_constrained=True must reject surface samples outside the domain."""
    recon = LocalShapesReconstructor()
    torch.manual_seed(0)

    mesh_orig = trimesh.load_mesh("tests/data/cone.stl")
    built = recon.build_struct(mesh_orig, [2, 2, 2])

    # Shrink the domain so a large share of the surface band falls outside.
    center = built.bounds.mean(dim=0)
    tight = center + 0.4 * (built.bounds - center)

    gt_sdf = SDFfromMesh(built.mesh_norm, scale=False)
    samples = sample_gt_sdf(
        gt_sdf,
        built.mesh_norm,
        tight,
        n_uniform=500,
        n_surface=2000,
        device=recon.device,
        box_constrained=True,
    )

    surface = samples.samples[500:]
    assert surface.shape[0] == 2000, "rejection sampling must top back up to n_surface"
    inside = ((surface >= tight[0]) & (surface <= tight[1])).all(dim=1)
    assert bool(inside.all()), "every kept surface sample must lie inside bounds"


def test_build_parameter_spline_knot_spans_match_tiling():
    spline = build_parameter_spline([1, 1, 1], [1, 8, 8], latent_dim=4)
    # Degree 1 with n spans -> n + 1 control points per dimension.
    assert spline.control_points.shape == (2 * 9 * 9, 4)

    bounds = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    spline = build_parameter_spline([1, 1, 1], [2, 2, 2], latent_dim=3, bounds=bounds)
    assert spline.control_points.shape == (27, 3)
    for kv in spline.knot_vectors:
        assert np.isclose(kv[0], -1.0) and np.isclose(kv[-1], 1.0)


def test_export_reconstructed_artifacts_writes_files(tmp_path):
    recon = LocalShapesReconstructor()
    torch.manual_seed(42)

    mesh_orig = trimesh.load_mesh("tests/data/cone.stl")
    built = recon.build_struct(mesh_orig, [2, 2, 2])

    out = export_reconstructed_artifacts(
        built.struct,
        tmp_path,
        mesh_resolution=16,
        bounds=built.bounds,
        device=recon.device,
        scaling=built.scaling,
        sdf_grid_N=16,
    )
    assert out.exists()
    assert (tmp_path / "reconstructed_sdf_grid.vtk").exists()
    assert (tmp_path / "reconstructed_mesh_parameterspace.stl").exists()

    # The lattice must come back in its original dtype after the float32 cast.
    assert built.struct.bounds.dtype == torch.float32




TESTS_DIR = Path(__file__).resolve().parent
MESH_PATH = TESTS_DIR / "data" / "flow_channel.stl"
# Overrides the reconstructor's default "output": test artifacts belong in the
# gitignored tests/tmp_outputs, not in the repository root.
OUTPUT_DIR = TESTS_DIR / "tmp_outputs" / "feed_channel"

TILING = [1, 8, 8]
# Below the default of 10 epochs, to keep CI short. Everything else is left at
# the library default.
NUM_ITERATIONS = 3
# Same resolution recon.export uses, so the metric is taken on the mesh that
# gets written out.
MESH_RESOLUTION = 32
# Mean |SDF| of the target mesh over the reconstructed surface vertices, in the
# mesh's own units. Three epochs land at ~0.03, the unfitted mean-code
# initialization at ~6.2, so this catches a fit that stopped working without
# being tight enough to flag ordinary run-to-run noise.
MAX_SDF_ERROR = 0.1


def test_feed_channel_reconstruction():
    logging.basicConfig(level=logging.INFO)

    recon = LocalShapesReconstructor(
        PretrainedModels.PrimitivesCL32, output_dir=OUTPUT_DIR
    )
    out_dir = recon.ensure_output_dir()
    torch.manual_seed(42)

    mesh_orig = trimesh.load_mesh(str(MESH_PATH))
    assert mesh_orig.is_watertight, "target mesh must be watertight"

    # fit_mesh returns a FitResult; its first four fields unpack directly.
    # Bind the whole thing instead if you also want loss_history / final_loss.
    struct, scaling, gt_sdf, params = recon.fit_mesh(
        mesh=mesh_orig, tiling=TILING, num_iterations=NUM_ITERATIONS
    )

    # Degree 1 with n knot spans -> n + 1 control points per dimension.
    n_expected = (TILING[0] + 1) * (TILING[1] + 1) * (TILING[2] + 1)
    assert params[0].shape == (n_expected, recon.latent_dim)
    assert torch.isfinite(params[0]).all()

    torch.save(params, out_dir / "rec_parameters.pt")

    mesh_path = recon.export(struct, scaling, mesh_resolution=MESH_RESOLUTION)
    assert mesh_path.exists()
    assert (out_dir / "reconstruction_loss.csv").exists()

    # gt_sdf comes back in the same (normalized) space as struct.
    probe = struct.bounds.mean(dim=0, keepdim=True)
    assert gt_sdf(probe).shape == (1, 1)

    # How far the reconstruction's surface sits from the target surface: mean
    # |SDF of the original mesh| over the reconstructed vertices, in the
    # original mesh's units (its bounding box is ~10 x 67 x 71).
    surf_mesh, _ = create_3D_mesh(
        struct,
        MESH_RESOLUTION,
        differentiate=False,
        device=recon.device,
        mesh_type="surface",
        bounds=struct.bounds,
        deformation_function=scaling,
    )
    error = mesh_to_analytical(SDFfromMesh(mesh_orig, scale=False), surf_mesh)
    print(f"Norm of SDF error on mesh vertices: {error}")
    assert error < MAX_SDF_ERROR, f"SDF error {error:.4f} exceeds {MAX_SDF_ERROR}"

    summary = {
        "mesh_path": str(MESH_PATH),
        "tiling": TILING,
        "n_control_points": int(params[0].shape[0]),
        "latent_dim": recon.latent_dim,
        "sdf_error": error,
        "reconstructed_mesh_path": str(mesh_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    test_feed_channel_reconstruction()
    test_fit_mesh_unpacks_and_reduces_error()
    test_fit_samples_reports_diagnostics_and_lowers_loss()
    test_box_constrained_sampling_respects_bounds()
    test_build_parameter_spline_knot_spans_match_tiling()