import trimesh
from DeepSDFStruct.pretrained_models import get_model, PretrainedModels
from DeepSDFStruct.SDF import SDFfromDeepSDF, SDFfromMesh, normalize_mesh_to_unit_cube
from DeepSDFStruct.mesh import create_3D_mesh, export_sdf_grid_vtk, export_surface_mesh
from DeepSDFStruct.deep_sdf.reconstruction import reconstruct_from_samples
from DeepSDFStruct.lattice_structure import LatticeSDFStruct
from DeepSDFStruct.parametrization import SplineParametrization
from DeepSDFStruct.torch_spline import TorchScaling
from DeepSDFStruct.sampling import sample_mesh_surface, random_sample_sdf
import splinepy
import torch
import numpy as np
from DeepSDFStruct.deep_sdf.metrics.chamfer import compute_trimesh_chamfer
from DeepSDFStruct.deep_sdf.metrics.mesh_to_analytical import mesh_to_analytical


def test_shape_reconstruction():
    model = get_model(PretrainedModels.Primitives)
    sdf = SDFfromDeepSDF(model)
    torch.manual_seed(42)

    mesh_orig = trimesh.load_mesh("tests/data/cone.stl")
    mesh, scale, shift = normalize_mesh_to_unit_cube(
        mesh_orig.copy(), shrink_factor=0.9
    )
    bounds = torch.tensor(
        [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], device=model.device, dtype=torch.float32
    )
    bounds = torch.tensor(mesh.bounds, device=model.device, dtype=torch.float32)
    print("Mesh bounds:", bounds)
    scaling = TorchScaling(
        scale_factors=scale, translation=shift, bounds=bounds, device=model.device
    )

    tiling = [2, 2, 2]
    param_spline_sp = splinepy.BSpline(
        [1, 1, 1],
        [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]],
        [[0.0] * model._trained_latent_vectors[0].shape[0]] * 8,
    )
    for i_box, n_box in enumerate(tiling):
        knots = np.linspace(0, 1, n_box + 1)[1:-1]
        print(f"Inserting {n_box - 1} knots at {knots} into spline dim {i_box}")
        param_spline_sp.insert_knots(i_box, knots)

    param_spline = SplineParametrization(param_spline_sp, device=model.device)
    for p in param_spline.parameters():
        torch.nn.init.normal_(p, mean=0.0, std=0.01)

    struct = LatticeSDFStruct(
        tiling=tiling, microtile=sdf, parametrization=param_spline, bounds=bounds
    )

    gt_sdf = SDFfromMesh(mesh, scale=False)

    uniform_samples = random_sample_sdf(
        gt_sdf, bounds, n_samples=5000, type="uniform", device=struct.get_device()
    )
    surface_samples = sample_mesh_surface(
        gt_sdf, mesh, n_samples=5000, stds=[0.25, 0.0001], device=struct.get_device()
    )

    SDF_samples = uniform_samples + surface_samples

    recon_param = reconstruct_from_samples(
        struct,
        SDF_samples,
        lr=5e-3,
        loss_fn="ClampedL1",
        num_iterations=50,
        batch_size=512,
        use_tanh_on_gt=False,
        loss_plot_path="tests/tmp_outputs/reconstruction_loss.png",
        optimizer_name="adam",
        deformation_function=None,
    )

    surf_mesh, derivative = create_3D_mesh(
        struct,
        32,
        differentiate=False,
        device=model.device,
        mesh_type="surface",
        bounds=bounds,
        deformation_function=scaling,
    )

    export_surface_mesh(
        "tests/tmp_outputs/reconstructed_cone.stl", surf_mesh.to_gus(), derivative
    )

    error = mesh_to_analytical(SDFfromMesh(mesh_orig, scale=False), surf_mesh)
    print(f"Norm of SDF error on mesh vertices: {error}")


if __name__ == "__main__":

    test_shape_reconstruction()
