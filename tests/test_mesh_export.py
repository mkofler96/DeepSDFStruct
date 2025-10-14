from DeepSDFStruct.pretrained_models import get_model, PretrainedModels
from DeepSDFStruct.SDF import SDFfromDeepSDF, SDFfromLineMesh, SDFfromMesh
from DeepSDFStruct.mesh import (
    generate_2D_surf_mesh,
    tetrahedralize_surface,
    create_3D_mesh,
    export_surface_mesh,
    export_sdf_grid_vtk,
)
from DeepSDFStruct.deep_sdf.reconstruction import (
    reconstruct_from_samples,
    reconstruct_deepLS_from_samples,
)
from DeepSDFStruct.lattice_structure import LatticeSDFStruct
from DeepSDFStruct.parametrization import SplineParametrization
from DeepSDFStruct.torch_spline import TorchSpline
from DeepSDFStruct.sampling import sample_mesh_surface, random_sample_sdf
import splinepy
import gustaf as _gus
import torch


def test_deepsdf_lattice_export():
    # Load a pretrained DeepSDF model
    model = get_model(PretrainedModels.AnalyticRoundCross)
    sdf = SDFfromDeepSDF(model)
    torch.manual_seed(42)

    # Define a spline-based deformation field
    deformation_spline = TorchSpline(
        splinepy.helpme.create.box(1.5, 1, 1).bspline, device=model.device
    )
    control_points = [
        [0.3],
        [0.3],
        [0.3],
        [0.3],
        [0.3],
        [0.3],
        [0.3],
        [0.3],
        [0.3],
        [0.3],
        [0.3],
        [0.3],
    ]
    param_spline = SplineParametrization(
        splinepy.BSpline(
            [0, 0, 0], [[0, 1 / 3, 2 / 3, 1], [0, 0.5, 1], [0, 0.5, 1]], control_points
        ),
        device=model.device,
    )

    # Create the lattice structure with deformation and microtile
    lattice_struct = LatticeSDFStruct(
        tiling=[3, 2, 2],
        deformation_spline=deformation_spline,
        microtile=sdf,
        parametrization=param_spline,
    )
    surf_mesh, derivative = create_3D_mesh(
        lattice_struct, 30, differentiate=True, mesh_type="surface", device=model.device
    )
    export_surface_mesh(
        "tests/tmp_outputs/mesh_with_derivative.vtk", surf_mesh.to_gus(), derivative
    )
    export_sdf_grid_vtk(
        lattice_struct, "tests/tmp_outputs/sdf.vtk", device=model.device
    )
    mesh = surf_mesh.to_gus()
    _gus.io.meshio.export("tests/tmp_outputs/faces.inp", mesh)
    _gus.io.meshio.export("tests/tmp_outputs/faces.obj", mesh)

    volumes, _ = tetrahedralize_surface(mesh)
    _gus.io.mfem.export("tests/tmp_outputs/volumes.mfem", volumes)

    # test other method to export 3D mesh:
    volume_mesh, derivative_vols = create_3D_mesh(
        lattice_struct, 30, differentiate=True, mesh_type="volume", device=model.device
    )
    _gus.io.meshio.export(
        fname="tests/tmp_outputs/volumes.inp", mesh=volume_mesh.to_gus()
    )

    # test reconstruction
    device = model.device
    gt_sdf = SDFfromMesh(mesh, scale=False)
    uniform_samples = random_sample_sdf(
        gt_sdf, mesh.bounds(), n_samples=int(1e3), type="uniform", device=device
    )
    surface_samples = sample_mesh_surface(
        gt_sdf, mesh, n_samples=int(1e3), stds=[0.0, 0.025], device=device
    )

    SDF_samples = uniform_samples + surface_samples
    lattice_struct.parametrization
    # set latent vector to 0.2
    for p in lattice_struct.parametrization.parameters():
        p.data *= 0
        p.data += 0.2
    recon_param_lattice = reconstruct_from_samples(
        lattice_struct,
        SDF_samples,
        device=model.device,
        lr=1e-3,
        loss_fn="ClampedL1",
        num_iterations=500,
        batch_size=len(SDF_samples.samples),
    )
    # set latent vector to 0.2
    for p in lattice_struct.parametrization.parameters():
        p.data *= 0
        p.data += 0.2
    recon_param_DeepLS = reconstruct_deepLS_from_samples(
        lattice_struct,
        SDF_samples,
        device=model.device,
        lr=1e-3,
        loss_fn="ClampedL1",
        num_iterations=500,
        batch_size=len(SDF_samples.samples),
    )
    print("Original parameters:")
    print(control_points)
    print("Reconstructed parameters from Lattice:")
    print(recon_param_lattice)
    print("Reconstructed parameters from DeepLS:")
    print(recon_param_DeepLS)
    torch.testing.assert_close(
        torch.tensor(control_points, device=device),
        recon_param_lattice[0],
        rtol=0.1,
        atol=0.1,
    )
    torch.testing.assert_close(
        torch.tensor(control_points, device=device),
        recon_param_DeepLS[0],
        rtol=0.1,
        atol=0.1,
    )


def test_2D_mesh_export():
    linemesh = _gus.io.meshio.load("tests/data/example_line_mesh.vtk")
    linemesh.vertices = linemesh.vertices[:, :2]

    sdf_from_linemesh = SDFfromLineMesh(linemesh, thickness=0.5)
    mesh = generate_2D_surf_mesh(sdf_from_linemesh, 300)
    _gus.io.meshio.export("tests/tmp_outputs/triangles.inp", mesh)

    sdf_from_linemesh = SDFfromLineMesh(linemesh, thickness=0.5, smoothness=0.1)
    mesh = generate_2D_surf_mesh(sdf_from_linemesh, 300)
    _gus.io.meshio.export("tests/tmp_outputs/triangles_smooth.inp", mesh)


if __name__ == "__main__":
    test_deepsdf_lattice_export()
    test_2D_mesh_export()
