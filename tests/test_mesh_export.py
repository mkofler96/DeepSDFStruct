from DeepSDFStruct.pretrained_models import get_model, PretrainedModels
from DeepSDFStruct.SDF import SDFfromDeepSDF, SDFfromLineMesh
from DeepSDFStruct.mesh import (
    generate_2D_surf_mesh,
    tetrahedralize_surface,
    create_3D_surface_mesh,
    export_surface_mesh_vtk,
    export_sdf_grid_vtk,
)
from DeepSDFStruct.lattice_structure import LatticeSDFStruct
from DeepSDFStruct.parametrization import SplineParametrization
from DeepSDFStruct.torch_spline import TorchSpline
import splinepy
import torch
import gustaf as _gus


def test_deepsdf_lattice_export():
    # Load a pretrained DeepSDF model
    model = get_model(PretrainedModels.AnalyticRoundCross)
    sdf = SDFfromDeepSDF(model)

    # Define a spline-based deformation field
    deformation_spline = TorchSpline(
        splinepy.helpme.create.box(1, 1, 1), device=model.device
    )

    param_spline = SplineParametrization(
        splinepy.BSpline(
            [1, 1, 1],
            [[-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1]],
            [[0.5], [0.5], [0.5], [0.5], [0.5], [0.5], [0.5], [0.5]],
        ),
        device=model.device,
    )

    # Create the lattice structure with deformation and microtile
    lattice_struct = LatticeSDFStruct(
        tiling=(2, 2, 2),
        deformation_spline=deformation_spline,
        microtile=sdf,
        parametrization=param_spline,
    )
    export_sdf_grid_vtk(lattice_struct, "sdf.vtk")
    surf_mesh, derivative = create_3D_surface_mesh(
        lattice_struct, 30, differentiate=True
    )
    export_surface_mesh_vtk(
        surf_mesh.vertices, surf_mesh.faces, "mesh_with_derivative.vtk", derivative
    )
    faces = surf_mesh.to_gus()
    _gus.io.meshio.export("faces.inp", faces)
    _gus.io.meshio.export("faces.obj", faces)

    volumes, _ = tetrahedralize_surface(faces)
    _gus.io.mfem.export("volumes.mfem", volumes)


def test_2D_mesh_export():
    linemesh = _gus.io.meshio.load("tests/data/example_line_mesh.vtk")
    linemesh.vertices = linemesh.vertices[:, :2]

    sdf_from_linemesh = SDFfromLineMesh(linemesh, thickness=0.5)
    mesh = generate_2D_surf_mesh(sdf_from_linemesh, 300)
    _gus.io.meshio.export("triangles.inp", mesh)

    sdf_from_linemesh = SDFfromLineMesh(linemesh, thickness=0.5, smoothness=0.1)
    mesh = generate_2D_surf_mesh(sdf_from_linemesh, 300)
    _gus.io.meshio.export("triangles_smooth.inp", mesh)


if __name__ == "__main__":
    test_deepsdf_lattice_export()
    test_2D_mesh_export()
