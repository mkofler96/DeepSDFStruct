from DeepSDFStruct.pretrained_models import get_model, PretrainedModels
from DeepSDFStruct.SDF import SDFfromDeepSDF, SDFfromLineMesh
from DeepSDFStruct.mesh import generate_2D_surf_mesh
from DeepSDFStruct.lattice_structure import LatticeSDFStruct, constantLatvec
import splinepy
import gustaf as gus
import torch
import gustaf as _gus


def test_deepsdf_lattice_evaluation():
    # Load a pretrained DeepSDF model
    model = get_model(PretrainedModels.RoundCross)
    sdf = SDFfromDeepSDF(model)

    # Set the latent vector and visualize a slice of the SDF
    sdf.set_latent_vec(torch.tensor([0.3]))

    # Define a spline-based deformation field
    deformation_spline = splinepy.helpme.create.box(2, 1, 1)

    # Create the lattice structure with deformation and microtile
    lattice_struct = LatticeSDFStruct(
        tiling=(6, 3, 3),
        deformation_spline=deformation_spline,
        microtile=sdf,
        parametrization_spline=constantLatvec([0.5]),
    )

    surf_mesh = lattice_struct.create_surface_mesh(30)
    faces = surf_mesh.to_gus()
    _gus.io.meshio.export("faces.inp", faces)


def test_2D_mesh_export():
    linemesh = gus.io.meshio.load("tests/data/example_line_mesh.vtk")
    linemesh.vertices = linemesh.vertices[:, :2]

    sdf_from_linemesh = SDFfromLineMesh(linemesh, thickness=0.5)
    mesh = generate_2D_surf_mesh(sdf_from_linemesh, 300)
    gus.io.meshio.export("triangles.inp", mesh)


if __name__ == "__main__":
    test_deepsdf_lattice_evaluation()
    test_2D_mesh_export()
