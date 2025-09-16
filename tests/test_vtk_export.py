from DeepSDFStruct.pretrained_models import get_model, PretrainedModels
from DeepSDFStruct.SDF import SDFfromDeepSDF
from DeepSDFStruct.mesh import export_sdf_grid_vtk
from DeepSDFStruct.lattice_structure import LatticeSDFStruct
from DeepSDFStruct.parametrization import Constant
from DeepSDFStruct.torch_spline import TorchSpline
import splinepy
import torch


def test_deepsdf_vtk_export():
    # Load a pretrained DeepSDF model
    model = get_model(PretrainedModels.RoundCross)
    sdf = SDFfromDeepSDF(model)

    # Set the latent vector and visualize a slice of the SDF
    sdf.set_latent_vec(torch.tensor([0.3]))

    # Define a spline-based deformation field
    deformation_spline = TorchSpline(
        splinepy.helpme.create.box(2, 1, 1), device=model.device
    )

    # Create the lattice structure with deformation and microtile
    lattice_struct = LatticeSDFStruct(
        tiling=(1, 1, 1),
        deformation_spline=deformation_spline,
        microtile=sdf,
        parametrization=Constant([0.5], device=model.device),
    )

    export_sdf_grid_vtk(lattice_struct, "sdf.vtk")

    # surf_mesh, derivative = create_3D_surface_mesh(
    #     lattice_struct, 30, differentiate=True
    # )
    # faces = surf_mesh.to_gus()
    # _gus.io.meshio.export("faces.inp", faces)
    # _gus.io.meshio.export("faces.obj", faces)

    # volumes, _ = tetrahedralize_surface(faces)
    # _gus.io.mfem.export("volumes.mfem", volumes)


if __name__ == "__main__":
    test_deepsdf_vtk_export()
