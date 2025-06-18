from DeepSDFStruct.pretrained_models import get_model, PretrainedModels
from DeepSDFStruct.SDF import SDFfromDeepSDF
from DeepSDFStruct.lattice_structure import LatticeSDFStruct, constantLatvec
import splinepy
import torch


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
        tiling=(6, 3, 1),
        deformation_spline=deformation_spline,
        microtile=sdf,
        parametrization_spline=constantLatvec([0.5]),
    )

    lattice_struct(
        torch.tensor([[0, 0, 0], [0, 1, 0]], dtype=torch.float32, device=model.device)
    )


if __name__ == "__main__":
    test_deepsdf_lattice_evaluation()
