from DeepSDFStruct.pretrained_models import get_model, PretrainedModels
from DeepSDFStruct.SDF import SDFfromDeepSDF, _cap_outside_of_unitcube
from DeepSDFStruct.lattice_structure import LatticeSDFStruct
from DeepSDFStruct.parametrization import Constant
from DeepSDFStruct.torch_spline import TorchSpline
import splinepy
import torch


def test_deepsdf_lattice_evaluation():
    # Load a pretrained DeepSDF model
    model = get_model(PretrainedModels.AnalyticRoundCross)
    sdf = SDFfromDeepSDF(model)

    # Set the latent vector and visualize a slice of the SDF
    sdf.set_latent_vec(torch.tensor([0.3]))

    # Define a spline-based deformation field
    deformation_spline = TorchSpline(
        splinepy.helpme.create.box(1, 1, 1).bspline, device=model.device
    )

    # Create the lattice structure with deformation and microtile
    lattice_struct = LatticeSDFStruct(
        tiling=(1, 1, 1),
        deformation_spline=deformation_spline,
        microtile=sdf,
        parametrization=Constant([0.5], device=model.device),
    )

    out = lattice_struct(
        torch.tensor(
            [[0, 0, 0], [0, 1, 0], [0.5, 0.5, 0.5]],
            dtype=torch.float32,
            device=model.device,
        )
    )
    print(out)


def test_cap_outside_unitcube():

    samples = torch.tensor(
        [
            [1.1, 0.0, 0.0],
            [-0.1, 0.0, 0.0],
            [0.0, 1.1, 0.0],
            [0.0, -0.1, 0.0],
            [0.0, 0.0, 1.1],
            [0.0, 0.0, -0.1],
            [0.1, 0.1, 0.1],
            [0.9, 0.9, 0.9],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    # true = inside, false = outside
    labels = [False, False, False, False, False, False, True, True]

    # Initial SDF values: all negative (e.g., "inside")
    sdf_values = torch.tensor([[-0.5]] * samples.shape[0], dtype=torch.float32)

    capped_sdf = _cap_outside_of_unitcube(samples, sdf_values.clone())

    # Check that the SDF is capped (increased) for points outside the cube
    for sample, sdf_value, label in zip(samples, capped_sdf, labels):
        if sdf_value < 1e-10:
            continue
        assertion = (sdf_value < 0) == label
        assert assertion, f"Incorrect capping on valid sample {sample}"
    assert capped_sdf[-1] < 1e10, "Point on the boundary is wrong"


if __name__ == "__main__":
    test_cap_outside_unitcube()
    test_deepsdf_lattice_evaluation()
