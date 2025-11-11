from DeepSDFStruct.local_shapes import LocalShapesSDF
from DeepSDFStruct.pretrained_models import get_model, PretrainedModels
from DeepSDFStruct.SDF import SDFfromDeepSDF, _cap_outside_of_unitcube
from DeepSDFStruct.lattice_structure import LatticeSDFStruct
from DeepSDFStruct.parametrization import Constant
from DeepSDFStruct.torch_spline import TorchSpline
import splinepy
import torch


def test_pretrained_model_evaluation():
    for pt_model in PretrainedModels:
        print(f"Testing pretrained model: {pt_model.name}")
        model = get_model(pt_model)
        sdf = SDFfromDeepSDF(model)

        # Set the latent vector and visualize a slice of the SDF
        sdf.set_latent_vec(model._trained_latent_vectors[0])

        out = sdf(
            torch.tensor(
                [[0, 0, 0], [0, 1, 0], [0.5, 0.5, 0.5]],
                dtype=torch.float32,
                device=model.device,
            )
        )
        print(out)


if __name__ == "__main__":
    test_pretrained_model_evaluation()