from DeepSDFStruct.pretrained_models import get_model, PretrainedModels
from DeepSDFStruct.SDF import SDFfromDeepSDF
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


def test_analytic_round_cross():

    model = get_model(PretrainedModels.AnalyticRoundCross)
    sdf = SDFfromDeepSDF(model)

    for radius in [0.2, 0.5, 0.7]:
        radius_tensor = torch.tensor([radius], dtype=torch.float32, device=model.device)
        sdf.set_latent_vec(radius_tensor)

        out = sdf(torch.tensor([[0, 0, 0]], dtype=torch.float32, device=model.device))
        torch.testing.assert_close(out[0], -radius_tensor)


if __name__ == "__main__":
    test_pretrained_model_evaluation()
    test_analytic_round_cross()
