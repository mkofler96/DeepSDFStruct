from enum import Enum
import importlib.resources
from DeepSDFStruct.deep_sdf.workspace import load_trained_model, load_latent_vectors
from DeepSDFStruct.deep_sdf.models import DeepSDFModel
import torch


class PretrainedModels(Enum):
    ChiAndCross = "chi_and_cross"
    AnalyticRoundCross = "analytic_round_cross"
    RoundCross = "round_cross"


# Maps enum entries to file paths
main_dir = importlib.resources.files("DeepSDFStruct")
_MODEL_REGISTRY = {
    PretrainedModels.ChiAndCross: main_dir / "trained_models" / "chi_and_cross",
    PretrainedModels.AnalyticRoundCross: main_dir
    / "trained_models"
    / "analytic_round_cross",
    PretrainedModels.RoundCross: main_dir / "trained_models" / "round_cross",
}


def get_model(model: str | PretrainedModels, checkpoint: str = "latest", device=None):
    """
    Load a pretrained model by name or enum.

    Args:
        model (str | PretrainedModels): model identifier
        checkpoint (str): checkpoint file name (default: 'latest')

    Returns:
        Trained PyTorch model
    """
    if isinstance(model, str):
        try:
            model_enum = PretrainedModels(model)
        except ValueError:
            raise ValueError(f"Unknown pretrained model name: {model}")
    else:
        model_enum = model

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = _MODEL_REGISTRY.get(model_enum)
    if not path:
        raise ValueError(f"Model path not registered for: {model_enum.name}")
    decoder = load_trained_model(path, checkpoint, device=device)
    latent_vectors = load_latent_vectors(path, checkpoint, device=device)
    decoder.eval()
    deep_sdf_model = DeepSDFModel(decoder, latent_vectors, device=device)
    return deep_sdf_model


def list_available_models():
    return list(PretrainedModels)
