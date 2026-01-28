"""
Pretrained DeepSDF Models
========================

This module provides access to pretrained DeepSDF neural network models for
common microstructure geometries. These models can be used directly for
geometry generation or as starting points for transfer learning.

Available Models
----------------

ChiAndCross
    Chi-shaped and cross-shaped lattice structures, commonly used in
    mechanical metamaterials.

AnalyticRoundCross
    Round cross-section variations with analytical parameterization,
    useful for smooth stress distribution.

RoundCross
    Standard round cross structures with various connectivity patterns.

Primitives
    Basic 3D geometric primitives (spheres, cylinders, cubes, etc.)
    for building more complex structures.

Primitives2D
    2D geometric primitives for planar structures and cross-sections.

Functions
---------

get_model(model, checkpoint='latest', device=None)
    Load a pretrained model by name or enum value. Returns a DeepSDFModel
    ready for inference.

list_available_models()
    Get a list of all available pretrained models.

The pretrained models are stored within the package and loaded on demand,
enabling quick prototyping and exploration without requiring training.

Examples
--------
Load and use a pretrained model::

    from DeepSDFStruct.pretrained_models import get_model, PretrainedModels
    import torch

    # Load a model
    model = get_model(PretrainedModels.RoundCross)

    # Generate geometry with latent code
    latent_code = torch.zeros(256)  # Use learned latent vector
    points = torch.rand(1000, 3)
    sdf_values = model(points, latent_code)

List available models::

    from DeepSDFStruct.pretrained_models import list_available_models

    models = list_available_models()
    for model in models:
        print(f"Available model: {model.value}")
"""

from enum import Enum
import importlib.resources
from DeepSDFStruct.deep_sdf.workspace import load_trained_model, load_latent_vectors
from DeepSDFStruct.deep_sdf.models import DeepSDFModel
import torch


class PretrainedModels(Enum):
    ChiAndCross = "chi_and_cross"
    AnalyticRoundCross = "analytic_round_cross"
    RoundCross = "round_cross"
    Primitives = "primitives"
    Primitives2D = "primitives_2d"


# Maps enum entries to file paths
main_dir = importlib.resources.files("DeepSDFStruct")
_MODEL_REGISTRY = {
    PretrainedModels.ChiAndCross: main_dir / "trained_models" / "chi_and_cross",
    PretrainedModels.AnalyticRoundCross: main_dir
    / "trained_models"
    / "analytic_round_cross",
    PretrainedModels.RoundCross: main_dir / "trained_models" / "round_cross",
    PretrainedModels.Primitives: main_dir / "trained_models" / "primitives",
    PretrainedModels.Primitives2D: main_dir / "trained_models" / "primitives_2d",
}


def get_model(
    model: str | PretrainedModels, checkpoint: str = "latest", device=None
) -> DeepSDFModel:
    """
    Load a pretrained model by name or enum.

    Args:
        model (str | PretrainedModels): model identifier
        checkpoint (str): checkpoint file name (default: 'latest')

    Returns:
        Trained PyTorch model
    """
    if isinstance(model, str):
        path = model
    else:
        model_enum = model
        path = _MODEL_REGISTRY.get(model_enum)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not path:
        raise ValueError(f"Model path {path} not found.")
    decoder = load_trained_model(path, checkpoint, device=device)
    latent_vectors = load_latent_vectors(path, checkpoint, device=device)
    decoder.eval()
    deep_sdf_model = DeepSDFModel(decoder, latent_vectors, device=device)
    return deep_sdf_model


def list_available_models():
    return list(PretrainedModels)
