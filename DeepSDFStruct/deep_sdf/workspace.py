#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import json
import os
import pathlib
import torch
from typing import TypedDict

from .networks.analytic_round_cross import RoundCrossDecoder
from .networks.deep_sdf_decoder import DeepSDFDecoder
from .networks.hierarchical_deep_sdf_decoder import HierachicalDeepSDFDecoder

screenshots_subdir = "Screenshots"
model_params_subdir = "ModelParameters"
optimizer_params_subdir = "OptimizerParameters"
latent_codes_subdir = "LatentCodes"
logs_filename = "Logs.pth"
logplot_filename = "Logs.png"
reconstructions_subdir = "Reconstructions"
reconstruction_meshes_subdir = "Meshes"
reconstruction_codes_subdir = "Codes"
specifications_filename = "specs.json"
data_source_map_filename = ".datasources.json"
evaluation_subdir = "Evaluation"
sdf_samples_subdir = "SdfSamples"
surface_samples_subdir = "SurfaceSamples"
normalization_param_subdir = "NormalizationParameters"
training_meshes_subdir = "TrainingMeshes"
experiment_summary_name = "training_summary.json"

# Map architecture name to Decoder class
ARCHITECTURES = {
    "analytic_round_cross": RoundCrossDecoder,
    "deep_sdf_decoder": DeepSDFDecoder,
    "hierarchical_deep_sdf_decoder": HierachicalDeepSDFDecoder,
}


def load_experiment_specifications(experiment_directory):

    filename = os.path.join(experiment_directory, specifications_filename)

    if not os.path.isfile(filename):
        raise Exception(
            f"The experiment directory ({experiment_directory}) does not include specifications file "
            + '"specs.json"'
        )
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def load_latent_vectors(experiment_directory, checkpoint, device):

    filename = os.path.join(
        experiment_directory, latent_codes_subdir, checkpoint + ".pth"
    )

    if not os.path.isfile(filename):
        raise Exception(
            f"The experiment directory ({experiment_directory}) does not include a latent code file"
            + f" for checkpoint '{checkpoint}'"
        )

    data = torch.load(filename, map_location=device, weights_only=True)

    if isinstance(data["latent_codes"], torch.Tensor):

        num_vecs = data["latent_codes"].size()[0]

        lat_vecs = []
        for i in range(num_vecs):
            lat_vecs.append(data["latent_codes"][i])

        return lat_vecs

    else:

        num_embeddings, embedding_dim = data["latent_codes"]["weight"].shape

        lat_vecs = torch.nn.Embedding(num_embeddings, embedding_dim, device=device)

        lat_vecs.load_state_dict(data["latent_codes"])

        return lat_vecs.weight.data.detach()


def load_model_parameters(
    experiment_directory, checkpoint, decoder: torch.nn.Module, device
):

    filename = os.path.join(
        experiment_directory, model_params_subdir, checkpoint + ".pth"
    )

    if not os.path.isfile(filename):
        raise Exception('model state dict "{}" does not exist'.format(filename))

    data = torch.load(filename, map_location=device, weights_only=True)

    decoder.load_state_dict(data["model_state_dict"])

    return data["epoch"]


def load_optimizer(
    experiment_directory, checkpoint, optimizer: torch.nn.Module, device
):

    filename = os.path.join(
        experiment_directory, optimizer_params_subdir, checkpoint + ".pth"
    )

    if not os.path.isfile(filename):
        raise Exception(f'optimizer state dict "{filename}" does not exist')

    data = torch.load(filename, map_location=device, weights_only=True)

    optimizer.load_state_dict(data["optimizer_state_dict"])

    return data["epoch"]


def get_data_source_map_filename(data_dir):
    return os.path.join(data_dir, data_source_map_filename)


def get_reconstructed_mesh_filename(
    experiment_dir,
    epoch,
    dataset,
    class_name,
    instance_name,
    create_dir=True,
    filetype="ply",
):
    fname_raw = os.path.join(
        experiment_dir,
        reconstructions_subdir,
        str(epoch),
        reconstruction_meshes_subdir,
        dataset,
        class_name,
        instance_name + "." + filetype,
    )
    fname = pathlib.Path(fname_raw)
    if not os.path.isdir(fname.parent) and create_dir:
        os.makedirs(fname.parent)
    return fname


def get_reconstructed_code_filename(
    experiment_dir, epoch, dataset, class_name, instance_name
):

    return os.path.join(
        experiment_dir,
        reconstructions_subdir,
        str(epoch),
        reconstruction_codes_subdir,
        dataset,
        class_name,
        instance_name + ".pth",
    )


def get_evaluation_dir(experiment_dir, checkpoint, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, evaluation_subdir, checkpoint)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_model_params_dir(experiment_dir, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, model_params_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_screenshots_dir(experiment_dir, create_if_nonexistent=True):

    dir = os.path.join(experiment_dir, screenshots_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_optimizer_params_dir(experiment_dir, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, optimizer_params_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_latent_codes_dir(experiment_dir, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, latent_codes_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_normalization_params_filename(
    data_dir, dataset_name, class_name, instance_name
):
    return os.path.join(
        data_dir,
        normalization_param_subdir,
        dataset_name,
        class_name,
        instance_name + ".npz",
    )


def init_decoder(experiment_specs, device, data_parallel):
    arch_name = experiment_specs["NetworkArch"]
    if arch_name not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {arch_name}")

    latent_size = experiment_specs["CodeLength"]
    DecoderClass = ARCHITECTURES[arch_name]

    decoder = DecoderClass(latent_size, **experiment_specs["NetworkSpecs"]).to(device)
    if data_parallel:
        decoder = torch.nn.DataParallel(decoder)
    return decoder


def load_trained_model(
    experiment_directory: str, checkpoint: str, device=None, data_parallel=False
):
    specs_filename = os.path.join(experiment_directory, "specs.json")
    with open(specs_filename, "r", encoding="utf-8") as f:
        experiment_specs = json.load(f)
    if device is None:
        device = get_default_device(device)

    filename = os.path.join(
        experiment_directory, model_params_subdir, checkpoint + ".pth"
    )

    if not os.path.isfile(filename):
        raise Exception('model state dict "{}" does not exist'.format(filename))

    data = torch.load(filename, map_location=device)
    decoder = init_decoder(experiment_specs, device, data_parallel)
    try:
        decoder.load_state_dict(data["model_state_dict"])
    except RuntimeError:
        state_dict = {}
        for k, v in data["model_state_dict"].items():
            new_key = k.replace("module.", "", 1) if k.startswith("module.") else k
            state_dict[new_key] = v
        decoder.load_state_dict(state_dict)
    decoder = decoder.to(device)
    return decoder


def get_default_device():

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def print_model_specifications(experiment_directory: str):
    specs = load_experiment_specifications(experiment_directory)
    print("Model Specifications:")
    for key in specs:
        print(f"  {key}: {specs[key]}")
    print("\n")


class ExperimentSummary(TypedDict):
    loss: float
    num_epochs: int
    timestamp: str
    host_name: str
    device: str
    training_duration: str
    data_dir: str
    version: str


def save_experiment_summary(experiment_directory: str, summary: ExperimentSummary):
    with open(os.path.join(experiment_directory, experiment_summary_name), "w") as f:
        json.dump(summary, f, indent=4)
