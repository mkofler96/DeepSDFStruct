from DeepSDFStruct.deep_sdf.training import (
    train_deep_sdf,
    reconstruct_meshs_from_latent,
    create_interpolated_meshes_from_latent,
)
from DeepSDFStruct.pretrained_models import get_model
from huggingface_hub import snapshot_download
import pytest
import torch

REVISION = "dbe58ebaa00057d5f15096c2b253c7efa91e19d3"

torch.set_default_dtype(torch.float32)
torch.set_default_device("cpu")


@pytest.fixture(scope="module")
def data_dir():
    return snapshot_download(
        "mkofler/lattice_structure_unit_cells",
        repo_type="dataset",
        revision=REVISION,
        ignore_patterns=["*.stl", "**/*.stl"],
    )


def test_train_homogenization_model(data_dir):
    exp_dir = "DeepSDFStruct/trained_models/test_experiment_homogenization"

    device = "cpu"
    train_deep_sdf(exp_dir, data_dir, device=device)


def test_train_hierarchical_model(data_dir):
    exp_dir = "DeepSDFStruct/trained_models/test_experiment_hierarchical"

    device = "cpu"
    train_deep_sdf(exp_dir, data_dir, device=device)


def test_train_model(data_dir):
    exp_dir = "DeepSDFStruct/trained_models/test_experiment"

    device = "cpu"
    train_deep_sdf(exp_dir, data_dir, device=device)


def test_continue_from(data_dir):
    exp_dir = "DeepSDFStruct/trained_models/test_experiment"

    device = "cpu"
    train_deep_sdf(exp_dir, data_dir, device=device, continue_from="1")


def test_latent_recon():
    exp_dir = "DeepSDFStruct/trained_models/analytic_round_cross"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    reconstruct_meshs_from_latent(exp_dir, filetype="obj", device=device)
    create_interpolated_meshes_from_latent(exp_dir, [1, 2, 3], 4, device=device)


def test_cpp_file_export():
    exp_dir = "DeepSDFStruct/trained_models/analytic_round_cross"
    model = get_model(exp_dir)
    model.export_libtorch_executable("tests/tmp_outputs/test_cpp_model.pt")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("error")
    data_dir = snapshot_download(
        "mkofler/lattice_structure_unit_cells",
        repo_type="dataset",
        revision=REVISION,
        ignore_patterns=["*.stl", "**/*.stl"],
    )
    test_train_homogenization_model(data_dir)
    test_train_hierarchical_model(data_dir)
    test_train_model(data_dir)
    test_continue_from(data_dir)
    test_latent_recon()
    test_cpp_file_export()
