from DeepSDFStruct.deep_sdf.training import (
    train_deep_sdf,
    reconstruct_meshs_from_latent,
    create_interpolated_meshes_from_latent,
)
from DeepSDFStruct.pretrained_models import get_model
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError
import pytest
import torch
import time

REVISION = "dbe58ebaa00057d5f15096c2b253c7efa91e19d3"


def snapshot_download_with_retry(*args, max_retries=3, **kwargs):
    """Download with retry on 429 rate limit errors."""
    for attempt in range(max_retries):
        try:
            return snapshot_download(*args, **kwargs)
        except HfHubHTTPError as e:
            if e.response and e.response.status_code == 429:
                if attempt < max_retries - 1:
                    wait_time = 60 * (attempt + 1)
                    print(
                        f"Rate limited (429). Waiting {wait_time}s before retry {attempt + 2}/{max_retries}"
                    )
                    time.sleep(wait_time)
                else:
                    raise
            raise


@pytest.fixture(scope="module", autouse=True)
def set_float32_dtype():
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cpu")
    yield


@pytest.fixture(scope="module")
def data_dir():
    return snapshot_download_with_retry(
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
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cpu")
    data_dir = snapshot_download_with_retry(
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
