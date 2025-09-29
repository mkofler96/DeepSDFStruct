from DeepSDFStruct.deep_sdf.training import (
    train_deep_sdf,
    reconstruct_meshs_from_latent,
    create_interpolated_meshes_from_latent,
)
from huggingface_hub import snapshot_download
import torch


def test_train_model():
    data_dir = snapshot_download(
        "mkofler/lattice_structure_unit_cells",
        repo_type="dataset",
        revision="b80339abc071df77ff81e8abc19ad4856d96ddbd",
    )
    exp_dir = "DeepSDFStruct/trained_models/test_experiment"

    device = "cpu"
    torch.set_default_device("cpu")
    train_deep_sdf(exp_dir, data_dir, device=device)


def test_latent_recon():
    exp_dir = "DeepSDFStruct/trained_models/test_experiment"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    reconstruct_meshs_from_latent(exp_dir, filetype="obj", device=device)
    create_interpolated_meshes_from_latent(exp_dir, [1, 2, 3], 4, device=device)


if __name__ == "__main__":
    test_latent_recon()
    test_train_model()
