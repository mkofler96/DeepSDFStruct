from DeepSDFStruct.deep_sdf.train_deep_sdf import train_deep_sdf
from huggingface_hub import snapshot_download


def test_train_model():
    data_dir = snapshot_download(
        "mkofler/lattice_structure_unit_cells", repo_type="dataset"
    )
    train_deep_sdf("DeepSDFStruct/trained_models/test_experiment", data_dir)


if __name__ == "__main__":
    test_train_model()
