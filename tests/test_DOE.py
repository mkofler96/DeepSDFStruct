from DeepSDFStruct.design_of_experiments import run_experiment, ExperimentSpecifications
from huggingface_hub import snapshot_download


def test_DOE():
    data_dir = snapshot_download(
        "mkofler/lattice_structure_unit_cells",
        repo_type="dataset",
        revision="b80339abc071df77ff81e8abc19ad4856d96ddbd",
    )
    specs = ExperimentSpecifications(
        "DeepSDFStruct/trained_models/test_experiment/specs.json"
    )
    # Define hyperparameter grid / overrides
    tuning_params = [
        {"NetworkSpecs": {"latent_in": [2], "dropout_prob": 0.2}},
        {"NetworkSpecs": {"latent_in": [3], "dropout_prob": 0.3}},
        {"NetworkSpecs": {"latent_in": [4], "dropout_prob": 0.4}},
    ]

    # Run experiments
    for i, overrides in enumerate(tuning_params):
        specs.update(overrides)
        exp_name = f"run_{i}"
        run_experiment(exp_name, data_dir, specs=specs, device="cpu")


if __name__ == "__main__":
    test_DOE()
