import os
import pathlib
import json
import mlflow
from urllib.parse import urlparse
from DeepSDFStruct.deep_sdf.training import train_deep_sdf


package_path = pathlib.Path(__file__).parent


def recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and k in d:
            recursive_update(d[k], v)
        else:
            d[k] = v


def create_experiment(
    exp_dir,
    overrides=None,
    base_spec_path=package_path / "trained_models" / "test_experiment" / "specs.json",
):
    """
    Create a new experiment directory and specs.json file, copying defaults and applying overrides.

    Args:
        exp_dir (str): Path to new experiment directory.
        overrides (dict): Hyperparameters to override from default specs.
        base_spec_path (str): Path to the default specs.json.
    """
    os.makedirs(exp_dir, exist_ok=True)

    # Load default specs
    with open(base_spec_path, "r") as f:
        specs = json.load(f)

    # Apply overrides
    if overrides:
        recursive_update(specs, overrides)

    # Write new specs.json
    specs_path = os.path.join(exp_dir, "specs.json")
    with open(specs_path, "w") as f:
        json.dump(specs, f, indent=4)

    return specs_path


def run_experiment(
    exp_name,
    data_dir,
    overrides=None,
    device="cpu",
    mlflow_experiment_name="DeepSDFStruct_Experiment",
    tracking_uri="mlruns",
):
    """
    Creates an experiment with overrides, trains the model, and logs everything to MLflow.

    Args:
        exp_name (str): Name of the experiment folder.
        data_dir (str): Path to the dataset.
        overrides (dict): Hyperparameters to override from base specs.
        device (str): Training device ('cpu' or 'cuda').
        mlflow_experiment_name (str): Name of the MLflow experiment.
        register_model (bool): If True, register the final model in MLflow Model Registry.
    """
    exp_dir = os.path.join("DeepSDFStruct/trained_models", exp_name)
    mlflow.set_tracking_uri(tracking_uri)
    # Start MLflow run
    mlflow.set_experiment(mlflow_experiment_name)
    with mlflow.start_run():
        parsed = urlparse(mlflow.get_artifact_uri())
        if parsed.path == "":
            raise NotImplementedError("Remote Tracking URI not implemented yet.")
        else:
            exp_dir = parsed.path
        create_experiment(exp_dir, overrides=overrides)
        specs_path = os.path.join(exp_dir, "specs.json")
        mlflow.log_artifact(specs_path)

        with open(specs_path, "r") as f:
            specs = json.load(f)
        mlflow.log_params(specs)

        summary = train_deep_sdf(exp_dir, data_dir, device=device)
        mlflow.set_tags(summary)

        if "loss" in summary.keys():
            mlflow.log_metric("train_loss", summary["loss"])
