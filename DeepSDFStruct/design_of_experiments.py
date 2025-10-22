import os
import pathlib
import json
import mlflow
from urllib.parse import urlparse
from DeepSDFStruct.deep_sdf.training import train_deep_sdf
import copy
from typing import Dict, Any, Union

package_path = pathlib.Path(__file__).parent


class ExperimentSpecifications(dict):
    """
    A dictionary-like class to hold and update experiment specifications.
    Supports recursive updates for nested dictionaries.
    Can be initialized from a dictionary or loaded from a file (JSON/YAML).
    """

    def __init__(self, specs: Union[Dict[str, Any], str, None] = None):
        """
        Initialize the ExperimentSpecifications.

        Args:
            specs: A dictionary of specifications, a file path (JSON/YAML),
                   or None for an empty specification set.
        """
        if isinstance(specs, str):
            loaded_specs = self._load_from_file(specs)
            super().__init__(copy.deepcopy(loaded_specs))
        else:
            super().__init__(copy.deepcopy(specs) if specs else {})

    @staticmethod
    def _load_from_file(filename: str) -> Dict[str, Any]:
        """
        Load specifications from a JSON or YAML file.
        """
        with open(filename, "r") as f:
            if filename.endswith(".json"):
                return json.load(f)
            else:
                raise ValueError("Unsupported file format. Use .json or .yaml/.yml")

    def save(self, filename: str) -> None:
        """
        Save current specifications to a JSON or YAML file.
        """
        with open(filename, "w") as f:
            if filename.endswith(".json"):
                json.dump(self, f, indent=4)

    def update(self, updates: Dict[str, Any]) -> None:
        """
        Recursively update the experiment specifications with new values.
        """

        def _recursive_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and isinstance(d.get(k), dict):
                    _recursive_update(d[k], v)
                else:
                    d[k] = v

        _recursive_update(self, updates)

    def copy(self):
        """Return a deep copy of the specifications."""
        return ExperimentSpecifications(copy.deepcopy(self))

    def __repr__(self):
        return f"ExperimentSpecifications({dict(self)})"


def create_experiment(exp_dir, specs):
    """
    Create a new experiment directory and specs.json file, copying defaults and applying overrides.

    Args:
        exp_dir (str): Path to new experiment directory.
        specs (dict): Dictionary containing the experiment specifications.
    """

    os.makedirs(exp_dir, exist_ok=True)

    # Write new specs.json
    specs_path = os.path.join(exp_dir, "specs.json")
    with open(specs_path, "w") as f:
        json.dump(specs, f, indent=4)

    return specs_path


def run_experiment(
    exp_name,
    data_dir,
    specs=None,
    device="cpu",
    mlflow_experiment_name="DeepSDFStruct_Experiment",
    tracking_uri="mlruns",
):
    """
    Creates an experiment with overrides, trains the model, and logs everything to MLflow.

    Args:
        exp_name (str): Name of the experiment folder.
        data_dir (str): Path to the dataset.
        specs (dict): Experiment specifications
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
        create_experiment(exp_dir, specs=specs)
        specs_path = os.path.join(exp_dir, "specs.json")
        mlflow.log_artifact(specs_path)

        with open(specs_path, "r") as f:
            specs = json.load(f)
        mlflow.log_params(specs)

        summary = train_deep_sdf(exp_dir, data_dir, device=device)
        mlflow.set_tags(summary)

        mlflow.log_metric("train_loss", summary["loss"])
    return summary["loss"]
