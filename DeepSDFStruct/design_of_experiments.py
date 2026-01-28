"""
Design of Experiments (DOE) for DeepSDF Training
================================================

This module provides tools for conducting systematic design of experiments
when training DeepSDF neural network models. It integrates with MLflow for
experiment tracking and supports automated hyperparameter sweeps.

Key Features
------------

ExperimentSpecifications
    A flexible dictionary-like class for managing experiment configurations
    with support for:
    - Loading specifications from JSON files
    - Recursive updates for nested parameter structures
    - Deep copying to prevent unintended modifications
    - Flattening nested configurations for logging
    - Saving configurations for reproducibility

MLflow Integration
    Automatic tracking of:
    - Training metrics (loss curves, validation scores)
    - Model checkpoints
    - Hyperparameters and configurations
    - Dataset information
    - Experiment metadata

The module simplifies the process of running large-scale hyperparameter
searches and ablation studies for DeepSDF model training, with built-in
versioning and reproducibility features.

Examples
--------
Run an experiment with custom specifications::

    from DeepSDFStruct.design_of_experiments import ExperimentSpecifications

    # Load base configuration
    specs = ExperimentSpecifications('config.json')

    # Update hyperparameters
    specs.update({
        'learning_rate': 0.001,
        'batch_size': 32,
        'num_epochs': 100
    })

    # Run training with tracking
    # train_with_specs(specs)

Create and modify experiment configurations::

    specs = ExperimentSpecifications({
        'model': {
            'layers': [512, 512, 512],
            'activation': 'relu'
        },
        'training': {
            'lr': 0.0005
        }
    })

    # Flatten for logging
    flat_params = specs.flatten()
    print(flat_params)  # {'model.layers': [...], 'model.activation': 'relu', ...}
"""

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

    def flatten(self, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
        """
        Flatten nested dictionary (including lists of dicts) into a single-level dict
        with dot-separated keys.
        """
        items = {}
        for k, v in self.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            if isinstance(v, dict):
                items.update(
                    ExperimentSpecifications(v).flatten(parent_key=new_key, sep=sep)
                )

            elif isinstance(v, list):
                # Check if the list contains dictionaries
                if all(isinstance(i, dict) for i in v):
                    for idx, elem in enumerate(v):
                        items.update(
                            ExperimentSpecifications(elem).flatten(
                                parent_key=f"{new_key}.{idx}", sep=sep
                            )
                        )
                else:
                    # Just convert the list/tuple to a string for MLflow
                    items[new_key] = str(v)

            else:
                items[new_key] = v

        return items

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
    batch_split=1,
    run_name=None,
    specs: ExperimentSpecifications | None = None,
    device="cpu",
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
    mlflow.set_tracking_uri(tracking_uri)
    # Start MLflow run
    mlflow.set_experiment(exp_name)
    with mlflow.start_run(run_name=run_name):
        parsed = urlparse(mlflow.get_artifact_uri())
        if parsed.path == "":
            raise NotImplementedError("Remote Tracking URI not implemented yet.")
        else:
            exp_dir = parsed.path
        create_experiment(exp_dir, specs=specs)
        specs_path = os.path.join(exp_dir, "specs.json")
        mlflow.log_artifact(specs_path)

        mlflow.log_params(specs.flatten())

        summary = train_deep_sdf(
            exp_dir, data_dir, device=device, batch_split=batch_split
        )
        mlflow.set_tags(summary)

        mlflow.log_metric("train_loss", summary["loss"])
    return summary["loss"]
