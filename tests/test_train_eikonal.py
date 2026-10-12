"""Tests for the optional Eikonal loss term in ``deep_sdf.training.train_deep_sdf``.

The dataset is a handful of analytic spheres written in the on-disk layout
``SDFSamples`` expects, so these run without network access. The experiment
specification is the committed ``trained_models/test_experiment/specs.json``,
copied into a tmp directory with keys overridden.
"""

import json
import pathlib

import numpy as np
import pytest
import torch

import DeepSDFStruct.deep_sdf.workspace as ws
from DeepSDFStruct.deep_sdf.training import load_logs, train_deep_sdf

SPECS_SOURCE = pathlib.Path("DeepSDFStruct/trained_models/test_experiment/specs.json")
RADII = (0.6, 0.7, 0.8, 0.9)
DATASET, CLASS_NAME = "synthetic", "spheres"


@pytest.fixture(autouse=True)
def _float32():
    saved = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(saved)


@pytest.fixture(scope="module")
def data_source(tmp_path_factory):
    root = tmp_path_factory.mktemp("eikonal_data")
    samples_dir = root / ws.sdf_samples_subdir / DATASET / CLASS_NAME
    samples_dir.mkdir(parents=True)
    (root / "splits").mkdir()

    rng = np.random.default_rng(0)
    for scene, radius in enumerate(RADII):
        pool = rng.uniform(-1.0, 1.0, size=(4000, 3)).astype(np.float32)
        distance = (np.linalg.norm(pool, axis=1) - radius).astype(np.float32)
        rows = np.concatenate([pool, distance[:, None]], axis=1)
        np.savez(
            samples_dir / f"{scene}.npz",
            pos=rows[distance > 0][:500],
            neg=rows[distance <= 0][:500],
        )

    json.dump(
        {DATASET: {CLASS_NAME: [str(i) for i in range(len(RADII))]}},
        (root / "splits" / "train.json").open("w"),
    )
    return root


def _make_experiment(tmp_path, **overrides):
    specs = json.loads(SPECS_SOURCE.read_text())
    specs.update(
        TrainSplit="splits/train.json",
        TestSplit="splits/train.json",
        SamplesPerScene=256,
        NumEpochs=2,
        LogFrequency=2,
        AdditionalSnapshots=[],
    )
    specs.update(overrides)
    experiment = tmp_path / "experiment"
    experiment.mkdir(parents=True, exist_ok=True)
    (experiment / "specs.json").write_text(json.dumps(specs, indent=2))
    return experiment


@pytest.mark.parametrize("batch_split", [1, 2])
def test_eikonal_loss_runs(tmp_path, data_source, batch_split):
    experiment = _make_experiment(tmp_path, EikonalLambda=0.1)

    summary = train_deep_sdf(
        experiment, str(data_source), batch_split=batch_split, device="cpu"
    )

    assert np.isfinite(summary["loss"])


def test_eikonal_loss_increases_the_logged_loss(tmp_path, data_source):
    """With identical seeds, the first logged batch loss differs only by the
    Eikonal term, which is non-negative."""
    without = _make_experiment(tmp_path / "a", NumEpochs=1, LogFrequency=1)
    with_eik = _make_experiment(
        tmp_path / "b", NumEpochs=1, LogFrequency=1, EikonalLambda=10.0
    )

    train_deep_sdf(without, str(data_source), device="cpu")
    train_deep_sdf(with_eik, str(data_source), device="cpu")

    loss_without = load_logs(without)[0]
    loss_with = load_logs(with_eik)[0]
    assert loss_with[0] > loss_without[0]
