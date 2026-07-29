"""Tests for ``deep_sdf.training_latent_field.train``.

This is the spline-latent-field trainer: instead of one latent vector per scene,
each scene owns a B-spline whose control points are learned, evaluated at the
query position to produce the latent code the decoder sees.

The dataset is generated here rather than downloaded. Each scene is an analytic
sphere of a different radius, written in the on-disk layout ``SDFSamples``
expects, so a real training run happens with no network access and in a couple of
seconds. The experiment specification is the committed
``trained_models/test_experiment_latent_field/specs.json``, alongside the other
test experiments, so the same run can be reproduced by hand; tests copy it into a
tmp directory (optionally overriding keys) so training artifacts never land in
the repository.
"""

import json
import pathlib
import shutil

import numpy as np
import pytest
import torch

import DeepSDFStruct.deep_sdf.workspace as ws
from DeepSDFStruct.deep_sdf.training_latent_field import (
    export_training_latent_fields_to_stl,
    train,
)

SPECS_SOURCE = pathlib.Path(
    "DeepSDFStruct/trained_models/test_experiment_latent_field/specs.json"
)
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
    """Analytic sphere scenes in the SdfSamples/splits layout, built once."""
    root = tmp_path_factory.mktemp("latent_field_data")
    samples_dir = root / ws.sdf_samples_subdir / DATASET / CLASS_NAME
    samples_dir.mkdir(parents=True)
    (root / "splits").mkdir()

    rng = np.random.default_rng(0)
    for scene, radius in enumerate(RADII):
        # Draw from a pool and keep a balanced set, so every scene has plenty of
        # both signs: SamplesPerScene is split evenly between pos and neg.
        pool = rng.uniform(-1.0, 1.0, size=(8000, 3)).astype(np.float32)
        distance = (np.linalg.norm(pool, axis=1) - radius).astype(np.float32)
        rows = np.concatenate([pool, distance[:, None]], axis=1)
        pos = rows[distance > 0][:1000]
        neg = rows[distance <= 0][:1000]
        assert len(pos) >= 512 and len(neg) >= 512, (scene, len(pos), len(neg))
        np.savez(samples_dir / f"{scene}.npz", pos=pos, neg=neg)

    json.dump(
        {DATASET: {CLASS_NAME: [str(i) for i in range(len(RADII))]}},
        (root / "splits" / "train.json").open("w"),
    )
    return root


def _make_experiment(tmp_path, **overrides):
    """Copy the committed specification into ``tmp_path``, overriding keys."""
    specs = json.loads(SPECS_SOURCE.read_text())
    specs.update(overrides)
    experiment = tmp_path / "experiment"
    experiment.mkdir(exist_ok=True)
    (experiment / "specs.json").write_text(json.dumps(specs, indent=2))
    return experiment


def _loss_log(experiment):
    return torch.load(pathlib.Path(experiment) / ws.logs_filename, weights_only=False)[
        "loss"
    ]


def _latent_checkpoint(experiment, name="latest.pth"):
    return torch.load(
        pathlib.Path(experiment) / ws.latent_codes_subdir / name, weights_only=False
    )


def _control_points(experiment, name="latest.pth"):
    """Per-scene spline control points from a latent-field checkpoint.

    The state dict also carries the spline's knot vectors, degrees and
    parametric bounds, so the learnable tensors have to be picked out by name.
    """
    state = _latent_checkpoint(experiment, name)["latent_fields_state_dict"]
    return {
        key.split(".")[0]: value
        for key, value in state.items()
        if key.endswith("control_points")
    }


# --------------------------------------------------------------------------
# a full run
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained(tmp_path_factory, data_source):
    """One complete 4-epoch run, shared by the assertions below.

    ``DataSource`` is baked into the specification because
    ``export_training_latent_fields_to_stl`` reads it from there rather than
    taking it as an argument.
    """
    tmp_path = tmp_path_factory.mktemp("latent_field_run")
    experiment = _make_experiment(tmp_path, DataSource=str(data_source))
    summary = train(experiment, data_source=str(data_source), device="cpu")
    return experiment, summary


def test_train_returns_a_summary(trained):
    experiment, summary = trained

    assert summary["num_epochs"] == 4
    assert np.isfinite(summary["loss"])
    assert summary["device"] == "cpu"
    assert summary["data_dir"]
    assert summary["version"]
    # Also persisted next to the checkpoints.
    on_disk = json.loads((experiment / "training_summary.json").read_text())
    assert on_disk["num_epochs"] == 4


def test_train_writes_every_checkpoint_kind(trained):
    experiment, _ = trained

    # latest.pth for each of the three state dicts, plus the logs and the plot.
    assert (experiment / ws.model_params_subdir / "latest.pth").is_file()
    assert (experiment / ws.optimizer_params_subdir / "latest.pth").is_file()
    assert (experiment / ws.latent_codes_subdir / "latest.pth").is_file()
    assert (experiment / ws.logs_filename).is_file()
    assert (experiment / ws.logplot_filename).is_file()


def test_additional_snapshots_are_written(trained):
    """AdditionalSnapshots: [2] must produce epoch-2 checkpoints."""
    experiment, _ = trained

    assert (experiment / ws.model_params_subdir / "2.pth").is_file()
    assert (experiment / ws.optimizer_params_subdir / "2.pth").is_file()
    assert (experiment / ws.latent_codes_subdir / "2.pth").is_file()


def test_latent_field_checkpoint_holds_one_spline_per_scene(trained):
    experiment, _ = trained
    control_points = _control_points(experiment)

    # One ModuleList entry per scene.
    assert set(control_points) == {str(i) for i in range(len(RADII))}
    for value in control_points.values():
        # Tiling [1,1,1] at degree 1 -> 2**3 control points, each CodeLength wide.
        assert value.shape == (8, 2)
    # Dummy latent codes are kept so load_latent_vectors/get_model still work.
    assert _latent_checkpoint(experiment)["latent_codes"].shape == (len(RADII), 2)


def test_scenes_learn_distinct_latent_fields(trained):
    """Each sphere has a different radius, so the fields must not stay identical."""
    experiment, _ = trained
    control_points = _control_points(experiment)

    stacked = torch.stack([control_points[str(i)] for i in range(len(RADII))])
    spread = stacked.std(dim=0).max().item()
    assert spread > 0.0


def test_training_reduces_the_loss(trained):
    experiment, _ = trained
    losses = _loss_log(experiment)

    assert len(losses) > 0
    assert all(np.isfinite(losses))
    # Compare the first and last epoch's worth rather than single batches, which
    # are noisy at this size.
    per_epoch = max(1, len(losses) // 4)
    assert np.mean(losses[-per_epoch:]) < np.mean(losses[:per_epoch])


def test_decoder_checkpoint_reloads_through_workspace(trained):
    """The saved decoder must be loadable by the normal workspace helper."""
    experiment, _ = trained

    decoder = ws.load_trained_model(str(experiment), "latest", device="cpu")

    assert decoder.geom_dimension == 3
    out = decoder(torch.zeros(4, 2 + 3))
    assert out.shape == (4, 1)


# --------------------------------------------------------------------------
# resuming
# --------------------------------------------------------------------------


def test_continue_from_resumes_and_extends_the_logs(tmp_path, data_source):
    experiment = _make_experiment(tmp_path, NumEpochs=2)
    train(experiment, data_source=str(data_source), device="cpu")
    first_losses = _loss_log(experiment)

    # Same directory, more epochs, resuming from the saved state.
    specs = json.loads((experiment / "specs.json").read_text())
    specs["NumEpochs"] = 4
    (experiment / "specs.json").write_text(json.dumps(specs))
    summary = train(
        experiment, data_source=str(data_source), device="cpu", continue_from="latest"
    )

    assert summary["num_epochs"] == 4
    # Resumed rather than restarted: the log grew instead of being replaced.
    assert len(_loss_log(experiment)) > len(first_losses)


def test_continue_from_tolerates_a_missing_checkpoint(tmp_path, data_source):
    """Each load is individually guarded, so a bad name trains from scratch."""
    experiment = _make_experiment(tmp_path, NumEpochs=1)

    summary = train(
        experiment,
        data_source=str(data_source),
        device="cpu",
        continue_from="does_not_exist",
    )

    assert np.isfinite(summary["loss"])


def test_accepts_a_pathlike_experiment_directory(tmp_path, data_source):
    """experiment_directory is stringified before use, so a Path is fine."""
    experiment = _make_experiment(tmp_path, NumEpochs=1)

    summary = train(
        pathlib.Path(experiment), data_source=str(data_source), device="cpu"
    )

    assert np.isfinite(summary["loss"])


def test_data_source_falls_back_to_the_specification(tmp_path, data_source):
    """Omitting data_source reads DataSource out of specs.json."""
    experiment = _make_experiment(tmp_path, NumEpochs=1, DataSource=str(data_source))

    summary = train(experiment, device="cpu")

    assert summary["data_dir"] == str(data_source)


# --------------------------------------------------------------------------
# specification variants
# --------------------------------------------------------------------------


@pytest.mark.parametrize("loss_type", ["ClampedL1", "L1", "MSE", "clampedl1", "mse"])
def test_supported_loss_types(tmp_path, data_source, loss_type):
    experiment = _make_experiment(tmp_path, NumEpochs=1, LossType=loss_type)

    summary = train(experiment, data_source=str(data_source), device="cpu")

    assert np.isfinite(summary["loss"])


def test_rejects_unknown_loss_type(tmp_path, data_source):
    experiment = _make_experiment(tmp_path, LossType="huber")

    with pytest.raises(ValueError, match="Unknown LossType: huber"):
        train(experiment, data_source=str(data_source), device="cpu")


@pytest.mark.parametrize("target", ["evaluated", "control_points", "both"])
def test_code_regularization_targets(tmp_path, data_source, target):
    experiment = _make_experiment(
        tmp_path,
        NumEpochs=1,
        CodeRegularizationTarget=target,
        CodeRegularizationLambda=0.1,
    )

    summary = train(experiment, data_source=str(data_source), device="cpu")

    assert np.isfinite(summary["loss"])


def test_eikonal_regularization_runs(tmp_path, data_source):
    experiment = _make_experiment(tmp_path, NumEpochs=1, EikonalLambda=0.1)

    summary = train(experiment, data_source=str(data_source), device="cpu")

    assert np.isfinite(summary["loss"])


def test_code_bound_clamps_the_control_points(tmp_path, data_source):
    bound = 0.005  # below SplineInitStd, so the projection must bite
    # LogFrequency=1 so the single epoch actually writes latest.pth -- see
    # test_checkpoints_are_only_written_on_the_log_frequency below.
    experiment = _make_experiment(
        tmp_path, NumEpochs=1, LogFrequency=1, CodeBound=bound
    )

    train(experiment, data_source=str(data_source), device="cpu")

    for value in _control_points(experiment).values():
        assert value.abs().max().item() <= bound + 1e-6


def test_tiling_must_match_the_geometric_dimension(tmp_path, data_source):
    experiment = _make_experiment(tmp_path, Tiling=[2, 2])

    with pytest.raises(ValueError, match="Tiling length must match geom_dimension=3"):
        train(experiment, data_source=str(data_source), device="cpu")


def test_learning_rate_schedule_must_have_two_entries(tmp_path, data_source):
    """One schedule for the decoder, one for the spline control points."""
    experiment = _make_experiment(
        tmp_path, LearningRateSchedule=[{"Type": "Constant", "Value": 1e-3}]
    )

    with pytest.raises(ValueError, match="list of two schedules"):
        train(experiment, data_source=str(data_source), device="cpu")


def test_non_trivial_tiling_grows_the_control_lattice(tmp_path, data_source):
    """Tiling [2,2,2] gives 27 control points per scene instead of 8."""
    experiment = _make_experiment(
        tmp_path, NumEpochs=1, LogFrequency=1, Tiling=[2, 2, 2]
    )

    train(experiment, data_source=str(data_source), device="cpu")

    for value in _control_points(experiment).values():
        assert value.shape == (27, 2)


def test_checkpoints_are_only_written_on_the_log_frequency(tmp_path, data_source):
    """Saving is tied to LogFrequency, with no unconditional save at the end.

    A run whose NumEpochs is not a multiple of LogFrequency therefore finishes
    without a latest.pth, and the final epoch's weights are not persisted.
    """
    experiment = _make_experiment(tmp_path, NumEpochs=1, LogFrequency=2)

    summary = train(experiment, data_source=str(data_source), device="cpu")

    # The run completed and reported a summary...
    assert summary["num_epochs"] == 1
    assert (experiment / "training_summary.json").is_file()
    # ...but epoch 1 never hit the every-2-epochs save.
    assert not (experiment / ws.latent_codes_subdir / "latest.pth").exists()
    assert not (experiment / ws.model_params_subdir / "latest.pth").exists()


def test_grad_clip_and_batch_split_run(tmp_path, data_source):
    experiment = _make_experiment(
        tmp_path, NumEpochs=1, GradientClipNorm=1.0, BatchSplit=2
    )

    summary = train(experiment, data_source=str(data_source), device="cpu")

    assert np.isfinite(summary["loss"])


def test_enforce_minmax_disabled(tmp_path, data_source):
    """With clamping off, raw distances reach the loss unclamped."""
    experiment = _make_experiment(tmp_path, NumEpochs=1, EnforceMinMax=False)

    summary = train(experiment, data_source=str(data_source), device="cpu")

    assert np.isfinite(summary["loss"])


# --------------------------------------------------------------------------
# exporting the trained fields to STL
# --------------------------------------------------------------------------


def test_export_writes_one_stl_per_scene(trained, tmp_path):
    experiment, _ = trained
    out_dir = tmp_path / "stls"

    export_training_latent_fields_to_stl(
        experiment,
        checkpoint="latest.pth",
        out_dir=str(out_dir),
        N_base=8,
        device="cpu",
    )

    # Capped by the default max_scenes=3, out of 4 training scenes.
    produced = sorted(p.name for p in out_dir.glob("*.stl"))
    assert produced == ["scene_00000.stl", "scene_00001.stl", "scene_00002.stl"]
    for stl in out_dir.glob("*.stl"):
        assert stl.stat().st_size > 0


def test_export_respects_max_scenes(trained, tmp_path):
    experiment, _ = trained
    out_dir = tmp_path / "one"

    export_training_latent_fields_to_stl(
        experiment,
        checkpoint="latest.pth",
        out_dir=str(out_dir),
        N_base=8,
        device="cpu",
        max_scenes=1,
    )

    assert sorted(p.name for p in out_dir.glob("*.stl")) == ["scene_00000.stl"]


def test_export_produces_a_readable_mesh(trained, tmp_path):
    """The STL must load back as a non-empty triangle mesh."""
    trimesh = pytest.importorskip("trimesh")
    experiment, _ = trained
    out_dir = tmp_path / "mesh"

    export_training_latent_fields_to_stl(
        experiment,
        checkpoint="latest.pth",
        out_dir=str(out_dir),
        N_base=12,
        device="cpu",
        max_scenes=1,
    )

    mesh = trimesh.load(out_dir / "scene_00000.stl")
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0


def test_export_defaults_to_a_directory_inside_the_experiment(trained):
    experiment, _ = trained

    export_training_latent_fields_to_stl(
        experiment, checkpoint="latest.pth", N_base=8, device="cpu", max_scenes=1
    )

    default_dir = experiment / "reconstructions_latent_field"
    assert (default_dir / "scene_00000.stl").is_file()


@pytest.mark.parametrize("checkpoint", ["latest", "latest.pth"])
def test_export_accepts_the_checkpoint_with_or_without_extension(
    trained, tmp_path, checkpoint
):
    """load_trained_model wants no extension, save_latent_fields writes one."""
    experiment, _ = trained
    out_dir = tmp_path / checkpoint.replace(".", "_")

    export_training_latent_fields_to_stl(
        experiment,
        checkpoint=checkpoint,
        out_dir=str(out_dir),
        N_base=8,
        device="cpu",
        max_scenes=1,
    )

    assert (out_dir / "scene_00000.stl").is_file()


def test_export_can_skip_existing_files(trained, tmp_path):
    experiment, _ = trained
    out_dir = tmp_path / "skip"
    out_dir.mkdir()
    sentinel = out_dir / "scene_00000.stl"
    sentinel.write_text("not really an stl")

    export_training_latent_fields_to_stl(
        experiment,
        checkpoint="latest.pth",
        out_dir=str(out_dir),
        N_base=8,
        device="cpu",
        max_scenes=1,
        overwrite=False,
    )

    # Left untouched rather than regenerated.
    assert sentinel.read_text() == "not really an stl"


def test_export_overwrites_by_default(trained, tmp_path):
    experiment, _ = trained
    out_dir = tmp_path / "overwrite"
    out_dir.mkdir()
    sentinel = out_dir / "scene_00000.stl"
    sentinel.write_text("stale")

    export_training_latent_fields_to_stl(
        experiment,
        checkpoint="latest.pth",
        out_dir=str(out_dir),
        N_base=8,
        device="cpu",
        max_scenes=1,
    )

    assert sentinel.read_text() != "stale"
    assert sentinel.stat().st_size > 0


def test_export_rejects_a_tiling_that_mismatches_the_decoder(trained, tmp_path):
    experiment, _ = trained
    # Same trained checkpoints, but a specification whose tiling is 2D.
    broken = tmp_path / "broken"
    broken.mkdir()
    shutil.copytree(experiment, broken / "experiment")
    specs = json.loads((broken / "experiment" / "specs.json").read_text())
    specs["Tiling"] = [2, 2]
    (broken / "experiment" / "specs.json").write_text(json.dumps(specs))

    with pytest.raises(ValueError, match="Tiling length must match geom_dimension=3"):
        export_training_latent_fields_to_stl(
            broken / "experiment", checkpoint="latest.pth", device="cpu"
        )


def test_committed_specification_stays_in_sync():
    """The checked-in specs.json must remain a valid latent-field experiment."""
    specs = json.loads(SPECS_SOURCE.read_text())

    assert specs["NetworkArch"] in ws.ARCHITECTURES
    assert len(specs["Tiling"]) == specs["NetworkSpecs"]["geom_dimension"]
    assert len(specs["LearningRateSchedule"]) == 2
    assert len(specs["BoundsParamSpace"]) == 2
    # Every scene contributes SamplesPerScene rows split evenly pos/neg.
    assert specs["SamplesPerScene"] % 2 == 0
    assert specs["ScenesPerBatch"] <= len(RADII)


def test_experiment_directory_is_not_polluted_by_the_tests():
    """Artifacts belong in tmp: only specs.json is tracked for this experiment."""
    committed = SPECS_SOURCE.parent
    assert sorted(p.name for p in committed.iterdir()) == ["specs.json"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
