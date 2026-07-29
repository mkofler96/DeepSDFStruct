"""Tests for the SDF/mesh error metrics in ``DeepSDFStruct.deep_sdf.metrics``.

These are pure numeric/IO helpers, so the expected values below are computed by
hand rather than pinned to a previous run.
"""

import json

import numpy as np
import pytest
import pyvista as pv
import trimesh

from DeepSDFStruct.deep_sdf.metrics.error_metrics import (
    compute_metrics_from_vtp,
    compute_near_surface_metrics,
    find_scalar_array,
    load_vtp_with_sdf,
    save_error_vtp,
    save_json,
)
from DeepSDFStruct.deep_sdf.metrics.mesh_to_analytical import chamfer_distance

# Four samples, three of which are within a 0.1 cutoff of the surface.
_POINTS = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
_GT_SDF = np.array([0.0, 0.05, 0.5, -0.02])
_PRED_SDF = np.array([0.1, 0.05, 0.9, -0.05])
# Masked absolute errors are [0.1, 0.0, 0.03]; the 0.5 sample is dropped.
_ABS_ERR = np.array([0.1, 0.0, 0.03])


def _write_vtp(path, points, sdf, array_name="SDF"):
    mesh = pv.PolyData(points)
    mesh.point_data[array_name] = sdf
    mesh.save(path)
    return path


def test_near_surface_metrics_match_hand_computed_values():
    metrics = compute_near_surface_metrics(_GT_SDF, _PRED_SDF, cutoff=0.1)

    assert metrics["cutoff"] == 0.1
    assert metrics["num_total_samples"] == 4
    assert metrics["num_near_surface_samples"] == 3
    assert metrics["fraction_near_surface"] == pytest.approx(0.75)
    assert metrics["mae"] == pytest.approx(_ABS_ERR.mean())
    assert metrics["mae"] == pytest.approx(0.13 / 3)
    assert metrics["median"] == pytest.approx(0.03)
    assert metrics["max"] == pytest.approx(0.1)
    assert metrics["rmse"] == pytest.approx(np.sqrt(0.0109 / 3))
    # Linear-interpolated quantiles over the sorted errors [0.0, 0.03, 0.1].
    assert metrics["p05"] == pytest.approx(0.003)
    assert metrics["p95"] == pytest.approx(0.093)
    # rmse >= mae whenever the errors are not all equal.
    assert metrics["rmse"] > metrics["mae"]

    # Every value must be a plain float so the dict is JSON-serializable.
    for key, value in metrics.items():
        assert isinstance(value, (int, float)), key
    json.dumps(metrics)


def test_near_surface_metrics_cutoff_selects_samples():
    """A wider cutoff must keep more samples and change the reported error."""
    tight = compute_near_surface_metrics(_GT_SDF, _PRED_SDF, cutoff=0.03)
    wide = compute_near_surface_metrics(_GT_SDF, _PRED_SDF, cutoff=1.0)

    assert tight["num_near_surface_samples"] == 2  # only 0.0 and -0.02
    assert wide["num_near_surface_samples"] == 4
    assert wide["fraction_near_surface"] == 1.0
    # The dropped sample has the largest error, so including it raises the max.
    assert wide["max"] == pytest.approx(0.4)
    assert tight["max"] == pytest.approx(0.1)


def test_near_surface_metrics_accepts_unflattened_input():
    """Inputs are reshaped, so an (N, 1) column must behave like an (N,) vector."""
    flat = compute_near_surface_metrics(_GT_SDF, _PRED_SDF, cutoff=0.1)
    column = compute_near_surface_metrics(
        _GT_SDF.reshape(-1, 1), _PRED_SDF.reshape(-1, 1), cutoff=0.1
    )
    assert flat == column


def test_near_surface_metrics_raises_when_cutoff_excludes_everything():
    # Every sample sits well away from the surface for this cutoff.
    far_gt = np.array([0.5, -0.7, 0.9])
    with pytest.raises(ValueError, match="No samples satisfy"):
        compute_near_surface_metrics(far_gt, np.zeros(3), cutoff=0.1)


def test_find_scalar_array_prefers_canonical_names():
    mesh = pv.PolyData(_POINTS)
    mesh.point_data["something_else"] = _GT_SDF
    mesh.point_data["sdf"] = _GT_SDF
    mesh.point_data["SDF"] = _PRED_SDF

    values, name = find_scalar_array(mesh)
    # "SDF" outranks "sdf", which outranks any unrecognized name.
    assert name == "SDF"
    np.testing.assert_allclose(values, _PRED_SDF)


def test_find_scalar_array_falls_back_to_sole_array():
    mesh = pv.PolyData(_POINTS)
    mesh.point_data["unconventional_name"] = _GT_SDF

    values, name = find_scalar_array(mesh)
    assert name == "unconventional_name"
    np.testing.assert_allclose(values, _GT_SDF)


def test_find_scalar_array_raises_when_ambiguous():
    mesh = pv.PolyData(_POINTS)
    mesh.point_data["first"] = _GT_SDF
    mesh.point_data["second"] = _PRED_SDF

    with pytest.raises(ValueError, match="Could not determine SDF scalar array"):
        find_scalar_array(mesh)


def test_save_and_load_error_vtp_round_trip(tmp_path):
    path = tmp_path / "err.vtp"
    save_error_vtp(_POINTS, _ABS_ERR[:1].repeat(4), path, array_name="custom_name")

    points, values, name = load_vtp_with_sdf(path)
    assert name == "custom_name"
    np.testing.assert_allclose(points, _POINTS)
    np.testing.assert_allclose(values, np.full(4, 0.1))


def test_save_json_creates_parent_directories(tmp_path):
    path = tmp_path / "nested" / "deeper" / "metrics.json"
    save_json(path, {"mae": 0.5})

    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8")) == {"mae": 0.5}


def test_compute_metrics_from_vtp_end_to_end(tmp_path):
    gt = _write_vtp(tmp_path / "gt.vtp", _POINTS, _GT_SDF)
    pred = _write_vtp(tmp_path / "pred.vtp", _POINTS, _PRED_SDF)
    out_json = tmp_path / "out" / "metrics.json"

    metrics = compute_metrics_from_vtp(gt, pred, cutoff=0.1, output_json_path=out_json)

    assert metrics == compute_near_surface_metrics(_GT_SDF, _PRED_SDF, cutoff=0.1)
    assert json.loads(out_json.read_text(encoding="utf-8")) == metrics

    # The per-point error cloud is written next to the prediction, and covers
    # every sample rather than only the near-surface ones.
    error_vtp = pred.with_name("sdf_error.vtp")
    assert error_vtp.is_file()
    points, errors, _ = load_vtp_with_sdf(error_vtp)
    np.testing.assert_allclose(points, _POINTS)
    np.testing.assert_allclose(errors, np.abs(_PRED_SDF - _GT_SDF))


def test_compute_metrics_from_vtp_without_json_output(tmp_path):
    gt = _write_vtp(tmp_path / "gt.vtp", _POINTS, _GT_SDF)
    pred = _write_vtp(tmp_path / "pred.vtp", _POINTS, _PRED_SDF)

    metrics = compute_metrics_from_vtp(gt, pred, cutoff=0.1)

    assert metrics["num_near_surface_samples"] == 3
    assert not list(tmp_path.glob("*.json"))


def test_compute_metrics_from_vtp_rejects_mismatched_point_counts(tmp_path):
    gt = _write_vtp(tmp_path / "gt.vtp", _POINTS, _GT_SDF)
    pred = _write_vtp(tmp_path / "pred.vtp", _POINTS[:3], _PRED_SDF[:3])

    with pytest.raises(ValueError, match="different shapes"):
        compute_metrics_from_vtp(gt, pred, cutoff=0.1)


def test_compute_metrics_from_vtp_rejects_misaligned_points(tmp_path):
    """The metric assumes both clouds store values at identical coordinates."""
    shifted = _POINTS + np.array([0.0, 0.0, 0.5])
    gt = _write_vtp(tmp_path / "gt.vtp", _POINTS, _GT_SDF)
    pred = _write_vtp(tmp_path / "pred.vtp", shifted, _PRED_SDF)

    with pytest.raises(ValueError, match="not identical"):
        compute_metrics_from_vtp(gt, pred, cutoff=0.1)


def test_chamfer_distance_is_near_zero_for_identical_meshes():
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)

    np.random.seed(0)
    cd = chamfer_distance(sphere, sphere, n_surface_samples=2000)

    # Not exactly zero: the two point sets are independent surface samples, so
    # the residual is sampling noise on a unit sphere.
    assert cd >= 0.0
    assert cd < 0.01


def test_chamfer_distance_grows_with_separation():
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    distances = []
    for offset in (0.0, 0.1, 0.5):
        shifted = sphere.copy()
        shifted.apply_translation([offset, 0.0, 0.0])
        np.random.seed(0)
        distances.append(chamfer_distance(sphere, shifted, n_surface_samples=2000))

    assert distances == sorted(distances)
    # A half-radius translation must dominate the sampling noise by far.
    assert distances[2] > 10 * distances[0]


def test_chamfer_distance_is_symmetric():
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    box = trimesh.creation.box(extents=[1.0, 1.0, 1.0])

    np.random.seed(0)
    forward = chamfer_distance(sphere, box, n_surface_samples=4000)
    np.random.seed(0)
    backward = chamfer_distance(box, sphere, n_surface_samples=4000)

    assert forward == pytest.approx(backward, rel=0.15)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
