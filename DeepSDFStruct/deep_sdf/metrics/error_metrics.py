import json
from pathlib import Path

import numpy as np
import pyvista as pv


def save_error_vtp(points, error, filename, array_name="sdf_error"):
    mesh = pv.PolyData(points)
    mesh.point_data[array_name] = error
    mesh.save(filename)


def find_scalar_array(mesh):
    preferred_names = ["SDF", "sdf", "distance", "distances", "signed_distance"]

    for name in preferred_names:
        if name in mesh.point_data:
            return np.asarray(mesh.point_data[name]).reshape(-1), name

    if len(mesh.point_data) == 1:
        name = list(mesh.point_data.keys())[0]
        return np.asarray(mesh.point_data[name]).reshape(-1), name

    raise ValueError(
        "Could not determine SDF scalar array automatically. "
        f"Available point_data arrays: {list(mesh.point_data.keys())}"
    )


def load_vtp_with_sdf(path: Path):
    mesh = pv.read(path)
    points = np.asarray(mesh.points)
    sdf, sdf_name = find_scalar_array(mesh)
    return points, sdf, sdf_name


def compute_near_surface_metrics(gt_sdf, pred_sdf, cutoff):
    gt_sdf = np.asarray(gt_sdf).reshape(-1)
    pred_sdf = np.asarray(pred_sdf).reshape(-1)

    mask = np.abs(gt_sdf) <= cutoff
    n_total = gt_sdf.shape[0]
    n_kept = int(mask.sum())

    if n_kept == 0:
        raise ValueError(
            f"No samples satisfy |gt_sdf| <= {cutoff}. Try a larger cutoff."
        )

    diff = pred_sdf[mask] - gt_sdf[mask]
    abs_err = np.abs(diff)

    metrics = {
        "cutoff": float(cutoff),
        "num_total_samples": int(n_total),
        "num_near_surface_samples": int(n_kept),
        "fraction_near_surface": float(n_kept / n_total),
        "mae": float(np.mean(abs_err)),
        "median": float(np.median(abs_err)),
        "p05": float(np.quantile(abs_err, 0.05)),
        "p95": float(np.quantile(abs_err, 0.95)),
        "max": float(np.max(abs_err)),
        "rmse": float(np.sqrt(np.mean(diff**2))),
    }

    return metrics


def save_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def compute_metrics_from_vtp(
    gt_vtp_path, pred_vtp_path, cutoff, output_json_path=None, atol=1e-10
):
    gt_vtp_path = Path(gt_vtp_path)
    pred_vtp_path = Path(pred_vtp_path)

    gt_points, gt_sdf, gt_name = load_vtp_with_sdf(gt_vtp_path)
    pred_points, pred_sdf, pred_name = load_vtp_with_sdf(pred_vtp_path)

    if gt_points.shape != pred_points.shape:
        raise ValueError(
            f"Point clouds have different shapes: {gt_points.shape} vs {pred_points.shape}"
        )

    if not np.allclose(gt_points, pred_points, atol=atol, rtol=1e-10):
        raise ValueError(
            "Point coordinates are not identical. "
            "This evaluation assumes GT and prediction are stored at the same sample points."
        )

    metrics = compute_near_surface_metrics(gt_sdf, pred_sdf, cutoff)

    print("\nLoaded arrays")
    print("-------------")
    print(f"GT scalar array:   {gt_name}")
    print(f"Pred scalar array: {pred_name}")

    print("\nNear-surface SDF metrics")
    print("------------------------")
    print(f"cutoff:   {metrics['cutoff']}")
    print(
        f"kept:     {metrics['num_near_surface_samples']} / {metrics['num_total_samples']}"
    )
    print(f"mae:      {metrics['mae']:.8f}")
    print(f"median:   {metrics['median']:.8f}")
    print(f"p95:      {metrics['p95']:.8f}")
    print(f"max:      {metrics['max']:.8f}")
    print(f"rmse:     {metrics['rmse']:.8f}")

    diff = pred_sdf - gt_sdf
    abs_err = np.abs(diff)
    error_vtp_path = pred_vtp_path.with_name("sdf_error.vtp")
    save_error_vtp(gt_points, abs_err, error_vtp_path)
    print(f"\nSaved error VTP to: {error_vtp_path}")

    if output_json_path is not None:
        output_json_path = Path(output_json_path)
        save_json(output_json_path, metrics)
        print(f"\nSaved metrics to: {output_json_path}")

    return metrics
