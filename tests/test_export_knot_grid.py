"""Tests for ``DeepSDFStruct.export_knot_grid``.

These are ParaView exporters, so each test writes a real file and reads it back
rather than inspecting intermediate arrays: the point of the module is that the
result opens correctly and carries the right metadata.
"""

import numpy as np
import pytest
import pyvista as pv
import splinepy
import torch

from DeepSDFStruct.export_knot_grid import (
    _greville_1d,
    export_control_lattice_paramspace,
    export_control_lattice_physical,
    export_control_points,
    export_control_volume_physical,
    export_design_volume_paramspace,
    export_knot_grid_paramspace,
)


def _make_spline(
    tiling=(2, 2, 2), degrees=(1, 1, 1), lo=(0.0, 0.0, 0.0), hi=(1.0, 1.0, 1.0)
):
    """A clamped trivariate B-spline subdivided into ``tiling`` knot spans."""
    knot_vectors = [
        [float(lo[d])] * (degrees[d] + 1) + [float(hi[d])] * (degrees[d] + 1)
        for d in range(3)
    ]
    ncp = int(np.prod([p + 1 for p in degrees]))
    spline = splinepy.BSpline(
        list(degrees), knot_vectors, [[0.0, 0.0, 0.0] for _ in range(ncp)]
    )
    for dim, n_box in enumerate(tiling):
        if n_box > 1:
            spline.insert_knots(dim, np.linspace(lo[dim], hi[dim], n_box + 1)[1:-1])
    return spline


def _lattice_points(n_per_dim, order="F"):
    """Control points laid out on an integer grid in the given flattening order."""
    n0, n1, n2 = n_per_dim
    ids = np.arange(n0 * n1 * n2)
    I, J, K = np.unravel_index(ids, (n0, n1, n2), order=order)
    return np.column_stack([I, J, K]).astype(float)


# --------------------------------------------------------------------------
# _greville_1d
# --------------------------------------------------------------------------


def test_greville_1d_degree_one_returns_interior_knots():
    # Clamped linear knot vector over [0, 1] with one interior knot.
    U = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
    # n = len(U) - p - 1 = 3, and for p == 1 each abscissa is a single knot.
    np.testing.assert_allclose(_greville_1d(U, 1), [0.0, 0.5, 1.0])


def test_greville_1d_degree_two_averages_p_knots():
    U = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    # n = 3; averages of two consecutive knots each.
    np.testing.assert_allclose(_greville_1d(U, 2), [0.0, 0.5, 1.0])


def test_greville_1d_degree_zero_uses_span_midpoints():
    U = np.array([0.0, 0.5, 1.0])
    # p == 0 takes its own branch: midpoint of each span.
    np.testing.assert_allclose(_greville_1d(U, 0), [0.25, 0.75])


def test_greville_1d_rejects_too_short_knot_vector():
    with pytest.raises(ValueError, match="Invalid knot vector length"):
        _greville_1d(np.array([0.0, 1.0]), 3)


def test_greville_1d_count_matches_control_points():
    """One abscissa per basis function, i.e. per control point in that axis."""
    spline = _make_spline(tiling=(3, 1, 1))
    degrees = np.array(spline.degrees, dtype=int)
    counts = [
        len(_greville_1d(np.asarray(kv, dtype=float), degrees[d]))
        for d, kv in enumerate(spline.knot_vectors)
    ]
    assert int(np.prod(counts)) == spline.control_points.shape[0]


# --------------------------------------------------------------------------
# export_knot_grid_paramspace
# --------------------------------------------------------------------------


def test_export_knot_grid_paramspace_builds_full_wireframe(tmp_path):
    path = tmp_path / "knots.vtp"
    export_knot_grid_paramspace(_make_spline(tiling=(2, 2, 2)), str(path))

    grid = pv.read(path)
    # 3 unique knots per axis -> 27 lattice nodes, deduplicated by get_point_id.
    assert grid.n_points == 27
    # One polyline per axis-aligned family: 3*3 lines in each of 3 directions.
    assert grid.n_lines == 27
    bounds = np.asarray(grid.bounds).reshape(3, 2)
    np.testing.assert_allclose(bounds[:, 0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(bounds[:, 1], [1.0, 1.0, 1.0])


def test_export_knot_grid_paramspace_handles_anisotropic_tiling(tmp_path):
    path = tmp_path / "knots.vtp"
    export_knot_grid_paramspace(_make_spline(tiling=(1, 2, 3)), str(path))

    grid = pv.read(path)
    nu, nv, nw = 2, 3, 4  # unique knots = tiling + 1
    assert grid.n_points == nu * nv * nw
    assert grid.n_lines == nv * nw + nu * nw + nu * nv


def test_export_knot_grid_paramspace_respects_bounds(tmp_path):
    path = tmp_path / "knots.vtp"
    spline = _make_spline(tiling=(2, 2, 2), lo=(-1.0, 0.0, 2.0), hi=(1.0, 4.0, 3.0))
    export_knot_grid_paramspace(spline, str(path))

    bounds = np.asarray(pv.read(path).bounds).reshape(3, 2)
    np.testing.assert_allclose(bounds[:, 0], [-1.0, 0.0, 2.0])
    np.testing.assert_allclose(bounds[:, 1], [1.0, 4.0, 3.0])


# --------------------------------------------------------------------------
# export_control_lattice_paramspace
# --------------------------------------------------------------------------


def test_export_control_lattice_paramspace_writes_index_metadata(tmp_path):
    path = tmp_path / "lattice.vtp"
    export_control_lattice_paramspace(_make_spline(tiling=(2, 2, 2)), str(path))

    cloud = pv.read(path)
    assert cloud.n_points == 27
    assert set(cloud.point_data) == {"id", "i", "j", "k"}
    np.testing.assert_array_equal(cloud["id"], np.arange(27))
    # Default order="F" makes i vary fastest.
    np.testing.assert_array_equal(cloud["i"][:3], [0, 1, 2])
    assert cloud["i"].max() == 2 and cloud["j"].max() == 2 and cloud["k"].max() == 2
    # No locked_idx given -> no locked array at all.
    assert "locked" not in cloud.point_data


def test_export_control_lattice_paramspace_order_c_varies_k_fastest(tmp_path):
    path = tmp_path / "lattice.vtp"
    export_control_lattice_paramspace(
        _make_spline(tiling=(2, 2, 2)), str(path), order="C"
    )

    cloud = pv.read(path)
    np.testing.assert_array_equal(cloud["k"][:3], [0, 1, 2])
    np.testing.assert_array_equal(cloud["i"][:3], [0, 0, 0])


def test_export_control_lattice_paramspace_marks_locked_points(tmp_path):
    path = tmp_path / "lattice.vtp"
    export_control_lattice_paramspace(
        _make_spline(tiling=(2, 2, 2)), str(path), locked_idx=[0, 5, 26]
    )

    cloud = pv.read(path)
    locked = cloud["locked"]
    assert locked.dtype == np.int32
    assert locked.sum() == 3
    np.testing.assert_array_equal(np.flatnonzero(locked), [0, 5, 26])


def test_export_control_lattice_paramspace_accepts_torch_locked_idx(tmp_path):
    path = tmp_path / "lattice.vtp"
    export_control_lattice_paramspace(
        _make_spline(tiling=(2, 2, 2)),
        str(path),
        locked_idx=torch.tensor([1, 2], dtype=torch.int64),
    )

    np.testing.assert_array_equal(np.flatnonzero(pv.read(path)["locked"]), [1, 2])


def test_export_control_lattice_paramspace_omits_locked_when_empty(tmp_path):
    path = tmp_path / "lattice.vtp"
    export_control_lattice_paramspace(
        _make_spline(tiling=(2, 2, 2)), str(path), locked_idx=[]
    )

    assert "locked" not in pv.read(path).point_data


@pytest.mark.parametrize("bad", [[-1], [27], [0, 999]])
def test_export_control_lattice_paramspace_rejects_out_of_range_locked_idx(
    tmp_path, bad
):
    with pytest.raises(IndexError, match="locked_idx out of bounds"):
        export_control_lattice_paramspace(
            _make_spline(tiling=(2, 2, 2)), str(tmp_path / "x.vtp"), locked_idx=bad
        )


# --------------------------------------------------------------------------
# export_design_volume_paramspace
# --------------------------------------------------------------------------


def test_export_design_volume_paramspace_writes_structured_grid(tmp_path):
    path = tmp_path / "volume.vts"
    export_design_volume_paramspace(_make_spline(tiling=(2, 2, 2)), path)

    grid = pv.read(path)
    assert grid.n_points == 27
    assert sorted(grid.dimensions) == [3, 3, 3]
    assert set(grid.point_data) == {"id", "i", "j", "k", "on_boundary"}
    # A 3x3x3 block has 27 nodes, of which only the centre is interior.
    assert grid["on_boundary"].sum() == 26


def test_export_design_volume_paramspace_all_boundary_when_thin(tmp_path):
    path = tmp_path / "volume.vts"
    export_design_volume_paramspace(_make_spline(tiling=(1, 1, 1)), path)

    grid = pv.read(path)
    assert grid.n_points == 8
    # Every corner of a 2x2x2 block lies on the boundary.
    assert grid["on_boundary"].sum() == 8


# --------------------------------------------------------------------------
# export_control_lattice_physical
# --------------------------------------------------------------------------


def test_export_control_lattice_physical_edge_count(tmp_path):
    path = tmp_path / "phys.vtp"
    n = (2, 2, 2)
    export_control_lattice_physical(_lattice_points(n), n, path)

    poly = pv.read(path)
    assert poly.n_points == 8
    # The 12 edges of a cube: (n0-1)*n1*n2 + n0*(n1-1)*n2 + n0*n1*(n2-1).
    assert poly.n_lines == 12
    assert set(poly.point_data) == {"id", "i", "j", "k"}


def test_export_control_lattice_physical_accepts_torch_input(tmp_path):
    path = tmp_path / "phys.vtp"
    n = (2, 3, 2)
    points = torch.tensor(_lattice_points(n), dtype=torch.float32)
    export_control_lattice_physical(points, n, path)

    poly = pv.read(path)
    assert poly.n_points == 12
    assert poly.n_lines == 1 * 3 * 2 + 2 * 2 * 2 + 2 * 3 * 1


def test_export_control_lattice_physical_boundary_only_drops_interior(tmp_path):
    full = tmp_path / "full.vtp"
    cage = tmp_path / "cage.vtp"
    n = (3, 3, 3)
    points = _lattice_points(n)
    export_control_lattice_physical(points, n, full)
    export_control_lattice_physical(points, n, cage, boundary_only=True)

    full_poly, cage_poly = pv.read(full), pv.read(cage)
    assert full_poly.n_points == 27
    # Only the single centre node is interior.
    assert cage_poly.n_points == 26
    # 54 axis edges total, minus the 6 that pass through the centre node.
    assert full_poly.n_lines == 54
    assert cage_poly.n_lines == 48
    # `id` still refers to the original full-lattice index after remapping.
    assert 13 not in set(cage_poly["id"])
    assert cage_poly["id"].max() == 26


def test_export_control_lattice_physical_degenerate_single_point(tmp_path):
    """A 1x1x1 lattice has no edges at all, so the lines branch is skipped."""
    path = tmp_path / "one.vtp"
    export_control_lattice_physical(np.zeros((1, 3)), (1, 1, 1), path)

    poly = pv.read(path)
    assert poly.n_points == 1
    assert poly.n_lines == 0


def test_export_control_lattice_physical_rejects_wrong_point_count(tmp_path):
    with pytest.raises(ValueError, match="implies 8"):
        export_control_lattice_physical(np.zeros((7, 3)), (2, 2, 2), tmp_path / "x.vtp")


# --------------------------------------------------------------------------
# export_control_volume_physical
# --------------------------------------------------------------------------


def test_export_control_volume_physical_writes_hex_volume(tmp_path):
    path = tmp_path / "vol.vts"
    n = (3, 3, 3)
    export_control_volume_physical(_lattice_points(n), n, path)

    grid = pv.read(path)
    assert grid.n_points == 27
    assert grid.dimensions == (3, 3, 3)
    assert grid["on_boundary"].sum() == 26
    # No undeformed reference -> no displacement arrays.
    assert "displacement" not in grid.point_data


def test_export_control_volume_physical_computes_displacement(tmp_path):
    path = tmp_path / "vol.vts"
    n = (2, 2, 2)
    undeformed = _lattice_points(n)
    deformed = undeformed + np.array([0.0, 3.0, 4.0])

    export_control_volume_physical(deformed, n, path, undeformed=undeformed)

    grid = pv.read(path)
    np.testing.assert_allclose(grid["displacement"], np.tile([0.0, 3.0, 4.0], (8, 1)))
    # |(0, 3, 4)| == 5 for every node.
    np.testing.assert_allclose(grid["displacement_mag"], np.full(8, 5.0))


def test_export_control_volume_physical_accepts_torch_tensors(tmp_path):
    path = tmp_path / "vol.vts"
    n = (2, 2, 2)
    undeformed = torch.tensor(_lattice_points(n), dtype=torch.float32)
    deformed = undeformed + 1.0

    export_control_volume_physical(deformed, n, path, undeformed=undeformed)

    grid = pv.read(path)
    np.testing.assert_allclose(
        grid["displacement_mag"], np.full(8, np.sqrt(3.0)), rtol=1e-6
    )


def test_export_control_volume_physical_requires_fortran_order(tmp_path):
    with pytest.raises(NotImplementedError, match="order='F'"):
        export_control_volume_physical(
            _lattice_points((2, 2, 2)), (2, 2, 2), tmp_path / "x.vts", order="C"
        )


def test_export_control_volume_physical_rejects_wrong_point_count(tmp_path):
    with pytest.raises(ValueError, match="implies 8"):
        export_control_volume_physical(np.zeros((5, 3)), (2, 2, 2), tmp_path / "x.vts")


def test_export_control_volume_physical_rejects_mismatched_undeformed(tmp_path):
    n = (2, 2, 2)
    with pytest.raises(ValueError, match="does not match control_points"):
        export_control_volume_physical(
            _lattice_points(n), n, tmp_path / "x.vts", undeformed=np.zeros((4, 3))
        )


# --------------------------------------------------------------------------
# export_control_points
# --------------------------------------------------------------------------


def test_export_control_points_round_trip(tmp_path):
    path = tmp_path / "cps.vtp"
    points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-1.0, 0.5, 0.25]])

    export_control_points(points, path)

    cloud = pv.read(path)
    assert cloud.n_points == 3
    np.testing.assert_allclose(cloud.points, points)


def test_export_control_points_accepts_torch_tensor(tmp_path):
    path = tmp_path / "cps.vtp"
    points = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)

    # A tensor on the autograd graph must be detached rather than raising.
    export_control_points(points, path)

    np.testing.assert_allclose(pv.read(path).points, [[1.0, 2.0, 3.0]])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
