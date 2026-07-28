"""Tests for the scene-construction helpers in ``generate_primitive_dataset``.

``generate_primitive_dataset`` itself writes a full sampled dataset to disk and
is exercised by generating data, not by the test suite. The geometry helpers it
composes are deterministic given a seeded generator and are covered here.
"""

import numpy as np
import pytest
import torch
import trimesh

from DeepSDFStruct.deep_sdf.generate_primitive_dataset import (
    _AnisoScaledSDF,
    _build_scene,
    _filter_to_bounds,
    _make_primitive,
    _place,
    _random_rotation_matrix,
)
from DeepSDFStruct.sampling import SampledSDF
from DeepSDFStruct.sdf_primitives import SphereSDF

UNIT_BOUNDS = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])


@pytest.fixture(autouse=True)
def _float32():
    saved = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(saved)


@pytest.fixture
def fixed_scale():
    """Pin ``_make_primitive``'s scale range so primitives are deterministic.

    The range is a function attribute that ``generate_primitive_dataset`` sets
    before use, so calling the helper standalone requires setting it here.
    """
    saved = getattr(_make_primitive, "scale_range", None)
    _make_primitive.scale_range = (0.5, 0.5)  # degenerate range -> exactly 0.5
    yield 0.5
    if saved is None:
        del _make_primitive.scale_range
    else:
        _make_primitive.scale_range = saved


# --------------------------------------------------------------------------
# _random_rotation_matrix
# --------------------------------------------------------------------------


def test_random_rotation_matrix_is_a_proper_rotation():
    rng = np.random.default_rng(0)
    for _ in range(20):
        R = _random_rotation_matrix(rng)
        assert R.shape == (3, 3)
        # Orthogonal with determinant +1: a rotation, not a reflection.
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)


def test_random_rotation_matrix_preserves_lengths():
    rng = np.random.default_rng(3)
    R = _random_rotation_matrix(rng)
    v = np.array([1.0, -2.0, 0.5])
    assert np.linalg.norm(R @ v) == pytest.approx(np.linalg.norm(v))


def test_random_rotation_matrix_is_reproducible_per_seed():
    a = _random_rotation_matrix(np.random.default_rng(42))
    b = _random_rotation_matrix(np.random.default_rng(42))
    c = _random_rotation_matrix(np.random.default_rng(43))

    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


# --------------------------------------------------------------------------
# _AnisoScaledSDF
# --------------------------------------------------------------------------


def test_aniso_scaled_sdf_zero_level_set_is_exact():
    """The scaled surface must still read zero, even though it is anisotropic."""
    sdf = _AnisoScaledSDF(
        SphereSDF(center=[0.0, 0.0, 0.0], radius=1.0), [2.0, 1.0, 1.0]
    )

    on_surface = torch.tensor([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    values = sdf._compute(on_surface).reshape(-1)

    torch.testing.assert_close(values, torch.zeros(3), atol=1e-6, rtol=0)


def test_aniso_scaled_sdf_applies_min_scale_correction():
    scale = [2.0, 3.0, 4.0]
    sdf = _AnisoScaledSDF(SphereSDF(center=[0.0, 0.0, 0.0], radius=1.0), scale)

    # correction == min(scale) keeps the field 1-Lipschitz.
    assert sdf.correction == pytest.approx(2.0)
    centre = sdf._compute(torch.zeros(1, 3)).reshape(-1)
    # Unit sphere reads -1 at its centre; scaled by min(scale).
    torch.testing.assert_close(centre, torch.tensor([-2.0]), atol=1e-6, rtol=0)


def test_aniso_scaled_sdf_never_overestimates_distance():
    """1-Lipschitz: the reported magnitude must not exceed the true distance."""
    scale = np.array([3.0, 1.0, 1.0])
    sdf = _AnisoScaledSDF(SphereSDF(center=[0.0, 0.0, 0.0], radius=1.0), scale.tolist())

    # Along the least-scaled axis the value is exact: surface at y = 1.
    value = sdf._compute(torch.tensor([[0.0, 2.0, 0.0]])).reshape(-1)
    torch.testing.assert_close(value, torch.tensor([1.0]), atol=1e-6, rtol=0)


def test_aniso_scaled_sdf_scales_domain_bounds():
    scale = [2.0, 3.0, 4.0]
    inner = SphereSDF(center=[0.0, 0.0, 0.0], radius=1.0)
    sdf = _AnisoScaledSDF(inner, scale)

    scaled = sdf._get_domain_bounds()
    expected = inner._get_domain_bounds() * torch.tensor(scale)
    torch.testing.assert_close(scaled, expected)


# --------------------------------------------------------------------------
# _place
# --------------------------------------------------------------------------


def test_place_keeps_sdf_and_mesh_coincident():
    canonical_sdf = SphereSDF(center=[0.0, 0.0, 0.0], radius=1.0)
    canonical_mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    R = _random_rotation_matrix(np.random.default_rng(7))
    center = np.array([1.0, -2.0, 0.5])

    sdf, mesh = _place(canonical_sdf, canonical_mesh, R, center)

    # The mesh surface must sit on the zero level set of the placed SDF.
    vertices = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32)
    values = sdf._compute(vertices).reshape(-1)
    torch.testing.assert_close(values, torch.zeros(len(values)), atol=1e-5, rtol=0)


def test_place_moves_the_centre_to_the_requested_point():
    canonical_sdf = SphereSDF(center=[0.0, 0.0, 0.0], radius=1.0)
    canonical_mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    center = np.array([3.0, 4.0, -5.0])

    sdf, mesh = _place(canonical_sdf, canonical_mesh, np.eye(3), center)

    # Deepest interior point of a unit sphere reads -1, now at `center`.
    value = sdf._compute(torch.tensor(center[None, :], dtype=torch.float32)).reshape(-1)
    torch.testing.assert_close(value, torch.tensor([-1.0]), atol=1e-5, rtol=0)
    np.testing.assert_allclose(mesh.centroid, center, atol=1e-6)


def test_place_does_not_mutate_the_canonical_mesh():
    canonical_mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    before = np.array(canonical_mesh.vertices)

    _place(
        SphereSDF(center=[0.0, 0.0, 0.0], radius=1.0),
        canonical_mesh,
        np.eye(3),
        np.array([10.0, 0.0, 0.0]),
    )

    np.testing.assert_array_equal(canonical_mesh.vertices, before)


# --------------------------------------------------------------------------
# _make_primitive
# --------------------------------------------------------------------------


def test_make_primitive_sphere_is_scaled_ellipsoid(fixed_scale):
    rng = np.random.default_rng(0)
    sdf, mesh, scale_vec = _make_primitive("sphere", rng)

    np.testing.assert_allclose(scale_vec, [fixed_scale] * 3)
    assert isinstance(sdf, _AnisoScaledSDF)
    # Mesh spans +/- the semi-axis in each direction.
    np.testing.assert_allclose(mesh.bounds[0], [-fixed_scale] * 3, atol=1e-6)
    np.testing.assert_allclose(mesh.bounds[1], [fixed_scale] * 3, atol=1e-6)


def test_make_primitive_box_uses_exact_extents(fixed_scale):
    rng = np.random.default_rng(0)
    sdf, mesh, scale_vec = _make_primitive("box", rng)

    # Boxes skip the anisotropic wrapper because BoxSDF is already exact.
    assert not isinstance(sdf, _AnisoScaledSDF)
    # extents = 2 * semi-size.
    np.testing.assert_allclose(mesh.extents, [2 * fixed_scale] * 3, atol=1e-6)


def test_make_primitive_cylinder_is_scaled(fixed_scale):
    rng = np.random.default_rng(0)
    sdf, mesh, _ = _make_primitive("cylinder", rng)

    assert isinstance(sdf, _AnisoScaledSDF)
    # Unit cylinder is radius 1, height 2 before the scale is applied.
    np.testing.assert_allclose(mesh.extents, [2 * fixed_scale] * 3, atol=1e-6)


def test_make_primitive_rejects_unknown_type(fixed_scale):
    with pytest.raises(ValueError, match="Unknown primitive type: torus"):
        _make_primitive("torus", np.random.default_rng(0))


def test_make_primitive_scale_is_independent_per_axis():
    saved = getattr(_make_primitive, "scale_range", None)
    _make_primitive.scale_range = (0.1, 1.0)
    try:
        _, _, scale_vec = _make_primitive("sphere", np.random.default_rng(1))
        assert scale_vec.shape == (3,)
        assert np.all((scale_vec >= 0.1) & (scale_vec <= 1.0))
        # Three independent draws, so they should not coincide.
        assert len(set(scale_vec.tolist())) == 3
    finally:
        if saved is None:
            del _make_primitive.scale_range
        else:
            _make_primitive.scale_range = saved


# --------------------------------------------------------------------------
# _build_scene
# --------------------------------------------------------------------------


def test_build_scene_unions_all_primitives(fixed_scale):
    n_primitives = 3
    scene_sdf, scene_mesh = _build_scene(
        ["sphere"], n_primitives, UNIT_BOUNDS, False, np.random.default_rng(0)
    )

    single = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    # The concatenated mesh carries every primitive's surface.
    assert len(scene_mesh.vertices) == n_primitives * len(single.vertices)
    # The union reads negative somewhere inside, i.e. the scene is non-empty.
    grid = torch.tensor(
        np.stack(np.meshgrid(*[np.linspace(-1, 1, 12)] * 3, indexing="ij"), -1).reshape(
            -1, 3
        ),
        dtype=torch.float32,
    )
    assert scene_sdf._compute(grid).min().item() < 0.0


def test_build_scene_single_primitive_skips_the_union(fixed_scale):
    scene_sdf, scene_mesh = _build_scene(
        ["box"], 1, UNIT_BOUNDS, False, np.random.default_rng(0)
    )

    # With one primitive the union loop body never runs.
    assert len(scene_mesh.vertices) == len(
        trimesh.creation.box(extents=[1, 1, 1]).vertices
    )
    assert scene_sdf is not None


def test_build_scene_margin_keeps_unrotated_primitives_strictly_inside(fixed_scale):
    """Without rotation the margin (max semi-size) fully contains each primitive."""
    _, scene_mesh = _build_scene(
        ["sphere", "box", "cylinder"], 6, UNIT_BOUNDS, False, np.random.default_rng(5)
    )

    assert np.all(scene_mesh.bounds[0] >= UNIT_BOUNDS[0] - 1e-6)
    assert np.all(scene_mesh.bounds[1] <= UNIT_BOUNDS[1] + 1e-6)


def test_build_scene_rotated_primitives_stay_roughly_inside(fixed_scale):
    """Rotation can push a corner past the margin -- bounded by the half-diagonal.

    The margin is ``max(scale_vec)``, so a rotated anisotropic primitive may poke
    out by up to its half-diagonal. Those samples are rejected downstream by
    ``_filter_to_bounds`` rather than prevented here.
    """
    _, scene_mesh = _build_scene(
        ["sphere", "box", "cylinder"], 6, UNIT_BOUNDS, True, np.random.default_rng(5)
    )

    slack = np.sqrt(3.0) * fixed_scale
    assert np.all(scene_mesh.bounds[0] >= UNIT_BOUNDS[0] - slack)
    assert np.all(scene_mesh.bounds[1] <= UNIT_BOUNDS[1] + slack)


def test_build_scene_handles_margin_larger_than_the_box():
    """An oversized primitive must not produce an inverted centre range."""
    saved = getattr(_make_primitive, "scale_range", None)
    _make_primitive.scale_range = (2.0, 2.0)  # margin exceeds the half-box
    try:
        scene_sdf, scene_mesh = _build_scene(
            ["box"], 2, UNIT_BOUNDS, False, np.random.default_rng(0)
        )
        # The min/max swap keeps rng.uniform's low <= high instead of erroring.
        assert scene_sdf is not None
        assert len(scene_mesh.vertices) > 0
    finally:
        if saved is None:
            del _make_primitive.scale_range
        else:
            _make_primitive.scale_range = saved


def test_build_scene_rotation_flag_changes_the_result(fixed_scale):
    _, unrotated = _build_scene(
        ["box"], 2, UNIT_BOUNDS, False, np.random.default_rng(11)
    )
    _, rotated = _build_scene(["box"], 2, UNIT_BOUNDS, True, np.random.default_rng(11))

    # An axis-aligned box scene differs from a randomly oriented one.
    assert not np.allclose(
        np.sort(unrotated.vertices, axis=0), np.sort(rotated.vertices, axis=0)
    )


# --------------------------------------------------------------------------
# _filter_to_bounds
# --------------------------------------------------------------------------


def test_filter_to_bounds_drops_outside_samples():
    samples = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # inside
            [2.0, 0.0, 0.0],  # outside in x
            [0.0, -5.0, 0.0],  # outside in y
            [0.5, 0.5, 0.5],  # inside
        ]
    )
    distances = torch.tensor([[0.1], [0.2], [0.3], [0.4]])

    out = _filter_to_bounds(
        SampledSDF(samples=samples, distances=distances), UNIT_BOUNDS
    )

    assert out.samples.shape == (2, 3)
    torch.testing.assert_close(out.distances.reshape(-1), torch.tensor([0.1, 0.4]))


def test_filter_to_bounds_is_inclusive_on_the_boundary():
    samples = torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
    distances = torch.tensor([[0.0], [0.0]])

    out = _filter_to_bounds(
        SampledSDF(samples=samples, distances=distances), UNIT_BOUNDS
    )

    # Points exactly on the box faces are kept.
    assert out.samples.shape == (2, 3)


def test_filter_to_bounds_can_return_empty():
    samples = torch.tensor([[9.0, 9.0, 9.0]])
    distances = torch.tensor([[1.0]])

    out = _filter_to_bounds(
        SampledSDF(samples=samples, distances=distances), UNIT_BOUNDS
    )

    assert out.samples.shape[0] == 0
    assert out.distances.shape[0] == 0


def test_filter_to_bounds_keeps_everything_when_box_is_large():
    samples = torch.rand(50, 3) * 2 - 1
    distances = torch.rand(50, 1)
    big = np.array([[-10.0] * 3, [10.0] * 3])

    out = _filter_to_bounds(SampledSDF(samples=samples, distances=distances), big)

    assert out.samples.shape == (50, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
