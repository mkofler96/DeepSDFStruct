"""Tests for the sample IO helpers in ``DeepSDFStruct.sampling`` and ``deep_sdf.data``.

``save_points_to_vtp`` takes a single ``(N, 4)`` array of ``[x, y, z, sdf]``. It is
on the ``also_save_vtk=True`` path, which no other test exercises, and its
signature is easy to call incorrectly -- so the shape contract is asserted here
explicitly.
"""

import numpy as np
import pytest
import pyvista as pv
import torch

from DeepSDFStruct.deep_sdf.data import (
    _read_pos_neg,
    read_sdf_samples_into_ram,
    remove_nans,
    unpack_sdf_samples,
)
from DeepSDFStruct.sampling import save_points_to_vtp

_POINTS = np.array(
    [
        [0.0, 0.0, 0.0, -0.5],
        [1.0, 0.0, 0.0, 0.25],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 1.5],
    ]
)


# --------------------------------------------------------------------------
# save_points_to_vtp
# --------------------------------------------------------------------------


def test_save_points_to_vtp_round_trip(tmp_path):
    path = tmp_path / "samples.vtp"
    save_points_to_vtp(path, _POINTS)

    cloud = pv.read(path)
    assert cloud.n_points == 4
    np.testing.assert_allclose(cloud.points, _POINTS[:, :3])
    # The fourth column lands in a point array named "SDF".
    np.testing.assert_allclose(cloud["SDF"], _POINTS[:, 3])


def test_save_points_to_vtp_writes_one_vertex_cell_per_point(tmp_path):
    path = tmp_path / "samples.vtp"
    save_points_to_vtp(path, _POINTS)

    # Written as vertex cells so ParaView renders the cloud without a filter.
    assert pv.read(path).n_cells == 4


def test_save_points_to_vtp_accepts_torch_tensor(tmp_path):
    path = tmp_path / "samples.vtp"
    tensor = torch.tensor(_POINTS, dtype=torch.float32, requires_grad=True)

    # A tensor on the autograd graph must be detached rather than raising.
    save_points_to_vtp(path, tensor)

    np.testing.assert_allclose(pv.read(path)["SDF"], _POINTS[:, 3], rtol=1e-6)


@pytest.mark.parametrize("bad_shape", [(4, 3), (4, 5), (4,)])
def test_save_points_to_vtp_rejects_wrong_shape(tmp_path, bad_shape):
    """The (N, 4) contract is enforced, not silently reinterpreted."""
    with pytest.raises(ValueError, match=r"Expected points of shape \(N,4\)"):
        save_points_to_vtp(tmp_path / "x.vtp", np.zeros(bad_shape))


def test_save_points_to_vtp_handles_empty_cloud(tmp_path):
    path = tmp_path / "empty.vtp"
    save_points_to_vtp(path, np.zeros((0, 4)))

    assert pv.read(path).n_points == 0


def test_save_points_to_vtp_accepts_str_filename(tmp_path):
    path = tmp_path / "samples.vtp"
    save_points_to_vtp(str(path), _POINTS)
    assert path.is_file()


# --------------------------------------------------------------------------
# deep_sdf.data loaders
# --------------------------------------------------------------------------


def _write_npz(path, pos, neg, legacy=False):
    keys = {"pos.npy": pos, "neg.npy": neg} if legacy else {"pos": pos, "neg": neg}
    np.savez(path, **keys)
    return path.with_suffix(".npz") if path.suffix != ".npz" else path


def test_read_pos_neg_supports_modern_keys(tmp_path):
    pos = np.array([[0.0, 0.0, 0.0, 0.5]])
    neg = np.array([[1.0, 1.0, 1.0, -0.5]])
    path = _write_npz(tmp_path / "s.npz", pos, neg)

    got_pos, got_neg = _read_pos_neg(np.load(path))

    np.testing.assert_allclose(got_pos, pos)
    np.testing.assert_allclose(got_neg, neg)


def test_read_pos_neg_falls_back_to_legacy_keys(tmp_path):
    """Older archives store 'pos.npy'/'neg.npy' rather than 'pos'/'neg'."""
    pos = np.array([[0.0, 0.0, 0.0, 0.5]])
    neg = np.array([[1.0, 1.0, 1.0, -0.5]])
    path = _write_npz(tmp_path / "legacy.npz", pos, neg, legacy=True)

    archive = np.load(path)
    assert "pos" not in archive
    got_pos, got_neg = _read_pos_neg(archive)

    np.testing.assert_allclose(got_pos, pos)
    np.testing.assert_allclose(got_neg, neg)


def test_read_sdf_samples_into_ram_returns_float_tensors(tmp_path):
    pos = np.array([[0.0, 0.0, 0.0, 0.5]], dtype=np.float64)
    neg = np.array([[1.0, 1.0, 1.0, -0.5], [2.0, 0.0, 0.0, -1.0]], dtype=np.float64)
    path = _write_npz(tmp_path / "s.npz", pos, neg)

    pos_tensor, neg_tensor = read_sdf_samples_into_ram(path)

    # Downcast to float32 regardless of the stored dtype.
    assert pos_tensor.dtype == torch.float32
    assert neg_tensor.dtype == torch.float32
    assert pos_tensor.shape == (1, 4)
    assert neg_tensor.shape == (2, 4)


def test_remove_nans_drops_rows_with_nan_distance():
    tensor = torch.tensor(
        [[0.0, 0.0, 0.0, 0.5], [1.0, 0.0, 0.0, float("nan")], [2.0, 0.0, 0.0, -0.5]]
    )

    # Only the distance column (index == geom_dimension) is inspected.
    out = remove_nans(tensor, geom_dimension=3)

    assert out.shape == (2, 4)
    torch.testing.assert_close(out[:, 3], torch.tensor([0.5, -0.5]))


def test_unpack_sdf_samples_concatenates_when_no_subsample(tmp_path):
    pos = np.array([[0.0, 0.0, 0.0, 0.5], [0.1, 0.0, 0.0, 0.6]])
    neg = np.array([[1.0, 1.0, 1.0, -0.5]])
    path = _write_npz(tmp_path / "s.npz", pos, neg)

    out = unpack_sdf_samples(path, geom_dimension=3, subsample=None)

    assert out.shape == (3, 4)
    assert out.dtype == torch.float32


def test_unpack_sdf_samples_drops_nan_rows(tmp_path):
    pos = np.array([[0.0, 0.0, 0.0, 0.5], [0.1, 0.0, 0.0, float("nan")]])
    neg = np.array([[1.0, 1.0, 1.0, -0.5]])
    path = _write_npz(tmp_path / "s.npz", pos, neg)

    out = unpack_sdf_samples(path, geom_dimension=3, subsample=None)

    # The NaN-distance positive sample is removed.
    assert out.shape == (2, 4)
    assert not torch.isnan(out).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
