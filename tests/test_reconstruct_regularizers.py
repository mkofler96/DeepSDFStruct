"""Tests for the optional regularization branches of ``reconstruct_from_samples``.

The reconstruction tests all run with the default ``code_reg_lambda=0`` and
``eikonal_lambda=0``, so those branches -- and the ``code_bound`` projection --
are never entered. Each test here runs a handful of iterations only: the point is
that the branch executes and stays finite, not that it converges.
"""

import numpy as np
import pytest
import torch

from DeepSDFStruct.deep_sdf.reconstruction import reconstruct_from_samples
from DeepSDFStruct.sampling import SampledSDF
from DeepSDFStruct.sdf_primitives import SphereSDF


@pytest.fixture(autouse=True)
def _float32():
    saved = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(saved)


@pytest.fixture
def samples():
    """A small set of signed distances taken from a unit sphere."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1.0, 1.0, size=(256, 3)).astype(np.float32)
    dist = np.linalg.norm(pts, axis=1, keepdims=True) - 0.5
    return SampledSDF(
        samples=torch.tensor(pts), distances=torch.tensor(dist.astype(np.float32))
    )


def _sphere():
    return SphereSDF(center=[0.0, 0.0, 0.0], radius=0.4)


def test_eikonal_regularization_runs_and_stays_finite(samples):
    """The eikonal term needs a second backward pass through the SDF gradient."""
    sdf = _sphere()

    reconstruct_from_samples(
        sdf, samples, num_iterations=3, batch_size=128, eikonal_lambda=0.1
    )

    for p in sdf.parameters():
        assert torch.isfinite(p).all()


def test_eikonal_regularization_tolerates_no_near_surface_samples():
    """The ``near_mask.any()`` guard must skip the term rather than divide by zero."""
    # Every |gt| is far above the 0.05 near-surface threshold.
    pts = torch.rand(64, 3)
    far = torch.full((64, 1), 5.0)
    sdf = _sphere()

    reconstruct_from_samples(
        sdf,
        SampledSDF(samples=pts, distances=far),
        num_iterations=2,
        batch_size=32,
        eikonal_lambda=0.1,
    )

    for p in sdf.parameters():
        assert torch.isfinite(p).all()


def test_code_regularization_runs_when_parametrization_present(samples):
    """``code_reg_lambda`` penalizes evaluated latent codes, if the SDF has any."""
    sdf = _sphere()
    # _parametrization is picked up with getattr(sdf, "parametrization", None),
    # and only needs to map clamped queries to codes.
    sdf.parametrization = torch.nn.Linear(3, 4)

    reconstruct_from_samples(
        sdf, samples, num_iterations=3, batch_size=128, code_reg_lambda=0.01
    )

    for p in sdf.parameters():
        assert torch.isfinite(p).all()


def test_code_regularization_skipped_without_parametrization(samples):
    """No parametrization means the branch is skipped, not an AttributeError."""
    sdf = _sphere()
    assert getattr(sdf, "parametrization", None) is None

    reconstruct_from_samples(
        sdf, samples, num_iterations=2, batch_size=128, code_reg_lambda=0.01
    )

    for p in sdf.parameters():
        assert torch.isfinite(p).all()


def test_code_bound_clamps_parametrization_parameters(samples):
    sdf = _sphere()
    parametrization = torch.nn.Linear(3, 4)
    # Start well outside the bound so the projection has to bite.
    torch.nn.init.constant_(parametrization.weight, 5.0)
    sdf.parametrization = parametrization

    reconstruct_from_samples(
        sdf, samples, num_iterations=2, batch_size=128, code_bound=0.1
    )

    for p in sdf.parametrization.parameters():
        assert p.abs().max().item() <= 0.1 + 1e-6


def test_both_regularizers_together(samples):
    sdf = _sphere()
    sdf.parametrization = torch.nn.Linear(3, 4)

    reconstruct_from_samples(
        sdf,
        samples,
        num_iterations=3,
        batch_size=128,
        code_reg_lambda=0.01,
        eikonal_lambda=0.1,
        grad_clip=1.0,
    )

    for p in sdf.parameters():
        assert torch.isfinite(p).all()


def test_rejects_unknown_optimizer(samples):
    with pytest.raises(NotImplementedError, match="Optimizer nope not available"):
        reconstruct_from_samples(_sphere(), samples, optimizer_name="nope")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
