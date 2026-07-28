"""Tests for ``QuantumDeepSDFDecoder``.

Requires the optional ``quantum`` extra (``uv sync --extra quantum``); the module
raises ImportError at import time without pennylane, so collection is skipped
rather than failed when the extra is absent.

These tests are deliberately self-contained: the fitting test below builds its
targets from an analytic sphere in-process instead of reusing the shared
``lattice_structure_unit_cells`` dataset that the classical training tests
download. That keeps the quantum path independent of network access and of the
other tests' fixtures, and keeps the circuit evaluations small -- statevector
simulation is far slower per sample than a classical MLP.
"""

import math

import pytest
import torch

pytest.importorskip("pennylane", reason="requires the optional 'quantum' extra")

from DeepSDFStruct.deep_sdf.networks.quantum_deep_sdf_decoder import (  # noqa: E402
    QuantumDeepSDFDecoder,
)


@pytest.fixture(autouse=True)
def _float32():
    saved = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(saved)


@pytest.fixture(scope="module")
def decoder():
    """Default-geometry decoder, built once: construction compiles the QNode."""
    torch.manual_seed(0)
    return QuantumDeepSDFDecoder(latent_size=16, geom_dimension=3, n_qubits=5)


# --------------------------------------------------------------------------
# shape and padding contract
# --------------------------------------------------------------------------


def test_maps_latent_plus_xyz_to_one_value_per_point(decoder):
    out = decoder(torch.randn(8, 19))

    # (N, latent + geom) -> (N, 1), matching DeepSDFDecoder's contract.
    assert out.shape == (8, 1)
    assert out.dtype == torch.float32


def test_output_lies_in_the_pauli_z_range(decoder):
    # A Pauli-Z expectation value is bounded, so no readout layer is needed.
    out = decoder(torch.randn(16, 19) * 5.0)

    assert torch.isfinite(out).all()
    assert out.min().item() >= -1.0
    assert out.max().item() <= 1.0


def test_pads_features_up_to_a_whole_number_of_blocks():
    torch.manual_seed(0)
    # 19 features over 5 qubits -> ceil(19/5) = 4 blocks -> 20 slots, 1 padded.
    decoder = QuantumDeepSDFDecoder(latent_size=16, geom_dimension=3, n_qubits=5)

    assert decoder.padded_dim == 20
    assert decoder(torch.randn(4, 19)).shape == (4, 1)


def test_exact_multiple_needs_no_padding():
    torch.manual_seed(0)
    # 20 features over 5 qubits divides exactly, taking the no-pad branch.
    decoder = QuantumDeepSDFDecoder(latent_size=17, geom_dimension=3, n_qubits=5)

    assert decoder.padded_dim == 20
    assert decoder(torch.randn(3, 20)).shape == (3, 1)


@pytest.mark.parametrize(
    "latent_size, n_qubits, expected_blocks",
    [(16, 5, 4), (5, 4, 2), (13, 8, 2), (1, 2, 2)],
)
def test_block_count_follows_ceil_division(latent_size, n_qubits, expected_blocks):
    torch.manual_seed(0)
    decoder = QuantumDeepSDFDecoder(
        latent_size=latent_size, geom_dimension=3, n_qubits=n_qubits
    )

    assert decoder.padded_dim == expected_blocks * n_qubits
    assert decoder.enc_weights.shape[1] == expected_blocks


def test_single_point_batch_is_supported(decoder):
    """Reconstruction evaluates one query at a time in places."""
    out = decoder(torch.randn(1, 19))
    assert out.shape == (1, 1)


def test_larger_batch_matches_per_row_evaluation(decoder):
    """Native broadcasting must agree with evaluating rows one at a time."""
    x = torch.randn(5, 19)

    batched = decoder(x)
    per_row = torch.cat([decoder(x[i : i + 1]) for i in range(len(x))])

    torch.testing.assert_close(batched, per_row, atol=1e-5, rtol=1e-4)


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------


def test_parameter_shapes_follow_the_documented_layout():
    torch.manual_seed(0)
    decoder = QuantumDeepSDFDecoder(
        latent_size=16,
        geom_dimension=3,
        n_qubits=5,
        n_variational_layers=2,
        n_repeats=3,
    )

    # (n_repeats, n_blocks, n_qubits)
    assert decoder.enc_weights.shape == (3, 4, 5)
    # (n_repeats, n_blocks, n_variational_layers, n_qubits, 3)
    assert decoder.q_weights.shape == (3, 4, 2, 5, 3)
    assert decoder.enc_weights.requires_grad
    assert decoder.q_weights.requires_grad


def test_weights_are_initialised_near_zero():
    torch.manual_seed(0)
    decoder = QuantumDeepSDFDecoder(latent_size=16, geom_dimension=3, n_qubits=5)

    # Near-zero init avoids barren plateaus: (-0.01*pi, +0.01*pi).
    limit = 0.01 * math.pi
    for name, param in decoder.named_parameters():
        assert param.abs().max().item() <= limit, name
    # Not all identical, i.e. actually randomised rather than zeros.
    assert decoder.q_weights.std().item() > 0.0


def test_absorbs_unused_classical_network_specs():
    """The decoder is a drop-in, so classical NetworkSpecs keys must not break it."""
    torch.manual_seed(0)
    decoder = QuantumDeepSDFDecoder(
        latent_size=16,
        geom_dimension=3,
        n_qubits=5,
        dims=[512, 512],
        dropout=[0, 1],
        dropout_prob=0.2,
        norm_layers=[0, 1],
        latent_in=[4],
        weight_norm=True,
        xyz_in_all=False,
        use_tanh=False,
        latent_dropout=False,
    )

    assert decoder(torch.randn(2, 19)).shape == (2, 1)


def test_geom_dimension_is_exposed(decoder):
    # deep_sdf_decoder consumers read this off the module.
    assert decoder.geom_dimension == 3
    assert decoder.n_qubits == 5


def test_registered_as_a_selectable_architecture():
    """workspace.py registers the arch in a try/except, so it appears only with
    the extra installed. ``NetworkArch: quantum_deep_sdf_decoder`` in an
    experiment's specs resolves through this table."""
    from DeepSDFStruct.deep_sdf.workspace import ARCHITECTURES

    assert ARCHITECTURES["quantum_deep_sdf_decoder"] is QuantumDeepSDFDecoder


# --------------------------------------------------------------------------
# gradients and fitting
# --------------------------------------------------------------------------


def test_gradients_reach_both_parameter_tensors(decoder):
    out = decoder(torch.randn(4, 19))
    decoder.zero_grad()
    out.sum().backward()

    # backprop through the statevector simulation must populate both tensors.
    for name, param in decoder.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
        assert param.grad.abs().sum().item() > 0.0, name


def test_gradients_flow_to_the_input(decoder):
    """Eikonal-style regularizers differentiate the SDF w.r.t. xyz."""
    x = torch.randn(3, 19, requires_grad=True)

    grad = torch.autograd.grad(decoder(x).sum(), x)[0]

    assert grad.shape == x.shape
    assert torch.isfinite(grad).all()


def test_fits_a_synthetic_analytic_target():
    """A few Adam steps on an in-process sphere SDF must reduce the loss.

    Targets come from an analytic sphere rather than the shared training
    dataset, so this needs no download and stays small enough for statevector
    simulation.
    """
    torch.manual_seed(0)
    # Each step is a forward and backward pass through a statevector simulation,
    # so this is sized to demonstrate progress, not to converge.
    latent_size, n_points, n_steps = 4, 16, 8
    decoder = QuantumDeepSDFDecoder(
        latent_size=latent_size, geom_dimension=3, n_qubits=4, n_repeats=1
    )

    xyz = torch.rand(n_points, 3) * 2.0 - 1.0
    # Signed distance to a sphere of radius 0.5, inside the [-1, 1] output range.
    target = (xyz.norm(dim=1, keepdim=True) - 0.5).clamp(-1.0, 1.0)
    latent = torch.zeros(n_points, latent_size)
    features = torch.cat([latent, xyz], dim=1)

    optimizer = torch.optim.Adam(decoder.parameters(), lr=0.1)
    losses = []
    for _ in range(n_steps):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(decoder(features), target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert all(math.isfinite(loss) for loss in losses)
    # Optimisation makes progress; the circuit is not stuck at init.
    assert losses[-1] < losses[0]


def test_is_deterministic_for_fixed_weights(decoder):
    """Exact statevector simulation: no shot noise, so repeats are identical."""
    x = torch.randn(4, 19)

    torch.testing.assert_close(decoder(x), decoder(x), atol=0.0, rtol=0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
