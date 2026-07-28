"""Tests for the feasibility-restoration pass of ``DeepSDFStruct.optimization.MMA``.

``_restore_feasibility`` takes an evaluation callback, so it can be driven with
analytic constraints whose least-norm correction is known in closed form. The
convention throughout is that ``g > tol`` means violated.
"""

import logging

import numpy as np
import pytest
import torch

from DeepSDFStruct.optimization import MMA


def _mma(n=3, lo=-100.0, hi=100.0, max_step=0.1):
    bounds = np.column_stack([np.full(n, lo), np.full(n, hi)])
    return MMA(torch.zeros(n), bounds, max_step=max_step, n_constraints=1)


def _linear_eval(a, b):
    """g(x) = a . x - b with constant Jacobian ``a``; feasible when a.x <= b."""
    a = np.asarray(a, dtype=float).reshape(1, -1)

    def restore_eval(x):
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        g = (a @ x).reshape(-1) - b
        return g, a

    return restore_eval


def test_returns_immediately_when_already_feasible():
    mma = _mma(n=3)
    x0 = np.zeros((3, 1))
    calls = []

    def restore_eval(x):
        calls.append(np.array(x))
        return np.array([-1.0]), np.ones((1, 3))

    out = mma._restore_feasibility(
        x0, restore_eval, tol=1e-6, max_steps=10, step_limit=1.0
    )

    np.testing.assert_array_equal(out, x0)
    # Evaluated once to learn it was already feasible, then returned.
    assert len(calls) == 1


def test_converges_onto_the_constraint_surface():
    mma = _mma(n=3)
    x0 = np.ones((3, 1))
    # sum(x) <= 1, starting at sum(x) == 3.
    restore_eval = _linear_eval([1.0, 1.0, 1.0], 1.0)

    out = mma._restore_feasibility(
        x0, restore_eval, tol=1e-9, max_steps=200, step_limit=1.0
    )

    g_final, _ = restore_eval(out)
    assert g_final.max() <= 1e-9
    # The least-norm direction is parallel to the gradient, so a symmetric
    # constraint moves every coordinate equally.
    np.testing.assert_allclose(out.reshape(-1), np.full(3, out[0, 0]), atol=1e-12)


def test_least_norm_correction_leaves_unrelated_variables_untouched():
    """Minimum-norm Gauss-Newton must not disturb variables outside the row."""
    mma = _mma(n=3)
    x0 = np.array([[2.0], [5.0], [7.0]])
    # Only x0 appears in the constraint x0 <= 1.
    restore_eval = _linear_eval([1.0, 0.0, 0.0], 1.0)

    out = mma._restore_feasibility(
        x0, restore_eval, tol=1e-9, max_steps=10, step_limit=10.0
    )

    assert out[0, 0] == pytest.approx(1.0, abs=1e-9)
    # Untouched exactly, not just approximately.
    assert out[1, 0] == 5.0
    assert out[2, 0] == 7.0


def test_only_violated_rows_drive_the_correction():
    mma = _mma(n=2)
    x0 = np.array([[2.0], [3.0]])

    def restore_eval(x):
        x = np.asarray(x, dtype=float).reshape(-1)
        # Row 0 is violated; row 1 is satisfied with a large margin.
        g = np.array([x[0] - 1.0, -5.0])
        J = np.array([[1.0, 0.0], [0.0, 1.0]])
        return g, J

    out = mma._restore_feasibility(
        x0, restore_eval, tol=1e-9, max_steps=10, step_limit=10.0
    )

    assert out[0, 0] == pytest.approx(1.0, abs=1e-9)
    # The satisfied row's variable must stay put.
    assert out[1, 0] == 3.0


def test_step_limit_caps_the_per_pass_move():
    mma = _mma(n=1)
    x0 = np.array([[100.0]])
    restore_eval = _linear_eval([1.0], 1.0)

    out = mma._restore_feasibility(
        x0, restore_eval, tol=1e-9, max_steps=1, step_limit=0.5
    )

    # One pass, capped: the ideal correction is -99 but only -0.5 is taken.
    assert out[0, 0] == pytest.approx(99.5)


def test_step_limit_none_falls_back_to_max_step():
    mma = _mma(n=1, max_step=0.25)
    x0 = np.array([[100.0]])
    restore_eval = _linear_eval([1.0], 1.0)

    out = mma._restore_feasibility(
        x0, restore_eval, tol=1e-9, max_steps=1, step_limit=None
    )

    assert out[0, 0] == pytest.approx(99.75)


def test_respects_box_bounds_and_reports_residual(caplog):
    """Clipping can make the constraint unreachable; the residual is logged."""
    mma = _mma(n=1, lo=0.0, hi=1.0)
    x0 = np.array([[0.0]])

    def restore_eval(x):
        x = np.asarray(x, dtype=float).reshape(-1)
        # Requires x >= 2, which the box bound of 1.0 forbids.
        return np.array([2.0 - x[0]]), np.array([[-1.0]])

    with caplog.at_level(logging.WARNING, logger="DeepSDFStruct"):
        out = mma._restore_feasibility(
            x0, restore_eval, tol=1e-9, max_steps=20, step_limit=0.5
        )

    # Never leaves the box.
    assert 0.0 <= out[0, 0] <= 1.0
    assert out[0, 0] == pytest.approx(1.0)
    g_final, _ = restore_eval(out)
    assert g_final.max() > 1e-9
    assert "residual violation" in caplog.text


def test_zero_gradient_direction_aborts_without_moving(caplog):
    mma = _mma(n=3)
    x0 = np.array([[1.0], [2.0], [3.0]])

    def restore_eval(x):
        # Violated, but every gradient entry is masked to zero (all variables
        # locked), so there is no direction to move in.
        return np.array([5.0]), np.zeros((1, 3))

    with caplog.at_level(logging.WARNING, logger="DeepSDFStruct"):
        out = mma._restore_feasibility(
            x0, restore_eval, tol=1e-9, max_steps=10, step_limit=1.0
        )

    np.testing.assert_array_equal(out, x0)
    assert "zero correction direction" in caplog.text


def test_stops_when_backtracking_cannot_improve(caplog):
    """A constraint that never improves must not loop forever."""
    mma = _mma(n=1)
    x0 = np.array([[0.0]])
    calls = []

    def restore_eval(x):
        calls.append(float(np.asarray(x).reshape(-1)[0]))
        # Constant violation regardless of x: no step can reduce it.
        return np.array([1.0]), np.array([[1.0]])

    with caplog.at_level(logging.WARNING, logger="DeepSDFStruct"):
        out = mma._restore_feasibility(
            x0, restore_eval, tol=1e-9, max_steps=50, step_limit=1.0
        )

    # Initial evaluation plus exactly one pass of 4 backtracking probes.
    assert len(calls) == 5
    np.testing.assert_array_equal(out, x0)
    assert "residual violation" in caplog.text


def test_max_steps_limits_the_number_of_passes():
    mma = _mma(n=1)
    x0 = np.array([[10.0]])
    restore_eval = _linear_eval([1.0], 0.0)

    out = mma._restore_feasibility(
        x0, restore_eval, tol=1e-9, max_steps=3, step_limit=1.0
    )

    # Three passes of at most 1.0 each, so it cannot have reached 0.
    assert out[0, 0] == pytest.approx(7.0)


def test_max_steps_below_one_still_takes_one_pass():
    mma = _mma(n=1)
    x0 = np.array([[10.0]])
    restore_eval = _linear_eval([1.0], 0.0)

    out = mma._restore_feasibility(
        x0, restore_eval, tol=1e-9, max_steps=0, step_limit=1.0
    )

    # max(1, int(max_steps)) guarantees progress rather than a no-op.
    assert out[0, 0] == pytest.approx(9.0)


def test_tolerance_defines_acceptance():
    mma = _mma(n=1)
    x0 = np.array([[1.05]])
    restore_eval = _linear_eval([1.0], 1.0)

    # A slack tolerance accepts the starting point untouched...
    out_slack = mma._restore_feasibility(
        x0, restore_eval, tol=0.1, max_steps=10, step_limit=1.0
    )
    np.testing.assert_array_equal(out_slack, x0)

    # ...while a tight one drives it down to the surface.
    out_tight = mma._restore_feasibility(
        x0, restore_eval, tol=1e-12, max_steps=50, step_limit=1.0
    )
    assert out_tight[0, 0] == pytest.approx(1.0, abs=1e-12)


def test_logs_success_when_restoration_succeeds(caplog):
    mma = _mma(n=1)
    x0 = np.array([[5.0]])
    restore_eval = _linear_eval([1.0], 0.0)

    with caplog.at_level(logging.INFO, logger="DeepSDFStruct"):
        mma._restore_feasibility(
            x0, restore_eval, tol=1e-9, max_steps=50, step_limit=10.0
        )

    assert "feasibility restoration" in caplog.text
    assert "residual violation" not in caplog.text


def test_degenerate_duplicate_rows_are_handled_by_the_tikhonov_guard():
    """Two identical constraint rows make J J^T singular without the guard."""
    mma = _mma(n=2)
    x0 = np.array([[3.0], [3.0]])

    def restore_eval(x):
        x = np.asarray(x, dtype=float).reshape(-1)
        val = x[0] + x[1] - 1.0
        # Same row twice: rank-deficient Gram matrix.
        return np.array([val, val]), np.array([[1.0, 1.0], [1.0, 1.0]])

    out = mma._restore_feasibility(
        x0, restore_eval, tol=1e-6, max_steps=200, step_limit=1.0
    )

    g_final, _ = restore_eval(out)
    assert np.all(np.isfinite(out))
    assert g_final.max() <= 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
