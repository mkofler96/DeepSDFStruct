"""Tests for ``DeepSDFStruct.optimization.MMA``.

Two halves:

``_restore_feasibility`` takes an evaluation callback, so it can be driven with
analytic constraints whose least-norm correction is known in closed form.

``step`` is exercised on three small multi-constraint problems whose optima are
known analytically -- one with both rows active, one with a slack row, and one
driving the hybrid GCMMA conservativeness loop. Gradients come from autograd so
each problem is stated once, and every problem is 2-variable, so a full solve is
a fraction of a second.

The convention throughout is that ``g > 0`` means violated.
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


# ==========================================================================
# MMA.step -- driven on small multi-constraint problems with known optima
# ==========================================================================


def _solve(f, g, x0, bounds, n_constraints, iterations, max_step=0.3, **step_kwargs):
    """Run ``iterations`` MMA steps on ``min f(x) s.t. g(x) <= 0``.

    ``f`` returns a scalar tensor and ``g`` a ``(m,)`` tensor; both gradients come
    from autograd, so each problem is stated once rather than differentiated by
    hand. Returns the final design and the optimizer.
    """
    parameters = torch.tensor(x0, dtype=torch.float64)
    optimizer = MMA(
        parameters,
        np.asarray(bounds, dtype=float),
        max_step=max_step,
        n_constraints=n_constraints,
    )

    for _ in range(iterations):
        x = parameters.detach().clone().requires_grad_(True)
        objective = f(x)
        objective.backward()
        dF = x.grad.clone()

        xg = parameters.detach().clone().requires_grad_(True)
        G = g(xg)
        dG = torch.stack(
            [
                torch.autograd.grad(G[i], xg, retain_graph=True)[0]
                for i in range(n_constraints)
            ]
        )
        optimizer.step(objective.detach(), dF, G.detach(), dG, **step_kwargs)

    return parameters.detach().numpy(), optimizer


# --- Problem A: both constraints active at the optimum ---------------------
# minimize  x1 + x2
# subject to  1/x1 - 1 <= 0,  1/x2 - 1 <= 0   (i.e. x1 >= 1 and x2 >= 1)
# Optimum (1, 1), f = 2, with both rows active. Nonlinear constraints, so MMA
# has to do real work rather than solve one linear subproblem.
_A_BOUNDS = [[0.5, 5.0], [0.5, 5.0]]


def _f_a(x):
    return x.sum()


def _g_a(x):
    return 1.0 / x - 1.0


def test_two_active_constraints_reach_the_analytic_optimum():
    x, optimizer = _solve(_f_a, _g_a, [3.0, 3.0], _A_BOUNDS, 2, iterations=15)

    np.testing.assert_allclose(x, [1.0, 1.0], atol=1e-4)
    assert x.sum() == pytest.approx(2.0, abs=1e-4)
    # Both rows are active: MMA approaches the boundary from the feasible side,
    # so each g sits just below zero.
    g = 1.0 / x - 1.0
    assert g.max() <= 1e-9
    np.testing.assert_allclose(g, [0.0, 0.0], atol=1e-4)
    assert optimizer.loop == 15


def test_step_writes_back_into_the_parameter_tensor():
    """The design tensor handed to MMA is updated in place, not replaced."""
    parameters = torch.tensor([3.0, 3.0], dtype=torch.float64)
    optimizer = MMA(
        parameters, np.asarray(_A_BOUNDS, dtype=float), max_step=0.3, n_constraints=2
    )
    before = parameters.clone()

    x = parameters.detach().clone().requires_grad_(True)
    obj = _f_a(x)
    obj.backward()
    G = _g_a(parameters.detach())
    dG = torch.diag(-1.0 / parameters.detach() ** 2)
    optimizer.step(obj.detach(), x.grad.clone(), G, dG)

    assert not torch.equal(parameters, before)
    # Same object, so callers holding a reference see the update.
    np.testing.assert_allclose(parameters.detach().numpy(), optimizer.x.reshape(-1))


def test_convergence_metric_decays():
    _, optimizer = _solve(_f_a, _g_a, [3.0, 3.0], _A_BOUNDS, 2, iterations=15)

    # ch is the relative design change; at a converged point it is ~0.
    assert optimizer.ch < 1e-6


def test_move_limit_caps_a_single_step():
    max_step = 0.05
    x, _ = _solve(_f_a, _g_a, [3.0, 3.0], _A_BOUNDS, 2, iterations=1, max_step=max_step)

    # One iteration cannot move any variable further than max_step.
    assert np.abs(x - 3.0).max() <= max_step + 1e-12


# --- Problem B: one active, one slack constraint ---------------------------
# minimize  (x1 - 3)^2 + (x2 - 3)^2
# subject to  x1 + x2 - 4 <= 0,  x1 - x2 - 5 <= 0
# Optimum (2, 2): the projection of (3, 3) onto the first row. The second row
# reads -5 there, so a comfortably inactive constraint must not perturb it.
_B_BOUNDS = [[0.0, 5.0], [0.0, 5.0]]


def _f_b(x):
    return ((x - 3.0) ** 2).sum()


def _g_b(x):
    return torch.stack([x[0] + x[1] - 4.0, x[0] - x[1] - 5.0])


def test_inactive_constraint_does_not_perturb_the_optimum():
    x, _ = _solve(_f_b, _g_b, [0.5, 0.5], _B_BOUNDS, 2, iterations=25)

    np.testing.assert_allclose(x, [2.0, 2.0], atol=1e-4)
    # First row active, second far from binding.
    assert x[0] + x[1] - 4.0 == pytest.approx(0.0, abs=1e-4)
    assert x[0] - x[1] - 5.0 == pytest.approx(-5.0, abs=1e-4)


def test_recovers_from_an_infeasible_start():
    """Started outside the feasible set, MMA must pull the design back in."""
    x0 = [3.0, 3.0]
    assert x0[0] + x0[1] - 4.0 > 0  # infeasible on row 0

    x, _ = _solve(_f_b, _g_b, x0, _B_BOUNDS, 2, iterations=25)

    np.testing.assert_allclose(x, [2.0, 2.0], atol=1e-4)
    assert x[0] + x[1] - 4.0 <= 1e-6


# --- Sign and direction matrix ---------------------------------------------
# Two mirrored problems with the SAME optimum (1, 1) but opposite constraint
# orientations, so a sign error in the constraint handling cannot satisfy both:
#
#   push_up   : minimize sum(x)          s.t.  1 - x <= 0   (drives x upward)
#   push_down : minimize sum((x - 5)^2)  s.t.  x - 1 <= 0   (drives x downward)
#
# Each is run from a feasible and an infeasible start, and with the objective
# shifted by a constant so that F(x0) comes out positive, exactly zero, or
# negative. A constant cannot move the optimum, so all twelve combinations must
# land on the same point.


def _f_up(x):
    return x.sum()


def _g_up(x):
    return 1.0 - x


def _f_down(x):
    return ((x - 5.0) ** 2).sum()


def _g_down(x):
    return x - 1.0


# name -> (f, g, bounds, feasible start, infeasible start)
_MIRRORED = {
    "push_up": (_f_up, _g_up, [[0.5, 5.0]] * 2, [3.0, 3.0], [0.6, 0.6]),
    "push_down": (_f_down, _g_down, [[0.0, 5.0]] * 2, [0.5, 0.5], [2.0, 2.0]),
}


def _offset_for(f, x0, sign):
    """Constant that puts F(x0) at the requested sign without moving the argmin."""
    f0 = float(f(torch.tensor(x0, dtype=torch.float64)))
    assert f0 > 0.0, "these problems are built with a positive objective at x0"
    return {"positive": 0.0, "zero": -f0, "negative": -2.0 * f0}[sign]


@pytest.mark.parametrize("objective_sign", ["positive", "zero", "negative"])
@pytest.mark.parametrize("start", ["feasible", "infeasible"])
@pytest.mark.parametrize("problem", ["push_up", "push_down"])
def test_optimum_is_invariant_to_objective_sign_and_start(
    problem, start, objective_sign
):
    f, g, bounds, feasible, infeasible = _MIRRORED[problem]
    x0 = feasible if start == "feasible" else infeasible

    # Precondition: the start really is on the side the case name claims.
    g0 = g(torch.tensor(x0, dtype=torch.float64)).numpy()
    if start == "feasible":
        assert g0.max() <= 0.0
    else:
        assert g0.max() > 0.0

    offset = _offset_for(f, x0, objective_sign)

    def shifted(x):
        return f(x) + offset

    # Precondition: F(x0) has the sign this case is meant to cover.
    f0 = float(shifted(torch.tensor(x0, dtype=torch.float64)))
    if objective_sign == "positive":
        assert f0 > 0.0
    elif objective_sign == "zero":
        assert f0 == pytest.approx(0.0, abs=1e-12)
    else:
        assert f0 < 0.0

    x, optimizer = _solve(shifted, g, x0, bounds, 2, iterations=30)

    np.testing.assert_allclose(x, [1.0, 1.0], atol=1e-4)
    # Both rows end up active, approached from the feasible side.
    g_final = g(torch.tensor(x)).numpy()
    assert g_final.max() <= 1e-6
    np.testing.assert_allclose(g_final, [0.0, 0.0], atol=1e-4)
    # Scale factor is a magnitude, never a signed value.
    assert float(optimizer.F0[0, 0]) > 0.0


@pytest.mark.parametrize(
    "problem, bounds, expected, binding",
    [
        # The constraint wants x >= 1 but the box floors it higher.
        ("push_up", [[1.2, 5.0]] * 2, 1.2, "lower"),
        # The objective pulls toward 5 and the constraint caps at 1, but the box
        # caps lower still.
        ("push_down", [[0.0, 0.8]] * 2, 0.8, "upper"),
    ],
)
def test_box_bound_overrides_the_constraint(problem, bounds, expected, binding):
    """When a bound is tighter than the constraint, the bound wins and the row
    goes slack -- i.e. the optimum sits at a strictly negative constraint value."""
    f, g, _default_bounds, feasible, _infeasible = _MIRRORED[problem]

    x, _ = _solve(f, g, feasible, bounds, 2, iterations=30)

    np.testing.assert_allclose(x, [expected, expected], atol=1e-4)
    lo = np.asarray(bounds, dtype=float)[:, 0]
    hi = np.asarray(bounds, dtype=float)[:, 1]
    assert np.all(x >= lo - 1e-9) and np.all(x <= hi + 1e-9)
    if binding == "lower":
        np.testing.assert_allclose(x, lo, atol=1e-4)
    else:
        np.testing.assert_allclose(x, hi, atol=1e-4)
    # The constraint is no longer active.
    np.testing.assert_allclose(g(torch.tensor(x)).numpy(), [-0.2, -0.2], atol=1e-4)


def test_zero_initial_objective_leaves_the_scale_unscaled():
    """F(x0) == 0 would divide by zero, so the normalization falls back to 1."""
    f, g, bounds, feasible, _ = _MIRRORED["push_up"]
    offset = _offset_for(f, feasible, "zero")

    x, optimizer = _solve(
        lambda x: f(x) + offset, g, feasible, bounds, 2, iterations=30
    )

    assert float(optimizer.F0[0, 0]) == 1.0
    assert np.all(np.isfinite(x))
    np.testing.assert_allclose(x, [1.0, 1.0], atol=1e-4)


def test_single_constraint_accepts_scalar_and_flat_gradient():
    """The docstring allows a scalar G and a flat (n,) dG when m == 1."""
    parameters = torch.tensor([3.0, 3.0], dtype=torch.float64)
    optimizer = MMA(
        parameters, np.asarray(_B_BOUNDS, dtype=float), max_step=0.3, n_constraints=1
    )

    for _ in range(25):
        x = parameters.detach().clone().requires_grad_(True)
        obj = _f_b(x)
        obj.backward()
        xg = parameters.detach().clone().requires_grad_(True)
        g = xg[0] + xg[1] - 4.0  # 0-dim tensor
        dg = torch.autograd.grad(g, xg)[0]  # flat (n,)
        optimizer.step(obj.detach(), x.grad.clone(), g.detach(), dg)

    np.testing.assert_allclose(parameters.detach().numpy(), [2.0, 2.0], atol=1e-4)


# --- Problem C: the hybrid GCMMA conservativeness loop --------------------
# minimize  (x1 - 2)^2 + (x2 - 2)^2
# subject to  x1 - 4 <= 0            (row 0, never active: the "expensive" row)
#             x1^2 + x2^2 - 2 <= 0   (row 1, the cheap geometry row)
# The objective pulls toward (2, 2), which lies outside the circle, so the
# optimum sits on it at (1, 1).
_C_BOUNDS = [[0.0, 3.0], [0.0, 3.0]]


def _f_c(x):
    return ((x - 2.0) ** 2).sum()


def _g_c(x):
    return torch.stack([x[0] - 4.0, (x**2).sum() - 2.0])


def _geom_eval(offset=0.0):
    """Cheap geometry-only re-evaluation of row 1.

    With ``offset > 0`` the truth reads systematically worse than the row the
    subproblem was given, which is what drives the rho escalation: gcmmasub's
    approximation is conservative by construction, so a row that agrees with its
    own model never triggers the inner loop.
    """

    def geom(x_np):
        x = np.asarray(x_np, dtype=float).reshape(-1)
        return np.array([float((x**2).sum() - 2.0) + offset])

    return geom


def test_gcmma_path_reaches_the_same_optimum_as_plain_mma():
    plain, _ = _solve(_f_c, _g_c, [0.5, 0.5], _C_BOUNDS, 2, iterations=20)
    gcmma, _ = _solve(
        _f_c,
        _g_c,
        [0.5, 0.5],
        _C_BOUNDS,
        2,
        iterations=20,
        geom_eval=_geom_eval(),
        geom_rows=[1],
        max_inner=4,
    )

    # On the circle, along the direction of (2, 2).
    np.testing.assert_allclose(gcmma, [1.0, 1.0], atol=1e-3)
    assert float((gcmma**2).sum()) == pytest.approx(2.0, abs=1e-3)
    np.testing.assert_allclose(gcmma, plain, atol=1e-3)


def test_gcmma_requires_both_geom_eval_and_geom_rows():
    """Either half missing falls back to a plain mmasub step."""
    calls = []

    def counting_geom(x_np):
        calls.append(x_np)
        return _geom_eval()(x_np)

    # geom_rows omitted, so do_inner stays False and geom_eval is never called.
    _solve(_f_c, _g_c, [0.5, 0.5], _C_BOUNDS, 2, iterations=3, geom_eval=counting_geom)
    assert calls == []

    # An empty geom_rows likewise disables the loop.
    _solve(
        _f_c,
        _g_c,
        [0.5, 0.5],
        _C_BOUNDS,
        2,
        iterations=3,
        geom_eval=counting_geom,
        geom_rows=[],
    )
    assert calls == []


def test_gcmma_escalates_rho_when_the_truth_beats_the_model(caplog):
    with caplog.at_level(logging.INFO, logger="DeepSDFStruct"):
        x, _ = _solve(
            _f_c,
            _g_c,
            [0.5, 0.5],
            _C_BOUNDS,
            2,
            iterations=12,
            geom_eval=_geom_eval(offset=0.5),
            geom_rows=[1],
            max_inner=4,
        )

    # The rho update fired rather than silently accepting the overshoot.
    assert "GCMMA rho update" in caplog.text
    # And the design still lands on the circle.
    np.testing.assert_allclose(x, [1.0, 1.0], atol=1e-2)


def test_gcmma_falls_back_to_least_worsening_candidate_when_exhausted(caplog):
    """With max_inner=1 there is no room to escalate, so the fallback is taken."""
    with caplog.at_level(logging.WARNING, logger="DeepSDFStruct"):
        x, _ = _solve(
            _f_c,
            _g_c,
            [0.5, 0.5],
            _C_BOUNDS,
            2,
            iterations=12,
            geom_eval=_geom_eval(offset=0.5),
            geom_rows=[1],
            max_inner=1,
        )

    assert "GCMMA inner loop exhausted" in caplog.text
    assert np.all(np.isfinite(x))
    # The fallback still accepts a usable design rather than stalling at the start.
    assert not np.allclose(x, [0.5, 0.5], atol=1e-3)


def test_step_with_restoration_keeps_the_geometry_row_feasible():
    """restore_eval runs after the step, projecting the accepted candidate back."""

    # Deliberately tight: row 1 must end up at or inside the circle.
    def restore_eval(x_np):
        x = np.asarray(x_np, dtype=float).reshape(-1)
        g = np.array([float((x**2).sum()) - 2.0])
        J = np.array([2.0 * x])
        return g, J

    x, _ = _solve(
        _f_c,
        _g_c,
        [0.5, 0.5],
        _C_BOUNDS,
        2,
        iterations=15,
        restore_eval=restore_eval,
        restore_tol=1e-4,
        restore_max_steps=8,
    )

    assert float((x**2).sum()) - 2.0 <= 1e-3
    np.testing.assert_allclose(x, [1.0, 1.0], atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
