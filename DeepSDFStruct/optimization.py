"""
Structural Optimization Utilities
==================================

This module provides tools for gradient-based optimization of SDF-based geometries,
with a focus on structural design problems. It integrates with TorchFEM for
finite element analysis and provides optimization algorithms suitable for
constrained design problems.

Key Features
------------

MMA Optimizer
    Implementation of the Method of Moving Asymptotes (MMA), a gradient-based
    algorithm well-suited for structural optimization with nonlinear constraints.
    MMA is particularly effective for:
    - Topology optimization
    - Shape optimization with constraints
    - Problems with expensive objective evaluations
    - Highly nonlinear design spaces

Finite Element Integration
    - Conversion between TorchFEM and PyVista mesh formats
    - Support for tetrahedral and hexahedral elements
    - Linear and quadratic element types
    - Integration with gradient computation

Mesh Quality Utilities
    - Signed volume computation for tetrahedra
    - Mesh quality metrics
    - Degeneracy detection

The module is designed to work seamlessly with differentiable SDF representations,
enabling gradient-based optimization of complex 3D structures.
"""

import torchfem.materials
import torchfem.solid
from torchfem.elements import Hexa1, Hexa2, Tetra1, Tetra2
import torch
import numpy as np
from mmapy import mmasub, gcmmasub, asymp, raaupdate
import pyvista
import logging
import DeepSDFStruct

logger = logging.getLogger(DeepSDFStruct.__name__)


def get_mesh_from_torchfem(Solid: torchfem.Solid) -> pyvista.UnstructuredGrid:
    """Convert a TorchFEM Solid mesh to PyVista UnstructuredGrid.

    This function enables visualization and export of TorchFEM finite element
    meshes using PyVista. It supports both tetrahedral and hexahedral elements
    with linear and quadratic shape functions.

    Parameters
    ----------
    Solid : torchfem.Solid
        TorchFEM solid mesh object containing nodes, elements, and element type.

    Returns
    -------
    pyvista.UnstructuredGrid
        PyVista mesh representation suitable for visualization and I/O.

    Raises
    ------
    NotImplementedError
        If input is not a torchfem.Solid object.

    Notes
    -----
    Supported element types:
    - Tetra1: 4-node linear tetrahedron
    - Tetra2: 10-node quadratic tetrahedron
    - Hexa1: 8-node linear hexahedron
    - Hexa2: 20-node quadratic hexahedron

    Examples
    --------
    >>> from DeepSDFStruct.optimization import get_mesh_from_torchfem
    >>> import torchfem
    >>>
    >>> # Assume we have a TorchFEM solid mesh
    >>> # solid = torchfem.Solid(...)
    >>>
    >>> # Convert to PyVista for visualization
    >>> pv_mesh = get_mesh_from_torchfem(solid)
    >>> pv_mesh.plot()
    """
    if not isinstance(Solid, torchfem.Solid):
        raise NotImplementedError("Currently only solid mesh is supported.")
    # VTK cell types
    etype = Solid.etype

    if etype is Tetra1 or isinstance(etype, Tetra1):
        cell_types = [pyvista.CellType.TETRA] * Solid.n_elem
    elif etype is Tetra2 or isinstance(etype, Tetra2):
        cell_types = [pyvista.CellType.QUADRATIC_TETRA] * Solid.n_elem
    elif etype is Hexa1 or isinstance(etype, Hexa1):
        cell_types = [pyvista.CellType.HEXAHEDRON] * Solid.n_elem
    elif etype is Hexa2 or isinstance(etype, Hexa2):
        cell_types = [pyvista.CellType.QUADRATIC_HEXAHEDRON] * Solid.n_elem
    else:
        raise TypeError(f"Unsupported element type: {etype} ({type(etype)})")

    # VTK element list
    el = len(Solid.elements[0]) * torch.ones(Solid.n_elem, dtype=Solid.elements.dtype)
    elements = torch.cat([el[:, None], Solid.elements], dim=1).view(-1).tolist()

    # Deformed node positions
    pos = Solid.nodes

    # Create unstructured mesh
    mesh = pyvista.UnstructuredGrid(elements, cell_types, pos.tolist())
    return mesh


def tet_signed_vol(vertices, tets):
    """Compute signed volumes of tetrahedral elements.

    Calculates the signed volume of each tetrahedron, which is positive for
    correctly oriented elements and negative for inverted elements. This is
    useful for detecting mesh degeneracies and enforcing mesh quality constraints.

    Parameters
    ----------
    vertices : torch.Tensor
        Vertex coordinates of shape (N, 3).
    tets : torch.Tensor
        Tetrahedral connectivity of shape (M, 4), where each row contains
        vertex indices [v0, v1, v2, v3].

    Returns
    -------
    torch.Tensor
        Signed volumes of shape (M,), one per tetrahedron. Positive volumes
        indicate correctly oriented elements.

    Notes
    -----
    The signed volume is computed as:
        V = (1/6) * ((v1-v0) × (v2-v0)) · (v3-v0)

    Examples
    --------
    >>> import torch
    >>> from DeepSDFStruct.optimization import tet_signed_vol
    >>>
    >>> # Define a simple tetrahedron
    >>> vertices = torch.tensor([
    ...     [0.0, 0.0, 0.0],
    ...     [1.0, 0.0, 0.0],
    ...     [0.0, 1.0, 0.0],
    ...     [0.0, 0.0, 1.0]
    ... ])
    >>> tets = torch.tensor([[0, 1, 2, 3]])
    >>> volumes = tet_signed_vol(vertices, tets)
    >>> print(f"Volume: {volumes[0]:.3f}")  # Should be 1/6 ≈ 0.167
    """
    v0 = vertices[tets[:, 0]]
    v1 = vertices[tets[:, 1]]
    v2 = vertices[tets[:, 2]]
    v3 = vertices[tets[:, 3]]
    vols = torch.einsum("ij,ij->i", torch.cross(v1 - v0, v2 - v0, dim=1), v3 - v0) / 6.0
    return vols


class MMA:
    """Method of Moving Asymptotes (MMA) optimizer for constrained problems.

    MMA is a gradient-based optimization algorithm designed for nonlinear
    constrained problems. It constructs convex subproblems using moving
    asymptotes and is particularly effective for structural optimization.

    The optimizer handles a single objective function and a single constraint,
    with box bounds on design variables. It automatically normalizes the
    objective by its initial value for better numerical behavior.

    Parameters
    ----------
    parameters : torch.Tensor
        Initial design variables (will be optimized in-place).
    bounds : array-like of shape (n, 2)
        Box constraints [[lower_1, upper_1], ..., [lower_n, upper_n]]
        for each design variable.
    max_step : float, default 0.1
        Maximum allowed change in design variables per iteration,
        as a fraction of the bound range.

    Attributes
    ----------
    parameters : torch.Tensor
        Current design variables (updated in-place each iteration).
    loop : int
        Current iteration number.
    x : ndarray
        Current design variables in numpy format.
    xold1, xold2 : ndarray
        Design variables from previous two iterations (for MMA history).

    Methods
    -------
    step(F, dF, G, dG)
        Perform one MMA optimization step given objective, constraint,
        and their gradients.

    Notes
    -----
    MMA was developed by Krister Svanberg and is widely used in topology
    optimization. It is particularly effective for problems where:
    - The objective and constraints are expensive to evaluate
    - Gradients are available (via automatic differentiation)
    - The design space is high-dimensional
    - Strong nonlinearity is present

    The implementation uses the mmapy package for the core MMA algorithm.

    Examples
    --------
    >>> import torch
    >>> from DeepSDFStruct.optimization import MMA
    >>>
    >>> # Define design variables
    >>> params = torch.ones(10, requires_grad=True)
    >>> bounds = [[0.0, 2.0]] * 10
    >>>
    >>> # Create optimizer
    >>> optimizer = MMA(params, bounds, max_step=0.1)
    >>>
    >>> # Optimization loop
    >>> for i in range(100):
    ...     # Compute objective and constraint
    ...     objective = (params ** 2).sum()
    ...     constraint = params.sum() - 5.0
    ...
    ...     # Compute gradients
    ...     dF = torch.autograd.grad(objective, params, create_graph=True)[0]
    ...     dG = torch.autograd.grad(constraint, params, create_graph=True)[0]
    ...
    ...     # MMA step
    ...     optimizer.step(objective, dF, constraint, dG)
    ...
    ...     if optimizer.ch < 1e-3:
    ...         break

    References
    ----------
    .. [1] Svanberg, K. (1987). "The method of moving asymptotes—a new method
           for structural optimization." International Journal for Numerical
           Methods in Engineering, 24(2), 359-373.
    .. [2] mmapy: Python implementation of MMA
           https://github.com/arjendeetman/mmapy
    """

    def __init__(self, parameters, bounds, max_step=0.1, n_constraints=1):
        self.max_step = max_step
        self.bounds = np.asarray(bounds, dtype=float)
        self.parameters = parameters

        self.m = n_constraints
        self.n = parameters.numel()

        self.x = parameters.detach().cpu().numpy().reshape(-1, 1)
        self.xold1 = self.x.copy()
        self.xold2 = self.x.copy()

        self.low = np.zeros((self.n, 1))
        self.upp = np.zeros((self.n, 1))

        self.a0_MMA = 1.0
        self.a_MMA = np.zeros((self.m, 1))
        self.c_MMA = 10000 * np.ones((self.m, 1))
        self.d_MMA = np.zeros((self.m, 1))

        self.loop = 0
        self.ch = 1.0
        self.F0 = None

    def _restore_feasibility(self, x, restore_eval, tol, max_steps, step_limit):
        """Project an accepted candidate back onto the cheap-constraint feasible set.

        Minimum-norm Gauss-Newton on the violated rows: solve ``J dx = -g`` for the
        smallest ``dx`` (so the objective is disturbed as little as possible to first
        order), capped at ``step_limit`` per pass, clipped to the box bounds, with
        backtracking on the TRUE values. Locked design variables never move because the
        supplied row gradients are already masked to zero there. Stops when every row
        reads ``g <= tol``, when ``max_steps`` passes are exhausted, or when backtracking
        cannot reduce the worst violation (evaluation-noise floor) -- the residual is
        logged either way.
        """
        lim = float(step_limit) if step_limit is not None else float(self.max_step)
        lo, hi = self.bounds[:, 0:1], self.bounds[:, 1:2]
        g, J = restore_eval(x)
        g = np.asarray(g, dtype=float).reshape(-1)
        v0 = float(g.max())
        if v0 <= tol:
            return x
        steps_used = 0
        for _ in range(max(1, int(max_steps))):
            viol = g > tol
            if not viol.any():
                break
            Jv = np.asarray(J, dtype=float).reshape(g.size, -1)[viol]
            gv = g[viol]
            # Least-norm correction onto g = 0 (strictly inside the tol-acceptance):
            # dx = Jv^T (Jv Jv^T)^-1 (-gv), tiny Tikhonov guard for degenerate rows.
            A = Jv @ Jv.T
            A += (
                1e-10 * max(float(np.trace(A)) / max(gv.size, 1), 0.0) + 1e-30
            ) * np.eye(gv.size)
            dx = (Jv.T @ np.linalg.solve(A, -gv)).reshape(-1, 1)
            nrm = float(np.abs(dx).max())
            if nrm <= 0.0:
                logger.warning(
                    "  feasibility restoration: zero correction direction (all row "
                    f"gradients masked/vanishing); residual max g = {g.max():+.3e}"
                )
                break
            if nrm > lim:
                dx *= lim / nrm
            improved = False
            for _bt in range(4):
                x_try = np.clip(x + dx, lo, hi)
                g_try, J_try = restore_eval(x_try)
                g_try = np.asarray(g_try, dtype=float).reshape(-1)
                if g_try.max() < g.max() - 1e-12:
                    x, g, J = x_try, g_try, J_try
                    improved = True
                    steps_used += 1
                    break
                dx *= 0.5
            if not improved:
                break
        if float(g.max()) > tol:
            logger.warning(
                f"  feasibility restoration: residual violation max g = {g.max():+.3e} "
                f"> tol {tol:.1e} after {steps_used} step(s) (started at {v0:+.3e}) -- "
                f"likely an evaluation-noise floor or a step_limit cap."
            )
        else:
            logger.info(
                f"  feasibility restoration: max g {v0:+.3e} -> {g.max():+.3e} "
                f"in {steps_used} step(s)"
            )
        return x

    def step(
        self,
        F,
        dF,
        G,
        dG,
        geom_eval=None,
        geom_rows=None,
        max_inner=1,
        feas_tol=0.05,
        restore_eval=None,
        restore_tol=5e-3,
        restore_max_steps=8,
        restore_step_limit=None,
    ):
        """Perform one MMA optimization step.

        Updates design variables by solving a convex subproblem constructed
        from the objective, constraint, and their gradients.

        Parameters
        ----------
        F : torch.Tensor or float
            Objective function value at current design.
        dF : torch.Tensor
            Gradient of objective w.r.t. design variables, shape (n,).
        G : torch.Tensor or float
            Constraint values at current design (≤ 0 is feasible), reshaped to
            (m, 1) for the ``m = n_constraints`` rows this instance was built
            with. A scalar is accepted when ``m == 1``.
        dG : torch.Tensor
            Gradient of the constraints w.r.t. design variables, reshaped to
            (m, n). For ``m == 1`` a flat (n,) tensor is accepted.
        geom_eval : callable, optional
            Cheap geometry-only re-evaluation ``x_np -> np.ndarray``. Given a
            candidate design vector it returns the *true* (nonlinear) constraint
            values ``g = value - target`` for the rows listed in ``geom_rows``,
            without running the expensive (CFD) objective/constraints. Enables the
            hybrid-GCMMA conservativeness loop; when ``None`` this is a plain MMA
            step.
        geom_rows : sequence of int, optional
            Row indices into ``G`` for the geometry-only constraints that
            ``geom_eval`` returns, in the same order. Required with ``geom_eval``.
        max_inner : int, optional
            Maximum GCMMA inner (conservativeness) iterations per step when
            ``geom_eval`` is supplied. Each rejected candidate raises the offending
            rows' curvature parameter rho (Svanberg 2002 ``raaupdate``) and re-solves;
            the move limit stays FIXED. Default 1 (no inner loop).
        feas_tol : float, optional
            TRUE-violation level below which a candidate is accepted outright, in the
            units of the constraint rows (pass normalized, "fraction over budget" rows
            and the default 0.05 reads "5% over budget is tolerated transiently").
            Must be well above 0: at an ACTIVE constraint the true value always reads
            slightly above the approximation (residual curvature, evaluation noise) --
            with a ~0 tolerance the inner loop fires on every boundary-riding step and
            burns max_inner solves per iteration for nothing.
        restore_eval : callable, optional
            Enables POST-STEP FEASIBILITY RESTORATION on the cheap geometry rows:
            ``x_np -> (g, J)`` returning the true values (k,) AND their gradients
            (k, n) -- masked for locked variables, in the same normalized units as the
            constraint rows. After the step is accepted (through whichever gate), the
            candidate is projected back onto the geometry-feasible set with
            minimum-norm Gauss-Newton passes, so geometry violations beyond
            ``restore_tol`` cannot survive an iteration. Independent of the GCMMA
            inner loop (works with or without ``geom_eval``). No CFD is invoked.
        restore_tol : float, optional
            Restoration target/trigger: rows with ``g <= restore_tol`` are left alone.
            Keep it above the geometry-evaluation noise floor (default 5e-3).
        restore_max_steps : int, optional
            Maximum Gauss-Newton passes per outer iteration (default 8; typically 1-2
            are used). Each pass costs one geometry evaluation plus up to 4 backtracks.
        restore_step_limit : float, optional
            Per-pass infinity-norm cap on the correction; defaults to ``max_step``.

        Notes
        -----
        The method automatically:
        - Normalizes the objective by its initial value
        - Enforces move limits based on max_step
        - Updates MMA history (xold1, xold2)
        - Computes and logs convergence metric (ch)
        - Updates self.parameters in-place

        The convergence metric ch is the relative change in design variables.
        """
        F_np = np.asarray(F.detach().cpu().numpy(), dtype=float).reshape(1, 1)
        dFdx_np = np.asarray(dF.detach().cpu().numpy(), dtype=float).reshape(self.n, 1)

        G_np = np.asarray(G.detach().cpu().numpy(), dtype=float).reshape(self.m, 1)
        dGdx_np = np.asarray(dG.detach().cpu().numpy(), dtype=float).reshape(
            self.m, self.n
        )

        if self.loop == 0:
            # Normalize by the MAGNITUDE of the initial objective. Dividing by a
            # signed F0 flips the sign of both F and dF whenever F(x0) < 0, which
            # turns the minimization into a maximization: the same problem with a
            # constant added to the objective (which cannot move the optimum) then
            # converges to a different point. An exactly-zero F(x0) would divide by
            # zero, so fall back to 1.0 and leave the objective unscaled.
            f0_mag = float(np.abs(F_np[0, 0]))
            self.F0 = np.full((1, 1), f0_mag if f0_mag > 0.0 else 1.0)

        F_np = F_np / self.F0
        dFdx_np = dFdx_np / self.F0

        # Hybrid GCMMA conservativeness loop (Svanberg 2002, CCSA). Solve the subproblem,
        # then -- when a cheap geometry-only re-evaluation callback is supplied -- check
        # whether any geometry row reads worse at the candidate than its own conservative
        # approximation predicted. If so, RAISE that row's curvature parameter rho
        # (raaupdate) and re-solve with the move limit FIXED: the subproblem then *sees*
        # the nonlinearity (e.g. KS-margin softmax curvature) and picks a genuinely
        # different direction, instead of re-scaling the same bad step. (The previous
        # move-limit-halving back-off converged to a null step of the SAME direction:
        # g_true -> g_now + noise floor as dx -> 0, so the loop burned max_inner halvings
        # every iteration and then accepted a candidate that ratcheted the violation up
        # ~1.5e-3/iter with the objective long converged.) Only the geometry rows are
        # re-evaluated: the (expensive, CFD-based) objective and remaining constraints
        # stay frozen at their approximation (f0valnew = f0app, fvalnew = fapp), so
        # raaupdate can never touch them and no extra primal/adjoint solves happen.
        do_inner = (
            geom_eval is not None and geom_rows is not None and len(geom_rows) > 0
        )
        pred_slack = 1e-6  # slack on "worse than the model" (rows are O(1) normalized)

        self.loop += 1
        xmin = np.maximum(self.x - float(self.max_step), self.bounds[:, 0:1])
        xmax = np.minimum(self.x + float(self.max_step), self.bounds[:, 1:2])

        if not do_inner:
            xmma, ymma, zmma, lam, xsi, eta, muMMA, zet, s, low, upp = mmasub(
                self.m,
                self.n,
                self.loop,
                self.x,
                xmin,
                xmax,
                self.xold1,
                self.xold2,
                F_np,
                dFdx_np,
                G_np,
                dGdx_np,
                self.low,
                self.upp,
                self.a0_MMA,
                self.a_MMA,
                self.c_MMA,
                self.d_MMA,
            )
        else:
            # Standard GCMMA constants (Svanberg's reference driver): epsimin is the
            # conservativeness slack, raa0eps/raaeps floor the curvature parameters.
            epsimin = 1e-7
            raa0eps = 1e-6
            raaeps = 1e-6 * np.ones((self.m, 1))
            rows = list(geom_rows)
            n_inner = max(1, int(max_inner))
            # asymp: asymptote update (same rule as mmasub) + fresh per-outer-iteration
            # initialization of raa0 (objective) / raa (constraint rows) from the
            # current gradients. The raa0/raa inputs are overwritten, so pass dummies.
            low, upp, raa0, raa = asymp(
                self.loop,
                self.n,
                self.x,
                self.xold1,
                self.xold2,
                xmin,
                xmax,
                self.low,
                self.upp,
                raa0eps,
                np.full((self.m, 1), raaeps[0, 0]),
                raa0eps,
                raaeps,
                dFdx_np,
                dGdx_np,
            )
            g_now = G_np[rows, 0]
            best_x, best_score = None, np.inf
            for inner in range(n_inner):
                xmma, ymma, zmma, lam, xsi, eta, muMMA, zet, s, f0app, fapp = gcmmasub(
                    self.m,
                    self.n,
                    self.loop,
                    epsimin,
                    self.x,
                    xmin,
                    xmax,
                    low,
                    upp,
                    raa0,
                    raa,
                    F_np,
                    dFdx_np,
                    G_np,
                    dGdx_np,
                    self.a0_MMA,
                    self.a_MMA,
                    self.c_MMA,
                    self.d_MMA,
                )
                # True (nonlinear) geometry values at the candidate (g = value - target,
                # > 0 infeasible) vs the conservative approximation at the same point
                # (fapp includes the current rho curvature, unlike a bare linearization).
                g_true = np.asarray(geom_eval(xmma), dtype=float).reshape(-1)
                g_app = np.asarray(fapp, dtype=float).reshape(-1)[rows]
                # Accept when, for every geometry row, at least one holds: the true
                # violation is small (<= feas_tol -- boundary riding always reads a bit
                # above the model), the approximation was conservative (g_true <= g_app),
                # or the candidate does not worsen the row vs the CURRENT point (progress
                # toward feasibility must never be rejected).
                overshoot = (
                    (g_true > feas_tol)
                    & (g_true > g_app + pred_slack)
                    & (g_true > g_now + pred_slack)
                )
                # Track the least-worsening candidate for the exhaustion fallback.
                score = float(np.max(g_true - g_now))
                if score < best_score:
                    best_score, best_x = score, xmma.copy()
                if not overshoot.any():
                    best_x = xmma
                    break
                if inner == n_inner - 1:
                    logger.warning(
                        f"  GCMMA inner loop exhausted ({n_inner} solves): accepting the "
                        f"least-worsening candidate (max geom-row increase "
                        f"{best_score:+.3e} vs current); rho escalation could not make "
                        f"the model conservative -- likely an evaluation noise floor."
                    )
                    break
                logger.info(
                    f"  GCMMA rho update (attempt {inner + 1}/{n_inner}): geom overshoot "
                    f"g_true={g_true[overshoot].tolist()} > g_app="
                    f"{g_app[overshoot].tolist()}; raa[geom]="
                    f"{np.asarray(raa).reshape(-1)[rows].tolist()}"
                )
                # Raise rho on every geometry row that read worse than its model
                # (Svanberg raaupdate: raa <- min(1.1*(raa + delta), 10*raa)). Objective
                # and CFD rows are frozen at their approximation, so only geometry rows
                # can be updated.
                fvalnew = np.asarray(fapp, dtype=float).reshape(self.m, 1).copy()
                fvalnew[rows, 0] = g_true
                raa0, raa = raaupdate(
                    xmma,
                    self.x,
                    xmin,
                    xmax,
                    low,
                    upp,
                    np.asarray(f0app, dtype=float).reshape(1, 1),
                    fvalnew,
                    np.asarray(f0app, dtype=float).reshape(1, 1),
                    np.asarray(fapp, dtype=float).reshape(self.m, 1),
                    raa0,
                    raa,
                    raa0eps,
                    raaeps,
                    epsimin,
                )
            xmma = best_x

        # Optional post-step feasibility restoration: whatever gate accepted the
        # candidate (feas_tol shortcut, conservativeness, improves-vs-current,
        # exhaustion fallback), project it back onto the cheap-constraint feasible
        # set before committing it as the new design.
        if restore_eval is not None:
            xmma = self._restore_feasibility(
                xmma,
                restore_eval,
                float(restore_tol),
                restore_max_steps,
                restore_step_limit,
            )

        self.xold2 = self.xold1.copy()
        self.xold1 = self.x.copy()
        self.x = xmma
        self.low = low
        self.upp = upp

        self.ch = np.abs(np.mean(self.x.T - self.xold1.T) / np.mean(self.x.T))

        with torch.no_grad():
            self.parameters.copy_(
                torch.tensor(
                    xmma.reshape(self.parameters.shape),
                    dtype=self.parameters.dtype,
                    device=self.parameters.device,
                )
            )

        logger.info(
            f"It.: {self.loop:4d} | J.: {F_np[0,0]:1.3e} | "
            f"G: {[float(g) for g in G_np[:, 0]]} | ch.: {self.ch:1.3e}"
        )
