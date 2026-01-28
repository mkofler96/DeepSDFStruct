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
from mmapy import mmasub
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
    if Solid.etype is Tetra1:
        cell_types = Solid.n_elem * [pyvista.CellType.TETRA]
    elif Solid.etype is Tetra2:
        cell_types = Solid.n_elem * [pyvista.CellType.QUADRATIC_TETRA]
    elif Solid.etype is Hexa1:
        cell_types = Solid.n_elem * [pyvista.CellType.HEXAHEDRON]
    elif Solid.etype is Hexa2:
        cell_types = Solid.n_elem * [pyvista.CellType.QUADRATIC_HEXAHEDRON]

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

    def __init__(self, parameters, bounds, max_step=0.1):
        self.max_step = max_step
        self.bounds = np.array(bounds)
        self.parameters = parameters
        self.m = 1
        self.n = len(parameters)
        self.x = parameters.detach().cpu().numpy()
        self.xold1 = parameters.detach().cpu().numpy()
        self.xold2 = parameters.detach().cpu().numpy()
        self.low = []
        self.upp = []
        self.a0_MMA = 1
        self.a_MMA = np.zeros((self.m, 1))
        self.c_MMA = 10000 * np.ones((self.m, 1))
        self.d_MMA = np.zeros((self.m, 1))

        self.loop = 0
        self.ch = 1.0
        self.F0 = None

    def step(self, F, dF, G, dG):
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
            Constraint function value at current design (≤ 0 is feasible).
        dG : torch.Tensor
            Gradient of constraint w.r.t. design variables, shape (n,).

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
        orig_shape = dF.shape
        F_np = F.detach().cpu().numpy().reshape(-1, 1)
        dFdx_np = dF.detach().cpu().numpy().reshape(-1, 1)
        G_np = G.detach().cpu().numpy().reshape(-1, 1)
        dGdx_np = dG.detach().cpu().numpy().reshape(1, -1)
        if self.loop == 0:
            self.F0 = F_np
        F_np = F_np / self.F0
        dFdx_np = dFdx_np / self.F0

        xmin = np.maximum(self.x - self.max_step, self.bounds[:, 0].reshape(-1, 1))
        xmax = np.minimum(self.x + self.max_step, self.bounds[:, 1].reshape(-1, 1))
        move = 0.1
        self.loop = self.loop + 1
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

        self.xold2 = self.xold1.copy()
        self.xold1 = self.x.copy()
        self.x = xmma
        self.upp = upp
        self.low = low

        ch = np.abs(np.mean(self.x.T - self.xold1.T) / np.mean(self.x.T))
        with torch.no_grad():
            self.parameters.copy_(
                torch.tensor(
                    xmma, dtype=self.parameters.dtype, device=self.parameters.device
                )
            )
        logger.info(
            "It.: {0:4} | J.: {1:1.3e} | Constr.:  {2:1.3e} | ch.: {3:1.3e}".format(
                self.loop, F_np[0][0], G_np[0][0], ch
            )
        )
