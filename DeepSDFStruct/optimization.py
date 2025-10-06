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
    if not isinstance(Solid, torchfem.Solid):
        raise NotImplementedError("Currently only solid mesh is supported.")
    # VTK cell types
    if isinstance(Solid.etype, Tetra1):
        cell_types = Solid.n_elem * [pyvista.CellType.TETRA]
    elif isinstance(Solid.etype, Tetra2):
        cell_types = Solid.n_elem * [pyvista.CellType.QUADRATIC_TETRA]
    elif isinstance(Solid.etype, Hexa1):
        cell_types = Solid.n_elem * [pyvista.CellType.HEXAHEDRON]
    elif isinstance(Solid.etype, Hexa2):
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
    v0 = vertices[tets[:, 0]]
    v1 = vertices[tets[:, 1]]
    v2 = vertices[tets[:, 2]]
    v3 = vertices[tets[:, 3]]
    vols = torch.einsum("ij,ij->i", torch.cross(v1 - v0, v2 - v0, dim=1), v3 - v0) / 6.0
    return vols


class MMA:
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
        orig_shape = dF.shape
        """
        performs an optimizer step
        """
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
                    xmma,
                    dtype=self.parameters.dtype,
                    device=self.parameters.device,
                )
            )
        logger.info(
            "It.: {0:4} | C.: {1:1.3e} | Constr.:  {2:1.3e} | ch.: {3:1.3e}".format(
                self.loop, F_np[0][0], G_np[0][0], ch
            )
        )
