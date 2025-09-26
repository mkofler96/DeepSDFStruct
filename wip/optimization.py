from DeepSDFStruct.pretrained_models import get_model, PretrainedModels
from DeepSDFStruct.SDF import SDFfromDeepSDF
from DeepSDFStruct.lattice_structure import LatticeSDFStruct
from DeepSDFStruct.torch_spline import TorchSpline
from DeepSDFStruct.mesh import create_3D_surface_mesh
from DeepSDFStruct.parametrization import SplineParametrization
import splinepy
import torchfem.materials
import torchfem.solid
import torch
import vtk
import numpy as np
from mmapy import mmasub
from datetime import datetime


def log(msg):
    """
    Logs a message with current time in HHMMSS format.
    """
    now = datetime.now().strftime("%H:%M:%S")
    print(f"{now} | {msg}")


def export_tet_mesh_vtk(verts, tets, u, filename):
    """
    Exports a tetrahedral mesh with displacement vectors to VTK.

    verts: (N, 3) torch tensor of vertex coordinates
    tets: (M, 4) torch tensor of tetrahedron indices
    u: (N*3,) or (N, 3) torch tensor of displacements
    filename: str, output filename (.vtk)
    """
    if u.ndim == 1:
        u = u.view(-1, 3)  # reshape to (N, 3)

    vtk_points = vtk.vtkPoints()
    for v in verts:
        vtk_points.InsertNextPoint(v.tolist())

    vtk_cells = vtk.vtkCellArray()
    for tet in tets:
        cell = vtk.vtkTetra()
        for i in range(4):
            cell.GetPointIds().SetId(i, int(tet[i]))
        vtk_cells.InsertNextCell(cell)

    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(vtk_points)
    grid.SetCells(vtk.VTK_TETRA, vtk_cells)

    # Add displacement vectors
    vectors = vtk.vtkDoubleArray()
    vectors.SetNumberOfComponents(3)
    vectors.SetName("displacement")
    for vec in u:
        vectors.InsertNextTuple(vec.tolist())
    grid.GetPointData().AddArray(vectors)
    grid.GetPointData().SetActiveVectors("displacement")

    # Write to file
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(grid)
    writer.Write()
    print(f"Tetrahedral mesh with displacements saved to {filename}")


torch.set_default_device("cuda")

log("Loading Model")
# Load a pretrained DeepSDF model
model = get_model(PretrainedModels.AnalyticRoundCross)
sdf = SDFfromDeepSDF(model)


def tet_signed_vol(vertices, tets):
    v0 = vertices[tets[:, 0]]
    v1 = vertices[tets[:, 1]]
    v2 = vertices[tets[:, 2]]
    v3 = vertices[tets[:, 3]]
    vols = torch.einsum("ij,ij->i", torch.cross(v1 - v0, v2 - v0, dim=1), v3 - v0) / 6.0
    return vols


cap_border_dict = {
    "x0": {"cap": 1, "measure": 0.05},
    "z1": {"cap": 1, "measure": 0.025},
}

param_spline_sp = splinepy.BSpline(
    [1, 0, 1],
    [[0, 0, 0.25, 0.5, 0.75, 1, 1], [0, 1], [0, 0, 0.5, 1, 1]],
    [[0.3] * model._trained_latent_vectors[0].shape[0]] * 3 * 5,
)
print("using spline with knot vectors:")
print(param_spline_sp.knot_vectors)
print(f"and control points: {param_spline_sp.control_points}")
tiling = [6, 3, 3]


param_spline = SplineParametrization(param_spline_sp, device=model.device)
# Define a spline-based deformation field
deformation_spline = TorchSpline(
    splinepy.helpme.create.box(2, 1, 1).bspline, device=model.device
)

# Create the lattice structure with deformation and microtile
lattice_struct = LatticeSDFStruct(
    tiling=tiling,
    deformation_spline=deformation_spline,
    microtile=sdf,
    parametrization=param_spline,
    cap_border_dict=cap_border_dict,
)


lr = 1e-2
param = lattice_struct.parametrization.parameters
# optimizer = torch.optim.Adam([param], lr=lr)
init_vol = None
init_compl = None
target_vol = 0.5


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
        """
        performs an optimizer step
        """
        F_np = F.detach().cpu().numpy().reshape(-1, 1)
        dFdx_np = dF.detach().cpu().numpy()
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
        log(
            "It.: {0:4} | C.: {1:1.3e} | Constr.:  {2:1.3e} | ch.: {3:1.3e}".format(
                self.loop, F_np[0][0], G_np[0][0], ch
            )
        )


bounds = np.zeros(param.shape) + np.array([0.15, 0.75])
optimizer = MMA(param, bounds)

for e in range(20):
    # log(
    #     f"Starting iteration with parameters: {lattice_struct.parametrization.parameters.T}"
    # )
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float32)
    mesh, derivative = create_3D_surface_mesh(
        lattice_struct, 20, differentiate=False, device=model.device
    )

    tets = mesh.faces
    verts = mesh.vertices

    # change ordering to fix negative jacobian
    perm = torch.tensor([0, 2, 1, 3])
    tets_reoredered = tets[:, perm]

    vols = tet_signed_vol(verts, tets_reoredered)
    if init_vol is None:
        init_vol = vols.sum().item()
        log(f"Initial volume: {init_vol} on {len(vols)} elements.")
    vol = vols.sum()
    mask = vols >= 0

    # keep only the good tets
    tets_clean = tets_reoredered[mask]

    # check how many were removed
    removed = (~mask).sum()
    # print(f"Removed {removed} negative volume tets")

    # Create model
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float64)
    verts_tf = verts.to("cpu").to(torch.float64)
    tets_tf = tets_clean.to("cpu")

    # Material
    material = torchfem.materials.IsotropicElasticity3D(E=1000.0, nu=0.3)

    cantilever = torchfem.solid.Solid(verts_tf, tets_tf, material)

    # Constrained displacement at left end [Node_IDs, DOFs]
    left_const_mask = cantilever.nodes[:, 0] < 1e-5
    cantilever.constraints[left_const_mask, :] = True

    # Load at tip [Node_ID, DOF]
    top_mask = cantilever.nodes[:, 2] > (1 - 1e-1)
    top_mask_no_side = top_mask & (~left_const_mask)
    num_nodes = top_mask_no_side.sum().item()
    cantilever.forces[top_mask_no_side, 2] = -100.0 / num_nodes

    # log("Starting Simulation")
    u, f, _, _, _ = cantilever.solve(
        rtol=1e-2, atol=1e-2, device="cpu", method="pardiso"
    )

    # Compute sensitivity of compliance w.r.t. element thicknesses
    compliance = torch.inner(f.ravel(), u.ravel())

    F = compliance
    G = vol - target_vol
    dF = torch.autograd.grad(F, param, retain_graph=True)[0]
    dG = torch.autograd.grad(G, param, retain_graph=True)[0]
    optimizer.step(F, dF, G, dG)
    # log(f"cmp: {compliance.item():.5f} volume: {vol.item():.5f}")


# torch.autograd.grad(compliance, cantilever.thickness)[0]
log("Writing Output")
export_tet_mesh_vtk(verts, tets_clean, u, "sim_out.vtk")
log("Finished ")
