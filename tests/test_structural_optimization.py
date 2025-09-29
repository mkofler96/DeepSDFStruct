from DeepSDFStruct.pretrained_models import get_model, PretrainedModels
from DeepSDFStruct.SDF import SDFfromDeepSDF
from DeepSDFStruct.lattice_structure import LatticeSDFStruct
from DeepSDFStruct.torch_spline import TorchSpline
from DeepSDFStruct.mesh import create_3D_mesh, torchVolumeMesh
from DeepSDFStruct.parametrization import SplineParametrization
from DeepSDFStruct.optimization import MMA, tet_signed_vol, get_mesh_from_torchfem
from DeepSDFStruct.utils import configure_logging
import torchfem.materials
import torchfem.solid
import splinepy
import torch
import logging
import numpy as np


logger = logging.getLogger(__name__)
configure_logging()


def test_structural_optimization(num_iter=1):
    # torch.set_default_device("cuda")

    logger.info("Loading Model")
    # Load a pretrained DeepSDF model
    model = get_model(PretrainedModels.AnalyticRoundCross)
    sdf = SDFfromDeepSDF(model)

    cap_border_dict = {
        "x0": {"cap": 1, "measure": 0.05},
        "z1": {"cap": 1, "measure": 0.025},
    }

    param_spline_sp = splinepy.BSpline(
        [1, 0, 1],
        [[0, 0, 0.25, 0.5, 0.75, 1, 1], [0, 1], [0, 0, 0.5, 1, 1]],
        [[0.3] * model._trained_latent_vectors[0].shape[0]] * 3 * 5,
    )
    logger.info("using spline with knot vectors:")
    logger.info(param_spline_sp.knot_vectors)
    logger.info(f"and control points: {param_spline_sp.control_points}")
    tiling = [2, 1, 1]

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
    param = next(lattice_struct.parametrization.parameters())
    # optimizer = torch.optim.Adam([param], lr=lr)
    init_vol = None
    init_compl = None
    target_vol = 0.5

    bounds = np.zeros(param.shape) + np.array([0.15, 0.75])
    optimizer = MMA(param, bounds)

    for e in range(num_iter):
        # log(
        #     f"Starting iteration with parameters: {lattice_struct.parametrization.parameters.T}"
        # )
        # torch.set_default_device("cuda")
        torch.set_default_dtype(torch.float32)
        mesh, derivative = create_3D_mesh(
            lattice_struct,
            20,
            mesh_type="volume",
            differentiate=False,
            device=model.device,
        )
        if isinstance(mesh, torchVolumeMesh):
            tets = mesh.volumes
            verts = mesh.vertices
        else:
            raise RuntimeError("Resulting mesh should be volume mesh.")

        # change ordering to fix negative jacobian
        perm = torch.tensor([0, 2, 1, 3])
        tets_reoredered = tets[:, perm]

        vols = tet_signed_vol(verts, tets_reoredered)
        if init_vol is None:
            init_vol = vols.sum().item()
            logger.info(f"Initial volume: {init_vol} on {len(vols)} elements.")
        vol = vols.sum()
        mask = vols >= 0

        # keep only the good tets
        tets_clean = tets_reoredered[mask]

        # check how many were removed
        removed = (~mask).sum()
        # print(f"Removed {removed} negative volume tets")

        # Create model
        # torch.set_default_device("cpu")
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

    # torch.autograd.grad(compliance, cantilever.thickness)[0]
    mesh = get_mesh_from_torchfem(cantilever)
    mesh.point_data["u"] = u.detach().cpu().numpy()
    out_file_name = "sim_out.vtk"
    logger.info(f"Writing Output to {out_file_name}")
    mesh.save(out_file_name)


if __name__ == "__main__":
    test_structural_optimization()
