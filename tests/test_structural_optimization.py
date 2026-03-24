from DeepSDFStruct.pretrained_models import get_model, PretrainedModels
from DeepSDFStruct.SDF import SDFfromDeepSDF, CappedBorderSDF
from DeepSDFStruct.lattice_structure import LatticeSDFStruct
from DeepSDFStruct.torch_spline import TorchSpline
from DeepSDFStruct.mesh import create_3D_mesh, torchVolumeMesh
from DeepSDFStruct.parametrization import SplineParametrization
from DeepSDFStruct.optimization import MMA, tet_signed_vol, get_mesh_from_torchfem
from DeepSDFStruct.utils import configure_logging
import DeepSDFStruct
import torchfem.materials
import torchfem.solid
import splinepy
import torch
import logging
import numpy as np

logger = logging.getLogger(DeepSDFStruct.__name__)
configure_logging()


def _build_structural_test_lattice(model):
    sdf = SDFfromDeepSDF(model)
    cap_border_dict = {
        "x0": {"cap": 1, "measure": 0.05},
        "z1": {"cap": 1, "measure": 0.1},
    }
    param_spline_sp = splinepy.BSpline(
        [1, 0, 1],
        [[0, 0, 0.25, 0.5, 0.75, 1, 1], [0, 1], [0, 0, 0.5, 1, 1]],
        [[0.3] * model._trained_latent_vectors[0].shape[0]] * 3 * 5,
    )
    logger.info("using spline with knot vectors:")
    logger.info(param_spline_sp.knot_vectors)
    logger.info(f"and control points: {param_spline_sp.control_points}")
    param_spline = SplineParametrization(param_spline_sp, device=model.device)
    deformation_spline = TorchSpline(
        splinepy.helpme.create.box(2, 1, 1).bspline, device=model.device
    )
    lattice_struct_uncapped = LatticeSDFStruct(
        tiling=[2, 1, 1],
        deformation_spline=deformation_spline,
        microtile=sdf,
        parametrization=param_spline,
    )
    lattice_struct = CappedBorderSDF(
        CappedBorderSDF(lattice_struct_uncapped, cap_border_dict)
    )
    return lattice_struct, lattice_struct_uncapped


def _filter_nonpositive_jacobian_elements(verts_tf, tets_tf):
    material = torchfem.materials.IsotropicElasticity3D(E=1000.0, nu=0.3)
    probe = torchfem.solid.Solid(verts_tf, tets_tf, material)
    nodes = probe.nodes[probe.elements, :]
    b = probe.etype.B(probe.etype.ipoints).to(nodes.dtype)
    J = torch.einsum("...iN, ANj -> ...Aij", b, nodes)
    detJ = torch.linalg.det(J)
    valid = detJ > 0.0
    if valid.all():
        return tets_tf, 0
    return tets_tf[valid], int((~valid).sum().item())


def test_structural_mesh_output_regression():
    model = get_model(PretrainedModels.AnalyticRoundCross)
    lattice_struct, _ = _build_structural_test_lattice(model)

    volume_mesh, _ = create_3D_mesh(
        lattice_struct, 30, mesh_type="volume", differentiate=False, device=model.device
    )
    surf_mesh, _ = create_3D_mesh(
        lattice_struct,
        30,
        mesh_type="surface",
        differentiate=False,
        device=model.device,
    )

    assert isinstance(volume_mesh, torchVolumeMesh)
    assert tuple(volume_mesh.vertices.shape) == (30423, 3)
    assert tuple(volume_mesh.volumes.shape) == (156168, 4)
    assert tuple(surf_mesh.vertices.shape) == (8432, 3)
    assert tuple(surf_mesh.faces.shape) == (16864, 3)

    expected_max = torch.tensor([2.0000002, 1.0000001, 1.0000001], device=model.device)
    torch.testing.assert_close(
        volume_mesh.vertices.min(dim=0).values,
        torch.zeros(3, device=model.device),
        atol=1e-8,
        rtol=0.0,
    )
    torch.testing.assert_close(
        volume_mesh.vertices.max(dim=0).values, expected_max, atol=1e-6, rtol=0.0
    )
    torch.testing.assert_close(
        volume_mesh.vertices.detach().sum(),
        torch.tensor(56512.44921875, device=model.device),
        atol=1e-3,
        rtol=0.0,
    )
    torch.testing.assert_close(
        surf_mesh.vertices.detach().sum(),
        torch.tensor(16229.1162109375, device=model.device),
        atol=1e-3,
        rtol=0.0,
    )

    perm = torch.tensor([0, 2, 1, 3], device=model.device)
    tets_reordered = volume_mesh.volumes[:, perm]
    vols = tet_signed_vol(volume_mesh.vertices, tets_reordered)
    n_negative = int((vols < 0).sum().item())
    n_zero = int((vols == 0).sum().item())
    assert (
        n_negative == 7
    ), f"Expected 7 inverted tets before filtering, got {n_negative}"
    assert n_zero == 0, f"Expected no zero-volume tets, got {n_zero}"

    verts_tf = volume_mesh.vertices.to("cpu").to(torch.float64)
    tets_tf = tets_reordered[vols >= 0].to("cpu")
    material = torchfem.materials.IsotropicElasticity3D(E=1000.0, nu=0.3)
    cantilever = torchfem.solid.Solid(verts_tf, tets_tf, material)

    nodes = cantilever.nodes[cantilever.elements, :]
    b = cantilever.etype.B(cantilever.etype.ipoints).to(nodes.dtype)
    J = torch.einsum("...iN, ANj -> ...Aij", b, nodes)
    detJ = torch.linalg.det(J)
    min_detJ = detJ.min().item()
    n_nonpositive = int((detJ <= 0.0).sum().item())
    assert n_nonpositive == 0, (
        f"Found {n_nonpositive} elements with non-positive Jacobian. "
        f"Minimum Jacobian determinant: {min_detJ}"
    )


def test_structural_optimization(num_iter=1):
    # torch.set_default_device("cuda")

    logger.info("Loading Model")
    # Load a pretrained DeepSDF model
    model = get_model(PretrainedModels.AnalyticRoundCross)
    lattice_struct, lattice_struct_uncapped = _build_structural_test_lattice(model)

    lr = 1e-2
    param = next(lattice_struct_uncapped.parametrization.parameters())
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
        mesh, _ = create_3D_mesh(
            lattice_struct,
            30,
            mesh_type="volume",
            differentiate=False,
            device=model.device,
        )
        surf_mesh, _ = create_3D_mesh(
            lattice_struct,
            30,
            mesh_type="surface",
            differentiate=False,
            device=model.device,
        )

        surf_trimesh = surf_mesh.to_trimesh()
        assert surf_trimesh.is_watertight
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

        tets_tf, removed_nonpositive_jac = _filter_nonpositive_jacobian_elements(
            verts_tf, tets_tf
        )
        if removed_nonpositive_jac > 0:
            logger.warning(
                f"Removed {removed_nonpositive_jac} elements with non-positive Jacobian"
            )
        cantilever = torchfem.solid.Solid(verts_tf, tets_tf, material)

        # Constrained displacement at left end [Node_IDs, DOFs]
        left_const_mask = cantilever.nodes[:, 0] < 1e-5
        cantilever.constraints[left_const_mask, :] = True

        # Load at tip [Node_ID, DOF]
        top_mask = cantilever.nodes[:, 2] > (1 - 1e-1)
        top_mask_no_side = top_mask & (~left_const_mask)
        num_nodes = top_mask_no_side.sum().item()
        cantilever.forces[top_mask_no_side, 2] = -100.0 / num_nodes
        cantilever.forces.requires_grad_(True)

        # log("Starting Simulation")
        u, f, _, _, _ = cantilever.solve(
            rtol=1e-2,
            atol=1e-2,
            device="cpu",
            method="spsolve",
            differentiable_parameters=cantilever.forces,
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
    assert abs(dF.sum().item()) > 1e-8, "Derivative of objective is zero"
    assert abs(dG.sum().item()) > 1e-8, "Derivative of constraint is zero"


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("error")
    test_structural_optimization()
