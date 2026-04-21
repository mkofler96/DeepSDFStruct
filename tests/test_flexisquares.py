from DeepSDFStruct.sdf_primitives import CircleSDF, RectangleSDF
from DeepSDFStruct.flexisquares.flexisquares import FlexiSquares
from DeepSDFStruct.torch_spline import TorchSpline
from DeepSDFStruct.pretrained_models import get_model, PretrainedModels
from DeepSDFStruct.SDF import (
    SDFfromDeepSDF,
    CappedBorderSDF,
    SDFfromLineMesh,
    DifferenceSDF,
)
from DeepSDFStruct.lattice_structure import LatticeSDFStruct
from DeepSDFStruct.parametrization import Constant
from DeepSDFStruct.mesh import create_2D_mesh
import numpy as np
import gustaf as gus
import splinepy
import torch


def test_flexisquares_linemesh_diag_and_straight():
    linemesh = gus.Edges(
        vertices=np.array([[-1, -1], [1, 1], [1, -1]]), edges=np.array([[0, 1], [1, 2]])
    )
    thickness = 0.4
    rectangle = RectangleSDF(center=[0, 0], extents=[3.9, 3.9])
    sdf_linemesh = SDFfromLineMesh(linemesh, thickness=thickness)
    sdf = DifferenceSDF(rectangle, sdf_linemesh)
    bounds = torch.tensor([[-2.0, -2.0], [2.0, 2.0]])
    mesh, _ = create_2D_mesh(sdf, 10, mesh_type="surface", bounds=bounds)
    mesh.export("linemesh.vtk")


def test_flexisquares_linemesh_diag1():
    linemesh = gus.Edges(
        vertices=np.array([[-2, -2], [2, 2]]), edges=np.array([[0, 1]])
    )
    thickness = 0.4
    rectangle = RectangleSDF(center=[0, 0], extents=[3.9, 3.9])
    sdf_linemesh = SDFfromLineMesh(linemesh, thickness=thickness)
    sdf = DifferenceSDF(rectangle, sdf_linemesh)
    bounds = torch.tensor([[-2.0, -2.0], [2.0, 2.0]])
    mesh, _ = create_2D_mesh(sdf, 2, mesh_type="surface", bounds=bounds)
    torch.testing.assert_close(
        mesh.vertices,
        torch.tensor(
            [
                [-1.2913, 0.1414],
                [-1.2913, 1.2913],
                [0.1414, -1.2913],
                [-0.5253, -0.5253],
                [0.5253, 0.5253],
                [-0.1414, 1.2913],
                [1.2913, -1.2913],
                [1.2913, -0.1414],
                [-0.5486, 0.5486],
                [0.5486, -0.5486],
            ]
        ),
        rtol=1e-4,
        atol=1e-4,
    )


def test_flexisquares_linemesh_diag2():
    linemesh = gus.Edges(
        vertices=np.array([[-2, 2], [2, -2]]), edges=np.array([[0, 1]])
    )
    thickness = 0.4
    rectangle = RectangleSDF(center=[0, 0], extents=[3.9, 3.9])
    sdf_linemesh = SDFfromLineMesh(linemesh, thickness=thickness)
    sdf = DifferenceSDF(rectangle, sdf_linemesh)
    bounds = torch.tensor([[-2.0, -2.0], [2.0, 2.0]])
    mesh, _ = create_2D_mesh(sdf, 2, mesh_type="surface", bounds=bounds)
    torch.testing.assert_close(
        mesh.vertices,
        torch.tensor(
            [
                [-1.2913, -1.2913],
                [-1.2913, -0.1414],
                [-0.1414, -1.2913],
                [-0.5253, 0.5253],
                [0.5253, -0.5253],
                [0.1414, 1.2913],
                [1.2913, 0.1414],
                [1.2913, 1.2913],
                [-0.5486, -0.5486],
                [0.5486, 0.5486],
            ]
        ),
        rtol=1e-4,
        atol=1e-4,
    )


def test_flexisquares_simple():
    circle_sdf = CircleSDF(center=(0.0, 0.0), radius=0.5)
    bounds = torch.tensor([[-1, -1], [1, 1]])
    fsq = FlexiSquares(device=bounds.device)
    verts, square_idx = fsq.construct_voxel_grid([10, 10], bounds=bounds)
    # this function is useful to plot the squares (voxels)
    # for square in square_idx:
    #     indices = square.index_select(0, torch.tensor([0, 1, 3, 2, 0]))
    #     ax.plot(verts[indices][:, 0], verts[indices][:, 1], color="black", lw=0.3)
    scalar_field = circle_sdf(verts)
    vd_surf, edges, L_dev = fsq(verts, scalar_field, square_idx)
    vd_surf_and_interior, faces, L_dev = fsq(
        verts, scalar_field, square_idx, output_tetmesh=True
    )
    # check that the first vertices are the boundary vertices:
    assert (
        vd_surf.shape[0] == 20
    ), f"Expected 20 surface vertices, got {vd_surf.shape[0]}"
    torch.testing.assert_close(vd_surf, vd_surf_and_interior[:20])
    assert (
        circle_sdf(vd_surf).abs() < 0.05
    ).all(), "Signed Distance Function not close to zero at border vertices"
    assert (circle_sdf(vd_surf) < 0.05).all(), "Some interior nodes are positive"
    mesh, _ = create_2D_mesh(circle_sdf, 40, mesh_type="surface", bounds=bounds)
    try:
        mesh, _ = create_2D_mesh(circle_sdf, 40, mesh_type="surface")
        raise ValueError(
            "This test should raise an error, because the bounds are not correct"
        )
    except RuntimeError as rError:
        print(f"The test correctly raised the runtimeerror {rError} ✅️")
    mesh.export("tests/tmp_outputs/circle_mesh.vtk")


def test_flexisquares_lattice_struct():
    model = get_model(PretrainedModels.AnalyticRoundCross)
    sdf = SDFfromDeepSDF(model)

    # Define a spline-based deformation field
    deformation_spline = TorchSpline(
        splinepy.helpme.create.box(2, 1).bspline, device=model.device
    )

    # Create the lattice structure with deformation and microtile
    lattice_struct = CappedBorderSDF(
        LatticeSDFStruct(
            tiling=(6, 3),
            microtile=sdf.to2D(axes=[0, 1], offset=0.5),
            parametrization=Constant([0.5], device=model.device),
        )
    )
    bounds = torch.tensor([[-0.05, -0.05], [1.05, 1.05]], device=model.device)
    fsq = FlexiSquares(device=model.device)
    verts, square_idx = fsq.construct_voxel_grid([60, 30], bounds=bounds)
    scalar_field = lattice_struct(verts) + 0.01

    vd_surf, edges, L_dev = fsq(verts, scalar_field, square_idx)

    vd_surf_and_interior, faces, L_dev = fsq(
        verts, scalar_field, square_idx, output_tetmesh=True
    )

    assert (lattice_struct(vd_surf) < 0.1).all(), "Some interior nodes are positive"

    # check if deformation spline works on extracted mesh
    vd = deformation_spline(vd_surf_and_interior)
    vd_surf_and_interior_manual = vd_surf_and_interior
    vd_surf_and_interior_manual[:, 0] = vd_surf_and_interior[:, 0] * 2
    torch.testing.assert_close(vd, vd_surf_and_interior_manual)


def test_near_zero_epsilon():
    """
    Test edge case where scalar values are exactly 0 or very close to it.
    This often causes division by zero in interpolation if not handled.
    """
    fsq = FlexiSquares(device="cpu")
    verts = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    square_idx = torch.tensor([[0, 1, 2, 3]])

    # Exactly zero on two corners
    scalar_field = torch.tensor([0.0, 0.0, 1.0, 1.0])

    vd, edges, _ = fsq(verts, scalar_field, square_idx)


if __name__ == "__main__":
    test_flexisquares_linemesh_diag_and_straight()
    test_flexisquares_linemesh_diag1()
    test_flexisquares_linemesh_diag2()
    test_flexisquares_simple()
    test_near_zero_epsilon()
    test_flexisquares_lattice_struct()
