from DeepSDFStruct.sdf_primitives import CylinderSDF
from DeepSDFStruct.flexisquares.flexisquares import FlexiSquares
from DeepSDFStruct.torch_spline import TorchSpline
from DeepSDFStruct.pretrained_models import get_model, PretrainedModels
from DeepSDFStruct.SDF import SDFfromDeepSDF
from DeepSDFStruct.lattice_structure import LatticeSDFStruct
from DeepSDFStruct.parametrization import Constant
import splinepy
import torch


def test_flexisquares_simple():
    circle_sdf = CylinderSDF(point=(0.0, 0.0, 0.0), axis=2, radius=0.5).to2D(
        axes=[0, 1]
    )
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


def test_flexisquares_lattice_struct():
    model = get_model(PretrainedModels.AnalyticRoundCross)
    sdf = SDFfromDeepSDF(model)

    # Set the latent vector and visualize a slice of the SDF
    sdf.set_latent_vec(torch.tensor([0.3]))

    # Define a spline-based deformation field
    deformation_spline = TorchSpline(
        splinepy.helpme.create.box(2, 1).bspline, device=model.device
    )

    # Create the lattice structure with deformation and microtile
    lattice_struct = LatticeSDFStruct(
        tiling=(6, 3),
        deformation_spline=deformation_spline,
        microtile=sdf.to2D(axes=[0, 1], offset=0.5),
        parametrization=Constant([0.5], device=model.device),
    )
    bounds = torch.tensor([[0, 0], [1, 1]], device=model.device)
    fsq = FlexiSquares(device=bounds.device)
    verts, square_idx = fsq.construct_voxel_grid([60, 30], bounds=bounds)
    scalar_field = lattice_struct(verts) + 0.01

    vd_surf, edges, L_dev = fsq(verts, scalar_field, square_idx)

    vd_surf_and_interior, faces, L_dev = fsq(
        verts, scalar_field, square_idx, output_tetmesh=True
    )

    assert (lattice_struct(vd_surf) < 0.05).all(), "Some interior nodes are positive"

    # check if deformation spline works on extracted mesh
    vd = deformation_spline(vd_surf_and_interior)
    vd_surf_and_interior_manual = vd_surf_and_interior
    vd_surf_and_interior_manual[:, 0] = vd_surf_and_interior[:, 0] * 2
    torch.testing.assert_close(vd, vd_surf_and_interior_manual)


if __name__ == "__main__":
    test_flexisquares_simple()
    test_flexisquares_lattice_struct()
