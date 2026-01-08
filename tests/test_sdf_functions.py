import gustaf as gus
import torch
import numpy as np
import itertools
from DeepSDFStruct.SDF import SDFfromLineMesh, CappedBorderSDF
from DeepSDFStruct.mesh import export_sdf_grid_vtk, export_surface_mesh, create_3D_mesh
from DeepSDFStruct.sdf_primitives import CrossMsSDF


def test_cap_border_dict():
    n = 21
    xs = torch.linspace(-1.0, 1.0, n)
    ys = torch.linspace(-1.0, 1.0, n)
    zs = torch.linspace(-1.0, 1.0, n)

    queries = torch.stack(torch.meshgrid(xs, ys, zs, indexing="ij"), dim=-1).reshape(
        -1, 3
    )

    base_sdf = CrossMsSDF(radius=0.2)
    export_sdf_grid_vtk(base_sdf, "sdf_nocap.vtk")
    base_sdf_vals = base_sdf(queries)

    axes = ["x", "y", "z"]
    sides = ["0", "1"]
    caps = [1, -1]

    dim_dict = {"x": 0, "y": 1, "z": 2}
    minstr_dict = {1: "plus", -1: "minus"}
    measure = 0.25

    for axis, side, cap in itertools.product(axes, sides, caps):
        minstr = minstr_dict[cap]
        name = f"{axis}{side}{caps}{minstr}"
        key = f"{axis}{side}"
        cap_border_dict = {key: {"cap": cap, "measure": measure}}

        capped_sdf = CappedBorderSDF(base_sdf, cap_border_dict)
        print("Evaluating capdict for cap dict")
        print(cap_border_dict)
        capped_sdf_vals = capped_sdf(queries)
        export_sdf_grid_vtk(capped_sdf, f"tests/tmp_outputs/sdf_{name}.vtk")
        surf_mesh, _ = create_3D_mesh(capped_sdf, 51, mesh_type="surface")
        export_surface_mesh(f"tests/tmp_outputs/sdf_{name}.obj", surf_mesh.to_gus())
        export_surface_mesh(f"tests/tmp_outputs/sdf_{name}.stl", surf_mesh.to_gus())
        interior_low = caps[0] + measure
        interior_high = caps[1] - measure

        inside_mask = (queries[:, dim_dict[axis]] > interior_low) & (
            queries[:, dim_dict[axis]] < interior_high
        )
        torch.testing.assert_close(
            capped_sdf_vals[inside_mask],
            base_sdf_vals[inside_mask],
            msg=f"Inside modified for {name}, cap={cap}, measure={measure}",
        )
        # raise NotImplementedError("stahp")


def test_sdf_from_linemesh():
    linemesh = gus.io.meshio.load("tests/data/example_line_mesh.vtk")
    linemesh.vertices = linemesh.vertices[:, :2]
    bounds = np.array([[-9.5, 0.0], [9.5, 20.0]])
    n_points = 100
    x = np.linspace(bounds[0, 0], bounds[1, 0], n_points)
    y = np.linspace(bounds[0, 1], bounds[1, 1], n_points)
    xx, yy = np.meshgrid(x, y)
    queries = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
    sdf_from_linemesh = SDFfromLineMesh(linemesh, thickness=0.5)
    sdf = sdf_from_linemesh(queries)
    np.testing.assert_almost_equal(sdf.min(), -0.25)
    np.testing.assert_array_compare(np.greater_equal, sdf.max(), 0)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("error")
    test_cap_border_dict()
    test_sdf_from_linemesh()
