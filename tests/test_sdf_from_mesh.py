import torch
import trimesh
from DeepSDFStruct.SDF import SDFfromMesh
from DeepSDFStruct.sdf_primitives import SphereSDF
import math


def test_icosphere_sdf_igl():
    mesh = trimesh.creation.icosphere(subdivisions=6, radius=1.0)

    sdf_sphere_from_mesh = SDFfromMesh(mesh, backend="igl", scale=False, threshold=0)
    sdf_sphere_from_analytical = SphereSDF(center=[0, 0, 0], radius=1.0)
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.0, 2.0, 2.0],
        ],
        dtype=torch.float32,
    )

    distances_mesh = sdf_sphere_from_mesh(points)
    distances_analyt = sdf_sphere_from_analytical(points)

    print("\n--- SDF Test Results ---")
    for i, p in enumerate(points):
        print(
            f"Point {p.tolist()}: Distance Mesh {distances_mesh[i,0]:.4f} | Distance Analytical {distances_analyt[i,0]:.4f}"
        )
    expected_distances = torch.tensor(
        [[-1.0], [-0.5], [0.0], [0.5], [math.sqrt(2**2 + 2**2 + 2**2) - 1]]
    )
    torch.testing.assert_close(
        distances_mesh,
        expected_distances,
        msg="Queried distances to mesh are wrong",
        atol=1e-4,
        rtol=1e-4,
    )
    torch.testing.assert_close(
        distances_mesh,
        expected_distances,
        msg="Queried analytical distances are wrong",
        atol=1e-4,
        rtol=1e-4,
    )


if __name__ == "__main__":
    test_icosphere_sdf_igl()
