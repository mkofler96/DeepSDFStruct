import gustaf as gus
import numpy as np

from DeepSDFStruct.SDF import SDFfromLineMesh


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
    test_sdf_from_linemesh()
