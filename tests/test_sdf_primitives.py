import torch
from math import pi, cos, sin
import pytest

from DeepSDFStruct.mesh import export_sdf_grid_vtk

from DeepSDFStruct.sdf_primitives import (
    SphereSDF,
    CylinderSDF,
    TorusSDF,
    PlaneSDF,
    CornerSpheresSDF,
    CrossMsSDF,
)
from DeepSDFStruct.SDF import TransformedSDF


@pytest.fixture
def queries():
    torch.manual_seed(42)
    return torch.rand(10, 3)


def test_sdf_primitives(queries):

    # instantiate primitives
    sphere = SphereSDF(center=[0.0, 0.0, 0.0], radius=0.5)
    cylinder_x = CylinderSDF(point=[0.0, 0.0, 0.0], axis="x", radius=0.3)
    torus = TorusSDF(center=[0.0, 0.0, 0.0], R=0.5, r=0.2)
    plane = PlaneSDF(point=[0.0, 0.0, 0.0], normal=[0.0, 1.0, 0.0])
    corner_spheres = CornerSpheresSDF(radius=0.2, limit=0.8)
    cross_ms = CrossMsSDF(radius=0.2)

    primitives = [sphere, cylinder_x, torus, plane, corner_spheres, cross_ms]

    # test each primitive
    for sdf in primitives:
        print(f"Testing {sdf.__class__.__name__}")
        _ = sdf(queries)


def test_rotated_cylinder(queries):
    """
    Rotate a cylinder along z-axis to align with y-axis.
    It should be equivalent to a cylinder originally along y-axis.
    """
    cyl_x = CylinderSDF(point=[0, 0, 0], axis="x", radius=0.3)
    theta = pi / 2  # rotate x -> y
    R = torch.tensor(
        [[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]],
        dtype=torch.float32,
    )

    rotated_cyl = TransformedSDF(cyl_x, rotation=R)
    cyl_y = CylinderSDF(point=[0, 0, 0], axis="y", radius=0.3)
    # queries = torch.tensor([[0.0, 0.0, 1.0]])
    val_rot = rotated_cyl._compute(queries)
    val_ref = cyl_y._compute(queries)
    assert torch.allclose(val_rot, val_ref, atol=1e-6)


def test_scaled_sphere(queries):
    """
    Scale a sphere and check equivalence with a sphere of different radius.
    """
    sphere_r03 = SphereSDF(center=[0, 0, 0], radius=0.3)
    sphere_r03_scaled = TransformedSDF(sphere_r03, scale=2.0)
    sphere_r06 = SphereSDF(center=[0, 0, 0], radius=0.6)
    queries = torch.tensor([[0.0, 0.0, 0.0]])
    val_r1_scaled = sphere_r03_scaled(queries)
    val_r2 = sphere_r06(queries)
    # export_sdf_grid_vtk(sphere_r03_scaled, "tests/tmp_outputs/sphere_r1_scaled.vtk")
    # export_sdf_grid_vtk(sphere_r06, "tests/tmp_outputs/sphere_r2.vtk")
    assert torch.allclose(val_r1_scaled, val_r2, atol=1e-6)


def test_rotated_sphere_equivalence(queries):
    """
    Rotate a sphere around any axis – should produce the same SDF.
    """
    sphere = SphereSDF(center=[0, 0, 0], radius=0.5)
    theta = pi / 3
    R = torch.tensor(
        [[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]],
        dtype=torch.float32,
    )

    rotated_sphere = TransformedSDF(sphere, rotation=R)
    # export_sdf_grid_vtk(sphere, "tests/tmp_outputs/sphere_orig.vtk")
    # export_sdf_grid_vtk(rotated_sphere, "tests/tmp_outputs/sphere_rotated.vtk")
    val_rot = rotated_sphere._compute(queries)
    val_ref = sphere._compute(queries)
    assert torch.allclose(val_rot, val_ref, atol=1e-6)


def test_translated_sphere(queries):
    """
    Translate a sphere – should be equivalent to a sphere with new center.
    """
    sphere_orig = SphereSDF(center=[0, 0, 0], radius=0.5)
    translation = torch.tensor([0.2, -0.1, 0.3])
    translated_sphere = TransformedSDF(sphere_orig, translation=translation)
    # in TransformedSDF, the query is shifted by translation, so we adjust center
    sphere_translated = SphereSDF(center=[0.2, -0.1, 0.3], radius=0.5)

    val_trans = translated_sphere._compute(queries)
    val_ref = sphere_translated._compute(queries)
    assert torch.allclose(val_trans, val_ref, atol=1e-6)


if __name__ == "__main__":
    queries = torch.rand(10, 3)
    test_rotated_cylinder(queries)
    test_rotated_sphere_equivalence(queries)
    test_scaled_sphere(queries)
    test_sdf_primitives(queries)
    test_translated_sphere(queries)
