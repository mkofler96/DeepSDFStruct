import torch
from math import pi, cos, sin
import pytest


from DeepSDFStruct.sdf_primitives import (
    SphereSDF,
    CylinderSDF,
    TorusSDF,
    PlaneSDF,
    CornerSpheresSDF,
    CrossMsSDF,
    CircleSDF,
    RectangleSDF,
)
from DeepSDFStruct.SDF import TransformedSDF

"""
Comprehensive SDF Tests
======================

Tests for all implemented SDF primitives and operations.
Validates against expected behavior and trimesh comparisons where applicable.
"""

import torch
import pytest
import numpy as np
import sys

sys.path.insert(0, ".")

from DeepSDFStruct.sdf_primitives import (
    # 2D primitives
    CircleSDF,
    RectangleSDF,
    LineSDF,
    RoundedRectangleSDF,
    EquilateralTriangleSDF,
    HexagonSDF,
    PolygonSDF,
    # 3D primitives
    SphereSDF,
    BoxSDF,
    RoundedBoxSDF,
    WireframeBoxSDF,
    TorusSDF,
    CylinderSDF,
    ConeSDF,
    PlaneSDF,
    CappedCylinderSDF,
    RoundedCylinderSDF,
    CappedConeSDF,
    RoundedConeSDF,
    SlabSDF,
    TetrahedronSDF,
    OctahedronSDF,
    DodecahedronSDF,
    IcosahedronSDF,
    CapsuleSDF,
    EllipsoidSDF,
    PyramidSDF,
    CornerSpheresSDF,
    CrossMsSDF,
)
from DeepSDFStruct.sdf_operations import (
    ElongateSDF,
    TwistSDF,
    BendLinearSDF,
    BendRadialSDF,
    DilateSDF,
    ErodeSDF,
    ShellSDF,
    RepeatSDF,
    MirrorSDF,
    CircularArraySDF,
    RevolveSDF,
)
from DeepSDFStruct.SDF import (
    SmoothUnionSDF,
    SmoothDifferenceSDF,
    SmoothIntersectionSDF,
    UnionSDF,
    DifferenceSDF,
    TransformedSDF,
    SDFBase,
)

# ==================== 2D Primitive Tests ====================


def test_circle_sdf_basic():
    """Test CircleSDF basic behavior."""
    circle = CircleSDF(center=[0.0, 0.0], radius=1.0)
    # Inside (negative)
    assert circle(torch.tensor([[0.0, 0.0]])).item() < 0
    # On surface (zero)
    assert abs(circle(torch.tensor([[1.0, 0.0]])).item()) < 1e-4
    # Outside (positive)
    assert circle(torch.tensor([[2.0, 0.0]])).item() > 0


def test_rectangle_sdf_basic():
    """Test RectangleSDF basic behavior."""
    rect = RectangleSDF(center=[0.0, 0.0], extents=[2.0, 1.0])
    # Center inside
    assert rect(torch.tensor([[0.0, 0.0]])).item() < 0
    # On edge
    assert abs(rect(torch.tensor([[1.0, 0.0]])).item()) < 1e-4
    # Outside
    assert rect(torch.tensor([[2.0, 0.0]])).item() > 0


def test_polygon_sdf():
    """Test PolygonSDF with triangle."""
    # Equilateral triangle vertices
    poly = PolygonSDF([[0, 0], [1, 0], [0.5, 0.866]])
    # Inside point (centroid)
    assert poly(torch.tensor([[0.5, 0.3]])).item() < 0
    # Domain bounds should be reasonable
    bounds = poly._get_domain_bounds()
    assert bounds.shape == (2, 2)


# ==================== 3D Primitive Tests ====================


def test_sphere_sdf_accuracy():
    """Test SphereSDF accuracy for various points."""
    sphere = SphereSDF(center=[0, 0, 0], radius=1.0)
    # Center should be -radius
    assert abs(sphere(torch.tensor([[0.0, 0.0, 0.0]])).item() - (-1.0)) < 1e-4
    # On surface
    assert abs(sphere(torch.tensor([[1.0, 0.0, 0.0]])).item()) < 1e-4
    # Outside
    assert abs(sphere(torch.tensor([[2.0, 0.0, 0.0]])).item() - 1.0) < 1e-4


def test_slab_sdf():
    """Test SlabSDF clipping."""
    slab = SlabSDF(z0=-1, z1=1)
    # Inside slab
    assert slab(torch.tensor([[0.0, 0.0, 0.0]])).item() < 0
    # Below slab
    assert slab(torch.tensor([[0.0, 0.0, -2.0]])).item() > 0
    # Above slab
    assert slab(torch.tensor([[0.0, 0.0, 2.0]])).item() > 0


def test_capped_cylinder():
    """Test CappedCylinderSDF."""
    cyl = CappedCylinderSDF([0, 0, -1], [0, 0, 1], 0.5)
    # Center should be inside
    assert cyl(torch.tensor([[0.0, 0.0, 0.0]])).item() < 0
    # On surface at side
    assert abs(cyl(torch.tensor([[0.5, 0.0, 0.0]])).item()) < 0.1


def test_rounded_cylinder():
    """Test RoundedCylinderSDF."""
    cyl = RoundedCylinderSDF(0.5, 0.1, 2)
    # Center should be inside
    assert cyl(torch.tensor([[0.0, 0.0, 0.0]])).item() < 0


def test_capped_cone():
    """Test CappedConeSDF."""
    cone = CappedConeSDF([0, 0, 0], [0, 0, 1], 1, 0.5)
    # Apex
    val = cone(torch.tensor([[0.0, 0.0, 0.0]])).item()


def test_rounded_cone():
    """Test RoundedConeSDF."""
    cone = RoundedConeSDF(1, 0.5, 2)
    # Apex
    val = cone(torch.tensor([[0.0, 0.0, 0.0]])).item()


def test_tetrahedron():
    tet = TetrahedronSDF(1.0)
    # Center should be inside
    assert tet(torch.tensor([[0.0, 0.0, 0.0]])).item() < 0


def test_octahedron():
    oct = OctahedronSDF(1.0)
    # Center should be inside
    assert oct(torch.tensor([[0.0, 0.0, 0.0]])).item() < 0


def test_dodecahedron():
    dod = DodecahedronSDF(1.0)
    # Center should be inside
    assert dod(torch.tensor([[0.0, 0.0, 0.0]])).item() < 0


def test_icosahedron():
    ico = IcosahedronSDF(1.0)
    # Center should be inside
    assert ico(torch.tensor([[0.0, 0.0, 0.0]])).item() < 0


# ==================== Transformation Tests ====================


def test_dilate_erode():
    """Test DilateSDF and ErodeSDF."""
    sphere = SphereSDF([0, 0, 0], 1.0)

    dilated = DilateSDF(sphere, 0.2)
    # Center should be more negative
    assert dilated(torch.tensor([[0.0, 0.0, 0.0]])).item() < -1.0

    eroded = ErodeSDF(sphere, 0.2)
    # Center should be less negative
    assert eroded(torch.tensor([[0.0, 0.0, 0.0]])).item() > -1.0


def test_shell():
    """Test ShellSDF."""
    sphere = SphereSDF([0, 0, 0], 1.0)
    shell = ShellSDF(sphere, 0.2)
    # Center should now be outside the shell (positive)
    assert shell(torch.tensor([[0.0, 0.0, 0.0]])).item() > 0


def test_twist():
    """Test TwistSDF transformation."""
    sphere = SphereSDF([0, 0, 0], 0.5)
    twisted = TwistSDF(sphere, np.pi / 2)  # 90 degree twist

    # At z=0, point should be on surface
    assert abs(twisted(torch.tensor([[0.5, 0.0, 0.0]])).item()) < 0.05

    # At z=1, point rotated 90 degrees should be on surface
    assert abs(twisted(torch.tensor([[0.0, 0.5, 1.0]])).item()) < 0.05


def test_repeat():
    """Test RepeatSDF infinite repetition."""
    sphere = SphereSDF([0, 0, 0], 0.3)
    repeated = RepeatSDF(sphere, [1.0, 1.0, 1.0])

    # Original location
    assert repeated(torch.tensor([[0.0, 0.0, 0.0]])).item() < 0

    # One spacing away - should also be inside due to repetition
    assert repeated(torch.tensor([[1.0, 0.0, 0.0]])).item() < 0


def test_mirror():
    """Test MirrorSDF symmetry."""
    sphere = SphereSDF([0.5, 0, 0], 0.3)
    mirrored = MirrorSDF(sphere, [0, 0, 0], [1, 0, 0])

    # Original location inside
    assert mirrored(torch.tensor([[0.5, 0, 0]])).item() < 0

    # Mirror location should also be inside
    assert mirrored(torch.tensor([[-0.5, 0, 0]])).item() < 0


def test_circular_array():
    """Test CircularArraySDF radial replication."""
    sphere = SphereSDF([1.0, 0, 0], 0.2)
    arrayed = CircularArraySDF(sphere, count=4, radius=1.0)

    # Original location on surface
    assert abs(arrayed(torch.tensor([[1.0, 0, 0]])).item()) < 0.05

    # 90 degrees rotated should also be on surface
    assert abs(arrayed(torch.tensor([[0.0, 1.0, 0]])).item()) < 0.05


def test_revolve():
    """Test RevolveSDF creates 3D from 2D."""
    from DeepSDFStruct.sdf_primitives import CircleSDF

    circle = CircleSDF([1.0, 0.0], 0.2)
    revolved = RevolveSDF(circle, axis="z")

    # On torus ring
    assert abs(revolved(torch.tensor([[1.0, 0, 0]])).item()) < 0.05


# ==================== Boolean Operation Tests ====================


def test_smooth_union():
    """Test SmoothUnionSDF blending."""
    sphere1 = SphereSDF([0, 0, 0], 1.0)
    sphere2 = SphereSDF([1.5, 0, 0], 1.0)

    # Sharp union
    sharp = UnionSDF(sphere1, sphere2)
    mid = torch.tensor([[0.75, 0, 0]])
    sharp_val = sharp(mid).item()

    # Smooth union - should be more negative (thicker junction)
    smooth = SmoothUnionSDF(sphere1, sphere2, k=0.2)
    smooth_val = smooth(mid).item()

    # Smooth should be more than sharp (blending adds material)
    assert smooth_val < sharp_val


def test_smooth_difference():
    """Test SmoothDifferenceSDF."""
    sphere_big = SphereSDF([0, 0, 0], 1.0)
    sphere_small = SphereSDF([0, 0, 0], 0.5)

    smooth_diff = SmoothDifferenceSDF(sphere_big, sphere_small, k=0.2)

    # Outside both should be positive
    assert smooth_diff(torch.tensor([[0.8, 0, 0]])).item() > 0

    # Inside big but outside small should be negative
    assert smooth_diff(torch.tensor([[0.6, 0, 0]])).item() < 0


def test_smooth_intersection():
    """Test SmoothIntersectionSDF."""
    sphere1 = SphereSDF([0, 0, 0], 1.0)
    sphere2 = SphereSDF([1.5, 0, 0], 1.0)

    smooth_int = SmoothIntersectionSDF(sphere1, sphere2, k=0.1)

    # Midpoint outside intersection should be positive
    assert smooth_int(torch.tensor([[0.75, 0, 0]])).item() > 0


# ==================== Transformation Invariance Tests ====================


def test_sphere_rotation_invariance():
    """Sphere rotation should produce identical SDF."""
    sphere = SphereSDF([0, 0, 0], 1.0)

    # Simple rotation matrix for 90 degrees around Z
    import math

    theta = math.pi / 2
    R = torch.tensor(
        [
            [math.cos(theta), -math.sin(theta), 0],
            [math.sin(theta), math.cos(theta), 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )

    rotated = TransformedSDF(sphere, rotationMatrix=R)

    test_point = torch.tensor([[0.5, 0.3, 0.2]])

    # Should be identical due to sphere symmetry
    assert abs(rotated(test_point).item() - sphere(test_point).item()) < 1e-5


def test_sphere_scaling():
    """Test sphere scaling equivalence."""
    sphere_r05 = SphereSDF([0, 0, 0], 0.5)
    scaled = TransformedSDF(sphere_r05, scaleFactor=2.0)
    sphere_r1 = SphereSDF([0, 0, 0], 1.0)

    test_point = torch.tensor([[0.0, 0.0, 0.0]])

    # Should be equivalent
    assert abs(scaled(test_point).item() - sphere_r1(test_point).item()) < 1e-5


# ==================== Comprehensive Import Test ====================


def test_all_primitives_callable():
    """Ensure all primitives can be instantiated and called."""
    primitives_2d = [
        CircleSDF([0, 0], 1),
        RectangleSDF([0, 0], [1, 1]),
        LineSDF([1, 0], [0, 0]),
        RoundedRectangleSDF([0, 0], [1, 1], 0.1),
        EquilateralTriangleSDF(1.0),
        HexagonSDF(1.0),
        PolygonSDF([[0, 0], [1, 0], [0.5, 0.866]]),
    ]

    primitives_3d = [
        SphereSDF([0, 0, 0], 1),
        BoxSDF([0, 0, 0], [1, 1, 1]),
        RoundedBoxSDF([0, 0, 0], [1, 1, 1], 0.1),
        WireframeBoxSDF([0, 0, 0], [1, 1, 1], 0.05),
        TorusSDF([0, 0, 0], [0, 0, 1], 1, 0.2),
        CylinderSDF([0, 0, 0], [0, 0, 1], 0.5, 2),
        ConeSDF([0, 0, 0], [0, 1, 0], 1, 2),
        PlaneSDF([0, 0, 0], [0, 1, 0]),
        CappedCylinderSDF([0, 0, -1], [0, 0, 1], 0.5),
        RoundedCylinderSDF(0.5, 0.1, 2),
        CappedConeSDF([0, 0, 0], [0, 0, 1], 1, 0.5),
        RoundedConeSDF(1, 0.5, 2),
        SlabSDF(z0=-1, z1=1),
        TetrahedronSDF(1.0),
        OctahedronSDF(1.0),
        DodecahedronSDF(1.0),
        IcosahedronSDF(1.0),
        CapsuleSDF([0, 0, -1], [0, 0, 1], 0.5),
        EllipsoidSDF([0, 0, 0], [1, 1, 1]),
        PyramidSDF(1.0),
        CornerSpheresSDF(0.2),
        CrossMsSDF(0.2),
    ]

    operations = [
        RepeatSDF(SphereSDF([0, 0, 0], 0.3), [1, 1, 1]),
        MirrorSDF(SphereSDF([0.5, 0, 0], 0.3), [0, 0, 0], [1, 0, 0]),
        CircularArraySDF(SphereSDF([1, 0, 0], 0.2), 4, 1.0),
        RevolveSDF(CircleSDF([1, 0], 0.2)),
    ]

    # Test all 2D primitives
    for sdf in primitives_2d:
        assert sdf.geometric_dim == 2
        result = sdf(torch.tensor([[0.0, 0.0]]))
        assert result.shape == (1, 1)

    # Test all 3D primitives
    for sdf in primitives_3d:
        assert sdf.geometric_dim == 3
        result = sdf(torch.tensor([[0.0, 0.0, 0.0]]))
        assert result.shape == (1, 1)

    # Test operations
    for sdf in operations:
        result = sdf(torch.tensor([[0.0, 0.0, 0.0]]))
        assert result.shape == (1, 1)


@pytest.fixture
def queries():
    torch.manual_seed(42)
    return torch.rand(10, 3)


def test_sdf_primitives(queries):

    # instantiate primitives
    sphere = SphereSDF(center=[0.0, 0.0, 0.0], radius=0.5)
    cylinder_x = CylinderSDF(
        point=[0.0, 0.0, 0.0], axis=[1, 0, 0], radius=0.3, height=1
    )
    torus = TorusSDF(
        center=[0.0, 0.0, 0.0], major_radius=0.5, minor_radius=0.2, axis=[0, 0, 1]
    )
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
    cyl_x = CylinderSDF(point=[0, 0, 0], axis=[1, 0, 0], radius=0.3, height=1)
    theta = pi / 2  # rotate x -> y
    R = torch.tensor(
        [[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]],
        dtype=torch.float32,
    )

    rotated_cyl = TransformedSDF(cyl_x, rotationMatrix=R)
    cyl_y = CylinderSDF(point=[0, 0, 0], axis=[0, 1, 0], radius=0.3, height=1)
    # queries = torch.tensor([[0.0, 0.0, 1.0]])
    val_rot = rotated_cyl._compute(queries)
    val_ref = cyl_y._compute(queries)
    assert torch.allclose(val_rot, val_ref, atol=1e-6)


def test_scaled_sphere(queries):
    """
    Scale a sphere and check equivalence with a sphere of different radius.
    """
    sphere_r03 = SphereSDF(center=[0, 0, 0], radius=0.3)
    sphere_r03_scaled = TransformedSDF(sphere_r03, scaleFactor=2.0)
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

    rotated_sphere = TransformedSDF(sphere, rotationMatrix=R)
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


def test_circle_2d():
    """Basic checks for CircleSDF (2D)."""
    circle = CircleSDF(center=[0.0, 0.0], radius=0.5)
    pts = torch.tensor([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    vals = circle(pts).reshape(-1)
    assert torch.allclose(vals[0], torch.tensor(-0.5), atol=1e-6)  # center
    assert torch.allclose(vals[1], torch.tensor(0.0), atol=1e-6)  # on surface
    assert torch.allclose(vals[2], torch.tensor(0.5), atol=1e-6)  # outside


def test_rectangle_2d():
    """Basic checks for RectangleSDF (2D)."""
    width = 0.5
    height = 0.2
    rect = RectangleSDF(center=[0.0, 0.0], extents=[width, height])
    pts = torch.tensor([[0.0, 0.0], [width / 2, 0.0], [1.0, 0.0]])
    vals = rect(pts).reshape(-1)
    assert torch.allclose(vals[0], torch.tensor(-height / 2), atol=1e-6)
    assert torch.allclose(vals[1], torch.tensor(0.0), atol=1e-6)  # on surface
    assert torch.allclose(vals[2], torch.tensor(0.75), atol=1e-6)  # outside


if __name__ == "__main__":
    print("Running comprehensive SDF tests...")
    queries = torch.rand(10, 3)
    test_rotated_cylinder(queries)
    test_rotated_sphere_equivalence(queries)
    test_scaled_sphere(queries)
    test_sdf_primitives(queries)
    test_translated_sphere(queries)
    test_circle_2d()
    test_rectangle_2d()
    # Run a few key tests

    test_circle_sdf_basic()
    print("✓ CircleSDF basic test passed")

    test_sphere_sdf_accuracy()
    print("✓ SphereSDF accuracy test passed")

    test_slab_sdf()
    print("✓ SlabSDF test passed")

    test_dilate_erode()
    print("✓ Dilate/Erode test passed")

    test_shell()
    print("✓ Shell test passed")

    test_twist()
    print("✓ Twist test passed")

    test_smooth_union()
    print("✓ SmoothUnion test passed")

    test_sphere_scaling()
    print("✓ Sphere scaling test passed")

    test_all_primitives_callable()
    print("✓ All primitives callable test passed")

    print("\n=== All basic tests passed! ===")
    print("Run with pytest for full test suite.")
