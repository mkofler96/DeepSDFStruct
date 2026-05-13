"""
Test ElongateSDF and wrapper SDFs with SmoothUnion operations.

This test ensures that wrapper SDFs properly preserve tensor shapes
when used in combination with SmoothUnionSDF and other composite operations.
"""

import pytest
import torch
from DeepSDFStruct.SDF import SmoothUnionSDF, UnionSDF
from DeepSDFStruct.sdf_operations import (
    ElongateSDF,
    TwistSDF,
    BendLinearSDF,
    BendRadialSDF,
    DilateSDF,
    ErodeSDF,
    ShellSDF,
    MirrorSDF,
)
from DeepSDFStruct.sdf_primitives import SphereSDF, RoundedBoxSDF, CapsuleSDF


class TestElongateSDF:
    """Test ElongateSDF wrapper."""

    def test_basic_elongate(self):
        """Test basic ElongateSDF functionality."""
        sphere = SphereSDF(center=[0, 0, 0], radius=0.5)
        elongated = ElongateSDF(sphere, size=[0.5, 0.5, 0.5])

        queries = torch.randn(100, 3)
        result = elongated(queries)

        assert result.shape == (100, 1), f"Expected shape (100, 1), got {result.shape}"
        assert torch.isfinite(result).all(), "Result contains NaN or Inf"

    def test_elongate_with_smooth_union(self):
        """Test ElongateSDF in SmoothUnion - reproduces the original bug."""
        sphere = SphereSDF(center=[0, 0, 0], radius=0.3)
        elongated_sphere = ElongateSDF(sphere, size=[0.5, 0.2, 0.2])
        capsule = CapsuleSDF(point_a=[1.0, 0, 0], point_b=[1.5, 0, 0], radius=0.2)
        box = RoundedBoxSDF(center=[-1.0, 0, 0], extents=[0.5, 0.5, 0.5], radius=0.1)

        smooth_union = SmoothUnionSDF(elongated_sphere, capsule, box, k=0.1)

        queries = torch.randn(1000, 3)
        result = smooth_union(queries)

        assert result.shape == (
            1000,
            1,
        ), f"Expected shape (1000, 1), got {result.shape}"
        assert torch.isfinite(result).all(), "Result contains NaN or Inf"

    def test_elongate_shape_preservation(self):
        """Test that ElongateSDF preserves input tensor shape."""
        sphere = SphereSDF(center=[0, 0, 0], radius=0.5)
        elongated = ElongateSDF(sphere, size=[0.3, 0.3, 0.3])

        for n_points in [1, 10, 100, 1000, 41616]:
            queries = torch.randn(n_points, 3)
            result = elongated(queries)
            assert result.shape == (
                n_points,
                1,
            ), f"Shape mismatch for {n_points} points: expected ({n_points}, 1), got {result.shape}"

    def test_elongate_gradient_flow(self):
        """Test that gradients flow through ElongateSDF."""
        sphere = SphereSDF(center=[0, 0, 0], radius=0.5)
        elongated = ElongateSDF(sphere, size=[0.3, 0.3, 0.3])

        queries = torch.randn(100, 3, requires_grad=True)
        result = elongated(queries)
        loss = result.sum()
        loss.backward()

        assert queries.grad is not None, "Gradients not computed"
        assert torch.isfinite(queries.grad).all(), "Gradients contain NaN or Inf"


class TestWrapperSDFsWithSmoothUnion:
    """Test various wrapper SDFs with SmoothUnion."""

    def test_twist_sdf_smooth_union(self):
        """Test TwistSDF in SmoothUnion."""
        sphere = SphereSDF(center=[0, 0, 0], radius=0.5)
        twisted = TwistSDF(sphere, k=0.5)
        box = RoundedBoxSDF(center=[1.0, 0, 0], extents=[0.3, 0.3, 0.3], radius=0.05)

        smooth_union = SmoothUnionSDF(twisted, box, k=0.1)

        queries = torch.randn(500, 3)
        result = smooth_union(queries)

        assert result.shape == (500, 1), f"Expected shape (500, 1), got {result.shape}"
        assert torch.isfinite(result).all(), "Result contains NaN or Inf"

    def test_bend_linear_sdf_smooth_union(self):
        """Test BendLinearSDF in SmoothUnion."""
        sphere = SphereSDF(center=[0, 0, 0], radius=0.3)
        bent = BendLinearSDF(sphere, p0=[0, 0, 0], p1=[0, 0, 1], v=[0.1, 0, 0])
        capsule = CapsuleSDF(point_a=[0.5, 0, 0], point_b=[1.0, 0, 0], radius=0.2)

        smooth_union = SmoothUnionSDF(bent, capsule, k=0.1)

        queries = torch.randn(500, 3)
        result = smooth_union(queries)

        assert result.shape == (500, 1), f"Expected shape (500, 1), got {result.shape}"
        assert torch.isfinite(result).all(), "Result contains NaN or Inf"

    def test_dilate_sdf_smooth_union(self):
        """Test DilateSDF in SmoothUnion."""
        sphere = SphereSDF(center=[0, 0, 0], radius=0.3)
        dilated = DilateSDF(sphere, r=0.1)
        box = RoundedBoxSDF(center=[0.8, 0, 0], extents=[0.3, 0.3, 0.3], radius=0.05)

        smooth_union = SmoothUnionSDF(dilated, box, k=0.1)

        queries = torch.randn(500, 3)
        result = smooth_union(queries)

        assert result.shape == (500, 1), f"Expected shape (500, 1), got {result.shape}"
        assert torch.isfinite(result).all(), "Result contains NaN or Inf"

    def test_mirror_sdf_smooth_union(self):
        """Test MirrorSDF in SmoothUnion."""
        sphere = SphereSDF(center=[0.5, 0.5, 0], radius=0.3)
        mirrored = MirrorSDF(sphere, plane_point=[0, 0, 0], plane_normal=[1, 0, 0])
        capsule = CapsuleSDF(point_a=[0, 0, 0], point_b=[0, 1, 0], radius=0.2)

        smooth_union = SmoothUnionSDF(mirrored, capsule, k=0.1)

        queries = torch.randn(500, 3)
        result = smooth_union(queries)

        assert result.shape == (500, 1), f"Expected shape (500, 1), got {result.shape}"
        assert torch.isfinite(result).all(), "Result contains NaN or Inf"


class TestFormula1CarScenario:
    """Test the specific scenario from the bug report - Formula 1 car chassis."""

    def test_main_chassis_smooth_union(self):
        """Test MainChassis configuration from bug report."""
        rounded_box = RoundedBoxSDF(
            center=[0, 0, 0.4], extents=[2.5, 0.8, 0.5], radius=0.15
        )
        elongated = ElongateSDF(rounded_box, size=[2.5, 0.4, 0.3])

        capsule = CapsuleSDF(point_a=[1.5, 0, 0.3], point_b=[2.5, 0, 0.3], radius=0.25)

        rounded_box2 = RoundedBoxSDF(
            center=[-1.5, 0, 0.5], extents=[1.5, 0.7, 0.4], radius=0.1
        )

        smooth_union = SmoothUnionSDF(elongated, capsule, rounded_box2, k=0.1)

        queries = torch.randn(41616, 3)
        result = smooth_union(queries)

        assert result.shape == (
            41616,
            1,
        ), f"Expected shape (41616, 1), got {result.shape}"
        assert torch.isfinite(result).all(), "Result contains NaN or Inf"

    def test_side_pod_smooth_union(self):
        """Test SidePod configuration from bug report."""
        rounded_box = RoundedBoxSDF(
            center=[-0.5, 0.55, 0.35], extents=[1.2, 0.35, 0.35], radius=0.12
        )

        from DeepSDFStruct.sdf_primitives import CappedConeSDF

        capped_cone = CappedConeSDF(
            point_a=[0.5, 0.55, 0.35], point_b=[-1.2, 0.55, 0.35], ra=0.25, rb=0.15
        )

        smooth_union = SmoothUnionSDF(rounded_box, capped_cone, k=0.15)

        queries = torch.randn(1000, 3)
        result = smooth_union(queries)

        assert result.shape == (
            1000,
            1,
        ), f"Expected shape (1000, 1), got {result.shape}"
        assert torch.isfinite(result).all(), "Result contains NaN or Inf"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
