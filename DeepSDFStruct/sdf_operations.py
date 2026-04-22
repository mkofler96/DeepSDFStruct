"""
SDF Operations and Transformations
==================================

This module provides various operations that can be applied to SDFs,
including space warping transformations like elongation, twisting,
bending, dilation/erosion, and shell creation.
"""

import torch
import numpy as np
from DeepSDFStruct.SDF import SDFBase


class ElongateSDF(SDFBase):
    """Elongate an SDF by adding material along axes."""

    def __init__(self, sdf: SDFBase, size):
        super().__init__()
        self.sdf = sdf
        self.size = torch.nn.Parameter(torch.as_tensor(size, dtype=torch.float32))

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        size = self.size.to(device=queries.device, dtype=queries.dtype)
        q = torch.abs(queries) - size
        w = torch.minimum(
            torch.maximum(torch.maximum(q[:, 0], q[:, 1]), q[:, 2]), torch.tensor(0.0)
        )
        sdf_q = self.sdf(torch.clamp(q, min=0.0))
        return torch.maximum(sdf_q, w).reshape(-1, 1)

    def _get_domain_bounds(self) -> torch.Tensor:
        b = self.sdf._get_domain_bounds()
        return torch.stack([b[0] - self.size, b[1] + self.size])


class TwistSDF(SDFBase):
    """Twist an SDF around Z-axis by angle k*z."""

    def __init__(self, sdf: SDFBase, k):
        super().__init__()
        self.sdf = sdf
        self.k = torch.nn.Parameter(torch.as_tensor(k, dtype=torch.float32))

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        k = self.k.to(device=queries.device, dtype=queries.dtype)
        x, y, z = queries[:, 0], queries[:, 1], queries[:, 2]
        c = torch.cos(k * z)
        s = torch.sin(k * z)
        x2 = c * x - s * y
        y2 = s * x + c * y
        rotated = torch.stack([x2, y2, z], dim=1)
        return self.sdf(rotated)

    def _get_domain_bounds(self) -> torch.Tensor:
        return self.sdf._get_domain_bounds()


class BendLinearSDF(SDFBase):
    """Bend SDF linearly between two control points."""

    def __init__(self, sdf: SDFBase, p0, p1, v):
        super().__init__()
        self.sdf = sdf
        self.p0 = torch.nn.Parameter(torch.as_tensor(p0, dtype=torch.float32))
        self.p1 = torch.nn.Parameter(torch.as_tensor(p1, dtype=torch.float32))
        self.v = torch.nn.Parameter(torch.as_tensor(v, dtype=torch.float32))

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        p0 = self.p0.to(device=queries.device, dtype=queries.dtype)
        p1 = self.p1.to(device=queries.device, dtype=queries.dtype)
        v = self.v.to(device=queries.device, dtype=queries.dtype)

        ab = p1 - p0
        t = torch.clamp(torch.sum((queries - p0) * ab, dim=1) / torch.sum(ab**2), 0, 1)
        t = t.reshape(-1, 1)
        return self.sdf(queries + t * v)

    def _get_domain_bounds(self) -> torch.Tensor:
        b = self.sdf._get_domain_bounds()
        return b


class BendRadialSDF(SDFBase):
    """Bend SDF radially."""

    def __init__(self, sdf: SDFBase, r0, r1, dz):
        super().__init__()
        self.sdf = sdf
        self.r0 = torch.nn.Parameter(torch.as_tensor(r0, dtype=torch.float32))
        self.r1 = torch.nn.Parameter(torch.as_tensor(r1, dtype=torch.float32))
        self.dz = torch.nn.Parameter(torch.as_tensor(dz, dtype=torch.float32))

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        r0 = self.r0.to(device=queries.device, dtype=queries.dtype)
        r1 = self.r1.to(device=queries.device, dtype=queries.dtype)
        dz = self.dz.to(device=queries.device, dtype=queries.dtype)

        x = queries[:, 0]
        y = queries[:, 1]
        r = torch.sqrt(x**2 + y**2)
        t = torch.clamp((r - r0) / (r1 - r0), 0, 1)
        z = queries[:, 2] - dz * t

        p = torch.stack([queries[:, 0], queries[:, 1], z], dim=1)
        return self.sdf(p)

    def _get_domain_bounds(self) -> torch.Tensor:
        return self.sdf._get_domain_bounds()


class DilateSDF(SDFBase):
    """Expand an SDF uniformly by distance r."""

    def __init__(self, sdf: SDFBase, r):
        super().__init__()
        self.sdf = sdf
        self.r = torch.nn.Parameter(torch.as_tensor(r, dtype=torch.float32))

    def _compute(self, queries) -> torch.Tensor:
        r = self.r.to(device=queries.device, dtype=queries.dtype)
        return self.sdf(queries) - r

    def _get_domain_bounds(self) -> torch.Tensor:
        b = self.sdf._get_domain_bounds()
        # Expand bounds by r
        return torch.stack([b[0] - self.r, b[1] + self.r])


class ErodeSDF(SDFBase):
    """Contract an SDF uniformly by distance r."""

    def __init__(self, sdf: SDFBase, r):
        super().__init__()
        self.sdf = sdf
        self.r = torch.nn.Parameter(torch.as_tensor(r, dtype=torch.float32))

    def _compute(self, queries) -> torch.Tensor:
        r = self.r.to(device=queries.device, dtype=queries.dtype)
        return self.sdf(queries) + r

    def _get_domain_bounds(self) -> torch.Tensor:
        return self.sdf._get_domain_bounds()


class ShellSDF(SDFBase):
    """Create a hollow shell of thickness t."""

    def __init__(self, sdf: SDFBase, thickness):
        super().__init__()
        self.sdf = sdf
        self.thickness = torch.nn.Parameter(
            torch.as_tensor(thickness, dtype=torch.float32)
        )

    def _compute(self, queries) -> torch.Tensor:
        t = self.thickness.to(device=queries.device, dtype=queries.dtype)
        return torch.abs(self.sdf(queries)) - t / 2

    def _get_domain_bounds(self) -> torch.Tensor:
        return self.sdf._get_domain_bounds()


class RepeatSDF(SDFBase):
    """Infinite or finite grid repetition of an SDF."""

    def __init__(self, sdf: SDFBase, spacing, count=None):
        super().__init__()
        self.sdf = sdf
        self.spacing = torch.nn.Parameter(torch.as_tensor(spacing, dtype=torch.float32))
        self.count = count  # None for infinite, odd number for finite

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        spacing = self.spacing.to(device=queries.device, dtype=queries.dtype)

        if self.count is None:
            # Infinite repetition - use modulo to map back to base domain
            eps = torch.tensor(1e-9, device=queries.device, dtype=queries.dtype)
            q = (queries + eps) % spacing - eps
        else:
            # Finite repetition
            half = (self.count - 1) * spacing / 2
            eps = torch.tensor(1e-9, device=queries.device, dtype=queries.dtype)
            q = (queries + eps) % spacing - eps
            # Clamp to finite range
            q = torch.clamp(q, -half, half)

        return self.sdf(q)

    def _get_domain_bounds(self) -> torch.Tensor:
        if self.count is None:
            # Infinite bounds
            return torch.tensor([[-1e9, -1e9, -1e9], [1e9, 1e9, 1e9]])
        else:
            b = self.sdf._get_domain_bounds()
            span = (self.count - 1) * self.spacing
            return torch.stack([b[0] - span / 2, b[1] + span / 2])


class MirrorSDF(SDFBase):
    """Reflect an SDF across a plane."""

    def __init__(self, sdf: SDFBase, plane_point, plane_normal):
        super().__init__()
        self.sdf = sdf
        self.point = torch.nn.Parameter(
            torch.as_tensor(plane_point, dtype=torch.float32)
        )
        self.normal = torch.nn.Parameter(
            torch.as_tensor(plane_normal, dtype=torch.float32)
        )

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        pt = self.point.to(device=queries.device, dtype=queries.dtype)
        nm = self.normal.to(device=queries.device, dtype=queries.dtype)
        nm = nm / torch.linalg.norm(nm)

        # Signed distance to plane
        d = torch.sum((queries - pt) * nm, dim=1, keepdim=True)

        # Distance from original SDF
        d_orig = self.sdf(queries)

        # Distance from reflected SDF
        queries_reflected = queries - 2 * d * nm
        d_reflected = self.sdf(queries_reflected)

        # Take minimum (union of original and mirror)
        return torch.minimum(d_orig, d_reflected)

    def _get_domain_bounds(self) -> torch.Tensor:
        return self.sdf._get_domain_bounds()


class CircularArraySDF(SDFBase):
    """Create count copies of an SDF rotated around an axis."""

    def __init__(self, sdf: SDFBase, count, radius, axis="z"):
        super().__init__()
        self.sdf = sdf
        self.count = count
        self.radius = torch.nn.Parameter(torch.as_tensor(radius, dtype=torch.float32))
        self.axis = axis

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        r = self.radius.to(device=queries.device, dtype=queries.dtype)
        da = 2 * torch.pi / self.count

        # Get coordinates based on axis
        if self.axis == "z":
            x, y, z = queries[:, 0], queries[:, 1], queries[:, 2]
            dist = torch.sqrt(x**2 + y**2)
            angle = torch.arctan2(y, x) % da
            # Evaluate at angle and angle-da
            q1 = torch.stack(
                [torch.cos(angle - da) * dist, torch.sin(angle - da) * dist, z], dim=1
            )
            q2 = torch.stack(
                [torch.cos(angle) * dist, torch.sin(angle) * dist, z], dim=1
            )
        elif self.axis == "x":
            y, z, x = queries[:, 1], queries[:, 2], queries[:, 0]
            dist = torch.sqrt(y**2 + z**2)
            angle = torch.arctan2(z, y) % da
            q1 = torch.stack(
                [torch.cos(angle - da) * dist, torch.sin(angle - da) * dist, x], dim=1
            )
            q2 = torch.stack(
                [torch.cos(angle) * dist, torch.sin(angle) * dist, x], dim=1
            )
        elif self.axis == "y":
            z, x, y = queries[:, 2], queries[:, 0], queries[:, 1]
            dist = torch.sqrt(z**2 + x**2)
            angle = torch.arctan2(x, z) % da
            q1 = torch.stack(
                [torch.cos(angle - da) * dist, torch.sin(angle - da) * dist, y], dim=1
            )
            q2 = torch.stack(
                [torch.cos(angle) * dist, torch.sin(angle) * dist, y], dim=1
            )
        else:
            raise ValueError(f"Invalid axis: {self.axis}. Must be 'x', 'y', or 'z'")

        d1 = self.sdf(q1)
        d2 = self.sdf(q2)
        return torch.minimum(d1, d2)

    def _get_domain_bounds(self) -> torch.Tensor:
        # Conservative bounds
        b = self.sdf._get_domain_bounds()
        r = self.radius
        if self.axis == "z":
            return torch.stack(
                [
                    torch.tensor(
                        [min(b[0, 0], b[1, 0]) - r, min(b[0, 1], b[1, 1]) - r, b[0, 2]]
                    ),
                    torch.tensor(
                        [max(b[0, 0], b[1, 0]) + r, max(b[0, 1], b[1, 1]) + r, b[1, 2]]
                    ),
                ]
            )
        else:
            return b


class RevolveSDF(SDFBase):
    """Revolve a 2D profile to create 3D surface."""

    def __init__(self, sdf_2d: SDFBase, axis="z", offset=0.0):
        super().__init__()
        if sdf_2d.geometric_dim != 2:
            raise ValueError("RevolveSDF requires a 2D SDF")
        self.sdf_2d = sdf_2d
        self.axis = axis
        self.offset = torch.nn.Parameter(torch.as_tensor(offset, dtype=torch.float32))

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        offset = self.offset.to(device=queries.device, dtype=queries.dtype)

        if self.axis == "z":
            # Use x,y plane distance as radius, z as height in 2D space
            radius = torch.sqrt(queries[:, 0] ** 2 + queries[:, 1] ** 2) - offset
            pts_2d = torch.stack([radius, queries[:, 2]], dim=1)
        elif self.axis == "x":
            # Use y,z plane
            radius = torch.sqrt(queries[:, 1] ** 2 + queries[:, 2] ** 2) - offset
            pts_2d = torch.stack([radius, queries[:, 0]], dim=1)
        elif self.axis == "y":
            # Use x,z plane
            radius = torch.sqrt(queries[:, 0] ** 2 + queries[:, 2] ** 2) - offset
            pts_2d = torch.stack([radius, queries[:, 1]], dim=1)
        else:
            raise ValueError(f"Invalid axis: {self.axis}. Must be 'x', 'y', or 'z'")

        return self.sdf_2d(pts_2d)

    def _get_domain_bounds(self) -> torch.Tensor:
        b = self.sdf_2d._get_domain_bounds()
        # Conservative estimate for 3D bounds based on 2D bounds
        return torch.stack(
            [
                torch.tensor([min(b[0, 0], b[1, 0]), min(b[0, 0], b[1, 0]), b[0, 1]]),
                torch.tensor([max(b[0, 0], b[1, 0]), max(b[0, 0], b[1, 0]), b[1, 1]]),
            ]
        )
