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


def cubic_bezier_distance(p, control_points, samples=500, chunk_size=1000, refine=True):
    """
    Compute distance from point p to cubic bezier curve using sampling with refinement.

    Samples the curve at many points and finds the closest one, then refines locally.

    Args:
        p: Query point(s), shape (N, 3)
        control_points: Control points tensor, shape (4, 3)
        samples: Number of samples along the curve
        chunk_size: Number of query points to process at a time (for memory efficiency)
        refine: Whether to refine the minimum with local optimization

    Returns:
        Tuple of (dist, t, closest_point) where:
            - dist: distance to curve, shape (N, 1)
            - t: parameter value at closest point, shape (N, 1)
            - closest_point: closest point on curve, shape (N, 3)
    """
    if p.dim() == 1:
        p = p.unsqueeze(0)

    # Move control points to same device as p
    control_points = control_points.to(device=p.device, dtype=p.dtype)

    p0, p1, p2, p3 = control_points

    # Sample the curve once (shared across all queries)
    t_samples = torch.linspace(0, 1, samples, device=p.device, dtype=p.dtype)
    u_samples = 1 - t_samples

    # Evaluate cubic bezier at all sample points
    t_vec = t_samples.unsqueeze(1)  # (samples, 1)
    u_vec = u_samples.unsqueeze(1)  # (samples, 1)

    curve_points = (
        (u_vec**3) * p0
        + (3 * u_vec**2 * t_vec) * p1
        + (3 * u_vec * t_vec**2) * p2
        + (t_vec**3) * p3
    )  # (samples, 3)

    # Initialize output tensors
    n_points = p.shape[0]
    dtype = p.dtype
    device = p.device

    min_dists = torch.full((n_points, 1), float("inf"), dtype=dtype, device=device)
    min_idx = torch.zeros(n_points, dtype=torch.long, device=device)

    # Process in chunks to reduce memory
    for i in range(0, n_points, chunk_size):
        end_idx = min(i + chunk_size, n_points)
        chunk = p[i:end_idx]  # (chunk_size, 3)

        # Compute distances for this chunk
        chunk_expanded = chunk.unsqueeze(1)  # (chunk_size, 1, 3)
        curve_expanded = curve_points.unsqueeze(0)  # (1, samples, 3)

        diff = chunk_expanded - curve_expanded  # (chunk_size, samples, 3)
        dists_chunk = torch.linalg.norm(diff, dim=2)  # (chunk_size, samples)

        # Find minimum for this chunk
        min_dists_chunk, min_idx_chunk = torch.min(
            dists_chunk, dim=1
        )  # (chunk_size), (chunk_size)

        min_dists[i:end_idx] = min_dists_chunk.unsqueeze(1)
        min_idx[i:end_idx] = min_idx_chunk

    # Get initial best t and points
    t_closest = t_samples[min_idx].unsqueeze(1)  # (N, 1)
    point_closest = curve_points[min_idx]  # (N, 3)

    # Local refinement around the minimum (optional)
    if refine:
        # Process refinement in chunks as well to save memory
        for i in range(0, n_points, chunk_size):
            end_idx = min(i + chunk_size, n_points)
            chunk = p[i:end_idx]
            t_chunk_base = t_closest[i:end_idx]

            # Create refined samples around each minimum for this chunk
            refine_points = 20
            refine_delta = 0.02  # Larger refinement window

            t_refine_base = t_chunk_base.squeeze(1)  # (chunk,)
            t_refine_start = torch.clamp(t_refine_base - refine_delta, 0, 1)
            t_refine_width = (
                torch.clamp(t_refine_base + refine_delta, 0, 1) - t_refine_start
            )

            min_dists_chunk = min_dists[i:end_idx].clone()
            min_t_chunk = t_chunk_base.clone()

            # Evaluate at multiple refinement points for this chunk
            for j in range(refine_points + 1):
                t_refine = t_refine_start + (t_refine_width * j / refine_points)
                t_refine_exp = t_refine.unsqueeze(1)  # (chunk, 1)

                # Evaluate bezier at this t
                u_val = 1 - t_refine_exp
                curve_at_t = (
                    (u_val**3) * p0
                    + (3 * u_val**2 * t_refine_exp) * p1
                    + (3 * u_val * t_refine_exp**2) * p2
                    + (t_refine_exp**3) * p3
                )

                # Compute distances
                diff = chunk - curve_at_t  # (chunk, 3)
                dists = torch.linalg.norm(diff, dim=1, keepdim=True)

                # Update minimum if better
                mask = dists < min_dists_chunk
                min_dists_chunk = torch.where(mask, dists, min_dists_chunk)
                min_t_chunk = torch.where(mask, t_refine_exp, min_t_chunk)

            # Update global results with refined chunk results
            min_dists[i:end_idx] = min_dists_chunk
            t_closest[i:end_idx] = min_t_chunk

        # Recompute closest points with refined t values
        t_closest_exp = t_closest
        point_closest = (
            (1 - t_closest_exp) ** 3 * p0
            + 3 * (1 - t_closest_exp) ** 2 * t_closest_exp * p1
            + 3 * (1 - t_closest_exp) * t_closest_exp**2 * p2
            + t_closest_exp**3 * p3
        )

    return min_dists, t_closest, point_closest


class SweepSDF(SDFBase):
    """Sweep a 2D profile along a cubic bezier curve with flat end caps."""

    def __init__(
        self,
        profile_sdf: SDFBase,
        trajectory,
        bezier_samples=500,
        cap_ends=True,
        chunk_size=1000,
        refine=True,
    ):
        super().__init__()
        assert profile_sdf.geometric_dim == 2, "SweepSDF requires a 2D profile SDF"
        self.profile_sdf = profile_sdf
        self.trajectory = trajectory
        self.bezier_samples = bezier_samples
        self.cap_ends = cap_ends
        self.chunk_size = chunk_size
        self.refine = refine

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        # Ensure queries has shape (N, 3)
        original_shape = queries.shape
        if queries.dim() == 1:
            queries = queries.unsqueeze(0)

        control_points = self.trajectory.control_points
        device = queries.device

        # Move control points to correct device
        control_points = control_points.to(device=device, dtype=queries.dtype)

        n_points = queries.shape[0]

        # Initialize output
        result = torch.zeros(n_points, 1, dtype=queries.dtype, device=device)

        # Process in chunks to reduce memory
        for i in range(0, n_points, self.chunk_size):
            end_idx = min(i + self.chunk_size, n_points)
            chunk_queries = queries[i:end_idx]  # (chunk, 3)

            # Find closest point and parameter t on bezier for this chunk
            dist_to_curve, t, closest_point = cubic_bezier_distance(
                chunk_queries,
                control_points,
                self.bezier_samples,
                chunk_size=self.chunk_size,
                refine=self.refine,
            )

            # Compute tangent at closest point
            t_values = t[:, 0:1]
            u = 1 - t_values
            p0, p1, p2, p3 = control_points

            tangent_per_point = (
                (3 * u**2) * (p1 - p0).unsqueeze(0)
                + (6 * u * t_values) * (p2 - p1).unsqueeze(0)
                + (3 * t_values**2) * (p3 - p2).unsqueeze(0)
            )

            tangent_norms = torch.linalg.norm(tangent_per_point, dim=1, keepdim=True)
            tangent = tangent_per_point / (tangent_norms + 1e-10)

            # Compute normal using curvature (second derivative)
            curvature_per_point = (6 * u) * (p2 - 2 * p1 + p0).unsqueeze(0) + (
                6 * t_values
            ) * (p3 - 2 * p2 + p1).unsqueeze(0)

            normal = torch.zeros_like(chunk_queries)
            curvature_norms = torch.linalg.norm(
                curvature_per_point, dim=1, keepdim=True
            )
            has_curvature = curvature_norms.squeeze(1) > 1e-10

            if has_curvature.any():
                normal_curv = curvature_per_point / (curvature_norms + 1e-10)
                proj = torch.sum(normal_curv * tangent, dim=1, keepdim=True) * tangent
                normal_curv = normal_curv - proj
                normal_curv_norms = torch.linalg.norm(normal_curv, dim=1, keepdim=True)
                normal_curv = normal_curv / (normal_curv_norms + 1e-10)
                normal = torch.where(
                    has_curvature.unsqueeze(1).expand_as(normal), normal_curv, normal
                )

            no_curvature = ~has_curvature
            if no_curvature.any():
                try_normal = torch.tensor(
                    [0.0, 1.0, 0.0], device=device, dtype=chunk_queries.dtype
                )
                try_normal = try_normal.unsqueeze(0).expand(chunk_queries.shape[0], -1)

                dot_with_tangent = torch.sum(tangent * try_normal, dim=1)

                parallel_to_y = torch.abs(dot_with_tangent) > 0.9
                n_base = torch.where(
                    parallel_to_y.unsqueeze(1).expand_as(try_normal),
                    torch.tensor(
                        [1.0, 0.0, 0.0], device=device, dtype=chunk_queries.dtype
                    )
                    .unsqueeze(0)
                    .expand(chunk_queries.shape[0], -1),
                    try_normal,
                )

                proj = torch.sum(n_base * tangent, dim=1, keepdim=True) * tangent
                n_perp = n_base - proj
                n_perp_norms = torch.linalg.norm(n_perp, dim=1, keepdim=True)
                n_perp = n_perp / (n_perp_norms + 1e-10)

                normal = torch.where(
                    no_curvature.unsqueeze(1).expand_as(normal), n_perp, normal
                )

            # Compute binormal
            binormal = torch.linalg.cross(tangent, normal, dim=1)
            binormal_norm = torch.linalg.norm(binormal, dim=1, keepdim=True)
            binormal = binormal / (binormal_norm + 1e-10)

            # Vector from closest point to query
            vec_to_query = chunk_queries - closest_point

            # Project onto normal and binormal to get profile coordinates
            coord_n = torch.sum(vec_to_query * normal, dim=1, keepdim=True)
            coord_b = torch.sum(vec_to_query * binormal, dim=1, keepdim=True)

            # Profile SDF expects 2D points (u, v)
            pts_2d = torch.cat([coord_n, coord_b], dim=1)

            # Evaluate profile SDF
            profile_dist = self.profile_sdf(pts_2d)

            chunk_result = profile_dist

            # Add end caps if requested
            if self.cap_ends:
                start_point = control_points[0]
                end_point = control_points[3]

                # Tangent at t=0: 3(P1 - P0)
                start_tangent = 3 * (control_points[1] - control_points[0])
                start_tangent_norm = torch.linalg.norm(start_tangent)
                if start_tangent_norm > 1e-10:
                    start_tangent = start_tangent / start_tangent_norm

                # Tangent at t=1: 3(P3 - P2)
                end_tangent = 3 * (control_points[3] - control_points[2])
                end_tangent_norm = torch.linalg.norm(end_tangent)
                if end_tangent_norm > 1e-10:
                    end_tangent = end_tangent / end_tangent_norm

                dist_to_start_plane = torch.sum(
                    (chunk_queries - start_point) * start_tangent, dim=1, keepdim=True
                )
                start_cap_dist = -dist_to_start_plane

                dist_to_end_plane = torch.sum(
                    (chunk_queries - end_point) * end_tangent, dim=1, keepdim=True
                )
                end_cap_dist = dist_to_end_plane

                chunk_result = torch.maximum(chunk_result, start_cap_dist)
                chunk_result = torch.maximum(chunk_result, end_cap_dist)

            result[i:end_idx] = chunk_result

        return result

    def _get_domain_bounds(self) -> torch.Tensor:
        cp = self.trajectory.control_points
        curve_min = cp.min(dim=0).values
        curve_max = cp.max(dim=0).values

        profile_bounds = self.profile_sdf._get_domain_bounds()
        profile_radius = max(abs(profile_bounds[0, 0]), abs(profile_bounds[1, 0]))

        lower = curve_min - profile_radius
        upper = curve_max + profile_radius

        return torch.stack([lower, upper])
