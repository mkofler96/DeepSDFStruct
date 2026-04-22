"""
Primitive SDF Shapes
====================

This module provides basic geometric primitive SDFs that can be used as building
blocks for more complex geometries. All primitives inherit from SDFBase and can
be combined using boolean operations.

Available Primitives
--------------------
- SphereSDF: Sphere with specified center and radius
- CylinderSDF: Infinite cylinder along a coordinate axis
- TorusSDF: Torus with major and minor radii
- PlaneSDF: Half-space defined by a point and normal vector
- CornerSpheresSDF: Cube with spherical cutouts at corners
- CrossMsSDF: Cross-shaped structure (intersection of three cylinders)
- RoundedBoxSDF: 3D axis-aligned box with rounded corners
- WireframeBoxSDF: 3D wireframe box SDF
- CapsuleSDF: SDF for a capsule (a line segment with a radius)
- EllipsoidSDF: SDF for an ellipsoid
- PyramidSDF: SDF for a pyramid

All primitives support PyTorch's automatic differentiation and can be used
in optimization workflows.
"""

from DeepSDFStruct.SDF import SDFBase
import torch
import numpy as np


class SphereSDF(SDFBase):
    """Signed distance function for a sphere.

    Computes the signed distance from query points to a sphere surface.
    The distance is negative inside the sphere, zero on the surface,
    and positive outside.

    Parameters
    ----------
    center : array-like of shape (3,)
        Center point of the sphere in 3D space.
    radius : float
        Radius of the sphere.

    Examples
    --------
    >>> import torch
    >>> from DeepSDFStruct.sdf_primitives import SphereSDF
    >>>
    >>> sphere = SphereSDF(center=[0, 0, 0], radius=1.0)
    >>> points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    >>> distances = sphere(points)
    >>> print(distances)  # [-1.0, 0.0] (center, surface)
    """

    def __init__(self, center, radius):
        super().__init__()
        # make center and radius trainable parameters and ensure correct dtype
        c = torch.as_tensor(center, dtype=torch.float32)
        r = torch.as_tensor(radius, dtype=torch.float32)
        self.center = torch.nn.Parameter(c)
        self.r = torch.nn.Parameter(r.reshape(()))  # scalar parameter

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        # ensure computations use same dtype/device as queries
        center = self.center.to(device=queries.device, dtype=queries.dtype)
        r = self.r.to(device=queries.device, dtype=queries.dtype)
        return (torch.linalg.norm(queries - center, dim=1) - r).reshape(-1, 1)

    def _get_domain_bounds(self) -> torch.Tensor:
        # Use same dtype/device as parameters
        center = self.center
        r = self.r

        # Compute axis-aligned bounding box of the sphere
        lower = center - r
        upper = center + r

        return torch.stack([lower, upper], dim=0)


class BoxSDF(SDFBase):
    def __init__(self, center, extents):
        """3D axis-aligned box SDF.

        extents defines full widths in x, y, z.
        Both center and extents are torch parameters.

                +--------+
               /|       /|
              +--------+ |
              | |      | |
              | +------|-+
              |/       |/
              +--------+

        """
        super().__init__(geometric_dim=3)

        c = torch.as_tensor(center, dtype=torch.float32)
        e = torch.as_tensor(extents, dtype=torch.float32)

        if c.numel() != 3 or e.numel() != 3:
            raise ValueError("center and extents must be length-3 for BoxSDF")

        self.center = torch.nn.Parameter(c.reshape(3))
        self.extents = torch.nn.Parameter(e.reshape(3))

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        center = self.center.to(device=queries.device, dtype=queries.dtype)
        half = self.extents.to(device=queries.device, dtype=queries.dtype) / 2.0

        q = queries - center  # (N,3)
        d = torch.abs(q) - half  # (N,3)

        zero = torch.tensor(0.0, device=queries.device, dtype=queries.dtype)

        # outside distance
        d_clamped = torch.maximum(d, zero)
        outside_dist = torch.linalg.norm(d_clamped, dim=1)

        # inside distance (negative inside)
        inside_dist = torch.minimum(
            torch.maximum(torch.maximum(d[:, 0], d[:, 1]), d[:, 2]), zero
        )

        sdf = (outside_dist + inside_dist).reshape(-1, 1)
        return sdf

    def _get_domain_bounds(self) -> torch.Tensor:
        center = self.center.detach()
        half = self.extents.detach() / 2.0

        lower = center - half
        upper = center + half

        return torch.stack([lower, upper], dim=0)


class CylinderSDF(SDFBase):
    """Signed distance function for an infinite cylinder.

    Creates a cylinder extending infinitely along a specified coordinate axis.
    The cylinder is defined by a point on the axis and a radius.

    Parameters
    ----------
    point : array-like of shape (3,)
        A point on the cylinder's axis.
    axis : str or int
        Axis direction: 'x'/0, 'y'/1, or 'z'/2.
    radius : float
        Radius of the cylinder.

    Examples
    --------
    >>> from DeepSDFStruct.sdf_primitives import CylinderSDF
    >>> import torch
    >>>
    >>> # Cylinder along z-axis
    >>> cylinder = CylinderSDF(point=[0, 0, 0], axis='z', radius=0.5)
    >>> points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    >>> distances = cylinder(points)
    >>> print(distances)  # [-0.5, 0.5] (on axis, outside)
    """

    def __init__(self, point, axis, radius, height):
        super().__init__()
        self.point = torch.tensor(point, dtype=torch.float32)
        self.axis = torch.tensor(axis, dtype=torch.float32)
        self.r = radius
        self.h = height

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        point = self.point.to(device=queries.device, dtype=queries.dtype)
        axis = self.axis.to(device=queries.device, dtype=queries.dtype)

        # normalize axis direction
        axis_norm = torch.linalg.norm(axis)
        if axis_norm == 0:
            raise ValueError("Cylinder axis vector must be non-zero")
        axis_unit = axis / axis_norm

        # vector from axis point to queries
        diff = queries - point  # (N,3)

        # project out axial component
        # dot = (diff · axis_unit)
        dot = torch.sum(diff * axis_unit, dim=1, keepdim=True)  # (N,1)
        diff_perp = diff - dot * axis_unit  # (N,3)

        # distance to axis
        dist_radial = torch.linalg.norm(diff_perp, dim=1, keepdim=True)

        d_r = dist_radial - self.r
        d_a = torch.abs(dot) - (self.h / 2.0)

        external_dist = torch.linalg.norm(
            torch.clamp(torch.cat([d_r, d_a], dim=1), min=0), dim=1, keepdim=True
        )
        internal_dist = torch.clamp(torch.max(d_r, d_a), max=0)

        return external_dist + internal_dist

    def _get_domain_bounds(self) -> torch.Tensor:
        """
        Calculates a bounding box that encapsulates the cylinder.
        Uses the radius and height to create a tight-ish box around the center.
        """
        # For a general oriented cylinder, the bounding box is a bit complex.
        # Here we use a conservative bound: a cube that fits the maximum extent.
        extent = max(self.r, self.h / 2.0)

        min_bound = self.point - extent
        max_bound = self.point + extent

        return torch.stack([min_bound, max_bound])


class ConeSDF(SDFBase):
    """Signed distance function for a finite cone.

    The cone is defined by:
    - apex point
    - axis direction
    - height
    - base radius

    Parameters
    ----------
    point : array-like of shape (3,)
        Apex position of the cone.
    axis : array-like of shape (3,)
        Direction vector of the cone axis.
    radius : float
        Radius at the base of the cone.
    height : float
        Height of the cone from apex to base.

    Example
    -------
    >>> cone = ConeSDF(point=[0,0,0], axis=[0,1,0], radius=1.0, height=2.0)
    """

    def __init__(self, apexpoint, axis, radius, height):
        super().__init__()
        apexpoint = torch.as_tensor(apexpoint, dtype=torch.float32)
        axis = torch.as_tensor(axis, dtype=torch.float32)
        radius = torch.as_tensor([radius], dtype=torch.float32)
        height = torch.as_tensor([height], dtype=torch.float32)
        self.point = torch.nn.Parameter(apexpoint)
        self.axis = torch.nn.Parameter(axis)
        self.radius = torch.nn.Parameter(radius)
        self.height = torch.nn.Parameter(height)
        self.geometric_dim = 3

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        point = self.point.to(device=queries.device, dtype=queries.dtype)
        axis = self.axis.to(device=queries.device, dtype=queries.dtype)

        # Normalize axis
        axis_norm = torch.linalg.norm(axis)
        if axis_norm == 0:
            raise ValueError("Cone axis vector must be non-zero")
        axis_unit = axis / axis_norm

        # Vector from apex to query points
        diff = queries - point  # (N,3)

        # Axial projection
        y = torch.sum(diff * axis_unit, dim=1, keepdim=True)  # (N,1)

        # Radial component
        radial_vec = diff - y * axis_unit
        x = torch.linalg.norm(radial_vec, dim=1, keepdim=True)

        # Linear radius scaling along height
        # radius(y) = (y/h) * r
        k = self.radius / self.height
        r_at_y = k * y

        # Distance to infinite cone surface
        d_side = x - r_at_y

        # Distance to base plane
        d_base = y - self.height

        # Distance to apex cap (prevent below apex)
        d_apex = -y

        # Outside distance (Euclidean corner handling)
        external = torch.linalg.norm(
            torch.clamp(torch.cat([d_side, d_base], dim=1), min=0), dim=1, keepdim=True
        )

        # Inside distance
        internal = torch.clamp(torch.max(d_side, torch.max(d_base, d_apex)), max=0)

        return external + internal

    def _get_domain_bounds(self) -> torch.Tensor:
        """
        Conservative bounding box.
        """
        extent = max(self.radius, self.height)
        min_bound = self.point - extent
        max_bound = self.point + extent
        return torch.stack([min_bound, max_bound])


class TorusSDF(SDFBase):
    """
    Torus SDF with trainable parameters.

    Parameters
    ----------
    center : array-like (3,)
        Center of the torus.
    axis : array-like (3,)
        Axis normal to the torus ring plane.
    major_radius : float
        Distance from center to tube center (R).
    minor_radius : float
        Tube radius (r).
    """

    def __init__(self, center, axis, major_radius, minor_radius):
        super().__init__()
        # store as trainable parameters
        c = torch.as_tensor(center, dtype=torch.float32)
        a = torch.as_tensor(axis, dtype=torch.float32)
        R = torch.as_tensor(major_radius, dtype=torch.float32)
        r = torch.as_tensor(minor_radius, dtype=torch.float32)

        self.center = torch.nn.Parameter(c)
        self.axis = torch.nn.Parameter(a)
        self.R = torch.nn.Parameter(R.reshape(()))  # scalar
        self.r = torch.nn.Parameter(r.reshape(()))  # scalar

        self.geometric_dim = 3

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        # align dtype/device
        center = self.center.to(device=queries.device, dtype=queries.dtype)
        axis = self.axis.to(device=queries.device, dtype=queries.dtype)
        R = self.R.to(device=queries.device, dtype=queries.dtype)
        r = self.r.to(device=queries.device, dtype=queries.dtype)

        # normalize axis
        axis_norm = torch.linalg.norm(axis)
        if axis_norm.item() == 0:
            raise ValueError("Torus axis must be non-zero")
        axis_unit = axis / axis_norm

        diff = queries - center  # (N,3)

        # axial projection (height from ring plane)
        y = torch.sum(diff * axis_unit, dim=1, keepdim=True)  # (N,1)

        # radial length in ring plane
        radial_vec = diff - y * axis_unit
        radial_len = torch.linalg.norm(radial_vec, dim=1, keepdim=True)  # (N,1)

        q = torch.cat([radial_len - R, y], dim=1)  # (N,2)
        out = torch.linalg.norm(q, dim=1, keepdim=True) - r  # (N,1)
        return out.reshape(-1, 1)

    def _get_domain_bounds(self) -> torch.Tensor:
        # conservative AABB
        center = self.center
        extent = (self.R + self.r).reshape(())
        lower = center - extent
        upper = center + extent
        return torch.stack([lower, upper], dim=0)


class PlaneSDF(SDFBase):
    def __init__(self, point, normal):
        super().__init__()
        self.point = torch.tensor(point, dtype=torch.float32)
        self.normal = torch.tensor(normal, dtype=torch.float32)
        self.normal = self.normal / torch.linalg.norm(self.normal)

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        return torch.matmul(queries - self.point, self.normal).reshape(-1, 1)

    def _get_domain_bounds(self) -> torch.Tensor:
        return torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])


class CornerSpheresSDF(SDFBase):
    def __init__(self, radius, limit=1.0):
        super().__init__()
        self.r = radius
        self.limit = limit

        # define the 8 corners of the cube
        self.corners = torch.tensor(
            [[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]],
            dtype=torch.float32,
        )

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:

        # start with the cube SDF
        output = torch.linalg.norm(queries, dim=1, ord=float("inf")) - self.limit

        # subtract spheres at corners
        for corner in self.corners:
            sphere_like = torch.linalg.norm(queries - corner, dim=1, ord=3) - self.r
            output = torch.maximum(output, -sphere_like)

        return output.reshape(-1, 1)

    def _get_domain_bounds(self) -> torch.Tensor:
        return torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])


class CrossMsSDF(SDFBase):
    def __init__(self, radius):
        super().__init__()
        self.r = radius

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        # start with the L∞ norm
        output = torch.linalg.norm(queries, dim=1, ord=float("inf"))

        # x-axis cylinder
        cylinder_x = torch.sqrt(queries[:, 1] ** 2 + queries[:, 2] ** 2) - self.r
        output = torch.minimum(output, cylinder_x)

        # y-axis cylinder
        cylinder_y = torch.sqrt(queries[:, 0] ** 2 + queries[:, 2] ** 2) - self.r
        output = torch.minimum(output, cylinder_y)

        # z-axis cylinder
        cylinder_z = torch.sqrt(queries[:, 0] ** 2 + queries[:, 1] ** 2) - self.r
        output = torch.minimum(output, cylinder_z)

        return output.reshape(-1, 1)

    def _get_domain_bounds(self) -> torch.Tensor:
        return torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])


class RoundedBoxSDF(SDFBase):
    """3D axis-aligned box with rounded corners."""

    def __init__(self, center, extents, radius):
        super().__init__(geometric_dim=3)
        c = torch.as_tensor(center, dtype=torch.float32)
        e = torch.as_tensor(extents, dtype=torch.float32)
        r = torch.as_tensor(radius, dtype=torch.float32)
        self.center = torch.nn.Parameter(c.reshape(3))
        self.extents = torch.nn.Parameter(e.reshape(3))
        self.radius = torch.nn.Parameter(r.reshape(()))

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        center = self.center.to(device=queries.device, dtype=queries.dtype)
        extents = self.extents.to(device=queries.device, dtype=queries.dtype)
        radius = self.radius.to(device=queries.device, dtype=queries.dtype)

        q = torch.abs(queries - center) - extents / 2.0 + radius

        outside_dist = torch.linalg.norm(torch.clamp(q, min=0.0), dim=1)
        inside_dist = torch.minimum(torch.max(q, dim=1).values, torch.tensor(0.0))

        return (outside_dist + inside_dist - radius).reshape(-1, 1)

    def _get_domain_bounds(self) -> torch.Tensor:
        center = self.center.detach()
        half = self.extents.detach() / 2.0
        lower = center - half
        upper = center + half
        return torch.stack([lower, upper], dim=0)


class WireframeBoxSDF(SDFBase):
    """3D wireframe box SDF."""

    def __init__(self, center, extents, thickness):
        super().__init__(geometric_dim=3)
        c = torch.as_tensor(center, dtype=torch.float32)
        e = torch.as_tensor(extents, dtype=torch.float32)
        t = torch.as_tensor(thickness, dtype=torch.float32)
        self.center = torch.nn.Parameter(c.reshape(3))
        self.extents = torch.nn.Parameter(e.reshape(3))
        self.thickness = torch.nn.Parameter(t.reshape(()))

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        center = self.center.to(device=queries.device, dtype=queries.dtype)
        extents = self.extents.to(device=queries.device, dtype=queries.dtype)
        thickness = self.thickness.to(device=queries.device, dtype=queries.dtype)

        p = torch.abs(queries - center) - extents / 2.0 - thickness / 2.0
        q = torch.abs(p + thickness / 2.0) - thickness / 2.0

        px, py, pz = p[:, 0], p[:, 1], p[:, 2]
        qx, qy, qz = q[:, 0], q[:, 1], q[:, 2]

        def g(a, b, c):
            return torch.linalg.norm(
                torch.clamp(torch.stack([a, b, c], dim=1), min=0.0), dim=1
            ) + torch.minimum(torch.max(torch.max(a, b), c), torch.tensor(0.0))

        return torch.minimum(
            torch.minimum(g(px, qy, qz), g(qx, py, qz)), g(qx, qy, pz)
        ).reshape(-1, 1)

    def _get_domain_bounds(self) -> torch.Tensor:
        center = self.center.detach()
        half = self.extents.detach() / 2.0
        lower = center - half
        upper = center + half
        return torch.stack([lower, upper], dim=0)


class CapsuleSDF(SDFBase):
    """SDF for a capsule (a line segment with a radius)."""

    def __init__(self, point_a, point_b, radius):
        super().__init__(geometric_dim=3)
        a = torch.as_tensor(point_a, dtype=torch.float32)
        b = torch.as_tensor(point_b, dtype=torch.float32)
        r = torch.as_tensor(radius, dtype=torch.float32)
        self.point_a = torch.nn.Parameter(a.reshape(3))
        self.point_b = torch.nn.Parameter(b.reshape(3))
        self.radius = torch.nn.Parameter(r.reshape(()))

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        point_a = self.point_a.to(device=queries.device, dtype=queries.dtype)
        point_b = self.point_b.to(device=queries.device, dtype=queries.dtype)
        radius = self.radius.to(device=queries.device, dtype=queries.dtype)

        pa = queries - point_a
        ba = point_b - point_a

        h = torch.clamp(
            torch.sum(pa * ba, dim=1) / torch.sum(ba * ba), 0.0, 1.0
        ).reshape(-1, 1)

        return (torch.linalg.norm(pa - h * ba, dim=1) - radius).reshape(-1, 1)

    def _get_domain_bounds(self) -> torch.Tensor:
        a = self.point_a.detach()
        b = self.point_b.detach()
        r = self.radius.detach()
        lower = torch.min(a, b) - r
        upper = torch.max(a, b) + r
        return torch.stack([lower, upper], dim=0)


class EllipsoidSDF(SDFBase):
    """SDF for an ellipsoid."""

    def __init__(self, center, extents):
        super().__init__(geometric_dim=3)
        c = torch.as_tensor(center, dtype=torch.float32)
        e = torch.as_tensor(extents, dtype=torch.float32)
        self.center = torch.nn.Parameter(c.reshape(3))
        self.extents = torch.nn.Parameter(e.reshape(3))

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        center = self.center.to(device=queries.device, dtype=queries.dtype)
        extents = self.extents.to(device=queries.device, dtype=queries.dtype)

        p = queries - center
        k0 = torch.linalg.norm(p / extents, dim=1)
        k1 = torch.linalg.norm(p / (extents * extents), dim=1)

        return (k0 * (k0 - 1.0) / k1).reshape(-1, 1)

    def _get_domain_bounds(self) -> torch.Tensor:
        center = self.center.detach()
        extents = self.extents.detach()
        lower = center - extents
        upper = center + extents
        return torch.stack([lower, upper], dim=0)


class PyramidSDF(SDFBase):
    """SDF for a pyramid."""

    def __init__(self, height):
        super().__init__(geometric_dim=3)
        h = torch.as_tensor(height, dtype=torch.float32)
        self.height = torch.nn.Parameter(h.reshape(()))

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        h = self.height.to(device=queries.device, dtype=queries.dtype)

        a = torch.abs(queries[:, :2]) - 0.5

        # This part is tricky to translate directly without a direct equivalent of the numpy advanced indexing in torch
        # For now, let's stick to a simplified version or assume a square base.
        # The original implementation handles non-square bases by swapping coordinates.

        px = a[:, 0]
        pz = a[:, 1]
        py = queries[:, 2]

        m2 = h * h + 0.25

        qx = pz
        qy = h * py - 0.5 * px
        qz = h * px + 0.5 * py

        s = torch.clamp(-qx, min=0.0)
        t = torch.clamp((qy - 0.5 * pz) / (m2 + 0.25), 0.0, 1.0)

        a_dist = m2 * (qx + s) ** 2 + qy**2
        b_dist = m2 * (qx + 0.5 * t) ** 2 + (qy - m2 * t) ** 2

        d2 = torch.where(
            (qy < 0) & (-qx * m2 - qy * 0.5 < 0),
            torch.min(a_dist, b_dist),
            torch.tensor(0.0, device=queries.device),
        )

        return (
            torch.sqrt((d2 + qz**2) / m2) * torch.sign(torch.max(qz, -py))
        ).reshape(-1, 1)

    def _get_domain_bounds(self) -> torch.Tensor:
        h = self.height.detach()
        return torch.tensor([[-0.5, -0.5, 0], [0.5, 0.5, h]])


class CircleSDF(SDFBase):
    """SDF for a 2D circle."""

    def __init__(self, center, radius):
        super().__init__(geometric_dim=2)
        self.center = torch.nn.Parameter(torch.as_tensor(center, dtype=torch.float32))
        self.radius = torch.nn.Parameter(
            torch.as_tensor(radius, dtype=torch.float32).reshape(())
        )

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        center = self.center.to(device=queries.device, dtype=queries.dtype)
        radius = self.radius.to(device=queries.device, dtype=queries.dtype)
        return (torch.linalg.norm(queries - center, dim=1) - radius).reshape(-1, 1)

    def _get_domain_bounds(self) -> torch.Tensor:
        r = self.radius
        c = self.center
        return torch.stack([c - r, c + r])


class RectangleSDF(SDFBase):
    """SDF for a 2D rectangle with center and extents."""

    def __init__(self, center, extents):
        super().__init__(geometric_dim=2)
        self.center = torch.nn.Parameter(torch.as_tensor(center, dtype=torch.float32))
        self.extents = torch.nn.Parameter(torch.as_tensor(extents, dtype=torch.float32))

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        center = self.center.to(device=queries.device, dtype=queries.dtype)
        half = self.extents.to(device=queries.device, dtype=queries.dtype) / 2.0
        q = torch.abs(queries - center) - half
        outside = torch.linalg.norm(torch.clamp(q, min=0.0), dim=1)
        inside = torch.minimum(torch.maximum(q[:, 0], q[:, 1]), torch.tensor(0.0))
        return (outside + inside).reshape(-1, 1)

    def _get_domain_bounds(self) -> torch.Tensor:
        c = self.center
        e = self.extents
        return torch.stack([c - e / 2, c + e / 2])


class LineSDF(SDFBase):
    """SDF for a 2D line (infinite line defined by normal vector and point)."""

    def __init__(self, normal, point):
        super().__init__(geometric_dim=2)
        self.normal = torch.nn.Parameter(torch.as_tensor(normal, dtype=torch.float32))
        self.point = torch.nn.Parameter(torch.as_tensor(point, dtype=torch.float32))

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        n = self.normal.to(device=queries.device, dtype=queries.dtype)
        p = self.point.to(device=queries.device, dtype=queries.dtype)
        n_normalized = n / torch.linalg.norm(n)
        return torch.matmul(queries - p, n_normalized).reshape(-1, 1)

    def _get_domain_bounds(self) -> torch.Tensor:
        return torch.tensor([[-1.0, -1.0], [1.0, 1.0]])


class RoundedRectangleSDF(SDFBase):
    """SDF for a 2D rectangle with rounded corners."""

    def __init__(self, center, extents, radius):
        super().__init__(geometric_dim=2)
        self.center = torch.nn.Parameter(torch.as_tensor(center, dtype=torch.float32))
        self.extents = torch.nn.Parameter(torch.as_tensor(extents, dtype=torch.float32))

        # Support single radius or tuple of 4 radii (one per quadrant)
        if isinstance(radius, (tuple, list)):
            if len(radius) == 1:
                self.radii = torch.nn.Parameter(
                    torch.as_tensor([radius[0]] * 4, dtype=torch.float32)
                )
            elif len(radius) == 4:
                self.radii = torch.nn.Parameter(
                    torch.as_tensor(radius, dtype=torch.float32)
                )
            else:
                raise ValueError("radius must be a single value or a tuple of 4 values")
        else:
            self.radii = torch.nn.Parameter(
                torch.as_tensor([radius] * 4, dtype=torch.float32)
            )

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        center = self.center.to(device=queries.device, dtype=queries.dtype)
        extents = self.extents.to(device=queries.device, dtype=queries.dtype)
        radii = self.radii.to(device=queries.device, dtype=queries.dtype)

        q = torch.abs(queries - center) - extents / 2.0

        # Select appropriate radius based on quadrant
        x, y = q[:, 0], q[:, 1]
        zeros = torch.zeros_like(x)
        r0 = torch.where((x > 0) & (y > 0), radii[0], zeros)
        r1 = torch.where((x > 0) & (y <= 0), radii[1], zeros)
        r2 = torch.where((x <= 0) & (y <= 0), radii[2], zeros)
        r3 = torch.where((x <= 0) & (y > 0), radii[3], zeros)
        r = r0 + r1 + r2 + r3

        q = q + r.unsqueeze(1)
        outside = torch.linalg.norm(torch.clamp(q, min=0.0), dim=1)
        inside = torch.minimum(torch.maximum(q[:, 0], q[:, 1]), torch.tensor(0.0))
        return (outside + inside - r).reshape(-1, 1)

    def _get_domain_bounds(self) -> torch.Tensor:
        c = self.center
        e = self.extents
        return torch.stack([c - e / 2, c + e / 2])


class EquilateralTriangleSDF(SDFBase):
    """SDF for a regular equilateral triangle."""

    def __init__(self, size=1.0):
        super().__init__(geometric_dim=2)
        self.size = torch.nn.Parameter(
            torch.as_tensor(size, dtype=torch.float32).reshape(())
        )

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        r_val = self.size.item() if isinstance(self.size, torch.Tensor) else self.size
        r = torch.tensor(r_val, device=queries.device, dtype=queries.dtype)
        k = torch.tensor(np.sqrt(3.0), device=queries.device, dtype=queries.dtype)
        p = queries.clone()

        # Apply abs and offset
        p[:, 0] = torch.abs(p[:, 0]) - r
        p[:, 1] = p[:, 1] + r / k

        # Conditional transformation
        w = (p[:, 0] + k * p[:, 1]) > 0
        q = torch.stack([p[:, 0] - k * p[:, 1], -k * p[:, 0] - p[:, 1]], dim=1) / 2.0
        p = torch.where(w.unsqueeze(1), q, p)

        # Clamp x
        p[:, 0] = p[:, 0] - torch.clamp(p[:, 0], min=-2.0 * r, max=0.0)

        # Compute signed distance
        return (-torch.linalg.norm(p, dim=1) * torch.sign(p[:, 1])).reshape(-1, 1)

    def _get_domain_bounds(self) -> torch.Tensor:
        s = self.size
        return torch.tensor([[-s, -s], [s, s]])


class HexagonSDF(SDFBase):
    """SDF for a regular hexagon."""

    def __init__(self, size=1.0):
        super().__init__(geometric_dim=2)
        self.size = torch.nn.Parameter(
            torch.as_tensor(size, dtype=torch.float32).reshape(())
        )

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        r = self.size
        k = torch.tensor(
            [-np.sqrt(3.0) / 2.0, 0.5, np.tan(np.pi / 6.0)],
            device=queries.device,
            dtype=queries.dtype,
        )
        p = torch.abs(queries)
        p = p - 2.0 * torch.clamp(torch.sum(p * k[:2], dim=1, keepdim=True), max=0) * k[
            :2
        ].unsqueeze(0)
        p = p - torch.stack(
            [
                torch.clamp(p[:, 0], min=-k[2] * r, max=k[2] * r),
                torch.ones_like(p[:, 0]) * r,
            ],
            dim=1,
        )
        return (torch.linalg.norm(p, dim=1) * torch.sign(p[:, 1])).reshape(-1, 1)

    def _get_domain_bounds(self) -> torch.Tensor:
        s = self.size
        return torch.tensor([[-s, -s], [s, s]])


class PolygonSDF(SDFBase):
    """SDF for a general convex polygon defined by vertices."""

    def __init__(self, vertices):
        super().__init__(geometric_dim=2)
        if len(vertices) < 3:
            raise ValueError("Polygon needs at least 3 vertices")
        self.vertices = torch.nn.Parameter(
            torch.as_tensor(vertices, dtype=torch.float32)
        )

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        """
        Signed distance for convex polygon using winding number method.
        Returns negative values for points inside the polygon.
        """
        points = self.vertices.to(device=queries.device, dtype=queries.dtype)
        p = queries
        n = len(points)

        # Initialize distance squared to distance from first vertex
        d = torch.sum((p - points[0]) ** 2, dim=1)

        # Initialize sign (positive = outside)
        s = torch.ones(len(p), device=p.device, dtype=p.dtype)

        for i in range(n):
            j = (i + n - 1) % n
            vi = points[i]
            vj = points[j]
            e = vj - vi
            w = p - vi

            # Compute distance to edge segment
            # Project w onto e, clamp to [0, 1], then compute perpendicular distance
            e_len_sq = torch.sum(e**2)
            if e_len_sq > 1e-12:
                t = torch.clamp(torch.sum(w * e, dim=1) / e_len_sq, 0, 1)
                b = w - t.unsqueeze(1) * e
                d = torch.minimum(d, torch.sum(b**2, dim=1))

            # Winding number computation using cross products
            # Check conditions: (p.y >= vi.y), (p.y < vj.y), (e.x * w.y > e.y * w.x)
            c1 = p[:, 1] >= vi[1]
            c2 = p[:, 1] < vj[1]
            c3 = e[0] * w[:, 1] > e[1] * w[:, 0]

            # Flip sign if point crosses edge (all conditions true or all false)
            all_c = c1 & c2 & c3
            all_not_c = (~c1) & (~c2) & (~c3)
            condition = all_c | all_not_c
            s = torch.where(condition, -s, s)

        return (s * torch.sqrt(d)).reshape(-1, 1)

    def _get_domain_bounds(self) -> torch.Tensor:
        v = self.vertices.detach()
        return torch.stack([v.min(dim=0).values, v.max(dim=0).values])


class SlabSDF(SDFBase):
    """SDF for a slab (clipping box defined by axis-aligned planes)."""

    def __init__(self, x0=None, y0=None, z0=None, x1=None, y1=None, z1=None):
        super().__init__()
        # Bounds for each axis
        self.bounds = {}
        if x0 is not None:
            self.bounds["x0"] = (
                float(x0) if not isinstance(x0, torch.nn.Parameter) else x0
            )
        if x1 is not None:
            self.bounds["x1"] = (
                float(x1) if not isinstance(x1, torch.nn.Parameter) else x1
            )
        if y0 is not None:
            self.bounds["y0"] = (
                float(y0) if not isinstance(y0, torch.nn.Parameter) else y0
            )
        if y1 is not None:
            self.bounds["y1"] = (
                float(y1) if not isinstance(y1, torch.nn.Parameter) else y1
            )
        if z0 is not None:
            self.bounds["z0"] = (
                float(z0) if not isinstance(z0, torch.nn.Parameter) else z0
            )
        if z1 is not None:
            self.bounds["z1"] = (
                float(z1) if not isinstance(z1, torch.nn.Parameter) else z1
            )

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        dist = torch.full(
            (len(queries),), -float("inf"), device=queries.device, dtype=queries.dtype
        )

        # For each plane constraint
        for axis, side, bound_key, direction in [
            ("x", 0, "x0", 1),
            ("x", 1, "x1", -1),
            ("y", 0, "y0", 1),
            ("y", 1, "y1", -1),
            ("z", 0, "z0", 1),
            ("z", 1, "z1", -1),
        ]:
            bound = self.bounds.get(bound_key)
            if bound is None:
                continue

            idx = {"x": 0, "y": 1, "z": 2}[axis]
            if side == 0:
                plane_dist = bound - queries[:, idx]
                dist = torch.maximum(dist, plane_dist)
            else:
                plane_dist = queries[:, idx] - bound
                dist = torch.maximum(dist, plane_dist)

        return dist.reshape(-1, 1)

    def _get_domain_bounds(self) -> torch.Tensor:
        return torch.tensor([[-1e9, -1e9, -1e9], [1e9, 1e9, 1e9]])


class CappedCylinderSDF(SDFBase):
    """SDF for a finite cylinder with exact end caps."""

    def __init__(self, point_a, point_b, radius):
        super().__init__()
        a = torch.as_tensor(point_a, dtype=torch.float32)
        b = torch.as_tensor(point_b, dtype=torch.float32)
        r = torch.as_tensor(radius, dtype=torch.float32)
        self.point_a = torch.nn.Parameter(a)
        self.point_b = torch.nn.Parameter(b)
        self.radius = torch.nn.Parameter(r.reshape(()))

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        a = self.point_a.to(device=queries.device, dtype=queries.dtype)
        b = self.point_b.to(device=queries.device, dtype=queries.dtype)
        r = self.radius.to(device=queries.device, dtype=queries.dtype)

        ba = b - a
        ba_length_sq = torch.sum(ba**2)
        pa = queries - a

        # Project onto cylinder axis, clamp to segment
        paba = torch.sum(pa * ba, dim=1)
        h = torch.clamp(paba / ba_length_sq, 0, 1)

        # Distance to cylinder side
        pa_ba_diff = pa * ba_length_sq - h.unsqueeze(1) * ba
        x = torch.sqrt(torch.sum(pa_ba_diff**2, dim=1)) - r * ba_length_sq

        # Distance to end caps
        y = torch.abs(paba) - ba_length_sq * 0.5

        x2 = x**2
        y2 = y**2 * ba_length_sq

        # Mix inside/outside properly
        d = torch.where(
            (x < 0) & (y < 0),
            -torch.minimum(x2, y2),
            torch.where(x > 0, x2, torch.tensor(0.0))
            + torch.where(y > 0, y2, torch.tensor(0.0)),
        )

        # Fix sign
        d = torch.sign(d + torch.tensor(1e-9)) * torch.sqrt(torch.abs(d))

        return (d / ba_length_sq).reshape(-1, 1)

    def _get_domain_bounds(self) -> torch.Tensor:
        a = self.point_a.detach()
        b = self.point_b.detach()
        r = self.radius.detach()
        lower = torch.minimum(a, b) - r
        upper = torch.maximum(a, b) + r
        return torch.stack([lower, upper])


class RoundedCylinderSDF(SDFBase):
    """SDF for a cylinder with smooth rounded transitions."""

    def __init__(self, ra, rb, h):
        super().__init__()
        self.ra = torch.nn.Parameter(torch.as_tensor(ra, dtype=torch.float32))
        self.rb = torch.nn.Parameter(torch.as_tensor(rb, dtype=torch.float32))
        self.h = torch.nn.Parameter(torch.as_tensor(h, dtype=torch.float32))

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        ra = self.ra.to(device=queries.device, dtype=queries.dtype)
        rb = self.rb.to(device=queries.device, dtype=queries.dtype)
        h = self.h.to(device=queries.device, dtype=queries.dtype)

        d = torch.stack(
            [
                torch.sqrt(queries[:, 0] ** 2 + queries[:, 1] ** 2) - ra + rb,
                torch.abs(queries[:, 2]) - h / 2 + rb,
            ],
            dim=1,
        )

        outside = torch.linalg.norm(torch.clamp(d, min=0.0), dim=1)
        inside = torch.minimum(torch.maximum(d[:, 0], d[:, 1]), torch.tensor(0.0))
        return (outside + inside - rb).reshape(-1, 1)

    def _get_domain_bounds(self) -> torch.Tensor:
        r = max(self.ra.detach().item(), self.rb.detach().item())
        h = self.h.detach().item()
        return torch.tensor([[-r, -r, -h / 2], [r, r, h / 2]])


class CappedConeSDF(SDFBase):
    """SDF for a finite cone with exact end caps."""

    def __init__(self, point_a, point_b, ra, rb):
        super().__init__()
        self.point_a = torch.nn.Parameter(torch.as_tensor(point_a, dtype=torch.float32))
        self.point_b = torch.nn.Parameter(torch.as_tensor(point_b, dtype=torch.float32))
        self.ra = torch.nn.Parameter(torch.as_tensor(ra, dtype=torch.float32))
        self.rb = torch.nn.Parameter(torch.as_tensor(rb, dtype=torch.float32))

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        a = self.point_a.to(device=queries.device, dtype=queries.dtype)
        b = self.point_b.to(device=queries.device, dtype=queries.dtype)
        ra = self.ra.to(device=queries.device, dtype=queries.dtype)
        rb = self.rb.to(device=queries.device, dtype=queries.dtype)

        rba = rb - ra
        baba = torch.sum((b - a) ** 2)
        p = queries
        papa = torch.sum((p - a) ** 2, dim=1)
        paba = torch.sum((p - a) * (b - a), dim=1) / baba

        # Distance to cone side
        q = torch.sqrt(torch.clamp(papa - paba * paba * baba, min=0.0))
        ca = torch.maximum(torch.tensor(0.0), q - torch.where(paba < 0.5, ra, rb))
        cay = torch.abs(paba - 0.5) - 0.5

        cax = ca
        cbx = (
            q
            - ra
            - (torch.clamp((rba * (q - ra) + paba * baba) / (rba * rba + baba), 0, 1))
            * rba
        )
        cby = paba - torch.clamp(
            (rba * (q - ra) + paba * baba) / (rba * rba + baba), 0, 1
        )

        s = torch.where((cbx < 0) & (cay < 0), -1, 1)

        a_dist = cax**2 + cay**2 * baba
        b_dist = cbx**2 + cby**2 * baba

        # Determine which distance to use
        mask = (ca < 0) & (cay < 0)
        d2 = torch.where(mask, torch.tensor(0.0), torch.minimum(a_dist, b_dist))

        d = torch.sign(d2) * torch.sqrt(torch.abs(d2)) * s
        return d.reshape(-1, 1)

    def _get_domain_bounds(self) -> torch.Tensor:
        r = max(self.ra.detach().item(), self.rb.detach().item())
        h = torch.linalg.norm(self.point_b.detach() - self.point_a.detach())
        return torch.tensor([[-r, -r, 0], [r, r, h]])


class RoundedConeSDF(SDFBase):
    """SDF for a cone with smooth rounded transitions."""

    def __init__(self, r1, r2, h):
        super().__init__()
        self.r1 = torch.nn.Parameter(torch.as_tensor(r1, dtype=torch.float32))
        self.r2 = torch.nn.Parameter(torch.as_tensor(r2, dtype=torch.float32))
        self.h = torch.nn.Parameter(torch.as_tensor(h, dtype=torch.float32))

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        r1 = self.r1.to(device=queries.device, dtype=queries.dtype)
        r2 = self.r2.to(device=queries.device, dtype=queries.dtype)
        h = self.h.to(device=queries.device, dtype=queries.dtype)

        q = torch.stack(
            [torch.sqrt(queries[:, 0] ** 2 + queries[:, 1] ** 2), queries[:, 2]], dim=1
        )
        b = (r1 - r2) / h
        a = torch.sqrt(1 - b * b)
        k = torch.sum(q * torch.stack([-b, a]), dim=1)

        c1 = torch.linalg.norm(q, dim=1) - r1
        c2 = torch.linalg.norm(q - torch.stack([torch.tensor(0.0), h]), dim=1) - r2
        c3 = torch.sum(q * torch.stack([a, b]), dim=1) - r1

        return torch.where(k < 0, c1, torch.where(k > a * h, c2, c3)).reshape(-1, 1)

    def _get_domain_bounds(self) -> torch.Tensor:
        r = max(self.r1.detach().item(), self.r2.detach().item())
        h = self.h.detach().item()
        return torch.tensor([[-r, -r, 0], [r, r, h]])


class TetrahedronSDF(SDFBase):
    """SDF for a regular tetrahedron."""

    def __init__(self, r=1.0):
        super().__init__()
        self.r = torch.nn.Parameter(torch.as_tensor(r, dtype=torch.float32))

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        r = self.r.to(device=queries.device, dtype=queries.dtype)
        result = (
            torch.maximum(
                torch.abs(queries[:, 0] + queries[:, 1]) - queries[:, 2],
                torch.abs(queries[:, 0] - queries[:, 1]) + queries[:, 2],
            )
            - r
        )
        return result.reshape(-1, 1) / torch.sqrt(torch.tensor(3.0))

    def _get_domain_bounds(self) -> torch.Tensor:
        r = self.r.item()
        return torch.tensor([[-r, -r, -r], [r, r, r]])


class OctahedronSDF(SDFBase):
    """SDF for a regular octahedron."""

    def __init__(self, r=1.0):
        super().__init__()
        self.r = torch.nn.Parameter(torch.as_tensor(r, dtype=torch.float32))

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        r = self.r.to(device=queries.device, dtype=queries.dtype)
        return (
            (torch.sum(torch.abs(queries), dim=1) - r)
            * torch.tan(torch.tensor(np.pi / 6))
        ).reshape(-1, 1)

    def _get_domain_bounds(self) -> torch.Tensor:
        r = self.r.item()
        return torch.tensor([[-r, -r, -r], [r, r, r]])


class DodecahedronSDF(SDFBase):
    """SDF for a regular dodecahedron."""

    def __init__(self, r=1.0):
        super().__init__()
        self.r = torch.nn.Parameter(torch.as_tensor(r, dtype=torch.float32))

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        r = self.r.to(device=queries.device, dtype=queries.dtype)

        # Golden ratio
        golden = (1 + torch.sqrt(torch.tensor(5.0))) / 2
        # Normalized vertices of dodecahedron face normals
        vn = torch.tensor([golden, 1, 0], dtype=torch.float32)
        vn = vn / torch.linalg.norm(vn)

        p = torch.abs(queries / r)
        a = torch.sum(p * vn, dim=1)
        b = torch.sum(
            p * torch.tensor([vn[1], vn[2], vn[0]], dtype=torch.float32), dim=1
        )
        c = torch.sum(
            p * torch.tensor([vn[2], vn[0], vn[1]], dtype=torch.float32), dim=1
        )

        q = (torch.maximum(torch.maximum(a, b), c) - vn[0]) * r
        return q.reshape(-1, 1)

    def _get_domain_bounds(self) -> torch.Tensor:
        r = self.r.item()
        return torch.tensor([[-r, -r, -r], [r, r, r]])


class IcosahedronSDF(SDFBase):
    """SDF for a regular icosahedron."""

    def __init__(self, r=1.0):
        super().__init__()
        self.r = torch.nn.Parameter(torch.as_tensor(r, dtype=torch.float32))

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        r = self.r.to(device=queries.device, dtype=queries.dtype)
        r_scaled = r * torch.tensor(0.8506507174597755)

        # Normalized icosahedron vertices
        vn_raw = torch.tensor(
            [(torch.sqrt(torch.tensor(5.0)) + 3) / 2, 1, 0], dtype=torch.float32
        )
        vn = vn_raw / torch.linalg.norm(vn_raw)

        w = torch.sqrt(torch.tensor(3.0)) / 3

        p = torch.abs(queries / r_scaled)
        a = torch.sum(p * vn, dim=1)

        vn2 = torch.tensor([vn[1], vn[2], vn[0]], dtype=torch.float32)
        b = torch.sum(p * vn2, dim=1)

        vn3 = torch.tensor([vn[2], vn[0], vn[1]], dtype=torch.float32)
        c = torch.sum(p * vn3, dim=1)

        d = torch.sum(p * torch.tensor([w, w, w]), dim=1) - vn[0]

        return (
            torch.maximum(torch.maximum(torch.maximum(a, b), c) - vn[0], d).reshape(
                -1, 1
            )
            * r_scaled
        )

    def _get_domain_bounds(self) -> torch.Tensor:
        r = self.r.item() * 0.8506507174597755
        return torch.tensor([[-r, -r, -r], [r, r, r]])
