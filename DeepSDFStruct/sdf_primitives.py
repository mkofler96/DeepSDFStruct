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
