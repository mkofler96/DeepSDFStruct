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


# New 2D primitives: CircleSDF and RectangleSDF
class CircleSDF(SDFBase):
    """2D circle SDF (geometric_dim=2). Center and radius are torch parameters."""

    def __init__(self, center, radius):
        super().__init__(geometric_dim=2)
        c = torch.as_tensor(center, dtype=torch.float32)
        r = torch.as_tensor(radius, dtype=torch.float32)
        if c.numel() != 2:
            raise ValueError("center must be length-2 for CircleSDF")
        self.center = torch.nn.Parameter(c.reshape(2))
        self.radius = torch.nn.Parameter(r.reshape(()))

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        # queries expected shape (N,2)
        if queries.shape[1] == 3:
            queries = queries[:, :2]
        center = self.center.to(device=queries.device, dtype=queries.dtype)
        r = self.radius.to(device=queries.device, dtype=queries.dtype)
        return (torch.linalg.norm(queries - center, dim=1) - r).reshape(-1, 1)

    def _get_domain_bounds(self) -> torch.Tensor:
        return torch.tensor([[-1.0, -1.0], [1.0, 1.0]])


class RectangleSDF(SDFBase):
    def __init__(self, center, extents):
        """2D axis-aligned rectangle SDF. half_extents defines half-widths in x and y.
        Both center and half_extents are torch parameters.
        SDF computed using standard box SDF formula in 2D.
        +---------------+
        |     center    |
        |       x       | extents[1]
        |               |  (height)
        +---------------+
            extents[0]
            (width)
        """
        super().__init__(geometric_dim=2)
        c = torch.as_tensor(center, dtype=torch.float32)
        h = torch.as_tensor(extents, dtype=torch.float32)
        if c.numel() != 2 or h.numel() != 2:
            raise ValueError(
                "center and half_extents must be length-2 for RectangleSDF"
            )
        self.center = torch.nn.Parameter(c.reshape(2))
        self.extents = torch.nn.Parameter(h.reshape(2))

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        center = self.center.to(device=queries.device, dtype=queries.dtype)
        half = self.extents.to(device=queries.device, dtype=queries.dtype) / 2
        if queries.shape[1] == 3:
            queries = queries[:, :2]
        q = queries - center  # (N,2)
        d = torch.abs(q) - half  # (N,2)
        # outside distance
        zero = torch.tensor(0.0, device=queries.device, dtype=queries.dtype)
        d_clamped = torch.maximum(d, zero)
        outside_dist = torch.linalg.norm(d_clamped, dim=1)
        # inside distance (negative when inside)
        inside_dist = torch.minimum(torch.maximum(d[:, 0], d[:, 1]), zero)
        sdf = (outside_dist + inside_dist).reshape(-1, 1)
        return sdf

    def _get_domain_bounds(self) -> torch.Tensor:
        center = self.center.detach()
        half = self.extents.detach() / 2.0

        lower = center - half
        upper = center + half

        return torch.stack([lower, upper], dim=0)
