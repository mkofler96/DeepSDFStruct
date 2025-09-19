from DeepSDFStruct.SDF import SDFBase
import torch


class SphereSDF(SDFBase):
    def __init__(self, center, radius):
        super().__init__()
        self.center = torch.tensor(center, dtype=torch.float32)
        self.r = radius

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        return (torch.linalg.norm(queries - self.center, dim=1) - self.r).reshape(-1, 1)

    def _get_domain_bounds(self) -> torch.Tensor:
        return torch.tensor([[-1, -1, -1], [1, 1, 1]], dtype=torch.float32)


class CylinderSDF(SDFBase):
    def __init__(self, point, axis, radius):
        super().__init__()
        self.point = torch.tensor(point, dtype=torch.float32)
        self.axis = axis
        self.r = radius

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        diff = queries - self.point
        if self.axis == "x":
            dist = torch.sqrt(diff[:, 1] ** 2 + diff[:, 2] ** 2)
        elif self.axis == "y":
            dist = torch.sqrt(diff[:, 0] ** 2 + diff[:, 2] ** 2)
        elif self.axis == "z":
            dist = torch.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'")
        return (dist - self.r).reshape(-1, 1)

    def _get_domain_bounds(self) -> torch.Tensor:
        return torch.tensor([[-1, -1, -1], [1, 1, 1]], dtype=torch.float32)


class TorusSDF(SDFBase):
    def __init__(self, center, R, r):
        super().__init__()
        self.center = torch.tensor(center, dtype=torch.float32)
        self.R = R
        self.r = r

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        p = queries - self.center
        q = torch.stack(
            [torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - self.R, p[:, 2]], dim=1
        )
        dist = torch.linalg.norm(q, dim=1) - self.r
        return dist.reshape(-1, 1)

    def _get_domain_bounds(self) -> torch.Tensor:
        return torch.tensor([[-1, -1, -1], [1, 1, 1]], dtype=torch.float32)


class PlaneSDF(SDFBase):
    def __init__(self, point, normal):
        super().__init__()
        self.point = torch.tensor(point, dtype=torch.float32)
        self.normal = torch.tensor(normal, dtype=torch.float32)
        self.normal = self.normal / torch.linalg.norm(self.normal)

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        return torch.matmul(queries - self.point, self.normal).reshape(-1, 1)

    def _get_domain_bounds(self) -> torch.Tensor:
        return torch.tensor([[-1, -1, -1], [1, 1, 1]], dtype=torch.float32)


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
        return torch.tensor([[-1, -1, -1], [1, 1, 1]], dtype=torch.float32)


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
        return torch.tensor([[-1, -1, -1], [1, 1, 1]], dtype=torch.float32)
