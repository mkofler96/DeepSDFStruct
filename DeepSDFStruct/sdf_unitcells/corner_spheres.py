from DeepSDFStruct.SDF import SDFBase
import torch


class CornerSpheresSDF(SDFBase):
    def __init__(self, radius, limit=1):
        self.r = radius
        self.limit = limit

    def SDF(self, xyz):
        output = torch.linalg.norm(xyz, axis=1, ord=torch.inf) - self.limit

        # substract corners
        corners = torch.array(torch.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(
            -1, 3
        )
        for corner in corners:
            sphere_like = torch.linalg.norm(xyz - corner, axis=1, ord=3) - self.r
            output = torch.maximum(output, -sphere_like)

        return output.reshape(-1, 1)
