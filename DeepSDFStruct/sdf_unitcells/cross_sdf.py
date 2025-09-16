from DeepSDFStruct.SDF import SDFBase
import torch


class CrossMsSDF(SDFBase):
    def __init__(self, radius):
        self.r = radius

    def SDF(self, xyz):
        output = torch.linalg.norm(xyz, axis=1, ord=torch.inf)

        # add x cylinder
        cylinder = torch.sqrt(xyz[:, 1] ** 2 + xyz[:, 2] ** 2) - self.r
        output = torch.minimum(output, cylinder)
        # add y cylinder
        cylinder = torch.sqrt(xyz[:, 0] ** 2 + xyz[:, 2] ** 2) - self.r
        output = torch.minimum(output, cylinder)
        # add z cylinder
        cylinder = torch.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2) - self.r
        output = torch.minimum(output, cylinder)

        return output.reshape(-1, 1)
