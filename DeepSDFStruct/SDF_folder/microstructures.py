import numpy as np


class CrossMsSDF:
    def __init__(self, radius):
        self.r = radius

    def SDF(self, xyz):
        output = np.linalg.norm(xyz, axis=1, ord=np.inf)

        # add x cylinder
        cylinder = np.sqrt(xyz[:, 1] ** 2 + xyz[:, 2] ** 2) - self.r
        output = np.minimum(output, cylinder)
        # add y cylinder
        cylinder = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 2] ** 2) - self.r
        output = np.minimum(output, cylinder)
        # add z cylinder
        cylinder = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2) - self.r
        output = np.minimum(output, cylinder)

        return output.reshape(-1, 1)


class CornerSpheresSDF:
    def __init__(self, radius, limit=1):
        self.r = radius
        self.limit = limit

    def SDF(self, xyz):
        output = np.linalg.norm(xyz, axis=1, ord=np.inf) - self.limit

        # substract corners
        corners = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)
        for corner in corners:
            sphere_like = np.linalg.norm(xyz - corner, axis=1, ord=3) - self.r
            output = np.maximum(output, -sphere_like)

        return output.reshape(-1, 1)
