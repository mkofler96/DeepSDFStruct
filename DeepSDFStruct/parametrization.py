from abc import ABC, abstractmethod
import torch
import splinepy as sp
from DeepSDFStruct.torch_spline import TorchSpline


class _Parametrization(ABC):
    parameters: torch.Tensor

    def __init__(self, parameters: torch.Tensor, device):
        self.parameters = parameters
        self.device = device

    @abstractmethod
    def __call__(self, queries: torch.Tensor) -> torch.Tensor:
        pass


class Constant(_Parametrization):
    def __init__(self, value, device):
        if type(value) is not torch.Tensor:
            value = torch.tensor(value, device=device)
        super().__init__(parameters=value, device=device)

    def __call__(self, queries: torch.Tensor) -> torch.Tensor:
        N = queries.shape[0]
        return self.parameters.expand(N, -1)


class SplineParametrization(_Parametrization):
    def __init__(self, spline: sp.BSpline | sp.Bezier | sp.NURBS, device):
        self.torch_spline = TorchSpline(spline, device=device)
        super().__init__(parameters=self.torch_spline.control_points, device=device)

    def __call__(self, queries: torch.Tensor) -> torch.Tensor:
        return self.torch_spline(queries)
