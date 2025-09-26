from abc import ABC, abstractmethod
import torch
import torch.nn
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

    @abstractmethod
    def set_param(self, parameter: torch.Tensor) -> torch.Tensor:
        pass


class Constant(_Parametrization):
    def __init__(self, value, device):
        if type(value) is not torch.Tensor:
            val_tensor = torch.tensor(value, device=device)
            value = torch.nn.Parameter(val_tensor)
        super().__init__(parameters=value, device=device)

    def __call__(self, queries: torch.Tensor) -> torch.Tensor:
        N = queries.shape[0]
        return self.parameters.expand(N, -1)

    def set_param(self, parameters: torch.Tensor):
        self.parameters = parameters


class SplineParametrization(_Parametrization):
    def __init__(self, spline: sp.BSpline | sp.Bezier | sp.NURBS, device):
        self.torch_spline = TorchSpline(spline, device=device)
        params = torch.nn.Parameter(self.torch_spline.control_points)
        super().__init__(parameters=params, device=device)

    def __call__(self, queries: torch.Tensor) -> torch.Tensor:
        return self.torch_spline(queries)

    def set_param(self, parameter: torch.Tensor):
        self.torch_spline.control_points = parameter
