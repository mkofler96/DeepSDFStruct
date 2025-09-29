import torch
import torch.nn as nn
import splinepy as sp
from DeepSDFStruct.torch_spline import TorchSpline


class Constant(nn.Module):
    def __init__(self, value, device=None, dtype=None):
        super().__init__()
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, device=device, dtype=dtype)
        self.param = nn.Parameter(value)

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        N = queries.shape[0]
        return self.param.expand(N, -1)

    def set_param(self, new_value: torch.Tensor):
        with torch.no_grad():
            self.param.copy_(new_value.to(self.param.device))


class SplineParametrization(nn.Module):
    def __init__(self, spline: sp.BSpline | sp.Bezier | sp.NURBS, device=None):
        super().__init__()
        self.torch_spline = TorchSpline(spline, device=device)

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        return self.torch_spline(queries)

    def set_param(self, new_value: torch.Tensor):
        with torch.no_grad():
            self.torch_spline.control_points.copy_(
                new_value.to(self.torch_spline.control_points.device)
            )
