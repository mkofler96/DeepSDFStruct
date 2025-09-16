import torch
import splinepy as sp


class TorchSpline(torch.nn.Module):
    def __init__(self, spline: sp.BSpline | sp.Bezier | sp.NURBS, device):
        super().__init__()
        # make control points learnable
        self.control_points = torch.nn.Parameter(
            torch.tensor(spline.control_points, dtype=torch.float32, device=device)
        )
        self.spline = spline
        self.device = device

    def forward(self, queries):
        # queries: [n_points, 3]
        # compute basis functions in numpy and convert to torch
        basis, supports = self.spline.basis_and_support(queries.detach().cpu().numpy())
        basis_eval = sp.utils.data.make_matrix(
            basis, supports, self.control_points.shape[0], as_array=True
        )
        basis_eval_torch = torch.tensor(
            basis_eval,
            device=self.control_points.device,
            dtype=self.control_points.dtype,
        )

        return basis_eval_torch @ self.control_points
