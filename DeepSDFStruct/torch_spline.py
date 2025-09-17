import torch
import splinepy as sp


class TorchSplineFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, queries, control_points, spline):
        """
        queries: [n_points, dim_in] tensor
        control_points: [n_control_points, dim_out] tensor
        spline: splinepy object
        """
        # Save for backward
        ctx.save_for_backward(queries, control_points)
        ctx.spline = spline

        # Evaluate basis functions using splinepy (NumPy)
        basis, supports = spline.basis_and_support(queries.detach().cpu().numpy())
        basis_eval = sp.utils.data.make_matrix(
            basis, supports, control_points.shape[0], as_array=True
        )
        basis_eval_torch = torch.tensor(
            basis_eval, dtype=control_points.dtype, device=control_points.device
        )
        ctx.basis_eval = basis_eval_torch

        return basis_eval_torch @ control_points

    @staticmethod
    def backward(ctx, grad_output):
        queries, control_points = ctx.saved_tensors
        basis_eval = ctx.basis_eval

        # Gradient w.r.t control points
        grad_control_points = basis_eval.T @ grad_output

        # Gradient w.r.t queries (via splinepy jacobian)
        grad_queries = None
        try:
            jacobian = ctx.spline.jacobian(
                queries.detach().cpu().numpy()
            )  # [n_points, dim_out, dim_in]
            jacobian_torch = torch.tensor(
                jacobian, dtype=grad_output.dtype, device=grad_output.device
            )
            grad_queries = torch.einsum("pi,pij->pj", grad_output, jacobian_torch)
        except:
            grad_queries = None  # If jacobian not implemented

        return grad_queries, grad_control_points, None  # None for spline


# ---------------------------------------------------------
class TorchSpline(torch.nn.Module):
    def __init__(self, spline: sp.BSpline | sp.Bezier | sp.NURBS, device="cpu"):
        super().__init__()
        self.device = device
        self.spline = spline

        # Learnable control points
        self.control_points = torch.nn.Parameter(
            torch.tensor(spline.control_points, dtype=torch.float32, device=device)
        )

    def forward(self, queries: torch.Tensor):
        return TorchSplineFunction.apply(queries, self.control_points, self.spline)
