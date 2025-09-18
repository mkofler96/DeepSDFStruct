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
        ctx.spline = spline

        # Evaluate basis functions using splinepy (NumPy)
        basis, supports = spline.basis_and_support(
            queries.clone().detach().cpu().numpy()
        )
        basis_eval_np = sp.utils.data.make_matrix(
            basis, supports, control_points.shape[0], as_array=True
        )
        basis_eval = torch.tensor(
            basis_eval_np, device=control_points.device, dtype=control_points.dtype
        )
        # Save for backward
        ctx.save_for_backward(queries, basis_eval, control_points)
        # Save separately for forward-mode
        ctx.queries = queries
        ctx.basis_eval = basis_eval
        ctx.control_points = control_points
        return basis_eval @ control_points

    @staticmethod
    def backward(ctx, grad_output):
        queries, basis_eval, control_points = ctx.saved_tensors

        # Gradient w.r.t control points
        grad_control_points = basis_eval.T @ grad_output

        # Gradient w.r.t queries (via splinepy jacobian)
        grad_queries = None

        jacobian = ctx.spline.jacobian(
            queries.detach().cpu().numpy()
        )  # [n_points, dim_out, dim_in]
        jacobian_torch = torch.tensor(
            jacobian, dtype=grad_output.dtype, device=grad_output.device
        )
        grad_queries = torch.einsum("pi,pij->pj", grad_output, jacobian_torch)

        return grad_queries, grad_control_points, None  # None for spline

    @staticmethod
    def jvp(ctx, queries_tangent, control_points_tangent, spline_tangent):
        queries = ctx.queries
        basis_eval = ctx.basis_eval
        control_points = ctx.control_points

        # Tangent due to control_points
        cp_tangent = control_points_tangent
        if cp_tangent is None:
            cp_tangent = torch.zeros_like(control_points)

        out_tangent_cp = basis_eval @ cp_tangent

        jacobian = ctx.spline.jacobian(
            queries.detach().cpu().numpy()
        )  # [n_points, dim_out, dim_in]
        jacobian_torch = torch.tensor(
            jacobian, dtype=queries.dtype, device=queries.device
        )
        out_tangent_queries = torch.bmm(
            queries_tangent.unsqueeze(1), jacobian_torch.transpose(1, 2)
        ).squeeze(1)

        return out_tangent_cp + out_tangent_queries


# ---------------------------------------------------------
class TorchSpline(torch.nn.Module):
    def __init__(self, spline: sp.BSpline | sp.Bezier | sp.NURBS, device="cpu"):
        super().__init__()
        self.device = device
        self.spline = spline

        self.control_points = torch.tensor(
            spline.control_points, dtype=torch.float32, device=device
        )

    def forward(self, queries: torch.Tensor):
        return TorchSplineFunction.apply(queries, self.control_points, self.spline)
