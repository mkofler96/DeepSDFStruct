import torch
import splinepy
from DeepSDFStruct.torch_spline import TorchSpline
import numpy as np


def test_torchspline_evaluation(np_rng, device="cpu"):
    """Check that TorchSpline evaluation matches splinepy evaluation"""

    # Random Bezier 2D -> 3D
    control_points = np_rng.random((16, 3))
    bezier = splinepy.Bezier(degrees=[3, 3], control_points=control_points)

    spline_module = TorchSpline(bezier, device=device)

    queries = torch.tensor(np_rng.random((5, 2)), dtype=torch.float32, device=device)

    # Torch output
    output_torch = spline_module(queries).detach().cpu().numpy()

    # Numpy output
    output_np = bezier.evaluate(queries.detach().cpu().numpy())

    assert np.allclose(output_torch, output_np, atol=1e-6)


def test_torchspline_derivative(np_rng, device="cpu", eps=1e-3):
    """Check that TorchSpline forward derivatives w.r.t input are correct via finite differences"""

    # Random Bezier 2D -> 3D
    control_points = np_rng.random((16, 3))
    bezier = splinepy.Bezier(degrees=[3, 3], control_points=control_points)
    spline_module = TorchSpline(bezier, device=device)

    queries = torch.tensor(
        np_rng.random((3, 2)), dtype=torch.float32, device=device, requires_grad=True
    )

    # Torch output
    output = spline_module(queries)

    # Finite differences approximation
    fd_jacobian = np.zeros((queries.shape[0], output.shape[1], queries.shape[1]))
    for i in range(queries.shape[0]):
        for j in range(queries.shape[1]):
            q_plus = queries.clone()
            q_plus[i, j] += eps
            q_minus = queries.clone()
            q_minus[i, j] -= eps
            f_plus = spline_module(q_plus).detach().cpu().numpy()[i]
            f_minus = spline_module(q_minus).detach().cpu().numpy()[i]
            fd_jacobian[i, :, j] = (f_plus - f_minus) / (2 * eps)

    # Torch autograd jacobian
    jacobian_torch = []
    for i in range(output.shape[1]):
        grad = (
            torch.autograd.grad(
                output[:, i].sum(), queries, create_graph=False, retain_graph=True
            )[0]
            .detach()
            .cpu()
            .numpy()
        )
        jacobian_torch.append(grad)
    jacobian_torch = np.stack(jacobian_torch, axis=1)

    fd_flat = fd_jacobian.flatten()
    jac_flat = jacobian_torch.flatten()

    np.testing.assert_allclose(
        fd_jacobian,
        jacobian_torch,
        rtol=1e-5,
        atol=1e-4,
        err_msg="Finite difference Jacobian does not match autograd Jacobian.\n"
        f"First 5 entries of FD Jacobian:\n{fd_flat[:5]}\n"
        f"First 5 entries of Autograd Jacobian:\n{jac_flat[:5]}",
    )


def test_torchspline_jacobian(np_rng, device="cpu"):
    """Check that TorchSpline forward matches splinepy Jacobian"""

    # Random Bezier 2D -> 3D
    control_points = np_rng.random((16, 3))
    bezier = splinepy.Bezier(degrees=[3, 3], control_points=control_points)
    spline_module = TorchSpline(bezier, device=device)

    queries_np = np_rng.random((3, 2))
    queries_torch = torch.tensor(queries_np, dtype=torch.float32, device=device)

    # Torch output
    output_torch = spline_module(queries_torch).detach().cpu().numpy()

    # SplinePy jacobian
    jacobian_np = bezier.jacobian(queries_np)  # shape: [n_points, dim_out, dim_in]

    # Check by applying jacobian to small random delta
    delta = np_rng.random(queries_np.shape) * 1e-5
    f0 = output_torch
    f1 = (
        spline_module(
            queries_torch + torch.tensor(delta, dtype=torch.float32, device=device)
        )
        .detach()
        .cpu()
        .numpy()
    )
    f1_approx = f0 + np.einsum("pij,pj->pi", jacobian_np, delta)

    assert np.allclose(f1, f1_approx, atol=1e-5)


def test_control_point_derivative(np_rng, device="cpu"):
    """Test that d(output)/d(control_points) matches the shape function (basis)"""

    # Random Bezier 2D -> 3D
    control_points = np_rng.random((16, 3))
    bezier = splinepy.Bezier(degrees=[3, 3], control_points=control_points)
    spline_module = TorchSpline(bezier, device=device)

    # Random queries
    queries = torch.tensor(
        np_rng.random((5, 2)), dtype=torch.float32, device=device, requires_grad=True
    )

    # Forward pass
    output = spline_module(queries)  # [n_points, dim_out]

    # Compute derivative w.r.t control points via autograd
    basis_autograd_list = []

    for i in range(output.shape[1]):
        grad_cp = torch.autograd.grad(
            output[:, i].sum(), spline_module.control_points, retain_graph=True
        )[
            0
        ]  # [n_control_points, dim_out]
        # Only need the column corresponding to output i
        basis_autograd_list.append(grad_cp[:, i].detach().cpu().numpy())

    basis_autograd = np.stack(
        basis_autograd_list, axis=1
    )  # [n_control_points, dim_out]

    # Compute shape function (basis) matrix directly from splinepy
    queries_np = queries.detach().cpu().numpy()
    basis, supports = bezier.basis_and_support(queries_np)
    basis_matrix = splinepy.utils.data.make_matrix(
        basis, supports, spline_module.control_points.shape[0], as_array=True
    )

    # The basis_matrix shape: [n_queries, n_control_points], need to transpose to match autograd?
    # Each output dimension behaves the same for vector-valued spline
    basis_expected = np.repeat(
        basis_matrix[:, :, np.newaxis], output.shape[1], axis=2
    )  # [n_queries, n_control_points, dim_out]
    # Sum along queries to match autograd sum over queries
    basis_expected_sum = basis_expected.sum(axis=0)  # [n_control_points, dim_out]

    # Compare
    np.testing.assert_allclose(
        basis_autograd,
        basis_expected_sum,
        rtol=1e-5,
        atol=1e-6,
        err_msg=f"Derivative w.r.t control points does not match shape functions.\n"
        f"First 5 entries of autograd derivative:\n{basis_autograd[:5]}\n"
        f"First 5 entries of expected basis:\n{basis_expected_sum[:5]}",
    )


if __name__ == "__main__":
    np_rng = np.random.default_rng(0)
    test_torchspline_evaluation(np_rng)
    test_torchspline_derivative(np_rng)
    test_torchspline_jacobian(np_rng)
    test_control_point_derivative(np_rng)
