import torch
import splinepy
from DeepSDFStruct.torch_spline import (
    # TorchSpline,
    generate_bbox_spline,
    torch_spline_1D,
    torch_spline_3D,
)
from DeepSDFStruct.torch_spline import TorchSpline
import numpy as np
import pytest


@pytest.fixture
def np_rng():
    return np.random.default_rng(0)


@pytest.fixture
def bspline():
    control_points = np.random.rand(64, 3)  # 4x4x4 Bezier -> 64 control points

    knot_vecs = [
        [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
    ]

    return splinepy.BSpline(
        degrees=[2, 2, 2], knot_vectors=knot_vecs, control_points=control_points
    )


def test_torchspline_evaluation(np_rng, bspline, device="cpu"):
    """Check that TorchSpline evaluation matches splinepy evaluation"""

    spline_module = TorchSpline(bspline, device=device)

    queries = torch.tensor(np_rng.random((5, 3)), dtype=torch.float32, device=device)

    # Torch output
    output_torch = spline_module(queries).detach().cpu().numpy()

    # Numpy output
    output_np = bspline.evaluate(queries.detach().cpu().numpy())

    assert np.allclose(output_torch, output_np, atol=1e-6)


def test_torchspline_derivative(np_rng, bspline, device="cpu", eps=1e-3):
    """Check that TorchSpline forward derivatives w.r.t input are correct via finite differences"""

    spline_module = TorchSpline(bspline, device=device)

    queries = torch.tensor(
        np_rng.random((5, 3)), dtype=torch.float32, device=device, requires_grad=True
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


def test_torchspline_autograd(np_rng, bspline, device="cpu"):

    torch_spline = TorchSpline(bspline, device=device)
    queries = torch.tensor(
        np_rng.random((5, 3)), dtype=torch.float64, device=device, requires_grad=True
    )
    control_points = torch.tensor(
        bspline.control_points, dtype=torch.float64, device=device, requires_grad=True
    )

    def func(q, cp):
        torch_spline.control_points = cp
        return torch_spline.forward(q)

    torch.autograd.gradcheck(func, (queries, control_points), check_forward_ad=False)

    queries_2 = torch.tensor(
        np_rng.random((5, 3)), dtype=torch.float64, device=device, requires_grad=True
    )
    control_points_2 = torch.tensor(
        bspline.control_points, dtype=torch.float64, device=device, requires_grad=True
    )

    torch.autograd.gradcheck(func, (queries_2, control_points_2), check_forward_ad=True)


def test_generate_bbox_spline():
    # Define bounding box
    bbox = np.array([[-1.0, -2.0, -3.0], [4.0, 5.0, 6.0]])

    spline = generate_bbox_spline(bbox)

    # Check that min/max corners match bbox
    cp = spline.control_points
    mins, maxs = cp.min(axis=0), cp.max(axis=0)
    np.testing.assert_allclose(mins, bbox[0])
    np.testing.assert_allclose(maxs, bbox[1])

    params = np.array(np.meshgrid([0, 1], [0, 1], [0, 1])).T.reshape(-1, 3)

    evals = spline.evaluate(params)

    expected = np.array(
        np.meshgrid(
            [bbox[0, 0], bbox[1, 0]], [bbox[0, 1], bbox[1, 1]], [bbox[0, 2], bbox[1, 2]]
        )
    ).T.reshape(-1, 3)

    # Order might differ, so compare sets
    assert set(map(tuple, np.round(evals, 8))) == set(map(tuple, np.round(expected, 8)))


def test_custom_torchspline_3D(bspline):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_queries = 1000

    queries = torch.rand((n_queries, 3), dtype=torch.float32, device=device)

    splinepy_eval = bspline.evaluate(queries.detach().cpu().numpy())

    splinepy_eval_as_tensor = torch.tensor(
        splinepy_eval, dtype=torch.float32, device=device
    )

    output_ip = torch_spline_3D(
        torch.tensor(bspline.knot_vectors, device=device, dtype=torch.float32),
        control_points=torch.tensor(
            bspline.control_points, device=device, dtype=torch.float32
        ),
        degrees=bspline.degrees,
        queries=queries,
    )

    diff = torch.linalg.norm(splinepy_eval_as_tensor - output_ip, axis=1)

    print("Max L2 difference:", diff.max())
    torch.testing.assert_close(
        diff.max(),
        torch.tensor(0, device=diff.device, dtype=diff.dtype),
        atol=1e-5,
        rtol=1e-5,
    )


def test_custom_torchspline_1D():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_queries = 1000

    queries = torch.rand((n_queries, 1), dtype=torch.float32, device=device)
    control_points = np.random.rand(4, 1)  # 4x4x4 Bezier -> 64 control points

    knot_vecs = [[0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]]
    bspline = splinepy.BSpline(
        degrees=[2], knot_vectors=knot_vecs, control_points=control_points
    )
    splinepy_eval = bspline.evaluate(queries.detach().cpu().numpy())

    splinepy_eval_as_tensor = torch.tensor(
        splinepy_eval, dtype=torch.float32, device=device
    )

    output_ip = torch_spline_1D(
        torch.tensor(knot_vecs, device=device, dtype=torch.float32),
        control_points=torch.tensor(control_points, device=device, dtype=torch.float32),
        degrees=[2],
        queries=queries,
    )

    diff = torch.linalg.norm(splinepy_eval_as_tensor - output_ip, axis=1)

    print("Max L2 difference:", diff.max())
    torch.testing.assert_close(
        diff.max(),
        torch.tensor(0, device=diff.device, dtype=diff.dtype),
        atol=1e-5,
        rtol=1e-5,
    )


if __name__ == "__main__":
    control_points = np.random.rand(64, 3)  # 4x4x4 Bezier -> 64 control points

    knot_vecs = [
        [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
    ]

    bspline_test = splinepy.BSpline(
        degrees=[2, 2, 2], knot_vectors=knot_vecs, control_points=control_points
    )
    test_custom_torchspline_3D(bspline_test)
    test_custom_torchspline_1D()
    test_generate_bbox_spline()
    np_rng = np.random.default_rng(0)
    test_torchspline_evaluation(np_rng, bspline_test)
    test_torchspline_derivative(np_rng, bspline_test)
    test_torchspline_autograd(np_rng, bspline_test)
