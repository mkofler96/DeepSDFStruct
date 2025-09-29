import torch
import splinepy as sp
import numpy as np


def torch_spline_1D(knot_vectors, control_points, degrees, queries: torch.tensor):
    knot_vector = knot_vectors[0]
    degree = degrees[0]
    queries_1D = queries[:, 0]
    assert (len(knot_vector) - degree - 1) == len(control_points)

    x = queries_1D

    n_basis = len(knot_vector) - degree - 1
    t_clamped = torch.clamp(x, knot_vector[degree], knot_vector[-degree - 1])
    k = torch.searchsorted(knot_vector, t_clamped, right=True) - 1
    k = torch.clamp(k, min=degree, max=n_basis - 1).view(-1)

    # For each t, pick degree+1 control points for that span
    i = k.view(-1, 1) + torch.arange(-degree, 1, device=queries_1D.device).view(
        1, -1
    )  # (n_queries, degree+1)
    cp_local = control_points[i]  # (n_queries, degree+1)

    # b[j] = N_{j,p}(t) for local control points
    d = cp_local.clone()

    # vectorized deboor from https://en.wikipedia.org/wiki/De_Boor%27s_algorithm
    for r in range(1, degree + 1):
        j_idx = torch.arange(degree, r - 1, -1, device=knot_vector.device)

        idx_j_plus_k = j_idx.unsqueeze(0) + k.unsqueeze(1)
        idx_j_plus_k_minus_p = idx_j_plus_k - degree
        idx_j_plus_k_plus_1_minus_r = idx_j_plus_k + 1 - r

        jkmp = knot_vector[idx_j_plus_k_minus_p]
        j1kmr = knot_vector[idx_j_plus_k_plus_1_minus_r]

        alpha = ((x.unsqueeze(-1) - jkmp) / (j1kmr - jkmp)).unsqueeze(-1)
        d[:, j_idx, :] = (1.0 - alpha) * d[:, j_idx - 1, :] + alpha * d[:, j_idx, :]

    y = d[:, degree]
    return y


def bspline_basis(t, p, queries: torch.tensor):
    x = queries

    n_basis = len(t) - p - 1
    t_clamped = torch.clamp(x, t[p], t[-p - 1])
    k = torch.searchsorted(t, t_clamped, right=True) - 1
    k = torch.clamp(k, min=p, max=n_basis - 1).view(-1)

    # For each t, pick degree+1 control points for that span
    i = k.view(-1, 1) + torch.arange(-p, 1, device=queries.device).view(
        1, -1
    )  # (n_queries, degree+1)

    # b[j] = N_{j,p}(t) for local control points
    d = (
        torch.eye(p + 1, device=queries.device, dtype=queries.dtype)
        .unsqueeze(0)
        .repeat(queries.shape[0], 1, 1)
    )

    # vectorized deboor from https://en.wikipedia.org/wiki/De_Boor%27s_algorithm
    for r in range(1, p + 1):
        j_idx = torch.arange(p, r - 1, -1, device=t.device)

        idx_j_plus_k = j_idx.unsqueeze(0) + k.unsqueeze(1)
        idx_j_plus_k_minus_p = idx_j_plus_k - p
        idx_j_plus_k_plus_1_minus_r = idx_j_plus_k + 1 - r

        jkmp = t[idx_j_plus_k_minus_p]
        j1kmr = t[idx_j_plus_k_plus_1_minus_r]

        alpha = ((x.unsqueeze(-1) - jkmp) / (j1kmr - jkmp)).unsqueeze(-1)
        d[:, j_idx, :] = (1.0 - alpha) * d[:, j_idx - 1, :] + alpha * d[:, j_idx, :]

    # Result
    y = d[:, p, :]
    return i, y


def torch_spline_3D(knot_vectors, control_points, degrees, queries):
    """
    Evaluate a 3D B-spline volume at query points.

    Args:
        tx, ty, tz: knot vectors for x, y, z
        px, py, pz: degrees
        cp: control points, shape (nx, ny, nz, d)
        qx, qy, qz: query coordinates, shape (N,)

    Returns:
        y: evaluated spline, shape (N, d)
    """
    tx, ty, tz = knot_vectors
    px, py, pz = degrees
    qx = queries[:, 0]
    qy = queries[:, 1]
    qz = queries[:, 2]
    # Basis functions along each axis
    ix, bx = bspline_basis(tx, px, qx)  # (N, px+1), (N, px+1)
    iy, by = bspline_basis(ty, py, qy)
    iz, bz = bspline_basis(tz, pz, qz)
    torch.testing.assert_close(bx.sum(dim=1), torch.ones_like(qx))
    torch.testing.assert_close(by.sum(dim=1), torch.ones_like(qy))
    torch.testing.assert_close(bz.sum(dim=1), torch.ones_like(qz))
    nx = len(tx) - px - 1
    ny = len(ty) - py - 1
    nz = len(tz) - pz - 1

    assert nx * ny * nz == control_points.shape[0]

    bx_ = bx[:, :, None, None]  # (N, px+1, 1, 1)
    by_ = by[:, None, :, None]  # (N, 1, py+1, 1)
    bz_ = bz[:, None, None, :]  # (N, 1, 1, pz+1)

    # Compute outer product of weights: (N, px+1, py+1, pz+1)
    weights = bx_ * by_ * bz_

    ix_ = ix[:, :, None, None]  # (N, px+1, 1, 1)
    iy_ = iy[:, None, :, None]  # (N, 1, py+1, 1)
    iz_ = iz[:, None, None, :]  # (N, 1, 1, pz+1)

    flat_idx = ix_ + nx * (iy_ + ny * iz_)
    cp_selected = control_points[flat_idx]
    y = (weights[:, :, :, :, None] * cp_selected).sum(dim=(1, 2, 3))  # (N,)

    return y


class TorchSpline(torch.nn.Module):
    """
    V2: uses custom torch spline as spline backend
    """

    spline: sp.BSpline

    def __init__(self, spline: sp.BSpline, device="cpu", dtype=torch.float32):
        # TODO register control points etc. as parameter
        super().__init__()
        self.device = device
        self.dtype = dtype  # dtype and device could be getter and setter
        self.spline = spline
        self.control_points = torch.nn.Parameter(
            torch.tensor(spline.control_points, dtype=dtype, device=device)
        )
        self.knot_vectors = [
            torch.tensor(knot, dtype=dtype, device=device)
            for knot in spline.knot_vectors
        ]
        self.degrees = torch.tensor(self.spline.degrees, dtype=int, device=device)

        match len(spline.degrees):
            case 1:
                self.spline_fun = torch_spline_1D
            case 2:
                raise NotImplementedError("2D spline not implemented yet")
            case 3:
                self.spline_fun = torch_spline_3D

    def forward(self, queries: torch.Tensor):
        # spline fun takes the following arguments:
        #     knot_vectors, control_points, degrees, queries: Any
        return self.spline_fun(
            self.knot_vectors, self.control_points, self.degrees, queries
        )


def generate_bbox_spline(bounds):
    """
    Takes bounding box and generates a spline box.

    Parameters
    ----------
    bounds : (2, 3) array-like
        [[xmin, ymin, zmin],
         [xmax, ymax, zmax]]

    Returns
    -------
    spline : splinepy.BSpline
        BSpline representing the bounding box.
    """
    bounds = np.asarray(bounds)
    mins, maxs = bounds[0], bounds[1]

    knots = [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]

    nx, ny, nz = 2, 2, 2

    # Create regular grid of control points
    xs = np.linspace(mins[0], maxs[0], nx)
    ys = np.linspace(mins[1], maxs[1], ny)
    zs = np.linspace(mins[2], maxs[2], nz)

    # Generate full grid
    control_points = np.array([[x, y, z] for z in zs for y in ys for x in xs])

    spline = sp.BSpline([1, 1, 1], knots, control_points)
    return spline
