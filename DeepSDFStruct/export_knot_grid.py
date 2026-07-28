import numpy as np
import pyvista as pv
import torch


def export_knot_grid_paramspace(spline, filename="knot_grid_paramspace.vtp"):
    """Export the tensor-product knot grid of a splinepy BSpline as a line
    network (PolyData) for ParaView.

    Parameters
    ----------
    spline : splinepy.BSpline
        The spline whose knot grid should be exported.
    filename : str
        Output file path (.vtp).
    """
    kvs = [np.asarray(kv, dtype=float) for kv in spline.knot_vectors]

    u_vals = np.unique(kvs[0])
    v_vals = np.unique(kvs[1])
    w_vals = np.unique(kvs[2])

    points = []
    lines = []
    point_dict = {}

    def get_point_id(p):
        key = tuple(np.round(p, 12))
        if key not in point_dict:
            point_dict[key] = len(points)
            points.append(p)
        return point_dict[key]

    for v in v_vals:
        for w in w_vals:
            ids = [get_point_id([u, v, w]) for u in u_vals]
            lines.append([len(ids)] + ids)

    for u in u_vals:
        for w in w_vals:
            ids = [get_point_id([u, v, w]) for v in v_vals]
            lines.append([len(ids)] + ids)

    for u in u_vals:
        for v in v_vals:
            ids = [get_point_id([u, v, w]) for w in w_vals]
            lines.append([len(ids)] + ids)

    points = np.array(points)
    lines = np.hstack(lines)

    grid = pv.PolyData(points)
    grid.lines = lines
    grid.save(filename)

    print(f"Knot grid exported to: {filename}")


def _greville_1d(U, p):
    n = len(U) - p - 1
    if n <= 0:
        raise ValueError(f"Invalid knot vector length {len(U)} for degree {p}")
    if p == 0:
        return 0.5 * (U[:n] + U[1 : n + 1])
    return np.array(
        [np.sum(U[i + 1 : i + p + 1]) / p for i in range(n)], dtype=float
    )


def export_control_lattice_paramspace(
    spline,
    filename="control_lattice_paramspace.vtp",
    locked_idx=None,
    order="F",
):
    """Export control-lattice Greville abscissae (parametric space) as a
    .vtp point cloud with id/i/j/k metadata. If locked_idx is given, adds
    a `locked` int32 point array (0 = free, 1 = locked) so ParaView can
    color free vs. locked control points.

    Parameters
    ----------
    spline : splinepy.BSpline
    filename : str
        Output file path (.vtp).
    locked_idx : array-like or torch.Tensor, optional
        Indices of locked control points. If None or empty, the `locked`
        array is omitted.
    order : str
        Flattening order for control point indices ('F' or 'C').
    """
    degrees = np.array(spline.degrees, dtype=int)
    kvs = [np.asarray(kv, dtype=float) for kv in spline.knot_vectors]

    g0 = _greville_1d(kvs[0], degrees[0])
    g1 = _greville_1d(kvs[1], degrees[1])
    g2 = _greville_1d(kvs[2], degrees[2])

    n0, n1, n2 = len(g0), len(g1), len(g2)
    N = n0 * n1 * n2
    ids = np.arange(N)

    I, J, K = np.unravel_index(ids, (n0, n1, n2), order=order)
    pts = np.column_stack([g0[I], g1[J], g2[K]])

    cloud = pv.PolyData(pts)
    cloud["id"] = ids
    cloud["i"] = I
    cloud["j"] = J
    cloud["k"] = K

    if locked_idx is not None:
        if torch.is_tensor(locked_idx):
            locked_idx = locked_idx.detach().to("cpu").numpy()
        locked_idx = np.asarray(locked_idx, dtype=int)
        if locked_idx.size > 0:
            if locked_idx.min() < 0 or locked_idx.max() >= N:
                raise IndexError(
                    f"locked_idx out of bounds. Valid range: [0, {N-1}]"
                )
            locked_flag = np.zeros((N,), dtype=np.int32)
            locked_flag[locked_idx] = 1
            cloud["locked"] = locked_flag

    cloud.save(filename)
    print(f"Control lattice exported to: {filename}")


def export_design_volume_paramspace(spline, filename="design_volume_paramspace.vts"):
    """Export the parametric design volume as a pyvista StructuredGrid
    (.vts). Points are placed at every unique knot intersection, so the
    volume shows the knot subdivision structure. Natural companion to
    `export_knot_grid_paramspace` (wireframe) and
    `export_control_lattice_paramspace` (Greville points) when you want
    a solid, colorable volume in ParaView.

    Parameters
    ----------
    spline : splinepy.BSpline
    filename : str or Path
        Output .vts file path.
    """
    kvs = [np.unique(np.asarray(kv, dtype=float)) for kv in spline.knot_vectors]
    u, v, w = kvs
    nu, nv, nw = len(u), len(v), len(w)
    N = nu * nv * nw

    U, V, W = np.meshgrid(u, v, w, indexing="ij")
    grid = pv.StructuredGrid(U, V, W)

    ids = np.arange(N)
    I, J, K = np.unravel_index(ids, (nu, nv, nw), order="F")
    on_boundary = (
        (I == 0) | (I == nu - 1)
        | (J == 0) | (J == nv - 1)
        | (K == 0) | (K == nw - 1)
    ).astype(np.int32)

    grid["id"] = ids
    grid["i"] = I
    grid["j"] = J
    grid["k"] = K
    grid["on_boundary"] = on_boundary

    grid.save(str(filename))
    print(f"Design volume (paramspace) exported to: {filename}")


def export_control_lattice_physical(
    control_points,
    n_per_dim,
    filename,
    order="F",
    boundary_only=False,
):
    """Export a deformed control lattice in physical space: points plus
    lines connecting (i,j,k) neighbors along each grid axis.

    Parameters
    ----------
    control_points : torch.Tensor or np.ndarray
        Shape (N, 3) where N == prod(n_per_dim).
    n_per_dim : sequence of 3 ints
        (n0, n1, n2) grid shape of the control lattice.
    filename : str or Path
        Output .vtp file path.
    order : str
        Flattening order for control point indices ('F' or 'C').
    boundary_only : bool
        If True, keep only the points and edges on the six outer faces
        of the lattice (the "boundary cage"). Point indices are remapped
        so `id` still refers to the original full-lattice index.
    """
    if isinstance(control_points, torch.Tensor):
        pts = control_points.detach().cpu().numpy()
    else:
        pts = np.asarray(control_points)

    n0, n1, n2 = (int(n) for n in n_per_dim)
    N = n0 * n1 * n2
    if pts.shape[0] != N:
        raise ValueError(
            f"control_points has {pts.shape[0]} rows but n_per_dim={n_per_dim} implies {N}"
        )

    ids = np.arange(N)
    I, J, K = np.unravel_index(ids, (n0, n1, n2), order=order)

    def flat(i, j, k):
        return np.ravel_multi_index((i, j, k), (n0, n1, n2), order=order)

    def _edge_block(i_range, j_range, k_range, di, dj, dk):
        i_idx, j_idx, k_idx = np.meshgrid(i_range, j_range, k_range, indexing="ij")
        a = flat(i_idx.ravel(), j_idx.ravel(), k_idx.ravel())
        b = flat(i_idx.ravel() + di, j_idx.ravel() + dj, k_idx.ravel() + dk)
        return np.column_stack([a, b]), i_idx.ravel(), j_idx.ravel(), k_idx.ravel()

    edges = []
    # +i edges: shared j, k → on boundary iff j or k on boundary
    if n0 > 1:
        edge, _, j_ed, k_ed = _edge_block(
            np.arange(n0 - 1), np.arange(n1), np.arange(n2), 1, 0, 0,
        )
        if boundary_only:
            mask = (j_ed == 0) | (j_ed == n1 - 1) | (k_ed == 0) | (k_ed == n2 - 1)
            edge = edge[mask]
        edges.append(edge)
    # +j edges: shared i, k → on boundary iff i or k on boundary
    if n1 > 1:
        edge, i_ed, _, k_ed = _edge_block(
            np.arange(n0), np.arange(n1 - 1), np.arange(n2), 0, 1, 0,
        )
        if boundary_only:
            mask = (i_ed == 0) | (i_ed == n0 - 1) | (k_ed == 0) | (k_ed == n2 - 1)
            edge = edge[mask]
        edges.append(edge)
    # +k edges: shared i, j → on boundary iff i or j on boundary
    if n2 > 1:
        edge, i_ed, j_ed, _ = _edge_block(
            np.arange(n0), np.arange(n1), np.arange(n2 - 1), 0, 0, 1,
        )
        if boundary_only:
            mask = (i_ed == 0) | (i_ed == n0 - 1) | (j_ed == 0) | (j_ed == n1 - 1)
            edge = edge[mask]
        edges.append(edge)

    edge_array = np.vstack(edges) if edges else np.zeros((0, 2), dtype=np.int64)

    if boundary_only:
        on_boundary = (
            (I == 0) | (I == n0 - 1)
            | (J == 0) | (J == n1 - 1)
            | (K == 0) | (K == n2 - 1)
        )
        keep_idx = np.flatnonzero(on_boundary)
        remap = -np.ones(N, dtype=np.int64)
        remap[keep_idx] = np.arange(keep_idx.size)
        pts_out = pts[keep_idx]
        ids_out = ids[keep_idx]
        I_out, J_out, K_out = I[keep_idx], J[keep_idx], K[keep_idx]
        edge_array = remap[edge_array]
    else:
        pts_out = pts
        ids_out = ids
        I_out, J_out, K_out = I, J, K

    if edge_array.size > 0:
        lines = np.column_stack(
            [np.full(edge_array.shape[0], 2, dtype=np.int64), edge_array]
        ).ravel()
    else:
        lines = np.array([], dtype=np.int64)

    poly = pv.PolyData(pts_out)
    if lines.size > 0:
        poly.lines = lines
    poly["id"] = ids_out
    poly["i"] = I_out
    poly["j"] = J_out
    poly["k"] = K_out
    poly.save(str(filename))
    print(f"Control lattice (physical) exported to: {filename}")


def export_control_volume_physical(
    control_points,
    n_per_dim,
    filename,
    undeformed=None,
    order="F",
):
    """Export a deformed control lattice as a StructuredGrid (.vts) — a
    solid hexahedral volume. In ParaView, render directly (adjust opacity)
    or apply `Extract Surface` to see only the six boundary faces. Color
    by any point array; `displacement_mag` is a natural choice.

    Parameters
    ----------
    control_points : torch.Tensor or np.ndarray
        Shape (N, 3) where N == prod(n_per_dim). Ordering must be F-order
        (i varies fastest), matching VTK's StructuredGrid convention.
    n_per_dim : sequence of 3 ints
        (n0, n1, n2) grid shape of the control lattice.
    filename : str or Path
        Output .vts file path.
    undeformed : torch.Tensor or np.ndarray, optional
        Same shape as `control_points`. If provided, adds `displacement`
        (vector) and `displacement_mag` (scalar) point arrays.
    order : str
        Flattening order ('F' required — VTK StructuredGrid uses F-order).
    """
    if order != "F":
        raise NotImplementedError(
            "StructuredGrid requires order='F' (i varies fastest)."
        )
    if isinstance(control_points, torch.Tensor):
        pts = control_points.detach().cpu().numpy()
    else:
        pts = np.asarray(control_points)

    n0, n1, n2 = (int(n) for n in n_per_dim)
    N = n0 * n1 * n2
    if pts.shape[0] != N:
        raise ValueError(
            f"control_points has {pts.shape[0]} rows but n_per_dim={n_per_dim} implies {N}"
        )

    grid = pv.StructuredGrid()
    grid.points = pts.astype(np.float64, copy=False)
    grid.dimensions = (n0, n1, n2)

    ids = np.arange(N)
    I, J, K = np.unravel_index(ids, (n0, n1, n2), order=order)
    on_boundary = (
        (I == 0) | (I == n0 - 1)
        | (J == 0) | (J == n1 - 1)
        | (K == 0) | (K == n2 - 1)
    ).astype(np.int32)

    grid["id"] = ids
    grid["i"] = I
    grid["j"] = J
    grid["k"] = K
    grid["on_boundary"] = on_boundary

    if undeformed is not None:
        if isinstance(undeformed, torch.Tensor):
            und = undeformed.detach().cpu().numpy()
        else:
            und = np.asarray(undeformed)
        if und.shape != pts.shape:
            raise ValueError(
                f"undeformed shape {und.shape} does not match control_points {pts.shape}"
            )
        disp = pts - und
        grid["displacement"] = disp
        grid["displacement_mag"] = np.linalg.norm(disp, axis=1)

    grid.save(str(filename))
    print(f"Control volume (physical) exported to: {filename}")


def export_control_points(control_points, file_path):
    """Export control points as a VTP point cloud.

    Parameters
    ----------
    control_points : torch.Tensor or np.ndarray
        Shape (N, 3)
    file_path : str or Path
        Output .vtp file path
    """
    if isinstance(control_points, torch.Tensor):
        points = control_points.detach().cpu().numpy()
    else:
        points = np.asarray(control_points)

    poly = pv.PolyData(points)
    poly.save(str(file_path))
