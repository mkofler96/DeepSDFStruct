import logging
import torch as _torch
import torch.autograd.functional
import tetgenpy
import numpy as np
import napf
import gustaf as gus
import pathlib
import os
import skimage
import triangle
import vtk

from functools import partial

from DeepSDFStruct.flexicubes.flexicubes import FlexiCubes
from DeepSDFStruct.lattice_structure import LatticeSDFStruct
from DeepSDFStruct.SDF import SDFBase

logger = logging.getLogger(__name__)


class torchSurfMesh:
    def __init__(self, vertices: _torch.tensor, faces: _torch.tensor):
        self.vertices = vertices
        self.faces = faces

    def to_gus(self):
        return gus.Faces(self.vertices.detach().cpu(), self.faces.detach().cpu())

    def to_trimesh():
        raise NotImplementedError("To trimesh functionality not implemented yet.")
        pass


def tetrahedralize_surface(surface_mesh: gus.Faces) -> tuple[gus.Volumes, np.ndarray]:
    logger.debug("Tetrahedralizing surface mesh")
    t_in = tetgenpy.TetgenIO()
    t_in.setup_plc(surface_mesh.vertices, surface_mesh.faces.tolist())
    # gus.show(dmesh)
    switch_command = "pYq"
    if logging.DEBUG <= logging.root.level:
        switch_command += "Q"
    t_out = tetgenpy.tetrahedralize(switch_command, t_in)  # pqa

    tets = np.vstack(t_out.tetrahedra())
    verts = t_out.points()

    kdt = napf.KDT(tree_data=verts, metric=1)

    distances, face_indices = kdt.knn_search(
        queries=surface_mesh.vertices, kneighbors=1, nthread=4
    )
    tol = 1e-6
    if distances.max() > tol:
        Warning("Not all surface nodes as included in the volumetric mesh.")
    volumes = gus.Volumes(verts, tets)
    surface_mesh_indices = face_indices
    return volumes, surface_mesh_indices


def export_volume_mesh(volume_mesh: gus.Volumes, filename: str, export_abaqus=False):
    """
    export a mesh and adds corresponding boundary conditions
    """
    filepath = pathlib.Path(filename)
    if not os.path.isdir(filepath.parent):
        os.makedirs(filepath.parent)
    logger.debug(
        f"Exporting mesh with {len(volume_mesh.volumes)} elements, {len(volume_mesh.vertices)} vertices to {filepath}"
    )
    gus.io.mfem.export(str(filepath), volume_mesh)


def export_abaqus_surf_mesh(surf_mesh: gus.Faces, filename: str):
    """
    export a mesh and adds corresponding boundary conditions
    """
    filepath = pathlib.Path(filename)
    if not os.path.isdir(filepath.parent):
        os.makedirs(filepath.parent)
    gus.io.meshio.export(str(filepath.with_suffix(".inp")), surf_mesh)


def generate_2D_surf_mesh(
    sdf: SDFBase, n_squares: int, n_elements: int = 50000, bounds=None
):
    n_points = 1000
    if bounds is None:
        bounds = sdf._get_domain_bounds()
    x = np.linspace(bounds[0, 0], bounds[1, 0], n_points)
    y = np.linspace(bounds[0, 1], bounds[1, 1], n_points)
    xx, yy = np.meshgrid(x, y)
    xx, yy = np.meshgrid(x, y)
    queries = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
    evaluated_sdf = sdf(queries)
    evaluated_sdf_orig_shape = evaluated_sdf.reshape(xx.shape)
    A = sdf_to_triangle_dict(x, y, evaluated_sdf_orig_shape, level=0)
    domain_area = (bounds[1, 0] - bounds[0, 0]) * (bounds[1, 1] - bounds[0, 1])
    min_a = domain_area / n_elements
    triangle_string = f"pqa{min_a}"
    logger.info("Calling triangulate with " + triangle_string)
    B = triangle.triangulate(A, triangle_string)
    mesh = gus.Faces(vertices=B["vertices"], faces=B["triangles"])
    return mesh


def sdf_to_triangle_dict(x, y, sdf, level=0.0, pad_value=1.0, collinear_tol=1e-8):
    """
    Convert an SDF array to a PSLG dict for triangle.triangulate.
    Includes domain boundary and hole polygons.

    Args:
        sdf (np.ndarray): 2D signed distance field.
        level (float): Contour level (usually 0 for boundary).
        pad_value (float): Value outside domain to close boundaries.

    Returns:
        dict: {"vertices": ..., "segments": ..., "holes": ...}
    """
    H, W = sdf.shape

    # Pad so contours touching domain edge get closed
    padded = np.pad(sdf, 1, mode="constant", constant_values=pad_value)
    contours = skimage.measure.find_contours(padded, level)
    contours = [c - 1 for c in contours]  # shift back after padding

    vertices = []
    segments = []
    holes = []

    v_offset = 0

    for contour in contours:
        # contour[:, 0] = row indices in [0, H), map to y
        # contour[:, 1] = col indices in [0, W), map to x
        poly_x = np.interp(contour[:, 1], np.arange(W), x)
        poly_y = np.interp(contour[:, 0], np.arange(H), y)
        poly = np.column_stack([poly_x, poly_y])
        poly = prune_collinear(poly, tol=collinear_tol)

        n = len(poly)
        vertices.extend(poly.tolist())
        segments.extend([[v_offset + i, v_offset + (i + 1) % n] for i in range(n)])

        # Orientation heuristic
        area = 0.5 * np.sum(
            poly[:, 0] * np.roll(poly[:, 1], -1) - poly[:, 1] * np.roll(poly[:, 0], -1)
        )
        if area > 0:
            centroid = poly.mean(axis=0)
            holes.append(centroid.tolist())

        v_offset += n

    _, unique_indices, counts = np.unique(
        vertices, axis=0, return_index=True, return_counts=True
    )
    duplicate_indices = np.where(counts > 1)[0]

    if len(duplicate_indices) > 0:
        logger.info(f"Found {len(duplicate_indices)} duplicate vertices:")
        for idx in duplicate_indices:
            logger.debug(vertices[idx])
    else:
        print("No duplicate vertices found.")

    vertices, idx = np.unique(vertices, axis=0, return_inverse=True)
    segments = np.array([[idx[s[0]], idx[s[1]]] for s in segments])

    return dict(
        vertices=np.array(vertices), segments=np.array(segments), holes=np.array(holes)
    )


def prune_collinear(points, tol=1e-9):
    """
    Remove nearly collinear points from a polyline (closed loop).
    Args:
        points (ndarray): Nx2 array of (x,y) vertices.
        tol (float): area tolerance. Smaller = stricter (keep more points).
    Returns:
        ndarray: pruned Nx2 array of vertices.
    """
    if len(points) <= 3:
        return points  # nothing to prune

    pruned = [points[0]]
    for i in range(1, len(points) - 1):
        a, b, c = points[i - 1], points[i], points[i + 1]
        # Compute twice the triangle area formed by (a,b,c)
        area = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        if area > tol:
            pruned.append(b)
    pruned.append(points[-1])
    return np.array(pruned)


def process_N_base_input(N, tiling):
    if isinstance(N, list):
        if len(N) != 3:
            raise ValueError("Number of grid points must be a list of 3 integers")
        N = _torch.tensor(N)
    elif isinstance(N, int):
        N = _torch.tensor([N, N, N])
    else:
        raise ValueError("Number of grid points must be a list or an integer")
    # add 1 on each side to slightly include the border
    N_mod = N * tiling + 1
    return N_mod


def _prepare_flexicubes_querypoints(N, device=None):
    """
    takes the tiling and a resolution as input
    output: DeepSDFStruct.flexicubes constructor, samples and cube indices
            the points are located in the region [0,1] with a margin of 0.025
            -> [-0.025, 1.025]
    """
    # check_tiling_input(tiling)
    if device is None:
        device = "cuda" if _torch.cuda.is_available() else "cpu"
    flexi_cubes_constructor = FlexiCubes(device=device)
    samples, cube_idx = flexi_cubes_constructor.construct_voxel_grid(
        resolution=tuple(N)
    )

    samples = samples * 1.1 + _torch.tensor([0.5, 0.5, 0.5], device=device)
    tolerance = 1e-6
    _torch._assert(
        _torch.all(samples.ge(-0.05 - tolerance) & samples.le(1.05 + tolerance)),
        "Samples are out of bounds",
    )

    return flexi_cubes_constructor, samples, cube_idx


def create_3D_surface_mesh(sdf: SDFBase, N_base, differentiate=False, device="cpu"):
    if type(sdf) is LatticeSDFStruct:
        tiling = _torch.tensor(sdf.tiling)
    else:
        tiling = _torch.tensor([1, 1, 1])
    N = process_N_base_input(N_base, tiling)

    constructor, samples, cube_idx = _prepare_flexicubes_querypoints(N, device=device)
    dVerts_dParams = None

    verts_fn = partial(
        _verts_from_params,
        sdf=sdf,
        samples=samples,
        constructor=constructor,
        cube_idx=cube_idx,
        N=N,
        return_faces=False,
    )
    verts, faces = _verts_from_params(
        p=sdf.parametrization.parameters,
        sdf=sdf,
        samples=samples,
        constructor=constructor,
        cube_idx=cube_idx,
        N=N,
        return_faces=True,
    )
    if differentiate:
        if sdf.parametrization is None:
            raise RuntimeError("No parametrization found for given SDF")
        dVerts_dParams = torch.autograd.functional.jacobian(
            verts_fn,
            sdf.parametrization.parameters,
            strategy="forward-mode",
            vectorize=True,
        )

    return torchSurfMesh(verts, faces), dVerts_dParams


def _verts_from_params(
    p: _torch.tensor,
    sdf: SDFBase,
    samples: _torch.tensor,
    constructor: FlexiCubes,
    cube_idx: _torch.tensor,
    N,
    return_faces=False,
):
    sdf.parametrization.set_param(p)
    sdf_values = sdf(samples)

    verts_local, faces, _ = constructor(
        voxelgrid_vertices=samples,
        scalar_field=sdf_values,
        cube_idx=cube_idx,
        resolution=tuple(N),
        output_tetmesh=True,
    )

    if sdf.deformation_spline is not None:
        verts_local = sdf.deformation_spline.forward(verts_local)
    if return_faces:
        return verts_local, faces
    else:
        return verts_local


def export_sdf_grid_vtk(sdf: SDFBase, filename, N=64, bounds=None, device="cpu"):
    if bounds is None:
        bounds = np.array([[0, 0, 0], [1, 1, 1]])
    # Generate grid points
    x = np.linspace(bounds[0, 0], bounds[1, 0], N)
    y = np.linspace(bounds[0, 1], bounds[1, 1], N)
    z = np.linspace(bounds[0, 2], bounds[1, 2], N)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

    # Evaluate SDF
    with _torch.no_grad():
        sdf_vals = sdf(_torch.tensor(points, dtype=_torch.float32, device=device))
    sdf_vals = sdf_vals.detach().cpu().numpy()

    # Create vtkPoints
    vtk_points = vtk.vtkPoints()
    for pt in points:
        vtk_points.InsertNextPoint(pt.tolist())

    # Create structured grid
    grid = vtk.vtkStructuredGrid()
    grid.SetDimensions(N, N, N)
    grid.SetPoints(vtk_points)

    # Add SDF scalar field
    vtk_array = vtk.vtkDoubleArray()
    vtk_array.SetName("SDF")
    vtk_array.SetNumberOfValues(len(sdf_vals))
    for i, val in enumerate(sdf_vals):
        vtk_array.SetValue(i, val)
    grid.GetPointData().SetScalars(vtk_array)

    # Write to file
    writer = vtk.vtkStructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(grid)
    writer.Write()
    print(f"SDF structured grid saved to {filename}")


def _export_surface_mesh_vtk(verts, faces, filename, dSurf=None):
    """
    verts: (N, 3) torch tensor
    faces: (M, 3) torch tensor
    dSurf: (N, 3, ..,..) torch tensor
    """
    vtk_points = vtk.vtkPoints()
    for v in verts:
        vtk_points.InsertNextPoint(v.tolist())

    vtk_cells = vtk.vtkCellArray()
    for f in faces:
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, f[0])
        triangle.GetPointIds().SetId(1, f[1])
        triangle.GetPointIds().SetId(2, f[2])
        vtk_cells.InsertNextCell(triangle)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetPolys(vtk_cells)

    # Add derivative vectors as a vector field
    extra_dims = dSurf.shape[2:]
    N = int(np.prod(extra_dims))

    # reshape to [n_nodes, N_derivatives, 3]
    dSurf_flat = dSurf.flatten(2)
    indices = [np.unravel_index(i, extra_dims) for i in range(N)]
    assert dSurf_flat.shape[2] == len(indices)
    for j, idx in enumerate(indices):
        idx_str = "".join(str(i + 1) for i in idx)
        name = f"derivative_[{idx_str}]"

        vectors = vtk.vtkDoubleArray()
        vectors.SetNumberOfComponents(3)
        vectors.SetName(name)

        for vec in dSurf_flat[:, :, j]:
            vectors.InsertNextTuple(vec.tolist())

        polydata.GetPointData().AddArray(vectors)
        polydata.GetPointData().SetActiveVectors(name)
    # Write to file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Write()
    print(f"Mesh saved to {filename}")


def export_surface_mesh(
    filename: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    mesh: gus.Faces | torchSurfMesh,
    dSurf=None,
):
    export_filename = pathlib.Path(filename)
    ext = export_filename.suffix.lower()
    if isinstance(mesh, torchSurfMesh):
        mesh = mesh.to_gus()
    match ext:
        case ".vtk":
            _export_surface_mesh_vtk(mesh.vertices, mesh.faces, export_filename, dSurf)
        case _:
            gus.io.meshio.export(export_filename, mesh)
